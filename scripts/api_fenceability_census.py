#!/usr/bin/env python3
"""Production Tool API Fence-Ability Census (`evidence/v3/reference/api_fenceability_census.json`).

Audits 14 widely deployed agent tool APIs across 6 infrastructure categories on the three
protocol primitives required by Complete Authority Mediation (C1/C2/C6):
  1. Native Idempotency Key (safe retry without double-charging/double-executing under lost reply)
  2. Conditional / Epoch CAS Write (rejecting stale authority epochs or concurrent revisions)
  3. Two-Phase Hold / Pre-Commit Cancel (stopping queued/in-flight work upon revocation)

Classifies each tool surface into one of four operational deployment tiers:
  - `natively_fenceable` (5/14): Sink natively exposes idempotency/preconditions + two-phase cancel.
  - `proxy_fenced` (4/14): Cooperative CAS/staging exists when wrapped by a local broker/staging shim.
  - `escrow_buffer_required` (3/14): Immediate dispatch is irrevocable; requires an outbound delay/escrow queue.
  - `uncooperative_irrevocable` (2/14): Side effects leave custody immediately once invoked; enters residual `u`
    unless confined in an ephemeral sandbox or blocked at the network waist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "evidence/v3/reference/api_fenceability_census.json"

APIS: list[dict] = [
    {
        "api_id": "stripe_payment_intents",
        "surface": "Stripe PaymentIntents",
        "category": "payments_ledger",
        "native_idempotency": "yes",
        "conditional_cas": "partial",
        "two_phase_cancel": "yes",
        "tier": "natively_fenceable",
        "mechanism_summary": "Idempotency-Key header (24h window) + capture_method='manual' hold with POST /v1/payment_intents/:id/cancel prior to capture.",
        "paper_note": "24h Idempotency-Key header + manual-capture hold with pre-capture cancel",
        "residual_u_status": "covered_pre_capture",
    },
    {
        "api_id": "postgresql_serializable",
        "surface": "PostgreSQL 16+ (2PC)",
        "category": "relational_database",
        "native_idempotency": "yes",
        "conditional_cas": "yes",
        "two_phase_cancel": "yes",
        "tier": "natively_fenceable",
        "mechanism_summary": "ON CONFLICT (permit_id) DO NOTHING + WHERE epoch >= floor AND rev = expected + PREPARE TRANSACTION / ROLLBACK PREPARED.",
        "paper_note": "Unique permit-key conflict guard + epoch/revision predicate + prepared rollback",
        "residual_u_status": "covered_cooperative_sink",
    },
    {
        "api_id": "sqlite_wal_sink",
        "surface": "SQLite 3 WAL Sink",
        "category": "relational_database",
        "native_idempotency": "yes",
        "conditional_cas": "yes",
        "two_phase_cancel": "yes",
        "tier": "natively_fenceable",
        "mechanism_summary": "PRIMARY KEY permit_id dedup + monotone epoch_floor table + pending->cancelled transition in BEGIN IMMEDIATE WAL.",
        "paper_note": "Primary-key permit dedup + monotone epoch-floor table in immediate WAL transaction",
        "residual_u_status": "covered_cooperative_sink",
    },
    {
        "api_id": "aws_s3_objects",
        "surface": "AWS S3 Objects",
        "category": "cloud_storage",
        "native_idempotency": "no",
        "conditional_cas": "yes",
        "two_phase_cancel": "yes",
        "tier": "natively_fenceable",
        "mechanism_summary": "If-None-Match / If-Match ETag conditional writes (2024+) + CreateMultipartUpload / AbortMultipartUpload before CompleteMultipartUpload.",
        "paper_note": "If-Match / If-None-Match ETag preconditions + abortable multipart upload",
        "residual_u_status": "covered_with_conditional_multipart",
    },
    {
        "api_id": "gcs_objects",
        "surface": "Google Cloud Storage",
        "category": "cloud_storage",
        "native_idempotency": "yes",
        "conditional_cas": "yes",
        "two_phase_cancel": "yes",
        "tier": "natively_fenceable",
        "mechanism_summary": "ifGenerationMatch / ifMetagenerationMatch preconditions + cancellable resumable upload session URI prior to final commit.",
        "paper_note": "Generation-match preconditions + cancellable resumable upload session",
        "residual_u_status": "covered_cooperative_sink",
    },
    {
        "api_id": "kubernetes_apiserver",
        "surface": "Kubernetes API (etcd)",
        "category": "cloud_control_plane",
        "native_idempotency": "partial",
        "conditional_cas": "yes",
        "two_phase_cancel": "partial",
        "tier": "proxy_fenced",
        "mechanism_summary": "metadata.resourceVersion optimistic CAS + ValidatingAdmissionWebhook epoch gate + dryRun=All preflight.",
        "paper_note": "Resource-version CAS + validating admission webhook checking epoch floor",
        "residual_u_status": "covered_via_admission_webhook",
    },
    {
        "api_id": "github_git_refs",
        "surface": "GitHub Git Refs \\& PRs",
        "category": "source_control",
        "native_idempotency": "no",
        "conditional_cas": "yes",
        "two_phase_cancel": "yes",
        "tier": "proxy_fenced",
        "mechanism_summary": "GraphQL UpdateRefInput.expectedHeadOid CAS (force=false) + staging on PR branch gated by merge-queue status check.",
        "paper_note": "Expected-head OID compare-and-swap (non-force) + staged PR merge-queue check",
        "residual_u_status": "covered_via_protected_ref_cas",
    },
    {
        "api_id": "linear_jira_issues",
        "surface": "Linear / Jira Issues",
        "category": "work_tracking",
        "native_idempotency": "no",
        "conditional_cas": "partial",
        "two_phase_cancel": "no",
        "tier": "proxy_fenced",
        "mechanism_summary": "Lacks native create idempotency; requires local broker permit_id dedup table and draft/triage state before public transition.",
        "paper_note": "Broker-side permit dedup table + draft triage state before public transition",
        "residual_u_status": "covered_via_broker_dedup_shim",
    },
    {
        "api_id": "posix_filesystem",
        "surface": "POSIX Filesystem",
        "category": "local_os",
        "native_idempotency": "no",
        "conditional_cas": "yes",
        "two_phase_cancel": "yes",
        "tier": "proxy_fenced",
        "mechanism_summary": "Stage writes in O_TMPFILE / private dir + unlink on fence, or commit atomically via renameat2(..., RENAME_NOREPLACE).",
        "paper_note": "Private staging file unlinked on fence + atomic non-overwriting rename commit",
        "residual_u_status": "covered_via_atomic_staging_shim",
    },
    {
        "api_id": "sendgrid_smtp_email",
        "surface": "SendGrid / SMTP Email",
        "category": "external_comms",
        "native_idempotency": "no",
        "conditional_cas": "no",
        "two_phase_cancel": "partial",
        "tier": "escrow_buffer_required",
        "mechanism_summary": "Immediate SMTP relay is irrevocable; requires local outbound hold queue or SendGrid batch_id scheduled_sends cancel window.",
        "paper_note": "Immediate SMTP is irrevocable; requires outbound hold queue or scheduled batch cancel",
        "residual_u_status": "requires_outbound_escrow_else_u",
    },
    {
        "api_id": "slack_chat_api",
        "surface": "Slack Web Chat API",
        "category": "external_comms",
        "native_idempotency": "no",
        "conditional_cas": "no",
        "two_phase_cancel": "partial",
        "tier": "escrow_buffer_required",
        "mechanism_summary": "chat.postMessage duplicates on retry and triggers push notifications immediately; requires chat.scheduleMessage + deleteScheduledMessage or local escrow.",
        "paper_note": "Direct post duplicates on retry; requires scheduled-message cancel window or local escrow",
        "residual_u_status": "requires_outbound_escrow_else_u",
    },
    {
        "api_id": "twilio_sms",
        "surface": "Twilio Programmable SMS",
        "category": "external_comms",
        "native_idempotency": "no",
        "conditional_cas": "no",
        "two_phase_cancel": "partial",
        "tier": "escrow_buffer_required",
        "mechanism_summary": "Carrier handoff is irrevocable; ScheduleType=Fixed allows Status=canceled only with >=15m lead time, else requires local broker hold.",
        "paper_note": "Carrier handoff is irrevocable; requires >=15 min scheduled hold or local escrow",
        "residual_u_status": "requires_outbound_escrow_else_u",
    },
    {
        "api_id": "local_subprocess_execve",
        "surface": "Arbitrary OS Subprocess",
        "category": "local_os",
        "native_idempotency": "no",
        "conditional_cas": "no",
        "two_phase_cancel": "no",
        "tier": "uncooperative_irrevocable",
        "mechanism_summary": "SIGKILL halts CPU execution mid-flight without rolling back partial side effects; safe only inside ephemeral container/overlayfs with network waist cut.",
        "paper_note": "Process kill cannot roll back partial side effects; safe only in ephemeral sandbox waist",
        "residual_u_status": "unmediated_residual_u_unless_sandboxed",
    },
    {
        "api_id": "outbound_http_webhooks",
        "surface": "Third-Party Webhooks",
        "category": "external_network",
        "native_idempotency": "no",
        "conditional_cas": "no",
        "two_phase_cancel": "no",
        "tier": "uncooperative_irrevocable",
        "mechanism_summary": "Once request bytes leave NIC to an uncooperative host, remote state mutation cannot be fenced; enters irreducible external exposure u.",
        "paper_note": "Once bytes leave NIC to an uncooperative host, remote state enters residual exposure $u$",
        "residual_u_status": "irreducible_external_u",
    },
]


def build_census() -> dict:
    tiers = [
        "natively_fenceable",
        "proxy_fenced",
        "escrow_buffer_required",
        "uncooperative_irrevocable",
    ]
    counts = {t: sum(1 for a in APIS if a["tier"] == t) for t in tiers}
    assert sum(counts.values()) == len(APIS) == 14
    primitive_counts = {
        "native_idempotency_yes": sum(1 for a in APIS if a["native_idempotency"] == "yes"),
        "conditional_cas_yes": sum(1 for a in APIS if a["conditional_cas"] == "yes"),
        "two_phase_cancel_yes": sum(1 for a in APIS if a["two_phase_cancel"] == "yes"),
    }
    return {
        "schema_version": 1,
        "suite": "api_fenceability_census",
        "total_apis_surveyed": len(APIS),
        "tier_counts": counts,
        "cooperative_or_shim_fenceable_count": counts["natively_fenceable"] + counts["proxy_fenced"],
        "escrow_or_residual_u_count": counts["escrow_buffer_required"] + counts["uncooperative_irrevocable"],
        "primitive_yes_counts": primitive_counts,
        "apis": APIS,
        "script_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        "claim_boundary": (
            "Architectural protocol audit of 14 production tool interfaces against C1/C2/C6 idempotency, "
            "conditional CAS, and two-phase cancellation requirements; documents where complete authority "
            "mediation applies natively (5/14), via a thin CAS/staging shim (4/14), via an outbound delay/escrow "
            "buffer (3/14), or fails into irreducible external residual u (2/14)."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    data = build_census()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "tier_counts": data["tier_counts"]}, indent=2))


if __name__ == "__main__":
    main()
