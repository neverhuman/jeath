//! Release-mode kernel guard on the V2 broker/sink path, through the public API
//! and the `authority-lab sink-v2` binary. No test hook is used here: each
//! divergence or kernel-less state is reachable from outside the crate.
use authority_lab::protocol_v2::{
    Broker, DispatchedPermit, Effect, KERNEL_DIVERGENCE, KERNEL_UNAVAILABLE, KernelGuardError,
    Permit, Proposal, Record, Sink, Status, same_dispatched_permit,
};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

fn transfer(id: &str, units: u64, epochs: Vec<u64>) -> Permit {
    Permit {
        actor: "alice".into(),
        proposal: Proposal {
            id: id.into(),
            epochs,
            expires: 1_000,
            effect: Effect::Transfer {
                recipient: "approved".into(),
                units,
            },
        },
    }
}

fn edit(id: &str, expected_revision: u64) -> Permit {
    Permit {
        actor: "alice".into(),
        proposal: Proposal {
            id: id.into(),
            epochs: vec![0, 0],
            expires: 1_000,
            effect: Effect::Edit {
                document: "toy.rs".into(),
                expected_revision,
                contents: "fn answer() -> u32 { 42 }".into(),
            },
        },
    }
}

fn guard_reason(err: &(dyn std::error::Error + 'static)) -> Option<&'static str> {
    err.downcast_ref::<KernelGuardError>().map(|e| e.reason)
}

fn snapshot<T: serde::Serialize>(state: &T) -> String {
    serde_json::to_string(state).unwrap()
}

const HALF: u64 = i64::MAX as u64;

/// Explicit base requirements and their compact spelling describe the same
/// authority. A durable record from either spelling must remain retryable.
#[test]
fn equivalent_required_spellings_survive_recovery_and_retry() {
    let compact = DispatchedPermit::from(transfer("canonical", 1, vec![0, 0]));
    let explicit = DispatchedPermit {
        required: vec![true, false],
        ..compact.clone()
    };
    assert!(same_dispatched_permit(&compact, &explicit).unwrap());
    let mut sink = Sink::default();
    sink.records.push(Record {
        permit: explicit.clone(),
        status: Status::Pending,
    });
    // Exercise the representation loaded by recovery, rather than acceptance's
    // normalization of a newly received permit.
    sink = serde_json::from_str(&snapshot(&sink)).unwrap();
    assert!(sink.to_exec_kernel().is_some());
    assert_eq!(
        sink.accept_dispatched(compact.clone()).unwrap()["status"],
        "pending"
    );
    assert_eq!(sink.complete("canonical").unwrap().0["status"], "committed");
    assert_eq!(
        sink.accept_dispatched(explicit).unwrap()["status"],
        "committed"
    );
    assert!(sink.complete("canonical").unwrap().1.is_none());

    let strengthened = DispatchedPermit {
        required: vec![true, true],
        ..compact.clone()
    };
    assert!(!same_dispatched_permit(&compact, &strengthened).unwrap());
    assert_eq!(
        sink.accept_dispatched(strengthened).unwrap()["status"],
        "denied"
    );
}

/// The sink does not budget-check direct deliveries. After two commits of
/// i64::MAX units the committed total is 2^64 - 2, so a third commit of 2 units
/// no longer fits. The verified kernel cancels it; the unverified Rust would
/// commit it. The guard refuses, and no effect is reported.
#[test]
fn sink_refuses_a_commit_the_verified_kernel_would_cancel() {
    let mut s = Sink::default();
    for id in ["a", "b"] {
        assert_eq!(
            s.accept(transfer(id, HALF, vec![0, 0])).unwrap()["status"],
            "pending"
        );
        assert_eq!(s.complete(id).unwrap().0["status"], "committed");
    }
    assert_eq!(
        s.accept(transfer("c", 2, vec![0, 0])).unwrap()["status"],
        "pending"
    );
    let before = snapshot(&s);
    let err = s.complete("c").unwrap_err();
    assert_eq!(guard_reason(&*err), Some(KERNEL_DIVERGENCE));
    assert_eq!(snapshot(&s), before);
    assert_eq!(s.status("c"), json!({"status": "pending"}));
}

/// The same divergence end to end through the release service: the reply is
/// the standard denial shape, the transaction is rolled back, the effects table
/// gains no row, and the sink's own audit stays consistent.
#[test]
fn sink_service_denies_kernel_divergence_and_records_no_effect() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "kernel-guard-v2-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let db = dir.join("sink.sqlite");
    let mut child = Command::new(env!("CARGO_BIN_EXE_authority-lab"))
        .arg("sink-v2")
        .arg(&db)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut call = |q: Value| -> Value {
        writeln!(stdin, "{q}").unwrap();
        stdin.flush().unwrap();
        let mut line = String::new();
        stdout.read_line(&mut line).unwrap();
        serde_json::from_str(&line).unwrap()
    };
    let permit = |id: &str, units: u64| json!({"actor": "alice", "proposal": transfer(id, units, vec![0, 0]).proposal});

    assert_eq!(
        call(json!({"op": "init"})),
        json!({"status": "initialized"})
    );
    for id in ["a", "b"] {
        assert_eq!(
            call(json!({"op": "accept", "permit": permit(id, HALF)})),
            json!({"status": "pending"})
        );
        assert_eq!(
            call(json!({"op": "complete", "id": id})),
            json!({"status": "committed"})
        );
    }
    assert_eq!(
        call(json!({"op": "accept", "permit": permit("c", 2)})),
        json!({"status": "pending"})
    );
    assert_eq!(
        call(json!({"op": "complete", "id": "c"})),
        json!({"status": "denied", "reason": KERNEL_DIVERGENCE})
    );
    assert_eq!(
        call(json!({"op": "status", "id": "c"})),
        json!({"status": "pending"})
    );
    let audit = call(json!({"op": "audit"}));
    assert_eq!(audit["consistent"], true, "{audit}");
    assert_eq!(audit["effects"], 2, "{audit}");
    drop(stdin);
    assert!(child.wait().unwrap().success());

    let conn = rusqlite::Connection::open(&db).unwrap();
    let ids: Vec<String> = conn
        .prepare("SELECT id FROM effects ORDER BY sequence")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(ids, ["a", "b"]);
}

/// Two records for one id have no kernel image. Rust alone would commit the
/// first and so produce a second effect for an id that is already committed.
#[test]
fn sink_fails_closed_on_a_state_with_no_kernel_image() {
    let mut s = Sink::default();
    s.accept(transfer("p", 2, vec![0, 0])).unwrap();
    s.records.push(Record {
        permit: s.records[0].permit.clone(),
        status: Status::Committed,
    });
    let before = snapshot(&s);
    for err in [
        s.complete("p").unwrap_err(),
        s.fence(0, 1).unwrap_err(),
        s.accept(transfer("q", 1, vec![0, 0])).unwrap_err(),
    ] {
        assert_eq!(guard_reason(&*err), Some(KERNEL_UNAVAILABLE));
    }
    assert_eq!(snapshot(&s), before);
    assert_eq!(
        s.audit_state("ok", &[])["verus_exec_kernel_verified"],
        false
    );
}

/// A ledger whose `charged` no longer equals pending + committed has no kernel
/// image. Rust alone would admit a second full-budget transfer.
#[test]
fn broker_fails_closed_on_a_corrupted_ledger() {
    let mut b = Broker::new(8);
    b.approve(transfer("p1", 8, vec![0, 0]), 0).unwrap();
    assert_eq!(
        b.submit(transfer("p1", 8, vec![0, 0]), 10).unwrap()["status"],
        "pending"
    );
    b.charged = 0;
    b.approve(transfer("p2", 8, vec![0, 0]), 0).unwrap();
    let before = snapshot(&b);
    for err in [
        b.submit(transfer("p2", 8, vec![0, 0]), 10).unwrap_err(),
        b.settle("p1", Status::Cancelled).unwrap_err(),
        b.renew(0).unwrap_err(),
    ] {
        assert_eq!(guard_reason(&*err), Some(KERNEL_UNAVAILABLE));
    }
    assert_eq!(snapshot(&b), before);
    // Withdrawal only removes authority and is deliberately not gated.
    assert_eq!(b.withdraw(1).unwrap()["status"], "admission_revoked");
    assert_eq!(b.audit_state("ok")["verus_exec_kernel_verified"], false);
}

/// An epoch vector for another issuer population is refused, as the kernel
/// refuses it (InvalidIssuerLength). It used to leave a cancelled tombstone
/// with no kernel image, which silently disabled the kernel comparison for
/// every later request at that sink.
#[test]
fn sink_denies_a_wrong_length_epoch_vector_without_a_tombstone() {
    let mut s = Sink::default();
    assert_eq!(
        s.accept(transfer("x", 1, vec![0, 0, 0])).unwrap(),
        json!({"status": "denied", "reason": "invalid_issuer_length"})
    );
    assert!(s.records.is_empty());
    let wrong_required = DispatchedPermit {
        required: vec![true, true, true],
        ..DispatchedPermit::from(transfer("y", 1, vec![0, 0]))
    };
    assert_eq!(
        s.accept_dispatched(wrong_required).unwrap(),
        json!({"status": "denied", "reason": "invalid_required_issuers"})
    );
    assert_eq!(
        s.accept(transfer("x", 1, vec![0, 0])).unwrap()["status"],
        "pending"
    );
    assert_eq!(s.complete("x").unwrap().0["status"], "committed");
    assert!(s.to_exec_kernel().is_some());
}

/// Normal operation: the exact replies of a full approve, submit, accept,
/// complete, settle, fence, withdraw and renew cycle are those of the
/// unguarded code, and both audits stay kernel-verified.
#[test]
fn normal_cycle_replies_are_unchanged() {
    let mut b = Broker::new(10);
    let mut s = Sink::default();
    let mut effects = Vec::new();

    let t = transfer("t", 4, vec![0, 0]);
    assert_eq!(
        b.approve(t.clone(), 0).unwrap(),
        json!({"status": "approved"})
    );
    let sub = b.submit(t.clone(), 10).unwrap();
    assert_eq!(sub["status"], "pending");
    let disp: DispatchedPermit = serde_json::from_value(sub["permit"].clone()).unwrap();
    assert_eq!(
        b.submit(t.clone(), 11).unwrap(),
        json!({"status": "pending", "permit": disp})
    );
    let mut altered = t.clone();
    altered.actor = "mallory".into();
    assert_eq!(
        b.submit(altered, 11).unwrap(),
        json!({"status": "denied", "reason": "payload_mismatch"})
    );
    assert_eq!(
        s.accept_dispatched(disp.clone()).unwrap(),
        json!({"status": "pending"})
    );
    assert_eq!(
        s.accept_dispatched(disp.clone()).unwrap(),
        json!({"status": "pending"})
    );
    let (resp, newly) = s.complete("t").unwrap();
    assert_eq!(resp, json!({"status": "committed"}));
    effects.push(("t".to_string(), newly.unwrap(), 4));
    assert_eq!(
        s.complete("t").unwrap(),
        (json!({"status": "committed"}), None)
    );
    assert_eq!(
        b.settle("t", Status::Pending).unwrap()["reason"],
        "not_authoritative_terminal"
    );
    assert_eq!(
        b.settle("t", Status::Committed).unwrap(),
        json!({"status": "committed"})
    );
    assert_eq!(
        b.settle("t", Status::Committed).unwrap()["reason"],
        "not_authoritative_terminal"
    );
    assert!(b.settle("missing", Status::Committed).is_err());
    assert!(s.complete("missing").is_err());

    // Edit: commits at the expected revision, a stale one is cancelled.
    for (id, rev, terminal) in [("e1", 0, "committed"), ("e2", 0, "cancelled")] {
        let p = edit(id, rev);
        b.approve(p.clone(), 0).unwrap();
        b.approve(p.clone(), 1).unwrap();
        let disp: DispatchedPermit =
            serde_json::from_value(b.submit(p, 10).unwrap()["permit"].clone()).unwrap();
        assert_eq!(
            s.accept_dispatched(disp).unwrap(),
            json!({"status": "pending"})
        );
        let (resp, newly) = s.complete(id).unwrap();
        assert_eq!(resp, json!({"status": terminal}));
        if let Some(p) = newly {
            effects.push((id.to_string(), p, 1));
        }
        assert_eq!(
            b.settle(id, serde_json::from_value(resp["status"].clone()).unwrap())
                .unwrap(),
            json!({"status": terminal})
        );
    }
    assert_eq!(s.revision, 1);

    // Revocation: a pending permit is cancelled by the fence, a late first
    // delivery becomes a tombstone, and admission needs renewal plus a new epoch.
    let late = transfer("late", 2, vec![0, 0]);
    b.approve(late.clone(), 0).unwrap();
    let late_disp: DispatchedPermit =
        serde_json::from_value(b.submit(late.clone(), 10).unwrap()["permit"].clone()).unwrap();
    let p = transfer("p", 1, vec![0, 0]);
    b.approve(p.clone(), 0).unwrap();
    let disp: DispatchedPermit =
        serde_json::from_value(b.submit(p, 10).unwrap()["permit"].clone()).unwrap();
    assert_eq!(
        s.accept_dispatched(disp).unwrap(),
        json!({"status": "pending"})
    );
    assert_eq!(
        b.withdraw(0).unwrap(),
        json!({"status": "admission_revoked", "issuer": 0, "floor": 1, "charged": 8, "outstanding": 3})
    );
    assert_eq!(
        s.fence(0, 1).unwrap(),
        json!({"status": "fenced", "issuer": 0, "floor": 1})
    );
    assert_eq!(
        s.fence(0, 0).unwrap(),
        json!({"status": "fenced", "issuer": 0, "floor": 1})
    );
    assert_eq!(
        s.complete("p").unwrap(),
        (json!({"status": "cancelled"}), None)
    );
    assert_eq!(
        s.accept_dispatched(late_disp).unwrap(),
        json!({"status": "cancelled"})
    );
    assert_eq!(
        b.settle("p", Status::Cancelled).unwrap(),
        json!({"status": "cancelled"})
    );
    assert_eq!(
        b.settle("late", Status::Cancelled).unwrap(),
        json!({"status": "cancelled"})
    );
    assert_eq!(b.charged, 5);
    let stale = transfer("stale", 1, vec![0, 0]);
    assert_eq!(
        b.approve(stale.clone(), 0).unwrap()["reason"],
        "withdrawn_or_stale"
    );
    assert_eq!(b.renew(0).unwrap(), json!({"status": "renewed"}));
    let fresh = transfer("fresh", 1, vec![1, 0]);
    b.approve(fresh.clone(), 0).unwrap();
    assert_eq!(b.submit(fresh, 10).unwrap()["status"], "pending");

    let b_audit = b.audit_state("ok");
    let s_audit = s.audit_state("ok", &effects);
    for audit in [&b_audit, &s_audit] {
        assert_eq!(audit["consistent"], true, "{audit}");
        assert_eq!(audit["verus_exec_kernel_verified"], true, "{audit}");
    }
}

/// Present only when debug assertions are compiled out, so its appearance in
/// `cargo test --release` output shows the guard tests above ran without them.
#[cfg(not(debug_assertions))]
#[test]
fn release_profile_has_debug_assertions_disabled() {
    assert!(!cfg!(debug_assertions));
}
