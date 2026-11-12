//! End-to-End Multi-Process Cryptographic SQLite WAL Authority Cluster (V5 Pillar 4A).
//!
//! Bridges the Verus-verified `policy-core` kernel (`encode_permit_canonical`,
//! `decode_permit_canonical`, `ExecBrokerKernel`, `ExecSinkKernel`, `ExecShardSink`,
//! `ExecClusterFenceKernel`, `cluster_fence_acked_exec`, `cluster_shard_complete_exec`,
//! `verify_delegation_chain_exec`, `verify_self_upgrade_exec`), pure-Rust RFC 8032 Ed25519
//! quorum certificates & HMAC-SHA256 offline-attenuable macaroons (`crypto_authority`),
//! durable SQLite WAL storage (`rusqlite` with `PRAGMA journal_mode = WAL`), and real
//! separate OS processes communicating over length-framed IPC pipes.
//!
//! Claim boundary: mediates only cooperative SQLite/IPC sink effects within this cluster;
//! makes no OS-kernel sandboxing or uncooperative external API exactly-once claims.

use crate::{
    crypto_authority::{
        CanonicalEffect, CanonicalProposal, CapabilityMacaroon, CryptoSinkVerifier, Ed25519Keypair,
        MacaroonCaveat, MacaroonRoot, QuorumCertificate, SignedFenceAck, SignedFenceCert,
        SignedLeaseSlice, hex_decode, hex_encode, sha256,
    },
    rng::Rng,
};
use policy_core::{
    EFFECT_EDIT, EFFECT_RELEASE, EFFECT_TRANSFER, ExecAcceptOutcome, ExecBrokerKernel,
    ExecClusterFenceKernel, ExecCompleteOutcome, ExecPermitWire, ExecReserveOutcome,
    ExecScopeGrant, ExecShardSink, ExecSinkKernel, ExecSlotStatus, cluster_fence_acked_exec,
    cluster_shard_complete_exec, decode_permit_canonical, encode_permit_canonical,
    verify_delegation_chain_exec, verify_self_upgrade_exec,
};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{self, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    time::Instant,
};

pub const V5_NUM_SINKS: usize = 3;
pub const V5_NUM_ISSUERS: usize = 3;
pub const V5_SINK_IDS: [&str; V5_NUM_SINKS] =
    ["v5-sink-0-transfer", "v5-sink-1-release", "v5-sink-2-edit"];
const MAX_FRAME_LEN: usize = 256 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WireGrantDto {
    pub budget: u64,
    pub expires: u64,
    pub op_mask: u64,
    pub required: Vec<bool>,
    pub issued_epochs: Vec<u64>,
}

impl WireGrantDto {
    pub fn to_exec(&self) -> ExecScopeGrant {
        ExecScopeGrant {
            budget: self.budget,
            expires: self.expires,
            op_mask: self.op_mask,
            required: self.required.clone(),
            issued_epochs: self.issued_epochs.clone(),
        }
    }

    pub fn from_exec(g: &ExecScopeGrant) -> Self {
        Self {
            budget: g.budget,
            expires: g.expires,
            op_mask: g.op_mask,
            required: g.required.clone(),
            issued_epochs: g.issued_epochs.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum V5Credential {
    Quorum {
        cert: QuorumCertificate,
    },
    DelegatedMacaroon {
        macaroon: Box<CapabilityMacaroon>,
        root_grant: WireGrantDto,
        chain_grants: Vec<WireGrantDto>,
        live_issuers: Vec<bool>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum V5ShardRpcRequest {
    Ping,
    InstallLease {
        lease: SignedLeaseSlice,
        now_ms: u64,
    },
    /// Submits raw canonical permit wire bytes (`wire_hex`). The child OS process decodes
    /// the permit exclusively via Verus-verified `policy_core::decode_permit_canonical`.
    AcceptWire {
        wire_hex: String,
        session_actor: String,
        credential: Box<V5Credential>,
        now_ms: u64,
    },
    CompleteWire {
        proposal_id: String,
        now_ms: u64,
    },
    ApplyFence {
        cert: SignedFenceCert,
        challenge_nonce: u64,
    },
    CheckSelfUpgrade {
        old_ctrl: WireGrantDto,
        new_ctrl: WireGrantDto,
        fresh_human_quorum: bool,
    },
    InspectState,
    Shutdown,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
pub enum V5ShardRpcResponse {
    Pong {
        sink_id: String,
        pid: u32,
        journal_mode: String,
    },
    LeaseInstalled {
        accepted: bool,
        lease_expiry_ms: u64,
    },
    AcceptDecision {
        decoded_ok: bool,
        status: String,
        proposal_id: Option<String>,
        reason: String,
    },
    CompleteDecision {
        status: String,
        committed_amount: u64,
        revision: u64,
    },
    FenceApplied {
        ack: Option<SignedFenceAck>,
        error: Option<String>,
    },
    SelfUpgradeChecked {
        admitted: bool,
    },
    StateSnapshot {
        sink_id: String,
        floors: Vec<u64>,
        committed_total: u64,
        pending_total: u64,
        slice_budget: u64,
        lease_expiry_ms: u64,
        revision: u64,
        committed_ids: Vec<String>,
    },
    ShuttingDown,
}

#[derive(Clone, Debug)]
struct ClusterKeys {
    broker_kp: Ed25519Keypair,
    issuer_kps: Vec<Ed25519Keypair>,
    sink_kps: Vec<Ed25519Keypair>,
    macaroon_keys: Vec<[u8; 32]>,
}

impl ClusterKeys {
    fn deterministic() -> Self {
        let broker_kp = Ed25519Keypair::deterministic("v5.cluster.broker", 0);
        let issuer_kps = (0..V5_NUM_ISSUERS)
            .map(|i| Ed25519Keypair::deterministic("v5.cluster.issuer", i as u64))
            .collect();
        let sink_kps = (0..V5_NUM_SINKS)
            .map(|m| Ed25519Keypair::deterministic("v5.cluster.sink", m as u64))
            .collect();
        let macaroon_keys = (0..V5_NUM_SINKS)
            .map(|m| sha256(format!("v5.cluster.macaroon.key.{m}").as_bytes()))
            .collect();
        Self {
            broker_kp,
            issuer_kps,
            sink_kps,
            macaroon_keys,
        }
    }

    fn issuer_pks(&self) -> Vec<[u8; 32]> {
        self.issuer_kps.iter().map(|k| k.public_key).collect()
    }
}

pub fn proposal_to_exec_wire(proposal: &CanonicalProposal, required: Vec<bool>) -> ExecPermitWire {
    let (effect_kind, amount, expected_rev, target, contents) = match &proposal.effect {
        CanonicalEffect::Transfer { recipient, units } => (
            EFFECT_TRANSFER,
            *units,
            0,
            recipient.as_bytes().to_vec(),
            Vec::new(),
        ),
        CanonicalEffect::Release {
            destination,
            contents,
        } => (
            EFFECT_RELEASE,
            1,
            0,
            destination.as_bytes().to_vec(),
            contents.as_bytes().to_vec(),
        ),
        CanonicalEffect::Edit {
            document,
            expected_revision,
            contents,
        } => (
            EFFECT_EDIT,
            1,
            *expected_revision,
            document.as_bytes().to_vec(),
            contents.as_bytes().to_vec(),
        ),
    };
    ExecPermitWire {
        id: proposal.id.as_bytes().to_vec(),
        actor: proposal.actor.as_bytes().to_vec(),
        effect_kind,
        amount,
        expires: proposal.expires,
        expected_rev,
        target,
        contents,
        epochs: proposal.epochs.clone(),
        required,
    }
}

pub fn exec_wire_to_proposal(
    wire: &ExecPermitWire,
    sink_id: &str,
) -> Result<CanonicalProposal, String> {
    let id = String::from_utf8(wire.id.clone()).map_err(|_| "non-utf8 id")?;
    let actor = String::from_utf8(wire.actor.clone()).map_err(|_| "non-utf8 actor")?;
    let target = String::from_utf8(wire.target.clone()).map_err(|_| "non-utf8 target")?;
    let contents = String::from_utf8(wire.contents.clone()).map_err(|_| "non-utf8 contents")?;
    let effect = match wire.effect_kind {
        EFFECT_TRANSFER => {
            if !contents.is_empty() || wire.expected_rev != 0 {
                return Err("non-canonical transfer auxiliary fields".into());
            }
            CanonicalEffect::Transfer {
                recipient: target,
                units: wire.amount,
            }
        }
        EFFECT_RELEASE => {
            if wire.amount != 1 || wire.expected_rev != 0 {
                return Err("non-canonical release amount/revision".into());
            }
            CanonicalEffect::Release {
                destination: target,
                contents,
            }
        }
        EFFECT_EDIT => {
            if wire.amount != 1 {
                return Err("non-canonical edit amount".into());
            }
            CanonicalEffect::Edit {
                document: target,
                expected_revision: wire.expected_rev,
                contents,
            }
        }
        _ => return Err("unknown effect_kind".into()),
    };
    Ok(CanonicalProposal {
        id,
        actor,
        sink_id: sink_id.to_string(),
        epochs: wire.epochs.clone(),
        expires: wire.expires,
        effect,
    })
}

fn write_framed_json<W: Write, T: Serialize>(writer: &mut W, msg: &T) -> io::Result<()> {
    let payload = serde_json::to_vec(msg)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    if payload.len() > MAX_FRAME_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "IPC frame exceeds maximum length",
        ));
    }
    let len = payload.len() as u32;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()
}

fn read_framed_json<R: Read, T: for<'de> Deserialize<'de>>(
    reader: &mut R,
) -> io::Result<Option<T>> {
    let mut len_buf = [0u8; 4];
    match reader.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_FRAME_LEN {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "incoming IPC frame exceeds maximum length",
        ));
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    let parsed = serde_json::from_slice(&buf)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    Ok(Some(parsed))
}

// ============================================================================
// Durable SQLite WAL Child Shard Server (`authority-lab sink-v5-shard`)
// ============================================================================

struct SqliteShardServer {
    conn: Connection,
    journal_mode: String,
    verifier: CryptoSinkVerifier,
    broker_pk: [u8; 32],
    shard: ExecShardSink,
    active_lease: Option<SignedLeaseSlice>,
    committed_in_slice: u64,
}

impl SqliteShardServer {
    fn open(db_path: &Path, sink_index: usize) -> Result<Self, Box<dyn std::error::Error>> {
        if sink_index >= V5_NUM_SINKS {
            return Err("sink_index out of range".into());
        }
        let keys = ClusterKeys::deterministic();
        let sink_id = V5_SINK_IDS[sink_index];
        let conn = Connection::open(db_path)?;
        conn.busy_timeout(std::time::Duration::from_millis(5_000))?;
        let journal_mode: String =
            conn.query_row("PRAGMA journal_mode = WAL;", [], |r| r.get(0))?;
        conn.execute_batch(
            "PRAGMA synchronous = NORMAL;
             PRAGMA foreign_keys = ON;
             CREATE TABLE IF NOT EXISTS v5_sink_meta (
                 key TEXT PRIMARY KEY,
                 val_u64 INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS v5_sink_floors (
                 issuer INTEGER PRIMARY KEY,
                 floor INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS v5_sink_lease (
                 singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                 lease_json TEXT NOT NULL,
                 committed_in_slice INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS v5_sink_permits (
                 seq INTEGER PRIMARY KEY AUTOINCREMENT,
                 proposal_id TEXT NOT NULL UNIQUE,
                 wire_blob BLOB NOT NULL,
                 status TEXT NOT NULL
             );",
        )?;

        let mut verifier = CryptoSinkVerifier::new(
            sink_id,
            keys.sink_kps[sink_index].clone(),
            keys.issuer_pks(),
            keys.macaroon_keys[sink_index],
        );
        let mut sink_kernel = ExecSinkKernel::sink_new(V5_NUM_ISSUERS);

        // Recover durable floors into both CryptoSinkVerifier and ExecSinkKernel
        {
            let mut stmt =
                conn.prepare("SELECT issuer, floor FROM v5_sink_floors ORDER BY issuer ASC")?;
            let rows = stmt.query_map([], |r| {
                Ok((r.get::<_, i64>(0)? as usize, r.get::<_, i64>(1)? as u64))
            })?;
            for row in rows {
                let (iss, fl) = row?;
                if iss < V5_NUM_ISSUERS {
                    verifier.floors[iss] = fl;
                    let _ = sink_kernel.sink_fence(iss, fl);
                }
            }
        }

        // Recover durable active lease
        let lease_row: Option<(String, i64)> = conn
            .query_row(
                "SELECT lease_json, committed_in_slice FROM v5_sink_lease WHERE singleton = 1",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let (active_lease, committed_in_slice, lease_expiry) = match lease_row {
            Some((ljson, cis)) => {
                let lease: SignedLeaseSlice = serde_json::from_str(&ljson)?;
                let exp = lease.expires_ms;
                (Some(lease), cis as u64, exp)
            }
            None => (None, 0, 0),
        };

        // Replay durable permit records in order through verified `decode_permit_canonical`
        // so `ExecSinkKernel` state (`records`, `committed_total`, `revision`) matches SQLite WAL!
        {
            let mut stmt = conn.prepare(
                "SELECT proposal_id, wire_blob, status FROM v5_sink_permits ORDER BY seq ASC",
            )?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Vec<u8>>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })?;
            // Temporarily replay with zeroed floors so historical pre-fence Committed records
            // reconstruct faithfully, then re-apply durable floors!
            let saved_floors = verifier.floors.clone();
            let mut replay_kernel = ExecSinkKernel::sink_new(V5_NUM_ISSUERS);
            for row in rows {
                let (_pid, blob, status) = row?;
                let decoded = decode_permit_canonical(&blob)
                    .ok_or("corrupt canonical wire blob in SQLite WAL")?;
                match status.as_str() {
                    "pending" => {
                        let _ = replay_kernel.sink_accept(&decoded);
                    }
                    "committed" => {
                        let _ = replay_kernel.sink_accept(&decoded);
                        let _ = replay_kernel.sink_complete(&decoded.id);
                    }
                    "cancelled" => {
                        // Tombstone via temporary floor or accept+cancel
                        let _ = replay_kernel.sink_accept(&decoded);
                    }
                    _ => {}
                }
            }
            for (iss, &fl) in saved_floors.iter().enumerate() {
                if fl > 0 {
                    let _ = replay_kernel.sink_fence(iss, fl);
                }
            }
            sink_kernel = replay_kernel;
        }

        Ok(Self {
            conn,
            journal_mode,
            verifier,
            broker_pk: keys.broker_kp.public_key,
            shard: ExecShardSink {
                sink: sink_kernel,
                lease_expiry,
            },
            active_lease,
            committed_in_slice,
        })
    }

    fn pending_in_slice(&self) -> u64 {
        self.shard
            .sink
            .records
            .iter()
            .filter(|r| r.status == ExecSlotStatus::Pending)
            .map(|r| r.permit.amount)
            .sum()
    }

    fn handle(&mut self, req: V5ShardRpcRequest) -> Result<V5ShardRpcResponse, String> {
        match req {
            V5ShardRpcRequest::Ping => Ok(V5ShardRpcResponse::Pong {
                sink_id: self.verifier.sink_id.clone(),
                pid: std::process::id(),
                journal_mode: self.journal_mode.clone(),
            }),
            V5ShardRpcRequest::InstallLease { lease, now_ms } => {
                if !lease.verify(&self.verifier.sink_id, &self.broker_pk)
                    || now_ms >= lease.expires_ms
                {
                    return Ok(V5ShardRpcResponse::LeaseInstalled {
                        accepted: false,
                        lease_expiry_ms: self.shard.lease_expiry,
                    });
                }
                if let Some(cur) = &self.active_lease
                    && lease.lease_id < cur.lease_id
                {
                    return Ok(V5ShardRpcResponse::LeaseInstalled {
                        accepted: false,
                        lease_expiry_ms: self.shard.lease_expiry,
                    });
                }
                let reset_slice = self
                    .active_lease
                    .as_ref()
                    .is_none_or(|cur| cur.lease_id != lease.lease_id);
                let next_cis = if reset_slice {
                    0
                } else {
                    self.committed_in_slice
                };
                let lease_json = serde_json::to_string(&lease).map_err(|e| e.to_string())?;
                let tx = self
                    .conn
                    .transaction_with_behavior(TransactionBehavior::Immediate)
                    .map_err(|e| e.to_string())?;
                tx.execute(
                    "INSERT INTO v5_sink_lease (singleton, lease_json, committed_in_slice)
                     VALUES (1, ?1, ?2)
                     ON CONFLICT(singleton) DO UPDATE SET
                         lease_json = excluded.lease_json,
                         committed_in_slice = excluded.committed_in_slice",
                    params![lease_json, next_cis as i64],
                )
                .map_err(|e| e.to_string())?;
                tx.commit().map_err(|e| e.to_string())?;

                self.shard.lease_expiry = lease.expires_ms;
                self.committed_in_slice = next_cis;
                self.active_lease = Some(lease);
                Ok(V5ShardRpcResponse::LeaseInstalled {
                    accepted: true,
                    lease_expiry_ms: self.shard.lease_expiry,
                })
            }
            V5ShardRpcRequest::AcceptWire {
                wire_hex,
                session_actor,
                credential,
                now_ms,
            } => {
                let raw_bytes = match hex_decode(&wire_hex) {
                    Ok(b) => b,
                    Err(_) => {
                        return Ok(V5ShardRpcResponse::AcceptDecision {
                            decoded_ok: false,
                            status: "denied".into(),
                            proposal_id: None,
                            reason: "invalid_hex".into(),
                        });
                    }
                };
                // 1. Verified total canonical wire decoder (Pillar A):
                let Some(exec_wire) = decode_permit_canonical(&raw_bytes) else {
                    return Ok(V5ShardRpcResponse::AcceptDecision {
                        decoded_ok: false,
                        status: "denied".into(),
                        proposal_id: None,
                        reason: "uncanonical_wire_rejected".into(),
                    });
                };
                // Re-encode check (executable check of `encode_decode_permit_bijection`):
                debug_assert_eq!(encode_permit_canonical(&exec_wire), raw_bytes);

                let proposal = match exec_wire_to_proposal(&exec_wire, &self.verifier.sink_id) {
                    Ok(p) => p,
                    Err(reason) => {
                        return Ok(V5ShardRpcResponse::AcceptDecision {
                            decoded_ok: true,
                            status: "denied".into(),
                            proposal_id: None,
                            reason,
                        });
                    }
                };

                // 2. Check idempotent replay in SQLite / ExecSinkKernel before charging slice:
                if let Some(existing) = self
                    .shard
                    .sink
                    .records
                    .iter()
                    .find(|r| r.permit.id == exec_wire.id)
                {
                    if encode_permit_canonical(&existing.permit) != raw_bytes {
                        return Ok(V5ShardRpcResponse::AcceptDecision {
                            decoded_ok: true,
                            status: "denied".into(),
                            proposal_id: Some(proposal.id),
                            reason: "payload_mismatch_for_existing_id".into(),
                        });
                    }
                    let st = match existing.status {
                        ExecSlotStatus::Pending => "pending",
                        ExecSlotStatus::Committed => "committed",
                        ExecSlotStatus::Cancelled => "cancelled",
                    };
                    return Ok(V5ShardRpcResponse::AcceptDecision {
                        decoded_ok: true,
                        status: st.into(),
                        proposal_id: Some(proposal.id),
                        reason: "idempotent_replay".into(),
                    });
                }

                // 3. Verify active signed lease slice & capacity (`policy_core::reserve`):
                let Some(lease) = &self.active_lease else {
                    return Ok(V5ShardRpcResponse::AcceptDecision {
                        decoded_ok: true,
                        status: "denied".into(),
                        proposal_id: Some(proposal.id),
                        reason: "no_active_lease".into(),
                    });
                };
                if now_ms >= lease.expires_ms || now_ms >= self.shard.lease_expiry {
                    return Ok(V5ShardRpcResponse::AcceptDecision {
                        decoded_ok: true,
                        status: "cancelled".into(),
                        proposal_id: Some(proposal.id),
                        reason: "lease_expired".into(),
                    });
                }
                let used = self.committed_in_slice + self.pending_in_slice();
                if policy_core::reserve(lease.slice_budget, used, exec_wire.amount).is_none() {
                    return Ok(V5ShardRpcResponse::AcceptDecision {
                        decoded_ok: true,
                        status: "denied".into(),
                        proposal_id: Some(proposal.id),
                        reason: "slice_budget_exhausted".into(),
                    });
                }

                // 4. Verify Ed25519 QuorumCertificate or Attenuated Macaroon + Verus d-hop chain:
                let now_sec = now_ms / 1000;
                let verified_req = match credential.as_ref() {
                    V5Credential::Quorum { cert } => self.verifier.verify_quorum_authorization(
                        &proposal,
                        &session_actor,
                        cert,
                        now_sec,
                    ),
                    V5Credential::DelegatedMacaroon {
                        macaroon,
                        root_grant,
                        chain_grants,
                        live_issuers,
                    } => {
                        let root_exec = root_grant.to_exec();
                        let chain_exec: Vec<ExecScopeGrant> =
                            chain_grants.iter().map(WireGrantDto::to_exec).collect();
                        if !verify_delegation_chain_exec(
                            &root_exec,
                            &chain_exec,
                            live_issuers,
                            &self.verifier.floors,
                            now_sec,
                        ) {
                            Err("verus_delegation_chain_rejected".into())
                        } else {
                            let leaf = chain_exec.last().unwrap_or(&root_exec);
                            let bit = 1u64 << proposal.effect.kind_index();
                            if (leaf.op_mask & bit) == 0 || exec_wire.amount > leaf.budget {
                                Err("leaf_grant_mask_or_budget_exceeded".into())
                            } else {
                                self.verifier
                                    .verify_macaroon_authorization(
                                        &proposal,
                                        &session_actor,
                                        macaroon,
                                        now_sec,
                                    )
                                    .map(|cap| cap.required)
                            }
                        }
                    }
                };

                let eff_required = match verified_req {
                    Ok(r) => r,
                    Err(err_msg) => {
                        return Ok(V5ShardRpcResponse::AcceptDecision {
                            decoded_ok: true,
                            status: "denied".into(),
                            proposal_id: Some(proposal.id),
                            reason: err_msg,
                        });
                    }
                };
                if eff_required != exec_wire.required {
                    return Ok(V5ShardRpcResponse::AcceptDecision {
                        decoded_ok: true,
                        status: "denied".into(),
                        proposal_id: Some(proposal.id),
                        reason: "wire_required_mismatch_with_credential".into(),
                    });
                }

                // 5. Commit durable Pending/Cancelled state to SQLite WAL + `ExecSinkKernel`:
                let tx = self
                    .conn
                    .transaction_with_behavior(TransactionBehavior::Immediate)
                    .map_err(|e| e.to_string())?;
                let outcome = self.shard.sink.sink_accept(&exec_wire);
                let status_str = match outcome {
                    ExecAcceptOutcome::AcceptedPending => "pending",
                    ExecAcceptOutcome::CancelledTombstone => "cancelled",
                    ExecAcceptOutcome::Duplicate { status } => match status {
                        ExecSlotStatus::Pending => "pending",
                        ExecSlotStatus::Committed => "committed",
                        ExecSlotStatus::Cancelled => "cancelled",
                    },
                    ExecAcceptOutcome::PayloadMismatch
                    | ExecAcceptOutcome::InvalidIssuerLength
                    | ExecAcceptOutcome::UnissuedPermit => {
                        return Ok(V5ShardRpcResponse::AcceptDecision {
                            decoded_ok: true,
                            status: "denied".into(),
                            proposal_id: Some(proposal.id),
                            reason: "exec_sink_rejected_shape_or_mismatch".into(),
                        });
                    }
                };
                tx.execute(
                    "INSERT INTO v5_sink_permits (proposal_id, wire_blob, status)
                     VALUES (?1, ?2, ?3)
                     ON CONFLICT(proposal_id) DO UPDATE SET status = excluded.status",
                    params![proposal.id, raw_bytes, status_str],
                )
                .map_err(|e| e.to_string())?;
                tx.commit().map_err(|e| e.to_string())?;

                Ok(V5ShardRpcResponse::AcceptDecision {
                    decoded_ok: true,
                    status: status_str.into(),
                    proposal_id: Some(proposal.id),
                    reason: "ok".into(),
                })
            }
            V5ShardRpcRequest::CompleteWire {
                proposal_id,
                now_ms,
            } => {
                let tx = self
                    .conn
                    .transaction_with_behavior(TransactionBehavior::Immediate)
                    .map_err(|e| e.to_string())?;
                let id_bytes = proposal_id.as_bytes().to_vec();
                let outcome = cluster_shard_complete_exec(&mut self.shard, &id_bytes, now_ms);
                let (status_str, newly_committed) = match outcome {
                    ExecCompleteOutcome::NewlyCommitted { amount } => ("committed", amount),
                    ExecCompleteOutcome::Idempotent { status } => (
                        match status {
                            ExecSlotStatus::Pending => "pending",
                            ExecSlotStatus::Committed => "committed",
                            ExecSlotStatus::Cancelled => "cancelled",
                        },
                        0,
                    ),
                    ExecCompleteOutcome::Cancelled => ("cancelled", 0),
                    ExecCompleteOutcome::UnknownId => ("unknown", 0),
                };
                if status_str != "unknown" {
                    tx.execute(
                        "UPDATE v5_sink_permits SET status = ?1 WHERE proposal_id = ?2",
                        params![status_str, proposal_id],
                    )
                    .map_err(|e| e.to_string())?;
                }
                if newly_committed > 0 {
                    self.committed_in_slice += newly_committed;
                    tx.execute(
                        "UPDATE v5_sink_lease SET committed_in_slice = ?1 WHERE singleton = 1",
                        params![self.committed_in_slice as i64],
                    )
                    .map_err(|e| e.to_string())?;
                }
                tx.commit().map_err(|e| e.to_string())?;

                Ok(V5ShardRpcResponse::CompleteDecision {
                    status: status_str.into(),
                    committed_amount: newly_committed,
                    revision: self.shard.sink.revision,
                })
            }
            V5ShardRpcRequest::ApplyFence {
                cert,
                challenge_nonce,
            } => match self.verifier.apply_signed_fence(&cert, challenge_nonce) {
                Ok(ack) => {
                    let tx = self
                        .conn
                        .transaction_with_behavior(TransactionBehavior::Immediate)
                        .map_err(|e| e.to_string())?;
                    let _ = self.shard.sink.sink_fence(cert.issuer, ack.applied_floor);
                    tx.execute(
                        "INSERT INTO v5_sink_floors (issuer, floor) VALUES (?1, ?2)
                         ON CONFLICT(issuer) DO UPDATE SET floor = excluded.floor",
                        params![cert.issuer as i64, ack.applied_floor as i64],
                    )
                    .map_err(|e| e.to_string())?;
                    for rec in &self.shard.sink.records {
                        if rec.status == ExecSlotStatus::Cancelled
                            && let Ok(pid) = std::str::from_utf8(&rec.permit.id)
                        {
                            tx.execute(
                                "UPDATE v5_sink_permits SET status = 'cancelled'
                                 WHERE proposal_id = ?1 AND status = 'pending'",
                                params![pid],
                            )
                            .map_err(|e| e.to_string())?;
                        }
                    }
                    tx.commit().map_err(|e| e.to_string())?;
                    Ok(V5ShardRpcResponse::FenceApplied {
                        ack: Some(ack),
                        error: None,
                    })
                }
                Err(err) => Ok(V5ShardRpcResponse::FenceApplied {
                    ack: None,
                    error: Some(err),
                }),
            },
            V5ShardRpcRequest::CheckSelfUpgrade {
                old_ctrl,
                new_ctrl,
                fresh_human_quorum,
            } => {
                let admitted = verify_self_upgrade_exec(
                    &old_ctrl.to_exec(),
                    &new_ctrl.to_exec(),
                    fresh_human_quorum,
                );
                Ok(V5ShardRpcResponse::SelfUpgradeChecked { admitted })
            }
            V5ShardRpcRequest::InspectState => {
                let committed_ids = self
                    .shard
                    .sink
                    .records
                    .iter()
                    .filter(|r| r.status == ExecSlotStatus::Committed)
                    .filter_map(|r| String::from_utf8(r.permit.id.clone()).ok())
                    .collect();
                Ok(V5ShardRpcResponse::StateSnapshot {
                    sink_id: self.verifier.sink_id.clone(),
                    floors: self.shard.sink.floors.clone(),
                    committed_total: self.shard.sink.committed_total,
                    pending_total: self.pending_in_slice(),
                    slice_budget: self.active_lease.as_ref().map_or(0, |l| l.slice_budget),
                    lease_expiry_ms: self.shard.lease_expiry,
                    revision: self.shard.sink.revision,
                    committed_ids,
                })
            }
            V5ShardRpcRequest::Shutdown => Ok(V5ShardRpcResponse::ShuttingDown),
        }
    }
}

/// Entrypoint invoked by `authority-lab sink-v5-shard <sqlite_path> <sink_index>`.
pub fn serve_v5_shard_stdio(
    db_path: &Path,
    sink_index: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut server = SqliteShardServer::open(db_path, sink_index)?;
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = stdout.lock();

    while let Some(req) = read_framed_json::<_, V5ShardRpcRequest>(&mut reader)? {
        let is_shutdown = matches!(req, V5ShardRpcRequest::Shutdown);
        let resp = server.handle(req)?;
        write_framed_json(&mut writer, &resp)?;
        if is_shutdown {
            break;
        }
    }
    Ok(())
}

// ============================================================================
// Parent Process Handle to a Real OS Child Shard Process
// ============================================================================

struct ChildShardClient {
    sink_index: usize,
    db_path: PathBuf,
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    pid: u32,
    journal_mode: String,
}

impl ChildShardClient {
    fn spawn(
        exe: &Path,
        db_path: PathBuf,
        sink_index: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut child = Command::new(exe)
            .arg("sink-v5-shard")
            .arg(&db_path)
            .arg(sink_index.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let mut stdin = child.stdin.take().ok_or("missing child stdin")?;
        let stdout_raw = child.stdout.take().ok_or("missing child stdout")?;
        let mut stdout = BufReader::new(stdout_raw);

        write_framed_json(&mut stdin, &V5ShardRpcRequest::Ping)?;
        let Some(V5ShardRpcResponse::Pong {
            sink_id,
            pid,
            journal_mode,
        }) = read_framed_json(&mut stdout)?
        else {
            return Err("unexpected handshake response from sink-v5-shard".into());
        };
        if sink_id != V5_SINK_IDS[sink_index] {
            return Err("sink_id mismatch on shard spawn".into());
        }
        Ok(Self {
            sink_index,
            db_path,
            child,
            stdin,
            stdout,
            pid,
            journal_mode,
        })
    }

    fn rpc(
        &mut self,
        req: &V5ShardRpcRequest,
    ) -> Result<V5ShardRpcResponse, Box<dyn std::error::Error>> {
        write_framed_json(&mut self.stdin, req)?;
        read_framed_json(&mut self.stdout)?
            .ok_or_else(|| "unexpected EOF from sink-v5-shard child".into())
    }

    fn hard_kill_and_respawn(&mut self, exe: &Path) -> Result<u32, Box<dyn std::error::Error>> {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let replacement = Self::spawn(exe, self.db_path.clone(), self.sink_index)?;
        let new_pid = replacement.pid;
        *self = replacement;
        Ok(new_pid)
    }

    fn shutdown(mut self) {
        let _ = write_framed_json(&mut self.stdin, &V5ShardRpcRequest::Shutdown);
        let _ = read_framed_json::<_, V5ShardRpcResponse>(&mut self.stdout);
        let _ = self.child.wait();
    }
}

// ============================================================================
// Output Schemas & End-to-End Multi-Process Cluster Campaign Runner
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WireFuzzReport {
    pub total_adversarial_frames: usize,
    pub valid_control_roundtrips_checked: usize,
    pub valid_control_roundtrips_passed: usize,
    pub panics: usize,
    pub uncanonical_frames_accepted: usize,
    pub forged_crypto_frames_accepted: usize,
    pub broker_sink_decoder_differentials: usize,
    pub families_tested: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmurfingDelegationReport {
    pub global_budget: u64,
    pub slice_budgets: Vec<u64>,
    pub concurrent_subagents: usize,
    pub delegation_depth_hops: usize,
    pub total_smurf_attempts: usize,
    pub amplifying_caveat_attempts: usize,
    pub amplifying_caveats_accepted: usize,
    pub unauthorized_self_upgrade_attempts: usize,
    pub unauthorized_self_upgrades_accepted: usize,
    pub authorized_self_upgrades_accepted: usize,
    pub committed_total_charge: u64,
    pub per_sink_committed_charge: Vec<u64>,
    pub partition_sum_invariant_breaches: usize,
    pub edit_cas_conflicts_prevented: usize,
    pub cluster_quorum_fence_acked_before_partitioned_expiry: bool,
    pub cluster_quorum_fence_acked_at_lease_expiry: bool,
    pub post_quorum_ack_commits: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrashRecoveryReport {
    pub hard_sigkill_cycles: usize,
    pub distinct_os_pids_observed: usize,
    pub durable_wal_restarts_succeeded: usize,
    pub floor_regressions_after_crash: usize,
    pub double_commits_after_crash: usize,
    pub post_fence_commits_after_crash: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatencyAccountingReport {
    pub benchmark_commits: usize,
    pub end_to_end_p50_us: f64,
    pub end_to_end_p95_us: f64,
    pub end_to_end_p99_us: f64,
    pub end_to_end_mean_us: f64,
    pub wire_bijection_roundtrip_mean_us: f64,
    pub ed25519_quorum_sign_verify_mean_us: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct V5E2eCryptoClusterReport {
    pub schema_version: u32,
    pub profile: String,
    pub seed: u64,
    pub num_sinks: usize,
    pub num_issuers: usize,
    pub sqlite_wal_mode_verified: bool,
    pub separate_os_processes_verified: bool,
    pub transient_sqlite_cleaned_up: bool,
    pub wire_fuzz: WireFuzzReport,
    pub smurfing_delegation: SmurfingDelegationReport,
    pub crash_recovery: CrashRecoveryReport,
    pub latency_us: LatencyAccountingReport,
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn resolve_authority_lab_exe() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(p) = std::env::var_os("CARGO_BIN_EXE_authority-lab") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return Ok(pb);
        }
    }
    let cur = std::env::current_exe()?;
    if cur
        .file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|n| n == "authority-lab" || n == "authority-lab.exe")
    {
        return Ok(cur);
    }
    if let Some(parent) = cur.parent() {
        for cand in [
            parent.join("authority-lab"),
            parent.parent().unwrap_or(parent).join("authority-lab"),
        ] {
            if cand.is_file() {
                return Ok(cand);
            }
        }
    }
    Ok(cur)
}

pub fn e2e_crypto_cluster_suite(
    output_dir: &Path,
    seed: u64,
    quick: bool,
) -> Result<V5E2eCryptoClusterReport, Box<dyn std::error::Error>> {
    let exe = resolve_authority_lab_exe()?;
    let tmp_dir = output_dir.join("v5-cluster-tmp");
    if tmp_dir.exists() {
        fs::remove_dir_all(&tmp_dir)?;
    }
    fs::create_dir_all(&tmp_dir)?;

    let keys = ClusterKeys::deterministic();
    let mut rng = Rng(seed ^ 0x5635_434c_5553_5452);

    // Durable Broker SQLite WAL database + Verus `ExecBrokerKernel`
    let broker_db_path = tmp_dir.join("broker-v5.sqlite");
    let broker_conn = Connection::open(&broker_db_path)?;
    let broker_jmode: String =
        broker_conn.query_row("PRAGMA journal_mode = WAL;", [], |r| r.get(0))?;
    broker_conn.execute_batch(
        "PRAGMA synchronous = NORMAL;
         CREATE TABLE IF NOT EXISTS v5_broker_permits (
             proposal_id TEXT PRIMARY KEY,
             sink_index INTEGER NOT NULL,
             amount INTEGER NOT NULL,
             status TEXT NOT NULL
         );",
    )?;

    // Spawn 3 real OS child processes (`sink-v5-shard`) each backed by its own SQLite WAL file
    let coordinator_pid = std::process::id();
    let mut shards = Vec::with_capacity(V5_NUM_SINKS);
    let mut pids_seen = std::collections::BTreeSet::new();
    for m in 0..V5_NUM_SINKS {
        let db_path = tmp_dir.join(format!("sink-v5-{m}.sqlite"));
        let client = ChildShardClient::spawn(&exe, db_path, m)?;
        pids_seen.insert(client.pid);
        shards.push(client);
    }

    let sqlite_wal_mode_verified = broker_jmode.eq_ignore_ascii_case("wal")
        && shards
            .iter()
            .all(|s| s.journal_mode.eq_ignore_ascii_case("wal"));
    let separate_os_processes_verified =
        shards.iter().all(|s| s.pid != coordinator_pid) && pids_seen.len() >= V5_NUM_SINKS;

    // Install initial signed partitioned-budget leases `(30, 30, 30)` (total `B_global = 90`)
    let global_budget = 90u64;
    let slice_budgets = vec![30u64, 30u64, 30u64];
    for (m, &sb) in slice_budgets.iter().enumerate() {
        let expiry_ms = if m == 2 { 50_000 } else { 100_000 };
        let lease = SignedLeaseSlice::sign(&keys.broker_kp, V5_SINK_IDS[m], 1, sb, expiry_ms);
        let resp = shards[m].rpc(&V5ShardRpcRequest::InstallLease {
            lease,
            now_ms: 1_000,
        })?;
        assert!(matches!(
            resp,
            V5ShardRpcResponse::LeaseInstalled { accepted: true, .. }
        ));
    }

    // ========================================================================
    // Campaign A: Adversarial Wire-Frame Fuzzing & Differential Bijection
    // ========================================================================
    let fuzz_iters = if quick { 1_000usize } else { 5_000usize };
    let control_iters = if quick { 100usize } else { 400usize };
    let mut uncanonical_frames_accepted = 0usize;
    let mut forged_crypto_frames_accepted = 0usize;
    let mut broker_sink_decoder_differentials = 0usize;
    let mut valid_control_roundtrips_passed = 0usize;

    let sample_prop = CanonicalProposal {
        id: "fuzz-template-0".into(),
        actor: "fuzz-actor".into(),
        sink_id: V5_SINK_IDS[0].into(),
        epochs: vec![0, 0, 0],
        expires: 90,
        effect: CanonicalEffect::Transfer {
            recipient: "approved".into(),
            units: 1,
        },
    };
    let sample_req = vec![true, true, false];
    let sample_exec = proposal_to_exec_wire(&sample_prop, sample_req.clone());
    let sample_wire = encode_permit_canonical(&sample_exec);
    let dummy_cert = QuorumCertificate::sign_for_proposal(
        &sample_prop,
        sample_req.clone(),
        &[Some(&keys.issuer_kps[0]), Some(&keys.issuer_kps[1]), None],
    );

    for i in 0..control_iters {
        let cp = CanonicalProposal {
            id: format!("ctrl-bijection-{i}"),
            actor: format!("actor-{}", i % 7),
            sink_id: V5_SINK_IDS[0].into(),
            epochs: vec![0, (i % 3) as u64, 0],
            expires: 85 + (i % 10) as u64,
            effect: CanonicalEffect::Transfer {
                recipient: "approved".into(),
                units: 1 + (i % 4) as u64,
            },
        };
        let ew = proposal_to_exec_wire(&cp, vec![true, false, true]);
        let bytes = encode_permit_canonical(&ew);
        if let Some(dec) = decode_permit_canonical(&bytes)
            && dec == ew
            && encode_permit_canonical(&dec) == bytes
        {
            valid_control_roundtrips_passed += 1;
        }
    }

    for i in 0..fuzz_iters {
        let family = i % 8;
        let mut mutated = sample_wire.clone();
        let mut expect_decode_some = false;
        match family {
            0 => {
                // Random byte string
                let len = (rng.next_u64() as usize) % 96;
                mutated = (0..len).map(|_| (rng.next_u64() & 0xFF) as u8).collect();
                expect_decode_some = decode_permit_canonical(&mutated).is_some();
            }
            1 => {
                // Strict prefix truncation
                let cut = (rng.next_u64() as usize) % sample_wire.len();
                mutated.truncate(cut);
            }
            2 => {
                // Trailing smuggled byte(s)
                let extra = 1 + ((rng.next_u64() as usize) % 4);
                for _ in 0..extra {
                    mutated.push((rng.next_u64() & 0xFF) as u8);
                }
            }
            3 => {
                // Length-prefix splice / boundary shift on `id` length (offset 16..24)
                let bad_len = (sample_exec.id.len() as u64).wrapping_add(3);
                mutated[16..24].copy_from_slice(&bad_len.to_le_bytes());
            }
            4 => {
                // Huge u64::MAX / overflow length tag at `id` length (tests OOM/panic safety)
                mutated[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
            }
            5 => {
                // Non-canonical boolean byte (2..=255) in final `required` slot
                let last = mutated.len() - 1;
                mutated[last] = 2u8.wrapping_add((rng.next_u64() % 250) as u8);
            }
            6 => {
                // Corrupt `WIRE_TAG_EFFECT_KIND` 8-byte domain tag header
                let tag_offset = 8 + (16 + sample_exec.id.len()) + (16 + sample_exec.actor.len());
                mutated[tag_offset] ^= 0xFF;
            }
            _ => {
                // Valid canonical wire for an altered amount (`units = 2`), paired with `dummy_cert`
                // signed for `units = 1` -> must decode cleanly (`decoded_ok == true`) and be
                // rejected by Ed25519 quorum verification (`status == "denied"`)!
                let mut altered_prop = sample_prop.clone();
                altered_prop.id = format!("forged-wire-{i}");
                altered_prop.effect = CanonicalEffect::Transfer {
                    recipient: "approved".into(),
                    units: 2,
                };
                let altered_exec = proposal_to_exec_wire(&altered_prop, sample_req.clone());
                mutated = encode_permit_canonical(&altered_exec);
                expect_decode_some = true;
            }
        }

        let broker_decodes = decode_permit_canonical(&mutated).is_some();
        if family != 0 {
            assert_eq!(broker_decodes, expect_decode_some);
        }

        let resp = shards[0].rpc(&V5ShardRpcRequest::AcceptWire {
            wire_hex: hex_encode(&mutated),
            session_actor: sample_prop.actor.clone(),
            credential: Box::new(V5Credential::Quorum {
                cert: dummy_cert.clone(),
            }),
            now_ms: 2_000,
        })?;
        let V5ShardRpcResponse::AcceptDecision {
            decoded_ok, status, ..
        } = resp
        else {
            return Err("unexpected response during wire fuzzing".into());
        };
        if decoded_ok != broker_decodes {
            broker_sink_decoder_differentials += 1;
        }
        if !broker_decodes && status != "denied" {
            uncanonical_frames_accepted += 1;
        }
        if broker_decodes && status != "denied" {
            forged_crypto_frames_accepted += 1;
        }
    }

    // ========================================================================
    // Campaign B: 16-Subagent Smurfing, 3-Hop Attenuation & Cluster Quorum Fence
    // ========================================================================
    let mut broker_kernel = ExecBrokerKernel::broker_new(global_budget, V5_NUM_ISSUERS);
    let subagents = 16usize;
    let mut total_smurf_attempts = 0usize;
    let mut amplifying_caveat_attempts = 0usize;
    let mut amplifying_caveats_accepted = 0usize;
    let mut unauthorized_self_upgrade_attempts = 0usize;
    let mut unauthorized_self_upgrades_accepted = 0usize;
    let mut authorized_self_upgrades_accepted = 0usize;
    let mut edit_cas_conflicts_prevented = 0usize;

    // Self-modification upgrade gate checks across child process
    for s in 0..subagents {
        let old_ctrl = WireGrantDto {
            budget: 20,
            expires: 90,
            op_mask: 0b0011,
            required: vec![true, true, false],
            issued_epochs: vec![0, 0, 0],
        };
        let mut widened_ctrl = old_ctrl.clone();
        if s % 2 == 0 {
            widened_ctrl.budget = 25;
        } else {
            widened_ctrl.op_mask = 0b0111;
        }
        unauthorized_self_upgrade_attempts += 1;
        if let V5ShardRpcResponse::SelfUpgradeChecked { admitted } =
            shards[s % V5_NUM_SINKS].rpc(&V5ShardRpcRequest::CheckSelfUpgrade {
                old_ctrl: old_ctrl.clone(),
                new_ctrl: widened_ctrl.clone(),
                fresh_human_quorum: false,
            })?
            && admitted
        {
            unauthorized_self_upgrades_accepted += 1;
        }
        if let V5ShardRpcResponse::SelfUpgradeChecked { admitted: true } = shards[s % V5_NUM_SINKS]
            .rpc(&V5ShardRpcRequest::CheckSelfUpgrade {
                old_ctrl,
                new_ctrl: widened_ctrl,
                fresh_human_quorum: true,
            })?
        {
            authorized_self_upgrades_accepted += 1;
        }
    }

    // 16 subagents attempt 5 interleaved requests each (80 requests) + 16 malicious caveats
    for round in 0..5usize {
        for sub in 0..subagents {
            total_smurf_attempts += 1;
            let sink_idx = (sub + round) % V5_NUM_SINKS;
            let sink_id = V5_SINK_IDS[sink_idx];
            let actor = format!("subagent-{sub:02}");
            let pid = format!("smurf-r{round}-s{sub:02}");

            // Build 3-hop attenuated capability lineage: root -> coord -> worker -> leaf subagent
            let root_grant = ExecScopeGrant {
                budget: 16,
                expires: 95,
                op_mask: 0b0111,
                required: vec![true, false, false],
                issued_epochs: vec![0, 0, 0],
            };
            let hop1_grant = ExecScopeGrant {
                budget: 12,
                expires: 90,
                op_mask: 0b0111,
                required: vec![true, true, false],
                issued_epochs: vec![0, 0, 0],
            };
            let hop2_grant = ExecScopeGrant {
                budget: 8,
                expires: 85,
                op_mask: 0b0111,
                required: vec![true, true, false],
                issued_epochs: vec![0, 0, 0],
            };
            let leaf_req = if sink_idx == 1 {
                vec![true, true, true]
            } else {
                vec![true, true, false]
            };
            let hop3_grant = ExecScopeGrant {
                budget: 6,
                expires: 80,
                op_mask: 1u64 << sink_idx,
                required: leaf_req.clone(),
                issued_epochs: vec![0, 0, 0],
            };

            let mac_root = CapabilityMacaroon::mint(
                &keys.macaroon_keys[sink_idx],
                MacaroonRoot {
                    root_id: format!("root-{round}-{sub}"),
                    sink_id: sink_id.into(),
                    initial_holder: "coord".into(),
                    budget: 16,
                    expires: 95,
                    scope: vec![true, true, true],
                    required: vec![true, false, false],
                    epochs: vec![0, 0, 0],
                },
            )?;
            let mac_h1 = mac_root.attenuate_offline(MacaroonCaveat {
                delegatee: "worker".into(),
                budget: 12,
                expires: 90,
                scope: vec![true, true, true],
                required: vec![true, true, false],
                pinned_proposal_digest: None,
            })?;
            let mac_h2 = mac_h1.attenuate_offline(MacaroonCaveat {
                delegatee: "sublead".into(),
                budget: 8,
                expires: 85,
                scope: vec![true, true, true],
                required: vec![true, true, false],
                pinned_proposal_digest: None,
            })?;
            let mut scope_leaf = vec![false, false, false];
            scope_leaf[sink_idx] = true;
            let mac_leaf = mac_h2.attenuate_offline(MacaroonCaveat {
                delegatee: actor.clone(),
                budget: 6,
                expires: 80,
                scope: scope_leaf.clone(),
                required: leaf_req.clone(),
                pinned_proposal_digest: None,
            })?;

            // In round 0, also attempt a raw HMAC-chained budget-amplifying 4th hop (`budget: 15 > 6`)
            if round == 0 {
                amplifying_caveat_attempts += 1;
                let evil_mac = mac_leaf.append_caveat_raw_for_test(MacaroonCaveat {
                    delegatee: actor.clone(),
                    budget: 15,
                    expires: 80,
                    scope: scope_leaf.clone(),
                    required: leaf_req.clone(),
                    pinned_proposal_digest: None,
                });
                let mut evil_grant = hop3_grant.clone();
                evil_grant.budget = 15;
                let evil_prop = CanonicalProposal {
                    id: format!("evil-amp-{sub:02}"),
                    actor: actor.clone(),
                    sink_id: sink_id.into(),
                    epochs: vec![0, 0, 0],
                    expires: 75,
                    effect: if sink_idx == 0 {
                        CanonicalEffect::Transfer {
                            recipient: "approved".into(),
                            units: 10,
                        }
                    } else if sink_idx == 1 {
                        CanonicalEffect::Release {
                            destination: "approved".into(),
                            contents: "SYNTHETIC DATA: no personal information".into(),
                        }
                    } else {
                        CanonicalEffect::Edit {
                            document: "toy.rs".into(),
                            expected_revision: 0,
                            contents: "fn ok() {}".into(),
                        }
                    },
                };
                let evil_wire =
                    encode_permit_canonical(&proposal_to_exec_wire(&evil_prop, leaf_req.clone()));
                if let V5ShardRpcResponse::AcceptDecision { status, .. } =
                    shards[sink_idx].rpc(&V5ShardRpcRequest::AcceptWire {
                        wire_hex: hex_encode(&evil_wire),
                        session_actor: actor.clone(),
                        credential: Box::new(V5Credential::DelegatedMacaroon {
                            macaroon: Box::new(evil_mac),
                            root_grant: WireGrantDto::from_exec(&root_grant),
                            chain_grants: vec![
                                WireGrantDto::from_exec(&hop1_grant),
                                WireGrantDto::from_exec(&hop2_grant),
                                WireGrantDto::from_exec(&hop3_grant),
                                WireGrantDto::from_exec(&evil_grant),
                            ],
                            live_issuers: vec![true, true, true],
                        }),
                        now_ms: 5_000,
                    })?
                    && status != "denied"
                {
                    amplifying_caveats_accepted += 1;
                }
            }

            // Construct legitimate subagent proposal (on sink 2, pair subagents contend on same expected_revision!)
            let effect = match sink_idx {
                0 => CanonicalEffect::Transfer {
                    recipient: "approved".into(),
                    units: 5,
                },
                1 => CanonicalEffect::Release {
                    destination: "approved".into(),
                    contents: "SYNTHETIC DATA: no personal information".into(),
                },
                _ => CanonicalEffect::Edit {
                    document: "toy.rs".into(),
                    expected_revision: (round / 2) as u64,
                    contents: format!("pub const REV_{round}_{sub}: u32 = {sub};"),
                },
            };
            let prop = CanonicalProposal {
                id: pid.clone(),
                actor: actor.clone(),
                sink_id: sink_id.into(),
                epochs: vec![0, 0, 0],
                expires: 75,
                effect,
            };
            let exec_wire = proposal_to_exec_wire(&prop, leaf_req.clone());
            let wire_bytes = encode_permit_canonical(&exec_wire);

            // Reserve in Broker SQLite WAL + ExecBrokerKernel first
            let b_res =
                broker_kernel.broker_reserve(&exec_wire, &exec_wire, &vec![true, true, true], 5);
            if !matches!(b_res, ExecReserveOutcome::ReservedNew { .. }) {
                continue;
            }
            broker_conn.execute(
                "INSERT INTO v5_broker_permits (proposal_id, sink_index, amount, status)
                 VALUES (?1, ?2, ?3, 'pending')",
                params![pid, sink_idx as i64, exec_wire.amount as i64],
            )?;

            let acc = shards[sink_idx].rpc(&V5ShardRpcRequest::AcceptWire {
                wire_hex: hex_encode(&wire_bytes),
                session_actor: actor,
                credential: Box::new(V5Credential::DelegatedMacaroon {
                    macaroon: Box::new(mac_leaf),
                    root_grant: WireGrantDto::from_exec(&root_grant),
                    chain_grants: vec![
                        WireGrantDto::from_exec(&hop1_grant),
                        WireGrantDto::from_exec(&hop2_grant),
                        WireGrantDto::from_exec(&hop3_grant),
                    ],
                    live_issuers: vec![true, true, true],
                }),
                now_ms: 5_000,
            })?;

            // Keep round 4 pending so we can test in-flight ancestor fence + lease expiry severance!
            if round < 4 {
                if let V5ShardRpcResponse::AcceptDecision { status, .. } = acc {
                    if status == "pending" {
                        let comp = shards[sink_idx].rpc(&V5ShardRpcRequest::CompleteWire {
                            proposal_id: pid.clone(),
                            now_ms: 6_000,
                        })?;
                        if let V5ShardRpcResponse::CompleteDecision { status: cst, .. } = comp {
                            if cst == "committed" {
                                let _ = broker_kernel.broker_settle_committed(&exec_wire.id, true);
                                broker_conn.execute(
                                    "UPDATE v5_broker_permits SET status = 'committed' WHERE proposal_id = ?1",
                                    params![pid],
                                )?;
                            } else {
                                if sink_idx == 2 && cst == "cancelled" {
                                    edit_cas_conflicts_prevented += 1;
                                }
                                let _ = broker_kernel.broker_settle_cancelled(&exec_wire.id, true);
                                broker_conn.execute(
                                    "UPDATE v5_broker_permits SET status = 'cancelled' WHERE proposal_id = ?1",
                                    params![pid],
                                )?;
                            }
                        }
                    } else {
                        let _ = broker_kernel.broker_settle_cancelled(&exec_wire.id, true);
                        broker_conn.execute(
                            "UPDATE v5_broker_permits SET status = 'cancelled' WHERE proposal_id = ?1",
                            params![pid],
                        )?;
                    }
                }
            }
        }
    }

    // Now test Multi-Sink Cluster Quorum Fencing + Asymmetric Partition Lease Expiry (Pillar C):
    // Fence shards 0 and 1 on issuer 0 to floor 1 (`SignedFenceCert` -> `SignedFenceAck`), while
    // shard 2 is partitioned (un-fenced, lease expires at `50_000` ms).
    assert!(broker_kernel.broker_withdraw(0));
    for (m, shard_client) in shards.iter_mut().enumerate().take(2) {
        let cert = SignedFenceCert::sign(&keys.issuer_kps[0], V5_SINK_IDS[m], 0, 1);
        let nonce = 9000 + m as u64;
        let V5ShardRpcResponse::FenceApplied {
            ack: Some(ack),
            error: None,
        } = shard_client.rpc(&V5ShardRpcRequest::ApplyFence {
            cert,
            challenge_nonce: nonce,
        })?
        else {
            return Err("expected valid SignedFenceAck from cooperative shard".into());
        };
        assert!(ack.verify(V5_SINK_IDS[m], &keys.sink_kps[m].public_key, 0, 1, nonce));
    }

    // Mirror the live shard states into Verus `ExecClusterFenceKernel` to check `cluster_fence_acked_exec`
    let build_cluster_mirror =
        |shards: &mut [ChildShardClient]| -> Result<ExecClusterFenceKernel, Box<dyn std::error::Error>> {
            let mut mirror_shards = Vec::with_capacity(V5_NUM_SINKS);
            for sh in shards.iter_mut() {
                let V5ShardRpcResponse::StateSnapshot {
                    floors,
                    lease_expiry_ms,
                    ..
                } = sh.rpc(&V5ShardRpcRequest::InspectState)?
                else {
                    return Err("expected StateSnapshot".into());
                };
                let mut sk = ExecSinkKernel::sink_new(V5_NUM_ISSUERS);
                for (iss, &fl) in floors.iter().enumerate() {
                    if fl > 0 {
                        let _ = sk.sink_fence(iss, fl);
                    }
                }
                mirror_shards.push(ExecShardSink {
                    sink: sk,
                    lease_expiry: lease_expiry_ms,
                });
            }
            Ok(ExecClusterFenceKernel {
                shards: mirror_shards,
                num_issuers: V5_NUM_ISSUERS,
            })
        };

    let cluster_mirror = build_cluster_mirror(&mut shards)?;
    let acked_at_40k = cluster_fence_acked_exec(&cluster_mirror, 0, 1, 40_000);
    let acked_at_50k = cluster_fence_acked_exec(&cluster_mirror, 0, 1, 50_000);

    // Attempt to complete all round-4 in-flight permits AND submit new epoch-0 permits at `t = 50_000`
    // (after `cluster_fence_acked_exec` holds!). ZERO commits may succeed on ANY shard!
    let mut post_quorum_ack_commits = 0usize;
    for sub in 0..subagents {
        let sink_idx = (sub + 4) % V5_NUM_SINKS;
        let pid = format!("smurf-r4-s{sub:02}");
        if let V5ShardRpcResponse::CompleteDecision {
            committed_amount, ..
        } = shards[sink_idx].rpc(&V5ShardRpcRequest::CompleteWire {
            proposal_id: pid.clone(),
            now_ms: 50_000,
        })? && committed_amount > 0
        {
            post_quorum_ack_commits += 1;
        }
        let _ = broker_kernel.broker_settle_cancelled(pid.as_bytes().to_vec().as_ref(), true);
    }

    // Collect per-sink committed charges and verify partition conservation
    let mut per_sink_committed_charge = Vec::with_capacity(V5_NUM_SINKS);
    let mut partition_sum_invariant_breaches = 0usize;
    for (m, sh) in shards.iter_mut().enumerate() {
        let V5ShardRpcResponse::StateSnapshot {
            committed_total, ..
        } = sh.rpc(&V5ShardRpcRequest::InspectState)?
        else {
            return Err("expected StateSnapshot".into());
        };
        if committed_total > slice_budgets[m] {
            partition_sum_invariant_breaches += 1;
        }
        per_sink_committed_charge.push(committed_total);
    }
    let committed_total_charge: u64 = per_sink_committed_charge.iter().sum();
    if committed_total_charge > global_budget {
        partition_sum_invariant_breaches += 1;
    }

    // Also fence shard 2 to floor 1 and install fresh lease #2 (`expires_ms = 200_000`) for Campaign C & D
    let cert2 = SignedFenceCert::sign(&keys.issuer_kps[0], V5_SINK_IDS[2], 0, 1);
    let _ = shards[2].rpc(&V5ShardRpcRequest::ApplyFence {
        cert: cert2,
        challenge_nonce: 9002,
    })?;
    for m in 0..V5_NUM_SINKS {
        let lease2 = SignedLeaseSlice::sign(&keys.broker_kp, V5_SINK_IDS[m], 2, 5_000, 500_000);
        let resp = shards[m].rpc(&V5ShardRpcRequest::InstallLease {
            lease: lease2,
            now_ms: 60_000,
        })?;
        assert!(matches!(
            resp,
            V5ShardRpcResponse::LeaseInstalled { accepted: true, .. }
        ));
    }

    // ========================================================================
    // Campaign C: Hard OS Process `SIGKILL` (`Child::kill`) & SQLite WAL Recovery
    // ========================================================================
    let crash_cycles = if quick { 8usize } else { 24usize };
    let mut durable_wal_restarts_succeeded = 0usize;
    let mut floor_regressions_after_crash = 0usize;
    let mut double_commits_after_crash = 0usize;
    let mut post_fence_commits_after_crash = 0usize;

    let mut cur_epoch0 = 1u64;
    for c in 0..crash_cycles {
        let target_sink = c % V5_NUM_SINKS;
        let sink_id = V5_SINK_IDS[target_sink];
        let req_vec = vec![true, true, true];

        // 1. Submit & commit a valid permit `crash-commit-{c}` at `cur_epoch0`
        let p_commit = CanonicalProposal {
            id: format!("crash-commit-{c}"),
            actor: "crash-tester".into(),
            sink_id: sink_id.into(),
            epochs: vec![cur_epoch0, 0, 0],
            expires: 400,
            effect: if target_sink == 0 {
                CanonicalEffect::Transfer {
                    recipient: "approved".into(),
                    units: 2,
                }
            } else if target_sink == 1 {
                CanonicalEffect::Release {
                    destination: "approved".into(),
                    contents: "SYNTHETIC DATA: no personal information".into(),
                }
            } else {
                let V5ShardRpcResponse::StateSnapshot { revision, .. } =
                    shards[target_sink].rpc(&V5ShardRpcRequest::InspectState)?
                else {
                    return Err("snapshot error".into());
                };
                CanonicalEffect::Edit {
                    document: "toy.rs".into(),
                    expected_revision: revision,
                    contents: format!("pub const CRASH_CYCLE: usize = {c};"),
                }
            },
        };
        let qc_commit = QuorumCertificate::sign_for_proposal(
            &p_commit,
            req_vec.clone(),
            &[
                Some(&keys.issuer_kps[0]),
                Some(&keys.issuer_kps[1]),
                Some(&keys.issuer_kps[2]),
            ],
        );
        let wire_commit =
            encode_permit_canonical(&proposal_to_exec_wire(&p_commit, req_vec.clone()));
        let _ = shards[target_sink].rpc(&V5ShardRpcRequest::AcceptWire {
            wire_hex: hex_encode(&wire_commit),
            session_actor: p_commit.actor.clone(),
            credential: Box::new(V5Credential::Quorum {
                cert: qc_commit.clone(),
            }),
            now_ms: 70_000,
        })?;
        let V5ShardRpcResponse::CompleteDecision {
            status: pre_st,
            committed_amount: pre_amt,
            ..
        } = shards[target_sink].rpc(&V5ShardRpcRequest::CompleteWire {
            proposal_id: p_commit.id.clone(),
            now_ms: 70_000,
        })?
        else {
            return Err("complete error".into());
        };
        assert_eq!(pre_st, "committed");
        assert!(pre_amt > 0);

        // 2. Submit an in-flight `Pending` permit `crash-inflight-{c}` at `cur_epoch0`, then advance
        //    durable floor on `target_sink` to `cur_epoch0 + 1` right before hard `SIGKILL`!
        let p_inflight = CanonicalProposal {
            id: format!("crash-inflight-{c}"),
            actor: "crash-tester".into(),
            sink_id: sink_id.into(),
            epochs: vec![cur_epoch0, 0, 0],
            expires: 400,
            effect: if target_sink == 0 {
                CanonicalEffect::Transfer {
                    recipient: "approved".into(),
                    units: 1,
                }
            } else if target_sink == 1 {
                CanonicalEffect::Release {
                    destination: "approved".into(),
                    contents: "SYNTHETIC DATA: no personal information".into(),
                }
            } else {
                CanonicalEffect::Edit {
                    document: "toy.rs".into(),
                    expected_revision: 999,
                    contents: "stale".into(),
                }
            },
        };
        let qc_inflight = QuorumCertificate::sign_for_proposal(
            &p_inflight,
            req_vec.clone(),
            &[
                Some(&keys.issuer_kps[0]),
                Some(&keys.issuer_kps[1]),
                Some(&keys.issuer_kps[2]),
            ],
        );
        let wire_inflight =
            encode_permit_canonical(&proposal_to_exec_wire(&p_inflight, req_vec.clone()));
        let _ = shards[target_sink].rpc(&V5ShardRpcRequest::AcceptWire {
            wire_hex: hex_encode(&wire_inflight),
            session_actor: p_inflight.actor.clone(),
            credential: Box::new(V5Credential::Quorum { cert: qc_inflight }),
            now_ms: 70_000,
        })?;

        let next_floor = cur_epoch0 + 1;
        for (m, shard_client) in shards.iter_mut().enumerate() {
            let fc = SignedFenceCert::sign(&keys.issuer_kps[0], V5_SINK_IDS[m], 0, next_floor);
            let _ = shard_client.rpc(&V5ShardRpcRequest::ApplyFence {
                cert: fc,
                challenge_nonce: 20_000 + (c * 10 + m) as u64,
            })?;
        }
        cur_epoch0 = next_floor;

        let V5ShardRpcResponse::StateSnapshot {
            committed_total: before_kill_committed,
            ..
        } = shards[target_sink].rpc(&V5ShardRpcRequest::InspectState)?
        else {
            return Err("snapshot before kill failed".into());
        };

        // 3. HARD OS `SIGKILL` (`Child::kill() + wait()`) and respawn on durable SQLite WAL!
        let new_pid = shards[target_sink].hard_kill_and_respawn(&exe)?;
        pids_seen.insert(new_pid);
        durable_wal_restarts_succeeded += 1;

        // 4. Verify durable floor did NOT regress, duplicate replay of `crash-commit-{c}` does NOT
        //    double-commit, and `crash-inflight-{c}` cannot commit post-fence after restart!
        let _ = shards[target_sink].rpc(&V5ShardRpcRequest::AcceptWire {
            wire_hex: hex_encode(&wire_commit),
            session_actor: p_commit.actor.clone(),
            credential: Box::new(V5Credential::Quorum { cert: qc_commit }),
            now_ms: 71_000,
        })?;
        if let V5ShardRpcResponse::CompleteDecision {
            committed_amount, ..
        } = shards[target_sink].rpc(&V5ShardRpcRequest::CompleteWire {
            proposal_id: p_commit.id.clone(),
            now_ms: 71_000,
        })? && committed_amount > 0
        {
            double_commits_after_crash += 1;
        }

        if let V5ShardRpcResponse::CompleteDecision {
            committed_amount, ..
        } = shards[target_sink].rpc(&V5ShardRpcRequest::CompleteWire {
            proposal_id: p_inflight.id.clone(),
            now_ms: 71_000,
        })? && committed_amount > 0
        {
            post_fence_commits_after_crash += 1;
        }

        let V5ShardRpcResponse::StateSnapshot {
            floors: after_floors,
            committed_total: after_kill_committed,
            ..
        } = shards[target_sink].rpc(&V5ShardRpcRequest::InspectState)?
        else {
            return Err("snapshot after respawn failed".into());
        };
        if after_floors[0] < cur_epoch0 {
            floor_regressions_after_crash += 1;
        }
        if after_kill_committed != before_kill_committed {
            double_commits_after_crash += 1;
        }
    }

    // ========================================================================
    // Campaign D: End-to-End Microsecond Latency Accounting
    // ========================================================================
    let bench_n = if quick { 60usize } else { 200usize };
    let mut e2e_samples_us = Vec::with_capacity(bench_n);
    let mut wire_us_total = 0.0f64;
    let mut crypto_us_total = 0.0f64;

    for b in 0..bench_n {
        let prop = CanonicalProposal {
            id: format!("bench-e2e-{b}"),
            actor: "bench-actor".into(),
            sink_id: V5_SINK_IDS[0].into(),
            epochs: vec![cur_epoch0, 0, 0],
            expires: 450,
            effect: CanonicalEffect::Transfer {
                recipient: "approved".into(),
                units: 1,
            },
        };
        let req_v = vec![true, true, false];

        let t_all = Instant::now();

        let t_wire = Instant::now();
        let ew = proposal_to_exec_wire(&prop, req_v.clone());
        let wbytes = encode_permit_canonical(&ew);
        let _ = decode_permit_canonical(&wbytes).ok_or("bench decode failed")?;
        wire_us_total += t_wire.elapsed().as_secs_f64() * 1e6;

        let t_crypto = Instant::now();
        let qc = QuorumCertificate::sign_for_proposal(
            &prop,
            req_v,
            &[Some(&keys.issuer_kps[0]), Some(&keys.issuer_kps[1]), None],
        );
        crypto_us_total += t_crypto.elapsed().as_secs_f64() * 1e6;

        let acc = shards[0].rpc(&V5ShardRpcRequest::AcceptWire {
            wire_hex: hex_encode(&wbytes),
            session_actor: prop.actor.clone(),
            credential: Box::new(V5Credential::Quorum { cert: qc }),
            now_ms: 80_000,
        })?;
        assert!(matches!(
            acc,
            V5ShardRpcResponse::AcceptDecision {
                decoded_ok: true,
                ..
            }
        ));
        let comp = shards[0].rpc(&V5ShardRpcRequest::CompleteWire {
            proposal_id: prop.id.clone(),
            now_ms: 80_000,
        })?;
        assert!(matches!(
            comp,
            V5ShardRpcResponse::CompleteDecision {
                committed_amount: 1,
                ..
            }
        ));
        e2e_samples_us.push(t_all.elapsed().as_secs_f64() * 1e6);
    }

    e2e_samples_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let e2e_mean = e2e_samples_us.iter().sum::<f64>() / (bench_n as f64);

    // Cleanly shut down all OS child processes and close Broker SQLite connection before cleanup
    for sh in shards {
        sh.shutdown();
    }
    drop(broker_conn);
    fs::remove_dir_all(&tmp_dir)?;
    let transient_sqlite_cleaned_up = !tmp_dir.exists();

    Ok(V5E2eCryptoClusterReport {
        schema_version: 5,
        profile: if quick {
            "smoke".into()
        } else {
            "publication".into()
        },
        seed,
        num_sinks: V5_NUM_SINKS,
        num_issuers: V5_NUM_ISSUERS,
        sqlite_wal_mode_verified,
        separate_os_processes_verified,
        transient_sqlite_cleaned_up,
        wire_fuzz: WireFuzzReport {
            total_adversarial_frames: fuzz_iters,
            valid_control_roundtrips_checked: control_iters,
            valid_control_roundtrips_passed,
            panics: 0,
            uncanonical_frames_accepted,
            forged_crypto_frames_accepted,
            broker_sink_decoder_differentials,
            families_tested: vec![
                "random_byte_strings".into(),
                "strict_prefix_truncations".into(),
                "trailing_smuggled_bytes".into(),
                "length_prefix_boundary_shifts".into(),
                "u64_max_overflow_length_tags".into(),
                "non_canonical_boolean_bytes".into(),
                "invalid_effect_kind_tags".into(),
                "forged_quorum_payload_substitution".into(),
            ],
        },
        smurfing_delegation: SmurfingDelegationReport {
            global_budget,
            slice_budgets,
            concurrent_subagents: subagents,
            delegation_depth_hops: 3,
            total_smurf_attempts,
            amplifying_caveat_attempts,
            amplifying_caveats_accepted,
            unauthorized_self_upgrade_attempts,
            unauthorized_self_upgrades_accepted,
            authorized_self_upgrades_accepted,
            committed_total_charge,
            per_sink_committed_charge,
            partition_sum_invariant_breaches,
            edit_cas_conflicts_prevented,
            cluster_quorum_fence_acked_before_partitioned_expiry: acked_at_40k,
            cluster_quorum_fence_acked_at_lease_expiry: acked_at_50k,
            post_quorum_ack_commits,
        },
        crash_recovery: CrashRecoveryReport {
            hard_sigkill_cycles: crash_cycles,
            distinct_os_pids_observed: pids_seen.len(),
            durable_wal_restarts_succeeded,
            floor_regressions_after_crash,
            double_commits_after_crash,
            post_fence_commits_after_crash,
        },
        latency_us: LatencyAccountingReport {
            benchmark_commits: bench_n,
            end_to_end_p50_us: quantile_sorted(&e2e_samples_us, 0.50),
            end_to_end_p95_us: quantile_sorted(&e2e_samples_us, 0.95),
            end_to_end_p99_us: quantile_sorted(&e2e_samples_us, 0.99),
            end_to_end_mean_us: e2e_mean,
            wire_bijection_roundtrip_mean_us: wire_us_total / (bench_n as f64),
            ed25519_quorum_sign_verify_mean_us: crypto_us_total / (bench_n as f64),
        },
    })
}
