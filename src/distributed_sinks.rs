//! Multi-sink distributed authority cluster, partitioned-budget bounded-lease
//! quorum fencing, and adversarial Jepsen-style fault nemesis over real UNIX sockets.
//!
//! Architecture:
//! - `CoordinatorBroker` + `M = 3` distinct sink worker threads (`sink-0-transfer`,
//!   `sink-1-release`, `sink-2-edit-ext`) communicating over real Linux `AF_UNIX`
//!   datagram sockets (`std::os::unix::net::UnixDatagram`).
//! - Global budget `B` is partitioned into non-overlapping lease slices `(B_0, B_1, B_2)`
//!   satisfying `sum(B_m) + B_unallocated <= B` with bounded lease TTL `Delta_lease`.
//! - Adversarial Jepsen Nemesis interposes on `(Broker <-> Sink_m)` and
//!   `(Proposer <-> Sink_m)` links with `PacketDrop`, `PacketDelay`, `PacketReorder`,
//!   `PacketDuplicate`, `AsymmetricPartition` (Broker<->Sink_k severed while
//!   Proposer<->Sink_k remains live), MITM forged/stale `FenceAck` injection, and
//!   simulated `CrashRestart` (wiping volatile state while recovering durable state).
//! - Contrasts 3 sink/protocol classes across `1,000` randomized Jepsen schedules:
//!   1. `CooperativeSignedFence`
//!   2. `PrematureBrokerOnlyAck`
//!   3. `UncooperativeExternalSink`

use crate::crypto_authority::{
    CanonicalEffect, CanonicalProposal, CapabilityMacaroon, CryptoRedTeamReport,
    CryptoSinkVerifier, Ed25519Keypair, MacaroonCaveat, MacaroonRoot, QuorumCertificate,
    SignedFenceAck, SignedFenceCert, SignedLeaseSlice, run_crypto_redteam_campaign, sha256,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs,
    io::{Read, Write},
    os::unix::net::UnixStream,
    path::Path,
    sync::Mutex,
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

pub const NUM_SINKS: usize = 3;
pub const SINK_IDS: [&str; NUM_SINKS] = ["sink-0-transfer", "sink-1-release", "sink-2-edit-ext"];

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SinkProtocolClass {
    CooperativeSignedFence,
    PrematureBrokerOnlyAck,
    UncooperativeExternalSink,
}

impl SinkProtocolClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CooperativeSignedFence => "cooperative_signed_fence",
            Self::PrematureBrokerOnlyAck => "premature_broker_only_ack",
            Self::UncooperativeExternalSink => "uncooperative_external_sink",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SinkRecordStatus {
    Pending,
    Committed,
    Cancelled,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommittedEffectRecord {
    pub proposal_id: String,
    pub sink_id: String,
    pub actor: String,
    pub charge: u64,
    pub epochs: Vec<u64>,
    pub required: Vec<bool>,
    pub commit_time_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DurableSinkSnapshot {
    floors: Vec<u64>,
    active_lease: Option<SignedLeaseSlice>,
    committed_in_slice: u64,
    records: HashMap<String, (CanonicalProposal, Vec<bool>, SinkRecordStatus)>,
    committed_effects: Vec<CommittedEffectRecord>,
    edit_revision: u64,
}

impl DurableSinkSnapshot {
    fn fresh(issuers: usize) -> Self {
        Self {
            floors: vec![0; issuers],
            active_lease: None,
            committed_in_slice: 0,
            records: HashMap::new(),
            committed_effects: Vec::new(),
            edit_revision: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AuthCredential {
    Quorum { cert: QuorumCertificate },
    Macaroon { macaroon: CapabilityMacaroon },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum SinkRpcRequest {
    ResetSchedule {
        class: SinkProtocolClass,
    },
    InstallLease {
        lease: SignedLeaseSlice,
        now_ms: u64,
    },
    Accept {
        proposal: CanonicalProposal,
        session_actor: String,
        credential: Box<AuthCredential>,
        now_ms: u64,
    },
    Complete {
        proposal_id: String,
        now_ms: u64,
    },
    ApplyFence {
        cert: SignedFenceCert,
        challenge_nonce: u64,
        now_ms: u64,
    },
    CrashRestart,
    InspectState,
    Shutdown,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
enum SinkRpcResponse {
    AckReset,
    LeaseInstalled {
        accepted: bool,
    },
    Decision {
        proposal_id: String,
        status: String,
        commit_time_ms: Option<u64>,
    },
    FenceAck {
        ack: Option<SignedFenceAck>,
        error: Option<String>,
    },
    Restarted {
        recovered_floors: Vec<u64>,
        recovered_effects: usize,
    },
    StateReport {
        sink_id: String,
        floors: Vec<u64>,
        committed_in_slice: u64,
        pending_in_slice: u64,
        slice_budget: u64,
        committed_effects: Vec<CommittedEffectRecord>,
    },
    ShuttingDown,
}

struct SinkRuntime {
    verifier: CryptoSinkVerifier,
    broker_pk: [u8; 32],
    class: SinkProtocolClass,
    durable: DurableSinkSnapshot,
    volatile_accept_counter: u64,
}

impl SinkRuntime {
    fn new(
        sink_id: &str,
        sink_kp: Ed25519Keypair,
        broker_pk: [u8; 32],
        issuer_pks: Vec<[u8; 32]>,
        macaroon_key: [u8; 32],
    ) -> Self {
        let issuers = issuer_pks.len();
        Self {
            verifier: CryptoSinkVerifier::new(sink_id, sink_kp, issuer_pks, macaroon_key),
            broker_pk,
            class: SinkProtocolClass::CooperativeSignedFence,
            durable: DurableSinkSnapshot::fresh(issuers),
            volatile_accept_counter: 0,
        }
    }

    fn pending_in_slice(&self) -> u64 {
        self.durable
            .records
            .values()
            .filter(|(_, _, st)| *st == SinkRecordStatus::Pending)
            .map(|(p, _, _)| p.effect.charge())
            .sum()
    }

    fn handle(&mut self, req: SinkRpcRequest) -> SinkRpcResponse {
        match req {
            SinkRpcRequest::ResetSchedule { class } => {
                self.class = class;
                self.durable = DurableSinkSnapshot::fresh(self.verifier.issuer_pks.len());
                self.verifier.floors.fill(0);
                self.volatile_accept_counter = 0;
                SinkRpcResponse::AckReset
            }
            SinkRpcRequest::InstallLease { lease, now_ms } => {
                if !lease.verify(&self.verifier.sink_id, &self.broker_pk)
                    || now_ms >= lease.expires_ms
                {
                    return SinkRpcResponse::LeaseInstalled { accepted: false };
                }
                let newer = self
                    .durable
                    .active_lease
                    .as_ref()
                    .is_none_or(|cur| lease.lease_id >= cur.lease_id);
                if !newer {
                    return SinkRpcResponse::LeaseInstalled { accepted: false };
                }
                if self
                    .durable
                    .active_lease
                    .as_ref()
                    .is_none_or(|cur| cur.lease_id != lease.lease_id)
                {
                    self.durable.committed_in_slice = 0;
                }
                self.durable.active_lease = Some(lease);
                SinkRpcResponse::LeaseInstalled { accepted: true }
            }
            SinkRpcRequest::Accept {
                proposal,
                session_actor,
                credential,
                now_ms,
            } => {
                // Idempotent replay handling for already-recorded proposal IDs
                if let Some((prev_prop, _, prev_status)) = self.durable.records.get(&proposal.id) {
                    if prev_prop != &proposal {
                        return SinkRpcResponse::Decision {
                            proposal_id: proposal.id,
                            status: "denied".into(),
                            commit_time_ms: None,
                        };
                    }
                    let st = match prev_status {
                        SinkRecordStatus::Pending => "pending",
                        SinkRecordStatus::Committed => "committed",
                        SinkRecordStatus::Cancelled => "cancelled",
                    };
                    return SinkRpcResponse::Decision {
                        proposal_id: proposal.id,
                        status: st.into(),
                        commit_time_ms: None,
                    };
                }

                // Verify active partitioned lease slice & capacity via policy_core::reserve
                let Some(lease) = &self.durable.active_lease else {
                    return SinkRpcResponse::Decision {
                        proposal_id: proposal.id,
                        status: "denied".into(),
                        commit_time_ms: None,
                    };
                };
                if now_ms >= lease.expires_ms {
                    return SinkRpcResponse::Decision {
                        proposal_id: proposal.id,
                        status: "cancelled".into(),
                        commit_time_ms: None,
                    };
                }
                let amount = proposal.effect.charge();
                let used = self.durable.committed_in_slice + self.pending_in_slice();
                if policy_core::reserve(lease.slice_budget, used, amount).is_none() {
                    return SinkRpcResponse::Decision {
                        proposal_id: proposal.id,
                        status: "denied".into(),
                        commit_time_ms: None,
                    };
                }

                // Verify cryptographic authority (Ed25519 QuorumCertificate or HMAC Macaroon)
                let now_sec = now_ms / 1000;
                let auth_req = match credential.as_ref() {
                    AuthCredential::Quorum { cert } => self.verifier.verify_quorum_authorization(
                        &proposal,
                        &session_actor,
                        cert,
                        now_sec,
                    ),
                    AuthCredential::Macaroon { macaroon } => self
                        .verifier
                        .verify_macaroon_authorization(&proposal, &session_actor, macaroon, now_sec)
                        .map(|cap| cap.required),
                };

                let Ok(eff_required) = auth_req else {
                    // Distinguish epoch-fenced cancellation from invalid capability denial
                    let base_req = proposal
                        .effect
                        .base_required(self.verifier.issuer_pks.len());
                    let is_fenced = proposal.epochs.len() == self.verifier.floors.len()
                        && !policy_core::unfenced(
                            &base_req,
                            &proposal.epochs,
                            &self.verifier.floors,
                        );
                    if is_fenced {
                        self.durable.records.insert(
                            proposal.id.clone(),
                            (proposal.clone(), base_req, SinkRecordStatus::Cancelled),
                        );
                        return SinkRpcResponse::Decision {
                            proposal_id: proposal.id,
                            status: "cancelled".into(),
                            commit_time_ms: None,
                        };
                    }
                    return SinkRpcResponse::Decision {
                        proposal_id: proposal.id,
                        status: "denied".into(),
                        commit_time_ms: None,
                    };
                };

                self.volatile_accept_counter += 1;
                self.durable.records.insert(
                    proposal.id.clone(),
                    (proposal.clone(), eff_required, SinkRecordStatus::Pending),
                );
                SinkRpcResponse::Decision {
                    proposal_id: proposal.id,
                    status: "pending".into(),
                    commit_time_ms: None,
                }
            }
            SinkRpcRequest::Complete {
                proposal_id,
                now_ms,
            } => {
                let Some((proposal, eff_required, status)) =
                    self.durable.records.get(&proposal_id).cloned()
                else {
                    return SinkRpcResponse::Decision {
                        proposal_id,
                        status: "denied".into(),
                        commit_time_ms: None,
                    };
                };

                match status {
                    SinkRecordStatus::Committed => SinkRpcResponse::Decision {
                        proposal_id,
                        status: "committed".into(),
                        commit_time_ms: None,
                    },
                    SinkRecordStatus::Cancelled => SinkRpcResponse::Decision {
                        proposal_id,
                        status: "cancelled".into(),
                        commit_time_ms: None,
                    },
                    SinkRecordStatus::Pending => {
                        let cooperative_precommit_check =
                            self.class != SinkProtocolClass::UncooperativeExternalSink;
                        let lease_live = self
                            .durable
                            .active_lease
                            .as_ref()
                            .is_some_and(|l| now_ms < l.expires_ms);
                        let unfenced = policy_core::unfenced(
                            &eff_required,
                            &proposal.epochs,
                            &self.verifier.floors,
                        );
                        let rev_ok = match &proposal.effect {
                            CanonicalEffect::Edit {
                                expected_revision, ..
                            } => policy_core::revision_matches(
                                self.durable.edit_revision,
                                *expected_revision,
                            ),
                            _ => true,
                        };

                        if (cooperative_precommit_check && (!lease_live || !unfenced)) || !rev_ok {
                            if let Some(entry) = self.durable.records.get_mut(&proposal_id) {
                                entry.2 = SinkRecordStatus::Cancelled;
                            }
                            return SinkRpcResponse::Decision {
                                proposal_id,
                                status: "cancelled".into(),
                                commit_time_ms: None,
                            };
                        }

                        let amount = proposal.effect.charge();
                        self.durable.committed_in_slice += amount;
                        if matches!(proposal.effect, CanonicalEffect::Edit { .. }) {
                            self.durable.edit_revision += 1;
                        }
                        if let Some(entry) = self.durable.records.get_mut(&proposal_id) {
                            entry.2 = SinkRecordStatus::Committed;
                        }
                        self.durable.committed_effects.push(CommittedEffectRecord {
                            proposal_id: proposal.id.clone(),
                            sink_id: self.verifier.sink_id.clone(),
                            actor: proposal.actor.clone(),
                            charge: amount,
                            epochs: proposal.epochs.clone(),
                            required: eff_required,
                            commit_time_ms: now_ms,
                        });
                        SinkRpcResponse::Decision {
                            proposal_id,
                            status: "committed".into(),
                            commit_time_ms: Some(now_ms),
                        }
                    }
                }
            }
            SinkRpcRequest::ApplyFence {
                cert,
                challenge_nonce,
                now_ms: _,
            } => match self.verifier.apply_signed_fence(&cert, challenge_nonce) {
                Ok(ack) => {
                    self.durable.floors = self.verifier.floors.clone();
                    if self.class != SinkProtocolClass::UncooperativeExternalSink {
                        for (prop, req, st) in self.durable.records.values_mut() {
                            if *st == SinkRecordStatus::Pending
                                && !policy_core::unfenced(req, &prop.epochs, &self.verifier.floors)
                            {
                                *st = SinkRecordStatus::Cancelled;
                            }
                        }
                    }
                    SinkRpcResponse::FenceAck {
                        ack: Some(ack),
                        error: None,
                    }
                }
                Err(e) => SinkRpcResponse::FenceAck {
                    ack: None,
                    error: Some(e),
                },
            },
            SinkRpcRequest::CrashRestart => {
                // Wipe volatile state and recoververifier floors from durable snapshot
                self.volatile_accept_counter = 0;
                self.verifier.floors = self.durable.floors.clone();
                SinkRpcResponse::Restarted {
                    recovered_floors: self.verifier.floors.clone(),
                    recovered_effects: self.durable.committed_effects.len(),
                }
            }
            SinkRpcRequest::InspectState => {
                let slice_budget = self
                    .durable
                    .active_lease
                    .as_ref()
                    .map_or(0, |l| l.slice_budget);
                SinkRpcResponse::StateReport {
                    sink_id: self.verifier.sink_id.clone(),
                    floors: self.verifier.floors.clone(),
                    committed_in_slice: self.durable.committed_in_slice,
                    pending_in_slice: self.pending_in_slice(),
                    slice_budget,
                    committed_effects: self.durable.committed_effects.clone(),
                }
            }
            SinkRpcRequest::Shutdown => SinkRpcResponse::ShuttingDown,
        }
    }
}

fn write_framed_packet(stream: &mut UnixStream, payload: &[u8]) -> std::io::Result<()> {
    let len = u32::try_from(payload.len()).expect("frame fits u32");
    stream.write_all(&len.to_le_bytes())?;
    stream.write_all(payload)?;
    stream.flush()
}

fn read_framed_packet(stream: &mut UnixStream) -> std::io::Result<Vec<u8>> {
    let mut hdr = [0u8; 4];
    stream.read_exact(&mut hdr)?;
    let len = u32::from_le_bytes(hdr) as usize;
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;
    Ok(buf)
}

struct SinkSocketWorker {
    proxy_sock: Mutex<UnixStream>,
    handle: Option<JoinHandle<()>>,
}

impl SinkSocketWorker {
    fn spawn(mut runtime: SinkRuntime) -> Self {
        let (proxy_sock, mut worker_sock) =
            UnixStream::pair().expect("create AF_UNIX stream socket pair");
        proxy_sock
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("set proxy read timeout");
        worker_sock
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("set worker read timeout");

        let handle = thread::spawn(move || {
            while let Ok(buf) = read_framed_packet(&mut worker_sock) {
                let req: SinkRpcRequest =
                    serde_json::from_slice(&buf).expect("decode sink RPC over AF_UNIX");
                let is_shutdown = matches!(req, SinkRpcRequest::Shutdown);
                let resp = runtime.handle(req);
                let encoded = serde_json::to_vec(&resp).expect("encode sink RPC reply");
                write_framed_packet(&mut worker_sock, &encoded)
                    .expect("send sink RPC reply over AF_UNIX");
                if is_shutdown {
                    break;
                }
            }
        });

        Self {
            proxy_sock: Mutex::new(proxy_sock),
            handle: Some(handle),
        }
    }

    fn call(&self, req: &SinkRpcRequest) -> SinkRpcResponse {
        let payload = serde_json::to_vec(req).expect("serialize SinkRpcRequest");
        let mut sock = self.proxy_sock.lock().expect("lock AF_UNIX proxy socket");
        write_framed_packet(&mut sock, &payload).expect("AF_UNIX send to sink thread");
        let reply = read_framed_packet(&mut sock).expect("AF_UNIX recv from sink thread");
        serde_json::from_slice(&reply).expect("deserialize SinkRpcResponse")
    }

    fn shutdown(mut self) {
        let _ = self.call(&SinkRpcRequest::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

// ============================================================================
// CoordinatorBroker & Partitioned-Budget Lease Ledger
// ============================================================================

#[derive(Clone, Debug)]
struct CoordinatorLeaseEntry {
    lease_id: u64,
    slice_budget: u64,
    expires_ms: u64,
    fenced_for_epoch: Vec<u64>,
}

struct CoordinatorBroker {
    global_budget: u64,
    unallocated_budget: u64,
    broker_kp: Ed25519Keypair,
    issuer_kps: Vec<Ed25519Keypair>,
    sink_pks: [[u8; 32]; NUM_SINKS],
    epochs: Vec<u64>,
    live: Vec<bool>,
    leases: [CoordinatorLeaseEntry; NUM_SINKS],
    next_challenge_nonce: u64,
}

impl CoordinatorBroker {
    fn new(
        global_budget: u64,
        broker_kp: Ed25519Keypair,
        issuer_kps: Vec<Ed25519Keypair>,
        sink_pks: [[u8; 32]; NUM_SINKS],
    ) -> Self {
        let n = issuer_kps.len();
        Self {
            global_budget,
            unallocated_budget: global_budget,
            broker_kp,
            issuer_kps,
            sink_pks,
            epochs: vec![0; n],
            live: vec![true; n],
            leases: std::array::from_fn(|_| CoordinatorLeaseEntry {
                lease_id: 0,
                slice_budget: 0,
                expires_ms: 0,
                fenced_for_epoch: vec![0; n],
            }),
            next_challenge_nonce: 10_000,
        }
    }

    fn check_partition_invariant(&self) -> bool {
        let sum_slices: u64 = self.leases.iter().map(|l| l.slice_budget).sum();
        sum_slices
            .checked_add(self.unallocated_budget)
            .is_some_and(|total| total <= self.global_budget)
    }

    fn allocate_lease_slice(
        &mut self,
        sink_idx: usize,
        slice_budget: u64,
        now_ms: u64,
        ttl_ms: u64,
    ) -> Result<SignedLeaseSlice, String> {
        if slice_budget == 0 || slice_budget > self.unallocated_budget {
            return Err("insufficient unallocated global budget for slice".into());
        }
        self.unallocated_budget -= slice_budget;
        let entry = &mut self.leases[sink_idx];
        entry.lease_id += 1;
        entry.slice_budget += slice_budget;
        entry.expires_ms = now_ms + ttl_ms;
        Ok(SignedLeaseSlice::sign(
            &self.broker_kp,
            SINK_IDS[sink_idx],
            entry.lease_id,
            entry.slice_budget,
            entry.expires_ms,
        ))
    }

    fn begin_withdrawal(&mut self, issuer: usize) -> (u64, [(SignedFenceCert, u64); NUM_SINKS]) {
        self.live[issuer] = false;
        self.epochs[issuer] += 1;
        let new_floor = self.epochs[issuer];
        let certs = std::array::from_fn(|m| {
            self.next_challenge_nonce += 1;
            let nonce = self.next_challenge_nonce;
            let cert =
                SignedFenceCert::sign(&self.issuer_kps[issuer], SINK_IDS[m], issuer, new_floor);
            (cert, nonce)
        });
        (new_floor, certs)
    }

    fn verify_and_record_fence_ack(
        &mut self,
        sink_idx: usize,
        issuer: usize,
        required_floor: u64,
        expected_nonce: u64,
        ack: &SignedFenceAck,
    ) -> bool {
        if ack.verify(
            SINK_IDS[sink_idx],
            &self.sink_pks[sink_idx],
            issuer,
            required_floor,
            expected_nonce,
        ) {
            self.leases[sink_idx].fenced_for_epoch[issuer] = policy_core::advance_floor(
                self.leases[sink_idx].fenced_for_epoch[issuer],
                ack.applied_floor,
            );
            true
        } else {
            false
        }
    }

    /// StrongRevokeAck holds iff EVERY sink with an active lease (`slice_budget > 0`)
    /// either returned a verified `SignedFenceAck` at `>= required_floor` OR its
    /// bounded lease TTL has strictly expired (`now_ms > expires_ms`) without renewal.
    fn can_emit_strong_revoke_ack(&self, issuer: usize, required_floor: u64, now_ms: u64) -> bool {
        for lease in &self.leases {
            if lease.slice_budget == 0 {
                continue;
            }
            let signed_fenced = lease.fenced_for_epoch[issuer] >= required_floor;
            let lease_expired = now_ms > lease.expires_ms;
            if !signed_fenced && !lease_expired {
                return false;
            }
        }
        true
    }
}

// ============================================================================
// Adversarial Jepsen-Style Fault Nemesis & Event-Driven Network Queue
// ============================================================================

#[derive(Clone, Debug)]
struct Rng64(u64);

impl Rng64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn index(&mut self, n: usize) -> usize {
        assert!(n > 0);
        let limit = u64::MAX - u64::MAX % (n as u64);
        loop {
            let x = self.next_u64();
            if x < limit {
                return (x % (n as u64)) as usize;
            }
        }
    }

    fn chance(&mut self, num: u64, den: u64) -> bool {
        (self.next_u64() % den) < num
    }
}

#[derive(Clone, Debug)]
enum ScheduledEventKind {
    BrokerFenceToSink {
        sink_idx: usize,
        issuer: usize,
        required_floor: u64,
        cert: SignedFenceCert,
        challenge_nonce: u64,
    },
    MitmSpoofedFenceAck {
        sink_idx: usize,
        issuer: usize,
        required_floor: u64,
        expected_nonce: u64,
    },
    ProposerAccept {
        sink_idx: usize,
        proposal: CanonicalProposal,
        session_actor: String,
        credential: Box<AuthCredential>,
    },
    ProposerComplete {
        sink_idx: usize,
        proposal_id: String,
    },
    SinkCrashRestart {
        sink_idx: usize,
    },
}

#[derive(Clone, Debug)]
struct ScheduledPacket {
    deliver_ms: u64,
    tie_breaker: u64,
    kind: ScheduledEventKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JepsenClassSummary {
    pub class: String,
    pub schedules_run: usize,
    pub asymmetric_partitions_injected: usize,
    pub packet_drops_injected: usize,
    pub packet_delays_injected: usize,
    pub packet_reorders_injected: usize,
    pub packet_duplicates_injected: usize,
    pub crash_restarts_injected: usize,
    pub mitm_forged_acks_injected: usize,
    pub mitm_forged_acks_rejected: usize,
    pub signed_fence_completions: usize,
    pub lease_expiry_completions: usize,
    pub post_ack_unauthorized_commits: usize,
    pub global_budget_breaches: usize,
    pub partition_invariant_breaches: usize,
    pub duplicate_effect_breaches: usize,
    pub max_committed_spend_observed: u64,
    pub global_budget_cap: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CryptoAndJepsenEvidence {
    pub schema_version: u32,
    pub transport: String,
    pub sinks_count: usize,
    pub sink_ids: Vec<String>,
    pub schedules_per_class: usize,
    pub crypto_redteam: CryptoRedTeamReport,
    pub jepsen_classes: Vec<JepsenClassSummary>,
    pub elapsed_ms: f64,
}

pub fn run_jepsen_multisink_campaign(
    seed: u64,
    schedules_per_class: usize,
) -> CryptoAndJepsenEvidence {
    run_jepsen_multisink_with_redteam(seed, schedules_per_class, 10_000)
}

pub fn crypto_and_jepsen_suite(seed: u64, quick: bool) -> CryptoAndJepsenEvidence {
    let (schedules, redteam_trials) = if quick { (200, 1_000) } else { (1_000, 10_000) };
    run_jepsen_multisink_with_redteam(seed, schedules, redteam_trials)
}

pub fn run_jepsen_multisink_with_redteam(
    seed: u64,
    schedules_per_class: usize,
    redteam_trials: usize,
) -> CryptoAndJepsenEvidence {
    let started = Instant::now();
    let crypto_redteam = run_crypto_redteam_campaign(seed ^ 0x517cc1b727220a95, redteam_trials);

    let broker_kp = Ed25519Keypair::deterministic("v4.cluster.broker", 0);
    let issuer_kps: Vec<Ed25519Keypair> = (0..3)
        .map(|i| Ed25519Keypair::deterministic("v4.cluster.issuer", i))
        .collect();
    let issuer_pks: Vec<[u8; 32]> = issuer_kps.iter().map(|k| k.public_key).collect();
    let sink_kps: [Ed25519Keypair; NUM_SINKS] =
        std::array::from_fn(|m| Ed25519Keypair::deterministic("v4.cluster.sink", m as u64));
    let sink_pks: [[u8; 32]; NUM_SINKS] = std::array::from_fn(|m| sink_kps[m].public_key);
    let macaroon_keys: [[u8; 32]; NUM_SINKS] = std::array::from_fn(|m| {
        let label = format!("v4.cluster.macaroon.{m}");
        sha256(label.as_bytes())
    });
    let attacker_kp = Ed25519Keypair::deterministic("v4.cluster.mitm", 999);

    let workers: [SinkSocketWorker; NUM_SINKS] = std::array::from_fn(|m| {
        SinkSocketWorker::spawn(SinkRuntime::new(
            SINK_IDS[m],
            sink_kps[m].clone(),
            broker_kp.public_key,
            issuer_pks.clone(),
            macaroon_keys[m],
        ))
    });

    let classes = [
        SinkProtocolClass::CooperativeSignedFence,
        SinkProtocolClass::PrematureBrokerOnlyAck,
        SinkProtocolClass::UncooperativeExternalSink,
    ];

    // Pre-mint 64 rotating per-schedule signed proposal sets so counterfactual runs across
    // the 3 classes share identical cryptographic inputs while verifying every AF_UNIX packet live.
    let cred_pool_len = 64usize;
    let key_refs: Vec<Option<&Ed25519Keypair>> = issuer_kps.iter().map(Some).collect();
    let mut cred_pool = Vec::with_capacity(cred_pool_len);
    for slot in 0..cred_pool_len {
        let p0_main = CanonicalProposal {
            id: format!("s{slot}-sink0-p0"),
            actor: "bob".into(),
            sink_id: SINK_IDS[0].into(),
            epochs: vec![0, 0, 0],
            expires: 100,
            effect: CanonicalEffect::Transfer {
                recipient: "approved".into(),
                units: 4,
            },
        };
        let root_mac0 = CapabilityMacaroon::mint(
            &macaroon_keys[0],
            MacaroonRoot {
                root_id: format!("root-s{slot}"),
                sink_id: SINK_IDS[0].into(),
                initial_holder: "alice".into(),
                budget: 6,
                expires: 120,
                scope: vec![true, false, false],
                required: vec![true, false, false],
                epochs: vec![0, 0, 0],
            },
        )
        .expect("mint root macaroon");
        let bob_mac0 = root_mac0
            .attenuate_offline(MacaroonCaveat {
                delegatee: "bob".into(),
                budget: 4,
                expires: 100,
                scope: vec![true, false, false],
                required: vec![true, false, false],
                pinned_proposal_digest: Some(p0_main.digest()),
            })
            .expect("offline attenuate macaroon");

        let p0_contender = CanonicalProposal {
            id: format!("s{slot}-sink0-overspend"),
            actor: "alice".into(),
            sink_id: SINK_IDS[0].into(),
            epochs: vec![0, 0, 0],
            expires: 100,
            effect: CanonicalEffect::Transfer {
                recipient: "approved".into(),
                units: 4,
            },
        };
        let qc0_contender = QuorumCertificate::sign_for_proposal(
            &p0_contender,
            vec![true, false, false],
            &key_refs,
        );

        let p1_rel = CanonicalProposal {
            id: format!("s{slot}-sink1-rel"),
            actor: "alice".into(),
            sink_id: SINK_IDS[1].into(),
            epochs: vec![0, 0, 0],
            expires: 100,
            effect: CanonicalEffect::Release {
                destination: "approved".into(),
                contents: "SYNTHETIC DATA: no personal information".into(),
            },
        };
        let qc1 = QuorumCertificate::sign_for_proposal(&p1_rel, vec![true, true, true], &key_refs);

        let p2_edit = CanonicalProposal {
            id: format!("s{slot}-sink2-edit"),
            actor: "alice".into(),
            sink_id: SINK_IDS[2].into(),
            epochs: vec![0, 0, 0],
            expires: 100,
            effect: CanonicalEffect::Edit {
                document: "toy.rs".into(),
                expected_revision: 0,
                contents: "fn answer() -> u32 { 42 }".into(),
            },
        };
        let qc2 =
            QuorumCertificate::sign_for_proposal(&p2_edit, vec![true, true, false], &key_refs);
        cred_pool.push((
            p0_main,
            bob_mac0,
            p0_contender,
            qc0_contender,
            p1_rel,
            qc1,
            p2_edit,
            qc2,
        ));
    }

    let mut summaries = Vec::with_capacity(classes.len());

    for class in classes {
        // Use the exact same schedule seed stream per class for paired counterfactual comparison!
        let mut rng = Rng64(seed);
        let mut asym_partitions = 0usize;
        let mut drops = 0usize;
        let mut delays = 0usize;
        let mut reorders = 0usize;
        let mut duplicates = 0usize;
        let mut crashes = 0usize;
        let mut mitm_injected = 0usize;
        let mut mitm_rejected = 0usize;
        let mut signed_fence_completions = 0usize;
        let mut lease_expiry_completions = 0usize;
        let mut post_ack_unauthorized_commits = 0usize;
        let mut global_budget_breaches = 0usize;
        let mut partition_invariant_breaches = 0usize;
        let mut duplicate_effect_breaches = 0usize;
        let mut max_committed_spend = 0u64;
        let global_budget = 12u64;
        let slice_allocs = [6u64, 3u64, 3u64];
        let lease_ttl_ms = 200u64;

        for sched_idx in 0..schedules_per_class {
            for w in &workers {
                let r = w.call(&SinkRpcRequest::ResetSchedule { class });
                assert!(matches!(r, SinkRpcResponse::AckReset));
            }

            let mut broker = CoordinatorBroker::new(
                global_budget,
                broker_kp.clone(),
                issuer_kps.clone(),
                sink_pks,
            );

            // Partition global budget B = 12 into (6, 3, 3) across Sink_0, Sink_1, Sink_2 at t = 10ms
            let lease_start_ms = 10u64;
            for m in 0..NUM_SINKS {
                let slice = broker
                    .allocate_lease_slice(m, slice_allocs[m], lease_start_ms, lease_ttl_ms)
                    .expect("allocate partitioned lease slice");
                let r = workers[m].call(&SinkRpcRequest::InstallLease {
                    lease: slice,
                    now_ms: lease_start_ms,
                });
                assert!(matches!(
                    r,
                    SinkRpcResponse::LeaseInstalled { accepted: true }
                ));
            }
            if !broker.check_partition_invariant() {
                partition_invariant_breaches += 1;
            }

            let (p0_main, bob_mac0, p0_contender, qc0_contender, p1_rel, qc1, p2_edit, qc2) =
                cred_pool[sched_idx % cred_pool_len].clone();

            // Choose Nemesis fault profile for this schedule
            let partitioned_sink: Option<usize> = if rng.chance(45, 100) {
                asym_partitions += 1;
                Some(rng.index(NUM_SINKS))
            } else {
                None
            };

            let withdraw_ms = 40 + (rng.next_u64() % 30); // withdrawal in [40, 69] ms; lease expires at 210 ms
            let revoked_issuer = 0usize; // required by all 3 proposals!
            let (required_floor, fence_certs) = broker.begin_withdrawal(revoked_issuer);

            let mut queue: Vec<ScheduledPacket> = Vec::with_capacity(24);
            let mut seq = 0u64;
            let mut push_pkt =
                |q: &mut Vec<ScheduledPacket>, deliver_ms: u64, kind: ScheduledEventKind| {
                    seq += 1;
                    q.push(ScheduledPacket {
                        deliver_ms,
                        tie_breaker: seq,
                        kind,
                    });
                };

            // Schedule Broker -> Sink_m Fence packets through Jepsen Nemesis
            for (m, (cert, nonce)) in fence_certs.iter().enumerate() {
                if partitioned_sink == Some(m) {
                    // Asymmetric partition: Broker <-> Sink_m severed while Proposer <-> Sink_m stays live!
                    drops += 1;
                    if rng.chance(50, 100) {
                        mitm_injected += 1;
                        push_pkt(
                            &mut queue,
                            withdraw_ms + 5,
                            ScheduledEventKind::MitmSpoofedFenceAck {
                                sink_idx: m,
                                issuer: revoked_issuer,
                                required_floor,
                                expected_nonce: *nonce,
                            },
                        );
                    }
                    continue;
                }

                if rng.chance(20, 100) {
                    // PacketDrop on Broker -> Sink_m
                    drops += 1;
                    continue;
                }

                let mut fence_arrival = withdraw_ms + 5 + (rng.next_u64() % 20);
                if rng.chance(30, 100) {
                    // PacketDelay: delay fence by +40..120ms
                    delays += 1;
                    fence_arrival += 40 + (rng.next_u64() % 80);
                }
                push_pkt(
                    &mut queue,
                    fence_arrival,
                    ScheduledEventKind::BrokerFenceToSink {
                        sink_idx: m,
                        issuer: revoked_issuer,
                        required_floor,
                        cert: cert.clone(),
                        challenge_nonce: *nonce,
                    },
                );
                if rng.chance(25, 100) {
                    // Duplicate fence packet
                    duplicates += 1;
                    push_pkt(
                        &mut queue,
                        fence_arrival + 7,
                        ScheduledEventKind::BrokerFenceToSink {
                            sink_idx: m,
                            issuer: revoked_issuer,
                            required_floor,
                            cert: cert.clone(),
                            challenge_nonce: *nonce,
                        },
                    );
                }
            }

            // Schedule Proposer -> Sink_m Accept & Complete (including pre-withdrawal in-flight,
            // post-withdrawal asymmetric-partition races, PacketReorder, and PacketDuplicate storms!)
            let proposals_per_sink = [
                (
                    0usize,
                    p0_main,
                    "bob".to_string(),
                    AuthCredential::Macaroon { macaroon: bob_mac0 },
                ),
                (
                    0usize,
                    p0_contender,
                    "alice".to_string(),
                    AuthCredential::Quorum {
                        cert: qc0_contender,
                    },
                ),
                (
                    1usize,
                    p1_rel,
                    "alice".to_string(),
                    AuthCredential::Quorum { cert: qc1 },
                ),
                (
                    2usize,
                    p2_edit,
                    "alice".to_string(),
                    AuthCredential::Quorum { cert: qc2 },
                ),
            ];

            for (m, prop, actor, cred) in proposals_per_sink {
                // Half the time Accept arrives BEFORE withdraw_ms (in-flight at withdrawal),
                // half the time Accept arrives AFTER withdraw_ms (racing during partition/delay)
                let accept_ms = if rng.chance(55, 100) {
                    20 + (rng.next_u64() % 18) // [20, 37] < withdraw_ms
                } else {
                    withdraw_ms + 10 + (rng.next_u64() % 180) // [withdraw+10, withdraw+189], spanning before & after lease expiry!
                };
                let mut complete_ms = accept_ms + 25 + (rng.next_u64() % 90);

                // Inject PacketReorder (`Complete` arrives before `Accept`, followed by retry `Complete`)
                if rng.chance(25, 100) {
                    reorders += 1;
                    let early_complete = accept_ms.saturating_sub(4);
                    push_pkt(
                        &mut queue,
                        early_complete,
                        ScheduledEventKind::ProposerComplete {
                            sink_idx: m,
                            proposal_id: prop.id.clone(),
                        },
                    );
                    complete_ms += 15;
                }

                push_pkt(
                    &mut queue,
                    accept_ms,
                    ScheduledEventKind::ProposerAccept {
                        sink_idx: m,
                        proposal: prop.clone(),
                        session_actor: actor.clone(),
                        credential: Box::new(cred.clone()),
                    },
                );
                push_pkt(
                    &mut queue,
                    complete_ms,
                    ScheduledEventKind::ProposerComplete {
                        sink_idx: m,
                        proposal_id: prop.id.clone(),
                    },
                );

                // Inject PacketDuplicate replay storm (duplicate Accept + duplicate Complete)
                if rng.chance(35, 100) {
                    duplicates += 1;
                    push_pkt(
                        &mut queue,
                        accept_ms + 3,
                        ScheduledEventKind::ProposerAccept {
                            sink_idx: m,
                            proposal: prop.clone(),
                            session_actor: actor,
                            credential: Box::new(cred),
                        },
                    );
                    push_pkt(
                        &mut queue,
                        complete_ms + 8,
                        ScheduledEventKind::ProposerComplete {
                            sink_idx: m,
                            proposal_id: prop.id,
                        },
                    );
                }
            }

            // Inject SimulatedCrashRestart on a random sink mid-schedule
            if rng.chance(40, 100) {
                crashes += 1;
                let crash_sink = rng.index(NUM_SINKS);
                let crash_ms = withdraw_ms + 8 + (rng.next_u64() % 60);
                push_pkt(
                    &mut queue,
                    crash_ms,
                    ScheduledEventKind::SinkCrashRestart {
                        sink_idx: crash_sink,
                    },
                );
            }

            // Sort network events chronologically
            queue.sort_by_key(|p| (p.deliver_ms, p.tie_breaker));

            // Track revocation acknowledgment timestamp under the active protocol class:
            // - PrematureBrokerOnlyAck: acks immediately at `withdraw_ms` without waiting for FenceAck or lease expiry!
            // - CooperativeSignedFence & UncooperativeExternalSink: Coordinator emits StrongRevokeAck at the earliest
            //   logical timestamp >= withdraw_ms where `can_emit_strong_revoke_ack` becomes true!
            let mut revoke_ack_time_ms: Option<u64> =
                if class == SinkProtocolClass::PrematureBrokerOnlyAck {
                    Some(withdraw_ms)
                } else {
                    None
                };
            let mut completed_via_lease_expiry = false;

            let max_lease_exp = broker
                .leases
                .iter()
                .map(|l| l.expires_ms)
                .max()
                .unwrap_or(withdraw_ms);

            for pkt in queue {
                let now_ms = pkt.deliver_ms;
                // Check if lease expiry alone already satisfied StrongRevokeAck before `now_ms`
                if revoke_ack_time_ms.is_none() && now_ms > max_lease_exp {
                    let expiry_ack_ms = max_lease_exp + 1;
                    if broker.can_emit_strong_revoke_ack(
                        revoked_issuer,
                        required_floor,
                        expiry_ack_ms,
                    ) {
                        revoke_ack_time_ms = Some(expiry_ack_ms);
                        completed_via_lease_expiry = true;
                    }
                }

                match pkt.kind {
                    ScheduledEventKind::BrokerFenceToSink {
                        sink_idx,
                        issuer,
                        required_floor: req_fl,
                        cert,
                        challenge_nonce,
                    } => {
                        let resp = workers[sink_idx].call(&SinkRpcRequest::ApplyFence {
                            cert,
                            challenge_nonce,
                            now_ms,
                        });
                        if let SinkRpcResponse::FenceAck {
                            ack: Some(signed_ack),
                            ..
                        } = resp
                        {
                            // Symmetric/return drop check: if partitioned_sink == Some(sink_idx), return is also dropped
                            if partitioned_sink != Some(sink_idx) {
                                let ok = broker.verify_and_record_fence_ack(
                                    sink_idx,
                                    issuer,
                                    req_fl,
                                    challenge_nonce,
                                    &signed_ack,
                                );
                                assert!(ok, "genuine Sink SignedFenceAck must verify");
                                if revoke_ack_time_ms.is_none()
                                    && now_ms >= withdraw_ms
                                    && broker.can_emit_strong_revoke_ack(issuer, req_fl, now_ms)
                                {
                                    revoke_ack_time_ms = Some(now_ms);
                                }
                            }
                        }
                    }
                    ScheduledEventKind::MitmSpoofedFenceAck {
                        sink_idx,
                        issuer,
                        required_floor: req_fl,
                        expected_nonce,
                    } => {
                        // Attacker forges a FenceAck signed by attacker_kp (or wrong nonce/floor)
                        let fake_frame = crate::crypto_authority::fence_ack_frame(
                            SINK_IDS[sink_idx],
                            issuer,
                            req_fl,
                            expected_nonce,
                        );
                        let forged = SignedFenceAck {
                            sink_id: SINK_IDS[sink_idx].into(),
                            issuer,
                            applied_floor: req_fl,
                            challenge_nonce: expected_nonce,
                            signature: attacker_kp.sign(&fake_frame),
                        };
                        if !broker.verify_and_record_fence_ack(
                            sink_idx,
                            issuer,
                            req_fl,
                            expected_nonce,
                            &forged,
                        ) {
                            mitm_rejected += 1;
                        }
                    }
                    ScheduledEventKind::ProposerAccept {
                        sink_idx,
                        proposal,
                        session_actor,
                        credential,
                    } => {
                        let _ = workers[sink_idx].call(&SinkRpcRequest::Accept {
                            proposal,
                            session_actor,
                            credential,
                            now_ms,
                        });
                    }
                    ScheduledEventKind::ProposerComplete {
                        sink_idx,
                        proposal_id,
                    } => {
                        let _ = workers[sink_idx].call(&SinkRpcRequest::Complete {
                            proposal_id,
                            now_ms,
                        });
                    }
                    ScheduledEventKind::SinkCrashRestart { sink_idx } => {
                        let r = workers[sink_idx].call(&SinkRpcRequest::CrashRestart);
                        assert!(matches!(r, SinkRpcResponse::Restarted { .. }));
                    }
                }
            }

            // If still unacked at end of packet trace, coordinator completes StrongRevokeAck at `max_lease_exp + 1`
            let ack_ms = match revoke_ack_time_ms {
                Some(t) => t,
                None => {
                    let t = max_lease_exp + 1;
                    assert!(broker.can_emit_strong_revoke_ack(revoked_issuer, required_floor, t));
                    completed_via_lease_expiry = true;
                    t
                }
            };
            if class != SinkProtocolClass::PrematureBrokerOnlyAck {
                if completed_via_lease_expiry {
                    lease_expiry_completions += 1;
                } else {
                    signed_fence_completions += 1;
                }
            }

            // Inspect all 3 sinks over AF_UNIX and verify global invariants
            let mut total_committed_spend = 0u64;
            let mut seen_ids = std::collections::HashSet::new();
            let mut schedule_had_post_ack_leak = false;

            for (m, w) in workers.iter().enumerate() {
                let SinkRpcResponse::StateReport {
                    committed_in_slice,
                    slice_budget,
                    committed_effects,
                    ..
                } = w.call(&SinkRpcRequest::InspectState)
                else {
                    panic!("expected StateReport from sink {m}");
                };
                if committed_in_slice > slice_budget || slice_budget != slice_allocs[m] {
                    global_budget_breaches += 1;
                }
                for eff in committed_effects {
                    total_committed_spend += eff.charge;
                    if !seen_ids.insert(eff.proposal_id.clone()) {
                        duplicate_effect_breaches += 1;
                    }
                    if eff.required.get(revoked_issuer) == Some(&true)
                        && eff.epochs.get(revoked_issuer).copied().unwrap_or(0) < required_floor
                        && eff.commit_time_ms > ack_ms
                    {
                        schedule_had_post_ack_leak = true;
                    }
                }
            }

            if schedule_had_post_ack_leak {
                post_ack_unauthorized_commits += 1;
            }
            if total_committed_spend > global_budget {
                global_budget_breaches += 1;
            }
            max_committed_spend = max_committed_spend.max(total_committed_spend);
            if !broker.check_partition_invariant() {
                partition_invariant_breaches += 1;
            }
        }

        summaries.push(JepsenClassSummary {
            class: class.as_str().into(),
            schedules_run: schedules_per_class,
            asymmetric_partitions_injected: asym_partitions,
            packet_drops_injected: drops,
            packet_delays_injected: delays,
            packet_reorders_injected: reorders,
            packet_duplicates_injected: duplicates,
            crash_restarts_injected: crashes,
            mitm_forged_acks_injected: mitm_injected,
            mitm_forged_acks_rejected: mitm_rejected,
            signed_fence_completions,
            lease_expiry_completions,
            post_ack_unauthorized_commits,
            global_budget_breaches,
            partition_invariant_breaches,
            duplicate_effect_breaches,
            max_committed_spend_observed: max_committed_spend,
            global_budget_cap: global_budget,
        });
    }

    for w in workers {
        w.shutdown();
    }

    CryptoAndJepsenEvidence {
        schema_version: 4,
        transport: "af_unix_stream_framed_threads".into(),
        sinks_count: NUM_SINKS,
        sink_ids: SINK_IDS.iter().map(|s| (*s).to_string()).collect(),
        schedules_per_class,
        crypto_redteam,
        jepsen_classes: summaries,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

pub fn write_v4_systems_evidence(
    output_path: &Path,
    seed: u64,
) -> Result<CryptoAndJepsenEvidence, String> {
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("create_dir_all {}: {e}", parent.display()))?;
    }
    let report = run_jepsen_multisink_campaign(seed, 1_000);
    let mut bytes =
        serde_json::to_vec_pretty(&report).map_err(|e| format!("serialize evidence: {e}"))?;
    bytes.push(b'\n');
    fs::write(output_path, bytes).map_err(|e| format!("write {}: {e}", output_path.display()))?;
    Ok(report)
}
