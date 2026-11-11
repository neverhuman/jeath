use authority_lab::protocol_v2::{
    Broker, DispatchedPermit, Effect, Permit, Proposal, Sink, Status,
};
use serde_json::json;
use std::collections::{HashSet, VecDeque};

type EffectsLog = Vec<(String, DispatchedPermit, u64)>;

fn transfer_permit(actor: &str, id: &str, units: u64, epochs: Vec<u64>, expires: u64) -> Permit {
    Permit {
        actor: actor.into(),
        proposal: Proposal {
            id: id.into(),
            epochs,
            expires,
            effect: Effect::Transfer {
                recipient: "approved".into(),
                units,
            },
        },
    }
}

fn release_permit(actor: &str, id: &str, epochs: Vec<u64>, expires: u64) -> Permit {
    Permit {
        actor: actor.into(),
        proposal: Proposal {
            id: id.into(),
            epochs,
            expires,
            effect: Effect::Release {
                destination: "approved".into(),
                contents: "SYNTHETIC DATA: no personal information".into(),
            },
        },
    }
}

fn edit_permit(actor: &str, id: &str, expected_rev: u64, epochs: Vec<u64>, expires: u64) -> Permit {
    Permit {
        actor: actor.into(),
        proposal: Proposal {
            id: id.into(),
            epochs,
            expires,
            effect: Effect::Edit {
                document: "toy.rs".into(),
                expected_revision: expected_rev,
                contents: "fn answer() -> u32 { 42 }".into(),
            },
        },
    }
}

fn assert_global_invariants(broker: &Broker, sink: &Sink, effects: &EffectsLog) {
    let b_audit = broker.audit_state("ok");
    assert_eq!(
        b_audit["consistent"], true,
        "broker audit failed: {b_audit}"
    );
    let s_audit = sink.audit_state("ok", effects);
    assert_eq!(s_audit["consistent"], true, "sink audit failed: {s_audit}");

    let committed_effect_charge: u64 = effects.iter().map(|(_, _, c)| *c).sum();
    assert!(
        committed_effect_charge <= broker.charged,
        "committed effects ({committed_effect_charge}) exceeded broker.charged ({})",
        broker.charged
    );
    assert!(
        broker.charged <= broker.budget,
        "broker.charged ({}) exceeded budget ({})",
        broker.charged,
        broker.budget
    );

    let mut seen = HashSet::new();
    for (id, _, _) in effects {
        assert!(seen.insert(id.clone()), "duplicate effect committed: {id}");
    }
}

fn enqueue_settle(queue: &mut Vec<(String, Status)>, item: (String, Status)) {
    if !queue.contains(&item) && queue.len() < 4 {
        queue.push(item);
    }
}

#[derive(Clone)]
struct AsyncWorld {
    broker: Broker,
    sink: Sink,
    effects: EffectsLog,
    in_flight_to_sink: Vec<DispatchedPermit>,
    in_flight_to_settle: Vec<(String, Status)>,
    withdrawn: bool,
    renewed_and_approved_p1: bool,
    fence_acked: bool,
    lost_reply_injected: bool,
}

impl AsyncWorld {
    fn canonical_key(&self) -> String {
        serde_json::to_string(&json!({
            "b": &self.broker,
            "s": &self.sink,
            "e": &self.effects,
            "q_sink": &self.in_flight_to_sink,
            "q_settle": &self.in_flight_to_settle,
            "w": self.withdrawn,
            "r": self.renewed_and_approved_p1,
            "f": self.fence_acked,
            "lr": self.lost_reply_injected,
        }))
        .expect("canonical state serialization")
    }
}

#[test]
fn exhaustive_revocation_renewal_crash_and_timeout_interleavings() {
    let mut init_broker = Broker::new(4);
    let p0 = transfer_permit("alice", "p0", 2, vec![0, 0], 1_000);
    let p1 = transfer_permit("alice", "p1", 2, vec![1, 0], 1_000);
    assert_eq!(
        init_broker.approve(p0.clone(), 0).unwrap()["status"],
        "approved"
    );

    let start = AsyncWorld {
        broker: init_broker,
        sink: Sink::default(),
        effects: vec![],
        in_flight_to_sink: vec![],
        in_flight_to_settle: vec![],
        withdrawn: false,
        renewed_and_approved_p1: false,
        fence_acked: false,
        lost_reply_injected: false,
    };

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    visited.insert(start.canonical_key());
    queue.push_back(start);

    let mut transitions_checked = 0usize;
    let mut witnessed_both_committed = false;
    let mut witnessed_fenced_cancel_and_p1_commit = false;
    let mut witnessed_lost_reply_retry_settlement = false;

    while let Some(state) = queue.pop_front() {
        assert_global_invariants(&state.broker, &state.sink, &state.effects);

        let p0_broker_settled = state
            .broker
            .records
            .iter()
            .any(|r| r.permit.proposal.id == "p0" && r.status == Status::Committed);
        if self_has_effect(&state.effects, "p0") && self_has_effect(&state.effects, "p1") {
            witnessed_both_committed = true;
        }
        if !self_has_effect(&state.effects, "p0")
            && self_has_effect(&state.effects, "p1")
            && state
                .broker
                .records
                .iter()
                .any(|r| r.permit.proposal.id == "p0" && r.status == Status::Cancelled)
        {
            witnessed_fenced_cancel_and_p1_commit = true;
        }
        if state.lost_reply_injected && p0_broker_settled {
            witnessed_lost_reply_retry_settlement = true;
        }

        let mut next_states = Vec::new();

        // 1. Proposer submits/retries p0 (or pre-commit crash on Broker, which is a no-op).
        {
            let mut pre_crash = state.clone();
            let snap = pre_crash.broker.clone();
            let _ = pre_crash.broker.submit(p0.clone(), 10).unwrap();
            pre_crash.broker = snap; // pre-commit crash rolls back SQLite Immediate tx
            assert_global_invariants(&pre_crash.broker, &pre_crash.sink, &pre_crash.effects);

            let mut ns = state.clone();
            let r = ns.broker.submit(p0.clone(), 10).unwrap();
            if r["status"] == "pending" && ns.in_flight_to_sink.len() < 2 {
                let dispatched: DispatchedPermit =
                    serde_json::from_value(r["permit"].clone()).unwrap();
                if !ns.in_flight_to_sink.contains(&dispatched) {
                    ns.in_flight_to_sink.push(dispatched);
                }
            }
            next_states.push(ns);
        }

        // 2. Full proposer-level retry (submit -> accept -> settle if terminal, fixing Bug #1).
        {
            let mut ns = state.clone();
            let r = ns.broker.submit(p0.clone(), 10).unwrap();
            if r["status"] == "pending" {
                let dispatched: DispatchedPermit =
                    serde_json::from_value(r["permit"].clone()).unwrap();
                let acc = ns.sink.accept_dispatched(dispatched).unwrap();
                match acc["status"].as_str() {
                    Some("committed") => {
                        let _ = ns.broker.settle("p0", Status::Committed).unwrap();
                    }
                    Some("cancelled") => {
                        let _ = ns.broker.settle("p0", Status::Cancelled).unwrap();
                    }
                    _ => {}
                }
            }
            next_states.push(ns);
        }

        // 3. Deliver (or duplicate) in-flight permit to Sink::accept_dispatched.
        for idx in 0..state.in_flight_to_sink.len() {
            let mut ns = state.clone();
            let permit = ns.in_flight_to_sink.remove(idx);
            let id = permit.proposal.id.clone();
            let resp = ns.sink.accept_dispatched(permit).unwrap();
            if resp["status"] == "cancelled" {
                enqueue_settle(&mut ns.in_flight_to_settle, (id, Status::Cancelled));
            }
            next_states.push(ns);
        }

        // 4. Sink::complete on any Pending record (normal + lost-reply post-commit crash + pre-commit rollback).
        for rec in &state.sink.records {
            if rec.status == Status::Pending {
                let id = rec.permit.proposal.id.clone();
                let req_epoch0 = rec.permit.proposal.epochs[0];
                let req_issuer0 = rec.permit.effective_required(2).unwrap()[0];

                // 4a. Pre-commit crash on Sink::complete rolls back cleanly.
                {
                    let mut snap_sink = state.sink.clone();
                    let _ = snap_sink.complete(&id).unwrap();
                    assert_global_invariants(&state.broker, &state.sink, &state.effects);
                }

                // 4b. Normal Sink::complete.
                let mut ns = state.clone();
                let (resp, newly_committed) = ns.sink.complete(&id).unwrap();
                if let Some(committed_permit) = newly_committed {
                    if ns.fence_acked && req_issuer0 {
                        assert!(
                            req_epoch0 >= ns.sink.floors[0],
                            "VIOLATION: old-epoch permit {id} (epoch {req_epoch0}) committed after fence floor {}!",
                            ns.sink.floors[0]
                        );
                    }
                    let c = committed_permit.proposal.effect.charge();
                    ns.effects.push((id.clone(), committed_permit, c));
                }
                let st: Status = serde_json::from_value(resp["status"].clone()).unwrap();
                enqueue_settle(&mut ns.in_flight_to_settle, (id.clone(), st));
                next_states.push(ns);

                // 4c. Post-commit crash on p0 (lost reply: effect durable, settlement dropped!).
                if id == "p0" && !state.lost_reply_injected {
                    let mut ns_lost = state.clone();
                    ns_lost.lost_reply_injected = true;
                    let (_resp, newly_committed) = ns_lost.sink.complete(&id).unwrap();
                    if let Some(committed_permit) = newly_committed {
                        if ns_lost.fence_acked && req_issuer0 {
                            assert!(req_epoch0 >= ns_lost.sink.floors[0]);
                        }
                        let c = committed_permit.proposal.effect.charge();
                        ns_lost.effects.push((id.clone(), committed_permit, c));
                    }
                    // Uncertain timeout probe: non-authoritative Pending status MUST NEVER refund!
                    let before_charged = ns_lost.broker.charged;
                    let denied_timeout = ns_lost.broker.settle(&id, Status::Pending).unwrap();
                    assert_eq!(denied_timeout["status"], "denied");
                    assert_eq!(ns_lost.broker.charged, before_charged);
                    next_states.push(ns_lost);
                }
            }
        }

        // 5. Deliver settlement to Broker::settle (plus duplicate settlement idempotence check).
        for idx in 0..state.in_flight_to_settle.len() {
            let mut ns = state.clone();
            let (id, st) = ns.in_flight_to_settle.remove(idx);
            let _ = ns.broker.settle(&id, st.clone()).unwrap();
            let dup = ns.broker.settle(&id, st).unwrap();
            assert_eq!(dup["status"], "denied");
            next_states.push(ns);
        }

        // 6. Withdraw issuer 0.
        if !state.withdrawn {
            let mut ns = state.clone();
            let rev = ns.broker.withdraw(0).unwrap();
            assert_eq!(rev["status"], "admission_revoked");
            assert_eq!(rev["floor"], 1);
            ns.withdrawn = true;
            next_states.push(ns);
        }

        // 7. Renew issuer 0 and approve fresh epoch-1 permit p1.
        if state.withdrawn && !state.renewed_and_approved_p1 {
            let mut ns = state.clone();
            assert_eq!(ns.broker.renew(0).unwrap()["status"], "renewed");
            // Old p0 approval was cleared by withdraw and must NOT be restored by renew!
            if !ns
                .broker
                .records
                .iter()
                .any(|r| r.permit.proposal.id == "p0")
            {
                assert_eq!(
                    ns.broker.submit(p0.clone(), 10).unwrap()["status"],
                    "denied"
                );
            }
            assert_eq!(
                ns.broker.approve(p1.clone(), 0).unwrap()["status"],
                "approved"
            );
            ns.renewed_and_approved_p1 = true;
            next_states.push(ns);
        }

        // 8. Submit fresh epoch-1 permit p1.
        if state.renewed_and_approved_p1 {
            let mut ns = state.clone();
            let r = ns.broker.submit(p1.clone(), 10).unwrap();
            if r["status"] == "pending" && ns.in_flight_to_sink.len() < 2 {
                let dispatched: DispatchedPermit =
                    serde_json::from_value(r["permit"].clone()).unwrap();
                if !ns.in_flight_to_sink.contains(&dispatched) {
                    ns.in_flight_to_sink.push(dispatched);
                }
            }
            next_states.push(ns);
        }

        // 9. Fence issuer 0 at floor = 1 on Sink.
        if state.withdrawn && !state.fence_acked {
            let mut ns = state.clone();
            let ack = ns.sink.fence(0, 1).unwrap();
            assert_eq!(ack["status"], "fenced");
            assert_eq!(ack["floor"], 1);
            ns.fence_acked = true;
            // Any pending p0 cancelled by fence can now settle as Cancelled.
            if ns
                .sink
                .records
                .iter()
                .any(|r| r.permit.proposal.id == "p0" && r.status == Status::Cancelled)
            {
                enqueue_settle(
                    &mut ns.in_flight_to_settle,
                    ("p0".into(), Status::Cancelled),
                );
            }
            next_states.push(ns);
        }

        for ns in next_states {
            transitions_checked += 1;
            assert_global_invariants(&ns.broker, &ns.sink, &ns.effects);
            let key = ns.canonical_key();
            if visited.insert(key) {
                queue.push_back(ns);
            }
        }
    }

    assert!(
        visited.len() >= 200 && transitions_checked >= 1000,
        "expected rich state space, got {} states / {} transitions",
        visited.len(),
        transitions_checked
    );
    assert!(witnessed_both_committed);
    assert!(witnessed_fenced_cancel_and_p1_commit);
    assert!(witnessed_lost_reply_retry_settlement);
}

fn self_has_effect(effects: &EffectsLog, id: &str) -> bool {
    effects.iter().any(|(eid, _, _)| eid == id)
}

#[derive(Clone)]
struct QuorumWorld {
    broker: Broker,
    sink: Sink,
    effects: EffectsLog,
    fenced_2: bool,
    fenced_1: bool,
}

impl QuorumWorld {
    fn key(&self) -> String {
        serde_json::to_string(&json!({
            "b": &self.broker,
            "s": &self.sink,
            "e": &self.effects,
            "f2": self.fenced_2,
            "f1": self.fenced_1,
        }))
        .unwrap()
    }
}

#[test]
fn exhaustive_three_issuer_quorum_split_and_noninterference() {
    let mut broker = Broker::with_issuers(6, 3).unwrap();
    let sink = Sink::with_issuers(3).unwrap();
    let p_xfer = transfer_permit("alice", "xfer-1of3", 2, vec![0, 0, 0], 1_000);
    let p_edit = edit_permit("alice", "edit-2of3", 0, vec![0, 0, 0], 1_000);
    let p_rel = release_permit("alice", "rel-3of3", vec![0, 0, 0], 1_000);

    broker.approve(p_xfer.clone(), 0).unwrap();
    for i in [0, 1] {
        broker.approve(p_edit.clone(), i).unwrap();
    }
    for i in [0, 1, 2] {
        broker.approve(p_rel.clone(), i).unwrap();
    }

    let start = QuorumWorld {
        broker,
        sink,
        effects: vec![],
        fenced_2: false,
        fenced_1: false,
    };

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    visited.insert(start.key());
    queue.push_back(start);

    let mut witnessed_edit_and_xfer_after_fence2 = false;
    let mut witnessed_xfer_after_fence1_and_fence2 = false;

    while let Some(st) = queue.pop_front() {
        assert_global_invariants(&st.broker, &st.sink, &st.effects);
        if st.fenced_2
            && !self_has_effect(&st.effects, "rel-3of3")
            && self_has_effect(&st.effects, "edit-2of3")
            && self_has_effect(&st.effects, "xfer-1of3")
        {
            witnessed_edit_and_xfer_after_fence2 = true;
        }
        if st.fenced_1
            && st.fenced_2
            && !self_has_effect(&st.effects, "rel-3of3")
            && !self_has_effect(&st.effects, "edit-2of3")
            && self_has_effect(&st.effects, "xfer-1of3")
        {
            witnessed_xfer_after_fence1_and_fence2 = true;
        }

        let mut nexts = Vec::new();
        for perm in [&p_xfer, &p_edit, &p_rel] {
            let id = perm.proposal.id.clone();
            if !st.broker.records.iter().any(|r| r.permit.proposal.id == id) {
                let mut ns = st.clone();
                let sub = ns.broker.submit(perm.clone(), 10).unwrap();
                if sub["status"] == "pending" {
                    let dispatched: DispatchedPermit =
                        serde_json::from_value(sub["permit"].clone()).unwrap();
                    let acc = ns.sink.accept_dispatched(dispatched).unwrap();
                    if acc["status"] == "cancelled" {
                        ns.broker.settle(&id, Status::Cancelled).unwrap();
                    }
                }
                nexts.push(ns);
            }
            if st
                .sink
                .records
                .iter()
                .any(|r| r.permit.proposal.id == id && r.status == Status::Pending)
            {
                let mut ns = st.clone();
                let (resp, newly) = ns.sink.complete(&id).unwrap();
                if let Some(cp) = newly {
                    if id == "rel-3of3" {
                        assert!(
                            !ns.fenced_2 && !ns.fenced_1,
                            "3-of-3 committed after fence!"
                        );
                    }
                    if id == "edit-2of3" {
                        assert!(!ns.fenced_1, "2-of-3 committed after issuer-1 fence!");
                    }
                    let c = cp.proposal.effect.charge();
                    ns.effects.push((id.clone(), cp, c));
                }
                let terminal: Status = serde_json::from_value(resp["status"].clone()).unwrap();
                ns.broker.settle(&id, terminal).unwrap();
                nexts.push(ns);
            }
        }

        if !st.fenced_2 {
            let mut ns = st.clone();
            let rev = ns.broker.withdraw(2).unwrap();
            ns.sink.fence(2, rev["floor"].as_u64().unwrap()).unwrap();
            ns.fenced_2 = true;
            for r in ns.sink.records.clone() {
                if r.status == Status::Cancelled {
                    let _ = ns.broker.settle(&r.permit.proposal.id, Status::Cancelled);
                }
            }
            nexts.push(ns);
        }

        if !st.fenced_1 {
            let mut ns = st.clone();
            let rev = ns.broker.withdraw(1).unwrap();
            ns.sink.fence(1, rev["floor"].as_u64().unwrap()).unwrap();
            ns.fenced_1 = true;
            for r in ns.sink.records.clone() {
                if r.status == Status::Cancelled {
                    let _ = ns.broker.settle(&r.permit.proposal.id, Status::Cancelled);
                }
            }
            nexts.push(ns);
        }

        for ns in nexts {
            assert_global_invariants(&ns.broker, &ns.sink, &ns.effects);
            if visited.insert(ns.key()) {
                queue.push_back(ns);
            }
        }
    }

    assert!(witnessed_edit_and_xfer_after_fence2);
    assert!(witnessed_xfer_after_fence1_and_fence2);
}

#[derive(Clone)]
struct DelegationWorld {
    broker: Broker,
    sink: Sink,
    effects: EffectsLog,
    ancestor_revoked: bool,
    ancestor_fenced: bool,
}

impl DelegationWorld {
    fn key(&self) -> String {
        serde_json::to_string(&json!({
            "b": &self.broker,
            "s": &self.sink,
            "e": &self.effects,
            "ar": self.ancestor_revoked,
            "af": self.ancestor_fenced,
        }))
        .unwrap()
    }
}

#[test]
fn subdelegation_attenuation_and_lineage_revocation_interleavings() {
    let mut broker = Broker::new(8);
    let sink = Sink::default();

    // Root delegation: budget=6, expires=500, scope=[Transfer, Release, false], grants=[true, true].
    assert_eq!(
        broker
            .delegate(&json!({
                "op": "delegate",
                "id": "root-cap",
                "budget": 6,
                "expires": 500,
                "scope": [true, true, false],
                "required": [true, true],
                "grants": [true, true],
                "epochs": [0, 0]
            }))
            .unwrap()["status"],
        "delegated"
    );

    // Mid delegation narrows budget (6 -> 4), expiry (500 -> 400), and scope ([T,T,F] -> [T,F,F]),
    // keeping strengthened required=[true, true] on Transfer!
    assert_eq!(
        broker
            .delegate(&json!({
                "op": "delegate",
                "parent": "root-cap",
                "id": "mid-cap",
                "budget": 4,
                "expires": 400,
                "scope": [true, false, false],
                "required": [true, true]
            }))
            .unwrap()["status"],
        "delegated"
    );

    // Every escalation attempt against mid-cap MUST be denied.
    for bad in [
        // Budget escalation (5 > 4)
        json!({"op":"delegate","parent":"mid-cap","id":"bad-b","budget":5,"expires":300,"scope":[true,false,false]}),
        // Expiry escalation (401 > 400)
        json!({"op":"delegate","parent":"mid-cap","id":"bad-e","budget":2,"expires":401,"scope":[true,false,false]}),
        // Scope widening ([F,T,F] not subset of [T,F,F])
        json!({"op":"delegate","parent":"mid-cap","id":"bad-s","budget":2,"expires":300,"scope":[false,true,false]}),
        // Dropping required issuer 1 ([T,F] not superset of [T,T])
        json!({"op":"delegate","parent":"mid-cap","id":"bad-r","budget":2,"expires":300,"scope":[true,false,false],"required":[true,false]}),
    ] {
        assert_eq!(broker.delegate(&bad).unwrap()["status"], "denied");
    }

    // Two valid leaf delegations of 3 units each under mid-cap (budget=4):
    // individually <= 4, but jointly 3 + 3 = 6 > 4!
    let leaf_a = transfer_permit("bob", "leaf-a", 3, vec![0, 0], 300);
    let leaf_b = transfer_permit("carol", "leaf-b", 3, vec![0, 0], 300);
    assert_eq!(
        broker
            .delegate(&json!({"op":"delegate","parent":"mid-cap","permit":leaf_a}))
            .unwrap()["status"],
        "delegated"
    );
    assert_eq!(
        broker
            .delegate(&json!({"op":"delegate","parent":"mid-cap","permit":leaf_b}))
            .unwrap()["status"],
        "delegated"
    );

    let start = DelegationWorld {
        broker,
        sink,
        effects: vec![],
        ancestor_revoked: false,
        ancestor_fenced: false,
    };

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    visited.insert(start.key());
    queue.push_back(start);

    let mut witnessed_cancel_refund_then_sibling_commit = false;

    while let Some(st) = queue.pop_front() {
        assert_global_invariants(&st.broker, &st.sink, &st.effects);
        let committed_total: u64 = st.effects.iter().map(|(_, _, c)| *c).sum();
        assert!(
            committed_total <= 4,
            "descendants exceeded mid-cap budget 4: got {committed_total}"
        );
        if self_has_effect(&st.effects, "leaf-b")
            && st
                .broker
                .records
                .iter()
                .any(|r| r.permit.proposal.id == "leaf-a" && r.status == Status::Cancelled)
        {
            witnessed_cancel_refund_then_sibling_commit = true;
        }

        let mut nexts = Vec::new();
        for leaf in [&leaf_a, &leaf_b] {
            let id = leaf.proposal.id.clone();
            if !st.broker.records.iter().any(|r| r.permit.proposal.id == id) {
                let mut ns = st.clone();
                let r = ns.broker.submit(leaf.clone(), 10).unwrap();
                if r["status"] == "pending" {
                    assert!(
                        !ns.ancestor_revoked,
                        "descendant admitted after ancestor withdrawal!"
                    );
                    let dispatched: DispatchedPermit =
                        serde_json::from_value(r["permit"].clone()).unwrap();
                    let acc = ns.sink.accept_dispatched(dispatched).unwrap();
                    if acc["status"] == "cancelled" {
                        ns.broker.settle(&id, Status::Cancelled).unwrap();
                    }
                }
                nexts.push(ns);
            }
            if st
                .sink
                .records
                .iter()
                .any(|r| r.permit.proposal.id == id && r.status == Status::Pending)
            {
                // Normal complete
                let mut ns = st.clone();
                let (resp, newly) = ns.sink.complete(&id).unwrap();
                if let Some(cp) = newly {
                    assert!(
                        !ns.ancestor_fenced,
                        "descendant committed after ancestor sink fence!"
                    );
                    let c = cp.proposal.effect.charge();
                    ns.effects.push((id.clone(), cp, c));
                }
                let terminal: Status = serde_json::from_value(resp["status"].clone()).unwrap();
                ns.broker.settle(&id, terminal).unwrap();
                nexts.push(ns);

                // Or pre-revocation voluntary/fenced cancellation on leaf-a to test lineage refund
                if id == "leaf-a" && !st.ancestor_revoked {
                    let mut ns_cancel = st.clone();
                    for r in &mut ns_cancel.sink.records {
                        if r.permit.proposal.id == "leaf-a" {
                            r.status = Status::Cancelled;
                        }
                    }
                    ns_cancel
                        .broker
                        .settle("leaf-a", Status::Cancelled)
                        .unwrap();
                    nexts.push(ns_cancel);
                }
            }
        }

        // Withdraw ancestor issuer 0 OR strengthened required steward issuer 1, fencing Sink!
        if !st.ancestor_revoked {
            for iss in [0usize, 1usize] {
                let mut ns = st.clone();
                let rev = ns.broker.withdraw(iss).unwrap();
                assert_eq!(rev["status"], "admission_revoked");
                ns.ancestor_revoked = true;
                ns.sink.fence(iss, rev["floor"].as_u64().unwrap()).unwrap();
                ns.ancestor_fenced = true;
                for r in ns.sink.records.clone() {
                    if r.status == Status::Cancelled {
                        let _ = ns.broker.settle(&r.permit.proposal.id, Status::Cancelled);
                    }
                }
                nexts.push(ns);
            }
        }

        for ns in nexts {
            assert_global_invariants(&ns.broker, &ns.sink, &ns.effects);
            if visited.insert(ns.key()) {
                queue.push_back(ns);
            }
        }
    }

    assert!(witnessed_cancel_refund_then_sibling_commit);
}

#[derive(Clone)]
struct DelegatedExtraStewardWorld {
    broker: Broker,
    sink: Sink,
    effects: EffectsLog,
    in_flight_to_sink: Vec<DispatchedPermit>,
    in_flight_to_settle: Vec<(String, Status)>,
    steward_withdrawn: bool,
    steward_fenced: bool,
    steward_renewed: bool,
}

impl DelegatedExtraStewardWorld {
    fn key(&self) -> String {
        serde_json::to_string(&json!({
            "b": &self.broker,
            "s": &self.sink,
            "e": &self.effects,
            "qs": &self.in_flight_to_sink,
            "qc": &self.in_flight_to_settle,
            "sw": self.steward_withdrawn,
            "sf": self.steward_fenced,
            "sr": self.steward_renewed,
        }))
        .unwrap()
    }
}

#[test]
fn delegated_transfer_extra_steward_fenced_interleavings() {
    let mut broker = Broker::new(8);
    let sink = Sink::default();
    let child = transfer_permit("bob", "delegated", 1, vec![0, 0], 400);

    assert_eq!(
        broker
            .delegate(&json!({
                "op": "delegate",
                "id": "root-cap",
                "budget": 8,
                "expires": 500,
                "scope": [true, false, false],
                "required": [true, true],
                "grants": [true, true],
                "epochs": [0, 0]
            }))
            .unwrap()["status"],
        "delegated"
    );
    assert_eq!(
        broker
            .delegate(&json!({
                "op": "delegate",
                "parent": "root-cap",
                "permit": child
            }))
            .unwrap()["status"],
        "delegated"
    );

    let start = DelegatedExtraStewardWorld {
        broker,
        sink,
        effects: vec![],
        in_flight_to_sink: vec![],
        in_flight_to_settle: vec![],
        steward_withdrawn: false,
        steward_fenced: false,
        steward_renewed: false,
    };

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    visited.insert(start.key());
    queue.push_back(start);

    let mut witnessed_in_flight_cancelled_by_steward_fence = false;

    while let Some(st) = queue.pop_front() {
        assert_global_invariants(&st.broker, &st.sink, &st.effects);
        if st.steward_fenced
            && !self_has_effect(&st.effects, "delegated")
            && st.broker.charged == 0
            && st
                .broker
                .records
                .iter()
                .any(|r| r.permit.proposal.id == "delegated" && r.status == Status::Cancelled)
        {
            witnessed_in_flight_cancelled_by_steward_fence = true;
        }

        let mut nexts = Vec::new();

        // 1. Proposer submits/retries child (never payload_mismatch on retry!).
        {
            let mut ns = st.clone();
            let r = ns.broker.submit(child.clone(), 10).unwrap();
            assert_ne!(r["reason"], "payload_mismatch");
            if r["status"] == "pending" {
                let dispatched: DispatchedPermit =
                    serde_json::from_value(r["permit"].clone()).unwrap();
                assert_eq!(dispatched.effective_required(2).unwrap(), vec![true, true]);
                if !ns.in_flight_to_sink.contains(&dispatched) && ns.in_flight_to_sink.len() < 2 {
                    ns.in_flight_to_sink.push(dispatched);
                }
            }
            nexts.push(ns);
        }

        // 2. Full proposer retry (submit -> accept_dispatched -> settle if terminal).
        {
            let mut ns = st.clone();
            let r = ns.broker.submit(child.clone(), 10).unwrap();
            assert_ne!(r["reason"], "payload_mismatch");
            if r["status"] == "pending" {
                let dispatched: DispatchedPermit =
                    serde_json::from_value(r["permit"].clone()).unwrap();
                let acc = ns.sink.accept_dispatched(dispatched).unwrap();
                match acc["status"].as_str() {
                    Some("committed") => {
                        let _ = ns.broker.settle("delegated", Status::Committed).unwrap();
                    }
                    Some("cancelled") => {
                        let _ = ns.broker.settle("delegated", Status::Cancelled).unwrap();
                    }
                    _ => {}
                }
            }
            nexts.push(ns);
        }

        // 3. Deliver in-flight DispatchedPermit to Sink.
        for idx in 0..st.in_flight_to_sink.len() {
            let mut ns = st.clone();
            let dp = ns.in_flight_to_sink.remove(idx);
            let acc = ns.sink.accept_dispatched(dp).unwrap();
            if acc["status"] == "cancelled" {
                enqueue_settle(
                    &mut ns.in_flight_to_settle,
                    ("delegated".into(), Status::Cancelled),
                );
            }
            nexts.push(ns);
        }

        // 4. Sink::complete on pending "delegated".
        if st
            .sink
            .records
            .iter()
            .any(|r| r.permit.proposal.id == "delegated" && r.status == Status::Pending)
        {
            let mut ns = st.clone();
            let (resp, newly) = ns.sink.complete("delegated").unwrap();
            if let Some(cp) = newly {
                assert!(
                    !ns.steward_fenced,
                    "VIOLATION: delegated Transfer with required=[true,true] committed after steward fence!"
                );
                let c = cp.proposal.effect.charge();
                ns.effects.push(("delegated".into(), cp, c));
            }
            let term: Status = serde_json::from_value(resp["status"].clone()).unwrap();
            enqueue_settle(&mut ns.in_flight_to_settle, ("delegated".into(), term));
            nexts.push(ns);
        }

        // 5. Deliver settlement to Broker.
        for idx in 0..st.in_flight_to_settle.len() {
            let mut ns = st.clone();
            let (id, term) = ns.in_flight_to_settle.remove(idx);
            let _ = ns.broker.settle(&id, term).unwrap();
            nexts.push(ns);
        }

        // 6. Withdraw steward issuer 1.
        if !st.steward_withdrawn {
            let mut ns = st.clone();
            let expected_out = usize::from(
                ns.broker
                    .records
                    .iter()
                    .any(|r| r.permit.proposal.id == "delegated" && r.status == Status::Pending),
            );
            let rev = ns.broker.withdraw(1).unwrap();
            assert_eq!(rev["status"], "admission_revoked");
            assert_eq!(rev["outstanding"].as_u64().unwrap() as usize, expected_out);
            ns.steward_withdrawn = true;
            nexts.push(ns);
        }

        // 7. Fence steward issuer 1 at floor = 1 on Sink.
        if st.steward_withdrawn && !st.steward_fenced {
            let mut ns = st.clone();
            let ack = ns.sink.fence(1, 1).unwrap();
            assert_eq!(ack["status"], "fenced");
            ns.steward_fenced = true;
            if ns
                .sink
                .records
                .iter()
                .any(|r| r.permit.proposal.id == "delegated" && r.status == Status::Cancelled)
            {
                enqueue_settle(
                    &mut ns.in_flight_to_settle,
                    ("delegated".into(), Status::Cancelled),
                );
            }
            nexts.push(ns);
        }

        // 8. Renew steward issuer 1 after withdraw (must NOT re-authorize epoch-0 leaf!).
        if st.steward_withdrawn && !st.steward_renewed {
            let mut ns = st.clone();
            assert_eq!(ns.broker.renew(1).unwrap()["status"], "renewed");
            ns.steward_renewed = true;
            nexts.push(ns);
        }

        for ns in nexts {
            assert_global_invariants(&ns.broker, &ns.sink, &ns.effects);
            if visited.insert(ns.key()) {
                queue.push_back(ns);
            }
        }
    }

    assert!(
        witnessed_in_flight_cancelled_by_steward_fence,
        "expected schedule witnessing in-flight delegated Transfer cancelled by steward fence!"
    );
}

#[test]
fn proposer_disposition_independence_across_four_strategies() {
    for strategy in [
        "guardian",
        "warrior_opportunistic",
        "replay_storm",
        "adaptive_boundary_seeker",
    ] {
        let mut broker = Broker::new(6);
        let mut sink = Sink::default();
        let mut effects: EffectsLog = Vec::new();

        let valid_1 = transfer_permit("alice", "ok-1", 3, vec![0, 0], 1_000);
        let valid_2 = transfer_permit("alice", "ok-2", 3, vec![0, 0], 1_000);
        let valid_over = transfer_permit("alice", "approved-but-over-cap", 2, vec![0, 0], 1_000);
        for p in [&valid_1, &valid_2, &valid_over] {
            broker.approve(p.clone(), 0).unwrap();
        }

        let candidates: Vec<(Permit, u64, bool)> = match strategy {
            "guardian" => vec![(valid_1.clone(), 10, true), (valid_2.clone(), 11, true)],
            "warrior_opportunistic" => vec![
                // Unapproved attacker transfer
                (
                    Permit {
                        actor: "alice".into(),
                        proposal: Proposal {
                            id: "unapproved".into(),
                            epochs: vec![0, 0],
                            expires: 1_000,
                            effect: Effect::Transfer {
                                recipient: "attacker".into(),
                                units: 6,
                            },
                        },
                    },
                    10,
                    false,
                ),
                // Actor impersonation on valid_1
                (
                    Permit {
                        actor: "mallory".into(),
                        proposal: valid_1.proposal.clone(),
                    },
                    10,
                    false,
                ),
                // Expired submission
                (valid_1.clone(), 1_000, false),
                // Valid 1 & Valid 2 consume all 6 units
                (valid_1.clone(), 10, true),
                (valid_2.clone(), 11, true),
                // Budget overrun even with valid approval!
                (valid_over.clone(), 12, false),
            ],
            "replay_storm" => vec![
                (valid_1.clone(), 10, true),
                (valid_1.clone(), 10, true),
                (valid_1.clone(), 10, true),
                // Mutated replay under same ID
                (
                    transfer_permit("alice", "ok-1", 4, vec![0, 0], 1_000),
                    10,
                    false,
                ),
                (valid_2.clone(), 12, true),
                (valid_over.clone(), 13, false),
            ],
            "adaptive_boundary_seeker" => {
                let over_utf8 = "é".repeat(65); // 130 UTF-8 bytes > 128
                vec![
                    (
                        transfer_permit("alice", &over_utf8, 1, vec![0, 0], 1_000),
                        10,
                        false,
                    ),
                    (
                        transfer_permit(&over_utf8, "ok-1", 3, vec![0, 0], 1_000),
                        10,
                        false,
                    ),
                    (
                        release_permit("alice", "steward-missing", vec![0, 0], 1_000),
                        10,
                        false,
                    ),
                    (valid_1.clone(), 10, true),
                    (valid_2.clone(), 11, true),
                    (valid_over.clone(), 12, false),
                ]
            }
            _ => unreachable!(),
        };

        let mut unauthorized_proposals = 0usize;
        let mut unauthorized_effects = 0usize;

        for (perm, now, is_authorized) in candidates {
            if !is_authorized {
                unauthorized_proposals += 1;
            }
            let id = perm.proposal.id.clone();
            let r = broker.submit(perm, now).unwrap();
            if r["status"] == "pending" {
                let dispatched: DispatchedPermit =
                    serde_json::from_value(r["permit"].clone()).unwrap();
                let acc = sink.accept_dispatched(dispatched).unwrap();
                if acc["status"] == "pending" {
                    let (cmp, newly) = sink.complete(&id).unwrap();
                    if let Some(cp) = newly {
                        if !is_authorized {
                            unauthorized_effects += 1;
                        }
                        let c = cp.proposal.effect.charge();
                        effects.push((id.clone(), cp, c));
                    }
                    let st: Status = serde_json::from_value(cmp["status"].clone()).unwrap();
                    let _ = broker.settle(&id, st).unwrap();
                }
            }
            assert_global_invariants(&broker, &sink, &effects);
        }

        if strategy == "guardian" {
            assert_eq!(unauthorized_proposals, 0);
        } else {
            assert!(unauthorized_proposals >= 2);
        }
        assert_eq!(
            unauthorized_effects, 0,
            "strategy {strategy} produced unauthorized effect!"
        );
        assert_eq!(broker.charged, 6);
        assert_eq!(effects.len(), 2);
    }
}
