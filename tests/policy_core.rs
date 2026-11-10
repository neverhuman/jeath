use authority_lab::protocol_v2::{Approval, Broker, Effect, Permit, Proposal};

#[test]
fn core_allows_useful_requests_and_rejects_changed_bytes() {
    let bytes = vec![0, 255, 17];
    assert!(policy_core::admit(
        10,
        8,
        2,
        true,
        3,
        3,
        4,
        5,
        7,
        7,
        &bytes,
        &bytes,
        &vec![true, true],
        &vec![true, true]
    ));
    assert!(!policy_core::same_bytes(&bytes, &vec![0, 255, 18]));
    assert!(!policy_core::same_bytes(&bytes, &vec![0, 255]));
    assert_eq!(policy_core::reserve(u64::MAX, u64::MAX, 1), None);
    assert_eq!(
        policy_core::reserve(u64::MAX, u64::MAX - 1, 1),
        Some(u64::MAX)
    );
    assert_eq!(policy_core::release(2, 2, false), 2);
    assert_eq!(policy_core::release(2, 2, true), 0);
    assert_eq!(policy_core::release(2, 5, true), 2);
    assert!(policy_core::revision_matches(4, 4));
    assert!(!policy_core::revision_matches(4, 5));
}

#[test]
fn arbitrary_finite_issuer_vectors_and_independent_withdrawal() {
    for n in 0..64 {
        let mut required = vec![true; n];
        let grants = vec![true; n];
        let mut live = vec![true; n];
        let epoch = vec![7; n];
        assert!(policy_core::current_grants(
            &required, &grants, &live, &epoch, &epoch
        ));
        if n > 0 {
            live[n - 1] = false;
            assert!(!policy_core::current_grants(
                &required, &grants, &live, &epoch, &epoch
            ));
            required[n - 1] = false;
            assert!(policy_core::current_grants(
                &required, &grants, &live, &epoch, &epoch
            ));
        }
    }
}

#[test]
fn verified_permissions_do_not_prove_task_correctness() {
    let mut b = Broker::new(1);
    let permit = Permit {
        actor: "alice".into(),
        proposal: Proposal {
            id: "bad-edit".into(),
            epochs: vec![0, 0],
            expires: 100,
            effect: Effect::Edit {
                document: "toy.rs".into(),
                expected_revision: 0,
                contents: "fn answer() -> u32 { 0 }".into(),
            },
        },
    };
    b.approvals.push(Approval {
        permit: permit.clone(),
        grants: vec![true, true],
    });
    assert_eq!(b.submit(permit, 1).unwrap()["status"], "pending");
    // The task's content oracle expects 42. Authorization alone admits 0.
}

#[test]
fn admit_multi_defensive_lengths_overflows_and_quorums() {
    let payload = vec![10, 20, 30, 40];
    let req = vec![true, false, true];
    let grants = vec![true, false, true];
    let live = vec![true, false, true];
    let epochs = vec![2, 99, 5];
    let requested = vec![2, 0, 5];

    // Valid admission: non-required middle issuer has grant=false, live=false, mismatched epoch.
    assert!(policy_core::admit_multi(
        100, 40, 60, 99, 100, &payload, &payload, &req, &grants, &live, &epochs, &requested
    ));

    // Zero amount, over-budget, overflow boundary, and expired timestamps fail closed.
    assert!(!policy_core::admit_multi(
        100, 40, 0, 99, 100, &payload, &payload, &req, &grants, &live, &epochs, &requested
    ));
    assert!(!policy_core::admit_multi(
        100, 41, 60, 99, 100, &payload, &payload, &req, &grants, &live, &epochs, &requested
    ));
    assert!(!policy_core::admit_multi(
        u64::MAX,
        u64::MAX,
        1,
        0,
        1,
        &payload,
        &payload,
        &req,
        &grants,
        &live,
        &epochs,
        &requested
    ));
    assert!(!policy_core::admit_multi(
        100, 40, 60, 100, 100, &payload, &payload, &req, &grants, &live, &epochs, &requested
    ));

    // Payload mismatch fails closed.
    let tampered = vec![10, 20, 30, 41];
    assert!(!policy_core::admit_multi(
        100, 40, 60, 99, 100, &payload, &tampered, &req, &grants, &live, &epochs, &requested
    ));

    // Every mismatched vector length returns false without panicking.
    assert!(!policy_core::admit_multi(
        100,
        0,
        1,
        0,
        10,
        &payload,
        &payload,
        &req,
        &vec![true, true],
        &live,
        &epochs,
        &requested
    ));
    assert!(!policy_core::admit_multi(
        100,
        0,
        1,
        0,
        10,
        &payload,
        &payload,
        &req,
        &grants,
        &vec![true, true],
        &epochs,
        &requested
    ));
    assert!(!policy_core::admit_multi(
        100,
        0,
        1,
        0,
        10,
        &payload,
        &payload,
        &req,
        &grants,
        &live,
        &vec![2, 99],
        &requested
    ));
    assert!(!policy_core::admit_multi(
        100,
        0,
        1,
        0,
        10,
        &payload,
        &payload,
        &req,
        &grants,
        &live,
        &epochs,
        &vec![2, 0]
    ));
    assert!(!policy_core::grants_ok(&vec![true], &vec![true, true]));
}

#[test]
fn admit_uncertain_monotonic_contraction_and_safety() {
    let payload = vec![1, 2, 3];
    let req = vec![true, true];
    let grants = vec![true, true];
    let live = vec![true, true];
    let epochs = vec![1, 1];

    // Inverted interval [8, 4] safely returns false.
    assert!(!policy_core::admit_uncertain(
        10, 8, 4, 1, 0, 10, &payload, &payload, &req, &grants, &live, &epochs, &epochs
    ));

    // Strict availability contraction witness: [2, 7] admits amount=3 under budget=10, while [2, 9] rejects.
    assert!(policy_core::admit_uncertain(
        10, 2, 7, 3, 0, 10, &payload, &payload, &req, &grants, &live, &epochs, &epochs
    ));
    assert!(!policy_core::admit_uncertain(
        10, 2, 9, 3, 0, 10, &payload, &payload, &req, &grants, &live, &epochs, &epochs
    ));

    // Exhaustive grid check of H2 monotonicity and soundness at unknown true_charged.
    let budget = 10u64;
    for lo_wide in 0..=12 {
        for lo_tight in lo_wide..=12 {
            for true_charged in lo_tight..=12 {
                for hi_tight in true_charged..=12 {
                    for hi_wide in hi_tight..=12 {
                        for amount in 0..=12 {
                            let ok_wide = policy_core::admit_uncertain(
                                budget, lo_wide, hi_wide, amount, 0, 5, &payload, &payload, &req,
                                &grants, &live, &epochs, &epochs,
                            );
                            let ok_tight = policy_core::admit_uncertain(
                                budget, lo_tight, hi_tight, amount, 0, 5, &payload, &payload, &req,
                                &grants, &live, &epochs, &epochs,
                            );
                            let ok_true = policy_core::admit_multi(
                                budget,
                                true_charged,
                                amount,
                                0,
                                5,
                                &payload,
                                &payload,
                                &req,
                                &grants,
                                &live,
                                &epochs,
                                &epochs,
                            );
                            if ok_wide {
                                assert!(ok_tight);
                            }
                            if ok_tight {
                                assert!(ok_true);
                                assert!(true_charged + amount <= budget);
                            }
                            if lo_tight == hi_tight {
                                assert_eq!(ok_tight, ok_true);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn cumulative_conservation_recycling_and_timeout_nonrefund() {
    let budget = 10u64;
    let mut charged = 0u64;
    let mut committed = 0u64;
    let mut inflight: Vec<u64> = Vec::new();

    // 20 authoritative reserve -> cancel cycles recycle capacity (> 80 total reserved)
    // without minting budget or drifting from `charged == committed + sum(inflight)`.
    for _ in 0..20 {
        let next = policy_core::reserve(budget, charged, 4).expect("4 fits in budget 10");
        charged = next;
        inflight.push(4);
        assert_eq!(charged, committed + inflight.iter().sum::<u64>());
        // Authoritative cancellation refunds 4.
        let amt = inflight.pop().unwrap();
        charged = policy_core::release(charged, amt, true);
        assert_eq!(charged, committed + inflight.iter().sum::<u64>());
    }

    // Reserve 6 (commits) and 3 (uncertain timeout: authoritative_no_effect = false).
    charged = policy_core::reserve(budget, charged, 6).unwrap();
    inflight.push(6);
    charged = policy_core::reserve(budget, charged, 3).unwrap();
    inflight.push(3);

    // Commit first reservation (slot 0 -> 6).
    let c0 = inflight.remove(0);
    charged = policy_core::release(charged, c0, false);
    committed += c0;
    assert_eq!(committed, 6);
    assert_eq!(charged, 9);

    // Uncertain timeout on second reservation (3) MUST NOT release capacity.
    charged = policy_core::release(charged, inflight[0], false);
    assert_eq!(charged, 9);
    // Attempting to reserve 2 now fails closed because 9 + 2 > 10.
    assert_eq!(policy_core::reserve(budget, charged, 2), None);

    // When the timed-out reservation later commits at the sink, committed == 9 <= 10.
    let c1 = inflight.remove(0);
    committed += c1;
    assert_eq!(committed, 9);
    assert!(committed <= budget);
}

#[test]
fn monotone_fencing_permanent_exclusion_and_noninterference() {
    assert_eq!(policy_core::advance_floor(5, 3), 5);
    assert_eq!(policy_core::advance_floor(5, 5), 5);
    assert_eq!(policy_core::advance_floor(5, 9), 9);
    assert_eq!(policy_core::advance_floor(u64::MAX, 0), u64::MAX);

    // Defensive length checks on unfenced.
    assert!(!policy_core::unfenced(
        &vec![true],
        &vec![0, 0],
        &vec![0, 0]
    ));
    assert!(!policy_core::unfenced(
        &vec![true, true],
        &vec![0],
        &vec![0, 0]
    ));
    assert!(!policy_core::unfenced(
        &vec![true, true],
        &vec![0, 0],
        &vec![0]
    ));

    let req_op_only = vec![true, false, false];
    let req_joint = vec![true, true, false];
    let permit_epochs = vec![2, 1, 0];
    let mut floors = vec![2, 1, 0];

    assert!(policy_core::unfenced(&req_op_only, &permit_epochs, &floors));
    assert!(policy_core::unfenced(&req_joint, &permit_epochs, &floors));

    // Issuer 1 (steward) fences to floor 2: blocks joint permit permanently, never blocks op-only!
    floors[1] = policy_core::advance_floor(floors[1], 2);
    assert!(policy_core::unfenced(&req_op_only, &permit_epochs, &floors));
    assert!(!policy_core::unfenced(&req_joint, &permit_epochs, &floors));

    // Subsequent arbitrary fence proposals (including lower proposed floor 0 on issuer 1,
    // and high floor 99 on non-required issuer 2) keep joint excluded and op-only unfenced.
    for (issuer, proposed) in [(1usize, 0u64), (2, 99), (0, 2), (1, 5)] {
        let prev = floors[issuer];
        floors[issuer] = policy_core::advance_floor(floors[issuer], proposed);
        assert!(floors[issuer] >= prev);
        assert!(policy_core::unfenced(&req_op_only, &permit_epochs, &floors));
        assert!(!policy_core::unfenced(&req_joint, &permit_epochs, &floors));
    }
}

#[test]
fn attenuates_defensive_checks_and_delegation_chain() {
    let root_scope = vec![true, true, true, false];
    let mid_scope = vec![true, true, false, false];
    let leaf_scope = vec![true, false, false, false];

    let root_req = vec![true, false, false];
    let mid_req = vec![true, true, false];
    let leaf_req = vec![true, true, true];

    // Valid chain: root (100, exp 50) -> mid (60, exp 40) -> leaf (25, exp 30).
    assert!(policy_core::attenuates(
        100,
        60,
        50,
        40,
        &root_scope,
        &mid_scope,
        &root_req,
        &mid_req
    ));
    assert!(policy_core::attenuates(
        60,
        25,
        40,
        30,
        &mid_scope,
        &leaf_scope,
        &mid_req,
        &leaf_req
    ));
    // Transitive root -> leaf attenuation holds.
    assert!(policy_core::attenuates(
        100,
        25,
        50,
        30,
        &root_scope,
        &leaf_scope,
        &root_req,
        &leaf_req
    ));

    // Escalation attempts fail closed: budget, expiry, scope widening, dropping required issuer.
    assert!(!policy_core::attenuates(
        60,
        61,
        40,
        30,
        &mid_scope,
        &leaf_scope,
        &mid_req,
        &leaf_req
    ));
    assert!(!policy_core::attenuates(
        60,
        25,
        40,
        41,
        &mid_scope,
        &leaf_scope,
        &mid_req,
        &leaf_req
    ));
    assert!(!policy_core::attenuates(
        60,
        25,
        40,
        30,
        &mid_scope,
        &vec![true, false, true, false],
        &mid_req,
        &leaf_req
    ));
    assert!(!policy_core::attenuates(
        60,
        25,
        40,
        30,
        &mid_scope,
        &leaf_scope,
        &mid_req,
        &vec![true, false, true]
    ));

    // Length mismatches fail closed without panicking.
    assert!(!policy_core::attenuates(
        100,
        25,
        50,
        30,
        &root_scope,
        &vec![true],
        &root_req,
        &leaf_req
    ));
    assert!(!policy_core::attenuates(
        100,
        25,
        50,
        30,
        &root_scope,
        &leaf_scope,
        &root_req,
        &vec![true]
    ));
}

#[test]
fn proposer_disposition_independence_across_streams() {
    #[derive(Clone)]
    struct Candidate {
        amount: u64,
        now: u64,
        expires: u64,
        payload: Vec<u8>,
        approved: Vec<u8>,
        grants: Vec<bool>,
    }

    let budget = 10u64;
    let req = vec![true, true];
    let live = vec![true, true];
    let epochs = vec![3, 3];
    let ok_bytes = vec![42, 1];

    let guardian = vec![
        Candidate {
            amount: 4,
            now: 1,
            expires: 10,
            payload: ok_bytes.clone(),
            approved: ok_bytes.clone(),
            grants: vec![true, true],
        },
        Candidate {
            amount: 6,
            now: 2,
            expires: 10,
            payload: ok_bytes.clone(),
            approved: ok_bytes.clone(),
            grants: vec![true, true],
        },
    ];
    let warrior = vec![
        Candidate {
            amount: 8,
            now: 1,
            expires: 10,
            payload: ok_bytes.clone(),
            approved: ok_bytes.clone(),
            grants: vec![true, true],
        },
        Candidate {
            amount: 8,
            now: 2,
            expires: 10,
            payload: ok_bytes.clone(),
            approved: ok_bytes.clone(),
            grants: vec![true, true],
        },
        Candidate {
            amount: 2,
            now: 15, // expired
            expires: 10,
            payload: ok_bytes.clone(),
            approved: ok_bytes.clone(),
            grants: vec![true, true],
        },
        Candidate {
            amount: 2,
            now: 3,
            expires: 10,
            payload: vec![99, 1], // tampered payload
            approved: ok_bytes.clone(),
            grants: vec![true, true],
        },
        Candidate {
            amount: 2,
            now: 3,
            expires: 10,
            payload: ok_bytes.clone(),
            approved: ok_bytes.clone(),
            grants: vec![true, false], // missing grant
        },
    ];

    for stream in [&guardian, &warrior] {
        let mut charged = 0u64;
        for c in stream {
            if policy_core::admit_multi(
                budget,
                charged,
                c.amount,
                c.now,
                c.expires,
                &c.payload,
                &c.approved,
                &req,
                &c.grants,
                &live,
                &epochs,
                &epochs,
            ) {
                // Every admitted candidate is genuinely valid and fits remaining budget.
                assert!(c.amount > 0 && charged + c.amount <= budget);
                assert!(
                    c.now < c.expires && c.payload == c.approved && c.grants == vec![true, true]
                );
                charged = policy_core::reserve(budget, charged, c.amount).unwrap();
            }
        }
        assert!(charged <= budget);
    }

    // Stateless per-call check diverges on warrior stream (admits 16 > 10).
    let stateless_spend: u64 = warrior
        .iter()
        .filter(|c| {
            c.amount > 0
                && c.amount <= budget
                && c.now < c.expires
                && c.payload == c.approved
                && c.grants == vec![true, true]
        })
        .map(|c| c.amount)
        .sum();
    assert_eq!(stateless_spend, 16);
    assert!(stateless_spend > budget);
}
