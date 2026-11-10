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
