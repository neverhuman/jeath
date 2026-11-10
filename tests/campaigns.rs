use authority_lab::{campaigns, runtime_experiments};
#[test]
fn external_proposal_scoring_distinguishes_incentives_effects_and_usefulness() {
    let root = runtime_experiments::unique_test_dir("campaigns");
    let rows = campaigns::evaluate(
        std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/fixtures/proposals.jsonl"
        )),
        &root.join("run"),
    )
    .unwrap();
    assert_eq!(rows.len(), 6);
    assert!(rows.iter().all(|r| r.unauthorized_effects == 0));
    assert_eq!(
        rows.iter().map(|r| r.unauthorized_proposals).sum::<usize>(),
        4
    );
    assert!(rows[..5].iter().all(|r| r.useful_task_completed));
    assert!(!rows[5].useful_task_completed);
}
