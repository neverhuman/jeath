use authority_lab::runtime_experiments;
#[test]
fn real_processes_and_persistent_effects() {
    let directory = runtime_experiments::unique_test_dir("runtime");
    let result = runtime_experiments::run(
        std::path::Path::new(env!("CARGO_BIN_EXE_authority-lab")),
        &directory.join("run"),
        true,
    )
    .unwrap();
    assert!(
        result
            .scenarios
            .iter()
            .all(|s| s.passed && s.audit.consistent())
    );
    assert_eq!(result.race.executed, 1);
    assert!(
        result
            .throughput
            .iter()
            .all(|b| b.executed == 64 && b.denials.is_empty())
    );
    runtime_experiments::write_json(&directory.join("result.json"), &result).unwrap();
}
