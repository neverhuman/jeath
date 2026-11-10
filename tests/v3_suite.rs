use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn v3_cli_quick_suite_emits_all_verified_artifacts() {
    let exe = env!("CARGO_BIN_EXE_authority-lab");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("test-evidence")
        .join(format!("v3-suite-{}-{stamp}", std::process::id()));

    let status = Command::new(exe)
        .args(["run-v3"])
        .arg(&out_dir)
        .args(["--quick", "--seed", "20261110"])
        .status()
        .expect("spawn authority-lab run-v3");
    assert!(status.success(), "run-v3 --quick failed: {status}");

    let expected_files = [
        "metadata.json",
        "uncertainty.json",
        "composition.json",
        "risk_horizon.json",
        "jarvis_ledger.json",
        "personas.json",
        "weighted_cuts.json",
        "interdiction.json",
        "coverage_v3.json",
        "audit_power.json",
        "copula.json",
        "monitors_v3.json",
        "assessment_variance.json",
        "selection_v3.json",
        "complete.json",
    ];
    for name in expected_files {
        let path = out_dir.join(name);
        assert!(path.is_file(), "missing V3 output file: {}", path.display());
    }

    let read_val = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&fs::read(out_dir.join(name)).expect(name)).expect(name)
    };

    let complete = read_val("complete.json");
    assert_eq!(complete["complete"], true);
    assert_eq!(complete["schema_version"], 3);

    let meta = read_val("metadata.json");
    assert_eq!(meta["schema_version"], 3);
    assert_eq!(meta["quick"], true);
    assert_eq!(meta["gpu_used"], false);

    let unc = read_val("uncertainty.json");
    assert_eq!(unc["algebraic_contraction_verified"], true);
    assert_eq!(unc["conservative_zero_violations_all_lags"], true);

    let comp = read_val("composition.json");
    assert_eq!(comp["b100_admissible"], 348_551);
    assert_eq!(comp["b100_overspend"], 171_700);
    assert_eq!(comp["b100_serialized_overspend"], 0);
    assert!(
        comp["rows"]
            .as_array()
            .expect("rows")
            .iter()
            .all(|r| r["closed_form_exact_match"] == true)
    );

    let cuts = read_val("weighted_cuts.json");
    assert_eq!(cuts["brute_force_discrepancies"], 0);

    let cov = read_val("coverage_v3.json");
    assert_eq!(cov["blind_inflates_gt_estimate_verified"], true);
    assert_eq!(cov["withheld_zero_critical_recall_verified"], true);

    let cop = read_val("copula.json");
    assert!(!cop["curves"].as_object().expect("curves").is_empty());

    let sel = read_val("selection_v3.json");
    assert!(
        sel["price_correlated_drift"]["max_finite_population_price_identity_error"]
            .as_f64()
            .expect("err")
            < 1e-12
    );

    // Record summary receipt in test-evidence for CI artifact glob compatibility.
    fs::write(
        out_dir.join("result.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "suite": "v3_suite",
            "passed": true,
            "files": expected_files.len(),
        }))
        .expect("json"),
    )
    .expect("write result.json");
}
