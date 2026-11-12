use authority_lab::v5_e2e_crypto_cluster::e2e_crypto_cluster_suite;
use std::{fs, path::Path};

#[test]
fn multi_process_sqlite_wal_crypto_cluster_invariants_hold() {
    let target_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("test-e2e-crypto-cluster-v5");
    if target_dir.exists() {
        fs::remove_dir_all(&target_dir).expect("clean prior test dir");
    }
    fs::create_dir_all(&target_dir).expect("create target test dir");

    let rep = e2e_crypto_cluster_suite(&target_dir, 20261112, true)
        .expect("multi-process SQLite WAL cluster suite must succeed");

    assert_eq!(rep.schema_version, 5);
    assert!(rep.sqlite_wal_mode_verified);
    assert!(rep.separate_os_processes_verified);
    assert!(rep.transient_sqlite_cleaned_up);

    // Campaign A: Zero panics, zero uncanonical accepts, zero parser differentials
    assert_eq!(rep.wire_fuzz.panics, 0);
    assert_eq!(rep.wire_fuzz.uncanonical_frames_accepted, 0);
    assert_eq!(rep.wire_fuzz.forged_crypto_frames_accepted, 0);
    assert_eq!(rep.wire_fuzz.broker_sink_decoder_differentials, 0);
    assert_eq!(
        rep.wire_fuzz.valid_control_roundtrips_passed,
        rep.wire_fuzz.valid_control_roundtrips_checked
    );

    // Campaign B: Zero smurfing conservation breaches, zero amplifying caveats, zero post-ack commits
    assert!(
        rep.smurfing_delegation.committed_total_charge <= rep.smurfing_delegation.global_budget
    );
    assert_eq!(rep.smurfing_delegation.partition_sum_invariant_breaches, 0);
    assert_eq!(rep.smurfing_delegation.amplifying_caveats_accepted, 0);
    assert_eq!(
        rep.smurfing_delegation.unauthorized_self_upgrades_accepted,
        0
    );
    assert_eq!(
        rep.smurfing_delegation.authorized_self_upgrades_accepted,
        rep.smurfing_delegation.unauthorized_self_upgrade_attempts
    );
    assert!(
        !rep.smurfing_delegation
            .cluster_quorum_fence_acked_before_partitioned_expiry
    );
    assert!(
        rep.smurfing_delegation
            .cluster_quorum_fence_acked_at_lease_expiry
    );
    assert_eq!(rep.smurfing_delegation.post_quorum_ack_commits, 0);
    assert!(rep.smurfing_delegation.edit_cas_conflicts_prevented > 0);

    // Campaign C: Hard OS SIGKILL + SQLite WAL restart preserves durable floors and idempotence
    assert_eq!(
        rep.crash_recovery.durable_wal_restarts_succeeded,
        rep.crash_recovery.hard_sigkill_cycles
    );
    assert_eq!(rep.crash_recovery.floor_regressions_after_crash, 0);
    assert_eq!(rep.crash_recovery.double_commits_after_crash, 0);
    assert_eq!(rep.crash_recovery.post_fence_commits_after_crash, 0);

    // Campaign D: Finite positive microsecond latency metrics
    assert!(rep.latency_us.end_to_end_p50_us > 0.0 && rep.latency_us.end_to_end_p99_us.is_finite());

    fs::remove_dir_all(&target_dir).expect("clean test dir");
}
