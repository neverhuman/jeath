use crate::runtime::{self, Audit, Denial, Error, Executor, Operation, Phase, Request};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, Barrier, mpsc},
    time::{Duration, Instant},
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut f, value)?;
    f.write_all(b"\n")?;
    Ok(())
}
fn ensure(condition: bool, message: &str) -> Result<()> {
    if !condition {
        return Err(std::io::Error::other(message).into());
    }
    Ok(())
}
pub fn spend(id: &str, amount: i64, epoch: i64) -> Request {
    Request {
        id: id.into(),
        actor: "actor0".into(),
        amount,
        epoch,
        expires_at: runtime::now() + 3600,
        operation: Operation::Spend {
            destination: "approved".into(),
        },
    }
}
pub fn provision(path: &Path, budget: i64) -> Result<Executor> {
    let mut e = Executor::open(path)?;
    e.initialize(budget, runtime::now() + 3600)?;
    Ok(e)
}
pub fn approve(e: &Executor, r: &Request) -> Result<()> {
    e.prepare(r)?;
    e.approve("operator", r)?;
    if !matches!(r.operation, Operation::Spend { .. }) {
        e.approve("steward", r)?;
    }
    Ok(())
}
fn denied(result: runtime::Result<runtime::Execution>, expected: Denial) -> Result<()> {
    match result {
        Err(Error::Denied(actual)) => ensure(
            actual == expected,
            &format!("expected {expected:?}, got {actual:?}"),
        ),
        other => Err(
            std::io::Error::other(format!("expected denial {expected:?}, got {other:?}")).into(),
        ),
    }
}
#[derive(Serialize)]
pub struct Scenario {
    pub name: String,
    pub passed: bool,
    pub audit: Audit,
}
#[derive(Serialize, Deserialize, Default)]
pub struct WorkerResult {
    pub executed: usize,
    pub replayed: usize,
    pub denied: BTreeMap<String, usize>,
    pub latency_us: Vec<f64>,
}
/// Internal benchmark worker; grant management is never exposed to its requests.
pub fn worker(db: &Path, requests: &Path, start: Option<&Path>, stop: Option<Phase>) -> Result<()> {
    let requests: Vec<Request> = serde_json::from_reader(fs::File::open(requests)?)?;
    let mut e = Executor::open(db)?;
    if let Some(start) = start {
        let deadline = Instant::now() + Duration::from_secs(15);
        while !start.exists() {
            ensure(Instant::now() < deadline, "start barrier timed out")?;
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    let mut out = WorkerResult::default();
    for r in requests {
        let begin = Instant::now();
        let result = e.execute_observed(&r, |phase| {
            if stop == Some(phase) {
                println!("fault-ready");
                std::io::stdout().flush().expect("flush fault marker");
                loop {
                    std::thread::park();
                }
            }
        });
        out.latency_us.push(begin.elapsed().as_secs_f64() * 1e6);
        match result {
            Ok(r) => {
                if r.replayed {
                    out.replayed += 1;
                } else {
                    out.executed += 1;
                }
            }
            Err(Error::Denied(d)) => *out.denied.entry(format!("{d:?}")).or_default() += 1,
            Err(e) => return Err(e.into()),
        }
    }
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
fn command(executable: &Path, db: &Path, requests: &Path, start: Option<&Path>) -> Command {
    let mut c = Command::new(executable);
    c.arg("worker")
        .arg(db)
        .arg(requests)
        .arg(start.unwrap_or(Path::new("-")));
    c
}
fn crash(executable: &Path, db: &Path, request_file: &Path, phase: &str) -> Result<()> {
    let mut child = command(executable, db, request_file, None)
        .arg(phase)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;
    let stdout = child.stdout.take().unwrap();
    let (send, receive) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut line = String::new();
        let result = BufReader::new(stdout).read_line(&mut line);
        let _ = send.send((result.is_ok(), line));
    });
    let response = receive.recv_timeout(Duration::from_secs(15));
    // Also clean up on missing markers. Never leave a parked child behind.
    let _ = child.kill();
    let status = child.wait()?;
    let _ = reader.join();
    let (read, line) = response?;
    ensure(
        read && line.trim() == "fault-ready" && !status.success(),
        "fault worker did not reach requested phase",
    )
}
#[derive(Serialize)]
pub struct Benchmark {
    pub workers: usize,
    pub attempts: usize,
    pub executed: usize,
    pub denials: BTreeMap<String, usize>,
    pub elapsed_seconds: f64,
    pub transactions_per_second: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    /// Raw observations, grouped by worker, with each worker's request order.
    pub operation_latency_us: Vec<f64>,
    pub audit: Audit,
}
fn parallel(
    executable: &Path,
    directory: &Path,
    workers: usize,
    per_worker: usize,
    budget: i64,
    amount: i64,
) -> Result<Benchmark> {
    fs::create_dir(directory)?;
    let db = directory.join("executor.sqlite");
    let e = provision(&db, budget)?;
    let start = directory.join("start");
    let mut files = Vec::new();
    for worker in 0..workers {
        let requests: Vec<_> = (0..per_worker)
            .map(|i| {
                let mut r = spend(&format!("worker{worker}-{i}"), amount, 0);
                r.actor = format!("actor{worker}");
                r
            })
            .collect();
        for r in &requests {
            approve(&e, r)?;
        }
        let path = directory.join(format!("requests-{worker}.json"));
        write_json(&path, &requests)?;
        files.push(path);
    }
    drop(e);
    let mut children = Vec::new();
    for path in &files {
        children.push(
            command(executable, &db, path, Some(&start))
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()?,
        );
    }
    let begin = Instant::now();
    fs::write(&start, b"start")?;
    let mut executed = 0;
    let mut latencies = Vec::new();
    let mut denials = BTreeMap::new();
    for child in children {
        let result = child.wait_with_output()?;
        ensure(result.status.success(), "benchmark worker failed")?;
        let out: WorkerResult = serde_json::from_slice(&result.stdout)?;
        executed += out.executed;
        latencies.extend(out.latency_us);
        for (key, n) in out.denied {
            *denials.entry(key).or_default() += n;
        }
    }
    let elapsed_seconds = begin.elapsed().as_secs_f64();
    let mut sorted_latencies = latencies.clone();
    sorted_latencies.sort_by(f64::total_cmp);
    let quantile = |p: f64| {
        sorted_latencies[((p * sorted_latencies.len() as f64).ceil() as usize).saturating_sub(1)]
    };
    let mut e = Executor::open(&db)?;
    let audit = e.audit()?;
    ensure(
        audit.consistent(),
        "independent runtime accounting disagrees",
    )?;
    ensure(
        executed as i64 * amount == audit.spent,
        "worker receipts disagree with observed effects",
    )?;
    Ok(Benchmark {
        workers,
        attempts: workers * per_worker,
        executed,
        denials,
        elapsed_seconds,
        transactions_per_second: executed as f64 / elapsed_seconds,
        p50_us: quantile(0.5),
        p95_us: quantile(0.95),
        p99_us: quantile(0.99),
        operation_latency_us: latencies,
        audit,
    })
}
#[derive(Serialize)]
pub struct RuntimeResult {
    pub sqlite_version: String,
    pub scenarios: Vec<Scenario>,
    pub race: Benchmark,
    pub throughput: Vec<Benchmark>,
}
fn checkpoint_stress(path: &Path, per_writer: usize) -> Result<Executor> {
    let e = provision(path, (4 * per_writer) as i64)?;
    let mut requests = Vec::new();
    for writer in 0..4 {
        let group: Vec<_> = (0..per_writer)
            .map(|i| spend(&format!("checkpoint-{writer}-{i}"), 1, 0))
            .collect();
        for r in &group {
            approve(&e, r)?;
        }
        requests.push(group);
    }
    drop(e);
    let barrier = Arc::new(Barrier::new(5));
    let mut threads = Vec::new();
    for group in requests {
        let path = path.to_path_buf();
        let barrier = barrier.clone();
        threads.push(std::thread::spawn(move || -> runtime::Result<()> {
            let mut e = Executor::open(path)?;
            barrier.wait();
            for r in group {
                e.execute(&r)?;
            }
            Ok(())
        }));
    }
    let checkpoint_path = path.to_path_buf();
    threads.push(std::thread::spawn(move || -> runtime::Result<()> {
        let e = Executor::open(checkpoint_path)?;
        barrier.wait();
        for _ in 0..500 {
            e.checkpoint()?;
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }));
    for thread in threads {
        thread
            .join()
            .map_err(|_| std::io::Error::other("checkpoint stress thread panicked"))??;
    }
    let mut e = Executor::open(path)?;
    ensure(
        e.audit()?.spent == (4 * per_writer) as i64,
        "checkpoint race lost a committed effect",
    )?;
    Ok(e)
}
pub fn run(executable: &Path, directory: &Path, small: bool) -> Result<RuntimeResult> {
    fs::create_dir(directory)?;
    let mut scenarios = Vec::new();
    let mut finish = |name: &str, e: &mut Executor| -> Result<()> {
        let audit = e.audit()?;
        ensure(audit.consistent(), name)?;
        scenarios.push(Scenario {
            name: name.into(),
            passed: true,
            audit,
        });
        Ok(())
    };
    // Useful effects in three database-resident domains.
    let mut e = provision(&directory.join("useful.sqlite"), 3)?;
    let r = spend("spend", 1, 0);
    approve(&e, &r)?;
    e.execute(&r)?;
    let mut release = spend("release", 1, 0);
    release.operation = Operation::Release {
        document: "synthetic".into(),
        destination: "approved".into(),
        purpose: "analysis".into(),
    };
    approve(&e, &release)?;
    e.execute(&release)?;
    let mut edit = spend("edit", 1, 0);
    edit.operation = Operation::Edit {
        document: "toy.rs".into(),
        expected_revision: 0,
        body: "fn answer() -> u32 { 42 }".into(),
    };
    approve(&e, &edit)?;
    e.execute(&edit)?;
    ensure(
        e.audit()?.observed_units == 3,
        "positive tasks did not execute",
    )?;
    finish("three_useful_domains", &mut e)?;

    let mut e = provision(&directory.join("grants.sqlite"), 4)?;
    e.prepare(&release)?;
    e.approve("operator", &release)?;
    denied(e.execute(&release), Denial::MissingGrant)?;
    e.approve("steward", &release)?;
    e.withdraw("steward", &release.id)?;
    denied(e.execute(&release), Denial::MissingGrant)?;
    e.approve("steward", &release)?;
    e.execute(&release)?;
    finish("required_steward_and_withdrawal", &mut e)?;

    let mut e = provision(&directory.join("payload.sqlite"), 2)?;
    let r = spend("bound", 1, 0);
    approve(&e, &r)?;
    let mut changed = r.clone();
    changed.amount = 2;
    denied(e.execute(&changed), Denial::PayloadMismatch)?;
    let mut bad = spend("destination", 1, 0);
    bad.operation = Operation::Spend {
        destination: "unapproved".into(),
    };
    approve(&e, &bad)?;
    denied(e.execute(&bad), Denial::Scope)?;
    let mut expired = spend("expired", 1, 0);
    expired.expires_at = runtime::now() - 1;
    approve(&e, &expired)?;
    denied(e.execute(&expired), Denial::Expired)?;
    let mut negative = spend("negative", -1, 0);
    denied(e.execute(&negative), Denial::InvalidRequest)?;
    negative.amount = i64::MAX;
    approve(&e, &negative)?;
    denied(e.execute(&negative), Denial::Budget)?;
    finish("payload_scope_expiry_and_numeric_boundaries", &mut e)?;

    let mut e = provision(&directory.join("revision.sqlite"), 3)?;
    approve(&e, &edit)?;
    e.execute(&edit)?;
    let mut stale = edit.clone();
    stale.id = "stale-edit".into();
    approve(&e, &stale)?;
    denied(e.execute(&stale), Denial::RevisionConflict)?;
    finish("stale_document_revision", &mut e)?;

    let path = directory.join("lifecycle.sqlite");
    let mut e = provision(&path, 2)?;
    let r = spend("first", 1, 0);
    approve(&e, &r)?;
    let first = e.execute(&r)?;
    let stale = spend("prepared-before-revoke", 1, 0);
    approve(&e, &stale)?;
    e.revoke()?;
    denied(e.execute(&stale), Denial::Revoked)?;
    e.renew()?;
    denied(e.execute(&stale), Denial::StaleEpoch)?;
    drop(e);
    let mut e = Executor::open(&path)?;
    ensure(
        e.execute(&r)?.receipt == first.receipt,
        "retry changed receipt after restart",
    )?;
    let r = spend("second", 1, 1);
    approve(&e, &r)?;
    e.execute(&r)?;
    let r = spend("overspend-after-renew", 1, 1);
    approve(&e, &r)?;
    denied(e.execute(&r), Denial::Budget)?;
    ensure(
        matches!(
            e.initialize(100, runtime::now() + 3600),
            Err(Error::Denied(Denial::AlreadyInitialized))
        ),
        "initialization reset existing policy",
    )?;
    ensure(
        e.audit()?.spent == 2 && e.audit()?.receipts == 2,
        "lifecycle duplicated or erased an effect",
    )?;
    finish("revoke_renew_restart_retry_preserve_ledger", &mut e)?;

    for phase in ["before", "after"] {
        let path = directory.join(format!("crash-{phase}.sqlite"));
        let e = provision(&path, 2)?;
        let r = spend("ambiguous-receipt", 1, 0);
        approve(&e, &r)?;
        drop(e);
        let file = directory.join(format!("crash-{phase}-request.json"));
        write_json(&file, &vec![r.clone()])?;
        crash(executable, &path, &file, phase)?;
        let mut e = Executor::open(&path)?;
        ensure(
            e.audit()?.spent == if phase == "before" { 0 } else { 1 },
            "process death broke atomicity",
        )?;
        let retry = e.execute(&r)?;
        ensure(
            retry.replayed == (phase == "after"),
            "ambiguous receipt had wrong replay outcome",
        )?;
        ensure(
            e.audit()?.spent == 1 && e.audit()?.receipts == 1,
            "retry duplicated the effect",
        )?;
        finish(&format!("process_kill_{phase}_commit"), &mut e)?;
    }
    let mut checkpointed = checkpoint_stress(
        &directory.join("checkpoint.sqlite"),
        if small { 4 } else { 16 },
    )?;
    finish(
        "concurrent_writes_and_500_passive_checkpoints",
        &mut checkpointed,
    )?;
    let race = parallel(executable, &directory.join("race"), 8, 1, 2, 2)?;
    ensure(
        race.executed == 1 && race.denials.get("Budget") == Some(&7),
        "concurrent processes exceeded allowance",
    )?;
    let per_worker = if small { 16 } else { 64 };
    let mut throughput = Vec::new();
    // Equal transaction count; setup and approvals excluded from operation latency.
    for workers in [1, 4] {
        throughput.push(parallel(
            executable,
            &directory.join(format!("throughput-{workers}")),
            workers,
            per_worker * 4 / workers,
            (per_worker * 4) as i64,
            1,
        )?);
    }
    Ok(RuntimeResult {
        sqlite_version: rusqlite::version().into(),
        scenarios,
        race,
        throughput,
    })
}
pub fn unique_test_dir(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/test-evidence")
        .join(format!("{label}-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&dir).expect("test evidence directory");
    dir
}
