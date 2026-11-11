use authority_lab::{
    campaigns, coverage, distributed_sinks, graph, model, monitors,
    runtime::{Executor, Phase, Request},
    runtime_experiments::{self, write_json},
    selection, stats, v3_core_models, v3_coverage, v3_graphs, v3_monitors, v3_selection,
    v4_neural_rl, v4_real_graphs,
};
use std::{fs, path::Path, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn main() {
    if let Err(error) = run() {
        eprintln!("authority-lab: {error}");
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let usage = "usage: authority-lab run OUTPUT [--quick] [--seed INTEGER]\n       authority-lab run-v3 OUTPUT [--quick] [--profile smoke|publication] [--seed INTEGER]\n       authority-lab model OUTPUT.json\n       authority-lab campaigns INPUT.jsonl OUTPUT\n       authority-lab execute DATABASE REQUEST.json\n       authority-lab inspect DATABASE\n       authority-lab run-v2 OUTPUT [--profile smoke|publication]\n       authority-lab analyze-v2 OUTPUT [--source]\n       authority-lab broker-v2|sink-v2 DATABASE (trusted JSONL only)\n       authority-lab version-v2";
    let command = args.first().map(String::as_str).unwrap_or("help");
    let arg =
        |n: usize| -> Result<&str> { args.get(n).map(String::as_str).ok_or_else(|| usage.into()) };
    match command {
        "version-v2" => println!(
            "{}",
            serde_json::json!({"sqlite":rusqlite::version(),
            "landlock_abi":authority_lab::isolation_v2::landlock_abi(),"protocol":"local-reservation-fenced-sink-v2"})
        ),
        "run-v2" | "analyze-v2" => {
            let script = if command == "run-v2" {
                "run_v2.py"
            } else {
                "analyze_v2.py"
            };
            let status = std::process::Command::new("python3")
                .arg(
                    Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("scripts")
                        .join(script),
                )
                .args(&args[1..])
                .status()?;
            if !status.success() {
                return Err(format!("{command} failed: {status}").into());
            }
        }
        "confine-v2" => {
            use std::os::unix::process::CommandExt;
            let exe = fs::canonicalize(arg(1)?)?;
            authority_lab::isolation_v2::confine(&exe)?;
            // No inherited non-stdio descriptors may reach the proposer.
            #[cfg(target_os = "linux")]
            if unsafe { libc::syscall(libc::SYS_close_range, 3u32, u32::MAX, 0) } != 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            let error = std::process::Command::new(exe)
                .args(&args[2..])
                .env_clear()
                .env("PATH", "/usr/bin:/bin")
                .env("PYTHONDONTWRITEBYTECODE", "1")
                .exec();
            return Err(error.into());
        }
        "probe-v2" => println!(
            "{}",
            authority_lab::isolation_v2::probe(Path::new(arg(1)?), arg(2)?.parse()?)
        ),
        "broker-v2" | "sink-v2" => {
            if args.len() != 2 {
                return Err(usage.into());
            }
            authority_lab::protocol_v2::serve(
                Path::new(arg(1)?),
                if command == "broker-v2" {
                    "broker"
                } else {
                    "sink"
                },
            )?;
        }
        "campaigns" => {
            if args.len() != 3 {
                return Err(
                    "usage: authority-lab campaigns INPUT.jsonl NEW_OUTPUT_DIRECTORY".into(),
                );
            }
            let results = campaigns::evaluate(Path::new(arg(1)?), Path::new(arg(2)?))?;
            println!(
                "evaluated {} campaigns; typed proposal scores are distinct from executed effects",
                results.len()
            );
        }
        "run" => {
            let output = Path::new(arg(1)?);
            let mut quick = false;
            let mut seed = 20261110u64;
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--quick" => quick = true,
                    "--seed" => {
                        i += 1;
                        seed = arg(i)?.parse()?;
                    }
                    _ => return Err(usage.into()),
                }
                i += 1;
            }
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::create_dir(output)?;
            let executable = std::env::current_exe()?;
            let start = Instant::now();
            write_json(
                &output.join("metadata.json"),
                &serde_json::json!({"schema_version":1,"seed":seed,
                "quick":quick,"package_version":env!("CARGO_PKG_VERSION"),"os":std::env::consts::OS,"arch":std::env::consts::ARCH,
                "rng":"splitmix64-box-muller-v1","hostname":std::env::var("HOSTNAME").ok(),"gpu_used":false}),
            )?;
            eprintln!("checking finite-state closure and mutations");
            write_json(&output.join("model.json"), &model::suite())?;
            eprintln!("testing SQLite processes, crash recovery, and contention");
            write_json(
                &output.join("runtime.json"),
                &runtime_experiments::run(&executable, &output.join("runtime"), quick)?,
            )?;
            write_json(&output.join("graphs.json"), &graph::suite())?;
            write_json(
                &output.join("coverage.json"),
                &coverage::run(seed, if quick { 200 } else { 2000 }),
            )?;
            eprintln!("training small synthetic monitors");
            write_json(
                &output.join("monitors.json"),
                &monitors::run(
                    seed,
                    if quick {
                        monitors::Config::quick()
                    } else {
                        monitors::Config::paper()
                    },
                ),
            )?;
            eprintln!("running reward-selection illustrations");
            write_json(
                &output.join("selection.json"),
                &selection::run(
                    seed,
                    if quick { 4 } else { 8 },
                    if quick { 100 } else { 400 },
                    if quick { 25 } else { 60 },
                ),
            )?;
            write_json(
                &output.join("statistics.json"),
                &serde_json::json!({"zero_of_3000_upper95":stats::binomial_upper(0,3000,0.05),
                "zero_failure_trials_for_one_in_million_95":2_995_731,"common_cause_joint":0.01+0.99*0.1f64.powi(4),
                "common_cause_product_of_marginals":0.109f64.powi(4),"unit":"independent Bernoulli episodes under a fixed protocol"}),
            )?;
            write_json(
                &output.join("complete.json"),
                &serde_json::json!({"complete":true,"elapsed_seconds":start.elapsed().as_secs_f64()}),
            )?;
            println!(
                "results: {} ({:.2}s)",
                output.display(),
                start.elapsed().as_secs_f64()
            );
        }
        "run-v3" => {
            let output = Path::new(arg(1)?);
            let mut quick = false;
            let mut seed = 20261110u64;
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--quick" => quick = true,
                    "--profile" => {
                        i += 1;
                        quick = match arg(i)? {
                            "smoke" => true,
                            "publication" => false,
                            _ => return Err(usage.into()),
                        };
                    }
                    "--seed" => {
                        i += 1;
                        seed = arg(i)?.parse()?;
                    }
                    _ => return Err(usage.into()),
                }
                i += 1;
            }
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::create_dir(output)?;
            let start = Instant::now();
            write_json(
                &output.join("metadata.json"),
                &serde_json::json!({
                    "schema_version": 3,
                    "seed": seed,
                    "quick": quick,
                    "profile": if quick { "smoke" } else { "publication" },
                    "package_version": env!("CARGO_PKG_VERSION"),
                    "os": std::env::consts::OS,
                    "arch": std::env::consts::ARCH,
                    "rng": "splitmix64-v3",
                    "hostname": std::env::var("HOSTNAME").ok(),
                    "gpu_used": false,
                }),
            )?;
            write_json(
                &output.join("uncertainty.json"),
                &v3_core_models::uncertainty_suite(seed, quick),
            )?;
            write_json(
                &output.join("composition.json"),
                &v3_core_models::composition_suite(),
            )?;
            write_json(
                &output.join("risk_horizon.json"),
                &v3_core_models::risk_horizon_suite(seed, quick),
            )?;
            let jp = v3_core_models::jarvis_and_personas_suite(seed, quick);
            write_json(&output.join("jarvis_ledger.json"), &jp.jarvis_ledger)?;
            write_json(&output.join("personas.json"), &jp.personas)?;
            write_json(
                &output.join("weighted_cuts.json"),
                &v3_graphs::weighted_cuts_suite(),
            )?;
            write_json(
                &output.join("interdiction.json"),
                &v3_graphs::interdiction_suite(seed, quick),
            )?;
            write_json(
                &output.join("coverage_v3.json"),
                &v3_coverage::coverage_v3_suite(seed, quick),
            )?;
            write_json(
                &output.join("audit_power.json"),
                &v3_coverage::audit_power_suite(),
            )?;
            write_json(&output.join("copula.json"), &v3_monitors::copula_suite())?;
            write_json(
                &output.join("monitors_v3.json"),
                &v3_monitors::monitors_v3_suite(seed, quick),
            )?;
            write_json(
                &output.join("assessment_variance.json"),
                &v3_monitors::assessment_variance_suite(seed, quick),
            )?;
            write_json(
                &output.join("selection_v3.json"),
                &v3_selection::selection_v3_suite(seed, quick),
            )?;
            write_json(
                &output.join("real_tool_cutsets.json"),
                &v4_real_graphs::real_tool_cutsets_suite(seed, quick),
            )?;
            write_json(
                &output.join("neural_post_training.json"),
                &v4_neural_rl::neural_post_training_suite(seed, quick),
            )?;
            write_json(
                &output.join("crypto_and_jepsen.json"),
                &distributed_sinks::crypto_and_jepsen_suite(seed, quick),
            )?;
            write_json(
                &output.join("complete.json"),
                &serde_json::json!({
                    "complete": true,
                    "schema_version": 3,
                    "elapsed_seconds": start.elapsed().as_secs_f64(),
                }),
            )?;
            println!(
                "v3 results: {} ({:.2}s)",
                output.display(),
                start.elapsed().as_secs_f64()
            );
        }
        "model" => write_json(Path::new(arg(1)?), &model::suite())?,
        "execute" => {
            if args.len() != 3 {
                return Err(usage.into());
            }
            let request: Request = serde_json::from_reader(fs::File::open(arg(2)?)?)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&Executor::open(arg(1)?)?.execute(&request)?)?
            );
        }
        "inspect" => println!(
            "{}",
            serde_json::to_string_pretty(&Executor::open(arg(1)?)?.audit()?)?
        ),
        "worker" => {
            let start = arg(3)?;
            let phase = match args.get(4).map(String::as_str) {
                None => None,
                Some("before") => Some(Phase::BeforeCommit),
                Some("after") => Some(Phase::AfterCommit),
                _ => return Err(usage.into()),
            };
            runtime_experiments::worker(
                Path::new(arg(1)?),
                Path::new(arg(2)?),
                if start == "-" {
                    None
                } else {
                    Some(Path::new(start))
                },
                phase,
            )?;
        }
        "help" | "--help" | "-h" => println!("{usage}"),
        _ => return Err(usage.into()),
    }
    Ok(())
}
