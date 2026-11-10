use authority_lab::{
    campaigns, coverage, graph, model, monitors,
    runtime::{Executor, Phase, Request},
    runtime_experiments::{self, write_json},
    selection, stats,
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
    let usage = "usage: authority-lab run OUTPUT [--quick] [--seed INTEGER]\n       authority-lab model OUTPUT.json\n       authority-lab campaigns INPUT.jsonl OUTPUT\n       authority-lab execute DATABASE REQUEST.json\n       authority-lab inspect DATABASE";
    let command = args.first().map(String::as_str).unwrap_or("help");
    let arg =
        |n: usize| -> Result<&str> { args.get(n).map(String::as_str).ok_or_else(|| usage.into()) };
    match command {
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
