//! Trusted provisioning demonstration; never give this process to a proposer.
use authority_lab::runtime::{Executor, Operation, Request, now};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: cargo run --example transaction -- NEW_DATABASE_PATH")?;
    let mut executor = Executor::open(path)?;
    executor.initialize(2, now() + 3600)?;
    let request = Request {
        id: "example-release".into(),
        actor: "assistant".into(),
        amount: 1,
        epoch: 0,
        expires_at: now() + 60,
        operation: Operation::Release {
            document: "synthetic".into(),
            destination: "approved".into(),
            purpose: "analysis".into(),
        },
    };
    executor.prepare(&request)?;
    executor.approve("operator", &request)?;
    executor.approve("steward", &request)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&executor.execute(&request)?)?
    );
    println!("{}", serde_json::to_string_pretty(&executor.audit()?)?);
    Ok(())
}
