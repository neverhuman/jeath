//! Local cooperative sink protocol. No network or external exactly-once claim.
use rusqlite::{Connection, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use std::{
    io::{self, BufRead, Write},
    path::Path,
    time::Duration,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Effect {
    Transfer {
        recipient: String,
        units: u64,
    },
    Release {
        destination: String,
        contents: String,
    },
    Edit {
        document: String,
        expected_revision: u64,
        contents: String,
    },
}
impl Effect {
    pub fn charge(&self) -> u64 {
        match self {
            Self::Transfer { units, .. } => *units,
            _ => 1,
        }
    }
    pub fn required(&self) -> Vec<bool> {
        vec![true, !matches!(self, Self::Transfer { .. })]
    }
    pub fn scoped(&self) -> bool {
        match self {
            Self::Transfer { recipient, units } => recipient == "approved" && *units > 0,
            Self::Release {
                destination,
                contents,
            } => destination == "approved" && contents == "SYNTHETIC DATA: no personal information",
            Self::Edit {
                document, contents, ..
            } => document == "toy.rs" && contents.len() <= 4096,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Proposal {
    pub id: String,
    pub epochs: Vec<u64>,
    pub expires: u64,
    pub effect: Effect,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Permit {
    pub actor: String,
    pub proposal: Proposal,
}
fn same_permit(a: &Permit, b: &Permit) -> Result<bool> {
    Ok(policy_core::same_bytes(
        &serde_json::to_vec(a)?,
        &serde_json::to_vec(b)?,
    ))
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Pending,
    Committed,
    Cancelled,
    Denied,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Record {
    pub permit: Permit,
    pub status: Status,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Approval {
    pub permit: Permit,
    pub grants: Vec<bool>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Broker {
    pub budget: u64,
    pub charged: u64,
    pub epochs: Vec<u64>,
    pub live: Vec<bool>,
    pub approvals: Vec<Approval>,
    pub records: Vec<Record>,
}
impl Broker {
    pub fn new(budget: u64) -> Self {
        Self {
            budget,
            charged: 0,
            epochs: vec![0, 0],
            live: vec![true, true],
            approvals: vec![],
            records: vec![],
        }
    }
    pub fn submit(&mut self, permit: Permit, timestamp: u64) -> Result<serde_json::Value> {
        let p = &permit.proposal;
        if let Some(r) = self.records.iter().find(|r| r.permit.proposal.id == p.id) {
            return Ok(if same_permit(&r.permit, &permit)? {
                serde_json::json!({"status":r.status,"permit":r.permit})
            } else {
                denied("payload_mismatch")
            });
        }
        if p.id.is_empty()
            || p.id.len() > 128
            || permit.actor.is_empty()
            || permit.actor.len() > 128
            || !p.effect.scoped()
        {
            return Ok(denied("scope"));
        }
        let Some(a) = self.approvals.iter().find(|a| a.permit.proposal.id == p.id) else {
            return Ok(denied("missing_grant"));
        };
        let required = p.effect.required();
        if !policy_core::current_grants(&required, &a.grants, &self.live, &self.epochs, &p.epochs) {
            return Ok(denied("withdrawn_or_stale_grant"));
        }
        let bytes = serde_json::to_vec(&permit)?;
        let approved = serde_json::to_vec(&a.permit)?;
        // Revisions are bound in the exact approved payload, then checked against
        // the authoritative object at the sink, where that revision lives.
        if !policy_core::admit(
            self.budget,
            self.charged,
            p.effect.charge(),
            true,
            0,
            0,
            timestamp,
            p.expires,
            0,
            0,
            &bytes,
            &approved,
            &required,
            &a.grants,
        ) {
            return Ok(denied("policy"));
        }
        self.charged = policy_core::reserve(self.budget, self.charged, p.effect.charge())
            .ok_or("reservation")?;
        self.records.push(Record {
            permit: permit.clone(),
            status: Status::Pending,
        });
        Ok(serde_json::json!({"status":"pending","permit":permit}))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Sink {
    pub floors: Vec<u64>,
    pub records: Vec<Record>,
    pub revision: u64,
    pub document: String,
}
impl Default for Sink {
    fn default() -> Self {
        Self {
            floors: vec![0, 0],
            records: vec![],
            revision: 0,
            document: "fn answer() -> u32 { 41 }".into(),
        }
    }
}
pub fn denied(reason: &str) -> serde_json::Value {
    serde_json::json!({"status":"denied","reason":reason})
}
fn issuer(v: &serde_json::Value) -> Result<usize> {
    let i = v["issuer"].as_u64().ok_or("issuer")? as usize;
    if i >= 2 {
        return Err("issuer out of range".into());
    }
    Ok(i)
}
fn text<'a>(v: &'a serde_json::Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .ok_or_else(|| format!("missing {key}").into())
}

/// Each process owns only its own database. The JSONL channel is a trusted
/// control channel; launch_v2.py exposes only submit/status to proposer children.
pub fn serve(path: &Path, role: &str) -> Result<()> {
    if rusqlite::version_number() < 3_051_003 {
        return Err("SQLite >= 3.51.3 required".into());
    }
    let mut db = Connection::open(path)?;
    db.busy_timeout(Duration::from_secs(10))?;
    db.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS state(singleton INTEGER PRIMARY KEY CHECK(singleton=1), body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS effects(sequence INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL, charge INTEGER NOT NULL);")?;
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.len() > 65536 {
            return Err("RPC too large".into());
        }
        let response = (|| -> Result<serde_json::Value> {
            let v: serde_json::Value = serde_json::from_str(&line)?;
            let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let op = text(&v, "op")?;
            let saved: Option<String> = tx.query_row("SELECT body FROM state WHERE singleton=1", [], |r| r.get(0)).optional()?;
            if op == "init" {
                if saved.is_some() { return Ok(denied("already_initialized")); }
                let body = if role == "broker" {
                    let b = v["budget"].as_u64().filter(|x| *x <= i64::MAX as u64).ok_or("budget")?;
                    serde_json::to_string(&Broker::new(b))?
                } else { serde_json::to_string(&Sink::default())? };
                tx.execute("INSERT INTO state VALUES(1,?1)", [body])?;
                tx.commit()?;
                return Ok(serde_json::json!({"status":"initialized"}));
            }
            let saved = saved.ok_or("not initialized")?;
            let response;
            let body;
            if role == "broker" {
                let mut b: Broker = serde_json::from_str(&saved)?;
                response = match op {
                    "approve" => {
                        let permit: Permit = serde_json::from_value(v["permit"].clone())?;
                        let i = issuer(&v)?;
                        if !b.live[i] || permit.proposal.epochs != b.epochs { denied("withdrawn_or_stale") }
                        else if let Some(a) = b.approvals.iter_mut().find(|a| a.permit.proposal.id == permit.proposal.id) {
                            if !same_permit(&a.permit, &permit)? { denied("payload_mismatch") }
                            else { a.grants[i] = true; serde_json::json!({"status":"approved"}) }
                        } else {
                            let mut grants = vec![false, false]; grants[i] = true;
                            b.approvals.push(Approval { permit, grants }); serde_json::json!({"status":"approved"})
                        }
                    }
                    "submit" => {
                        let proposal: Proposal = serde_json::from_value(v["proposal"].clone())?;
                        b.submit(Permit { actor: text(&v, "session_actor")?.into(), proposal }, crate::runtime::now() as u64)?
                    }
                    "status" => {
                        let id = text(&v, "id")?;
                        let actor = text(&v, "session_actor")?;
                        match b.records.iter().find(|r| r.permit.proposal.id == id && r.permit.actor == actor) {
                            Some(r) => serde_json::json!({"status":r.status}), None => denied("unknown_id"),
                        }
                    }
                    "withdraw" => {
                        let i = issuer(&v)?;
                        let old = b.epochs[i];
                        b.epochs[i] = old.checked_add(1).ok_or("epoch exhausted")?;
                        b.live[i] = false;
                        // Approval bits are not restored by renewal.
                        for a in &mut b.approvals { a.grants[i] = false; }
                        serde_json::json!({"status":"admission_revoked","issuer":i,"floor":b.epochs[i],"charged":b.charged,
                            "outstanding":b.records.iter().filter(|r| r.status == Status::Pending && r.permit.proposal.effect.required()[i]).map(|r| r.permit.proposal.effect.charge()).sum::<u64>()})
                    }
                    "renew" => { b.live[issuer(&v)?] = true; serde_json::json!({"status":"renewed"}) }
                    "settle" => {
                        // Trusted sink response only. A timeout is never a certificate.
                        let id = text(&v, "id")?;
                        let r = b.records.iter_mut().find(|r| r.permit.proposal.id == id).ok_or("unknown id")?;
                        let status: Status = serde_json::from_value(v["sink_status"].clone())?;
                        if r.status != Status::Pending || !matches!(status, Status::Committed | Status::Cancelled) { denied("not_authoritative_terminal") }
                        else {
                            b.charged = policy_core::release(b.charged, r.permit.proposal.effect.charge(), status == Status::Cancelled);
                            r.status = status; serde_json::json!({"status":r.status})
                        }
                    }
                    "inspect" => serde_json::to_value(&b)?,
                    _ => denied("unknown_control_operation"),
                };
                body = serde_json::to_string(&b)?;
            } else {
                let mut s: Sink = serde_json::from_str(&saved)?;
                response = match op {
                    "accept" => {
                        let permit: Permit = serde_json::from_value(v["permit"].clone())?;
                        let p = &permit.proposal;
                        if let Some(r) = s.records.iter().find(|r| r.permit.proposal.id == p.id) {
                            if same_permit(&r.permit, &permit)? { serde_json::json!({"status":r.status}) } else { denied("payload_mismatch") }
                        } else if !policy_core::unfenced(&p.effect.required(), &p.epochs, &s.floors) {
                            // A durable cancellation tombstone also covers a
                            // previously issued permit whose first delivery is late.
                            s.records.push(Record { permit, status: Status::Cancelled });
                            serde_json::json!({"status":"cancelled"})
                        }
                        else { s.records.push(Record { permit, status: Status::Pending }); serde_json::json!({"status":"pending"}) }
                    }
                    "complete" => {
                        let id = text(&v, "id")?;
                        let r = s.records.iter_mut().find(|r| r.permit.proposal.id == id).ok_or("unknown id")?;
                        if r.status == Status::Pending {
                            let p = &r.permit.proposal;
                            let revision = match &p.effect { Effect::Edit { expected_revision, .. } => policy_core::revision_matches(s.revision, *expected_revision), _ => true };
                            if !policy_core::unfenced(&p.effect.required(), &p.epochs, &s.floors) || !revision {
                                r.status = Status::Cancelled;
                            } else {
                                if let Effect::Edit { contents, .. } = &p.effect {
                                    s.revision = s.revision.checked_add(1).ok_or("revision exhausted")?;
                                    s.document = contents.clone();
                                }
                                tx.execute("INSERT INTO effects(id,payload,charge) VALUES(?1,?2,?3)", params![id, serde_json::to_string(&r.permit)?, i64::try_from(p.effect.charge())?])?;
                                r.status = Status::Committed;
                            }
                        }
                        serde_json::json!({"status":r.status})
                    }
                    "fence" => {
                        let i = issuer(&v)?;
                        let floor = v["floor"].as_u64().ok_or("floor")?;
                        s.floors[i] = s.floors[i].max(floor);
                        for r in &mut s.records {
                            if r.status == Status::Pending && !policy_core::unfenced(&r.permit.proposal.effect.required(), &r.permit.proposal.epochs, &s.floors) { r.status = Status::Cancelled; }
                        }
                        serde_json::json!({"status":"fenced","issuer":i,"floor":s.floors[i]})
                    }
                    "status" => {
                        let id = text(&v, "id")?;
                        match s.records.iter().find(|r| r.permit.proposal.id == id) {
                            Some(r) => serde_json::json!({"status":r.status}),
                            // Absence alone is not authoritative: an earlier send may arrive later.
                            None => serde_json::json!({"status":"unknown"}),
                        }
                    }
                    "inspect" => serde_json::to_value(&s)?,
                    _ => denied("unknown_sink_operation"),
                };
                body = serde_json::to_string(&s)?;
            }
            tx.execute("UPDATE state SET body=?1 WHERE singleton=1", [body])?;
            if let Some(delay) = v["delay_before_commit_ms"].as_u64() {
                std::thread::sleep(Duration::from_millis(delay.min(1000)));
            }
            if v["crash"].as_str() == Some("before_commit") { std::process::exit(86); }
            tx.commit()?;
            if v["crash"].as_str() == Some("after_commit") { std::process::exit(87); }
            Ok(response)
        })().unwrap_or_else(|e| denied(&e.to_string()));
        serde_json::to_writer(&mut out, &response)?;
        writeln!(out)?;
        out.flush()?;
    }
    Ok(())
}
use rusqlite::OptionalExtension;
