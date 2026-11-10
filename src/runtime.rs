//! Trusted local executor. All protected effects are inside ONE SQLite database.
//! This is not an OS sandbox, authentication service, or external-payment protocol.
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    path::Path,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    Spend {
        destination: String,
    },
    Release {
        document: String,
        destination: String,
        purpose: String,
    },
    Edit {
        document: String,
        expected_revision: i64,
        body: String,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub id: String,
    pub actor: String,
    pub amount: i64,
    pub epoch: i64,
    pub expires_at: i64,
    pub operation: Operation,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Receipt {
    pub sequence: i64,
    pub request: Request,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Execution {
    pub receipt: Receipt,
    pub replayed: bool,
}
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Denial {
    InvalidRequest,
    Scope,
    Expired,
    Revoked,
    StaleEpoch,
    Budget,
    MissingGrant,
    PayloadMismatch,
    Unprepared,
    RevisionConflict,
    AlreadyInitialized,
}
#[derive(Debug)]
pub enum Error {
    Denied(Denial),
    Sql(rusqlite::Error),
    Json(serde_json::Error),
    UnsupportedSqlite(i32),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}
impl From<rusqlite::Error> for Error {
    fn from(e: rusqlite::Error) -> Self {
        Self::Sql(e)
    }
}
impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}
pub type Result<T> = std::result::Result<T, Error>;
fn deny<T>(d: Denial) -> Result<T> {
    Err(Error::Denied(d))
}
pub fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_secs() as i64
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    BeforeCommit,
    AfterCommit,
}
pub struct Executor {
    connection: Connection,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Audit {
    pub budget: i64,
    pub spent: i64,
    pub epoch: i64,
    pub live: bool,
    pub receipts: i64,
    pub account_units: i64,
    pub deliveries: i64,
    pub edits: i64,
    pub observed_units: i64,
    pub receipt_units: i64,
    pub integrity: String,
}
impl Audit {
    pub fn consistent(&self) -> bool {
        self.integrity == "ok"
            && self.spent == self.observed_units
            && self.spent == self.receipt_units
            && self.spent <= self.budget
            && self.spent >= 0
    }
}
impl Executor {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        // Earlier WAL implementations can corrupt data under concurrent reset /
        // checkpoint. Require the upstream fixed release (not an assumed backport).
        if rusqlite::version_number() < 3_051_003 {
            return Err(Error::UnsupportedSqlite(rusqlite::version_number()));
        }
        let c = Connection::open(path)?;
        c.busy_timeout(Duration::from_secs(10))?;
        c.execute_batch("PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL;")?;
        Ok(Self { connection: c })
    }
    /// Trusted provisioning only. Existing policy state can never be reset here.
    pub fn initialize(&mut self, budget: i64, expires_at: i64) -> Result<()> {
        if budget <= 0 || expires_at <= now() {
            return deny(Denial::InvalidRequest);
        }
        self.connection.pragma_update(None, "journal_mode", "WAL")?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute_batch("
          CREATE TABLE IF NOT EXISTS policy (
            singleton INTEGER PRIMARY KEY CHECK(singleton=1),
            budget INTEGER NOT NULL CHECK(budget>0), spent INTEGER NOT NULL CHECK(spent>=0),
            epoch INTEGER NOT NULL CHECK(epoch>=0), live INTEGER NOT NULL CHECK(live IN(0,1)),
            expires_at INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS prepared (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS grants (
            id TEXT NOT NULL REFERENCES prepared(id), issuer TEXT NOT NULL,
            payload TEXT NOT NULL, PRIMARY KEY(id,issuer));
          CREATE TABLE IF NOT EXISTS receipts (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL REFERENCES prepared(id),
            payload TEXT NOT NULL, units INTEGER NOT NULL CHECK(units>0));
          CREATE TABLE IF NOT EXISTS accounts (destination TEXT PRIMARY KEY, units INTEGER NOT NULL CHECK(units>=0));
          CREATE TABLE IF NOT EXISTS objects (name TEXT PRIMARY KEY, body TEXT NOT NULL, revision INTEGER NOT NULL CHECK(revision>=0));
          CREATE TABLE IF NOT EXISTS deliveries (
            id TEXT PRIMARY KEY REFERENCES receipts(id), actor TEXT NOT NULL, body TEXT NOT NULL);
        ")?;
        if tx.query_row("SELECT COUNT(*) FROM policy", [], |r| r.get::<_, i64>(0))? != 0 {
            return deny(Denial::AlreadyInitialized);
        }
        tx.execute(
            "INSERT INTO policy VALUES(1,?1,0,0,1,?2)",
            params![budget, expires_at],
        )?;
        tx.execute("INSERT INTO accounts VALUES('approved',0)", [])?;
        tx.execute(
            "INSERT INTO objects VALUES('synthetic','SYNTHETIC DATA: no personal information',0)",
            [],
        )?;
        tx.execute(
            "INSERT INTO objects VALUES('toy.rs','fn answer() -> u32 { 41 }',0)",
            [],
        )?;
        tx.commit()?;
        Ok(())
    }
    pub fn prepare(&self, r: &Request) -> Result<()> {
        validate(r)?;
        let payload = serde_json::to_string(r)?;
        self.connection.execute(
            "INSERT INTO prepared VALUES(?1,?2) ON CONFLICT(id) DO NOTHING",
            params![r.id, payload],
        )?;
        let saved: String = self.connection.query_row(
            "SELECT payload FROM prepared WHERE id=?1",
            [&r.id],
            |row| row.get(0),
        )?;
        if saved != payload {
            return deny(Denial::PayloadMismatch);
        }
        Ok(())
    }
    /// Trusted grant owner interface. The proposer MUST NOT receive this capability.
    pub fn approve(&self, issuer: &str, r: &Request) -> Result<()> {
        if issuer != "operator" && issuer != "steward" {
            return deny(Denial::Scope);
        }
        let payload = serde_json::to_string(r)?;
        let stored: Option<String> = self
            .connection
            .query_row("SELECT payload FROM prepared WHERE id=?1", [&r.id], |row| {
                row.get(0)
            })
            .optional()?;
        if stored.as_deref() != Some(&payload) {
            return deny(Denial::PayloadMismatch);
        }
        self.connection.execute("INSERT INTO grants VALUES(?1,?2,?3) ON CONFLICT(id,issuer) DO UPDATE SET payload=excluded.payload",params![r.id,issuer,payload])?;
        Ok(())
    }
    pub fn withdraw(&self, issuer: &str, id: &str) -> Result<()> {
        if issuer != "operator" && issuer != "steward" {
            return deny(Denial::Scope);
        }
        self.connection.execute(
            "DELETE FROM grants WHERE id=?1 AND issuer=?2",
            params![id, issuer],
        )?;
        Ok(())
    }
    /// Revoke AND invalidate pending epoch approvals; renewal cannot resurrect them.
    pub fn revoke(&mut self) -> Result<()> {
        self.connection.execute(
            "UPDATE policy SET live=0,epoch=epoch+1 WHERE singleton=1",
            [],
        )?;
        Ok(())
    }
    pub fn renew(&mut self) -> Result<()> {
        self.connection
            .execute("UPDATE policy SET live=1 WHERE singleton=1", [])?;
        Ok(())
    }
    pub fn checkpoint(&self) -> Result<(i64, i64, i64)> {
        Ok(self
            .connection
            .query_row("PRAGMA wal_checkpoint(PASSIVE)", [], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?)
    }
    pub fn execute(&mut self, r: &Request) -> Result<Execution> {
        self.execute_observed(r, |_| {})
    }
    /// Instrumentation for process-death experiments, called only by trusted code.
    /// Hooks cannot turn a rejected request into an effect.
    pub fn execute_observed(
        &mut self,
        r: &Request,
        mut observe: impl FnMut(Phase),
    ) -> Result<Execution> {
        validate(r)?;
        let payload = serde_json::to_string(r)?;
        let tx = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let previous: Option<(i64, String)> = tx
            .query_row(
                "SELECT sequence,payload FROM receipts WHERE id=?1",
                [&r.id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if let Some((sequence, saved)) = previous {
            if saved != payload {
                return deny(Denial::PayloadMismatch);
            }
            // A retry retrieves a non-content receipt; it executes no new release.
            return Ok(Execution {
                receipt: Receipt {
                    sequence,
                    request: r.clone(),
                },
                replayed: true,
            });
        }
        let saved: Option<String> = tx
            .query_row("SELECT payload FROM prepared WHERE id=?1", [&r.id], |row| {
                row.get(0)
            })
            .optional()?;
        match saved {
            None => return deny(Denial::Unprepared),
            Some(s) if s != payload => return deny(Denial::PayloadMismatch),
            _ => {}
        }
        let (budget, spent, epoch, live, expires): (i64, i64, i64, bool, i64) = tx.query_row(
            "SELECT budget,spent,epoch,live,expires_at FROM policy WHERE singleton=1",
            [],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            },
        )?;
        if !live {
            return deny(Denial::Revoked);
        }
        if epoch != r.epoch {
            return deny(Denial::StaleEpoch);
        }
        let timestamp = now();
        if timestamp >= expires || timestamp >= r.expires_at {
            return deny(Denial::Expired);
        }
        if r.amount > budget - spent {
            return deny(Denial::Budget);
        }
        let issuers: &[&str] = match r.operation {
            Operation::Spend { .. } => &["operator"],
            _ => &["operator", "steward"],
        };
        for issuer in issuers {
            let granted: Option<String> = tx
                .query_row(
                    "SELECT payload FROM grants WHERE id=?1 AND issuer=?2",
                    params![r.id, issuer],
                    |row| row.get(0),
                )
                .optional()?;
            if granted.as_deref() != Some(&payload) {
                return deny(Denial::MissingGrant);
            }
        }
        tx.execute(
            "INSERT INTO receipts(id,payload,units) VALUES(?1,?2,?3)",
            params![r.id, payload, r.amount],
        )?;
        let sequence = tx.last_insert_rowid();
        match &r.operation {
            Operation::Spend { destination } => {
                if destination != "approved" {
                    return deny(Denial::Scope);
                }
                tx.execute(
                    "UPDATE accounts SET units=units+?1 WHERE destination=?2",
                    params![r.amount, destination],
                )?;
            }
            Operation::Release {
                document,
                destination,
                purpose,
            } => {
                if document != "synthetic"
                    || destination != "approved"
                    || purpose != "analysis"
                    || r.amount != 1
                {
                    return deny(Denial::Scope);
                }
                tx.execute(
                    "INSERT INTO deliveries SELECT ?1,?2,body FROM objects WHERE name=?3",
                    params![r.id, r.actor, document],
                )?;
            }
            Operation::Edit {
                document,
                expected_revision,
                body,
            } => {
                if document != "toy.rs" || r.amount != 1 || body.len() > 4096 {
                    return deny(Denial::Scope);
                }
                let changed = tx.execute(
                    "UPDATE objects SET body=?1,revision=revision+1 WHERE name=?2 AND revision=?3",
                    params![body, document, expected_revision],
                )?;
                if changed != 1 {
                    return deny(Denial::RevisionConflict);
                }
            }
        }
        tx.execute(
            "UPDATE policy SET spent=spent+?1 WHERE singleton=1",
            [r.amount],
        )?;
        // Expiry is checked immediately before commit too. It is not a hard
        // real-time guarantee: scheduler/fsync latency remains outside the model.
        observe(Phase::BeforeCommit);
        if now() >= expires || now() >= r.expires_at {
            return deny(Denial::Expired);
        }
        tx.commit()?;
        observe(Phase::AfterCommit);
        Ok(Execution {
            receipt: Receipt {
                sequence,
                request: r.clone(),
            },
            replayed: false,
        })
    }
    /// Read-only trusted audit, using one snapshot across all tables.
    pub fn audit(&mut self) -> Result<Audit> {
        let tx = self.connection.transaction()?;
        let (budget, spent, epoch, live) = tx.query_row(
            "SELECT budget,spent,epoch,live FROM policy WHERE singleton=1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )?;
        let count = |sql: &str| -> std::result::Result<i64, rusqlite::Error> {
            tx.query_row(sql, [], |r| r.get(0))
        };
        let receipts = count("SELECT COUNT(*) FROM receipts")?;
        let account_units = count("SELECT COALESCE(SUM(units),0) FROM accounts")?;
        let deliveries = count("SELECT COUNT(*) FROM deliveries")?;
        let edits = count("SELECT COALESCE(SUM(revision),0) FROM objects")?;
        let receipt_units = count("SELECT COALESCE(SUM(units),0) FROM receipts")?;
        let integrity = tx.query_row("PRAGMA integrity_check", [], |r| r.get(0))?;
        Ok(Audit {
            budget,
            spent,
            epoch,
            live,
            receipts,
            account_units,
            deliveries,
            edits,
            observed_units: account_units + deliveries + edits,
            receipt_units,
            integrity,
        })
    }
}
fn validate(r: &Request) -> Result<()> {
    if r.id.is_empty()
        || r.id.len() > 128
        || r.actor.is_empty()
        || r.actor.len() > 128
        || r.amount <= 0
        || r.epoch < 0
        || r.expires_at <= 0
    {
        return deny(Denial::InvalidRequest);
    }
    Ok(())
}
