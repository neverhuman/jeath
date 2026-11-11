//! Local cooperative sink protocol. No network or external exactly-once claim.
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    io::{self, Write},
    path::Path,
    time::Duration,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub const MAX_RPC_BYTES: usize = 65_536;

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
        self.required_for(2)
    }
    pub fn required_for(&self, issuers: usize) -> Vec<bool> {
        let mut req = vec![false; issuers];
        if issuers == 0 {
            return req;
        }
        match self {
            Self::Transfer { .. } => {
                req[0] = true;
            }
            Self::Edit { .. } => {
                req[0] = true;
                if issuers > 1 {
                    req[1] = true;
                }
            }
            Self::Release { .. } => {
                req.fill(true);
            }
        }
        req
    }
    pub fn kind_index(&self) -> usize {
        match self {
            Self::Transfer { .. } => 0,
            Self::Release { .. } => 1,
            Self::Edit { .. } => 2,
        }
    }
    pub fn scope_vec(&self) -> Vec<bool> {
        let mut s = vec![false; 3];
        s[self.kind_index()] = true;
        s
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
    pub fn to_exec_fields(&self) -> (u8, u64, u64, Vec<u8>, Vec<u8>) {
        match self {
            Self::Transfer { recipient, units } => (
                policy_core::EFFECT_TRANSFER,
                *units,
                0,
                recipient.as_bytes().to_vec(),
                Vec::new(),
            ),
            Self::Release {
                destination,
                contents,
            } => (
                policy_core::EFFECT_RELEASE,
                1,
                0,
                destination.as_bytes().to_vec(),
                contents.as_bytes().to_vec(),
            ),
            Self::Edit {
                document,
                expected_revision,
                contents,
            } => (
                policy_core::EFFECT_EDIT,
                1,
                *expected_revision,
                document.as_bytes().to_vec(),
                contents.as_bytes().to_vec(),
            ),
        }
    }
}

fn min_required_for_scope(scope: &[bool], issuers: usize) -> Vec<bool> {
    let mut req = vec![false; issuers];
    if issuers == 0 {
        return req;
    }
    if scope.get(1) == Some(&true) {
        req.fill(true);
    } else if scope.get(2) == Some(&true) {
        req[0] = true;
        if issuers > 1 {
            req[1] = true;
        }
    } else if scope.first() == Some(&true) {
        req[0] = true;
    }
    req
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
impl Permit {
    pub fn to_exec_wire(&self) -> policy_core::ExecPermitWire {
        let (effect_kind, amount, expected_rev, target, contents) =
            self.proposal.effect.to_exec_fields();
        policy_core::ExecPermitWire {
            id: self.proposal.id.as_bytes().to_vec(),
            actor: self.actor.as_bytes().to_vec(),
            effect_kind,
            amount,
            expires: self.proposal.expires,
            expected_rev,
            target,
            contents,
            epochs: self.proposal.epochs.clone(),
            required: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DispatchedPermit {
    pub actor: String,
    pub proposal: Proposal,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required: Vec<bool>,
}
impl From<Permit> for DispatchedPermit {
    fn from(p: Permit) -> Self {
        Self {
            actor: p.actor,
            proposal: p.proposal,
            required: vec![],
        }
    }
}
impl DispatchedPermit {
    pub fn base_permit(&self) -> Permit {
        Permit {
            actor: self.actor.clone(),
            proposal: self.proposal.clone(),
        }
    }
    pub fn to_exec_wire(&self) -> policy_core::ExecPermitWire {
        let mut wire = self.base_permit().to_exec_wire();
        wire.required = self.required.clone();
        wire
    }
    pub fn to_exec_wire_for_issuers(&self, issuers: usize) -> Result<policy_core::ExecPermitWire> {
        let mut wire = self.base_permit().to_exec_wire();
        wire.required = self.effective_required(issuers)?;
        Ok(wire)
    }
    pub fn effective_required(&self, issuers: usize) -> Result<Vec<bool>> {
        let base = self.proposal.effect.required_for(issuers);
        if self.required.is_empty() {
            return Ok(base);
        }
        if self.required.len() != issuers {
            return Err("permit required length mismatch".into());
        }
        for (&b, &r) in base.iter().zip(&self.required) {
            if b && !r {
                return Err("permit cannot weaken effect required issuers".into());
            }
        }
        Ok(self.required.clone())
    }
    pub fn normalized(mut self, issuers: usize) -> Result<(Self, Vec<bool>)> {
        let base = self.proposal.effect.required_for(issuers);
        let eff = self.effective_required(issuers)?;
        self.required = if eff == base { vec![] } else { eff.clone() };
        Ok((self, eff))
    }
}

pub fn same_permit(a: &Permit, b: &Permit) -> Result<bool> {
    Ok(policy_core::same_permit_canonical(
        &a.to_exec_wire(),
        &b.to_exec_wire(),
    ))
}
pub fn same_dispatched_permit(a: &DispatchedPermit, b: &DispatchedPermit) -> Result<bool> {
    Ok(policy_core::same_permit_canonical(
        &a.to_exec_wire(),
        &b.to_exec_wire(),
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
    pub permit: DispatchedPermit,
    pub status: Status,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Approval {
    pub permit: Permit,
    pub grants: Vec<bool>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Delegation {
    pub id: String,
    pub parent: Option<String>,
    pub permit: Option<Permit>,
    pub budget: u64,
    pub charged: u64,
    pub expires: u64,
    pub scope: Vec<bool>,
    pub required: Vec<bool>,
    pub grants: Vec<bool>,
    pub epochs: Vec<u64>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Broker {
    pub budget: u64,
    pub charged: u64,
    pub epochs: Vec<u64>,
    pub live: Vec<bool>,
    pub approvals: Vec<Approval>,
    pub records: Vec<Record>,
    #[serde(default)]
    pub delegations: Vec<Delegation>,
}
impl Broker {
    pub fn new(budget: u64) -> Self {
        Self::with_issuers(budget, 2).expect("valid default issuer count")
    }
    pub fn with_issuers(budget: u64, issuers: usize) -> Result<Self> {
        if issuers == 0 || issuers > 64 {
            return Err("issuers out of range".into());
        }
        Ok(Self {
            budget,
            charged: 0,
            epochs: vec![0; issuers],
            live: vec![true; issuers],
            approvals: vec![],
            records: vec![],
            delegations: vec![],
        })
    }
    pub fn issuers(&self) -> usize {
        self.epochs.len()
    }
    fn check_issuer(&self, i: usize) -> Result<usize> {
        if i >= self.epochs.len() {
            return Err("issuer out of range".into());
        }
        Ok(i)
    }
    fn delegation_chain(&self, start_idx: usize) -> Result<(Vec<usize>, Option<usize>)> {
        let mut chain = vec![start_idx];
        let mut cur = start_idx;
        let max_steps = self.delegations.len() + 1;
        for _ in 0..max_steps {
            let Some(parent_id) = &self.delegations[cur].parent else {
                return Ok((chain, None));
            };
            if let Some(p_idx) = self.delegations.iter().position(|d| &d.id == parent_id) {
                if chain.contains(&p_idx) {
                    return Err("delegation cycle".into());
                }
                chain.push(p_idx);
                cur = p_idx;
            } else if let Some(a_idx) = self
                .approvals
                .iter()
                .position(|a| &a.permit.proposal.id == parent_id)
            {
                return Ok((chain, Some(a_idx)));
            } else {
                return Err("unknown delegation ancestor".into());
            }
        }
        Err("delegation depth exceeded".into())
    }
    fn approval_delegated_charged(&self, approval_id: &str) -> u64 {
        self.delegations
            .iter()
            .filter(|d| d.parent.as_deref() == Some(approval_id))
            .map(|d| d.charged)
            .sum()
    }
    pub fn approve(&mut self, permit: Permit, issuer: usize) -> Result<serde_json::Value> {
        let i = self.check_issuer(issuer)?;
        if !self.live[i] || permit.proposal.epochs != self.epochs {
            return Ok(denied("withdrawn_or_stale"));
        }
        if self.delegations.iter().any(|d| {
            d.id == permit.proposal.id
                || d.permit
                    .as_ref()
                    .is_some_and(|dp| dp.proposal.id == permit.proposal.id)
        }) {
            return Ok(denied("id_conflict"));
        }
        if let Some(a) = self
            .approvals
            .iter_mut()
            .find(|a| a.permit.proposal.id == permit.proposal.id)
        {
            if !same_permit(&a.permit, &permit)? {
                Ok(denied("payload_mismatch"))
            } else {
                a.grants[i] = true;
                Ok(serde_json::json!({"status": "approved"}))
            }
        } else {
            let mut grants = vec![false; self.issuers()];
            grants[i] = true;
            self.approvals.push(Approval { permit, grants });
            Ok(serde_json::json!({"status": "approved"}))
        }
    }
    pub fn delegate(&mut self, v: &serde_json::Value) -> Result<serde_json::Value> {
        let n = self.issuers();
        let permit_opt: Option<Permit> = if v.get("permit").is_some_and(|x| !x.is_null()) {
            Some(serde_json::from_value(v["permit"].clone())?)
        } else {
            None
        };
        let id = if let Some(ref p) = permit_opt {
            if let Some(explicit_id) = v.get("id").and_then(|x| x.as_str()) {
                if explicit_id != p.proposal.id {
                    return Ok(denied("id_mismatch"));
                }
            }
            p.proposal.id.clone()
        } else {
            text(v, "id")?.to_string()
        };
        if id.is_empty() || id.len() > 128 {
            return Ok(denied("scope"));
        }
        if let Some(ref p) = permit_opt {
            if p.actor.is_empty() || p.actor.len() > 128 || !p.proposal.effect.scoped() {
                return Ok(denied("scope"));
            }
        }
        if self.approvals.iter().any(|a| a.permit.proposal.id == id)
            || self
                .delegations
                .iter()
                .any(|d| d.id == id || d.permit.as_ref().is_some_and(|dp| dp.proposal.id == id))
        {
            return Ok(denied("duplicate_delegation"));
        }

        let child_budget = match (
            v.get("budget").and_then(|x| x.as_u64()),
            permit_opt.as_ref(),
        ) {
            (Some(b), Some(p)) => {
                if p.proposal.effect.charge() > b {
                    return Ok(denied("attenuation"));
                }
                b
            }
            (Some(b), None) => b,
            (None, Some(p)) => p.proposal.effect.charge(),
            (None, None) => return Err("missing budget".into()),
        };
        if child_budget == 0 {
            return Ok(denied("attenuation"));
        }

        let child_expires = match (
            v.get("expires").and_then(|x| x.as_u64()),
            permit_opt.as_ref(),
        ) {
            (Some(e), Some(p)) => {
                if p.proposal.expires > e {
                    return Ok(denied("attenuation"));
                }
                p.proposal.expires
            }
            (Some(e), None) => e,
            (None, Some(p)) => p.proposal.expires,
            (None, None) => return Err("missing expires".into()),
        };

        let child_scope: Vec<bool> = if v.get("scope").is_some_and(|x| !x.is_null()) {
            serde_json::from_value(v["scope"].clone())?
        } else if let Some(ref p) = permit_opt {
            p.proposal.effect.scope_vec()
        } else {
            vec![true, true, true]
        };
        if child_scope.len() != 3 || !child_scope.iter().any(|&x| x) {
            return Ok(denied("attenuation"));
        }
        if let Some(ref p) = permit_opt {
            if child_scope.get(p.proposal.effect.kind_index()) != Some(&true) {
                return Ok(denied("attenuation"));
            }
        }

        let (
            parent_opt,
            parent_budget,
            parent_expires,
            parent_scope,
            parent_required,
            parent_grants,
            parent_epochs,
        ) = if v.get("parent").is_none_or(|x| x.is_null()) {
            let p_scope = vec![true, true, true];
            let p_req = min_required_for_scope(&child_scope, n);
            let p_grants: Vec<bool> = if v.get("grants").is_some_and(|x| !x.is_null()) {
                serde_json::from_value(v["grants"].clone())?
            } else {
                self.live.clone()
            };
            let p_epochs: Vec<u64> = if v.get("epochs").is_some_and(|x| !x.is_null()) {
                serde_json::from_value(v["epochs"].clone())?
            } else {
                self.epochs.clone()
            };
            if p_grants.len() != n || p_epochs.len() != n {
                return Ok(denied("attenuation"));
            }
            for j in 0..n {
                if p_grants[j] && (!self.live[j] || p_epochs[j] != self.epochs[j]) {
                    return Ok(denied("withdrawn_or_stale_grant"));
                }
            }
            (
                None,
                self.budget,
                u64::MAX,
                p_scope,
                p_req,
                p_grants,
                p_epochs,
            )
        } else {
            let parent_id = text(v, "parent")?;
            if let Some(p_idx) = self.delegations.iter().position(|d| d.id == parent_id) {
                let (chain, root_app) = self.delegation_chain(p_idx)?;
                if let Some(a_idx) = root_app {
                    let pa = &self.approvals[a_idx];
                    let a_req = pa.permit.proposal.effect.required_for(n);
                    if !policy_core::current_grants(
                        &a_req,
                        &pa.grants,
                        &self.live,
                        &self.epochs,
                        &pa.permit.proposal.epochs,
                    ) {
                        return Ok(denied("withdrawn_or_stale_grant"));
                    }
                    if self.records.iter().any(|r| {
                        r.permit.proposal.id == pa.permit.proposal.id
                            && r.status != Status::Cancelled
                    }) {
                        return Ok(denied("delegated_budget"));
                    }
                }
                for &idx in &chain {
                    let anc = &self.delegations[idx];
                    if !policy_core::current_grants(
                        &anc.required,
                        &anc.grants,
                        &self.live,
                        &self.epochs,
                        &anc.epochs,
                    ) {
                        return Ok(denied("withdrawn_or_stale_grant"));
                    }
                }
                let pd = &self.delegations[p_idx];
                if self
                    .records
                    .iter()
                    .any(|r| r.permit.proposal.id == pd.id && r.status != Status::Cancelled)
                {
                    return Ok(denied("delegated_budget"));
                }
                (
                    Some(parent_id.to_string()),
                    pd.budget,
                    pd.expires,
                    pd.scope.clone(),
                    pd.required.clone(),
                    pd.grants.clone(),
                    pd.epochs.clone(),
                )
            } else if let Some(pa) = self
                .approvals
                .iter()
                .find(|a| a.permit.proposal.id == parent_id)
            {
                let a_req = pa.permit.proposal.effect.required_for(n);
                if !policy_core::current_grants(
                    &a_req,
                    &pa.grants,
                    &self.live,
                    &self.epochs,
                    &pa.permit.proposal.epochs,
                ) {
                    return Ok(denied("withdrawn_or_stale_grant"));
                }
                if self
                    .records
                    .iter()
                    .any(|r| r.permit.proposal.id == parent_id && r.status != Status::Cancelled)
                {
                    return Ok(denied("delegated_budget"));
                }
                (
                    Some(parent_id.to_string()),
                    pa.permit.proposal.effect.charge(),
                    pa.permit.proposal.expires,
                    pa.permit.proposal.effect.scope_vec(),
                    a_req,
                    pa.grants.clone(),
                    pa.permit.proposal.epochs.clone(),
                )
            } else {
                return Ok(denied("unknown_parent"));
            }
        };

        let min_child_req = min_required_for_scope(&child_scope, n);
        let child_required: Vec<bool> = if v.get("required").is_some_and(|x| !x.is_null()) {
            serde_json::from_value(v["required"].clone())?
        } else {
            (0..n)
                .map(|j| parent_required[j] || min_child_req[j])
                .collect()
        };
        if child_required.len() != n || (0..n).any(|j| min_child_req[j] && !child_required[j]) {
            return Ok(denied("attenuation"));
        }

        if !policy_core::attenuates(
            parent_budget,
            child_budget,
            parent_expires,
            child_expires,
            &parent_scope,
            &child_scope,
            &parent_required,
            &child_required,
        ) {
            return Ok(denied("attenuation"));
        }

        let child_epochs: Vec<u64> = if let Some(ref p) = permit_opt {
            p.proposal.epochs.clone()
        } else if v.get("epochs").is_some_and(|x| !x.is_null()) {
            serde_json::from_value(v["epochs"].clone())?
        } else {
            parent_epochs.clone()
        };

        let child_grants: Vec<bool> = if v.get("grants").is_some_and(|x| !x.is_null()) {
            let g: Vec<bool> = serde_json::from_value(v["grants"].clone())?;
            if g.len() != n {
                return Ok(denied("attenuation"));
            }
            for j in 0..n {
                if g[j]
                    && !parent_grants[j]
                    && (!self.live[j] || child_epochs.get(j) != Some(&self.epochs[j]))
                {
                    return Ok(denied("withdrawn_or_stale_grant"));
                }
            }
            g
        } else {
            parent_grants
        };

        if !policy_core::current_grants(
            &child_required,
            &child_grants,
            &self.live,
            &self.epochs,
            &child_epochs,
        ) {
            return Ok(denied("withdrawn_or_stale_grant"));
        }

        self.delegations.push(Delegation {
            id: id.clone(),
            parent: parent_opt,
            permit: permit_opt,
            budget: child_budget,
            charged: 0,
            expires: child_expires,
            scope: child_scope,
            required: child_required,
            grants: child_grants,
            epochs: child_epochs,
        });
        Ok(serde_json::json!({"status": "delegated", "id": id}))
    }
    pub fn submit(&mut self, permit: Permit, timestamp: u64) -> Result<serde_json::Value> {
        let p = &permit.proposal;
        if let Some(r) = self.records.iter().find(|r| r.permit.proposal.id == p.id) {
            return Ok(if same_permit(&r.permit.base_permit(), &permit)? {
                serde_json::json!({"status": r.status, "permit": r.permit})
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

        let n = self.issuers();
        let amount = p.effect.charge();
        let bytes = policy_core::encode_permit_canonical(&permit.to_exec_wire());

        if let Some(d_idx) = self
            .delegations
            .iter()
            .position(|d| d.permit.as_ref().is_some_and(|dp| dp.proposal.id == p.id))
        {
            let (chain, root_app) = self.delegation_chain(d_idx)?;
            if let Some(a_idx) = root_app {
                let pa = &self.approvals[a_idx];
                let a_req = pa.permit.proposal.effect.required_for(n);
                if !policy_core::current_grants(
                    &a_req,
                    &pa.grants,
                    &self.live,
                    &self.epochs,
                    &pa.permit.proposal.epochs,
                ) {
                    return Ok(denied("withdrawn_or_stale_grant"));
                }
                if self.records.iter().any(|r| {
                    r.permit.proposal.id == pa.permit.proposal.id && r.status != Status::Cancelled
                }) {
                    return Ok(denied("delegated_budget"));
                }
                let used = self.approval_delegated_charged(&pa.permit.proposal.id);
                if policy_core::reserve(pa.permit.proposal.effect.charge(), used, amount).is_none()
                {
                    return Ok(denied("policy"));
                }
            }
            let leaf = &self.delegations[d_idx];
            let approved = policy_core::encode_permit_canonical(
                &leaf.permit.as_ref().ok_or("missing permit")?.to_exec_wire(),
            );
            for &idx in &chain {
                let anc = &self.delegations[idx];
                if anc.scope.get(p.effect.kind_index()) != Some(&true) {
                    return Ok(denied("scope"));
                }
                if !policy_core::current_grants(
                    &anc.required,
                    &anc.grants,
                    &self.live,
                    &self.epochs,
                    &anc.epochs,
                ) {
                    return Ok(denied("withdrawn_or_stale_grant"));
                }
                if !policy_core::admit_multi(
                    anc.budget,
                    anc.charged,
                    amount,
                    timestamp,
                    anc.expires,
                    &bytes,
                    &approved,
                    &anc.required,
                    &anc.grants,
                    &self.live,
                    &self.epochs,
                    &anc.epochs,
                ) {
                    return Ok(denied("policy"));
                }
            }
            if !policy_core::admit_multi(
                self.budget,
                self.charged,
                amount,
                timestamp,
                p.expires,
                &bytes,
                &approved,
                &leaf.required,
                &leaf.grants,
                &self.live,
                &self.epochs,
                &p.epochs,
            ) {
                return Ok(denied("policy"));
            }
            let base_req = p.effect.required_for(n);
            let bound_req = if leaf.required == base_req {
                vec![]
            } else {
                leaf.required.clone()
            };
            let dispatched = DispatchedPermit {
                actor: permit.actor,
                proposal: permit.proposal,
                required: bound_req,
            };
            self.charged =
                policy_core::reserve(self.budget, self.charged, amount).ok_or("reservation")?;
            for &idx in &chain {
                let anc = &mut self.delegations[idx];
                anc.charged = policy_core::reserve(anc.budget, anc.charged, amount)
                    .ok_or("delegation reservation")?;
            }
            self.records.push(Record {
                permit: dispatched.clone(),
                status: Status::Pending,
            });
            return Ok(serde_json::json!({"status": "pending", "permit": dispatched}));
        }

        let Some(a) = self.approvals.iter().find(|a| a.permit.proposal.id == p.id) else {
            return Ok(denied("missing_grant"));
        };
        if self.approval_delegated_charged(&p.id) > 0 {
            return Ok(denied("delegated_budget"));
        }
        let required = p.effect.required_for(n);
        if !policy_core::current_grants(&required, &a.grants, &self.live, &self.epochs, &p.epochs) {
            return Ok(denied("withdrawn_or_stale_grant"));
        }
        let approved = policy_core::encode_permit_canonical(&a.permit.to_exec_wire());
        // Revisions are bound in the exact approved payload, then checked against
        // the authoritative object at the sink, where that revision lives.
        if !policy_core::admit_multi(
            self.budget,
            self.charged,
            amount,
            timestamp,
            p.expires,
            &bytes,
            &approved,
            &required,
            &a.grants,
            &self.live,
            &self.epochs,
            &p.epochs,
        ) {
            return Ok(denied("policy"));
        }
        self.charged =
            policy_core::reserve(self.budget, self.charged, amount).ok_or("reservation")?;
        let dispatched = DispatchedPermit::from(permit);
        self.records.push(Record {
            permit: dispatched.clone(),
            status: Status::Pending,
        });
        Ok(serde_json::json!({"status": "pending", "permit": dispatched}))
    }
    pub fn status(&self, id: &str, actor: &str) -> serde_json::Value {
        match self
            .records
            .iter()
            .find(|r| r.permit.proposal.id == id && r.permit.actor == actor)
        {
            Some(r) => serde_json::json!({"status": r.status}),
            None => denied("unknown_id"),
        }
    }
    pub fn withdraw(&mut self, issuer: usize) -> Result<serde_json::Value> {
        let i = self.check_issuer(issuer)?;
        let n = self.issuers();
        let old = self.epochs[i];
        self.epochs[i] = old.checked_add(1).ok_or("epoch exhausted")?;
        self.live[i] = false;
        // Approval and delegation grant bits are not restored by renewal.
        for a in &mut self.approvals {
            if i < a.grants.len() {
                a.grants[i] = false;
            }
        }
        for d in &mut self.delegations {
            if i < d.grants.len() {
                d.grants[i] = false;
            }
        }
        let mut outstanding = 0u64;
        for r in &self.records {
            if r.status == Status::Pending && r.permit.effective_required(n)?.get(i) == Some(&true)
            {
                outstanding += r.permit.proposal.effect.charge();
            }
        }
        Ok(serde_json::json!({
            "status": "admission_revoked",
            "issuer": i,
            "floor": self.epochs[i],
            "charged": self.charged,
            "outstanding": outstanding,
        }))
    }
    pub fn renew(&mut self, issuer: usize) -> Result<serde_json::Value> {
        let i = self.check_issuer(issuer)?;
        self.live[i] = true;
        Ok(serde_json::json!({"status": "renewed"}))
    }
    pub fn settle(&mut self, id: &str, sink_status: Status) -> Result<serde_json::Value> {
        // Trusted sink response only. A timeout is never a certificate.
        let r_idx = self
            .records
            .iter()
            .position(|r| r.permit.proposal.id == id)
            .ok_or("unknown id")?;
        if self.records[r_idx].status != Status::Pending
            || !matches!(sink_status, Status::Committed | Status::Cancelled)
        {
            return Ok(denied("not_authoritative_terminal"));
        }
        let amount = self.records[r_idx].permit.proposal.effect.charge();
        let cancelled = sink_status == Status::Cancelled;
        self.charged = policy_core::release(self.charged, amount, cancelled);
        if let Some(d_idx) = self
            .delegations
            .iter()
            .position(|d| d.permit.as_ref().is_some_and(|dp| dp.proposal.id == id))
        {
            let (chain, _) = self.delegation_chain(d_idx)?;
            for idx in chain {
                let anc = &mut self.delegations[idx];
                anc.charged = policy_core::release(anc.charged, amount, cancelled);
            }
        }
        self.records[r_idx].status = sink_status.clone();
        Ok(serde_json::json!({"status": sink_status}))
    }
    pub fn to_exec_kernel(&self) -> Option<policy_core::ExecBrokerKernel> {
        let n = self.issuers();
        if n == 0 || self.live.len() != n {
            return None;
        }
        let mut records = Vec::with_capacity(self.records.len());
        let mut inflight_ridxs = Vec::new();
        let mut inflight_amounts = Vec::new();
        let mut pending = 0u64;
        let mut committed = 0u64;
        let mut ids = HashSet::new();
        for (r_idx, r) in self.records.iter().enumerate() {
            if !ids.insert(r.permit.proposal.id.as_str()) {
                return None;
            }
            let wire = r.permit.to_exec_wire_for_issuers(n).ok()?;
            if wire.amount == 0 || wire.amount > self.budget {
                return None;
            }
            let (status, inflight_pos) = match r.status {
                Status::Pending => {
                    let pos = inflight_ridxs.len();
                    pending = pending.checked_add(wire.amount)?;
                    inflight_ridxs.push(r_idx);
                    inflight_amounts.push(wire.amount);
                    (policy_core::ExecSlotStatus::Pending, pos)
                }
                Status::Committed => {
                    committed = committed.checked_add(wire.amount)?;
                    (policy_core::ExecSlotStatus::Committed, 0)
                }
                Status::Cancelled => (policy_core::ExecSlotStatus::Cancelled, 0),
                Status::Denied => return None,
            };
            records.push(policy_core::ExecBrokerRecord {
                permit: wire,
                status,
                inflight_pos,
            });
        }
        if pending.checked_add(committed) != Some(self.charged) || self.charged > self.budget {
            return None;
        }
        Some(policy_core::ExecBrokerKernel {
            budget: self.budget,
            charged: self.charged,
            committed,
            pending,
            live: self.live.clone(),
            epochs: self.epochs.clone(),
            records,
            inflight_ridxs,
            inflight_amounts,
        })
    }
    pub fn audit_state(&self, integrity: &str) -> serde_json::Value {
        let n = self.issuers();
        let mut pending_charge = 0u64;
        let mut committed_charge = 0u64;
        let mut cancelled_charge = 0u64;
        let mut ids = HashSet::new();
        let mut unique_ids = true;
        let mut statuses_valid = true;
        let mut permits_valid = true;
        for r in &self.records {
            if !ids.insert(r.permit.proposal.id.clone()) {
                unique_ids = false;
            }
            if r.permit.effective_required(n).is_err() {
                permits_valid = false;
            }
            let c = r.permit.proposal.effect.charge();
            match r.status {
                Status::Pending => pending_charge = pending_charge.saturating_add(c),
                Status::Committed => committed_charge = committed_charge.saturating_add(c),
                Status::Cancelled => cancelled_charge = cancelled_charge.saturating_add(c),
                Status::Denied => statuses_valid = false,
            }
        }
        let delegations_ok = self
            .delegations
            .iter()
            .all(|d| d.charged <= d.budget && d.required.len() == n);
        let sum_active = pending_charge.checked_add(committed_charge);
        let exec_kernel_ok = self.to_exec_kernel().is_some_and(|k| {
            k.charged == self.charged
                && k.pending == pending_charge
                && k.committed == committed_charge
                && k.charged <= k.budget
        });
        let consistent = integrity == "ok"
            && unique_ids
            && statuses_valid
            && permits_valid
            && delegations_ok
            && self.epochs.len() == self.live.len()
            && !self.epochs.is_empty()
            && sum_active == Some(self.charged)
            && self.charged <= self.budget
            && exec_kernel_ok;
        serde_json::json!({
            "status": if consistent { "ok" } else { "corrupt" },
            "consistent": consistent,
            "integrity": integrity,
            "budget": self.budget,
            "charged": self.charged,
            "pending_charge": pending_charge,
            "committed_charge": committed_charge,
            "cancelled_charge": cancelled_charge,
            "records": self.records.len(),
            "delegations_consistent": delegations_ok,
            "verus_exec_kernel_verified": exec_kernel_ok,
        })
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
        Self::with_issuers(2).expect("valid default issuer count")
    }
}
impl Sink {
    pub fn with_issuers(issuers: usize) -> Result<Self> {
        if issuers == 0 || issuers > 64 {
            return Err("issuers out of range".into());
        }
        Ok(Self {
            floors: vec![0; issuers],
            records: vec![],
            revision: 0,
            document: "fn answer() -> u32 { 41 }".into(),
        })
    }
    pub fn issuers(&self) -> usize {
        self.floors.len()
    }
    pub fn to_exec_kernel(&self) -> Option<policy_core::ExecSinkKernel> {
        let n = self.issuers();
        if n == 0 {
            return None;
        }
        let mut records = Vec::with_capacity(self.records.len());
        let mut committed_total = 0u64;
        let mut ids = HashSet::new();
        for r in &self.records {
            if !ids.insert(r.permit.proposal.id.as_str()) {
                return None;
            }
            if r.permit.proposal.epochs.len() != n {
                return None;
            }
            let wire = r.permit.to_exec_wire_for_issuers(n).ok()?;
            let status = match r.status {
                Status::Pending => {
                    if !policy_core::unfenced(&wire.required, &wire.epochs, &self.floors) {
                        return None;
                    }
                    policy_core::ExecSlotStatus::Pending
                }
                Status::Committed => {
                    committed_total = committed_total.checked_add(wire.amount)?;
                    policy_core::ExecSlotStatus::Committed
                }
                Status::Cancelled => policy_core::ExecSlotStatus::Cancelled,
                Status::Denied => return None,
            };
            records.push(policy_core::ExecSinkRecord {
                permit: wire,
                status,
            });
        }
        Some(policy_core::ExecSinkKernel {
            floors: self.floors.clone(),
            revision: self.revision,
            committed_total,
            records,
        })
    }
    pub fn accept(&mut self, permit: Permit) -> Result<serde_json::Value> {
        self.accept_dispatched(DispatchedPermit::from(permit))
    }
    pub fn accept_dispatched(&mut self, permit: DispatchedPermit) -> Result<serde_json::Value> {
        let n = self.issuers();
        let (permit, eff_req) = match permit.normalized(n) {
            Ok(v) => v,
            Err(_) => return Ok(denied("invalid_required_issuers")),
        };
        let mut exec_shadow = if permit.proposal.epochs.len() == n {
            self.to_exec_kernel()
        } else {
            None
        };
        let exec_outcome = if let Some(ref mut k) = exec_shadow {
            let mut wire = permit.base_permit().to_exec_wire();
            wire.required = eff_req.clone();
            Some(k.sink_accept(&wire))
        } else {
            None
        };
        let resp = if let Some(r) = self
            .records
            .iter()
            .find(|r| r.permit.proposal.id == permit.proposal.id)
        {
            if same_dispatched_permit(&r.permit, &permit)? {
                if let Some(out) = exec_outcome {
                    debug_assert!(matches!(
                        out,
                        policy_core::ExecAcceptOutcome::Duplicate { .. }
                    ));
                }
                serde_json::json!({"status": r.status})
            } else {
                if let Some(out) = exec_outcome {
                    debug_assert_eq!(out, policy_core::ExecAcceptOutcome::PayloadMismatch);
                }
                denied("payload_mismatch")
            }
        } else if !policy_core::unfenced(&eff_req, &permit.proposal.epochs, &self.floors) {
            // A durable cancellation tombstone also covers a
            // previously issued permit whose first delivery is late.
            if let Some(out) = exec_outcome {
                debug_assert_eq!(out, policy_core::ExecAcceptOutcome::CancelledTombstone);
            }
            self.records.push(Record {
                permit,
                status: Status::Cancelled,
            });
            serde_json::json!({"status": "cancelled"})
        } else {
            if let Some(out) = exec_outcome {
                debug_assert_eq!(out, policy_core::ExecAcceptOutcome::AcceptedPending);
            }
            self.records.push(Record {
                permit,
                status: Status::Pending,
            });
            serde_json::json!({"status": "pending"})
        };
        Ok(resp)
    }
    pub fn complete(&mut self, id: &str) -> Result<(serde_json::Value, Option<DispatchedPermit>)> {
        let n = self.issuers();
        let mut exec_shadow = self.to_exec_kernel();
        let exec_outcome = exec_shadow
            .as_mut()
            .map(|k| k.sink_complete(&id.as_bytes().to_vec()));
        let r = self
            .records
            .iter_mut()
            .find(|r| r.permit.proposal.id == id)
            .ok_or("unknown id")?;
        let mut newly_committed = None;
        if r.status == Status::Pending {
            let eff_req = r.permit.effective_required(n)?;
            let p = &r.permit.proposal;
            let revision = match &p.effect {
                Effect::Edit {
                    expected_revision, ..
                } => policy_core::revision_matches(self.revision, *expected_revision),
                _ => true,
            };
            if !policy_core::unfenced(&eff_req, &p.epochs, &self.floors) || !revision {
                r.status = Status::Cancelled;
                if let Some(out) = exec_outcome {
                    debug_assert_eq!(out, policy_core::ExecCompleteOutcome::Cancelled);
                }
            } else {
                if let Effect::Edit { contents, .. } = &p.effect {
                    self.revision = self.revision.checked_add(1).ok_or("revision exhausted")?;
                    self.document = contents.clone();
                }
                r.status = Status::Committed;
                newly_committed = Some(r.permit.clone());
                if let Some(out) = exec_outcome {
                    debug_assert_eq!(
                        out,
                        policy_core::ExecCompleteOutcome::NewlyCommitted {
                            amount: p.effect.charge()
                        }
                    );
                }
            }
        } else if let Some(out) = exec_outcome {
            debug_assert!(matches!(
                out,
                policy_core::ExecCompleteOutcome::Idempotent { .. }
            ));
        }
        if let Some(ref k) = exec_shadow {
            debug_assert_eq!(k.revision, self.revision);
        }
        Ok((serde_json::json!({"status": r.status}), newly_committed))
    }
    pub fn fence(&mut self, issuer: usize, floor: u64) -> Result<serde_json::Value> {
        let n = self.issuers();
        if issuer >= n {
            return Err("issuer out of range".into());
        }
        let mut exec_shadow = self.to_exec_kernel();
        if let Some(ref mut k) = exec_shadow {
            let ok = k.sink_fence(issuer, floor);
            debug_assert!(ok);
        }
        self.floors[issuer] = policy_core::advance_floor(self.floors[issuer], floor);
        for r in &mut self.records {
            if r.status == Status::Pending {
                let eff_req = r.permit.effective_required(n)?;
                if !policy_core::unfenced(&eff_req, &r.permit.proposal.epochs, &self.floors) {
                    r.status = Status::Cancelled;
                }
            }
        }
        if let Some(ref k) = exec_shadow {
            debug_assert_eq!(k.floors, self.floors);
        }
        Ok(serde_json::json!({
            "status": "fenced",
            "issuer": issuer,
            "floor": self.floors[issuer],
        }))
    }
    pub fn status(&self, id: &str) -> serde_json::Value {
        match self.records.iter().find(|r| r.permit.proposal.id == id) {
            Some(r) => serde_json::json!({"status": r.status}),
            // Absence alone is not authoritative: an earlier send may arrive later.
            None => serde_json::json!({"status": "unknown"}),
        }
    }
    pub fn audit_state(
        &self,
        integrity: &str,
        effects_rows: &[(String, DispatchedPermit, u64)],
    ) -> serde_json::Value {
        let n = self.issuers();
        let mut ids = HashSet::new();
        let mut unique_ids = true;
        let mut committed_records = Vec::new();
        let mut pending_count = 0usize;
        let mut cancelled_count = 0usize;
        let mut pending_unfenced = true;
        let mut statuses_valid = true;
        for r in &self.records {
            if !ids.insert(r.permit.proposal.id.clone()) {
                unique_ids = false;
            }
            let Ok(eff_req) = r.permit.effective_required(n) else {
                statuses_valid = false;
                continue;
            };
            match r.status {
                Status::Pending => {
                    pending_count += 1;
                    if !policy_core::unfenced(&eff_req, &r.permit.proposal.epochs, &self.floors) {
                        pending_unfenced = false;
                    }
                }
                Status::Committed => committed_records.push(&r.permit),
                Status::Cancelled => cancelled_count += 1,
                Status::Denied => statuses_valid = false,
            }
        }
        let mut effects_match = committed_records.len() == effects_rows.len();
        let mut effects_charge_sum = 0u64;
        if effects_match {
            let mut committed_by_id = std::collections::HashMap::new();
            for permit in &committed_records {
                committed_by_id.insert(permit.proposal.id.as_str(), *permit);
            }
            let mut seen_effects = HashSet::new();
            for (eid, epermit, echarge) in effects_rows {
                if !seen_effects.insert(eid.as_str()) {
                    effects_match = false;
                    break;
                }
                effects_charge_sum = effects_charge_sum.saturating_add(*echarge);
                match committed_by_id.get(eid.as_str()) {
                    Some(cpermit) => {
                        let same = same_dispatched_permit(cpermit, epermit).unwrap_or(false);
                        if !same || *echarge != cpermit.proposal.effect.charge() {
                            effects_match = false;
                            break;
                        }
                    }
                    None => {
                        effects_match = false;
                        break;
                    }
                }
            }
        }
        let edit_commits = effects_rows
            .iter()
            .filter(|(_, p, _)| matches!(p.proposal.effect, Effect::Edit { .. }))
            .count() as u64;
        let exec_kernel_ok = self.to_exec_kernel().is_none_or(|k| {
            k.floors == self.floors
                && k.revision == self.revision
                && (!effects_match || k.committed_total == effects_charge_sum)
        });
        let consistent = integrity == "ok"
            && !self.floors.is_empty()
            && unique_ids
            && statuses_valid
            && pending_unfenced
            && effects_match
            && self.revision == edit_commits
            && exec_kernel_ok;
        serde_json::json!({
            "status": if consistent { "ok" } else { "corrupt" },
            "consistent": consistent,
            "integrity": integrity,
            "effects": effects_rows.len(),
            "committed_permits": committed_records.len(),
            "pending_permits": pending_count,
            "cancelled_permits": cancelled_count,
            "pending_unfenced": pending_unfenced,
            "revision": self.revision,
            "floors": self.floors,
            "verus_exec_kernel_verified": exec_kernel_ok,
        })
    }
}

pub fn denied(reason: &str) -> serde_json::Value {
    serde_json::json!({"status": "denied", "reason": reason})
}
fn issuer_arg(v: &serde_json::Value) -> Result<usize> {
    let i = v["issuer"].as_u64().ok_or("issuer")?;
    usize::try_from(i).map_err(|_| "issuer out of range".into())
}
fn text<'a>(v: &'a serde_json::Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .ok_or_else(|| format!("missing {key}").into())
}

pub const DEFAULT_RPC_DEADLINE_MS: i32 = 500;

enum RpcFrame {
    Eof,
    Oversize,
    InvalidUtf8,
    Timeout,
    Line(String),
}

fn rpc_deadline_ms() -> i32 {
    std::env::var("JEATH_RPC_DEADLINE_MS")
        .ok()
        .and_then(|s| s.parse::<i32>().ok())
        .map(|v| v.clamp(5, 60_000))
        .unwrap_or(DEFAULT_RPC_DEADLINE_MS)
}

#[cfg(unix)]
struct FdLineBuffer {
    fd: libc::c_int,
    buf: Vec<u8>,
    pos: usize,
    filled: usize,
}

#[cfg(unix)]
impl FdLineBuffer {
    fn new(fd: libc::c_int) -> Self {
        Self {
            fd,
            buf: vec![0u8; 8192],
            pos: 0,
            filled: 0,
        }
    }

    fn consume(&mut self, amt: usize) {
        self.pos = (self.pos + amt).min(self.filled);
    }

    fn fill_chunk(&mut self, timeout_ms: Option<i32>) -> io::Result<Option<&[u8]>> {
        if self.pos < self.filled {
            return Ok(Some(&self.buf[self.pos..self.filled]));
        }
        if let Some(ms) = timeout_ms {
            loop {
                let mut pfd = libc::pollfd {
                    fd: self.fd,
                    events: libc::POLLIN,
                    revents: 0,
                };
                let rc = unsafe { libc::poll(&mut pfd, 1, ms.max(0)) };
                if rc == 0 {
                    return Ok(None);
                }
                if rc < 0 {
                    let err = io::Error::last_os_error();
                    if err.kind() == io::ErrorKind::Interrupted {
                        continue;
                    }
                    return Err(err);
                }
                break;
            }
        }
        loop {
            let n = unsafe { libc::read(self.fd, self.buf.as_mut_ptr().cast(), self.buf.len()) };
            if n < 0 {
                let err = io::Error::last_os_error();
                if err.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(err);
            }
            self.pos = 0;
            self.filled = n as usize;
            return Ok(Some(&self.buf[..self.filled]));
        }
    }
}

#[cfg(unix)]
fn read_rpc_frame_fd_with_deadline(
    reader: &mut FdLineBuffer,
    limit: usize,
    deadline_ms: i32,
) -> io::Result<RpcFrame> {
    let mut buf = Vec::new();
    let mut oversize = false;
    let mut started_at: Option<std::time::Instant> = None;
    loop {
        let timeout_opt = if let Some(t0) = started_at {
            let elapsed = t0.elapsed().as_millis();
            if elapsed >= deadline_ms.max(1) as u128 {
                return Ok(RpcFrame::Timeout);
            }
            Some((deadline_ms.max(1) as u128 - elapsed) as i32)
        } else {
            None
        };
        let Some(chunk) = reader.fill_chunk(timeout_opt)? else {
            return Ok(RpcFrame::Timeout);
        };
        if chunk.is_empty() {
            if buf.is_empty() && !oversize {
                return Ok(RpcFrame::Eof);
            }
            break;
        }
        if started_at.is_none() {
            started_at = Some(std::time::Instant::now());
        }
        if let Some(pos) = chunk.iter().position(|&b| b == b'\n') {
            let part = &chunk[..pos];
            if !oversize {
                let effective_len = if part.ends_with(b"\r") {
                    buf.len() + part.len() - 1
                } else {
                    buf.len() + part.len()
                };
                if effective_len > limit {
                    oversize = true;
                    buf.clear();
                } else {
                    buf.extend_from_slice(part);
                }
            }
            reader.consume(pos + 1);
            break;
        } else {
            if !oversize {
                if buf.len() + chunk.len() > limit + 1 {
                    oversize = true;
                    buf.clear();
                } else {
                    buf.extend_from_slice(chunk);
                }
            }
            let n = chunk.len();
            reader.consume(n);
        }
    }
    if oversize {
        return Ok(RpcFrame::Oversize);
    }
    if buf.ends_with(b"\r") {
        buf.pop();
    }
    if buf.len() > limit {
        return Ok(RpcFrame::Oversize);
    }
    match String::from_utf8(buf) {
        Ok(line) => Ok(RpcFrame::Line(line)),
        Err(_) => Ok(RpcFrame::InvalidUtf8),
    }
}

#[cfg(any(test, not(unix)))]
fn read_rpc_frame<R: io::BufRead>(reader: &mut R, limit: usize) -> io::Result<RpcFrame> {
    let mut buf = Vec::new();
    let mut oversize = false;
    loop {
        let chunk = reader.fill_buf()?;
        if chunk.is_empty() {
            if buf.is_empty() && !oversize {
                return Ok(RpcFrame::Eof);
            }
            break;
        }
        if let Some(pos) = chunk.iter().position(|&b| b == b'\n') {
            let part = &chunk[..pos];
            if !oversize {
                let effective_len = if part.ends_with(b"\r") {
                    buf.len() + part.len() - 1
                } else {
                    buf.len() + part.len()
                };
                if effective_len > limit {
                    oversize = true;
                    buf.clear();
                } else {
                    buf.extend_from_slice(part);
                }
            }
            reader.consume(pos + 1);
            break;
        } else {
            if !oversize {
                if buf.len() + chunk.len() > limit + 1 {
                    oversize = true;
                    buf.clear();
                } else {
                    buf.extend_from_slice(chunk);
                }
            }
            let n = chunk.len();
            reader.consume(n);
        }
    }
    if oversize {
        return Ok(RpcFrame::Oversize);
    }
    if buf.ends_with(b"\r") {
        buf.pop();
    }
    if buf.len() > limit {
        return Ok(RpcFrame::Oversize);
    }
    match String::from_utf8(buf) {
        Ok(line) => Ok(RpcFrame::Line(line)),
        Err(_) => Ok(RpcFrame::InvalidUtf8),
    }
}

/// Each process owns only its own database. The JSONL channel is a trusted
/// control channel; launch_v2.py exposes only submit/status to proposer children.
pub fn serve(path: &Path, role: &str) -> Result<()> {
    if rusqlite::version_number() < 3_051_003 {
        return Err("SQLite >= 3.51.3 required".into());
    }
    let mut db = Connection::open(path)?;
    db.busy_timeout(Duration::from_secs(10))?;
    db.execute_batch(
        "PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
        CREATE TABLE IF NOT EXISTS state(singleton INTEGER PRIMARY KEY CHECK(singleton=1), body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS effects(sequence INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL, charge INTEGER NOT NULL);",
    )?;
    let stdin = io::stdin();
    let deadline_ms = rpc_deadline_ms();
    #[cfg(unix)]
    let mut fd_reader = {
        use std::os::fd::AsRawFd;
        FdLineBuffer::new(stdin.as_raw_fd())
    };
    #[cfg(not(unix))]
    let mut reader = stdin.lock();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    loop {
        #[cfg(unix)]
        let frame = read_rpc_frame_fd_with_deadline(&mut fd_reader, MAX_RPC_BYTES, deadline_ms)?;
        #[cfg(not(unix))]
        let frame = read_rpc_frame(&mut reader, MAX_RPC_BYTES)?;
        let line = match frame {
            RpcFrame::Eof => break,
            RpcFrame::Oversize => {
                let response = denied("rpc_too_large");
                serde_json::to_writer(&mut out, &response)?;
                writeln!(out)?;
                out.flush()?;
                continue;
            }
            RpcFrame::InvalidUtf8 => {
                let response = denied("invalid_utf8");
                serde_json::to_writer(&mut out, &response)?;
                writeln!(out)?;
                out.flush()?;
                continue;
            }
            RpcFrame::Timeout => {
                let response = denied("rpc_frame_timeout");
                serde_json::to_writer(&mut out, &response)?;
                writeln!(out)?;
                out.flush()?;
                continue;
            }
            RpcFrame::Line(line) => line,
        };
        let response = (|| -> Result<serde_json::Value> {
            let v: serde_json::Value = serde_json::from_str(&line)?;
            let tx = db.transaction_with_behavior(TransactionBehavior::Immediate)?;
            let op = text(&v, "op")?;
            let saved: Option<String> = tx
                .query_row("SELECT body FROM state WHERE singleton=1", [], |r| r.get(0))
                .optional()?;
            if op == "init" {
                if saved.is_some() {
                    return Ok(denied("already_initialized"));
                }
                let issuers = match v.get("issuers").filter(|x| !x.is_null()) {
                    Some(val) => usize::try_from(val.as_u64().ok_or("issuers")?)
                        .map_err(|_| "issuers out of range")?,
                    None => 2usize,
                };
                let body = if role == "broker" {
                    let b = v["budget"]
                        .as_u64()
                        .filter(|x| *x <= i64::MAX as u64)
                        .ok_or("budget")?;
                    serde_json::to_string(&Broker::with_issuers(b, issuers)?)?
                } else {
                    serde_json::to_string(&Sink::with_issuers(issuers)?)?
                };
                tx.execute("INSERT INTO state VALUES(1,?1)", [body])?;
                tx.commit()?;
                return Ok(serde_json::json!({"status": "initialized"}));
            }
            let saved = saved.ok_or("not initialized")?;
            let response;
            let mut mutated = true;
            let body;
            if role == "broker" {
                let mut b: Broker = serde_json::from_str(&saved)?;
                response = match op {
                    "approve" => {
                        let permit: Permit = serde_json::from_value(v["permit"].clone())?;
                        b.approve(permit, issuer_arg(&v)?)?
                    }
                    "delegate" => b.delegate(&v)?,
                    "submit" => {
                        let proposal: Proposal = serde_json::from_value(v["proposal"].clone())?;
                        b.submit(
                            Permit {
                                actor: text(&v, "session_actor")?.into(),
                                proposal,
                            },
                            crate::runtime::now() as u64,
                        )?
                    }
                    "status" => {
                        mutated = false;
                        b.status(text(&v, "id")?, text(&v, "session_actor")?)
                    }
                    "withdraw" => b.withdraw(issuer_arg(&v)?)?,
                    "renew" => b.renew(issuer_arg(&v)?)?,
                    "settle" => {
                        let sink_status: Status = serde_json::from_value(v["sink_status"].clone())?;
                        b.settle(text(&v, "id")?, sink_status)?
                    }
                    "inspect" => {
                        mutated = false;
                        serde_json::to_value(&b)?
                    }
                    "audit" => {
                        mutated = false;
                        let integrity: String =
                            tx.query_row("PRAGMA integrity_check", [], |r| r.get(0))?;
                        b.audit_state(&integrity)
                    }
                    _ => {
                        mutated = false;
                        denied("unknown_control_operation")
                    }
                };
                body = serde_json::to_string(&b)?;
            } else {
                let mut s: Sink = serde_json::from_str(&saved)?;
                response = match op {
                    "accept" => {
                        let permit: DispatchedPermit = serde_json::from_value(v["permit"].clone())?;
                        s.accept_dispatched(permit)?
                    }
                    "complete" => {
                        let id = text(&v, "id")?;
                        let (resp, newly_committed) = s.complete(id)?;
                        if let Some(permit) = newly_committed {
                            let charge = i64::try_from(permit.proposal.effect.charge())?;
                            tx.execute(
                                "INSERT INTO effects(id,payload,charge) VALUES(?1,?2,?3)",
                                params![id, serde_json::to_string(&permit)?, charge],
                            )?;
                        }
                        resp
                    }
                    "fence" => {
                        let floor = v["floor"].as_u64().ok_or("floor")?;
                        s.fence(issuer_arg(&v)?, floor)?
                    }
                    "status" => {
                        mutated = false;
                        s.status(text(&v, "id")?)
                    }
                    "inspect" => {
                        mutated = false;
                        serde_json::to_value(&s)?
                    }
                    "audit" => {
                        mutated = false;
                        let integrity: String =
                            tx.query_row("PRAGMA integrity_check", [], |r| r.get(0))?;
                        let mut stmt = tx
                            .prepare("SELECT id, payload, charge FROM effects ORDER BY sequence")?;
                        let mut rows = Vec::new();
                        let mut mapped = stmt.query([])?;
                        while let Some(row) = mapped.next()? {
                            let id: String = row.get(0)?;
                            let payload: String = row.get(1)?;
                            let charge: i64 = row.get(2)?;
                            let permit: DispatchedPermit = serde_json::from_str(&payload)?;
                            let ucharge = u64::try_from(charge)?;
                            rows.push((id, permit, ucharge));
                        }
                        drop(mapped);
                        drop(stmt);
                        s.audit_state(&integrity, &rows)
                    }
                    _ => {
                        mutated = false;
                        denied("unknown_sink_operation")
                    }
                };
                body = serde_json::to_string(&s)?;
            }
            if mutated {
                tx.execute("UPDATE state SET body=?1 WHERE singleton=1", [body])?;
            }
            if let Some(delay) = v["delay_before_commit_ms"].as_u64() {
                std::thread::sleep(Duration::from_millis(delay.min(1000)));
            }
            if v["crash"].as_str() == Some("before_commit") {
                std::process::exit(86);
            }
            tx.commit()?;
            if v["crash"].as_str() == Some("after_commit") {
                std::process::exit(87);
            }
            Ok(response)
        })()
        .unwrap_or_else(|e| denied(&e.to_string()));
        serde_json::to_writer(&mut out, &response)?;
        writeln!(out)?;
        out.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn read_rpc_frame_drains_oversize_and_recovers() {
        let mut stream = Vec::new();
        stream.extend_from_slice(&vec![b'a'; MAX_RPC_BYTES + 10]);
        stream.push(b'\n');
        stream.extend_from_slice(br#"{"op":"inspect"}"#);
        stream.push(b'\n');
        let mut cursor = Cursor::new(stream);
        assert!(matches!(
            read_rpc_frame(&mut cursor, MAX_RPC_BYTES).unwrap(),
            RpcFrame::Oversize
        ));
        match read_rpc_frame(&mut cursor, MAX_RPC_BYTES).unwrap() {
            RpcFrame::Line(s) => assert_eq!(s, r#"{"op":"inspect"}"#),
            _ => panic!("expected valid second line after oversize frame"),
        }
        assert!(matches!(
            read_rpc_frame(&mut cursor, MAX_RPC_BYTES).unwrap(),
            RpcFrame::Eof
        ));
    }

    #[cfg(unix)]
    #[test]
    fn read_rpc_frame_fd_enforces_mid_frame_slowloris_deadline_without_idle_false_timeout() {
        use std::os::fd::AsRawFd;
        use std::os::unix::net::UnixStream;
        let (mut writer, reader_sock) = UnixStream::pair().unwrap();
        let mut fb = FdLineBuffer::new(reader_sock.as_raw_fd());

        // Idle pause before frame start must not trigger a timeout, then valid line succeeds.
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(80));
            writer.write_all(br#"{"op":"inspect"}"#).unwrap();
            writer.write_all(b"\n").unwrap();
            writer.flush().unwrap();
            // Now send an incomplete slowloris partial frame without newline and stall!
            writer.write_all(br#"{"op":"submit","partial":"#).unwrap();
            writer.flush().unwrap();
            std::thread::sleep(Duration::from_millis(160));
            writer
        });

        match read_rpc_frame_fd_with_deadline(&mut fb, MAX_RPC_BYTES, 40).unwrap() {
            RpcFrame::Line(s) => assert_eq!(s, r#"{"op":"inspect"}"#),
            _ => panic!("expected complete frame after idle wait"),
        }

        let t0 = std::time::Instant::now();
        assert!(matches!(
            read_rpc_frame_fd_with_deadline(&mut fb, MAX_RPC_BYTES, 40).unwrap(),
            RpcFrame::Timeout
        ));
        assert!(t0.elapsed() < Duration::from_millis(140));
        let _writer = handle.join().unwrap();
    }

    #[test]
    fn broker_and_sink_match_verus_exec_kernels_and_canonical_wire() {
        let mut b = Broker::new(10);
        let mut s = Sink::default();
        let p = Permit {
            actor: "alice".into(),
            proposal: Proposal {
                id: "p-1".into(),
                epochs: vec![0, 0],
                expires: 500,
                effect: Effect::Transfer {
                    recipient: "approved".into(),
                    units: 4,
                },
            },
        };
        assert_eq!(b.approve(p.clone(), 0).unwrap()["status"], "approved");
        // Tampered units under same ID must fail canonical wire admission.
        let mut tampered = p.clone();
        tampered.proposal.effect = Effect::Transfer {
            recipient: "approved".into(),
            units: 5,
        };
        assert_eq!(b.submit(tampered, 10).unwrap()["status"], "denied");

        let sub = b.submit(p.clone(), 10).unwrap();
        assert_eq!(sub["status"], "pending");
        let disp: DispatchedPermit = serde_json::from_value(sub["permit"].clone()).unwrap();
        assert_eq!(
            s.accept_dispatched(disp.clone()).unwrap()["status"],
            "pending"
        );
        let (comp, newly) = s.complete("p-1").unwrap();
        assert_eq!(comp["status"], "committed");
        assert!(newly.is_some());
        assert_eq!(
            b.settle("p-1", Status::Committed).unwrap()["status"],
            "committed"
        );
        assert_eq!(b.audit_state("ok")["verus_exec_kernel_verified"], true);
        assert_eq!(
            s.audit_state("ok", &[("p-1".into(), disp, 4)])["verus_exec_kernel_verified"],
            true
        );
    }
}
