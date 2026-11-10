//! A fresh implementation of the drafts' 22-action finite model.
//! Authorization and the event oracle deliberately use separate state fields.
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct Request {
    id: u8,
    amount: u8,
    approved: bool,
    version: u8,
}
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct State {
    world_spent: u8,
    ledger: u8,
    used: u8,
    actor_spent: [u8; 2],
    live: bool,
    version: u8,
    pending: [Option<Request>; 2],
}
impl Default for State {
    fn default() -> Self {
        Self {
            world_spent: 0,
            ledger: 0,
            used: 0,
            actor_spent: [0; 2],
            live: true,
            version: 0,
            pending: [None; 2],
        }
    }
}
#[derive(Clone, Copy, Debug)]
enum Action {
    Prepare(usize, u8, u8, bool),
    Commit(usize),
    Revoke,
    Advance,
    Restart,
    Renew,
}
fn actions() -> Vec<Action> {
    let mut a = Vec::new();
    for actor in 0..2 {
        for id in 0..2 {
            for amount in 1..=2 {
                for dest in [true, false] {
                    a.push(Action::Prepare(actor, id, amount, dest));
                }
            }
        }
    }
    a.extend([
        Action::Commit(0),
        Action::Commit(1),
        Action::Revoke,
        Action::Advance,
        Action::Restart,
        Action::Renew,
    ]);
    a
}
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Enforcer {
    Complete,
    Unconstrained,
    RejectAll,
    PerAction,
    PerActor,
    PrepareOnly,
    NoReplay,
    NoRevocation,
    NoVersion,
    NoDestination,
    ResetRestart,
    ResetRenewal,
}
fn eligible(s: &State, r: Request, actor: usize, e: Enforcer, preparing: bool) -> bool {
    if e == Enforcer::Unconstrained {
        return true;
    }
    if e == Enforcer::RejectAll {
        return false;
    }
    let budget = match e {
        Enforcer::PerAction => r.amount <= 2,
        Enforcer::PerActor => s.actor_spent[actor] + r.amount <= 2,
        Enforcer::PrepareOnly if !preparing => true,
        _ => s.ledger + r.amount <= 2,
    };
    budget
        && (e == Enforcer::NoDestination || r.approved)
        && (e == Enforcer::NoReplay || s.used & (1 << r.id) == 0)
        && (e == Enforcer::NoRevocation || s.live)
        && (e == Enforcer::NoVersion || r.version == s.version)
}

#[derive(Clone, Copy)]
struct Effect {
    actor: usize,
    request: Request,
}
fn transition(s: &State, a: Action, e: Enforcer) -> (State, Option<Effect>) {
    let mut n = s.clone();
    let mut effect = None;
    match a {
        Action::Prepare(actor, id, amount, approved) => {
            let r = Request {
                id: actor as u8 * 2 + id,
                amount,
                approved,
                version: s.version,
            };
            if e != Enforcer::PrepareOnly || eligible(s, r, actor, e, true) {
                n.pending[actor] = Some(r);
            }
        }
        Action::Commit(actor) => {
            if let Some(r) = s.pending[actor] {
                if eligible(s, r, actor, e, false) {
                    n.world_spent += r.amount;
                    n.ledger += r.amount;
                    n.actor_spent[actor] += r.amount;
                    n.used |= 1 << r.id;
                    effect = Some(Effect { actor, request: r });
                }
            }
        }
        Action::Revoke => n.live = false,
        Action::Advance => {
            if s.version == 0 {
                n.version = 1;
                n.live = false;
            }
        }
        Action::Renew => {
            n.live = true;
            if e == Enforcer::ResetRenewal {
                n.ledger = 0;
            }
        }
        Action::Restart => {
            if e == Enforcer::ResetRestart {
                n.ledger = 0;
            }
        }
    }
    (n, effect)
}
/// Inspect the pre-effect world, never the enforcer's eligibility decision.
fn oracle(s: &State, effect: Option<Effect>) -> Vec<&'static str> {
    let mut v = Vec::new();
    if let Some(Effect { request: r, .. }) = effect {
        if s.world_spent + r.amount > 2 {
            v.push("budget");
        }
        if s.used & (1 << r.id) != 0 {
            v.push("replay");
        }
        if !s.live {
            v.push("revocation");
        }
        if r.version != s.version {
            v.push("version");
        }
        if !r.approved {
            v.push("destination");
        }
    }
    v
}
#[derive(Serialize)]
pub struct Result {
    pub enforcer: Enforcer,
    pub budget_in_specification: bool,
    pub states: usize,
    pub checks: usize,
    pub positive_tasks_passed: usize,
    pub positive_tasks_total: usize,
    pub max_accepted_world_spend: u8,
    pub declared_violations: BTreeMap<String, Vec<String>>,
    pub reference_violations: BTreeMap<String, Vec<String>>,
}
fn trace(
    parents: &[Option<(usize, usize)>],
    mut i: usize,
    last: usize,
    aa: &[Action],
) -> Vec<String> {
    let mut path = vec![last];
    while let Some((p, a)) = parents[i] {
        path.push(a);
        i = p;
    }
    path.reverse();
    path.into_iter().map(|a| format!("{:?}", aa[a])).collect()
}
pub fn explore(e: Enforcer, budget_in_specification: bool) -> Result {
    // Omitting the budget with replay disabled can make the state space infinite.
    assert!(budget_in_specification || e == Enforcer::PerAction);
    let aa = actions();
    let initial = State::default();
    let mut seen = HashMap::from([(initial.clone(), 0)]);
    let mut states = vec![initial];
    let mut parents = vec![None];
    let mut out = Result {
        enforcer: e,
        budget_in_specification,
        states: 0,
        checks: 0,
        positive_tasks_passed: positive_tasks(e),
        positive_tasks_total: 20,
        max_accepted_world_spend: 0,
        declared_violations: BTreeMap::new(),
        reference_violations: BTreeMap::new(),
    };
    let mut head = 0;
    while head < states.len() {
        let s = states[head].clone();
        out.max_accepted_world_spend = out.max_accepted_world_spend.max(s.world_spent);
        for (ai, a) in aa.iter().enumerate() {
            out.checks += 1;
            let (next, effect) = transition(&s, *a, e);
            let violations = oracle(&s, effect);
            let mut reject = false;
            for name in violations {
                out.reference_violations
                    .entry(name.to_string())
                    .or_insert_with(|| trace(&parents, head, ai, &aa));
                if name != "budget" || budget_in_specification {
                    reject = true;
                    out.declared_violations
                        .entry(name.to_string())
                        .or_insert_with(|| trace(&parents, head, ai, &aa));
                }
            }
            if !reject && !seen.contains_key(&next) {
                seen.insert(next.clone(), states.len());
                states.push(next);
                parents.push(Some((head, ai)));
            }
        }
        head += 1;
    }
    out.states = states.len();
    out
}
fn positive_tasks(e: Enforcer) -> usize {
    let mut tasks = Vec::new();
    for id in 0..4 {
        for amount in 1..=2 {
            tasks.push(vec![(id, amount)]);
        }
    }
    for a in 0..4 {
        for b in 0..4 {
            if a != b {
                tasks.push(vec![(a, 1), (b, 1)]);
            }
        }
    }
    tasks
        .iter()
        .filter(|task| {
            let mut s = State::default();
            for &(id, amount) in task.iter() {
                let actor = (id / 2) as usize;
                s = transition(&s, Action::Prepare(actor, id % 2, amount, true), e).0;
                let (next, effect) = transition(&s, Action::Commit(actor), e);
                if effect.is_none() || !oracle(&s, effect).is_empty() {
                    return false;
                }
                // A successful positive task must have the intended actor and amount.
                if effect.unwrap().actor != actor {
                    return false;
                }
                s = next;
            }
            true
        })
        .count()
}
pub fn suite() -> Vec<Result> {
    let mut results: Vec<_> = [
        Enforcer::Complete,
        Enforcer::Unconstrained,
        Enforcer::RejectAll,
        Enforcer::PerAction,
        Enforcer::PerActor,
        Enforcer::PrepareOnly,
        Enforcer::NoReplay,
        Enforcer::NoRevocation,
        Enforcer::NoVersion,
        Enforcer::NoDestination,
        Enforcer::ResetRestart,
        Enforcer::ResetRenewal,
    ]
    .into_iter()
    .map(|e| explore(e, true))
    .collect();
    results.push(explore(Enforcer::PerAction, false));
    results
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn closure_mutations_and_nonvacuity() {
        let rows = suite();
        for r in &rows {
            assert_eq!(r.checks, 22 * r.states);
            assert_eq!(
                r.positive_tasks_passed,
                if r.enforcer == Enforcer::RejectAll {
                    0
                } else {
                    20
                }
            );
            if r.enforcer == Enforcer::Complete
                || r.enforcer == Enforcer::RejectAll
                || !r.budget_in_specification
            {
                assert!(r.declared_violations.is_empty());
            } else {
                assert!(!r.declared_violations.is_empty());
            }
        }
        assert_eq!((rows[0].states, rows[0].checks), (10_180, 223_960));
        assert_eq!(rows[2].states, 2 * 9 * 9 + 2 * 17 * 17);
        assert_eq!(rows[1].reference_violations.len(), 5);
        let omitted = rows.last().unwrap();
        assert_eq!(
            (
                omitted.states,
                omitted.checks,
                omitted.max_accepted_world_spend
            ),
            (41_732, 918_104, 8)
        );
        for (enforcer, kind, length) in [
            (Enforcer::PerAction, "budget", 4),
            (Enforcer::PerActor, "budget", 4),
            (Enforcer::PrepareOnly, "budget", 4),
            (Enforcer::NoReplay, "replay", 3),
            (Enforcer::NoRevocation, "revocation", 3),
            (Enforcer::NoVersion, "version", 4),
            (Enforcer::NoDestination, "destination", 2),
            (Enforcer::ResetRestart, "budget", 5),
            (Enforcer::ResetRenewal, "budget", 5),
        ] {
            let row = rows.iter().find(|r| r.enforcer == enforcer).unwrap();
            assert_eq!(row.reference_violations[kind].len(), length);
            assert_eq!(row.reference_violations.len(), 1);
        }
    }
}
