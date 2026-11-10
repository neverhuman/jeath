//! Deterministic executable decisions. Parsing, durable snapshots, identity and
//! normalized-effect construction are caller obligations, not verified here.
// Keep the verification interface flat and use Verus's Vec view specifications.
#![allow(clippy::ptr_arg, clippy::too_many_arguments)]
use vstd::prelude::*;

verus! {
pub fn same_bytes(a: &Vec<u8>, b: &Vec<u8>) -> (ok: bool)
    ensures ok == (a@ == b@),
{
    if a.len() != b.len() { return false; }
    let mut i = 0;
    while i < a.len()
        invariant a.len() == b.len(), i <= a.len(),
            forall|j: int| 0 <= j < i ==> a@[j] == b@[j],
        decreases a.len() - i,
    {
        if a[i] != b[i] { return false; }
        i += 1;
    }
    assert(a@ =~= b@);
    true
}

/// An arbitrary finite issuer vector, not a fixed actor population.
pub fn grants_ok(required: &Vec<bool>, granted: &Vec<bool>) -> (ok: bool)
    ensures ok == (required.len() == granted.len() &&
        forall|i: int| 0 <= i < required.len() ==> (required@[i] ==> granted@[i])),
{
    if required.len() != granted.len() { return false; }
    let mut i = 0;
    while i < required.len()
        invariant required.len() == granted.len(), i <= required.len(),
            forall|j: int| 0 <= j < i ==> (required@[j] ==> granted@[j]),
        decreases required.len() - i,
    {
        if required[i] && !granted[i] { return false; }
        i += 1;
    }
    true
}

pub fn current_grants(required: &Vec<bool>, grants: &Vec<bool>, live: &Vec<bool>,
    epochs: &Vec<u64>, requested: &Vec<u64>) -> (ok: bool)
    ensures ok == (required.len() == grants.len() && required.len() == live.len()
        && required.len() == epochs.len() && required.len() == requested.len()
        && forall|i: int| 0 <= i < required.len() ==> (required@[i] ==>
            grants@[i] && live@[i] && epochs@[i] == requested@[i])),
{
    if required.len() != grants.len() || required.len() != live.len()
        || required.len() != epochs.len() || required.len() != requested.len() { return false; }
    let mut i = 0;
    while i < required.len()
        invariant required.len() == grants.len(), required.len() == live.len(),
            required.len() == epochs.len(), required.len() == requested.len(), i <= required.len(),
            forall|j: int| 0 <= j < i ==> (required@[j] ==>
                grants@[j] && live@[j] && epochs@[j] == requested@[j]),
        decreases required.len() - i,
    {
        if required[i] && (!grants[i] || !live[i] || epochs[i] != requested[i]) { return false; }
        i += 1;
    }
    true
}

pub fn revision_matches(current: u64, expected: u64) -> (ok: bool)
    ensures ok == (current == expected),
{ current == expected }

pub fn reserve(budget: u64, charged: u64, amount: u64) -> (next: Option<u64>)
    ensures
        next.is_some() == (amount > 0 && amount <= budget && charged <= budget - amount),
        next.is_some() ==> next.unwrap() == charged + amount,
        next.is_some() ==> next.unwrap() <= budget,
{
    if amount == 0 || amount > budget || charged > budget - amount { None }
    else { Some(charged + amount) }
}

pub fn admit(budget: u64, charged: u64, amount: u64, live: bool,
    epoch: u64, requested_epoch: u64, now: u64, expires: u64,
    revision: u64, expected_revision: u64, payload: &Vec<u8>, approved: &Vec<u8>,
    required: &Vec<bool>, granted: &Vec<bool>) -> (ok: bool)
    ensures ok == (amount > 0 && amount <= budget && charged <= budget - amount
        && live && epoch == requested_epoch && now < expires
        && revision == expected_revision && payload@ == approved@
        && required.len() == granted.len()
        && forall|i: int| 0 <= i < required.len() ==> (required@[i] ==> granted@[i])),
{
    reserve(budget, charged, amount).is_some() && live && epoch == requested_epoch
        && now < expires && revision == expected_revision
        && same_bytes(payload, approved) && grants_ok(required, granted)
}

/// Sink epochs are durable lower bounds, independently for every required issuer.
pub fn unfenced(required: &Vec<bool>, epochs: &Vec<u64>, floors: &Vec<u64>) -> (ok: bool)
    ensures ok == (required.len() == epochs.len() && required.len() == floors.len()
        && forall|i: int| 0 <= i < required.len() ==> (required@[i] ==> epochs@[i] >= floors@[i])),
{
    if required.len() != epochs.len() || required.len() != floors.len() { return false; }
    let mut i = 0;
    while i < required.len()
        invariant required.len() == epochs.len(), required.len() == floors.len(),
            i <= required.len(),
            forall|j: int| 0 <= j < i ==> (required@[j] ==> epochs@[j] >= floors@[j]),
        decreases required.len() - i,
    {
        if required[i] && epochs[i] < floors[i] { return false; }
        i += 1;
    }
    true
}

/// Renewal and recovery cannot mint budget. No timeout input exists.
pub fn release(charged: u64, amount: u64, authoritative_no_effect: bool) -> (next: u64)
    ensures next <= charged,
        !authoritative_no_effect ==> next == charged,
        authoritative_no_effect && amount <= charged ==> next == charged - amount,
{
    if authoritative_no_effect && amount <= charged { charged - amount } else { charged }
}

pub open spec fn sum(xs: Seq<nat>) -> nat
    decreases xs.len(),
{
    if xs.len() == 0 { 0 } else { xs.last() + sum(xs.drop_last()) }
}

/// Outstanding-effect exposure: any subset of reserved nonnegative charges is
/// bounded by all outstanding charges. Actor identities do not occur in the rule.
pub proof fn exposure(charges: Seq<nat>, effects: Seq<nat>)
    requires charges.len() == effects.len(),
        forall|i: int| 0 <= i < charges.len() ==> effects[i] <= charges[i],
    ensures sum(effects) <= sum(charges),
    decreases charges.len(),
{
    if charges.len() > 0 {
        exposure(charges.drop_last(), effects.drop_last());
    }
}

/// Induction over arbitrary finite execution prefixes. A reservation adds at
/// most remaining capacity; settlement/recovery/renewal do not add charges.
pub proof fn prefix_bound(states: Seq<int>, deltas: Seq<int>, budget: int)
    requires states.len() == deltas.len() + 1, 0 <= states[0] <= budget,
        forall|i: int| 0 <= i < deltas.len() ==>
            (#[trigger] states[i + 1] == states[i] + deltas[i]
            && (deltas[i] > 0 ==> deltas[i] <= budget - states[i])
            && (deltas[i] <= 0 ==> -deltas[i] <= states[i])),
    ensures 0 <= states.last() <= budget,
    decreases deltas.len(),
{
    if deltas.len() > 0 {
        prefix_bound(states.drop_last(), deltas.drop_last(), budget);
        let i = deltas.len() - 1;
        assert(states[i + 1] == states[i] + deltas[i]);
    }
}

pub proof fn fenced_zero(required: Seq<bool>, epochs: Seq<u64>, floors: Seq<u64>, issuer: int)
    requires required.len() == epochs.len(), required.len() == floors.len(),
        0 <= issuer < required.len(), required[issuer], epochs[issuer] < floors[issuer],
    ensures !(forall|i: int| 0 <= i < required.len() ==> (required[i] ==> epochs[i] >= floors[i])),
{
}

/// Positive permission theorem; the executable predicate is an equivalence.
pub proof fn positive_budget(budget: u64, charged: u64, amount: u64)
    requires amount > 0, amount <= budget, charged <= budget - amount,
    ensures charged + amount <= budget,
{
}
}
