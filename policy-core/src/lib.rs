//! Deterministic executable decisions. Parsing, durable snapshots, identity and
//! normalized-effect construction are caller obligations, not verified here.
// Keep the verification interface flat and use Verus's Vec view specifications.
#![allow(clippy::ptr_arg, clippy::too_many_arguments)]
use vstd::prelude::*;

verus! {

pub fn same_bytes(a: &Vec<u8>, b: &Vec<u8>) -> (ok: bool)
    ensures
        ok == (a@ == b@),
{
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len()
        invariant
            a.len() == b.len(),
            i <= a.len(),
            forall|j: int| 0 <= j < i ==> #[trigger] a@[j] == b@[j],
        decreases a.len() - i,
    {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    assert(a@ =~= b@);
    true
}

/// An arbitrary finite issuer vector, not a fixed actor population.
pub fn grants_ok(required: &Vec<bool>, granted: &Vec<bool>) -> (ok: bool)
    ensures
        ok == (required.len() == granted.len() && forall|i: int|
            0 <= i < required.len() ==> ((#[trigger] required@[i]) ==> granted@[i])),
{
    if required.len() != granted.len() {
        return false;
    }
    let mut i = 0;
    while i < required.len()
        invariant
            required.len() == granted.len(),
            i <= required.len(),
            forall|j: int| 0 <= j < i ==> ((#[trigger] required@[j]) ==> granted@[j]),
        decreases required.len() - i,
    {
        if required[i] && !granted[i] {
            return false;
        }
        i += 1;
    }
    true
}

pub fn current_grants(
    required: &Vec<bool>,
    grants: &Vec<bool>,
    live: &Vec<bool>,
    epochs: &Vec<u64>,
    requested: &Vec<u64>,
) -> (ok: bool)
    ensures
        ok == (required.len() == grants.len()
            && required.len() == live.len()
            && required.len() == epochs.len()
            && required.len() == requested.len()
            && forall|i: int|
                0 <= i < required.len() ==> ((#[trigger] required@[i]) ==> grants@[i] && live@[i]
                    && epochs@[i] == requested@[i])),
{
    if required.len() != grants.len()
        || required.len() != live.len()
        || required.len() != epochs.len()
        || required.len() != requested.len()
    {
        return false;
    }
    let mut i = 0;
    while i < required.len()
        invariant
            required.len() == grants.len(),
            required.len() == live.len(),
            required.len() == epochs.len(),
            required.len() == requested.len(),
            i <= required.len(),
            forall|j: int|
                0 <= j < i ==> ((#[trigger] required@[j]) ==> grants@[j] && live@[j] && epochs@[j]
                    == requested@[j]),
        decreases required.len() - i,
    {
        if required[i] && (!grants[i] || !live[i] || epochs[i] != requested[i]) {
            return false;
        }
        i += 1;
    }
    true
}

pub fn revision_matches(current: u64, expected: u64) -> (ok: bool)
    ensures
        ok == (current == expected),
{
    current == expected
}

pub fn reserve(budget: u64, charged: u64, amount: u64) -> (next: Option<u64>)
    ensures
        next.is_some() == (amount > 0 && amount <= budget && charged <= budget - amount),
        next.is_some() ==> next.unwrap() == charged + amount,
        next.is_some() ==> next.unwrap() <= budget,
{
    if amount == 0 || amount > budget || charged > budget - amount {
        None
    } else {
        Some(charged + amount)
    }
}

pub fn admit(
    budget: u64,
    charged: u64,
    amount: u64,
    live: bool,
    epoch: u64,
    requested_epoch: u64,
    now: u64,
    expires: u64,
    revision: u64,
    expected_revision: u64,
    payload: &Vec<u8>,
    approved: &Vec<u8>,
    required: &Vec<bool>,
    granted: &Vec<bool>,
) -> (ok: bool)
    ensures
        ok == (amount > 0
            && amount <= budget
            && charged <= budget - amount
            && live
            && epoch == requested_epoch
            && now < expires
            && revision == expected_revision
            && payload@ == approved@
            && required.len() == granted.len()
            && forall|i: int|
                0 <= i < required.len() ==> ((#[trigger] required@[i]) ==> granted@[i])),
{
    reserve(budget, charged, amount).is_some()
        && live
        && epoch == requested_epoch
        && now < expires
        && revision == expected_revision
        && same_bytes(payload, approved)
        && grants_ok(required, granted)
}

/// Multi-issuer admission predicate over arbitrary finite issuer populations.
/// Total defensive checks: length mismatches or overflow attempts safely return false.
pub fn admit_multi(
    budget: u64,
    charged: u64,
    amount: u64,
    now: u64,
    expires: u64,
    payload: &Vec<u8>,
    approved: &Vec<u8>,
    required: &Vec<bool>,
    grants: &Vec<bool>,
    live: &Vec<bool>,
    epochs: &Vec<u64>,
    requested: &Vec<u64>,
) -> (ok: bool)
    ensures
        ok == (amount > 0
            && amount <= budget
            && charged <= budget - amount
            && now < expires
            && payload@ == approved@
            && required.len() == grants.len()
            && required.len() == live.len()
            && required.len() == epochs.len()
            && required.len() == requested.len()
            && forall|i: int|
                0 <= i < required.len() ==> ((#[trigger] required@[i]) ==> grants@[i] && live@[i]
                    && epochs@[i] == requested@[i])),
{
    reserve(budget, charged, amount).is_some()
        && now < expires
        && same_bytes(payload, approved)
        && current_grants(required, grants, live, epochs, requested)
}

/// Conservative admission under state-uncertainty interval [charged_lower, charged_upper].
/// Inverted intervals or length/budget mismatches safely return false.
pub fn admit_uncertain(
    budget: u64,
    charged_lower: u64,
    charged_upper: u64,
    amount: u64,
    now: u64,
    expires: u64,
    payload: &Vec<u8>,
    approved: &Vec<u8>,
    required: &Vec<bool>,
    grants: &Vec<bool>,
    live: &Vec<bool>,
    epochs: &Vec<u64>,
    requested: &Vec<u64>,
) -> (ok: bool)
    ensures
        ok == (charged_lower <= charged_upper
            && amount > 0
            && amount <= budget
            && charged_upper <= budget - amount
            && now < expires
            && payload@ == approved@
            && required.len() == grants.len()
            && required.len() == live.len()
            && required.len() == epochs.len()
            && required.len() == requested.len()
            && forall|i: int|
                0 <= i < required.len() ==> ((#[trigger] required@[i]) ==> grants@[i] && live@[i]
                    && epochs@[i] == requested@[i])),
{
    charged_lower <= charged_upper
        && admit_multi(
            budget,
            charged_upper,
            amount,
            now,
            expires,
            payload,
            approved,
            required,
            grants,
            live,
            epochs,
            requested,
        )
}

/// Monotone sink epoch floor update.
pub fn advance_floor(current: u64, proposed: u64) -> (next: u64)
    ensures
        next == if proposed > current { proposed } else { current },
        next >= current,
        next >= proposed,
{
    if proposed > current {
        proposed
    } else {
        current
    }
}

/// Sub-delegation capability attenuation check: child cannot expand budget, expiry,
/// allowed target scope, or drop any required issuer. Length mismatches return false.
pub fn attenuates(
    parent_budget: u64,
    child_budget: u64,
    parent_expires: u64,
    child_expires: u64,
    parent_scope: &Vec<bool>,
    child_scope: &Vec<bool>,
    parent_required: &Vec<bool>,
    child_required: &Vec<bool>,
) -> (ok: bool)
    ensures
        ok == (child_budget <= parent_budget
            && child_expires <= parent_expires
            && parent_scope.len() == child_scope.len()
            && parent_required.len() == child_required.len()
            && (forall|i: int|
                0 <= i < parent_scope.len() ==> ((#[trigger] child_scope@[i]) ==> parent_scope@[i]))
            && (forall|i: int|
                0 <= i < parent_required.len() ==> ((#[trigger] parent_required@[i])
                    ==> child_required@[i]))),
{
    child_budget <= parent_budget
        && child_expires <= parent_expires
        && grants_ok(child_scope, parent_scope)
        && grants_ok(parent_required, child_required)
}

/// Sink epochs are durable lower bounds, independently for every required issuer.
pub fn unfenced(required: &Vec<bool>, epochs: &Vec<u64>, floors: &Vec<u64>) -> (ok: bool)
    ensures
        ok == (required.len() == epochs.len()
            && required.len() == floors.len()
            && forall|i: int|
                0 <= i < required.len() ==> ((#[trigger] required@[i]) ==> epochs@[i]
                    >= floors@[i])),
{
    if required.len() != epochs.len() || required.len() != floors.len() {
        return false;
    }
    let mut i = 0;
    while i < required.len()
        invariant
            required.len() == epochs.len(),
            required.len() == floors.len(),
            i <= required.len(),
            forall|j: int|
                0 <= j < i ==> ((#[trigger] required@[j]) ==> epochs@[j] >= floors@[j]),
        decreases required.len() - i,
    {
        if required[i] && epochs[i] < floors[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Renewal and recovery cannot mint budget. No timeout input exists.
pub fn release(charged: u64, amount: u64, authoritative_no_effect: bool) -> (next: u64)
    ensures
        next <= charged,
        !authoritative_no_effect ==> next == charged,
        authoritative_no_effect && amount <= charged ==> next == charged - amount,
        authoritative_no_effect && amount > charged ==> next == charged,
{
    if authoritative_no_effect && amount <= charged {
        charged - amount
    } else {
        charged
    }
}

pub open spec fn sum(xs: Seq<nat>) -> nat
    decreases xs.len(),
{
    if xs.len() == 0 {
        0
    } else {
        xs.last() + sum(xs.drop_last())
    }
}

/// Outstanding-effect exposure: any subset of reserved nonnegative charges is
/// bounded by all outstanding charges. Actor identities do not occur in the rule.
pub proof fn exposure(charges: Seq<nat>, effects: Seq<nat>)
    requires
        charges.len() == effects.len(),
        forall|i: int| 0 <= i < charges.len() ==> #[trigger] effects[i] <= charges[i],
    ensures
        sum(effects) <= sum(charges),
    decreases charges.len(),
{
    if charges.len() > 0 {
        exposure(charges.drop_last(), effects.drop_last());
    }
}

/// Induction over arbitrary finite execution prefixes. A reservation adds at
/// most remaining capacity; settlement/recovery/renewal do not add charges.
pub proof fn prefix_bound(states: Seq<int>, deltas: Seq<int>, budget: int)
    requires
        states.len() == deltas.len() + 1,
        0 <= states[0] <= budget,
        forall|i: int|
            0 <= i < deltas.len() ==> (#[trigger] states[i + 1] == states[i] + deltas[i]
                && (deltas[i] > 0 ==> deltas[i] <= budget - states[i])
                && (deltas[i] <= 0 ==> -deltas[i] <= states[i])),
    ensures
        0 <= states.last() <= budget,
    decreases deltas.len(),
{
    if deltas.len() > 0 {
        prefix_bound(states.drop_last(), deltas.drop_last(), budget);
        let i = deltas.len() - 1;
        assert(states[i + 1] == states[i] + deltas[i]);
    }
}

pub proof fn fenced_zero(required: Seq<bool>, epochs: Seq<u64>, floors: Seq<u64>, issuer: int)
    requires
        required.len() == epochs.len(),
        required.len() == floors.len(),
        0 <= issuer < required.len(),
        required[issuer],
        epochs[issuer] < floors[issuer],
    ensures
        !(forall|i: int|
            0 <= i < required.len() ==> ((#[trigger] required[i]) ==> epochs[i] >= floors[i])),
{
}

/// Positive permission theorem; the executable predicate is an equivalence.
pub proof fn positive_budget(budget: u64, charged: u64, amount: u64)
    requires
        amount > 0,
        amount <= budget,
        charged <= budget - amount,
    ensures
        charged + amount <= budget,
{
}

// ============================================================================
// H1: Cumulative Ledger Conservation & Lifecycle/Timeout Vulnerability Proofs
// ============================================================================

pub struct LedgerState {
    pub budget: nat,
    pub charged: nat,
    pub committed: nat,
    pub inflight: Seq<nat>,
}

pub enum LedgerStep {
    Reserve { amount: nat },
    CommitSettle { slot: int },
    CancelRelease { slot: int },
    UncertainTimeout,
    Withdraw,
    Renew,
    Restart,
}

pub open spec fn ledger_inv(s: LedgerState) -> bool {
    s.charged == s.committed + sum(s.inflight) && s.charged <= s.budget
}

pub open spec fn step_ledger(s: LedgerState, step: LedgerStep) -> LedgerState {
    match step {
        LedgerStep::Reserve { amount } => {
            if amount > 0 && amount <= s.budget && s.charged + amount <= s.budget {
                LedgerState {
                    charged: s.charged + amount,
                    inflight: s.inflight.push(amount),
                    ..s
                }
            } else {
                s
            }
        }
        LedgerStep::CommitSettle { slot } => {
            if 0 <= slot < s.inflight.len() {
                LedgerState {
                    committed: s.committed + s.inflight[slot],
                    inflight: s.inflight.remove(slot),
                    ..s
                }
            } else {
                s
            }
        }
        LedgerStep::CancelRelease { slot } => {
            if 0 <= slot < s.inflight.len() && s.inflight[slot] <= s.charged {
                LedgerState {
                    charged: (s.charged - s.inflight[slot]) as nat,
                    inflight: s.inflight.remove(slot),
                    ..s
                }
            } else {
                s
            }
        }
        LedgerStep::UncertainTimeout
        | LedgerStep::Withdraw
        | LedgerStep::Renew
        | LedgerStep::Restart => s,
    }
}

pub open spec fn run_ledger(s: LedgerState, trace: Seq<LedgerStep>) -> LedgerState
    decreases trace.len(),
{
    if trace.len() == 0 {
        s
    } else {
        step_ledger(run_ledger(s, trace.drop_last()), trace.last())
    }
}

pub proof fn sum_push(xs: Seq<nat>, v: nat)
    ensures
        sum(xs.push(v)) == sum(xs) + v,
{
    assert(xs.push(v).drop_last() =~= xs);
}

pub proof fn sum_remove(xs: Seq<nat>, slot: int)
    requires
        0 <= slot < xs.len(),
    ensures
        sum(xs) == xs[slot] + sum(xs.remove(slot)),
    decreases xs.len(),
{
    if slot == xs.len() - 1 {
        assert(xs.remove(slot) =~= xs.drop_last());
    } else {
        sum_remove(xs.drop_last(), slot);
        assert(xs.remove(slot).last() == xs.last());
        assert(xs.remove(slot).drop_last() =~= xs.drop_last().remove(slot));
    }
}

pub proof fn ledger_step_preserves_inv(s: LedgerState, step: LedgerStep)
    requires
        ledger_inv(s),
    ensures
        ledger_inv(step_ledger(s, step)),
        step_ledger(s, step).budget == s.budget,
        step_ledger(s, step).committed >= s.committed,
{
    match step {
        LedgerStep::Reserve { amount } => {
            if amount > 0 && amount <= s.budget && s.charged + amount <= s.budget {
                sum_push(s.inflight, amount);
            }
        }
        LedgerStep::CommitSettle { slot } => {
            if 0 <= slot < s.inflight.len() {
                sum_remove(s.inflight, slot);
            }
        }
        LedgerStep::CancelRelease { slot } => {
            if 0 <= slot < s.inflight.len() {
                sum_remove(s.inflight, slot);
            }
        }
        LedgerStep::UncertainTimeout
        | LedgerStep::Withdraw
        | LedgerStep::Renew
        | LedgerStep::Restart => {}
    }
}

/// H1: Inductive spend + reservation conservation across arbitrary-length traces of
/// reserve, commit, cancel, uncertain timeout, withdraw, renew, and restart steps.
pub proof fn cumulative_conservation_induction(init: LedgerState, trace: Seq<LedgerStep>)
    requires
        ledger_inv(init),
    ensures
        ledger_inv(run_ledger(init, trace)),
        run_ledger(init, trace).budget == init.budget,
        run_ledger(init, trace).committed + sum(run_ledger(init, trace).inflight)
            == run_ledger(init, trace).charged,
        run_ledger(init, trace).committed <= run_ledger(init, trace).charged,
        run_ledger(init, trace).charged <= init.budget,
    decreases trace.len(),
{
    if trace.len() > 0 {
        cumulative_conservation_induction(init, trace.drop_last());
        ledger_step_preserves_inv(run_ledger(init, trace.drop_last()), trace.last());
    }
}

/// Faulty lifecycle transition that resets `charged` and `inflight` on renewal or restart
/// while committed external effects persist in the world.
pub open spec fn step_ledger_reset_on_lifecycle(s: LedgerState, step: LedgerStep) -> LedgerState {
    match step {
        LedgerStep::Renew | LedgerStep::Restart => {
            LedgerState { charged: 0, inflight: Seq::empty(), ..s }
        }
        _ => step_ledger(s, step),
    }
}

pub open spec fn run_ledger_reset_on_lifecycle(
    s: LedgerState,
    trace: Seq<LedgerStep>,
) -> LedgerState
    decreases trace.len(),
{
    if trace.len() == 0 {
        s
    } else {
        step_ledger_reset_on_lifecycle(
            run_ledger_reset_on_lifecycle(s, trace.drop_last()),
            trace.last(),
        )
    }
}

/// H1 contrast: resetting ledger charge on renewal or restart permits 2 * budget > budget
/// committed spend, whereas persistent `run_ledger` caps committed spend at `budget`.
pub proof fn reset_vulnerability_contrast(budget: nat)
    requires
        budget > 0,
    ensures
        ({
            let init = LedgerState {
                budget,
                charged: 0,
                committed: 0,
                inflight: Seq::empty(),
            };
            let renew_trace = seq![
                LedgerStep::Reserve { amount: budget },
                LedgerStep::CommitSettle { slot: 0 },
                LedgerStep::Renew,
                LedgerStep::Reserve { amount: budget },
                LedgerStep::CommitSettle { slot: 0 },
            ];
            let restart_trace = seq![
                LedgerStep::Reserve { amount: budget },
                LedgerStep::CommitSettle { slot: 0 },
                LedgerStep::Restart,
                LedgerStep::Reserve { amount: budget },
                LedgerStep::CommitSettle { slot: 0 },
            ];
            run_ledger_reset_on_lifecycle(init, renew_trace).committed == 2 * budget
                && run_ledger_reset_on_lifecycle(init, renew_trace).committed > budget
                && run_ledger_reset_on_lifecycle(init, restart_trace).committed == 2 * budget
                && run_ledger_reset_on_lifecycle(init, restart_trace).committed > budget
                && run_ledger(init, renew_trace).committed == budget
                && run_ledger(init, restart_trace).committed == budget
        }),
{
    reveal_with_fuel(run_ledger_reset_on_lifecycle, 6);
    reveal_with_fuel(run_ledger, 6);

    let init = LedgerState { budget, charged: 0, committed: 0, inflight: Seq::empty() };
    let r = LedgerStep::Reserve { amount: budget };
    let c = LedgerStep::CommitSettle { slot: 0 };

    let renew_trace = seq![r, c, LedgerStep::Renew, r, c];
    assert(renew_trace.drop_last() =~= seq![r, c, LedgerStep::Renew, r]);
    assert(renew_trace.drop_last().drop_last() =~= seq![r, c, LedgerStep::Renew]);
    assert(renew_trace.drop_last().drop_last().drop_last() =~= seq![r, c]);
    assert(renew_trace.drop_last().drop_last().drop_last().drop_last() =~= seq![r]);
    assert(renew_trace.drop_last().drop_last().drop_last().drop_last().drop_last()
        =~= Seq::<LedgerStep>::empty());

    let restart_trace = seq![r, c, LedgerStep::Restart, r, c];
    assert(restart_trace.drop_last() =~= seq![r, c, LedgerStep::Restart, r]);
    assert(restart_trace.drop_last().drop_last() =~= seq![r, c, LedgerStep::Restart]);
    assert(restart_trace.drop_last().drop_last().drop_last() =~= seq![r, c]);
    assert(restart_trace.drop_last().drop_last().drop_last().drop_last() =~= seq![r]);
    assert(restart_trace.drop_last().drop_last().drop_last().drop_last().drop_last()
        =~= Seq::<LedgerStep>::empty());

    assert(Seq::<nat>::empty().push(budget).remove(0) =~= Seq::<nat>::empty());
}

/// Faulty timeout handler that refunds `inflight[0]` on an unacknowledged timeout
/// while the permit remains live in `inflight` at the sink.
pub open spec fn step_ledger_refund_timeout(s: LedgerState, step: LedgerStep) -> LedgerState {
    match step {
        LedgerStep::UncertainTimeout => {
            if s.inflight.len() > 0 && s.inflight[0] <= s.charged {
                LedgerState { charged: (s.charged - s.inflight[0]) as nat, ..s }
            } else {
                s
            }
        }
        _ => step_ledger(s, step),
    }
}

pub open spec fn run_ledger_refund_timeout(s: LedgerState, trace: Seq<LedgerStep>) -> LedgerState
    decreases trace.len(),
{
    if trace.len() == 0 {
        s
    } else {
        step_ledger_refund_timeout(
            run_ledger_refund_timeout(s, trace.drop_last()),
            trace.last(),
        )
    }
}

/// H1 contrast: refunding an uncertain timeout (`release` without authoritative cancellation)
/// permits `budget + q > budget` committed spend when the timed-out permit commits at the sink.
pub proof fn timeout_refund_vulnerability(budget: nat, q: nat)
    requires
        0 < q <= budget,
    ensures
        ({
            let init = LedgerState {
                budget,
                charged: 0,
                committed: 0,
                inflight: Seq::empty(),
            };
            let trace = seq![
                LedgerStep::Reserve { amount: q },
                LedgerStep::UncertainTimeout,
                LedgerStep::Reserve { amount: budget },
                LedgerStep::CommitSettle { slot: 0 },
                LedgerStep::CommitSettle { slot: 0 },
            ];
            run_ledger_refund_timeout(init, trace).committed == budget + q
                && run_ledger_refund_timeout(init, trace).committed > budget
                && run_ledger(init, trace).committed <= budget
        }),
{
    reveal_with_fuel(run_ledger_refund_timeout, 6);

    let init = LedgerState { budget, charged: 0, committed: 0, inflight: Seq::empty() };
    let s0 = LedgerStep::Reserve { amount: q };
    let s1 = LedgerStep::UncertainTimeout;
    let s2 = LedgerStep::Reserve { amount: budget };
    let s3 = LedgerStep::CommitSettle { slot: 0 };
    let trace = seq![s0, s1, s2, s3, s3];

    assert(trace.drop_last() =~= seq![s0, s1, s2, s3]);
    assert(trace.drop_last().drop_last() =~= seq![s0, s1, s2]);
    assert(trace.drop_last().drop_last().drop_last() =~= seq![s0, s1]);
    assert(trace.drop_last().drop_last().drop_last().drop_last() =~= seq![s0]);
    assert(trace.drop_last().drop_last().drop_last().drop_last().drop_last()
        =~= Seq::<LedgerStep>::empty());

    let inf2 = Seq::<nat>::empty().push(q).push(budget);
    assert(inf2[0] == q);
    assert(inf2.remove(0) =~= Seq::<nat>::empty().push(budget));
    assert(Seq::<nat>::empty().push(budget)[0] == budget);
    assert(Seq::<nat>::empty().push(budget).remove(0) =~= Seq::<nat>::empty());

    cumulative_conservation_induction(init, trace);
}

// ============================================================================
// H2: State-Uncertainty Monotone Admissibility Contraction
// ============================================================================

pub open spec fn current_grants_spec(
    required: Seq<bool>,
    grants: Seq<bool>,
    live: Seq<bool>,
    epochs: Seq<u64>,
    requested: Seq<u64>,
) -> bool {
    required.len() == grants.len()
        && required.len() == live.len()
        && required.len() == epochs.len()
        && required.len() == requested.len()
        && forall|i: int|
            0 <= i < required.len() ==> ((#[trigger] required[i]) ==> grants[i] && live[i]
                && epochs[i] == requested[i])
}

pub open spec fn admit_multi_spec(
    budget: u64,
    charged: u64,
    amount: u64,
    now: u64,
    expires: u64,
    payload: Seq<u8>,
    approved: Seq<u8>,
    required: Seq<bool>,
    grants: Seq<bool>,
    live: Seq<bool>,
    epochs: Seq<u64>,
    requested: Seq<u64>,
) -> bool {
    amount > 0
        && amount <= budget
        && charged <= budget - amount
        && now < expires
        && payload == approved
        && current_grants_spec(required, grants, live, epochs, requested)
}

pub open spec fn admit_uncertain_spec(
    budget: u64,
    charged_lower: u64,
    charged_upper: u64,
    amount: u64,
    now: u64,
    expires: u64,
    payload: Seq<u8>,
    approved: Seq<u8>,
    required: Seq<bool>,
    grants: Seq<bool>,
    live: Seq<bool>,
    epochs: Seq<u64>,
    requested: Seq<u64>,
) -> bool {
    charged_lower <= charged_upper
        && admit_multi_spec(
            budget,
            charged_upper,
            amount,
            now,
            expires,
            payload,
            approved,
            required,
            grants,
            live,
            epochs,
            requested,
        )
}

/// H2: Interval containment B = [lo_tight, hi_tight] <= B' = [lo_wide, hi_wide]
/// monotonically contracts `admit_uncertain` (admissible under B' ==> admissible under B)
/// and guarantees exact budget conservation for every true state in [lo_tight, hi_tight].
pub proof fn uncertainty_monotonicity(
    budget: u64,
    lo_tight: u64,
    hi_tight: u64,
    lo_wide: u64,
    hi_wide: u64,
    true_charged: u64,
    amount: u64,
    now: u64,
    expires: u64,
    payload: Seq<u8>,
    approved: Seq<u8>,
    required: Seq<bool>,
    grants: Seq<bool>,
    live: Seq<bool>,
    epochs: Seq<u64>,
    requested: Seq<u64>,
)
    requires
        lo_wide <= lo_tight,
        lo_tight <= true_charged,
        true_charged <= hi_tight,
        hi_tight <= hi_wide,
    ensures
        admit_uncertain_spec(
            budget,
            lo_wide,
            hi_wide,
            amount,
            now,
            expires,
            payload,
            approved,
            required,
            grants,
            live,
            epochs,
            requested,
        ) ==> admit_uncertain_spec(
            budget,
            lo_tight,
            hi_tight,
            amount,
            now,
            expires,
            payload,
            approved,
            required,
            grants,
            live,
            epochs,
            requested,
        ),
        admit_uncertain_spec(
            budget,
            lo_tight,
            hi_tight,
            amount,
            now,
            expires,
            payload,
            approved,
            required,
            grants,
            live,
            epochs,
            requested,
        ) ==> (admit_multi_spec(
            budget,
            true_charged,
            amount,
            now,
            expires,
            payload,
            approved,
            required,
            grants,
            live,
            epochs,
            requested,
        ) && true_charged + amount <= budget),
{
}

// ============================================================================
// H3: Monotone Multi-Issuer Epoch Fencing & Independent Non-Interference
// ============================================================================

pub open spec fn advance_floor_spec(current: u64, proposed: u64) -> u64 {
    if proposed > current {
        proposed
    } else {
        current
    }
}

pub open spec fn unfenced_spec(required: Seq<bool>, epochs: Seq<u64>, floors: Seq<u64>) -> bool {
    required.len() == epochs.len()
        && required.len() == floors.len()
        && forall|i: int|
            0 <= i < required.len() ==> ((#[trigger] required[i]) ==> epochs[i] >= floors[i])
}

pub open spec fn apply_fence(floors: Seq<u64>, issuer: int, proposed: u64) -> Seq<u64> {
    if 0 <= issuer < floors.len() {
        floors.update(issuer, advance_floor_spec(floors[issuer], proposed))
    } else {
        floors
    }
}

pub open spec fn apply_fences(floors: Seq<u64>, fences: Seq<(int, u64)>) -> Seq<u64>
    decreases fences.len(),
{
    if fences.len() == 0 {
        floors
    } else {
        let prev = apply_fences(floors, fences.drop_last());
        let (issuer, proposed) = fences.last();
        apply_fence(prev, issuer, proposed)
    }
}

/// H3: Coordinatewise monotonicity of sink epoch floors across arbitrary fence sequences.
pub proof fn monotone_fencing_induction(floors: Seq<u64>, fences: Seq<(int, u64)>)
    ensures
        apply_fences(floors, fences).len() == floors.len(),
        forall|i: int| 0 <= i < floors.len() ==> #[trigger] apply_fences(floors, fences)[i] >= floors[i],
    decreases fences.len(),
{
    if fences.len() > 0 {
        monotone_fencing_induction(floors, fences.drop_last());
    }
}

/// H3: Once required issuer `issuer` fences above `permit_epochs[issuer]`, the permit
/// remains permanently excluded across all subsequent fence updates on any issuers.
pub proof fn permanent_post_fence_exclusion(
    required: Seq<bool>,
    permit_epochs: Seq<u64>,
    floors: Seq<u64>,
    issuer: int,
    fence_floor: u64,
    suffix: Seq<(int, u64)>,
)
    requires
        required.len() == permit_epochs.len(),
        required.len() == floors.len(),
        0 <= issuer < required.len(),
        required[issuer],
        permit_epochs[issuer] < fence_floor,
    ensures
        !unfenced_spec(
            required,
            permit_epochs,
            apply_fences(apply_fence(floors, issuer, fence_floor), suffix),
        ),
{
    let after_first = apply_fence(floors, issuer, fence_floor);
    monotone_fencing_induction(after_first, suffix);
    let final_floors = apply_fences(after_first, suffix);
    assert(after_first[issuer] >= fence_floor);
    assert(final_floors[issuer] >= after_first[issuer]);
    fenced_zero(required, permit_epochs, final_floors, issuer);
}

/// H3: Advancing floors, withdrawing liveness, changing grant bits, or bumping epochs
/// for any non-required issuer (`!required[issuer]`) has zero effect on `unfenced_spec`
/// or `current_grants_spec`.
pub proof fn independent_issuer_noninterference(
    required: Seq<bool>,
    grants: Seq<bool>,
    live: Seq<bool>,
    epochs: Seq<u64>,
    requested: Seq<u64>,
    floors: Seq<u64>,
    issuer: int,
    new_grant: bool,
    new_live: bool,
    new_epoch: u64,
    proposed_floor: u64,
)
    requires
        0 <= issuer < required.len(),
        !required[issuer],
    ensures
        unfenced_spec(required, requested, apply_fence(floors, issuer, proposed_floor))
            == unfenced_spec(required, requested, floors),
        (grants.len() == required.len()
            && live.len() == required.len()
            && epochs.len() == required.len()) ==> (current_grants_spec(
            required,
            grants.update(issuer, new_grant),
            live.update(issuer, new_live),
            epochs.update(issuer, new_epoch),
            requested,
        ) == current_grants_spec(required, grants, live, epochs, requested)),
{
    let next_floors = apply_fence(floors, issuer, proposed_floor);
    if unfenced_spec(required, requested, floors) {
        assert forall|i: int|
            0 <= i < required.len() && (#[trigger] required[i]) implies requested[i]
            >= next_floors[i] by {
            assert(i != issuer);
            assert(next_floors[i] == floors[i]);
        }
    }
    if unfenced_spec(required, requested, next_floors) {
        assert forall|i: int|
            0 <= i < required.len() && (#[trigger] required[i]) implies requested[i]
            >= floors[i] by {
            assert(i != issuer);
            assert(next_floors[i] == floors[i]);
        }
    }
    if grants.len() == required.len()
        && live.len() == required.len()
        && epochs.len() == required.len()
    {
        let g2 = grants.update(issuer, new_grant);
        let l2 = live.update(issuer, new_live);
        let e2 = epochs.update(issuer, new_epoch);
        if current_grants_spec(required, grants, live, epochs, requested) {
            assert forall|i: int|
                0 <= i < required.len() && (#[trigger] required[i]) implies g2[i] && l2[i]
                && e2[i] == requested[i] by {
                assert(i != issuer);
            }
        }
        if current_grants_spec(required, g2, l2, e2, requested) {
            assert forall|i: int|
                0 <= i < required.len() && (#[trigger] required[i]) implies grants[i] && live[i]
                && epochs[i] == requested[i] by {
                assert(i != issuer);
            }
        }
    }
}

// ============================================================================
// H4: Sub-Delegation Capability Attenuation & Chain Transitivity
// ============================================================================

pub struct CapSpec {
    pub budget: u64,
    pub expires: u64,
    pub scope: Seq<bool>,
    pub required: Seq<bool>,
}

pub open spec fn attenuates_spec(parent: CapSpec, child: CapSpec) -> bool {
    child.budget <= parent.budget
        && child.expires <= parent.expires
        && parent.scope.len() == child.scope.len()
        && parent.required.len() == child.required.len()
        && (forall|i: int|
            0 <= i < parent.scope.len() ==> ((#[trigger] child.scope[i]) ==> parent.scope[i]))
        && (forall|i: int|
            0 <= i < parent.required.len() ==> ((#[trigger] parent.required[i])
                ==> child.required[i]))
}

pub open spec fn valid_under_cap(
    cap: CapSpec,
    charged: u64,
    amount: u64,
    now: u64,
    target: int,
    grants: Seq<bool>,
    live: Seq<bool>,
    epochs: Seq<u64>,
    requested: Seq<u64>,
    floors: Seq<u64>,
) -> bool {
    amount > 0
        && amount <= cap.budget
        && charged <= cap.budget - amount
        && now < cap.expires
        && 0 <= target < cap.scope.len()
        && cap.scope[target]
        && current_grants_spec(cap.required, grants, live, epochs, requested)
        && unfenced_spec(cap.required, requested, floors)
}

/// H4: Any operation admitted and unfenced under an attenuated child capability
/// satisfies every budget, expiry, target-scope, broker-grant, and sink-fence
/// obligation of its parent (including ancestor withdrawal/fencing).
pub proof fn attenuation_monotonicity(
    parent: CapSpec,
    child: CapSpec,
    charged: u64,
    amount: u64,
    now: u64,
    target: int,
    grants: Seq<bool>,
    live: Seq<bool>,
    epochs: Seq<u64>,
    requested: Seq<u64>,
    floors: Seq<u64>,
)
    requires
        attenuates_spec(parent, child),
        valid_under_cap(
            child,
            charged,
            amount,
            now,
            target,
            grants,
            live,
            epochs,
            requested,
            floors,
        ),
    ensures
        valid_under_cap(
            parent,
            charged,
            amount,
            now,
            target,
            grants,
            live,
            epochs,
            requested,
            floors,
        ),
{
    assert(child.scope[target] ==> parent.scope[target]);
    assert forall|i: int|
        0 <= i < parent.required.len() && (#[trigger] parent.required[i]) implies grants[i]
        && live[i] && epochs[i] == requested[i] by {
        assert(child.required[i]);
    }
    assert forall|i: int|
        0 <= i < parent.required.len() && (#[trigger] parent.required[i]) implies requested[i]
        >= floors[i] by {
        assert(child.required[i]);
    }
}

pub proof fn attenuation_transitive(root: CapSpec, mid: CapSpec, leaf: CapSpec)
    requires
        attenuates_spec(root, mid),
        attenuates_spec(mid, leaf),
    ensures
        attenuates_spec(root, leaf),
{
    assert forall|i: int|
        0 <= i < root.scope.len() && (#[trigger] leaf.scope[i]) implies root.scope[i] by {
        assert(mid.scope[i]);
    }
    assert forall|i: int|
        0 <= i < root.required.len() && (#[trigger] root.required[i]) implies leaf.required[i] by {
        assert(mid.required[i]);
    }
}

pub open spec fn valid_delegation_chain(chain: Seq<CapSpec>) -> bool {
    chain.len() >= 1
        && forall|m: int|
            #![trigger chain[m]]
            0 <= m < chain.len() - 1 ==> attenuates_spec(chain[m], chain[m + 1])
}

/// H4: Transitivity of capability attenuation across arbitrary delegation chain depth.
pub proof fn attenuation_chain_induction(chain: Seq<CapSpec>)
    requires
        valid_delegation_chain(chain),
    ensures
        attenuates_spec(chain[0], chain.last()),
    decreases chain.len(),
{
    if chain.len() > 1 {
        let prefix = chain.drop_last();
        assert forall|m: int| #![trigger prefix[m]] 0 <= m < prefix.len() - 1 implies attenuates_spec(
            prefix[m],
            prefix[m + 1],
        ) by {
            assert(prefix[m] == chain[m]);
            assert(prefix[m + 1] == chain[m + 1]);
        }
        attenuation_chain_induction(prefix);
        let k = chain.len() - 2;
        assert(prefix.last() == chain[k]);
        assert(chain.last() == chain[k + 1]);
        assert(attenuates_spec(chain[k], chain[k + 1]));
        attenuation_transitive(chain[0], chain[k], chain.last());
    }
}

// ============================================================================
// H5: Proposer Disposition Independence & Ungated/Stateless Divergence
// ============================================================================

pub struct CandidateProposal {
    pub amount: nat,
    pub now: u64,
    pub expires: u64,
    pub payload_matches: bool,
    pub grants_current: bool,
}

pub struct RunOutcome {
    pub charged: nat,
    pub admitted_valid: nat,
    pub admitted_unauthorized: nat,
}

pub open spec fn proposal_authorized(budget: nat, charged: nat, p: CandidateProposal) -> bool {
    p.amount > 0
        && p.amount <= budget
        && charged + p.amount <= budget
        && p.now < p.expires
        && p.payload_matches
        && p.grants_current
}

pub open spec fn record_decision(
    budget: nat,
    prev: RunOutcome,
    p: CandidateProposal,
    admit_decision: bool,
) -> RunOutcome {
    if !admit_decision {
        prev
    } else if proposal_authorized(budget, prev.charged, p) {
        RunOutcome {
            charged: prev.charged + p.amount,
            admitted_valid: prev.admitted_valid + 1,
            admitted_unauthorized: prev.admitted_unauthorized,
        }
    } else {
        RunOutcome {
            charged: prev.charged + p.amount,
            admitted_valid: prev.admitted_valid,
            admitted_unauthorized: prev.admitted_unauthorized + 1,
        }
    }
}

pub open spec fn empty_outcome() -> RunOutcome {
    RunOutcome { charged: 0, admitted_valid: 0, admitted_unauthorized: 0 }
}

pub open spec fn gated_run(budget: nat, proposals: Seq<CandidateProposal>) -> RunOutcome
    decreases proposals.len(),
{
    if proposals.len() == 0 {
        empty_outcome()
    } else {
        let prev = gated_run(budget, proposals.drop_last());
        let p = proposals.last();
        record_decision(budget, prev, p, proposal_authorized(budget, prev.charged, p))
    }
}

pub open spec fn ungated_run(budget: nat, proposals: Seq<CandidateProposal>) -> RunOutcome
    decreases proposals.len(),
{
    if proposals.len() == 0 {
        empty_outcome()
    } else {
        let prev = ungated_run(budget, proposals.drop_last());
        let p = proposals.last();
        record_decision(budget, prev, p, true)
    }
}

pub open spec fn stateless_per_call_run(
    budget: nat,
    proposals: Seq<CandidateProposal>,
) -> RunOutcome
    decreases proposals.len(),
{
    if proposals.len() == 0 {
        empty_outcome()
    } else {
        let prev = stateless_per_call_run(budget, proposals.drop_last());
        let p = proposals.last();
        let stateless_ok = p.amount > 0
            && p.amount <= budget
            && p.now < p.expires
            && p.payload_matches
            && p.grants_current;
        record_decision(budget, prev, p, stateless_ok)
    }
}

/// H5: Across ANY finite proposal sequence (cooperative, guardian, warrior, replay,
/// or adaptive), the core gate admits zero unauthorized proposals and bounds total
/// admitted charge at `budget`.
pub proof fn proposer_disposition_independence(budget: nat, proposals: Seq<CandidateProposal>)
    ensures
        gated_run(budget, proposals).admitted_unauthorized == 0,
        gated_run(budget, proposals).charged <= budget,
    decreases proposals.len(),
{
    if proposals.len() > 0 {
        proposer_disposition_independence(budget, proposals.drop_last());
    }
}

/// H5 contrast: without stateful external gating, proposer disposition determines safety:
/// both ungated execution and stateless per-call checking admit unauthorized spend
/// (`2 * budget > budget`) on an opportunistic/warrior proposal stream that the core gate caps.
pub proof fn ungated_disposition_diverges(budget: nat)
    requires
        budget > 0,
    ensures
        ({
            let p = CandidateProposal {
                amount: budget,
                now: 0,
                expires: 10,
                payload_matches: true,
                grants_current: true,
            };
            let warrior_stream = seq![p, p];
            gated_run(budget, warrior_stream).admitted_unauthorized == 0
                && gated_run(budget, warrior_stream).admitted_valid == 1
                && gated_run(budget, warrior_stream).charged == budget
                && ungated_run(budget, warrior_stream).admitted_unauthorized == 1
                && ungated_run(budget, warrior_stream).charged == 2 * budget
                && ungated_run(budget, warrior_stream).charged > budget
                && stateless_per_call_run(budget, warrior_stream).admitted_unauthorized == 1
                && stateless_per_call_run(budget, warrior_stream).charged == 2 * budget
                && stateless_per_call_run(budget, warrior_stream).charged > budget
        }),
{
    reveal_with_fuel(gated_run, 3);
    reveal_with_fuel(ungated_run, 3);
    reveal_with_fuel(stateless_per_call_run, 3);

    let p = CandidateProposal {
        amount: budget,
        now: 0,
        expires: 10,
        payload_matches: true,
        grants_current: true,
    };
    let warrior_stream = seq![p, p];
    assert(warrior_stream.drop_last() =~= seq![p]);
    assert(warrior_stream.drop_last().drop_last() =~= Seq::<CandidateProposal>::empty());
}

} // verus!
