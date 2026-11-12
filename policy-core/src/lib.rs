//! Deterministic executable decisions. Parsing, durable snapshots, identity and
//! normalized-effect construction are caller obligations, not verified here.
// Keep the verification interface flat and use Verus's Vec view specifications.
#![allow(
    clippy::ptr_arg,
    clippy::too_many_arguments,
    clippy::question_mark,
    clippy::len_zero,
    clippy::assign_op_pattern
)]
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

// ============================================================================
// V4 Pillar 1A: Canonical Permit Wire Framing & Injectivity (H1 / Wire Integrity)
// ============================================================================

pub const EFFECT_TRANSFER: u8 = 0;
pub const EFFECT_RELEASE: u8 = 1;
pub const EFFECT_EDIT: u8 = 2;

pub const WIRE_TAG_PERMIT_V1: u64 = 0x5045_524d_4954_0001;
pub const WIRE_TAG_PERMIT_OVERFLOW: u64 = 0x5045_524d_4954_ffff;
pub const WIRE_TAG_ID: u64 = 1;
pub const WIRE_TAG_ACTOR: u64 = 2;
pub const WIRE_TAG_EFFECT_KIND: u64 = 3;
pub const WIRE_TAG_AMOUNT: u64 = 4;
pub const WIRE_TAG_EXPIRES: u64 = 5;
pub const WIRE_TAG_EXPECTED_REV: u64 = 6;
pub const WIRE_TAG_TARGET: u64 = 7;
pub const WIRE_TAG_CONTENTS: u64 = 8;
pub const WIRE_TAG_EPOCHS: u64 = 9;
pub const WIRE_TAG_REQUIRED: u64 = 10;

pub struct SpecPermitWire {
    pub id: Seq<u8>,
    pub actor: Seq<u8>,
    pub effect_kind: u8,
    pub amount: u64,
    pub expires: u64,
    pub expected_rev: u64,
    pub target: Seq<u8>,
    pub contents: Seq<u8>,
    pub epochs: Seq<u64>,
    pub required: Seq<bool>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ExecPermitWire {
    pub id: Vec<u8>,
    pub actor: Vec<u8>,
    pub effect_kind: u8,
    pub amount: u64,
    pub expires: u64,
    pub expected_rev: u64,
    pub target: Vec<u8>,
    pub contents: Vec<u8>,
    pub epochs: Vec<u64>,
    pub required: Vec<bool>,
}

impl View for ExecPermitWire {
    type V = SpecPermitWire;

    open spec fn view(&self) -> SpecPermitWire {
        SpecPermitWire {
            id: self.id@,
            actor: self.actor@,
            effect_kind: self.effect_kind,
            amount: self.amount,
            expires: self.expires,
            expected_rev: self.expected_rev,
            target: self.target@,
            contents: self.contents@,
            epochs: self.epochs@,
            required: self.required@,
        }
    }
}

pub open spec fn spec_permit_fits_u64(p: SpecPermitWire) -> bool {
    p.id.len() <= u64::MAX as nat
        && p.actor.len() <= u64::MAX as nat
        && p.target.len() <= u64::MAX as nat
        && p.contents.len() <= u64::MAX as nat
        && p.epochs.len() <= u64::MAX as nat
        && p.required.len() <= u64::MAX as nat
}

#[verifier::opaque]
pub open spec fn u64_le_bytes_spec(x: u64) -> Seq<u8> {
    let b0 = (x % 256) as u8;
    let x1 = x / 256;
    let b1 = (x1 % 256) as u8;
    let x2 = x1 / 256;
    let b2 = (x2 % 256) as u8;
    let x3 = x2 / 256;
    let b3 = (x3 % 256) as u8;
    let x4 = x3 / 256;
    let b4 = (x4 % 256) as u8;
    let x5 = x4 / 256;
    let b5 = (x5 % 256) as u8;
    let x6 = x5 / 256;
    let b6 = (x6 % 256) as u8;
    let x7 = x6 / 256;
    let b7 = (x7 % 256) as u8;
    seq![b0, b1, b2, b3, b4, b5, b6, b7]
}

pub open spec fn bool_byte_spec(b: bool) -> u8 {
    if b {
        1u8
    } else {
        0u8
    }
}

pub open spec fn encode_u64_seq_spec(xs: Seq<u64>) -> Seq<u8>
    decreases xs.len(),
{
    if xs.len() == 0 {
        Seq::empty()
    } else {
        encode_u64_seq_spec(xs.drop_last()) + u64_le_bytes_spec(xs.last())
    }
}

pub open spec fn encode_bool_seq_spec(bs: Seq<bool>) -> Seq<u8>
    decreases bs.len(),
{
    if bs.len() == 0 {
        Seq::empty()
    } else {
        encode_bool_seq_spec(bs.drop_last()).push(bool_byte_spec(bs.last()))
    }
}

pub open spec fn encode_nat_varint_spec(n: nat) -> Seq<u8>
    decreases n,
{
    if n < 128 {
        seq![n as u8]
    } else {
        seq![((n % 128) + 128) as u8] + encode_nat_varint_spec(n / 128)
    }
}

pub open spec fn encode_bytes_fixed_spec(tag: u64, bytes: Seq<u8>) -> Seq<u8> {
    u64_le_bytes_spec(tag) + u64_le_bytes_spec(bytes.len() as u64) + bytes
}

pub open spec fn encode_u64_field_spec(tag: u64, val: u64) -> Seq<u8> {
    u64_le_bytes_spec(tag) + u64_le_bytes_spec(val)
}

pub open spec fn encode_u8_field_spec(tag: u64, val: u8) -> Seq<u8> {
    u64_le_bytes_spec(tag) + seq![val]
}

#[verifier::opaque]
pub open spec fn encode_permit_fixed_spec(p: SpecPermitWire) -> Seq<u8> {
    u64_le_bytes_spec(WIRE_TAG_PERMIT_V1)
        + (encode_bytes_fixed_spec(WIRE_TAG_ID, p.id)
            + (encode_bytes_fixed_spec(WIRE_TAG_ACTOR, p.actor)
                + (encode_u8_field_spec(WIRE_TAG_EFFECT_KIND, p.effect_kind)
                    + (encode_u64_field_spec(WIRE_TAG_AMOUNT, p.amount)
                        + (encode_u64_field_spec(WIRE_TAG_EXPIRES, p.expires)
                            + (encode_u64_field_spec(WIRE_TAG_EXPECTED_REV, p.expected_rev)
                                + (encode_bytes_fixed_spec(WIRE_TAG_TARGET, p.target)
                                    + (encode_bytes_fixed_spec(WIRE_TAG_CONTENTS, p.contents)
                                        + (encode_u64_seq_framed_spec(WIRE_TAG_EPOCHS, p.epochs)
                                            + encode_bool_seq_framed_spec(
                                                WIRE_TAG_REQUIRED,
                                                p.required,
                                            ))))))))))
}

pub open spec fn encode_u64_seq_framed_spec(tag: u64, xs: Seq<u64>) -> Seq<u8> {
    u64_le_bytes_spec(tag) + u64_le_bytes_spec(xs.len() as u64) + encode_u64_seq_spec(xs)
}

pub open spec fn encode_bool_seq_framed_spec(tag: u64, bs: Seq<bool>) -> Seq<u8> {
    u64_le_bytes_spec(tag) + u64_le_bytes_spec(bs.len() as u64) + encode_bool_seq_spec(bs)
}

#[verifier::opaque]
pub open spec fn encode_permit_unbounded_spec(p: SpecPermitWire) -> Seq<u8> {
    u64_le_bytes_spec(WIRE_TAG_PERMIT_OVERFLOW)
        + (encode_nat_varint_spec(p.id.len())
            + (p.id
                + (encode_nat_varint_spec(p.actor.len())
                    + (p.actor
                        + (seq![p.effect_kind]
                            + (u64_le_bytes_spec(p.amount)
                                + (u64_le_bytes_spec(p.expires)
                                    + (u64_le_bytes_spec(p.expected_rev)
                                        + (encode_nat_varint_spec(p.target.len())
                                            + (p.target
                                                + (encode_nat_varint_spec(p.contents.len())
                                                    + (p.contents
                                                        + (encode_nat_varint_spec(p.epochs.len())
                                                            + (encode_u64_seq_spec(p.epochs)
                                                                + (encode_nat_varint_spec(
                                                                    p.required.len(),
                                                                ) + encode_bool_seq_spec(
                                                                    p.required,
                                                                ))))))))))))))))
}

pub open spec fn encode_permit_spec(p: SpecPermitWire) -> Seq<u8> {
    if spec_permit_fits_u64(p) {
        encode_permit_fixed_spec(p)
    } else {
        encode_permit_unbounded_spec(p)
    }
}

pub proof fn u64_le_bytes_injective(x: u64, y: u64)
    ensures
        u64_le_bytes_spec(x).len() == 8,
        u64_le_bytes_spec(y).len() == 8,
        u64_le_bytes_spec(x) == u64_le_bytes_spec(y) ==> x == y,
{
    reveal(u64_le_bytes_spec);
    let sx = u64_le_bytes_spec(x);
    let sy = u64_le_bytes_spec(y);
    if sx == sy {
        assert(sx[0] == sy[0]);
        assert(sx[1] == sy[1]);
        assert(sx[2] == sy[2]);
        assert(sx[3] == sy[3]);
        assert(sx[4] == sy[4]);
        assert(sx[5] == sy[5]);
        assert(sx[6] == sy[6]);
        assert(sx[7] == sy[7]);
    }
}

proof fn seq_append_cancel_left(a1: Seq<u8>, b1: Seq<u8>, a2: Seq<u8>, b2: Seq<u8>)
    requires
        a1.len() == a2.len(),
        a1 + b1 == a2 + b2,
    ensures
        a1 == a2,
        b1 == b2,
{
    let k = a1.len() as int;
    let total = (a1 + b1).len() as int;
    assert(a1 =~= (a1 + b1).subrange(0, k));
    assert(a2 =~= (a2 + b2).subrange(0, k));
    assert(b1 =~= (a1 + b1).subrange(k, total));
    assert(b2 =~= (a2 + b2).subrange(k, total));
}

pub proof fn encode_u64_seq_len(xs: Seq<u64>)
    ensures
        encode_u64_seq_spec(xs).len() == 8 * xs.len(),
    decreases xs.len(),
{
    if xs.len() > 0 {
        encode_u64_seq_len(xs.drop_last());
        u64_le_bytes_injective(xs.last(), xs.last());
    }
}

pub proof fn encode_u64_seq_injective(xs: Seq<u64>, ys: Seq<u64>)
    requires
        xs.len() == ys.len(),
        encode_u64_seq_spec(xs) == encode_u64_seq_spec(ys),
    ensures
        xs == ys,
    decreases xs.len(),
{
    if xs.len() == 0 {
        assert(xs =~= ys);
    } else {
        encode_u64_seq_len(xs.drop_last());
        encode_u64_seq_len(ys.drop_last());
        seq_append_cancel_left(
            encode_u64_seq_spec(xs.drop_last()),
            u64_le_bytes_spec(xs.last()),
            encode_u64_seq_spec(ys.drop_last()),
            u64_le_bytes_spec(ys.last()),
        );
        encode_u64_seq_injective(xs.drop_last(), ys.drop_last());
        u64_le_bytes_injective(xs.last(), ys.last());
        assert(xs =~= xs.drop_last().push(xs.last()));
        assert(ys =~= ys.drop_last().push(ys.last()));
    }
}

pub proof fn encode_bool_seq_len(bs: Seq<bool>)
    ensures
        encode_bool_seq_spec(bs).len() == bs.len(),
    decreases bs.len(),
{
    if bs.len() > 0 {
        encode_bool_seq_len(bs.drop_last());
    }
}

proof fn encode_bool_seq_index(bs: Seq<bool>, i: int)
    requires
        0 <= i < bs.len(),
    ensures
        encode_bool_seq_spec(bs).len() == bs.len(),
        encode_bool_seq_spec(bs)[i] == bool_byte_spec(bs[i]),
    decreases bs.len(),
{
    encode_bool_seq_len(bs);
    if i == bs.len() - 1 {
        assert(bs.last() == bs[i]);
    } else {
        encode_bool_seq_index(bs.drop_last(), i);
        assert(bs.drop_last()[i] == bs[i]);
    }
}

pub proof fn encode_bool_seq_injective(bs1: Seq<bool>, bs2: Seq<bool>)
    requires
        bs1.len() == bs2.len(),
        encode_bool_seq_spec(bs1) == encode_bool_seq_spec(bs2),
    ensures
        bs1 == bs2,
{
    assert forall|i: int| 0 <= i < bs1.len() implies #[trigger] bs1[i] == bs2[i] by {
        encode_bool_seq_index(bs1, i);
        encode_bool_seq_index(bs2, i);
        assert(encode_bool_seq_spec(bs1)[i] == encode_bool_seq_spec(bs2)[i]);
    }
    assert(bs1 =~= bs2);
}

proof fn encode_nat_varint_nonempty(n: nat)
    ensures
        encode_nat_varint_spec(n).len() >= 1,
    decreases n,
{
    if n >= 128 {
        encode_nat_varint_nonempty(n / 128);
    }
}

proof fn encode_nat_varint_prefix_free(n1: nat, r1: Seq<u8>, n2: nat, r2: Seq<u8>)
    requires
        encode_nat_varint_spec(n1) + r1 == encode_nat_varint_spec(n2) + r2,
    ensures
        n1 == n2,
        r1 == r2,
    decreases n1,
{
    encode_nat_varint_nonempty(n1);
    encode_nat_varint_nonempty(n2);
    let lhs = encode_nat_varint_spec(n1) + r1;
    let rhs = encode_nat_varint_spec(n2) + r2;
    assert(lhs[0] == encode_nat_varint_spec(n1)[0]);
    assert(rhs[0] == encode_nat_varint_spec(n2)[0]);
    assert(lhs[0] == rhs[0]);
    if n1 < 128 {
        assert(lhs[0] < 128);
        assert(n2 < 128);
        assert((n1 as u8) == (n2 as u8));
        assert(n1 == n2);
        seq_append_cancel_left(
            encode_nat_varint_spec(n1),
            r1,
            encode_nat_varint_spec(n2),
            r2,
        );
    } else {
        assert(lhs[0] >= 128);
        assert(n2 >= 128);
        let h1 = seq![((n1 % 128) + 128) as u8];
        let h2 = seq![((n2 % 128) + 128) as u8];
        let t1 = encode_nat_varint_spec(n1 / 128);
        let t2 = encode_nat_varint_spec(n2 / 128);
        assert((h1 + t1) + r1 =~= h1 + (t1 + r1));
        assert((h2 + t2) + r2 =~= h2 + (t2 + r2));
        seq_append_cancel_left(h1, t1 + r1, h2, t2 + r2);
        assert(h1[0] == h2[0]);
        assert(n1 % 128 == n2 % 128);
        encode_nat_varint_prefix_free(n1 / 128, r1, n2 / 128, r2);
    }
}

proof fn peel_bytes_fixed(
    tag: u64,
    b1: Seq<u8>,
    rest1: Seq<u8>,
    b2: Seq<u8>,
    rest2: Seq<u8>,
)
    requires
        b1.len() <= u64::MAX as nat,
        b2.len() <= u64::MAX as nat,
        encode_bytes_fixed_spec(tag, b1) + rest1 == encode_bytes_fixed_spec(tag, b2) + rest2,
    ensures
        b1 == b2,
        rest1 == rest2,
{
    u64_le_bytes_injective(tag, tag);
    u64_le_bytes_injective(b1.len() as u64, b2.len() as u64);
    let t = u64_le_bytes_spec(tag);
    let l1 = u64_le_bytes_spec(b1.len() as u64);
    let l2 = u64_le_bytes_spec(b2.len() as u64);
    assert(encode_bytes_fixed_spec(tag, b1) + rest1 =~= t + (l1 + (b1 + rest1)));
    assert(encode_bytes_fixed_spec(tag, b2) + rest2 =~= t + (l2 + (b2 + rest2)));
    seq_append_cancel_left(t, l1 + (b1 + rest1), t, l2 + (b2 + rest2));
    seq_append_cancel_left(l1, b1 + rest1, l2, b2 + rest2);
    assert(b1.len() == b2.len());
    seq_append_cancel_left(b1, rest1, b2, rest2);
}

proof fn peel_u64_fixed(tag: u64, v1: u64, rest1: Seq<u8>, v2: u64, rest2: Seq<u8>)
    requires
        encode_u64_field_spec(tag, v1) + rest1 == encode_u64_field_spec(tag, v2) + rest2,
    ensures
        v1 == v2,
        rest1 == rest2,
{
    u64_le_bytes_injective(tag, tag);
    u64_le_bytes_injective(v1, v2);
    let t = u64_le_bytes_spec(tag);
    let x1 = u64_le_bytes_spec(v1);
    let x2 = u64_le_bytes_spec(v2);
    assert(encode_u64_field_spec(tag, v1) + rest1 =~= t + (x1 + rest1));
    assert(encode_u64_field_spec(tag, v2) + rest2 =~= t + (x2 + rest2));
    seq_append_cancel_left(t, x1 + rest1, t, x2 + rest2);
    seq_append_cancel_left(x1, rest1, x2, rest2);
}

proof fn peel_u8_fixed(tag: u64, v1: u8, rest1: Seq<u8>, v2: u8, rest2: Seq<u8>)
    requires
        encode_u8_field_spec(tag, v1) + rest1 == encode_u8_field_spec(tag, v2) + rest2,
    ensures
        v1 == v2,
        rest1 == rest2,
{
    u64_le_bytes_injective(tag, tag);
    let t = u64_le_bytes_spec(tag);
    let x1 = seq![v1];
    let x2 = seq![v2];
    assert(encode_u8_field_spec(tag, v1) + rest1 =~= t + (x1 + rest1));
    assert(encode_u8_field_spec(tag, v2) + rest2 =~= t + (x2 + rest2));
    seq_append_cancel_left(t, x1 + rest1, t, x2 + rest2);
    seq_append_cancel_left(x1, rest1, x2, rest2);
    assert(x1[0] == x2[0]);
}

proof fn encode_permit_fixed_injective(p1: SpecPermitWire, p2: SpecPermitWire)
    requires
        spec_permit_fits_u64(p1),
        spec_permit_fits_u64(p2),
        encode_permit_fixed_spec(p1) == encode_permit_fixed_spec(p2),
    ensures
        p1 == p2,
{
    reveal(encode_permit_fixed_spec);
    u64_le_bytes_injective(WIRE_TAG_PERMIT_V1, WIRE_TAG_PERMIT_V1);
    let h = u64_le_bytes_spec(WIRE_TAG_PERMIT_V1);

    let r1_9 = encode_bool_seq_framed_spec(WIRE_TAG_REQUIRED, p1.required);
    let r2_9 = encode_bool_seq_framed_spec(WIRE_TAG_REQUIRED, p2.required);

    let r1_8 = encode_u64_seq_framed_spec(WIRE_TAG_EPOCHS, p1.epochs) + r1_9;
    let r2_8 = encode_u64_seq_framed_spec(WIRE_TAG_EPOCHS, p2.epochs) + r2_9;

    let r1_7 = encode_bytes_fixed_spec(WIRE_TAG_CONTENTS, p1.contents) + r1_8;
    let r2_7 = encode_bytes_fixed_spec(WIRE_TAG_CONTENTS, p2.contents) + r2_8;

    let r1_6 = encode_bytes_fixed_spec(WIRE_TAG_TARGET, p1.target) + r1_7;
    let r2_6 = encode_bytes_fixed_spec(WIRE_TAG_TARGET, p2.target) + r2_7;

    let r1_5 = encode_u64_field_spec(WIRE_TAG_EXPECTED_REV, p1.expected_rev) + r1_6;
    let r2_5 = encode_u64_field_spec(WIRE_TAG_EXPECTED_REV, p2.expected_rev) + r2_6;

    let r1_4 = encode_u64_field_spec(WIRE_TAG_EXPIRES, p1.expires) + r1_5;
    let r2_4 = encode_u64_field_spec(WIRE_TAG_EXPIRES, p2.expires) + r2_5;

    let r1_3 = encode_u64_field_spec(WIRE_TAG_AMOUNT, p1.amount) + r1_4;
    let r2_3 = encode_u64_field_spec(WIRE_TAG_AMOUNT, p2.amount) + r2_4;

    let r1_2 = encode_u8_field_spec(WIRE_TAG_EFFECT_KIND, p1.effect_kind) + r1_3;
    let r2_2 = encode_u8_field_spec(WIRE_TAG_EFFECT_KIND, p2.effect_kind) + r2_3;

    let r1_1 = encode_bytes_fixed_spec(WIRE_TAG_ACTOR, p1.actor) + r1_2;
    let r2_1 = encode_bytes_fixed_spec(WIRE_TAG_ACTOR, p2.actor) + r2_2;

    let r1_0 = encode_bytes_fixed_spec(WIRE_TAG_ID, p1.id) + r1_1;
    let r2_0 = encode_bytes_fixed_spec(WIRE_TAG_ID, p2.id) + r2_1;

    seq_append_cancel_left(h, r1_0, h, r2_0);

    peel_bytes_fixed(WIRE_TAG_ID, p1.id, r1_1, p2.id, r2_1);
    peel_bytes_fixed(WIRE_TAG_ACTOR, p1.actor, r1_2, p2.actor, r2_2);
    peel_u8_fixed(WIRE_TAG_EFFECT_KIND, p1.effect_kind, r1_3, p2.effect_kind, r2_3);
    peel_u64_fixed(WIRE_TAG_AMOUNT, p1.amount, r1_4, p2.amount, r2_4);
    peel_u64_fixed(WIRE_TAG_EXPIRES, p1.expires, r1_5, p2.expires, r2_5);
    peel_u64_fixed(WIRE_TAG_EXPECTED_REV, p1.expected_rev, r1_6, p2.expected_rev, r2_6);
    peel_bytes_fixed(WIRE_TAG_TARGET, p1.target, r1_7, p2.target, r2_7);
    peel_bytes_fixed(WIRE_TAG_CONTENTS, p1.contents, r1_8, p2.contents, r2_8);

    u64_le_bytes_injective(WIRE_TAG_EPOCHS, WIRE_TAG_EPOCHS);
    u64_le_bytes_injective(p1.epochs.len() as u64, p2.epochs.len() as u64);
    let te = u64_le_bytes_spec(WIRE_TAG_EPOCHS);
    let le1 = u64_le_bytes_spec(p1.epochs.len() as u64);
    let le2 = u64_le_bytes_spec(p2.epochs.len() as u64);
    let be1 = encode_u64_seq_spec(p1.epochs);
    let be2 = encode_u64_seq_spec(p2.epochs);
    assert(r1_8 =~= te + (le1 + (be1 + r1_9)));
    assert(r2_8 =~= te + (le2 + (be2 + r2_9)));
    seq_append_cancel_left(te, le1 + (be1 + r1_9), te, le2 + (be2 + r2_9));
    seq_append_cancel_left(le1, be1 + r1_9, le2, be2 + r2_9);
    assert(p1.epochs.len() == p2.epochs.len());
    encode_u64_seq_len(p1.epochs);
    encode_u64_seq_len(p2.epochs);
    seq_append_cancel_left(be1, r1_9, be2, r2_9);
    encode_u64_seq_injective(p1.epochs, p2.epochs);

    u64_le_bytes_injective(WIRE_TAG_REQUIRED, WIRE_TAG_REQUIRED);
    u64_le_bytes_injective(p1.required.len() as u64, p2.required.len() as u64);
    let tr = u64_le_bytes_spec(WIRE_TAG_REQUIRED);
    let lr1 = u64_le_bytes_spec(p1.required.len() as u64);
    let lr2 = u64_le_bytes_spec(p2.required.len() as u64);
    let br1 = encode_bool_seq_spec(p1.required);
    let br2 = encode_bool_seq_spec(p2.required);
    assert(r1_9 =~= tr + (lr1 + br1));
    assert(r2_9 =~= tr + (lr2 + br2));
    seq_append_cancel_left(tr, lr1 + br1, tr, lr2 + br2);
    seq_append_cancel_left(lr1, br1, lr2, br2);
    assert(p1.required.len() == p2.required.len());
    encode_bool_seq_injective(p1.required, p2.required);
}

proof fn encode_permit_unbounded_injective(p1: SpecPermitWire, p2: SpecPermitWire)
    requires
        encode_permit_unbounded_spec(p1) == encode_permit_unbounded_spec(p2),
    ensures
        p1 == p2,
{
    reveal(encode_permit_unbounded_spec);
    u64_le_bytes_injective(WIRE_TAG_PERMIT_OVERFLOW, WIRE_TAG_PERMIT_OVERFLOW);
    let h = u64_le_bytes_spec(WIRE_TAG_PERMIT_OVERFLOW);

    let r1_9 = encode_nat_varint_spec(p1.required.len()) + encode_bool_seq_spec(p1.required);
    let r2_9 = encode_nat_varint_spec(p2.required.len()) + encode_bool_seq_spec(p2.required);

    let r1_8 = encode_nat_varint_spec(p1.epochs.len()) + (encode_u64_seq_spec(p1.epochs) + r1_9);
    let r2_8 = encode_nat_varint_spec(p2.epochs.len()) + (encode_u64_seq_spec(p2.epochs) + r2_9);

    let r1_7 = encode_nat_varint_spec(p1.contents.len()) + (p1.contents + r1_8);
    let r2_7 = encode_nat_varint_spec(p2.contents.len()) + (p2.contents + r2_8);

    let r1_6 = encode_nat_varint_spec(p1.target.len()) + (p1.target + r1_7);
    let r2_6 = encode_nat_varint_spec(p2.target.len()) + (p2.target + r2_7);

    let r1_5 = u64_le_bytes_spec(p1.expected_rev) + r1_6;
    let r2_5 = u64_le_bytes_spec(p2.expected_rev) + r2_6;

    let r1_4 = u64_le_bytes_spec(p1.expires) + r1_5;
    let r2_4 = u64_le_bytes_spec(p2.expires) + r2_5;

    let r1_3 = u64_le_bytes_spec(p1.amount) + r1_4;
    let r2_3 = u64_le_bytes_spec(p2.amount) + r2_4;

    let r1_2 = seq![p1.effect_kind] + r1_3;
    let r2_2 = seq![p2.effect_kind] + r2_3;

    let r1_1 = encode_nat_varint_spec(p1.actor.len()) + (p1.actor + r1_2);
    let r2_1 = encode_nat_varint_spec(p2.actor.len()) + (p2.actor + r2_2);

    let r1_0 = encode_nat_varint_spec(p1.id.len()) + (p1.id + r1_1);
    let r2_0 = encode_nat_varint_spec(p2.id.len()) + (p2.id + r2_1);

    seq_append_cancel_left(h, r1_0, h, r2_0);

    encode_nat_varint_prefix_free(p1.id.len(), p1.id + r1_1, p2.id.len(), p2.id + r2_1);
    seq_append_cancel_left(p1.id, r1_1, p2.id, r2_1);

    encode_nat_varint_prefix_free(
        p1.actor.len(),
        p1.actor + r1_2,
        p2.actor.len(),
        p2.actor + r2_2,
    );
    seq_append_cancel_left(p1.actor, r1_2, p2.actor, r2_2);

    seq_append_cancel_left(seq![p1.effect_kind], r1_3, seq![p2.effect_kind], r2_3);
    assert(seq![p1.effect_kind][0] == seq![p2.effect_kind][0]);

    u64_le_bytes_injective(p1.amount, p2.amount);
    seq_append_cancel_left(
        u64_le_bytes_spec(p1.amount),
        r1_4,
        u64_le_bytes_spec(p2.amount),
        r2_4,
    );

    u64_le_bytes_injective(p1.expires, p2.expires);
    seq_append_cancel_left(
        u64_le_bytes_spec(p1.expires),
        r1_5,
        u64_le_bytes_spec(p2.expires),
        r2_5,
    );

    u64_le_bytes_injective(p1.expected_rev, p2.expected_rev);
    seq_append_cancel_left(
        u64_le_bytes_spec(p1.expected_rev),
        r1_6,
        u64_le_bytes_spec(p2.expected_rev),
        r2_6,
    );

    encode_nat_varint_prefix_free(
        p1.target.len(),
        p1.target + r1_7,
        p2.target.len(),
        p2.target + r2_7,
    );
    seq_append_cancel_left(p1.target, r1_7, p2.target, r2_7);

    encode_nat_varint_prefix_free(
        p1.contents.len(),
        p1.contents + r1_8,
        p2.contents.len(),
        p2.contents + r2_8,
    );
    seq_append_cancel_left(p1.contents, r1_8, p2.contents, r2_8);

    encode_nat_varint_prefix_free(
        p1.epochs.len(),
        encode_u64_seq_spec(p1.epochs) + r1_9,
        p2.epochs.len(),
        encode_u64_seq_spec(p2.epochs) + r2_9,
    );
    encode_u64_seq_len(p1.epochs);
    encode_u64_seq_len(p2.epochs);
    seq_append_cancel_left(
        encode_u64_seq_spec(p1.epochs),
        r1_9,
        encode_u64_seq_spec(p2.epochs),
        r2_9,
    );
    encode_u64_seq_injective(p1.epochs, p2.epochs);

    encode_nat_varint_prefix_free(
        p1.required.len(),
        encode_bool_seq_spec(p1.required),
        p2.required.len(),
        encode_bool_seq_spec(p2.required),
    );
    encode_bool_seq_injective(p1.required, p2.required);
}

proof fn encode_permit_fixed_header(p: SpecPermitWire)
    ensures
        encode_permit_fixed_spec(p).len() >= 8,
        encode_permit_fixed_spec(p).subrange(0, 8) == u64_le_bytes_spec(WIRE_TAG_PERMIT_V1),
{
    reveal(encode_permit_fixed_spec);
    u64_le_bytes_injective(WIRE_TAG_PERMIT_V1, WIRE_TAG_PERMIT_V1);
    let h = u64_le_bytes_spec(WIRE_TAG_PERMIT_V1);
    assert(encode_permit_fixed_spec(p).subrange(0, 8) =~= h);
}

proof fn encode_permit_unbounded_header(p: SpecPermitWire)
    ensures
        encode_permit_unbounded_spec(p).len() >= 8,
        encode_permit_unbounded_spec(p).subrange(0, 8)
            == u64_le_bytes_spec(WIRE_TAG_PERMIT_OVERFLOW),
{
    reveal(encode_permit_unbounded_spec);
    u64_le_bytes_injective(WIRE_TAG_PERMIT_OVERFLOW, WIRE_TAG_PERMIT_OVERFLOW);
    let h = u64_le_bytes_spec(WIRE_TAG_PERMIT_OVERFLOW);
    assert(encode_permit_unbounded_spec(p).subrange(0, 8) =~= h);
}

/// H1 Wire Integrity: Canonical fixed-width tag + length-prefixed permit wire encoding
/// is unconditionally injective across all `SpecPermitWire` values, eliminating cross-field
/// concatenation and framing ambiguity.
pub proof fn encode_permit_injective(p1: SpecPermitWire, p2: SpecPermitWire)
    ensures
        encode_permit_spec(p1) == encode_permit_spec(p2) ==> p1 == p2,
{
    if encode_permit_spec(p1) == encode_permit_spec(p2) {
        let f1 = spec_permit_fits_u64(p1);
        let f2 = spec_permit_fits_u64(p2);
        if f1 && f2 {
            encode_permit_fixed_injective(p1, p2);
        } else if !f1 && !f2 {
            encode_permit_unbounded_injective(p1, p2);
        } else {
            if f1 {
                encode_permit_fixed_header(p1);
                encode_permit_unbounded_header(p2);
            } else {
                encode_permit_unbounded_header(p1);
                encode_permit_fixed_header(p2);
            }
            u64_le_bytes_injective(WIRE_TAG_PERMIT_V1, WIRE_TAG_PERMIT_OVERFLOW);
            assert(false);
        }
    }
}

pub fn push_u64_le(out: &mut Vec<u8>, x: u64)
    ensures
        final(out)@ == old(out)@ + u64_le_bytes_spec(x),
{
    reveal(u64_le_bytes_spec);
    let b0 = (x % 256) as u8;
    let x1 = x / 256;
    let b1 = (x1 % 256) as u8;
    let x2 = x1 / 256;
    let b2 = (x2 % 256) as u8;
    let x3 = x2 / 256;
    let b3 = (x3 % 256) as u8;
    let x4 = x3 / 256;
    let b4 = (x4 % 256) as u8;
    let x5 = x4 / 256;
    let b5 = (x5 % 256) as u8;
    let x6 = x5 / 256;
    let b6 = (x6 % 256) as u8;
    let x7 = x6 / 256;
    let b7 = (x7 % 256) as u8;
    out.push(b0);
    out.push(b1);
    out.push(b2);
    out.push(b3);
    out.push(b4);
    out.push(b5);
    out.push(b6);
    out.push(b7);
    assert(out@ =~= old(out)@ + u64_le_bytes_spec(x));
}

fn push_slice_bytes(out: &mut Vec<u8>, src: &Vec<u8>)
    ensures
        final(out)@ == old(out)@ + src@,
{
    let mut i: usize = 0;
    while i < src.len()
        invariant
            i <= src.len(),
            out@ == old(out)@ + src@.subrange(0, i as int),
        decreases src.len() - i,
    {
        let b = src[i];
        out.push(b);
        assert(src@.subrange(0, i as int + 1) =~= src@.subrange(0, i as int).push(b));
        assert(out@ =~= old(out)@ + src@.subrange(0, i as int + 1));
        i += 1;
    }
    assert(src@.subrange(0, src.len() as int) =~= src@);
}

fn push_bytes_framed(out: &mut Vec<u8>, tag: u64, src: &Vec<u8>)
    ensures
        src@.len() <= u64::MAX as nat,
        final(out)@ == old(out)@ + encode_bytes_fixed_spec(tag, src@),
{
    push_u64_le(out, tag);
    push_u64_le(out, src.len() as u64);
    push_slice_bytes(out, src);
    assert(out@ =~= old(out)@ + encode_bytes_fixed_spec(tag, src@));
}

fn push_u64_seq(out: &mut Vec<u8>, xs: &Vec<u64>)
    ensures
        final(out)@ == old(out)@ + encode_u64_seq_spec(xs@),
{
    let mut i: usize = 0;
    assert(xs@.subrange(0, 0) =~= Seq::<u64>::empty());
    while i < xs.len()
        invariant
            i <= xs.len(),
            out@ == old(out)@ + encode_u64_seq_spec(xs@.subrange(0, i as int)),
        decreases xs.len() - i,
    {
        let v = xs[i];
        push_u64_le(out, v);
        proof {
            let sub_next = xs@.subrange(0, i as int + 1);
            assert(sub_next.drop_last() =~= xs@.subrange(0, i as int));
            assert(sub_next.last() == v);
            assert(out@ =~= old(out)@ + encode_u64_seq_spec(sub_next));
        }
        i += 1;
    }
    assert(xs@.subrange(0, xs.len() as int) =~= xs@);
}

fn push_bool_seq(out: &mut Vec<u8>, bs: &Vec<bool>)
    ensures
        final(out)@ == old(out)@ + encode_bool_seq_spec(bs@),
{
    let mut i: usize = 0;
    assert(bs@.subrange(0, 0) =~= Seq::<bool>::empty());
    while i < bs.len()
        invariant
            i <= bs.len(),
            out@ == old(out)@ + encode_bool_seq_spec(bs@.subrange(0, i as int)),
        decreases bs.len() - i,
    {
        let b = bs[i];
        let byte = if b { 1u8 } else { 0u8 };
        out.push(byte);
        proof {
            let sub_next = bs@.subrange(0, i as int + 1);
            assert(sub_next.drop_last() =~= bs@.subrange(0, i as int));
            assert(sub_next.last() == b);
            assert(out@ =~= old(out)@ + encode_bool_seq_spec(sub_next));
        }
        i += 1;
    }
    assert(bs@.subrange(0, bs.len() as int) =~= bs@);
}

fn push_u8_framed(out: &mut Vec<u8>, tag: u64, val: u8)
    ensures
        final(out)@ == old(out)@ + encode_u8_field_spec(tag, val),
{
    push_u64_le(out, tag);
    out.push(val);
    assert(out@ =~= old(out)@ + encode_u8_field_spec(tag, val));
}

fn push_u64_framed(out: &mut Vec<u8>, tag: u64, val: u64)
    ensures
        final(out)@ == old(out)@ + encode_u64_field_spec(tag, val),
{
    push_u64_le(out, tag);
    push_u64_le(out, val);
    assert(out@ =~= old(out)@ + encode_u64_field_spec(tag, val));
}

fn push_u64_seq_framed(out: &mut Vec<u8>, tag: u64, xs: &Vec<u64>)
    ensures
        xs@.len() <= u64::MAX as nat,
        final(out)@ == old(out)@ + encode_u64_seq_framed_spec(tag, xs@),
{
    push_u64_le(out, tag);
    push_u64_le(out, xs.len() as u64);
    push_u64_seq(out, xs);
    assert(out@ =~= old(out)@ + encode_u64_seq_framed_spec(tag, xs@));
}

fn push_bool_seq_framed(out: &mut Vec<u8>, tag: u64, bs: &Vec<bool>)
    ensures
        bs@.len() <= u64::MAX as nat,
        final(out)@ == old(out)@ + encode_bool_seq_framed_spec(tag, bs@),
{
    push_u64_le(out, tag);
    push_u64_le(out, bs.len() as u64);
    push_bool_seq(out, bs);
    assert(out@ =~= old(out)@ + encode_bool_seq_framed_spec(tag, bs@));
}

/// Verified canonical permit wire encoder with explicit fixed-width little-endian `u64`
/// domain/field tags and byte/element length prefixes.
pub fn encode_permit_canonical(p: &ExecPermitWire) -> (out: Vec<u8>)
    ensures
        spec_permit_fits_u64(p@),
        out@ == encode_permit_fixed_spec(p@),
        out@ == encode_permit_spec(p@),
{
    let mut out: Vec<u8> = Vec::new();
    push_u64_le(&mut out, WIRE_TAG_PERMIT_V1);
    push_bytes_framed(&mut out, WIRE_TAG_ID, &p.id);
    push_bytes_framed(&mut out, WIRE_TAG_ACTOR, &p.actor);
    push_u8_framed(&mut out, WIRE_TAG_EFFECT_KIND, p.effect_kind);
    push_u64_framed(&mut out, WIRE_TAG_AMOUNT, p.amount);
    push_u64_framed(&mut out, WIRE_TAG_EXPIRES, p.expires);
    push_u64_framed(&mut out, WIRE_TAG_EXPECTED_REV, p.expected_rev);
    push_bytes_framed(&mut out, WIRE_TAG_TARGET, &p.target);
    push_bytes_framed(&mut out, WIRE_TAG_CONTENTS, &p.contents);
    push_u64_seq_framed(&mut out, WIRE_TAG_EPOCHS, &p.epochs);
    push_bool_seq_framed(&mut out, WIRE_TAG_REQUIRED, &p.required);
    proof {
        reveal(encode_permit_fixed_spec);
        assert(out@ =~= encode_permit_fixed_spec(p@));
    }
    out
}

pub fn same_u64s(a: &Vec<u64>, b: &Vec<u64>) -> (ok: bool)
    ensures
        ok == (a@ == b@),
{
    if a.len() != b.len() {
        return false;
    }
    let mut i: usize = 0;
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

pub fn same_bools(a: &Vec<bool>, b: &Vec<bool>) -> (ok: bool)
    ensures
        ok == (a@ == b@),
{
    if a.len() != b.len() {
        return false;
    }
    let mut i: usize = 0;
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

/// Verified structural and canonical-wire permit equality predicate.
pub fn same_permit_canonical(a: &ExecPermitWire, b: &ExecPermitWire) -> (eq: bool)
    ensures
        eq == (a@ == b@),
        eq == (encode_permit_spec(a@) == encode_permit_spec(b@)),
{
    let eq = same_bytes(&a.id, &b.id)
        && same_bytes(&a.actor, &b.actor)
        && a.effect_kind == b.effect_kind
        && a.amount == b.amount
        && a.expires == b.expires
        && a.expected_rev == b.expected_rev
        && same_bytes(&a.target, &b.target)
        && same_bytes(&a.contents, &b.contents)
        && same_u64s(&a.epochs, &b.epochs)
        && same_bools(&a.required, &b.required);
    proof {
        encode_permit_injective(a@, b@);
    }
    eq
}

fn clone_vec_u8(src: &Vec<u8>) -> (out: Vec<u8>)
    ensures
        out@ == src@,
{
    let mut out: Vec<u8> = Vec::new();
    let mut i: usize = 0;
    while i < src.len()
        invariant
            i <= src.len(),
            out@ == src@.subrange(0, i as int),
        decreases src.len() - i,
    {
        let v = src[i];
        out.push(v);
        assert(src@.subrange(0, i as int + 1) =~= src@.subrange(0, i as int).push(v));
        i += 1;
    }
    assert(src@.subrange(0, src.len() as int) =~= src@);
    out
}

fn clone_vec_u64(src: &Vec<u64>) -> (out: Vec<u64>)
    ensures
        out@ == src@,
{
    let mut out: Vec<u64> = Vec::new();
    let mut i: usize = 0;
    while i < src.len()
        invariant
            i <= src.len(),
            out@ == src@.subrange(0, i as int),
        decreases src.len() - i,
    {
        let v = src[i];
        out.push(v);
        assert(src@.subrange(0, i as int + 1) =~= src@.subrange(0, i as int).push(v));
        i += 1;
    }
    assert(src@.subrange(0, src.len() as int) =~= src@);
    out
}

fn clone_vec_bool(src: &Vec<bool>) -> (out: Vec<bool>)
    ensures
        out@ == src@,
{
    let mut out: Vec<bool> = Vec::new();
    let mut i: usize = 0;
    while i < src.len()
        invariant
            i <= src.len(),
            out@ == src@.subrange(0, i as int),
        decreases src.len() - i,
    {
        let v = src[i];
        out.push(v);
        assert(src@.subrange(0, i as int + 1) =~= src@.subrange(0, i as int).push(v));
        i += 1;
    }
    assert(src@.subrange(0, src.len() as int) =~= src@);
    out
}

fn clone_vec_usize(src: &Vec<usize>) -> (out: Vec<usize>)
    ensures
        out@ == src@,
{
    let mut out: Vec<usize> = Vec::new();
    let mut i: usize = 0;
    while i < src.len()
        invariant
            i <= src.len(),
            out@ == src@.subrange(0, i as int),
        decreases src.len() - i,
    {
        let v = src[i];
        out.push(v);
        assert(src@.subrange(0, i as int + 1) =~= src@.subrange(0, i as int).push(v));
        i += 1;
    }
    assert(src@.subrange(0, src.len() as int) =~= src@);
    out
}

pub fn clone_permit_wire(p: &ExecPermitWire) -> (out: ExecPermitWire)
    ensures
        out@ == p@,
{
    ExecPermitWire {
        id: clone_vec_u8(&p.id),
        actor: clone_vec_u8(&p.actor),
        effect_kind: p.effect_kind,
        amount: p.amount,
        expires: p.expires,
        expected_rev: p.expected_rev,
        target: clone_vec_u8(&p.target),
        contents: clone_vec_u8(&p.contents),
        epochs: clone_vec_u64(&p.epochs),
        required: clone_vec_bool(&p.required),
    }
}

// ============================================================================
// V4 Pillar 1B: Executable State-Machine Kernel & Forward-Simulation Refinement
// ============================================================================

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ExecSlotStatus {
    Pending,
    Committed,
    Cancelled,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ExecReserveOutcome {
    ReservedNew { slot: usize },
    DuplicateExisting { status: ExecSlotStatus },
    PayloadMismatch,
    DeniedPolicy,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ExecSettleOutcome {
    Settled { slot: usize, amount: u64 },
    UnknownId,
    NotPending { status: ExecSlotStatus },
    SinkNotAuthoritative,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ExecAcceptOutcome {
    AcceptedPending,
    CancelledTombstone,
    Duplicate { status: ExecSlotStatus },
    PayloadMismatch,
    InvalidIssuerLength,
    UnissuedPermit,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ExecCompleteOutcome {
    NewlyCommitted { amount: u64 },
    Cancelled,
    Idempotent { status: ExecSlotStatus },
    UnknownId,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ExecBrokerRecord {
    pub permit: ExecPermitWire,
    pub status: ExecSlotStatus,
    pub inflight_pos: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ExecBrokerKernel {
    pub budget: u64,
    pub charged: u64,
    pub committed: u64,
    pub pending: u64,
    pub live: Vec<bool>,
    pub epochs: Vec<u64>,
    pub records: Vec<ExecBrokerRecord>,
    pub inflight_ridxs: Vec<usize>,
    pub inflight_amounts: Vec<u64>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ExecSinkRecord {
    pub permit: ExecPermitWire,
    pub status: ExecSlotStatus,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ExecSinkKernel {
    pub floors: Vec<u64>,
    pub revision: u64,
    pub committed_total: u64,
    pub records: Vec<ExecSinkRecord>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ExecAuthorityKernel {
    pub broker: ExecBrokerKernel,
    pub sink: ExecSinkKernel,
}

pub open spec fn seq_u64_to_nat(xs: Seq<u64>) -> Seq<nat>
    decreases xs.len(),
{
    if xs.len() == 0 {
        Seq::empty()
    } else {
        seq_u64_to_nat(xs.drop_last()).push(xs.last() as nat)
    }
}

impl View for ExecBrokerKernel {
    type V = LedgerState;

    open spec fn view(&self) -> LedgerState {
        LedgerState {
            budget: self.budget as nat,
            charged: self.charged as nat,
            committed: self.committed as nat,
            inflight: seq_u64_to_nat(self.inflight_amounts@),
        }
    }
}

impl View for ExecAuthorityKernel {
    type V = LedgerState;

    open spec fn view(&self) -> LedgerState {
        self.broker@
    }
}

pub open spec fn broker_committed_sum(recs: Seq<ExecBrokerRecord>) -> nat
    decreases recs.len(),
{
    if recs.len() == 0 {
        0
    } else {
        let add = if recs.last().status == ExecSlotStatus::Committed {
            recs.last().permit@.amount as nat
        } else {
            0nat
        };
        broker_committed_sum(recs.drop_last()) + add
    }
}

pub open spec fn broker_pending_sum(recs: Seq<ExecBrokerRecord>) -> nat
    decreases recs.len(),
{
    if recs.len() == 0 {
        0
    } else {
        let add = if recs.last().status == ExecSlotStatus::Pending {
            recs.last().permit@.amount as nat
        } else {
            0nat
        };
        broker_pending_sum(recs.drop_last()) + add
    }
}

pub open spec fn broker_active_sum(recs: Seq<ExecBrokerRecord>) -> nat {
    broker_committed_sum(recs) + broker_pending_sum(recs)
}

pub open spec fn sink_committed_sum(recs: Seq<ExecSinkRecord>) -> nat
    decreases recs.len(),
{
    if recs.len() == 0 {
        0
    } else {
        let add = if recs.last().status == ExecSlotStatus::Committed {
            recs.last().permit@.amount as nat
        } else {
            0nat
        };
        sink_committed_sum(recs.drop_last()) + add
    }
}

pub open spec fn find_broker_record_spec(
    recs: Seq<ExecBrokerRecord>,
    id: Seq<u8>,
) -> Option<int>
    decreases recs.len(),
{
    if recs.len() == 0 {
        None
    } else if recs.last().permit@.id == id {
        Some(recs.len() - 1)
    } else {
        find_broker_record_spec(recs.drop_last(), id)
    }
}

pub open spec fn find_sink_record_spec(recs: Seq<ExecSinkRecord>, id: Seq<u8>) -> Option<int>
    decreases recs.len(),
{
    if recs.len() == 0 {
        None
    } else if recs.last().permit@.id == id {
        Some(recs.len() - 1)
    } else {
        find_sink_record_spec(recs.drop_last(), id)
    }
}

pub open spec fn slot_ledger_projection(
    budget: u64,
    recs: Seq<ExecBrokerRecord>,
    id: Seq<u8>,
) -> LedgerState {
    match find_broker_record_spec(recs, id) {
        None => LedgerState {
            budget: budget as nat,
            charged: 0,
            committed: 0,
            inflight: Seq::empty(),
        },
        Some(idx) => {
            let amt = recs[idx].permit@.amount as nat;
            match recs[idx].status {
                ExecSlotStatus::Pending => LedgerState {
                    budget: budget as nat,
                    charged: amt,
                    committed: 0,
                    inflight: seq![amt],
                },
                ExecSlotStatus::Committed => LedgerState {
                    budget: budget as nat,
                    charged: amt,
                    committed: amt,
                    inflight: Seq::empty(),
                },
                ExecSlotStatus::Cancelled => LedgerState {
                    budget: budget as nat,
                    charged: 0,
                    committed: 0,
                    inflight: Seq::empty(),
                },
            }
        }
    }
}

pub open spec fn broker_records_unique_ids(recs: Seq<ExecBrokerRecord>) -> bool {
    forall|i: int, j: int|
        0 <= i < j < recs.len() ==> (#[trigger] recs[i].permit@.id) != (#[trigger] recs[j].permit@.id)
}

pub open spec fn sink_records_unique_ids(recs: Seq<ExecSinkRecord>) -> bool {
    forall|i: int, j: int|
        0 <= i < j < recs.len() ==> (#[trigger] recs[i].permit@.id) != (#[trigger] recs[j].permit@.id)
}

pub open spec fn sink_rev_ok_spec(revision: u64, p: SpecPermitWire) -> bool {
    if p.effect_kind == EFFECT_EDIT {
        revision == p.expected_rev && revision < u64::MAX
    } else {
        true
    }
}

proof fn seq_u64_to_nat_len(xs: Seq<u64>)
    ensures
        seq_u64_to_nat(xs).len() == xs.len(),
    decreases xs.len(),
{
    if xs.len() > 0 {
        seq_u64_to_nat_len(xs.drop_last());
    }
}

proof fn seq_u64_to_nat_index(xs: Seq<u64>, i: int)
    requires
        0 <= i < xs.len(),
    ensures
        seq_u64_to_nat(xs).len() == xs.len(),
        seq_u64_to_nat(xs)[i] == xs[i] as nat,
    decreases xs.len(),
{
    seq_u64_to_nat_len(xs);
    if i < xs.len() - 1 {
        seq_u64_to_nat_index(xs.drop_last(), i);
    }
}

proof fn seq_u64_to_nat_push(xs: Seq<u64>, v: u64)
    ensures
        seq_u64_to_nat(xs.push(v)) == seq_u64_to_nat(xs).push(v as nat),
{
    assert(xs.push(v).drop_last() =~= xs);
}

proof fn seq_u64_to_nat_remove(xs: Seq<u64>, slot: int)
    requires
        0 <= slot < xs.len(),
    ensures
        seq_u64_to_nat(xs.remove(slot)) == seq_u64_to_nat(xs).remove(slot),
{
    let lhs = seq_u64_to_nat(xs.remove(slot));
    let rhs = seq_u64_to_nat(xs).remove(slot);
    seq_u64_to_nat_len(xs);
    seq_u64_to_nat_len(xs.remove(slot));
    assert forall|j: int| 0 <= j < lhs.len() implies #[trigger] lhs[j] == rhs[j] by {
        seq_u64_to_nat_index(xs.remove(slot), j);
        if j < slot {
            seq_u64_to_nat_index(xs, j);
        } else {
            seq_u64_to_nat_index(xs, j + 1);
        }
    }
    assert(lhs =~= rhs);
}

proof fn find_broker_record_some(recs: Seq<ExecBrokerRecord>, id: Seq<u8>)
    ensures
        find_broker_record_spec(recs, id) is Some ==> ({
            let idx = find_broker_record_spec(recs, id).unwrap();
            0 <= idx < recs.len() && recs[idx].permit@.id == id
        }),
    decreases recs.len(),
{
    if recs.len() > 0 && recs.last().permit@.id != id {
        find_broker_record_some(recs.drop_last(), id);
    }
}

proof fn find_broker_record_none(recs: Seq<ExecBrokerRecord>, id: Seq<u8>, i: int)
    requires
        find_broker_record_spec(recs, id) is None,
        0 <= i < recs.len(),
    ensures
        recs[i].permit@.id != id,
    decreases recs.len(),
{
    if i < recs.len() - 1 {
        find_broker_record_none(recs.drop_last(), id, i);
    }
}

proof fn find_broker_record_unique_idx(recs: Seq<ExecBrokerRecord>, idx: int)
    requires
        broker_records_unique_ids(recs),
        0 <= idx < recs.len(),
    ensures
        find_broker_record_spec(recs, recs[idx].permit@.id) == Some(idx),
    decreases recs.len(),
{
    let last_i = recs.len() - 1;
    if idx == last_i {
        assert(recs.last().permit@.id == recs[idx].permit@.id);
    } else {
        assert(recs[idx].permit@.id != recs[last_i].permit@.id);
        let pref = recs.drop_last();
        assert forall|i: int, j: int|
            0 <= i < j < pref.len() implies (#[trigger] pref[i].permit@.id)
            != (#[trigger] pref[j].permit@.id) by {
            assert(pref[i] == recs[i]);
            assert(pref[j] == recs[j]);
        }
        find_broker_record_unique_idx(pref, idx);
    }
}

proof fn find_broker_record_push(
    recs: Seq<ExecBrokerRecord>,
    rec: ExecBrokerRecord,
    id: Seq<u8>,
)
    ensures
        find_broker_record_spec(recs.push(rec), id) == if rec.permit@.id == id {
            Some(recs.len() as int)
        } else {
            find_broker_record_spec(recs, id)
        },
{
    assert(recs.push(rec).drop_last() =~= recs);
}

proof fn find_broker_record_update_same_permit(
    recs: Seq<ExecBrokerRecord>,
    idx: int,
    rec: ExecBrokerRecord,
    id: Seq<u8>,
)
    requires
        0 <= idx < recs.len(),
        rec.permit@.id == recs[idx].permit@.id,
    ensures
        find_broker_record_spec(recs.update(idx, rec), id) == find_broker_record_spec(recs, id),
    decreases recs.len(),
{
    let u = recs.update(idx, rec);
    if idx == recs.len() - 1 {
        assert(u.drop_last() =~= recs.drop_last());
    } else {
        assert(u.last() == recs.last());
        assert(u.drop_last() =~= recs.drop_last().update(idx, rec));
        find_broker_record_update_same_permit(recs.drop_last(), idx, rec, id);
    }
}

proof fn broker_sums_same_status_amount(recs1: Seq<ExecBrokerRecord>, recs2: Seq<ExecBrokerRecord>)
    requires
        recs1.len() == recs2.len(),
        forall|i: int|
            0 <= i < recs1.len() ==> ((#[trigger] recs1[i]).status == recs2[i].status
                && recs1[i].permit@.amount == recs2[i].permit@.amount),
    ensures
        broker_committed_sum(recs1) == broker_committed_sum(recs2),
        broker_pending_sum(recs1) == broker_pending_sum(recs2),
    decreases recs1.len(),
{
    if recs1.len() > 0 {
        let p1 = recs1.drop_last();
        let p2 = recs2.drop_last();
        assert forall|i: int|
            0 <= i < p1.len() implies ((#[trigger] p1[i]).status == p2[i].status
            && p1[i].permit@.amount == p2[i].permit@.amount) by {
            assert(p1[i] == recs1[i]);
            assert(p2[i] == recs2[i]);
        }
        broker_sums_same_status_amount(p1, p2);
    }
}

proof fn broker_sums_on_commit(recs: Seq<ExecBrokerRecord>, idx: int, rec: ExecBrokerRecord)
    requires
        0 <= idx < recs.len(),
        recs[idx].status == ExecSlotStatus::Pending,
        rec.status == ExecSlotStatus::Committed,
        rec.permit@.amount == recs[idx].permit@.amount,
    ensures
        broker_committed_sum(recs.update(idx, rec))
            == broker_committed_sum(recs) + recs[idx].permit@.amount as nat,
        broker_pending_sum(recs)
            == broker_pending_sum(recs.update(idx, rec)) + recs[idx].permit@.amount as nat,
    decreases recs.len(),
{
    let u = recs.update(idx, rec);
    if idx == recs.len() - 1 {
        assert(u.drop_last() =~= recs.drop_last());
    } else {
        assert(u.last() == recs.last());
        assert(u.drop_last() =~= recs.drop_last().update(idx, rec));
        broker_sums_on_commit(recs.drop_last(), idx, rec);
    }
}

proof fn broker_sums_on_cancel(recs: Seq<ExecBrokerRecord>, idx: int, rec: ExecBrokerRecord)
    requires
        0 <= idx < recs.len(),
        recs[idx].status != ExecSlotStatus::Cancelled,
        rec.status == ExecSlotStatus::Cancelled,
        rec.permit@.amount == recs[idx].permit@.amount,
    ensures
        recs[idx].status == ExecSlotStatus::Pending ==> (
            broker_committed_sum(recs.update(idx, rec)) == broker_committed_sum(recs)
            && broker_pending_sum(recs)
                == broker_pending_sum(recs.update(idx, rec)) + recs[idx].permit@.amount as nat
        ),
        broker_active_sum(recs)
            == broker_active_sum(recs.update(idx, rec)) + recs[idx].permit@.amount as nat,
    decreases recs.len(),
{
    let u = recs.update(idx, rec);
    if idx == recs.len() - 1 {
        assert(u.drop_last() =~= recs.drop_last());
    } else {
        assert(u.last() == recs.last());
        assert(u.drop_last() =~= recs.drop_last().update(idx, rec));
        broker_sums_on_cancel(recs.drop_last(), idx, rec);
    }
}

proof fn find_sink_record_some(recs: Seq<ExecSinkRecord>, id: Seq<u8>)
    ensures
        find_sink_record_spec(recs, id) is Some ==> ({
            let idx = find_sink_record_spec(recs, id).unwrap();
            0 <= idx < recs.len() && recs[idx].permit@.id == id
        }),
    decreases recs.len(),
{
    if recs.len() > 0 && recs.last().permit@.id != id {
        find_sink_record_some(recs.drop_last(), id);
    }
}

proof fn find_sink_record_none(recs: Seq<ExecSinkRecord>, id: Seq<u8>, i: int)
    requires
        find_sink_record_spec(recs, id) is None,
        0 <= i < recs.len(),
    ensures
        recs[i].permit@.id != id,
    decreases recs.len(),
{
    if i < recs.len() - 1 {
        find_sink_record_none(recs.drop_last(), id, i);
    }
}

proof fn find_sink_record_unique_idx(recs: Seq<ExecSinkRecord>, idx: int)
    requires
        sink_records_unique_ids(recs),
        0 <= idx < recs.len(),
    ensures
        find_sink_record_spec(recs, recs[idx].permit@.id) == Some(idx),
    decreases recs.len(),
{
    let last_i = recs.len() - 1;
    if idx == last_i {
        assert(recs.last().permit@.id == recs[idx].permit@.id);
    } else {
        assert(recs[idx].permit@.id != recs[last_i].permit@.id);
        let pref = recs.drop_last();
        assert forall|i: int, j: int|
            0 <= i < j < pref.len() implies (#[trigger] pref[i].permit@.id)
            != (#[trigger] pref[j].permit@.id) by {
            assert(pref[i] == recs[i]);
            assert(pref[j] == recs[j]);
        }
        find_sink_record_unique_idx(pref, idx);
    }
}

proof fn find_sink_record_push(recs: Seq<ExecSinkRecord>, rec: ExecSinkRecord, id: Seq<u8>)
    ensures
        find_sink_record_spec(recs.push(rec), id) == if rec.permit@.id == id {
            Some(recs.len() as int)
        } else {
            find_sink_record_spec(recs, id)
        },
{
    assert(recs.push(rec).drop_last() =~= recs);
}

proof fn find_sink_record_update_same_permit(
    recs: Seq<ExecSinkRecord>,
    idx: int,
    rec: ExecSinkRecord,
    id: Seq<u8>,
)
    requires
        0 <= idx < recs.len(),
        rec.permit@.id == recs[idx].permit@.id,
    ensures
        find_sink_record_spec(recs.update(idx, rec), id) == find_sink_record_spec(recs, id),
    decreases recs.len(),
{
    let u = recs.update(idx, rec);
    if idx == recs.len() - 1 {
        assert(u.drop_last() =~= recs.drop_last());
    } else {
        assert(u.last() == recs.last());
        assert(u.drop_last() =~= recs.drop_last().update(idx, rec));
        find_sink_record_update_same_permit(recs.drop_last(), idx, rec, id);
    }
}

proof fn sink_sum_on_commit(recs: Seq<ExecSinkRecord>, idx: int, rec: ExecSinkRecord)
    requires
        0 <= idx < recs.len(),
        recs[idx].status == ExecSlotStatus::Pending,
        rec.status == ExecSlotStatus::Committed,
        rec.permit@.amount == recs[idx].permit@.amount,
    ensures
        sink_committed_sum(recs.update(idx, rec))
            == sink_committed_sum(recs) + recs[idx].permit@.amount as nat,
    decreases recs.len(),
{
    let u = recs.update(idx, rec);
    if idx == recs.len() - 1 {
        assert(u.drop_last() =~= recs.drop_last());
    } else {
        assert(u.last() == recs.last());
        assert(u.drop_last() =~= recs.drop_last().update(idx, rec));
        sink_sum_on_commit(recs.drop_last(), idx, rec);
    }
}

proof fn sink_sum_on_cancel_pending(recs: Seq<ExecSinkRecord>, idx: int, rec: ExecSinkRecord)
    requires
        0 <= idx < recs.len(),
        recs[idx].status == ExecSlotStatus::Pending,
        rec.status == ExecSlotStatus::Cancelled,
    ensures
        sink_committed_sum(recs.update(idx, rec)) == sink_committed_sum(recs),
    decreases recs.len(),
{
    let u = recs.update(idx, rec);
    if idx == recs.len() - 1 {
        assert(u.drop_last() =~= recs.drop_last());
    } else {
        assert(u.last() == recs.last());
        assert(u.drop_last() =~= recs.drop_last().update(idx, rec));
        sink_sum_on_cancel_pending(recs.drop_last(), idx, rec);
    }
}

/// Cross-kernel dominance lemma: whenever every active (`Pending` or `Committed`) sink
/// record maps by unique permit ID to a non-cancelled broker record of equal amount,
/// the sum of committed sink spend plus any chosen pending sink record is bounded by
/// the broker's active (`committed + pending`) charge.
pub proof fn sink_committed_le_broker_active(
    s_recs: Seq<ExecSinkRecord>,
    b_recs: Seq<ExecBrokerRecord>,
)
    requires
        sink_records_unique_ids(s_recs),
        forall|i: int|
            0 <= i < s_recs.len() && (#[trigger] s_recs[i]).status == ExecSlotStatus::Committed
                ==> ({
                    let opt = find_broker_record_spec(b_recs, s_recs[i].permit@.id);
                    opt is Some
                        && 0 <= opt.unwrap() < b_recs.len()
                        && b_recs[opt.unwrap()].permit@.id == s_recs[i].permit@.id
                        && b_recs[opt.unwrap()].permit@.amount == s_recs[i].permit@.amount
                        && b_recs[opt.unwrap()].status != ExecSlotStatus::Cancelled
                }),
    ensures
        sink_committed_sum(s_recs) <= broker_active_sum(b_recs),
    decreases s_recs.len(),
{
    if s_recs.len() > 0 {
        let prefix = s_recs.drop_last();
        let last = s_recs.last();
        assert forall|i: int, j: int|
            0 <= i < j < prefix.len() implies (#[trigger] prefix[i].permit@.id)
            != (#[trigger] prefix[j].permit@.id) by {
            assert(prefix[i] == s_recs[i]);
            assert(prefix[j] == s_recs[j]);
        }
        if last.status != ExecSlotStatus::Committed {
            assert forall|i: int|
                0 <= i < prefix.len()
                    && (#[trigger] prefix[i]).status == ExecSlotStatus::Committed implies ({
                let opt = find_broker_record_spec(b_recs, prefix[i].permit@.id);
                opt is Some
                    && 0 <= opt.unwrap() < b_recs.len()
                    && b_recs[opt.unwrap()].permit@.id == prefix[i].permit@.id
                    && b_recs[opt.unwrap()].permit@.amount == prefix[i].permit@.amount
                    && b_recs[opt.unwrap()].status != ExecSlotStatus::Cancelled
            }) by {
                assert(prefix[i] == s_recs[i]);
            }
            sink_committed_le_broker_active(prefix, b_recs);
        } else {
            let last_idx = s_recs.len() - 1;
            assert(s_recs[last_idx] == last);
            let b_idx = find_broker_record_spec(b_recs, last.permit@.id).unwrap();
            let ghost_cancelled = ExecBrokerRecord {
                status: ExecSlotStatus::Cancelled,
                ..b_recs[b_idx]
            };
            let b_minus = b_recs.update(b_idx, ghost_cancelled);
            broker_sums_on_cancel(b_recs, b_idx, ghost_cancelled);

            assert forall|i: int|
                0 <= i < prefix.len()
                    && (#[trigger] prefix[i]).status == ExecSlotStatus::Committed implies ({
                let opt = find_broker_record_spec(b_minus, prefix[i].permit@.id);
                opt is Some
                    && 0 <= opt.unwrap() < b_minus.len()
                    && b_minus[opt.unwrap()].permit@.id == prefix[i].permit@.id
                    && b_minus[opt.unwrap()].permit@.amount == prefix[i].permit@.amount
                    && b_minus[opt.unwrap()].status != ExecSlotStatus::Cancelled
            }) by {
                assert(prefix[i] == s_recs[i]);
                assert(s_recs[i].permit@.id != s_recs[last_idx].permit@.id);
                find_broker_record_update_same_permit(
                    b_recs,
                    b_idx,
                    ghost_cancelled,
                    prefix[i].permit@.id,
                );
                let opt = find_broker_record_spec(b_recs, prefix[i].permit@.id);
                assert(opt.unwrap() != b_idx);
                assert(b_minus[opt.unwrap()] == b_recs[opt.unwrap()]);
            }
            sink_committed_le_broker_active(prefix, b_minus);
        }
    }
}

#[verifier::opaque]
pub open spec fn broker_rec_to_inflight_inv(
    recs: Seq<ExecBrokerRecord>,
    ridxs: Seq<usize>,
    budget: u64,
) -> bool {
    forall|r: int|
        0 <= r < recs.len() ==> ((#[trigger] recs[r]).permit@.amount > 0
            && recs[r].permit@.amount <= budget
            && (recs[r].status == ExecSlotStatus::Pending ==> ({
                let k = recs[r].inflight_pos as int;
                0 <= k < ridxs.len() && ridxs[k] as int == r
            })))
}

#[verifier::opaque]
pub open spec fn sink_records_issued_by_broker(
    s_recs: Seq<ExecSinkRecord>,
    b_recs: Seq<ExecBrokerRecord>,
) -> bool {
    forall|s_idx: int|
        0 <= s_idx < s_recs.len() ==> ({
            let sr = #[trigger] s_recs[s_idx];
            let b_opt = find_broker_record_spec(b_recs, sr.permit@.id);
            b_opt is Some
                && 0 <= b_opt.unwrap() < b_recs.len()
                && b_recs[b_opt.unwrap()].permit@ == sr.permit@
        })
}

#[verifier::opaque]
pub open spec fn broker_terminal_matches_sink(
    b_recs: Seq<ExecBrokerRecord>,
    s_recs: Seq<ExecSinkRecord>,
) -> bool {
    forall|b_idx: int|
        0 <= b_idx < b_recs.len() ==> ({
            let br = #[trigger] b_recs[b_idx];
            (br.status == ExecSlotStatus::Committed ==> ({
                let s_opt = find_sink_record_spec(s_recs, br.permit@.id);
                s_opt is Some
                    && 0 <= s_opt.unwrap() < s_recs.len()
                    && s_recs[s_opt.unwrap()].status == ExecSlotStatus::Committed
            })) && (br.status == ExecSlotStatus::Cancelled ==> ({
                let s_opt = find_sink_record_spec(s_recs, br.permit@.id);
                s_opt is Some
                    && 0 <= s_opt.unwrap() < s_recs.len()
                    && s_recs[s_opt.unwrap()].status == ExecSlotStatus::Cancelled
            }))
        })
}

#[verifier::opaque]
pub open spec fn ridxs_pairwise_distinct(ridxs: Seq<usize>) -> bool {
    forall|k1: int, k2: int|
        0 <= k1 < k2 < ridxs.len() ==> (#[trigger] ridxs[k1]) != (#[trigger] ridxs[k2])
}

impl ExecBrokerKernel {
    pub open spec fn broker_inv(&self) -> bool {
        self.live@.len() == self.epochs@.len()
            && self.epochs@.len() > 0
            && self.charged as nat == self.committed as nat + self.pending as nat
            && self.charged <= self.budget
            && self.committed as nat == broker_committed_sum(self.records@)
            && self.pending as nat == broker_pending_sum(self.records@)
            && self.pending as nat == sum(seq_u64_to_nat(self.inflight_amounts@))
            && self.inflight_ridxs@.len() == self.inflight_amounts@.len()
            && broker_records_unique_ids(self.records@)
            && ridxs_pairwise_distinct(self.inflight_ridxs@)
            && (forall|k: int|
                0 <= k < self.inflight_ridxs@.len() ==> ({
                    let r = #[trigger] self.inflight_ridxs@[k] as int;
                    0 <= r < self.records@.len()
                        && self.records@[r].status == ExecSlotStatus::Pending
                        && self.records@[r].inflight_pos as int == k
                        && self.records@[r].permit@.amount == self.inflight_amounts@[k]
                        && self.inflight_amounts@[k] > 0
                }))
            && broker_rec_to_inflight_inv(self.records@, self.inflight_ridxs@, self.budget)
    }

    pub open spec fn project_slot(&self, id: Seq<u8>) -> LedgerState {
        slot_ledger_projection(self.budget, self.records@, id)
    }

    pub fn broker_new(budget: u64, issuers: usize) -> (b: Self)
        requires
            issuers > 0,
        ensures
            b.broker_inv(),
            b.epochs@.len() == issuers as nat,
            b.budget == budget,
            b.charged == 0,
            b.committed == 0,
            b.pending == 0,
            b.records@.len() == 0,
            b@ == (LedgerState {
                budget: budget as nat,
                charged: 0,
                committed: 0,
                inflight: Seq::empty(),
            }),
            ledger_inv(b@),
    {
        let mut live: Vec<bool> = Vec::new();
        let mut epochs: Vec<u64> = Vec::new();
        let mut i: usize = 0;
        while i < issuers
            invariant
                i <= issuers,
                live@.len() == i as nat,
                epochs@.len() == i as nat,
            decreases issuers - i,
        {
            live.push(true);
            epochs.push(0);
            i += 1;
        }
        let b = ExecBrokerKernel {
            budget,
            charged: 0,
            committed: 0,
            pending: 0,
            live,
            epochs,
            records: Vec::new(),
            inflight_ridxs: Vec::new(),
            inflight_amounts: Vec::new(),
        };
        assert(seq_u64_to_nat(b.inflight_amounts@) =~= Seq::<nat>::empty());
        assert(ridxs_pairwise_distinct(b.inflight_ridxs@)) by {
            reveal(ridxs_pairwise_distinct);
        }
        assert(broker_rec_to_inflight_inv(b.records@, b.inflight_ridxs@, b.budget)) by {
            reveal(broker_rec_to_inflight_inv);
        }
        b
    }

    fn find_record_exec(&self, id: &Vec<u8>) -> (res: Option<usize>)
        requires
            self.broker_inv(),
        ensures
            match res {
                Some(idx) => {
                    (idx as int) < self.records@.len()
                        && find_broker_record_spec(self.records@, id@) == Some(idx as int)
                        && self.records@[idx as int].permit@.id == id@
                }
                None => find_broker_record_spec(self.records@, id@) is None,
            },
    {
        let mut i: usize = 0;
        while i < self.records.len()
            invariant
                self.broker_inv(),
                i <= self.records.len(),
                forall|j: int| 0 <= j < i as int ==> (#[trigger] self.records@[j]).permit@.id != id@,
            decreases self.records.len() - i,
        {
            if same_bytes(&self.records[i].permit.id, id) {
                proof {
                    find_broker_record_unique_idx(self.records@, i as int);
                }
                return Some(i);
            }
            i += 1;
        }
        proof {
            if find_broker_record_spec(self.records@, id@) is Some {
                find_broker_record_some(self.records@, id@);
                assert(false);
            }
        }
        None
    }

    pub fn broker_withdraw(&mut self, issuer: usize) -> (ok: bool)
        requires
            old(self).broker_inv(),
        ensures
            final(self).broker_inv(),
            final(self).epochs@.len() == old(self).epochs@.len(),
            final(self).records@ == old(self).records@,
            final(self).budget == old(self).budget,
            final(self).charged == old(self).charged,
            final(self).committed == old(self).committed,
            final(self).pending == old(self).pending,
            final(self)@ == step_ledger(old(self)@, LedgerStep::Withdraw),
            forall|id: Seq<u8>|
                #[trigger] final(self).project_slot(id)
                    == step_ledger(old(self).project_slot(id), LedgerStep::Withdraw),
            ok == (issuer < old(self).epochs@.len() && old(self).epochs@[issuer as int] < u64::MAX),
            ok ==> (!final(self).live@[issuer as int]
                && final(self).epochs@[issuer as int] == old(self).epochs@[issuer as int] + 1),
    {
        if issuer >= self.epochs.len() || self.epochs[issuer] == u64::MAX {
            return false;
        }
        let next_epoch = self.epochs[issuer] + 1;
        self.epochs.set(issuer, next_epoch);
        self.live.set(issuer, false);
        true
    }

    pub fn broker_renew(&mut self, issuer: usize) -> (ok: bool)
        requires
            old(self).broker_inv(),
        ensures
            final(self).broker_inv(),
            final(self).epochs@.len() == old(self).epochs@.len(),
            final(self).records@ == old(self).records@,
            final(self).budget == old(self).budget,
            final(self).charged == old(self).charged,
            final(self).committed == old(self).committed,
            final(self).pending == old(self).pending,
            final(self)@ == step_ledger(old(self)@, LedgerStep::Renew),
            forall|id: Seq<u8>|
                #[trigger] final(self).project_slot(id)
                    == step_ledger(old(self).project_slot(id), LedgerStep::Renew),
            ok == (issuer < old(self).epochs@.len()),
            ok ==> (final(self).live@[issuer as int]
                && final(self).epochs@ == old(self).epochs@),
    {
        if issuer >= self.epochs.len() {
            return false;
        }
        self.live.set(issuer, true);
        true
    }

    pub fn broker_uncertain_timeout(&mut self)
        requires
            old(self).broker_inv(),
        ensures
            final(self).broker_inv(),
            final(self).epochs@.len() == old(self).epochs@.len(),
            final(self).records@ == old(self).records@,
            final(self).budget == old(self).budget,
            final(self).charged == old(self).charged,
            final(self).committed == old(self).committed,
            final(self).pending == old(self).pending,
            final(self)@ == step_ledger(old(self)@, LedgerStep::UncertainTimeout),
            forall|id: Seq<u8>|
                #[trigger] final(self).project_slot(id)
                    == step_ledger(old(self).project_slot(id), LedgerStep::UncertainTimeout),
    {
    }

    pub fn broker_reserve(
        &mut self,
        permit: &ExecPermitWire,
        approved: &ExecPermitWire,
        grants: &Vec<bool>,
        now: u64,
    ) -> (out: ExecReserveOutcome)
        requires
            old(self).broker_inv(),
        ensures
            final(self).broker_inv(),
            final(self).epochs@.len() == old(self).epochs@.len(),
            final(self).budget == old(self).budget,
            match out {
                ExecReserveOutcome::ReservedNew { slot } => {
                    permit@.amount > 0
                        && now < permit@.expires
                        && permit@ == approved@
                        && current_grants_spec(
                            permit@.required,
                            grants@,
                            old(self).live@,
                            old(self).epochs@,
                            permit@.epochs,
                        )
                        && find_broker_record_spec(old(self).records@, permit@.id) is None
                        && find_broker_record_spec(final(self).records@, permit@.id)
                            == Some(old(self).records@.len() as int)
                        && final(self).records@.len() == old(self).records@.len() + 1
                        && (forall|r: int|
                            0 <= r < old(self).records@.len() ==> #[trigger] final(self).records@[r]
                                == old(self).records@[r])
                        && final(self).records@[old(self).records@.len() as int].permit@
                            == permit@
                        && final(self).records@[old(self).records@.len() as int].status
                            == ExecSlotStatus::Pending
                        && slot as int == old(self).inflight_amounts@.len()
                        && final(self)@
                            == step_ledger(
                                old(self)@,
                                LedgerStep::Reserve { amount: permit@.amount as nat },
                            )
                        && final(self).project_slot(permit@.id)
                            == step_ledger(
                                old(self).project_slot(permit@.id),
                                LedgerStep::Reserve { amount: permit@.amount as nat },
                            )
                }
                ExecReserveOutcome::DuplicateExisting { status } => {
                    find_broker_record_spec(old(self).records@, permit@.id) is Some
                        && old(self).records@[find_broker_record_spec(
                            old(self).records@,
                            permit@.id,
                        ).unwrap()].permit@ == permit@
                        && status
                            == old(self).records@[find_broker_record_spec(
                                old(self).records@,
                                permit@.id,
                            ).unwrap()].status
                        && final(self).records@ == old(self).records@
                        && final(self)@ == old(self)@
                        && final(self).project_slot(permit@.id)
                            == old(self).project_slot(permit@.id)
                }
                ExecReserveOutcome::PayloadMismatch | ExecReserveOutcome::DeniedPolicy => {
                    final(self).records@ == old(self).records@
                        && final(self)@ == old(self)@
                        && final(self).project_slot(permit@.id)
                            == old(self).project_slot(permit@.id)
                }
            },
            forall|other_id: Seq<u8>|
                other_id != permit@.id ==> #[trigger] final(self).project_slot(other_id)
                    == old(self).project_slot(other_id),
    {
        if let Some(existing_idx) = self.find_record_exec(&permit.id) {
            if same_permit_canonical(&self.records[existing_idx].permit, permit) {
                return ExecReserveOutcome::DuplicateExisting {
                    status: self.records[existing_idx].status,
                };
            } else {
                return ExecReserveOutcome::PayloadMismatch;
            }
        }

        let next_charged_opt = reserve(self.budget, self.charged, permit.amount);
        let ok = next_charged_opt.is_some()
            && now < permit.expires
            && same_permit_canonical(permit, approved)
            && current_grants(
                &permit.required,
                grants,
                &self.live,
                &self.epochs,
                &permit.epochs,
            );
        if !ok {
            return ExecReserveOutcome::DeniedPolicy;
        }

        let next_charged = next_charged_opt.unwrap();
        let r_idx = self.records.len();
        let slot = self.inflight_amounts.len();
        let rec = ExecBrokerRecord {
            permit: clone_permit_wire(permit),
            status: ExecSlotStatus::Pending,
            inflight_pos: slot,
        };

        proof {
            let old_recs = self.records@;
            assert forall|i: int| 0 <= i < old_recs.len() implies (#[trigger] old_recs[i]).permit@.id
                != rec.permit@.id by {
                find_broker_record_none(old_recs, permit@.id, i);
            }
            seq_u64_to_nat_push(self.inflight_amounts@, permit.amount);
            sum_push(seq_u64_to_nat(self.inflight_amounts@), permit.amount as nat);
            find_broker_record_push(old_recs, rec, permit@.id);
        }

        self.charged = next_charged;
        self.pending += permit.amount;
        self.records.push(rec);
        self.inflight_ridxs.push(r_idx);
        self.inflight_amounts.push(permit.amount);

        proof {
            assert(self.records@.drop_last() =~= old(self).records@);
            assert(ridxs_pairwise_distinct(self.inflight_ridxs@)) by {
                reveal(ridxs_pairwise_distinct);
                assert forall|k1: int, k2: int|
                    0 <= k1 < k2 < self.inflight_ridxs@.len()
                    implies (#[trigger] self.inflight_ridxs@[k1])
                    != (#[trigger] self.inflight_ridxs@[k2]) by {
                    if k2 < old(self).inflight_ridxs@.len() {
                        assert(self.inflight_ridxs@[k1] == old(self).inflight_ridxs@[k1]);
                        assert(self.inflight_ridxs@[k2] == old(self).inflight_ridxs@[k2]);
                    } else {
                        assert(self.inflight_ridxs@[k1] == old(self).inflight_ridxs@[k1]);
                        assert((old(self).inflight_ridxs@[k1] as int) < old(self).records@.len());
                        assert(self.inflight_ridxs@[k2] == r_idx);
                    }
                }
            }
            assert(broker_rec_to_inflight_inv(
                self.records@,
                self.inflight_ridxs@,
                self.budget,
            )) by {
                reveal(broker_rec_to_inflight_inv);
                assert forall|r: int|
                    0 <= r < self.records@.len()
                    implies ((#[trigger] self.records@[r]).permit@.amount > 0
                    && self.records@[r].permit@.amount <= self.budget
                    && (self.records@[r].status == ExecSlotStatus::Pending ==> ({
                        let k = self.records@[r].inflight_pos as int;
                        0 <= k < self.inflight_ridxs@.len() && self.inflight_ridxs@[k] as int == r
                    }))) by {
                    if r < old(self).records@.len() {
                        assert(self.records@[r] == old(self).records@[r]);
                        if self.records@[r].status == ExecSlotStatus::Pending {
                            let k = self.records@[r].inflight_pos as int;
                            assert(self.inflight_ridxs@[k] == old(self).inflight_ridxs@[k]);
                        }
                    }
                }
            }
            assert forall|other_id: Seq<u8>|
                other_id != permit@.id implies #[trigger] self.project_slot(other_id)
                == old(self).project_slot(other_id) by {
                find_broker_record_push(old(self).records@, rec, other_id);
                if find_broker_record_spec(old(self).records@, other_id) is Some {
                    find_broker_record_some(old(self).records@, other_id);
                }
            }
            assert(Seq::<nat>::empty().push(permit@.amount as nat) =~= seq![permit@.amount as nat]);
        }

        ExecReserveOutcome::ReservedNew { slot }
    }

    fn remove_inflight_at(&mut self, slot: usize, target_r: usize, new_status: ExecSlotStatus)
        requires
            old(self).broker_inv(),
            new_status != ExecSlotStatus::Pending,
            (slot as int) < old(self).inflight_ridxs@.len(),
            old(self).inflight_ridxs@[slot as int] == target_r,
            (target_r as int) < old(self).records@.len(),
            old(self).records@[target_r as int].status == ExecSlotStatus::Pending,
        ensures
            final(self).live@ == old(self).live@,
            final(self).epochs@ == old(self).epochs@,
            final(self).budget == old(self).budget,
            final(self).charged == old(self).charged,
            final(self).committed == old(self).committed,
            final(self).pending == old(self).pending,
            final(self).inflight_ridxs@ == old(self).inflight_ridxs@.remove(slot as int),
            final(self).inflight_amounts@ == old(self).inflight_amounts@.remove(slot as int),
            final(self).records@.len() == old(self).records@.len(),
            broker_records_unique_ids(final(self).records@),
            forall|r: int|
                0 <= r < final(self).records@.len() ==> ((#[trigger] final(self).records@[r]).permit@
                    == old(self).records@[r].permit@
                    && final(self).records@[r].status
                        == (if r == target_r as int {
                            new_status
                        } else {
                            old(self).records@[r].status
                        })),
            new_status == ExecSlotStatus::Committed ==> (
                broker_committed_sum(final(self).records@)
                    == broker_committed_sum(old(self).records@)
                        + old(self).records@[target_r as int].permit@.amount as nat
                && broker_pending_sum(old(self).records@)
                    == broker_pending_sum(final(self).records@)
                        + old(self).records@[target_r as int].permit@.amount as nat
            ),
            new_status == ExecSlotStatus::Cancelled ==> (
                broker_committed_sum(final(self).records@)
                    == broker_committed_sum(old(self).records@)
                && broker_pending_sum(old(self).records@)
                    == broker_pending_sum(final(self).records@)
                        + old(self).records@[target_r as int].permit@.amount as nat
            ),
            ridxs_pairwise_distinct(final(self).inflight_ridxs@),
            forall|k: int|
                0 <= k < final(self).inflight_ridxs@.len() ==> ({
                    let r = #[trigger] final(self).inflight_ridxs@[k] as int;
                    0 <= r < final(self).records@.len()
                        && final(self).records@[r].status == ExecSlotStatus::Pending
                        && final(self).records@[r].inflight_pos as int == k
                        && final(self).records@[r].permit@.amount
                            == final(self).inflight_amounts@[k]
                        && final(self).inflight_amounts@[k] > 0
                }),
            broker_rec_to_inflight_inv(
                final(self).records@,
                final(self).inflight_ridxs@,
                final(self).budget,
            ),
    {
        let settled_rec = ExecBrokerRecord {
            permit: clone_permit_wire(&self.records[target_r].permit),
            status: new_status,
            inflight_pos: 0,
        };
        proof {
            if matches!(new_status, ExecSlotStatus::Committed) {
                broker_sums_on_commit(self.records@, target_r as int, settled_rec);
            } else {
                broker_sums_on_cancel(self.records@, target_r as int, settled_rec);
            }
        }
        self.records.set(target_r, settled_rec);
        let ghost pre_recs = self.records@;
        let ghost pre_ridxs = self.inflight_ridxs@;
        let ghost pre_amts = self.inflight_amounts@;

        let n = self.inflight_ridxs.len();
        proof {
            assert forall|k: int| 0 <= k < n as int && k != slot as int implies ({
                let r = #[trigger] pre_ridxs[k] as int;
                0 <= r < pre_recs.len()
                    && r != target_r as int
                    && pre_recs[r].status == ExecSlotStatus::Pending
                    && pre_recs[r].permit@.amount == pre_amts[k]
                    && pre_amts[k] > 0
                    && self.records@[r].inflight_pos as int == k
            }) by {
                assert(pre_ridxs[k] != pre_ridxs[slot as int]) by {
                    reveal(ridxs_pairwise_distinct);
                    if k < slot as int {
                        assert(pre_ridxs[k] != pre_ridxs[slot as int]);
                    } else {
                        assert(pre_ridxs[slot as int] != pre_ridxs[k]);
                    }
                };
                let r = pre_ridxs[k] as int;
                assert(r != target_r as int);
                assert(pre_recs[r] == old(self).records@[r]);
            }
        }

        let mut i: usize = slot;
        while i + 1 < n
            invariant
                slot <= i < n,
                n == pre_ridxs.len(),
                pre_ridxs.len() == pre_amts.len(),
                self.inflight_ridxs.len() == n,
                self.inflight_amounts.len() == n,
                self.records.len() == pre_recs.len(),
                pre_recs.len() == old(self).records@.len(),
                self.live@ == old(self).live@,
                self.epochs@ == old(self).epochs@,
                self.budget == old(self).budget,
                self.charged == old(self).charged,
                self.committed == old(self).committed,
                self.pending == old(self).pending,
                ridxs_pairwise_distinct(pre_ridxs),
                forall|k: int|
                    0 <= k < n as int && k != slot as int ==> ({
                        let r = #[trigger] pre_ridxs[k] as int;
                        0 <= r < pre_recs.len()
                            && r != target_r as int
                            && pre_recs[r].status == ExecSlotStatus::Pending
                            && pre_recs[r].permit@.amount == pre_amts[k]
                            && pre_amts[k] > 0
                            && (if (slot as int) < k && k <= (i as int) {
                                self.records@[r].inflight_pos as int == k - 1
                            } else {
                                self.records@[r].inflight_pos as int == k
                            })
                    }),
                forall|k: int| 0 <= k < slot as int ==> #[trigger] self.inflight_ridxs@[k] == pre_ridxs[k],
                forall|k: int| 0 <= k < slot as int ==> #[trigger] self.inflight_amounts@[k] == pre_amts[k],
                forall|k: int|
                    (slot as int) <= k && k < (i as int) ==> #[trigger] self.inflight_ridxs@[k]
                        == pre_ridxs[k + 1],
                forall|k: int|
                    (slot as int) <= k && k < (i as int) ==> #[trigger] self.inflight_amounts@[k]
                        == pre_amts[k + 1],
                forall|k: int|
                    (i as int) + 1 <= k && k < (n as int) ==> #[trigger] self.inflight_ridxs@[k]
                        == pre_ridxs[k],
                forall|k: int|
                    (i as int) + 1 <= k && k < (n as int) ==> #[trigger] self.inflight_amounts@[k]
                        == pre_amts[k],
                forall|r: int|
                    0 <= r < self.records@.len() ==> ((#[trigger] self.records@[r]).permit@
                        == pre_recs[r].permit@
                        && self.records@[r].status == pre_recs[r].status),
            decreases n - i,
        {
            proof {
                assert(self.inflight_ridxs@[i as int + 1] == pre_ridxs[i as int + 1]);
                assert(self.inflight_amounts@[i as int + 1] == pre_amts[i as int + 1]);
            }
            let moved_r = self.inflight_ridxs[i + 1];
            let moved_amt = self.inflight_amounts[i + 1];
            self.inflight_ridxs.set(i, moved_r);
            self.inflight_amounts.set(i, moved_amt);

            let ghost prev_recs = self.records@;
            let moved_rec = ExecBrokerRecord {
                permit: clone_permit_wire(&self.records[moved_r].permit),
                status: self.records[moved_r].status,
                inflight_pos: i,
            };
            self.records.set(moved_r, moved_rec);

            proof {
                assert forall|k: int| 0 <= k < n as int && k != slot as int implies ({
                    let r = #[trigger] pre_ridxs[k] as int;
                    0 <= r < pre_recs.len()
                        && r != target_r as int
                        && pre_recs[r].status == ExecSlotStatus::Pending
                        && pre_recs[r].permit@.amount == pre_amts[k]
                        && pre_amts[k] > 0
                        && (if (slot as int) < k && k <= (i as int) + 1 {
                            self.records@[r].inflight_pos as int == k - 1
                        } else {
                            self.records@[r].inflight_pos as int == k
                        })
                }) by {
                    let r = pre_ridxs[k] as int;
                    if k == (i as int) + 1 {
                        assert(r == moved_r as int);
                        assert(self.records@[r].inflight_pos == i);
                    } else {
                        assert(r != moved_r as int) by {
                            reveal(ridxs_pairwise_distinct);
                            if k < (i as int) + 1 {
                                assert(pre_ridxs[k] != pre_ridxs[(i as int) + 1]);
                            } else {
                                assert(pre_ridxs[(i as int) + 1] != pre_ridxs[k]);
                            }
                        };
                        assert(self.records@[r] == prev_recs[r]);
                    }
                }
            }
            i += 1;
        }

        let _ = self.inflight_ridxs.pop();
        let _ = self.inflight_amounts.pop();

        proof {
            assert(self.inflight_ridxs@ =~= pre_ridxs.remove(slot as int));
            assert(self.inflight_amounts@ =~= pre_amts.remove(slot as int));
            broker_sums_same_status_amount(pre_recs, self.records@);
            assert forall|i1: int, i2: int|
                0 <= i1 < i2 < self.records@.len() implies (#[trigger] self.records@[i1].permit@.id)
                != (#[trigger] self.records@[i2].permit@.id) by {
                assert(self.records@[i1].permit@.id == old(self).records@[i1].permit@.id);
                assert(self.records@[i2].permit@.id == old(self).records@[i2].permit@.id);
            }
            assert(ridxs_pairwise_distinct(self.inflight_ridxs@)) by {
                reveal(ridxs_pairwise_distinct);
                assert forall|k1: int, k2: int|
                    0 <= k1 < k2 < self.inflight_ridxs@.len()
                    implies (#[trigger] self.inflight_ridxs@[k1])
                    != (#[trigger] self.inflight_ridxs@[k2]) by {
                    let o1 = if k1 < slot as int { k1 } else { k1 + 1 };
                    let o2 = if k2 < slot as int { k2 } else { k2 + 1 };
                    assert(self.inflight_ridxs@[k1] == pre_ridxs[o1]);
                    assert(self.inflight_ridxs@[k2] == pre_ridxs[o2]);
                }
            }
            assert forall|k: int| 0 <= k < self.inflight_ridxs@.len() implies ({
                let r = #[trigger] self.inflight_ridxs@[k] as int;
                0 <= r < self.records@.len()
                    && self.records@[r].status == ExecSlotStatus::Pending
                    && self.records@[r].inflight_pos as int == k
                    && self.records@[r].permit@.amount == self.inflight_amounts@[k]
                    && self.inflight_amounts@[k] > 0
            }) by {
                let ok = if k < slot as int { k } else { k + 1 };
                assert(self.inflight_ridxs@[k] == pre_ridxs[ok]);
                assert(self.inflight_amounts@[k] == pre_amts[ok]);
            }
            assert(broker_rec_to_inflight_inv(
                self.records@,
                self.inflight_ridxs@,
                self.budget,
            )) by {
                reveal(broker_rec_to_inflight_inv);
                assert forall|r: int| 0 <= r < self.records@.len() implies (
                    (#[trigger] self.records@[r]).permit@.amount > 0
                        && self.records@[r].permit@.amount <= self.budget
                        && (self.records@[r].status == ExecSlotStatus::Pending ==> ({
                            let k = self.records@[r].inflight_pos as int;
                            0 <= k < self.inflight_ridxs@.len() && self.inflight_ridxs@[k] as int == r
                        }))
                ) by {
                    assert(self.records@[r].permit@ == old(self).records@[r].permit@);
                    if self.records@[r].status == ExecSlotStatus::Pending {
                        assert(r != target_r as int);
                        assert(old(self).records@[r].status == ExecSlotStatus::Pending);
                        let ok = old(self).records@[r].inflight_pos as int;
                        assert(0 <= ok < n as int);
                        assert(pre_ridxs[ok] as int == r);
                        assert(ok != slot as int);
                        if ok < slot as int {
                            assert(self.records@[pre_ridxs[ok] as int].inflight_pos as int == ok);
                            assert(self.inflight_ridxs@[ok] == pre_ridxs[ok]);
                        } else {
                            assert(
                                self.records@[pre_ridxs[ok] as int].inflight_pos as int == ok - 1
                            );
                            assert(self.inflight_ridxs@[ok - 1] == pre_ridxs[ok]);
                        }
                    }
                }
            }
        }
    }

    pub fn broker_settle_committed(
        &mut self,
        id: &Vec<u8>,
        authoritative_sink_committed: bool,
    ) -> (out: ExecSettleOutcome)
        requires
            old(self).broker_inv(),
        ensures
            final(self).broker_inv(),
            final(self).epochs@.len() == old(self).epochs@.len(),
            final(self).budget == old(self).budget,
            !authoritative_sink_committed ==> out == ExecSettleOutcome::SinkNotAuthoritative,
            match out {
                ExecSettleOutcome::Settled { slot, amount } => {
                    authoritative_sink_committed
                        && find_broker_record_spec(old(self).records@, id@) is Some
                        && old(self).records@[find_broker_record_spec(
                            old(self).records@,
                            id@,
                        ).unwrap()].status == ExecSlotStatus::Pending
                        && amount
                            == old(self).records@[find_broker_record_spec(
                                old(self).records@,
                                id@,
                            ).unwrap()].permit@.amount
                        && (slot as int) < old(self).inflight_amounts@.len()
                        && old(self).inflight_amounts@[slot as int] == amount
                        && final(self).records@.len() == old(self).records@.len()
                        && (forall|r: int|
                            0 <= r < final(self).records@.len()
                                ==> ((#[trigger] final(self).records@[r]).permit@
                                    == old(self).records@[r].permit@
                                    && final(self).records@[r].status
                                        == (if r
                                            == find_broker_record_spec(
                                                old(self).records@,
                                                id@,
                                            ).unwrap()
                                        {
                                            ExecSlotStatus::Committed
                                        } else {
                                            old(self).records@[r].status
                                        })))
                        && final(self)@
                            == step_ledger(
                                old(self)@,
                                LedgerStep::CommitSettle { slot: slot as int },
                            )
                        && final(self).project_slot(id@)
                            == step_ledger(
                                old(self).project_slot(id@),
                                LedgerStep::CommitSettle { slot: 0 },
                            )
                }
                _ => {
                    final(self).records@ == old(self).records@
                        && final(self)@ == old(self)@
                        && final(self).project_slot(id@) == old(self).project_slot(id@)
                }
            },
            forall|other_id: Seq<u8>|
                other_id != id@ ==> #[trigger] final(self).project_slot(other_id)
                    == old(self).project_slot(other_id),
    {
        if !authoritative_sink_committed {
            return ExecSettleOutcome::SinkNotAuthoritative;
        }
        let Some(r_idx) = self.find_record_exec(id) else {
            return ExecSettleOutcome::UnknownId;
        };
        let cur_status = self.records[r_idx].status;
        if !matches!(cur_status, ExecSlotStatus::Pending) {
            return ExecSettleOutcome::NotPending { status: cur_status };
        }
        let slot = self.records[r_idx].inflight_pos;
        let amount = self.records[r_idx].permit.amount;

        proof {
            assert((slot as int) < self.inflight_ridxs@.len()
                && self.inflight_ridxs@[slot as int] == r_idx) by {
                reveal(broker_rec_to_inflight_inv);
            }
            seq_u64_to_nat_len(self.inflight_amounts@);
            seq_u64_to_nat_index(self.inflight_amounts@, slot as int);
            sum_remove(seq_u64_to_nat(self.inflight_amounts@), slot as int);
            seq_u64_to_nat_remove(self.inflight_amounts@, slot as int);
        }

        self.remove_inflight_at(slot, r_idx, ExecSlotStatus::Committed);
        self.pending -= amount;
        self.committed += amount;

        proof {
            assert forall|qid: Seq<u8>|
                #[trigger] find_broker_record_spec(self.records@, qid)
                == find_broker_record_spec(old(self).records@, qid) by {
                assert(self.records@ =~= old(self).records@.update(
                    r_idx as int,
                    self.records@[r_idx as int],
                ) || self.records@.len() == old(self).records@.len());
                // Prove via uniqueness & None/Some preservation:
                if find_broker_record_spec(old(self).records@, qid) is Some {
                    let j = find_broker_record_spec(old(self).records@, qid).unwrap();
                    find_broker_record_some(old(self).records@, qid);
                    assert(self.records@[j].permit@.id == qid);
                    find_broker_record_unique_idx(self.records@, j);
                } else {
                    if find_broker_record_spec(self.records@, qid) is Some {
                        let j = find_broker_record_spec(self.records@, qid).unwrap();
                        find_broker_record_some(self.records@, qid);
                        find_broker_record_none(old(self).records@, qid, j);
                        assert(false);
                    }
                }
            }
            assert(seq![amount as nat].remove(0) =~= Seq::<nat>::empty());
            assert forall|other_id: Seq<u8>|
                other_id != id@ implies #[trigger] self.project_slot(other_id)
                == old(self).project_slot(other_id) by {
                if find_broker_record_spec(old(self).records@, other_id) is Some {
                    let j = find_broker_record_spec(old(self).records@, other_id).unwrap();
                    find_broker_record_some(old(self).records@, other_id);
                    assert(j != r_idx as int);
                }
            }
        }

        ExecSettleOutcome::Settled { slot, amount }
    }

    pub fn broker_settle_cancelled(
        &mut self,
        id: &Vec<u8>,
        authoritative_sink_cancelled: bool,
    ) -> (out: ExecSettleOutcome)
        requires
            old(self).broker_inv(),
        ensures
            final(self).broker_inv(),
            final(self).epochs@.len() == old(self).epochs@.len(),
            final(self).budget == old(self).budget,
            !authoritative_sink_cancelled ==> out == ExecSettleOutcome::SinkNotAuthoritative,
            match out {
                ExecSettleOutcome::Settled { slot, amount } => {
                    authoritative_sink_cancelled
                        && find_broker_record_spec(old(self).records@, id@) is Some
                        && old(self).records@[find_broker_record_spec(
                            old(self).records@,
                            id@,
                        ).unwrap()].status == ExecSlotStatus::Pending
                        && amount
                            == old(self).records@[find_broker_record_spec(
                                old(self).records@,
                                id@,
                            ).unwrap()].permit@.amount
                        && (slot as int) < old(self).inflight_amounts@.len()
                        && old(self).inflight_amounts@[slot as int] == amount
                        && final(self).records@.len() == old(self).records@.len()
                        && (forall|r: int|
                            0 <= r < final(self).records@.len()
                                ==> ((#[trigger] final(self).records@[r]).permit@
                                    == old(self).records@[r].permit@
                                    && final(self).records@[r].status
                                        == (if r
                                            == find_broker_record_spec(
                                                old(self).records@,
                                                id@,
                                            ).unwrap()
                                        {
                                            ExecSlotStatus::Cancelled
                                        } else {
                                            old(self).records@[r].status
                                        })))
                        && final(self)@
                            == step_ledger(
                                old(self)@,
                                LedgerStep::CancelRelease { slot: slot as int },
                            )
                        && final(self).project_slot(id@)
                            == step_ledger(
                                old(self).project_slot(id@),
                                LedgerStep::CancelRelease { slot: 0 },
                            )
                }
                _ => {
                    final(self).records@ == old(self).records@
                        && final(self)@ == old(self)@
                        && final(self).project_slot(id@) == old(self).project_slot(id@)
                }
            },
            forall|other_id: Seq<u8>|
                other_id != id@ ==> #[trigger] final(self).project_slot(other_id)
                    == old(self).project_slot(other_id),
    {
        if !authoritative_sink_cancelled {
            return ExecSettleOutcome::SinkNotAuthoritative;
        }
        let Some(r_idx) = self.find_record_exec(id) else {
            return ExecSettleOutcome::UnknownId;
        };
        let cur_status = self.records[r_idx].status;
        if !matches!(cur_status, ExecSlotStatus::Pending) {
            return ExecSettleOutcome::NotPending { status: cur_status };
        }
        let slot = self.records[r_idx].inflight_pos;
        let amount = self.records[r_idx].permit.amount;

        proof {
            assert((slot as int) < self.inflight_ridxs@.len()
                && self.inflight_ridxs@[slot as int] == r_idx) by {
                reveal(broker_rec_to_inflight_inv);
            }
            seq_u64_to_nat_len(self.inflight_amounts@);
            seq_u64_to_nat_index(self.inflight_amounts@, slot as int);
            sum_remove(seq_u64_to_nat(self.inflight_amounts@), slot as int);
            seq_u64_to_nat_remove(self.inflight_amounts@, slot as int);
        }

        self.remove_inflight_at(slot, r_idx, ExecSlotStatus::Cancelled);
        self.charged = release(self.charged, amount, true);
        self.pending -= amount;

        proof {
            assert forall|qid: Seq<u8>|
                #[trigger] find_broker_record_spec(self.records@, qid)
                == find_broker_record_spec(old(self).records@, qid) by {
                if find_broker_record_spec(old(self).records@, qid) is Some {
                    let j = find_broker_record_spec(old(self).records@, qid).unwrap();
                    find_broker_record_some(old(self).records@, qid);
                    assert(self.records@[j].permit@.id == qid);
                    find_broker_record_unique_idx(self.records@, j);
                } else {
                    if find_broker_record_spec(self.records@, qid) is Some {
                        let j = find_broker_record_spec(self.records@, qid).unwrap();
                        find_broker_record_some(self.records@, qid);
                        find_broker_record_none(old(self).records@, qid, j);
                        assert(false);
                    }
                }
            }
            assert(seq![amount as nat].remove(0) =~= Seq::<nat>::empty());
            assert forall|other_id: Seq<u8>|
                other_id != id@ implies #[trigger] self.project_slot(other_id)
                == old(self).project_slot(other_id) by {
                if find_broker_record_spec(old(self).records@, other_id) is Some {
                    let j = find_broker_record_spec(old(self).records@, other_id).unwrap();
                    find_broker_record_some(old(self).records@, other_id);
                    assert(j != r_idx as int);
                }
            }
        }

        ExecSettleOutcome::Settled { slot, amount }
    }
}

impl ExecSinkKernel {
    pub open spec fn sink_wf(&self) -> bool {
        self.floors@.len() > 0
            && sink_records_unique_ids(self.records@)
            && self.committed_total as nat == sink_committed_sum(self.records@)
            && (forall|i: int|
                0 <= i < self.records@.len() ==> ((#[trigger] self.records@[i]).permit@.required.len()
                    == self.floors@.len()
                    && self.records@[i].permit@.epochs.len() == self.floors@.len()))
    }

    pub open spec fn sink_pending_unfenced(&self) -> bool {
        forall|i: int|
            0 <= i < self.records@.len() && (#[trigger] self.records@[i]).status == ExecSlotStatus::Pending
                ==> unfenced_spec(
                    self.records@[i].permit@.required,
                    self.records@[i].permit@.epochs,
                    self.floors@,
                )
    }

    pub open spec fn sink_inv(&self) -> bool {
        self.sink_wf() && self.sink_pending_unfenced()
    }

    pub fn sink_new(issuers: usize) -> (s: Self)
        requires
            issuers > 0,
        ensures
            s.sink_inv(),
            s.floors@.len() == issuers as nat,
            forall|k: int| 0 <= k < issuers as int ==> #[trigger] s.floors@[k] == 0,
            s.revision == 0,
            s.committed_total == 0,
            s.records@.len() == 0,
    {
        let mut floors: Vec<u64> = Vec::new();
        let mut i: usize = 0;
        while i < issuers
            invariant
                i <= issuers,
                floors@.len() == i as nat,
                forall|k: int| 0 <= k < i as int ==> #[trigger] floors@[k] == 0,
            decreases issuers - i,
        {
            floors.push(0);
            i += 1;
        }
        ExecSinkKernel {
            floors,
            revision: 0,
            committed_total: 0,
            records: Vec::new(),
        }
    }

    fn find_record_exec(&self, id: &Vec<u8>) -> (res: Option<usize>)
        requires
            self.sink_wf(),
        ensures
            match res {
                Some(idx) => {
                    (idx as int) < self.records@.len()
                        && find_sink_record_spec(self.records@, id@) == Some(idx as int)
                        && self.records@[idx as int].permit@.id == id@
                }
                None => find_sink_record_spec(self.records@, id@) is None,
            },
    {
        let mut i: usize = 0;
        while i < self.records.len()
            invariant
                self.sink_wf(),
                i <= self.records.len(),
                forall|j: int| 0 <= j < i as int ==> (#[trigger] self.records@[j]).permit@.id != id@,
            decreases self.records.len() - i,
        {
            if same_bytes(&self.records[i].permit.id, id) {
                proof {
                    find_sink_record_unique_idx(self.records@, i as int);
                }
                return Some(i);
            }
            i += 1;
        }
        proof {
            if find_sink_record_spec(self.records@, id@) is Some {
                find_sink_record_some(self.records@, id@);
                assert(false);
            }
        }
        None
    }

    pub fn sink_is_committed(&self, id: &Vec<u8>) -> (ok: bool)
        requires
            self.sink_wf(),
        ensures
            ok == (match find_sink_record_spec(self.records@, id@) {
                Some(idx) => self.records@[idx].status == ExecSlotStatus::Committed,
                None => false,
            }),
    {
        match self.find_record_exec(id) {
            Some(idx) => matches!(self.records[idx].status, ExecSlotStatus::Committed),
            None => false,
        }
    }

    pub fn sink_is_cancelled(&self, id: &Vec<u8>) -> (ok: bool)
        requires
            self.sink_wf(),
        ensures
            ok == (match find_sink_record_spec(self.records@, id@) {
                Some(idx) => self.records@[idx].status == ExecSlotStatus::Cancelled,
                None => false,
            }),
    {
        match self.find_record_exec(id) {
            Some(idx) => matches!(self.records[idx].status, ExecSlotStatus::Cancelled),
            None => false,
        }
    }

    pub fn sink_accept(&mut self, permit: &ExecPermitWire) -> (out: ExecAcceptOutcome)
        requires
            old(self).sink_inv(),
        ensures
            final(self).sink_inv(),
            final(self).floors@ == old(self).floors@,
            final(self).revision == old(self).revision,
            final(self).committed_total == old(self).committed_total,
            !(out is UnissuedPermit),
            match out {
                ExecAcceptOutcome::InvalidIssuerLength => {
                    (permit@.required.len() != old(self).floors@.len()
                        || permit@.epochs.len() != old(self).floors@.len())
                        && final(self).records@ == old(self).records@
                }
                ExecAcceptOutcome::Duplicate { status } => {
                    find_sink_record_spec(old(self).records@, permit@.id) is Some
                        && old(self).records@[find_sink_record_spec(
                            old(self).records@,
                            permit@.id,
                        ).unwrap()].permit@ == permit@
                        && status
                            == old(self).records@[find_sink_record_spec(
                                old(self).records@,
                                permit@.id,
                            ).unwrap()].status
                        && final(self).records@ == old(self).records@
                }
                ExecAcceptOutcome::PayloadMismatch => {
                    find_sink_record_spec(old(self).records@, permit@.id) is Some
                        && final(self).records@ == old(self).records@
                }
                ExecAcceptOutcome::AcceptedPending => {
                    find_sink_record_spec(old(self).records@, permit@.id) is None
                        && unfenced_spec(permit@.required, permit@.epochs, old(self).floors@)
                        && final(self).records@.len() == old(self).records@.len() + 1
                        && (forall|i: int|
                            0 <= i < old(self).records@.len() ==> #[trigger] final(self).records@[i]
                                == old(self).records@[i])
                        && final(self).records@[old(self).records@.len() as int].permit@ == permit@
                        && final(self).records@[old(self).records@.len() as int].status
                            == ExecSlotStatus::Pending
                }
                ExecAcceptOutcome::CancelledTombstone => {
                    find_sink_record_spec(old(self).records@, permit@.id) is None
                        && !unfenced_spec(permit@.required, permit@.epochs, old(self).floors@)
                        && final(self).records@.len() == old(self).records@.len() + 1
                        && (forall|i: int|
                            0 <= i < old(self).records@.len() ==> #[trigger] final(self).records@[i]
                                == old(self).records@[i])
                        && final(self).records@[old(self).records@.len() as int].permit@ == permit@
                        && final(self).records@[old(self).records@.len() as int].status
                            == ExecSlotStatus::Cancelled
                }
                ExecAcceptOutcome::UnissuedPermit => false,
            },
    {
        if permit.required.len() != self.floors.len() || permit.epochs.len() != self.floors.len() {
            return ExecAcceptOutcome::InvalidIssuerLength;
        }
        if let Some(idx) = self.find_record_exec(&permit.id) {
            if same_permit_canonical(&self.records[idx].permit, permit) {
                return ExecAcceptOutcome::Duplicate { status: self.records[idx].status };
            } else {
                return ExecAcceptOutcome::PayloadMismatch;
            }
        }
        let ok = unfenced(&permit.required, &permit.epochs, &self.floors);
        let status = if ok {
            ExecSlotStatus::Pending
        } else {
            ExecSlotStatus::Cancelled
        };
        let rec = ExecSinkRecord { permit: clone_permit_wire(permit), status };
        proof {
            let old_recs = self.records@;
            assert forall|i: int| 0 <= i < old_recs.len() implies (#[trigger] old_recs[i]).permit@.id
                != rec.permit@.id by {
                find_sink_record_none(old_recs, permit@.id, i);
            }
        }
        self.records.push(rec);
        assert(self.records@.drop_last() =~= old(self).records@);
        if ok {
            ExecAcceptOutcome::AcceptedPending
        } else {
            ExecAcceptOutcome::CancelledTombstone
        }
    }

    pub fn sink_fence(&mut self, issuer: usize, floor: u64) -> (ok: bool)
        requires
            old(self).sink_inv(),
        ensures
            final(self).sink_inv(),
            final(self).floors@.len() == old(self).floors@.len(),
            final(self).floors@ == apply_fence(old(self).floors@, issuer as int, floor),
            final(self).revision == old(self).revision,
            final(self).committed_total == old(self).committed_total,
            final(self).records@.len() == old(self).records@.len(),
            ok == (issuer < old(self).floors@.len()),
            ok ==> final(self).floors@[issuer as int] >= floor,
            forall|k: int|
                0 <= k < final(self).floors@.len() ==> #[trigger] final(self).floors@[k]
                    >= old(self).floors@[k],
            forall|i: int|
                0 <= i < final(self).records@.len() ==> ((#[trigger] final(self).records@[i]).permit@
                    == old(self).records@[i].permit@
                    && (old(self).records@[i].status == ExecSlotStatus::Committed
                        ==> final(self).records@[i].status == ExecSlotStatus::Committed)
                    && (old(self).records@[i].status == ExecSlotStatus::Cancelled
                        ==> final(self).records@[i].status == ExecSlotStatus::Cancelled)
                    && (old(self).records@[i].status == ExecSlotStatus::Pending ==> (
                        final(self).records@[i].status == ExecSlotStatus::Pending
                        || final(self).records@[i].status == ExecSlotStatus::Cancelled
                    ))
                    && (ok
                        && old(self).records@[i].permit@.required[issuer as int]
                        && old(self).records@[i].permit@.epochs[issuer as int] < floor
                        ==> final(self).records@[i].status != ExecSlotStatus::Pending)),
    {
        if issuer >= self.floors.len() {
            return false;
        }
        let next_floor = advance_floor(self.floors[issuer], floor);
        self.floors.set(issuer, next_floor);

        let mut i: usize = 0;
        while i < self.records.len()
            invariant
                i <= self.records.len(),
                issuer < self.floors.len(),
                self.floors@.len() == old(self).floors@.len(),
                self.floors@ == apply_fence(old(self).floors@, issuer as int, floor),
                self.revision == old(self).revision,
                self.committed_total == old(self).committed_total,
                self.committed_total as nat == sink_committed_sum(self.records@),
                self.records@.len() == old(self).records@.len(),
                sink_records_unique_ids(self.records@),
                forall|j: int|
                    0 <= j < self.records@.len() ==> ((#[trigger] self.records@[j]).permit@
                        == old(self).records@[j].permit@
                        && self.records@[j].permit@.required.len() == self.floors@.len()
                        && self.records@[j].permit@.epochs.len() == self.floors@.len()
                        && (old(self).records@[j].status == ExecSlotStatus::Committed
                            ==> self.records@[j].status == ExecSlotStatus::Committed)
                        && (old(self).records@[j].status == ExecSlotStatus::Cancelled
                            ==> self.records@[j].status == ExecSlotStatus::Cancelled)
                        && (old(self).records@[j].status == ExecSlotStatus::Pending ==> (
                            self.records@[j].status == ExecSlotStatus::Pending
                            || self.records@[j].status == ExecSlotStatus::Cancelled
                        ))),
                forall|j: int|
                    (i as int) <= j && j < self.records@.len()
                        ==> (#[trigger] self.records@[j]).status == old(self).records@[j].status,
                forall|j: int|
                    0 <= j < i as int && (#[trigger] self.records@[j]).status == ExecSlotStatus::Pending
                        ==> unfenced_spec(
                            self.records@[j].permit@.required,
                            self.records@[j].permit@.epochs,
                            self.floors@,
                        ),
            decreases self.records.len() - i,
        {
            if matches!(self.records[i].status, ExecSlotStatus::Pending)
                && !unfenced(
                    &self.records[i].permit.required,
                    &self.records[i].permit.epochs,
                    &self.floors,
                )
            {
                let cancelled_rec = ExecSinkRecord {
                    permit: clone_permit_wire(&self.records[i].permit),
                    status: ExecSlotStatus::Cancelled,
                };
                proof {
                    sink_sum_on_cancel_pending(self.records@, i as int, cancelled_rec);
                }
                self.records.set(i, cancelled_rec);
                proof {
                    assert forall|a: int, b: int|
                        0 <= a < b < self.records@.len()
                        implies (#[trigger] self.records@[a].permit@.id)
                        != (#[trigger] self.records@[b].permit@.id) by {
                        assert(self.records@[a].permit@.id == old(self).records@[a].permit@.id);
                        assert(self.records@[b].permit@.id == old(self).records@[b].permit@.id);
                    }
                }
            }
            i += 1;
        }
        proof {
            assert forall|j: int|
                0 <= j < self.records@.len()
                    && old(self).records@[j].permit@.required[issuer as int]
                    && old(self).records@[j].permit@.epochs[issuer as int] < floor
                implies (#[trigger] self.records@[j]).status != ExecSlotStatus::Pending by {
                if self.records@[j].status == ExecSlotStatus::Pending {
                    fenced_zero(
                        self.records@[j].permit@.required,
                        self.records@[j].permit@.epochs,
                        self.floors@,
                        issuer as int,
                    );
                    assert(false);
                }
            }
        }
        true
    }

    pub fn sink_complete(&mut self, id: &Vec<u8>) -> (out: ExecCompleteOutcome)
        requires
            old(self).sink_wf(),
        ensures
            final(self).sink_wf(),
            old(self).sink_inv() ==> final(self).sink_inv(),
            final(self).floors@ == old(self).floors@,
            final(self).records@.len() == old(self).records@.len(),
            forall|j: int|
                0 <= j < final(self).records@.len() ==> (#[trigger] final(self).records@[j]).permit@
                    == old(self).records@[j].permit@,
            match find_sink_record_spec(old(self).records@, id@) {
                None => {
                    out == ExecCompleteOutcome::UnknownId
                        && final(self).records@ == old(self).records@
                        && final(self).revision == old(self).revision
                        && final(self).committed_total == old(self).committed_total
                }
                Some(idx) => {
                    let old_rec = old(self).records@[idx];
                    (forall|j: int|
                        0 <= j < final(self).records@.len() && j != idx
                            ==> #[trigger] final(self).records@[j] == old(self).records@[j])
                        && if old_rec.status != ExecSlotStatus::Pending {
                            out == (ExecCompleteOutcome::Idempotent { status: old_rec.status })
                                && final(self).records@ == old(self).records@
                                && final(self).revision == old(self).revision
                                && final(self).committed_total == old(self).committed_total
                        } else {
                            let can_commit = unfenced_spec(
                                old_rec.permit@.required,
                                old_rec.permit@.epochs,
                                old(self).floors@,
                            ) && sink_rev_ok_spec(old(self).revision, old_rec.permit@)
                                && old(self).committed_total + old_rec.permit@.amount <= u64::MAX;
                            ((out is NewlyCommitted) == can_commit)
                                && if can_commit {
                                    out == (ExecCompleteOutcome::NewlyCommitted {
                                        amount: old_rec.permit@.amount,
                                    }) && final(self).records@[idx].status
                                        == ExecSlotStatus::Committed
                                        && final(self).committed_total
                                            == old(self).committed_total + old_rec.permit@.amount
                                        && final(self).revision
                                            == if old_rec.permit@.effect_kind == EFFECT_EDIT {
                                                (old(self).revision + 1) as u64
                                            } else {
                                                old(self).revision
                                            }
                                        && unfenced_spec(
                                            old_rec.permit@.required,
                                            old_rec.permit@.epochs,
                                            final(self).floors@,
                                        )
                                } else {
                                    out == ExecCompleteOutcome::Cancelled
                                        && final(self).records@[idx].status
                                            == ExecSlotStatus::Cancelled
                                        && final(self).committed_total == old(self).committed_total
                                        && final(self).revision == old(self).revision
                                }
                        }
                }
            },
    {
        let Some(idx) = self.find_record_exec(id) else {
            return ExecCompleteOutcome::UnknownId;
        };
        let cur_status = self.records[idx].status;
        if !matches!(cur_status, ExecSlotStatus::Pending) {
            return ExecCompleteOutcome::Idempotent { status: cur_status };
        }

        let fence_ok = unfenced(
            &self.records[idx].permit.required,
            &self.records[idx].permit.epochs,
            &self.floors,
        );
        let is_edit = self.records[idx].permit.effect_kind == EFFECT_EDIT;
        let rev_ok = if is_edit {
            revision_matches(self.revision, self.records[idx].permit.expected_rev)
                && self.revision < u64::MAX
        } else {
            true
        };
        let amount = self.records[idx].permit.amount;
        let fit_ok = self.committed_total <= u64::MAX - amount;

        if !fence_ok || !rev_ok || !fit_ok {
            let cancelled_rec = ExecSinkRecord {
                permit: clone_permit_wire(&self.records[idx].permit),
                status: ExecSlotStatus::Cancelled,
            };
            proof {
                sink_sum_on_cancel_pending(self.records@, idx as int, cancelled_rec);
            }
            self.records.set(idx, cancelled_rec);
            proof {
                assert forall|a: int, b: int|
                    0 <= a < b < self.records@.len()
                    implies (#[trigger] self.records@[a].permit@.id)
                    != (#[trigger] self.records@[b].permit@.id) by {
                    assert(self.records@[a].permit@.id == old(self).records@[a].permit@.id);
                    assert(self.records@[b].permit@.id == old(self).records@[b].permit@.id);
                }
            }
            ExecCompleteOutcome::Cancelled
        } else {
            let committed_rec = ExecSinkRecord {
                permit: clone_permit_wire(&self.records[idx].permit),
                status: ExecSlotStatus::Committed,
            };
            proof {
                sink_sum_on_commit(self.records@, idx as int, committed_rec);
            }
            if is_edit {
                self.revision += 1;
            }
            self.committed_total += amount;
            self.records.set(idx, committed_rec);
            proof {
                assert forall|a: int, b: int|
                    0 <= a < b < self.records@.len()
                    implies (#[trigger] self.records@[a].permit@.id)
                    != (#[trigger] self.records@[b].permit@.id) by {
                    assert(self.records@[a].permit@.id == old(self).records@[a].permit@.id);
                    assert(self.records@[b].permit@.id == old(self).records@[b].permit@.id);
                }
            }
            ExecCompleteOutcome::NewlyCommitted { amount }
        }
    }
}

impl ExecAuthorityKernel {
    pub open spec fn inv(&self) -> bool {
        self.broker.broker_inv()
            && self.sink.sink_inv()
            && self.broker.epochs@.len() == self.sink.floors@.len()
            && sink_records_issued_by_broker(self.sink.records@, self.broker.records@)
            && broker_terminal_matches_sink(self.broker.records@, self.sink.records@)
    }

    pub open spec fn project_slot(&self, id: Seq<u8>) -> LedgerState {
        self.broker.project_slot(id)
    }

    pub fn exec_new(budget: u64, issuers: usize) -> (k: Self)
        requires
            issuers > 0,
        ensures
            k.inv(),
            k.broker.budget == budget,
            k.broker.charged == 0,
            k.broker.committed == 0,
            k.broker.pending == 0,
            k.sink.committed_total == 0,
            k@ == (LedgerState {
                budget: budget as nat,
                charged: 0,
                committed: 0,
                inflight: Seq::empty(),
            }),
            ledger_inv(k@),
    {
        let k = ExecAuthorityKernel {
            broker: ExecBrokerKernel::broker_new(budget, issuers),
            sink: ExecSinkKernel::sink_new(issuers),
        };
        proof {
            assert(sink_records_issued_by_broker(k.sink.records@, k.broker.records@)) by {
                reveal(sink_records_issued_by_broker);
            }
            assert(broker_terminal_matches_sink(k.broker.records@, k.sink.records@)) by {
                reveal(broker_terminal_matches_sink);
            }
        }
        k
    }

    pub fn exec_withdraw(&mut self, issuer: usize) -> (ok: bool)
        requires
            old(self).inv(),
        ensures
            final(self).inv(),
            final(self)@ == step_ledger(old(self)@, LedgerStep::Withdraw),
            forall|id: Seq<u8>|
                #[trigger] final(self).project_slot(id)
                    == step_ledger(old(self).project_slot(id), LedgerStep::Withdraw),
            ok == (issuer < old(self).broker.epochs@.len()
                && old(self).broker.epochs@[issuer as int] < u64::MAX),
    {
        self.broker.broker_withdraw(issuer)
    }

    pub fn exec_renew(&mut self, issuer: usize) -> (ok: bool)
        requires
            old(self).inv(),
        ensures
            final(self).inv(),
            final(self)@ == step_ledger(old(self)@, LedgerStep::Renew),
            forall|id: Seq<u8>|
                #[trigger] final(self).project_slot(id)
                    == step_ledger(old(self).project_slot(id), LedgerStep::Renew),
            ok == (issuer < old(self).broker.epochs@.len()),
    {
        self.broker.broker_renew(issuer)
    }

    pub fn exec_uncertain_timeout(&mut self)
        requires
            old(self).inv(),
        ensures
            final(self).inv(),
            final(self)@ == step_ledger(old(self)@, LedgerStep::UncertainTimeout),
            forall|id: Seq<u8>|
                #[trigger] final(self).project_slot(id)
                    == step_ledger(old(self).project_slot(id), LedgerStep::UncertainTimeout),
    {
        self.broker.broker_uncertain_timeout();
    }

    pub fn exec_reserve(
        &mut self,
        permit: &ExecPermitWire,
        approved: &ExecPermitWire,
        grants: &Vec<bool>,
        now: u64,
    ) -> (out: ExecReserveOutcome)
        requires
            old(self).inv(),
        ensures
            final(self).inv(),
            final(self).sink == old(self).sink,
            out is ReservedNew ==> (
                final(self)@
                    == step_ledger(
                        old(self)@,
                        LedgerStep::Reserve { amount: permit@.amount as nat },
                    )
                && final(self).project_slot(permit@.id)
                    == step_ledger(
                        old(self).project_slot(permit@.id),
                        LedgerStep::Reserve { amount: permit@.amount as nat },
                    )
            ),
            !(out is ReservedNew) ==> (
                final(self)@ == old(self)@
                && final(self).project_slot(permit@.id) == old(self).project_slot(permit@.id)
            ),
            forall|other_id: Seq<u8>|
                other_id != permit@.id ==> #[trigger] final(self).project_slot(other_id)
                    == old(self).project_slot(other_id),
    {
        let out = self.broker.broker_reserve(permit, approved, grants, now);
        proof {
            if let ExecReserveOutcome::ReservedNew { .. } = out {
                let new_rec = self.broker.records@[old(self).broker.records@.len() as int];
                assert(self.broker.records@ =~= old(self).broker.records@.push(new_rec));
                assert(sink_records_issued_by_broker(
                    self.sink.records@,
                    self.broker.records@,
                )) by {
                    reveal(sink_records_issued_by_broker);
                    assert forall|s_idx: int|
                        0 <= s_idx < self.sink.records@.len() implies ({
                        let sr = #[trigger] self.sink.records@[s_idx];
                        let b_opt = find_broker_record_spec(self.broker.records@, sr.permit@.id);
                        b_opt is Some
                            && 0 <= b_opt.unwrap() < self.broker.records@.len()
                            && self.broker.records@[b_opt.unwrap()].permit@ == sr.permit@
                    }) by {
                        let sr = self.sink.records@[s_idx];
                        let old_opt = find_broker_record_spec(
                            old(self).broker.records@,
                            sr.permit@.id,
                        );
                        assert(old_opt is Some);
                        assert(sr.permit@.id != permit@.id);
                        find_broker_record_push(
                            old(self).broker.records@,
                            new_rec,
                            sr.permit@.id,
                        );
                    }
                }
                assert(broker_terminal_matches_sink(
                    self.broker.records@,
                    self.sink.records@,
                )) by {
                    reveal(broker_terminal_matches_sink);
                    assert forall|b_idx: int|
                        0 <= b_idx < self.broker.records@.len() implies ({
                        let br = #[trigger] self.broker.records@[b_idx];
                        (br.status == ExecSlotStatus::Committed ==> ({
                            let s_opt = find_sink_record_spec(self.sink.records@, br.permit@.id);
                            s_opt is Some
                                && 0 <= s_opt.unwrap() < self.sink.records@.len()
                                && self.sink.records@[s_opt.unwrap()].status
                                    == ExecSlotStatus::Committed
                        })) && (br.status == ExecSlotStatus::Cancelled ==> ({
                            let s_opt = find_sink_record_spec(self.sink.records@, br.permit@.id);
                            s_opt is Some
                                && 0 <= s_opt.unwrap() < self.sink.records@.len()
                                && self.sink.records@[s_opt.unwrap()].status
                                    == ExecSlotStatus::Cancelled
                        }))
                    }) by {
                        if b_idx < old(self).broker.records@.len() {
                            assert(self.broker.records@[b_idx] == old(self).broker.records@[b_idx]);
                        }
                    }
                }
            }
        }
        out
    }

    pub fn exec_sink_accept(&mut self, permit: &ExecPermitWire) -> (out: ExecAcceptOutcome)
        requires
            old(self).inv(),
        ensures
            final(self).inv(),
            final(self)@ == old(self)@,
            final(self).sink.committed_total == old(self).sink.committed_total,
            forall|id: Seq<u8>|
                #[trigger] final(self).project_slot(id) == old(self).project_slot(id),
            (find_sink_record_spec(old(self).sink.records@, permit@.id) is None
                && !unfenced_spec(permit@.required, permit@.epochs, old(self).sink.floors@))
                ==> !(out is AcceptedPending),
    {
        let Some(b_idx) = self.broker.find_record_exec(&permit.id) else {
            return ExecAcceptOutcome::UnissuedPermit;
        };
        if !same_permit_canonical(&self.broker.records[b_idx].permit, permit) {
            return ExecAcceptOutcome::PayloadMismatch;
        }
        let out = self.sink.sink_accept(permit);
        proof {
            if matches!(
                out,
                ExecAcceptOutcome::AcceptedPending | ExecAcceptOutcome::CancelledTombstone
            ) {
                let new_sr = self.sink.records@[old(self).sink.records@.len() as int];
                assert(self.sink.records@ =~= old(self).sink.records@.push(new_sr));
                assert(sink_records_issued_by_broker(
                    self.sink.records@,
                    self.broker.records@,
                )) by {
                    reveal(sink_records_issued_by_broker);
                    assert forall|s_idx: int|
                        0 <= s_idx < self.sink.records@.len() implies ({
                        let sr = #[trigger] self.sink.records@[s_idx];
                        let b_opt = find_broker_record_spec(self.broker.records@, sr.permit@.id);
                        b_opt is Some
                            && 0 <= b_opt.unwrap() < self.broker.records@.len()
                            && self.broker.records@[b_opt.unwrap()].permit@ == sr.permit@
                    }) by {
                        if s_idx < old(self).sink.records@.len() {
                            assert(self.sink.records@[s_idx] == old(self).sink.records@[s_idx]);
                        }
                    }
                }
                assert(broker_terminal_matches_sink(
                    self.broker.records@,
                    self.sink.records@,
                )) by {
                    reveal(broker_terminal_matches_sink);
                    assert forall|idx: int|
                        0 <= idx < self.broker.records@.len() implies ({
                        let br = #[trigger] self.broker.records@[idx];
                        (br.status == ExecSlotStatus::Committed ==> ({
                            let s_opt = find_sink_record_spec(self.sink.records@, br.permit@.id);
                            s_opt is Some
                                && 0 <= s_opt.unwrap() < self.sink.records@.len()
                                && self.sink.records@[s_opt.unwrap()].status
                                    == ExecSlotStatus::Committed
                        })) && (br.status == ExecSlotStatus::Cancelled ==> ({
                            let s_opt = find_sink_record_spec(self.sink.records@, br.permit@.id);
                            s_opt is Some
                                && 0 <= s_opt.unwrap() < self.sink.records@.len()
                                && self.sink.records@[s_opt.unwrap()].status
                                    == ExecSlotStatus::Cancelled
                        }))
                    }) by {
                        let br = self.broker.records@[idx];
                        if br.status == ExecSlotStatus::Committed
                            || br.status == ExecSlotStatus::Cancelled
                        {
                            let old_s = find_sink_record_spec(
                                old(self).sink.records@,
                                br.permit@.id,
                            );
                            assert(old_s is Some);
                            assert(br.permit@.id != permit@.id);
                            find_sink_record_push(old(self).sink.records@, new_sr, br.permit@.id);
                        }
                    }
                }
            }
        }
        out
    }

    pub fn exec_sink_fence(&mut self, issuer: usize, floor: u64) -> (ok: bool)
        requires
            old(self).inv(),
        ensures
            final(self).inv(),
            final(self)@ == old(self)@,
            final(self).sink.floors@ == apply_fence(old(self).sink.floors@, issuer as int, floor),
            final(self).sink.committed_total == old(self).sink.committed_total,
            ok == (issuer < old(self).sink.floors@.len()),
            ok ==> final(self).sink.floors@[issuer as int] >= floor,
            forall|id: Seq<u8>|
                #[trigger] final(self).project_slot(id) == old(self).project_slot(id),
    {
        let ok = self.sink.sink_fence(issuer, floor);
        proof {
            assert forall|qid: Seq<u8>|
                #[trigger] find_sink_record_spec(self.sink.records@, qid)
                == find_sink_record_spec(old(self).sink.records@, qid) by {
                if find_sink_record_spec(old(self).sink.records@, qid) is Some {
                    let j = find_sink_record_spec(old(self).sink.records@, qid).unwrap();
                    find_sink_record_some(old(self).sink.records@, qid);
                    assert(self.sink.records@[j].permit@.id == qid);
                    find_sink_record_unique_idx(self.sink.records@, j);
                } else {
                    if find_sink_record_spec(self.sink.records@, qid) is Some {
                        let j = find_sink_record_spec(self.sink.records@, qid).unwrap();
                        find_sink_record_some(self.sink.records@, qid);
                        find_sink_record_none(old(self).sink.records@, qid, j);
                        assert(false);
                    }
                }
            }
            assert(sink_records_issued_by_broker(self.sink.records@, self.broker.records@)) by {
                reveal(sink_records_issued_by_broker);
                assert forall|s_idx: int|
                    0 <= s_idx < self.sink.records@.len() implies ({
                    let sr = #[trigger] self.sink.records@[s_idx];
                    let b_opt = find_broker_record_spec(self.broker.records@, sr.permit@.id);
                    b_opt is Some
                        && 0 <= b_opt.unwrap() < self.broker.records@.len()
                        && self.broker.records@[b_opt.unwrap()].permit@ == sr.permit@
                }) by {
                    assert(self.sink.records@[s_idx].permit@ == old(self).sink.records@[s_idx].permit@);
                }
            }
            assert(broker_terminal_matches_sink(self.broker.records@, self.sink.records@)) by {
                reveal(broker_terminal_matches_sink);
            }
        }
        ok
    }

    pub fn exec_sink_complete(&mut self, id: &Vec<u8>) -> (out: ExecCompleteOutcome)
        requires
            old(self).inv(),
        ensures
            final(self).inv(),
            final(self)@ == old(self)@,
            final(self).sink.committed_total <= final(self).broker.charged,
            final(self).broker.charged <= final(self).broker.budget,
            forall|qid: Seq<u8>|
                #[trigger] final(self).project_slot(qid) == old(self).project_slot(qid),
            match find_sink_record_spec(old(self).sink.records@, id@) {
                Some(idx) => {
                    let sr = old(self).sink.records@[idx];
                    !unfenced_spec(sr.permit@.required, sr.permit@.epochs, old(self).sink.floors@)
                        ==> (!(out is NewlyCommitted)
                            && final(self).sink.committed_total == old(self).sink.committed_total)
                }
                None => {
                    out == ExecCompleteOutcome::UnknownId
                        && final(self).sink.committed_total == old(self).sink.committed_total
                }
            },
    {
        let out = self.sink.sink_complete(id);
        proof {
            assert forall|qid: Seq<u8>|
                #[trigger] find_sink_record_spec(self.sink.records@, qid)
                == find_sink_record_spec(old(self).sink.records@, qid) by {
                if find_sink_record_spec(old(self).sink.records@, qid) is Some {
                    let j = find_sink_record_spec(old(self).sink.records@, qid).unwrap();
                    find_sink_record_some(old(self).sink.records@, qid);
                    assert(self.sink.records@[j].permit@.id == qid);
                    find_sink_record_unique_idx(self.sink.records@, j);
                } else {
                    if find_sink_record_spec(self.sink.records@, qid) is Some {
                        let j = find_sink_record_spec(self.sink.records@, qid).unwrap();
                        find_sink_record_some(self.sink.records@, qid);
                        find_sink_record_none(old(self).sink.records@, qid, j);
                        assert(false);
                    }
                }
            }
            assert(sink_records_issued_by_broker(self.sink.records@, self.broker.records@)) by {
                reveal(sink_records_issued_by_broker);
                assert forall|s_idx: int|
                    0 <= s_idx < self.sink.records@.len() implies ({
                    let sr = #[trigger] self.sink.records@[s_idx];
                    let b_opt = find_broker_record_spec(self.broker.records@, sr.permit@.id);
                    b_opt is Some
                        && 0 <= b_opt.unwrap() < self.broker.records@.len()
                        && self.broker.records@[b_opt.unwrap()].permit@ == sr.permit@
                }) by {
                    assert(self.sink.records@[s_idx].permit@ == old(self).sink.records@[s_idx].permit@);
                }
            }
            assert(broker_terminal_matches_sink(self.broker.records@, self.sink.records@)) by {
                reveal(broker_terminal_matches_sink);
                assert forall|b_idx: int|
                    0 <= b_idx < self.broker.records@.len() implies ({
                    let br = #[trigger] self.broker.records@[b_idx];
                    (br.status == ExecSlotStatus::Committed ==> ({
                        let s_opt = find_sink_record_spec(self.sink.records@, br.permit@.id);
                        s_opt is Some
                            && 0 <= s_opt.unwrap() < self.sink.records@.len()
                            && self.sink.records@[s_opt.unwrap()].status == ExecSlotStatus::Committed
                    })) && (br.status == ExecSlotStatus::Cancelled ==> ({
                        let s_opt = find_sink_record_spec(self.sink.records@, br.permit@.id);
                        s_opt is Some
                            && 0 <= s_opt.unwrap() < self.sink.records@.len()
                            && self.sink.records@[s_opt.unwrap()].status == ExecSlotStatus::Cancelled
                    }))
                }) by {
                    let br = self.broker.records@[b_idx];
                    if br.status == ExecSlotStatus::Committed || br.status == ExecSlotStatus::Cancelled
                    {
                        let s_idx = find_sink_record_spec(old(self).sink.records@, br.permit@.id)
                            .unwrap();
                        assert(old(self).sink.records@[s_idx].status != ExecSlotStatus::Pending);
                        assert(
                            self.sink.records@[s_idx].status == old(self).sink.records@[s_idx].status
                        );
                    }
                }
            }
            authority_kernel_conservation(self);
        }
        out
    }

    pub fn exec_settle_committed(&mut self, id: &Vec<u8>) -> (out: ExecSettleOutcome)
        requires
            old(self).inv(),
        ensures
            final(self).inv(),
            final(self).sink == old(self).sink,
            match out {
                ExecSettleOutcome::Settled { slot, .. } => {
                    final(self)@
                        == step_ledger(old(self)@, LedgerStep::CommitSettle { slot: slot as int })
                        && final(self).project_slot(id@)
                            == step_ledger(
                                old(self).project_slot(id@),
                                LedgerStep::CommitSettle { slot: 0 },
                            )
                }
                _ => {
                    final(self)@ == old(self)@
                        && final(self).project_slot(id@) == old(self).project_slot(id@)
                }
            },
            forall|other_id: Seq<u8>|
                other_id != id@ ==> #[trigger] final(self).project_slot(other_id)
                    == old(self).project_slot(other_id),
    {
        let auth = self.sink.sink_is_committed(id);
        let out = self.broker.broker_settle_committed(id, auth);
        proof {
            if let ExecSettleOutcome::Settled { .. } = out {
                find_sink_record_some(self.sink.records@, id@);
                assert forall|qid: Seq<u8>|
                    #[trigger] find_broker_record_spec(self.broker.records@, qid)
                    == find_broker_record_spec(old(self).broker.records@, qid) by {
                    if find_broker_record_spec(old(self).broker.records@, qid) is Some {
                        let j = find_broker_record_spec(old(self).broker.records@, qid).unwrap();
                        find_broker_record_some(old(self).broker.records@, qid);
                        assert(self.broker.records@[j].permit@.id == qid);
                        find_broker_record_unique_idx(self.broker.records@, j);
                    } else {
                        if find_broker_record_spec(self.broker.records@, qid) is Some {
                            let j = find_broker_record_spec(self.broker.records@, qid).unwrap();
                            find_broker_record_some(self.broker.records@, qid);
                            find_broker_record_none(old(self).broker.records@, qid, j);
                            assert(false);
                        }
                    }
                }
                assert(sink_records_issued_by_broker(
                    self.sink.records@,
                    self.broker.records@,
                )) by {
                    reveal(sink_records_issued_by_broker);
                    assert forall|s_idx: int|
                        0 <= s_idx < self.sink.records@.len() implies ({
                        let sr = #[trigger] self.sink.records@[s_idx];
                        let b_opt = find_broker_record_spec(self.broker.records@, sr.permit@.id);
                        b_opt is Some
                            && 0 <= b_opt.unwrap() < self.broker.records@.len()
                            && self.broker.records@[b_opt.unwrap()].permit@ == sr.permit@
                    }) by {
                        let sr = self.sink.records@[s_idx];
                        let b_idx = find_broker_record_spec(
                            old(self).broker.records@,
                            sr.permit@.id,
                        ).unwrap();
                        assert(self.broker.records@[b_idx].permit@ == old(self).broker.records@[b_idx].permit@);
                    }
                }
                assert(broker_terminal_matches_sink(
                    self.broker.records@,
                    self.sink.records@,
                )) by {
                    reveal(broker_terminal_matches_sink);
                    assert forall|b_idx: int|
                        0 <= b_idx < self.broker.records@.len() implies ({
                        let br = #[trigger] self.broker.records@[b_idx];
                        (br.status == ExecSlotStatus::Committed ==> ({
                            let s_opt = find_sink_record_spec(self.sink.records@, br.permit@.id);
                            s_opt is Some
                                && 0 <= s_opt.unwrap() < self.sink.records@.len()
                                && self.sink.records@[s_opt.unwrap()].status
                                    == ExecSlotStatus::Committed
                        })) && (br.status == ExecSlotStatus::Cancelled ==> ({
                            let s_opt = find_sink_record_spec(self.sink.records@, br.permit@.id);
                            s_opt is Some
                                && 0 <= s_opt.unwrap() < self.sink.records@.len()
                                && self.sink.records@[s_opt.unwrap()].status
                                    == ExecSlotStatus::Cancelled
                        }))
                    }) by {
                        let target_b = find_broker_record_spec(old(self).broker.records@, id@)
                            .unwrap();
                        find_broker_record_some(old(self).broker.records@, id@);
                        if b_idx == target_b {
                            assert(self.broker.records@[b_idx].permit@.id == id@);
                        } else {
                            assert(self.broker.records@[b_idx].status == old(self).broker.records@[b_idx].status);
                            assert(self.broker.records@[b_idx].permit@ == old(self).broker.records@[b_idx].permit@);
                        }
                    }
                }
            }
        }
        out
    }

    pub fn exec_settle_cancelled(&mut self, id: &Vec<u8>) -> (out: ExecSettleOutcome)
        requires
            old(self).inv(),
        ensures
            final(self).inv(),
            final(self).sink == old(self).sink,
            match out {
                ExecSettleOutcome::Settled { slot, .. } => {
                    final(self)@
                        == step_ledger(old(self)@, LedgerStep::CancelRelease { slot: slot as int })
                        && final(self).project_slot(id@)
                            == step_ledger(
                                old(self).project_slot(id@),
                                LedgerStep::CancelRelease { slot: 0 },
                            )
                }
                _ => {
                    final(self)@ == old(self)@
                        && final(self).project_slot(id@) == old(self).project_slot(id@)
                }
            },
            forall|other_id: Seq<u8>|
                other_id != id@ ==> #[trigger] final(self).project_slot(other_id)
                    == old(self).project_slot(other_id),
    {
        let auth = self.sink.sink_is_cancelled(id);
        let out = self.broker.broker_settle_cancelled(id, auth);
        proof {
            if let ExecSettleOutcome::Settled { .. } = out {
                find_sink_record_some(self.sink.records@, id@);
                assert forall|qid: Seq<u8>|
                    #[trigger] find_broker_record_spec(self.broker.records@, qid)
                    == find_broker_record_spec(old(self).broker.records@, qid) by {
                    if find_broker_record_spec(old(self).broker.records@, qid) is Some {
                        let j = find_broker_record_spec(old(self).broker.records@, qid).unwrap();
                        find_broker_record_some(old(self).broker.records@, qid);
                        assert(self.broker.records@[j].permit@.id == qid);
                        find_broker_record_unique_idx(self.broker.records@, j);
                    } else {
                        if find_broker_record_spec(self.broker.records@, qid) is Some {
                            let j = find_broker_record_spec(self.broker.records@, qid).unwrap();
                            find_broker_record_some(self.broker.records@, qid);
                            find_broker_record_none(old(self).broker.records@, qid, j);
                            assert(false);
                        }
                    }
                }
                assert(sink_records_issued_by_broker(
                    self.sink.records@,
                    self.broker.records@,
                )) by {
                    reveal(sink_records_issued_by_broker);
                    assert forall|s_idx: int|
                        0 <= s_idx < self.sink.records@.len() implies ({
                        let sr = #[trigger] self.sink.records@[s_idx];
                        let b_opt = find_broker_record_spec(self.broker.records@, sr.permit@.id);
                        b_opt is Some
                            && 0 <= b_opt.unwrap() < self.broker.records@.len()
                            && self.broker.records@[b_opt.unwrap()].permit@ == sr.permit@
                    }) by {
                        let sr = self.sink.records@[s_idx];
                        let b_idx = find_broker_record_spec(
                            old(self).broker.records@,
                            sr.permit@.id,
                        ).unwrap();
                        assert(self.broker.records@[b_idx].permit@ == old(self).broker.records@[b_idx].permit@);
                    }
                }
                assert(broker_terminal_matches_sink(
                    self.broker.records@,
                    self.sink.records@,
                )) by {
                    reveal(broker_terminal_matches_sink);
                    assert forall|b_idx: int|
                        0 <= b_idx < self.broker.records@.len() implies ({
                        let br = #[trigger] self.broker.records@[b_idx];
                        (br.status == ExecSlotStatus::Committed ==> ({
                            let s_opt = find_sink_record_spec(self.sink.records@, br.permit@.id);
                            s_opt is Some
                                && 0 <= s_opt.unwrap() < self.sink.records@.len()
                                && self.sink.records@[s_opt.unwrap()].status
                                    == ExecSlotStatus::Committed
                        })) && (br.status == ExecSlotStatus::Cancelled ==> ({
                            let s_opt = find_sink_record_spec(self.sink.records@, br.permit@.id);
                            s_opt is Some
                                && 0 <= s_opt.unwrap() < self.sink.records@.len()
                                && self.sink.records@[s_opt.unwrap()].status
                                    == ExecSlotStatus::Cancelled
                        }))
                    }) by {
                        let target_b = find_broker_record_spec(old(self).broker.records@, id@)
                            .unwrap();
                        find_broker_record_some(old(self).broker.records@, id@);
                        if b_idx == target_b {
                            assert(self.broker.records@[b_idx].permit@.id == id@);
                        } else {
                            assert(self.broker.records@[b_idx].status == old(self).broker.records@[b_idx].status);
                            assert(self.broker.records@[b_idx].permit@ == old(self).broker.records@[b_idx].permit@);
                        }
                    }
                }
            }
        }
        out
    }
}

/// Bridge theorem: any valid `ExecBrokerKernel` state satisfies the ghost `ledger_inv`
/// and exact multi-record conservation `committed + pending == charged <= budget`.
pub proof fn broker_inv_implies_ledger_inv(b: &ExecBrokerKernel)
    requires
        b.broker_inv(),
    ensures
        ledger_inv(b@),
        b.committed + b.pending == b.charged,
        b.charged <= b.budget,
        forall|id: Seq<u8>| ledger_inv(#[trigger] b.project_slot(id)),
{
    assert forall|id: Seq<u8>| ledger_inv(#[trigger] b.project_slot(id)) by {
        reveal(broker_rec_to_inflight_inv);
        if let Some(idx) = find_broker_record_spec(b.records@, id) {
            find_broker_record_some(b.records@, id);
            let amt = b.records@[idx].permit@.amount as nat;
            assert(sum(seq![amt]) == amt + sum(seq![amt].drop_last()));
            assert(seq![amt].drop_last() =~= Seq::<nat>::empty());
        }
    }
}

/// End-to-end executable conservation theorem across coupled Broker + Sink kernels:
/// `sink.committed_total <= broker.committed + broker.pending == broker.charged <= broker.budget`.
pub proof fn authority_kernel_conservation(k: &ExecAuthorityKernel)
    requires
        k.inv(),
    ensures
        ledger_inv(k@),
        k.broker.committed + k.broker.pending == k.broker.charged,
        k.broker.charged <= k.broker.budget,
        k.sink.committed_total <= k.broker.charged,
        k.sink.committed_total <= k.broker.budget,
{
    broker_inv_implies_ledger_inv(&k.broker);
    assert forall|i: int|
        0 <= i < k.sink.records@.len()
            && (#[trigger] k.sink.records@[i]).status == ExecSlotStatus::Committed implies ({
        let opt = find_broker_record_spec(k.broker.records@, k.sink.records@[i].permit@.id);
        opt is Some
            && 0 <= opt.unwrap() < k.broker.records@.len()
            && k.broker.records@[opt.unwrap()].permit@.id == k.sink.records@[i].permit@.id
            && k.broker.records@[opt.unwrap()].permit@.amount
                == k.sink.records@[i].permit@.amount
            && k.broker.records@[opt.unwrap()].status != ExecSlotStatus::Cancelled
    }) by {
        let sr = k.sink.records@[i];
        assert(({
            let b_opt = find_broker_record_spec(k.broker.records@, sr.permit@.id);
            b_opt is Some
                && 0 <= b_opt.unwrap() < k.broker.records@.len()
                && k.broker.records@[b_opt.unwrap()].permit@ == sr.permit@
        })) by {
            reveal(sink_records_issued_by_broker);
        }
        let b_idx = find_broker_record_spec(k.broker.records@, sr.permit@.id).unwrap();
        find_sink_record_unique_idx(k.sink.records@, i);
        if k.broker.records@[b_idx].status == ExecSlotStatus::Cancelled {
            assert(({
                let s_opt = find_sink_record_spec(
                    k.sink.records@,
                    k.broker.records@[b_idx].permit@.id,
                );
                s_opt is Some
                    && 0 <= s_opt.unwrap() < k.sink.records@.len()
                    && k.sink.records@[s_opt.unwrap()].status == ExecSlotStatus::Cancelled
            })) by {
                reveal(broker_terminal_matches_sink);
            }
            let s_idx = find_sink_record_spec(k.sink.records@, sr.permit@.id).unwrap();
            assert(s_idx == i);
            assert(false);
        }
    }
    sink_committed_le_broker_active(k.sink.records@, k.broker.records@);
}

/// End-to-end executable post-fence exclusion theorem: once `sink.floors[issuer] >= fence_floor`
/// (and across any subsequent monotone fence suffix), no permit requiring `issuer` with
/// `permit.epochs[issuer] < fence_floor` can reside in `Pending` at the sink or satisfy
/// the pre-commit gate of `exec_sink_complete`.
pub proof fn authority_post_fence_exclusion(
    k: &ExecAuthorityKernel,
    permit: SpecPermitWire,
    issuer: int,
    fence_floor: u64,
    suffix: Seq<(int, u64)>,
)
    requires
        k.inv(),
        0 <= issuer < k.sink.floors@.len(),
        k.sink.floors@[issuer] >= fence_floor,
        permit.required.len() == k.sink.floors@.len(),
        permit.epochs.len() == k.sink.floors@.len(),
        permit.required[issuer],
        permit.epochs[issuer] < fence_floor,
    ensures
        !unfenced_spec(permit.required, permit.epochs, k.sink.floors@),
        !unfenced_spec(
            permit.required,
            permit.epochs,
            apply_fences(k.sink.floors@, suffix),
        ),
        forall|s_idx: int|
            0 <= s_idx < k.sink.records@.len()
                && (#[trigger] k.sink.records@[s_idx]).permit@ == permit
                ==> k.sink.records@[s_idx].status != ExecSlotStatus::Pending,
{
    fenced_zero(permit.required, permit.epochs, k.sink.floors@, issuer);
    monotone_fencing_induction(k.sink.floors@, suffix);
    let future_floors = apply_fences(k.sink.floors@, suffix);
    assert(future_floors[issuer] >= k.sink.floors@[issuer]);
    fenced_zero(permit.required, permit.epochs, future_floors, issuer);
    assert forall|s_idx: int|
        0 <= s_idx < k.sink.records@.len()
            && (#[trigger] k.sink.records@[s_idx]).permit@ == permit
        implies k.sink.records@[s_idx].status != ExecSlotStatus::Pending by {
        if k.sink.records@[s_idx].status == ExecSlotStatus::Pending {
            assert(unfenced_spec(permit.required, permit.epochs, k.sink.floors@));
            assert(false);
        }
    }
}

// ============================================================================
// V5 Pillar A: Verified Canonical Wire Decoder & Non-Malleable Bijection
// ============================================================================

#[verifier::opaque]
pub open spec fn le_bytes_to_u64_spec(b: Seq<u8>) -> u64
    recommends
        b.len() == 8,
{
    ((b[0] as u64)
        + 256u64
            * ((b[1] as u64)
                + 256u64
                    * ((b[2] as u64)
                        + 256u64
                            * ((b[3] as u64)
                                + 256u64
                                    * ((b[4] as u64)
                                        + 256u64
                                            * ((b[5] as u64)
                                                + 256u64
                                                    * ((b[6] as u64)
                                                        + 256u64 * (b[7] as u64)))))))) as u64
}

pub proof fn lemma_u64_le_roundtrip(x: u64)
    ensures
        u64_le_bytes_spec(x).len() == 8,
        le_bytes_to_u64_spec(u64_le_bytes_spec(x)) == x,
{
    reveal(u64_le_bytes_spec);
    reveal(le_bytes_to_u64_spec);
}

pub proof fn lemma_le_bytes_u64_canonical(
    b0: u8,
    b1: u8,
    b2: u8,
    b3: u8,
    b4: u8,
    b5: u8,
    b6: u8,
    b7: u8,
)
    ensures
        (b0 as nat)
            + 256nat
                * ((b1 as nat)
                    + 256nat
                        * ((b2 as nat)
                            + 256nat
                                * ((b3 as nat)
                                    + 256nat
                                        * ((b4 as nat)
                                            + 256nat
                                                * ((b5 as nat)
                                                    + 256nat
                                                        * ((b6 as nat)
                                                            + 256nat * (b7 as nat)))))))
            <= u64::MAX as nat,
        ({
            let x = ((b0 as u64)
                + 256u64
                    * ((b1 as u64)
                        + 256u64
                            * ((b2 as u64)
                                + 256u64
                                    * ((b3 as u64)
                                        + 256u64
                                            * ((b4 as u64)
                                                + 256u64
                                                    * ((b5 as u64)
                                                        + 256u64
                                                            * ((b6 as u64)
                                                                + 256u64 * (b7 as u64))))))))
                as u64;
            u64_le_bytes_spec(x) == seq![b0, b1, b2, b3, b4, b5, b6, b7]
                && le_bytes_to_u64_spec(seq![b0, b1, b2, b3, b4, b5, b6, b7]) == x
        }),
{
    reveal(u64_le_bytes_spec);
    reveal(le_bytes_to_u64_spec);
    let n7: nat = b7 as nat;
    let n6: nat = (b6 as nat) + 256nat * n7;
    let n5: nat = (b5 as nat) + 256nat * n6;
    let n4: nat = (b4 as nat) + 256nat * n5;
    let n3: nat = (b3 as nat) + 256nat * n4;
    let n2: nat = (b2 as nat) + 256nat * n3;
    let n1: nat = (b1 as nat) + 256nat * n2;
    let n0: nat = (b0 as nat) + 256nat * n1;
    assert(n7 <= 255);
    assert(n6 <= 65535);
    assert(n5 <= 16777215);
    assert(n4 <= 4294967295);
    assert(n3 <= 1099511627775);
    assert(n2 <= 281474976710655);
    assert(n1 <= 72057594037927935);
    assert(n0 <= 18446744073709551615);
    let x: u64 = n0 as u64;
    assert(x % 256 == b0 as u64);
    assert(x / 256 == n1 as u64);
    let x1 = x / 256;
    assert(x1 % 256 == b1 as u64);
    assert(x1 / 256 == n2 as u64);
    let x2 = x1 / 256;
    assert(x2 % 256 == b2 as u64);
    assert(x2 / 256 == n3 as u64);
    let x3 = x2 / 256;
    assert(x3 % 256 == b3 as u64);
    assert(x3 / 256 == n4 as u64);
    let x4 = x3 / 256;
    assert(x4 % 256 == b4 as u64);
    assert(x4 / 256 == n5 as u64);
    let x5 = x4 / 256;
    assert(x5 % 256 == b5 as u64);
    assert(x5 / 256 == n6 as u64);
    let x6 = x5 / 256;
    assert(x6 % 256 == b6 as u64);
    assert(x6 / 256 == n7 as u64);
    let x7 = x6 / 256;
    assert(x7 % 256 == b7 as u64);
    assert(u64_le_bytes_spec(x) =~= seq![b0, b1, b2, b3, b4, b5, b6, b7]);
}

pub open spec fn spec_read_u64_le(s: Seq<u8>, pos: int) -> Option<(u64, int)> {
    if 0 <= pos && s.len() - pos >= 8 {
        Some((le_bytes_to_u64_spec(s.subrange(pos, pos + 8)), pos + 8))
    } else {
        None
    }
}

pub open spec fn spec_read_tag(s: Seq<u8>, pos: int, expected_tag: u64) -> Option<int> {
    match spec_read_u64_le(s, pos) {
        Some((tag, p1)) => {
            if tag == expected_tag {
                Some(p1)
            } else {
                None
            }
        }
        None => None,
    }
}

pub open spec fn spec_read_u8_framed(
    s: Seq<u8>,
    pos: int,
    expected_tag: u64,
) -> Option<(u8, int)> {
    match spec_read_tag(s, pos, expected_tag) {
        Some(p1) => {
            if s.len() - p1 >= 1 {
                Some((s[p1], p1 + 1))
            } else {
                None
            }
        }
        None => None,
    }
}

pub open spec fn spec_read_u64_framed(
    s: Seq<u8>,
    pos: int,
    expected_tag: u64,
) -> Option<(u64, int)> {
    match spec_read_tag(s, pos, expected_tag) {
        Some(p1) => spec_read_u64_le(s, p1),
        None => None,
    }
}

pub open spec fn spec_read_bytes_framed(
    s: Seq<u8>,
    pos: int,
    expected_tag: u64,
) -> Option<(Seq<u8>, int)> {
    match spec_read_tag(s, pos, expected_tag) {
        Some(p1) => match spec_read_u64_le(s, p1) {
            Some((n, p2)) => {
                if (n as int) <= s.len() - p2 {
                    Some((s.subrange(p2, p2 + n as int), p2 + n as int))
                } else {
                    None
                }
            }
            None => None,
        },
        None => None,
    }
}

pub open spec fn spec_decode_u64_elems(s: Seq<u8>, pos: int, count: nat) -> Seq<u64>
    decreases count,
{
    if count == 0 {
        Seq::empty()
    } else {
        let prev = (count - 1) as nat;
        spec_decode_u64_elems(s, pos, prev).push(le_bytes_to_u64_spec(
            s.subrange(pos + 8 * (prev as int), pos + 8 * (count as int)),
        ))
    }
}

pub open spec fn spec_read_u64_seq_framed(
    s: Seq<u8>,
    pos: int,
    expected_tag: u64,
) -> Option<(Seq<u64>, int)> {
    match spec_read_tag(s, pos, expected_tag) {
        Some(p1) => match spec_read_u64_le(s, p1) {
            Some((n, p2)) => {
                if 8 * (n as int) <= s.len() - p2 {
                    Some((
                        spec_decode_u64_elems(s, p2, n as nat),
                        p2 + 8 * (n as int),
                    ))
                } else {
                    None
                }
            }
            None => None,
        },
        None => None,
    }
}

pub open spec fn spec_decode_bool_elems(s: Seq<u8>, pos: int, count: nat) -> Option<Seq<bool>>
    decreases count,
{
    if count == 0 {
        Some(Seq::empty())
    } else {
        let prev = (count - 1) as nat;
        match spec_decode_bool_elems(s, pos, prev) {
            Some(prefix) => {
                let raw = s[pos + (prev as int)];
                if raw <= 1u8 {
                    Some(prefix.push(raw == 1u8))
                } else {
                    None
                }
            }
            None => None,
        }
    }
}

pub open spec fn spec_read_bool_seq_framed(
    s: Seq<u8>,
    pos: int,
    expected_tag: u64,
) -> Option<(Seq<bool>, int)> {
    match spec_read_tag(s, pos, expected_tag) {
        Some(p1) => match spec_read_u64_le(s, p1) {
            Some((n, p2)) => {
                if (n as int) <= s.len() - p2 {
                    match spec_decode_bool_elems(s, p2, n as nat) {
                        Some(bs) => Some((bs, p2 + n as int)),
                        None => None,
                    }
                } else {
                    None
                }
            }
            None => None,
        },
        None => None,
    }
}

pub open spec fn spec_decode_permit(s: Seq<u8>) -> Option<SpecPermitWire> {
    match spec_read_tag(s, 0, WIRE_TAG_PERMIT_V1) {
        None => None,
        Some(pos0) => match spec_read_bytes_framed(s, pos0, WIRE_TAG_ID) {
            None => None,
            Some((id, pos1)) => match spec_read_bytes_framed(s, pos1, WIRE_TAG_ACTOR) {
                None => None,
                Some((actor, pos2)) => match spec_read_u8_framed(s, pos2, WIRE_TAG_EFFECT_KIND) {
                    None => None,
                    Some((effect_kind, pos3)) => {
                        match spec_read_u64_framed(s, pos3, WIRE_TAG_AMOUNT) {
                            None => None,
                            Some((amount, pos4)) => {
                                match spec_read_u64_framed(s, pos4, WIRE_TAG_EXPIRES) {
                                    None => None,
                                    Some((expires, pos5)) => match spec_read_u64_framed(
                                        s,
                                        pos5,
                                        WIRE_TAG_EXPECTED_REV,
                                    ) {
                                        None => None,
                                        Some((expected_rev, pos6)) => {
                                            match spec_read_bytes_framed(s, pos6, WIRE_TAG_TARGET) {
                                                None => None,
                                                Some((target, pos7)) => {
                                                    match spec_read_bytes_framed(
                                                        s,
                                                        pos7,
                                                        WIRE_TAG_CONTENTS,
                                                    ) {
                                                        None => None,
                                                        Some((contents, pos8)) => {
                                                            match spec_read_u64_seq_framed(
                                                                s,
                                                                pos8,
                                                                WIRE_TAG_EPOCHS,
                                                            ) {
                                                                None => None,
                                                                Some((epochs, pos9)) => {
                                                                    match spec_read_bool_seq_framed(
                                                                        s,
                                                                        pos9,
                                                                        WIRE_TAG_REQUIRED,
                                                                    ) {
                                                                        None => None,
                                                                        Some((required, pos10)) => {
                                                                            if pos10 == s.len()
                                                                                && epochs.len()
                                                                                    == required
                                                                                        .len()
                                                                            {
                                                                                Some(
                                                                                    SpecPermitWire {
                                                                                        id,
                                                                                        actor,
                                                                                        effect_kind,
                                                                                        amount,
                                                                                        expires,
                                                                                        expected_rev,
                                                                                        target,
                                                                                        contents,
                                                                                        epochs,
                                                                                        required,
                                                                                    },
                                                                                )
                                                                            } else {
                                                                                None
                                                                            }
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    },
                                }
                            }
                        }
                    }
                },
            },
        },
    }
}

proof fn lemma_spec_read_u64_le_sound(s: Seq<u8>, pos: int)
    requires
        spec_read_u64_le(s, pos) is Some,
    ensures
        ({
            let (val, next_pos) = spec_read_u64_le(s, pos).unwrap();
            0 <= pos && next_pos == pos + 8 && next_pos <= s.len()
                && s.subrange(pos, next_pos) == u64_le_bytes_spec(val)
        }),
{
    let sub = s.subrange(pos, pos + 8);
    lemma_le_bytes_u64_canonical(
        sub[0], sub[1], sub[2], sub[3], sub[4], sub[5], sub[6], sub[7],
    );
    assert(sub =~= seq![sub[0], sub[1], sub[2], sub[3], sub[4], sub[5], sub[6], sub[7]]);
}

proof fn lemma_spec_read_u64_le_complete(s: Seq<u8>, pos: int, x: u64, rest: Seq<u8>)
    requires
        0 <= pos <= s.len(),
        s.subrange(pos, s.len() as int) == u64_le_bytes_spec(x) + rest,
    ensures
        spec_read_u64_le(s, pos) == Some((x, pos + 8)),
        s.subrange(pos + 8, s.len() as int) == rest,
{
    lemma_u64_le_roundtrip(x);
    let tail = s.subrange(pos, s.len() as int);
    assert(tail.len() >= 8);
    assert(s.subrange(pos, pos + 8) =~= tail.subrange(0, 8));
    assert(tail.subrange(0, 8) =~= u64_le_bytes_spec(x));
    assert(s.subrange(pos + 8, s.len() as int) =~= tail.subrange(8, tail.len() as int));
    assert(tail.subrange(8, tail.len() as int) =~= rest);
}

proof fn lemma_spec_read_tag_sound(s: Seq<u8>, pos: int, tag: u64)
    requires
        spec_read_tag(s, pos, tag) is Some,
    ensures
        ({
            let p1 = spec_read_tag(s, pos, tag).unwrap();
            0 <= pos && p1 == pos + 8 && p1 <= s.len()
                && s.subrange(pos, p1) == u64_le_bytes_spec(tag)
        }),
{
    lemma_spec_read_u64_le_sound(s, pos);
}

proof fn lemma_spec_read_tag_complete(s: Seq<u8>, pos: int, tag: u64, rest: Seq<u8>)
    requires
        0 <= pos <= s.len(),
        s.subrange(pos, s.len() as int) == u64_le_bytes_spec(tag) + rest,
    ensures
        spec_read_tag(s, pos, tag) == Some(pos + 8),
        s.subrange(pos + 8, s.len() as int) == rest,
{
    lemma_spec_read_u64_le_complete(s, pos, tag, rest);
}

proof fn lemma_spec_read_u8_framed_sound(s: Seq<u8>, pos: int, tag: u64)
    requires
        spec_read_u8_framed(s, pos, tag) is Some,
    ensures
        ({
            let (v, np) = spec_read_u8_framed(s, pos, tag).unwrap();
            0 <= pos && np == pos + 9 && np <= s.len()
                && s.subrange(pos, np) == encode_u8_field_spec(tag, v)
        }),
{
    lemma_spec_read_tag_sound(s, pos, tag);
    let (v, np) = spec_read_u8_framed(s, pos, tag).unwrap();
    assert(s.subrange(pos, np) =~= s.subrange(pos, pos + 8) + seq![v]);
}

proof fn lemma_spec_read_u8_framed_complete(
    s: Seq<u8>,
    pos: int,
    tag: u64,
    v: u8,
    rest: Seq<u8>,
)
    requires
        0 <= pos <= s.len(),
        s.subrange(pos, s.len() as int) == encode_u8_field_spec(tag, v) + rest,
    ensures
        spec_read_u8_framed(s, pos, tag) == Some((v, pos + 9)),
        s.subrange(pos + 9, s.len() as int) == rest,
{
    assert(encode_u8_field_spec(tag, v) + rest =~= u64_le_bytes_spec(tag) + (seq![v] + rest));
    lemma_spec_read_tag_complete(s, pos, tag, seq![v] + rest);
    let tail1 = s.subrange(pos + 8, s.len() as int);
    assert(s[pos + 8] == tail1[0]);
    assert(tail1[0] == v);
    assert(s.subrange(pos + 9, s.len() as int) =~= tail1.subrange(1, tail1.len() as int));
    assert(tail1.subrange(1, tail1.len() as int) =~= rest);
}

proof fn lemma_spec_read_u64_framed_sound(s: Seq<u8>, pos: int, tag: u64)
    requires
        spec_read_u64_framed(s, pos, tag) is Some,
    ensures
        ({
            let (v, np) = spec_read_u64_framed(s, pos, tag).unwrap();
            0 <= pos && np == pos + 16 && np <= s.len()
                && s.subrange(pos, np) == encode_u64_field_spec(tag, v)
        }),
{
    lemma_spec_read_tag_sound(s, pos, tag);
    lemma_spec_read_u64_le_sound(s, pos + 8);
    let (_, np) = spec_read_u64_framed(s, pos, tag).unwrap();
    assert(s.subrange(pos, np) =~= s.subrange(pos, pos + 8) + s.subrange(pos + 8, np));
}

proof fn lemma_spec_read_u64_framed_complete(
    s: Seq<u8>,
    pos: int,
    tag: u64,
    v: u64,
    rest: Seq<u8>,
)
    requires
        0 <= pos <= s.len(),
        s.subrange(pos, s.len() as int) == encode_u64_field_spec(tag, v) + rest,
    ensures
        spec_read_u64_framed(s, pos, tag) == Some((v, pos + 16)),
        s.subrange(pos + 16, s.len() as int) == rest,
{
    assert(encode_u64_field_spec(tag, v) + rest
        =~= u64_le_bytes_spec(tag) + (u64_le_bytes_spec(v) + rest));
    lemma_spec_read_tag_complete(s, pos, tag, u64_le_bytes_spec(v) + rest);
    lemma_spec_read_u64_le_complete(s, pos + 8, v, rest);
}

proof fn lemma_spec_read_bytes_framed_sound(s: Seq<u8>, pos: int, tag: u64)
    requires
        spec_read_bytes_framed(s, pos, tag) is Some,
    ensures
        ({
            let (b, np) = spec_read_bytes_framed(s, pos, tag).unwrap();
            0 <= pos && pos <= np && np <= s.len()
                && b.len() <= u64::MAX as nat
                && s.subrange(pos, np) == encode_bytes_fixed_spec(tag, b)
        }),
{
    lemma_spec_read_tag_sound(s, pos, tag);
    lemma_spec_read_u64_le_sound(s, pos + 8);
    let (n, p2) = spec_read_u64_le(s, pos + 8).unwrap();
    let (b, np) = spec_read_bytes_framed(s, pos, tag).unwrap();
    assert(b.len() == n as nat);
    assert(s.subrange(pos, np)
        =~= (s.subrange(pos, pos + 8) + s.subrange(pos + 8, p2)) + s.subrange(p2, np));
}

proof fn lemma_spec_read_bytes_framed_complete(
    s: Seq<u8>,
    pos: int,
    tag: u64,
    b: Seq<u8>,
    rest: Seq<u8>,
)
    requires
        0 <= pos <= s.len(),
        b.len() <= u64::MAX as nat,
        s.subrange(pos, s.len() as int) == encode_bytes_fixed_spec(tag, b) + rest,
    ensures
        spec_read_bytes_framed(s, pos, tag) == Some((b, pos + 16 + b.len() as int)),
        s.subrange(pos + 16 + b.len() as int, s.len() as int) == rest,
{
    let n: u64 = b.len() as u64;
    assert(encode_bytes_fixed_spec(tag, b) + rest
        =~= u64_le_bytes_spec(tag) + (u64_le_bytes_spec(n) + (b + rest)));
    lemma_spec_read_tag_complete(s, pos, tag, u64_le_bytes_spec(n) + (b + rest));
    lemma_spec_read_u64_le_complete(s, pos + 8, n, b + rest);
    let p2 = pos + 16;
    let tail2 = s.subrange(p2, s.len() as int);
    assert(s.subrange(p2, p2 + n as int) =~= tail2.subrange(0, n as int));
    assert(tail2.subrange(0, n as int) =~= b);
    assert(s.subrange(p2 + n as int, s.len() as int) =~= tail2.subrange(n as int, tail2.len() as int));
    assert(tail2.subrange(n as int, tail2.len() as int) =~= rest);
}

proof fn lemma_spec_decode_u64_elems_sound(s: Seq<u8>, pos: int, count: nat)
    requires
        0 <= pos,
        pos + 8 * (count as int) <= s.len(),
    ensures
        spec_decode_u64_elems(s, pos, count).len() == count,
        s.subrange(pos, pos + 8 * (count as int))
            == encode_u64_seq_spec(spec_decode_u64_elems(s, pos, count)),
    decreases count,
{
    if count == 0 {
        assert(s.subrange(pos, pos) =~= Seq::<u8>::empty());
    } else {
        let prev = (count - 1) as nat;
        lemma_spec_decode_u64_elems_sound(s, pos, prev);
        let mid = pos + 8 * (prev as int);
        let end = pos + 8 * (count as int);
        lemma_spec_read_u64_le_sound(s, mid);
        let out = spec_decode_u64_elems(s, pos, count);
        let prev_out = spec_decode_u64_elems(s, pos, prev);
        assert(out.drop_last() =~= prev_out);
        assert(out.last() == le_bytes_to_u64_spec(s.subrange(mid, end)));
        assert(s.subrange(pos, end) =~= s.subrange(pos, mid) + s.subrange(mid, end));
    }
}

proof fn lemma_spec_decode_u64_elems_complete(s: Seq<u8>, pos: int, xs: Seq<u64>)
    requires
        0 <= pos,
        pos + 8 * (xs.len() as int) <= s.len(),
        s.subrange(pos, pos + 8 * (xs.len() as int)) == encode_u64_seq_spec(xs),
    ensures
        spec_decode_u64_elems(s, pos, xs.len()) == xs,
    decreases xs.len(),
{
    if xs.len() == 0 {
        assert(xs =~= Seq::<u64>::empty());
    } else {
        let prev = xs.drop_last();
        encode_u64_seq_len(prev);
        lemma_u64_le_roundtrip(xs.last());
        let mid = pos + 8 * (prev.len() as int);
        let end = pos + 8 * (xs.len() as int);
        let sub = s.subrange(pos, end);
        assert(s.subrange(pos, mid) =~= sub.subrange(0, 8 * (prev.len() as int)));
        assert(sub.subrange(0, 8 * (prev.len() as int)) =~= encode_u64_seq_spec(prev));
        assert(s.subrange(mid, end) =~= sub.subrange(8 * (prev.len() as int), 8 * (xs.len() as int)));
        assert(sub.subrange(8 * (prev.len() as int), 8 * (xs.len() as int))
            =~= u64_le_bytes_spec(xs.last()));
        lemma_spec_decode_u64_elems_complete(s, pos, prev);
        assert(xs =~= prev.push(xs.last()));
    }
}

proof fn lemma_spec_read_u64_seq_framed_sound(s: Seq<u8>, pos: int, tag: u64)
    requires
        spec_read_u64_seq_framed(s, pos, tag) is Some,
    ensures
        ({
            let (xs, np) = spec_read_u64_seq_framed(s, pos, tag).unwrap();
            0 <= pos && pos <= np && np <= s.len()
                && xs.len() <= u64::MAX as nat
                && s.subrange(pos, np) == encode_u64_seq_framed_spec(tag, xs)
        }),
{
    lemma_spec_read_tag_sound(s, pos, tag);
    lemma_spec_read_u64_le_sound(s, pos + 8);
    let (n, p2) = spec_read_u64_le(s, pos + 8).unwrap();
    lemma_spec_decode_u64_elems_sound(s, p2, n as nat);
    let (xs, np) = spec_read_u64_seq_framed(s, pos, tag).unwrap();
    assert(xs.len() == n as nat);
    assert(s.subrange(pos, np)
        =~= (s.subrange(pos, pos + 8) + s.subrange(pos + 8, p2)) + s.subrange(p2, np));
}

proof fn lemma_spec_read_u64_seq_framed_complete(
    s: Seq<u8>,
    pos: int,
    tag: u64,
    xs: Seq<u64>,
    rest: Seq<u8>,
)
    requires
        0 <= pos <= s.len(),
        xs.len() <= u64::MAX as nat,
        s.subrange(pos, s.len() as int) == encode_u64_seq_framed_spec(tag, xs) + rest,
    ensures
        spec_read_u64_seq_framed(s, pos, tag) == Some((xs, pos + 16 + 8 * (xs.len() as int))),
        s.subrange(pos + 16 + 8 * (xs.len() as int), s.len() as int) == rest,
{
    let n: u64 = xs.len() as u64;
    encode_u64_seq_len(xs);
    let body = encode_u64_seq_spec(xs);
    assert(encode_u64_seq_framed_spec(tag, xs) + rest
        =~= u64_le_bytes_spec(tag) + (u64_le_bytes_spec(n) + (body + rest)));
    lemma_spec_read_tag_complete(s, pos, tag, u64_le_bytes_spec(n) + (body + rest));
    lemma_spec_read_u64_le_complete(s, pos + 8, n, body + rest);
    let p2 = pos + 16;
    let np = p2 + 8 * (xs.len() as int);
    let tail2 = s.subrange(p2, s.len() as int);
    assert(s.subrange(p2, np) =~= tail2.subrange(0, 8 * (xs.len() as int)));
    assert(tail2.subrange(0, 8 * (xs.len() as int)) =~= body);
    lemma_spec_decode_u64_elems_complete(s, p2, xs);
    assert(s.subrange(np, s.len() as int) =~= tail2.subrange(8 * (xs.len() as int), tail2.len() as int));
    assert(tail2.subrange(8 * (xs.len() as int), tail2.len() as int) =~= rest);
}

proof fn lemma_spec_decode_bool_elems_sound(s: Seq<u8>, pos: int, count: nat)
    requires
        0 <= pos,
        pos + (count as int) <= s.len(),
        spec_decode_bool_elems(s, pos, count) is Some,
    ensures
        ({
            let bs = spec_decode_bool_elems(s, pos, count).unwrap();
            bs.len() == count
                && s.subrange(pos, pos + (count as int)) == encode_bool_seq_spec(bs)
        }),
    decreases count,
{
    if count == 0 {
        assert(s.subrange(pos, pos) =~= Seq::<u8>::empty());
    } else {
        let prev = (count - 1) as nat;
        lemma_spec_decode_bool_elems_sound(s, pos, prev);
        let bs = spec_decode_bool_elems(s, pos, count).unwrap();
        let prev_bs = spec_decode_bool_elems(s, pos, prev).unwrap();
        let raw = s[pos + (prev as int)];
        assert(bs.drop_last() =~= prev_bs);
        assert(bool_byte_spec(bs.last()) == raw);
        assert(s.subrange(pos, pos + (count as int))
            =~= s.subrange(pos, pos + (prev as int)).push(raw));
    }
}

proof fn lemma_spec_decode_bool_elems_complete(s: Seq<u8>, pos: int, bs: Seq<bool>)
    requires
        0 <= pos,
        pos + (bs.len() as int) <= s.len(),
        s.subrange(pos, pos + (bs.len() as int)) == encode_bool_seq_spec(bs),
    ensures
        spec_decode_bool_elems(s, pos, bs.len()) == Some(bs),
    decreases bs.len(),
{
    if bs.len() == 0 {
        assert(bs =~= Seq::<bool>::empty());
    } else {
        let prev = bs.drop_last();
        encode_bool_seq_len(prev);
        let end = pos + (bs.len() as int);
        let mid = pos + (prev.len() as int);
        let sub = s.subrange(pos, end);
        assert(s.subrange(pos, mid) =~= sub.subrange(0, prev.len() as int));
        assert(sub.subrange(0, prev.len() as int) =~= encode_bool_seq_spec(prev));
        lemma_spec_decode_bool_elems_complete(s, pos, prev);
        assert(s[mid] == sub[prev.len() as int]);
        assert(sub[prev.len() as int] == bool_byte_spec(bs.last()));
        assert(bs =~= prev.push(bs.last()));
    }
}

proof fn lemma_spec_decode_bool_elems_none(s: Seq<u8>, pos: int, bad_end: nat, total: nat)
    requires
        bad_end <= total,
        spec_decode_bool_elems(s, pos, bad_end) is None,
    ensures
        spec_decode_bool_elems(s, pos, total) is None,
    decreases total - bad_end,
{
    if bad_end < total {
        lemma_spec_decode_bool_elems_none(s, pos, (bad_end + 1) as nat, total);
    }
}

proof fn lemma_spec_read_bool_seq_framed_sound(s: Seq<u8>, pos: int, tag: u64)
    requires
        spec_read_bool_seq_framed(s, pos, tag) is Some,
    ensures
        ({
            let (bs, np) = spec_read_bool_seq_framed(s, pos, tag).unwrap();
            0 <= pos && pos <= np && np <= s.len()
                && bs.len() <= u64::MAX as nat
                && s.subrange(pos, np) == encode_bool_seq_framed_spec(tag, bs)
        }),
{
    lemma_spec_read_tag_sound(s, pos, tag);
    lemma_spec_read_u64_le_sound(s, pos + 8);
    let (n, p2) = spec_read_u64_le(s, pos + 8).unwrap();
    lemma_spec_decode_bool_elems_sound(s, p2, n as nat);
    let (bs, np) = spec_read_bool_seq_framed(s, pos, tag).unwrap();
    assert(bs.len() == n as nat);
    assert(s.subrange(pos, np)
        =~= (s.subrange(pos, pos + 8) + s.subrange(pos + 8, p2)) + s.subrange(p2, np));
}

proof fn lemma_spec_read_bool_seq_framed_complete(
    s: Seq<u8>,
    pos: int,
    tag: u64,
    bs: Seq<bool>,
    rest: Seq<u8>,
)
    requires
        0 <= pos <= s.len(),
        bs.len() <= u64::MAX as nat,
        s.subrange(pos, s.len() as int) == encode_bool_seq_framed_spec(tag, bs) + rest,
    ensures
        spec_read_bool_seq_framed(s, pos, tag) == Some((bs, pos + 16 + (bs.len() as int))),
        s.subrange(pos + 16 + (bs.len() as int), s.len() as int) == rest,
{
    let n: u64 = bs.len() as u64;
    encode_bool_seq_len(bs);
    let body = encode_bool_seq_spec(bs);
    assert(encode_bool_seq_framed_spec(tag, bs) + rest
        =~= u64_le_bytes_spec(tag) + (u64_le_bytes_spec(n) + (body + rest)));
    lemma_spec_read_tag_complete(s, pos, tag, u64_le_bytes_spec(n) + (body + rest));
    lemma_spec_read_u64_le_complete(s, pos + 8, n, body + rest);
    let p2 = pos + 16;
    let np = p2 + (bs.len() as int);
    let tail2 = s.subrange(p2, s.len() as int);
    assert(s.subrange(p2, np) =~= tail2.subrange(0, bs.len() as int));
    assert(tail2.subrange(0, bs.len() as int) =~= body);
    lemma_spec_decode_bool_elems_complete(s, p2, bs);
    assert(s.subrange(np, s.len() as int) =~= tail2.subrange(bs.len() as int, tail2.len() as int));
    assert(tail2.subrange(bs.len() as int, tail2.len() as int) =~= rest);
}

proof fn lemma_spec_decode_permit_sound(s: Seq<u8>)
    requires
        spec_decode_permit(s) is Some,
    ensures
        ({
            let p = spec_decode_permit(s).unwrap();
            spec_permit_fits_u64(p)
                && p.epochs.len() == p.required.len()
                && encode_permit_fixed_spec(p) == s
                && encode_permit_spec(p) == s
        }),
{
    reveal(encode_permit_fixed_spec);
    lemma_spec_read_tag_sound(s, 0, WIRE_TAG_PERMIT_V1);
    let pos0 = spec_read_tag(s, 0, WIRE_TAG_PERMIT_V1).unwrap();
    lemma_spec_read_bytes_framed_sound(s, pos0, WIRE_TAG_ID);
    let (_, pos1) = spec_read_bytes_framed(s, pos0, WIRE_TAG_ID).unwrap();
    lemma_spec_read_bytes_framed_sound(s, pos1, WIRE_TAG_ACTOR);
    let (_, pos2) = spec_read_bytes_framed(s, pos1, WIRE_TAG_ACTOR).unwrap();
    lemma_spec_read_u8_framed_sound(s, pos2, WIRE_TAG_EFFECT_KIND);
    let (_, pos3) = spec_read_u8_framed(s, pos2, WIRE_TAG_EFFECT_KIND).unwrap();
    lemma_spec_read_u64_framed_sound(s, pos3, WIRE_TAG_AMOUNT);
    let (_, pos4) = spec_read_u64_framed(s, pos3, WIRE_TAG_AMOUNT).unwrap();
    lemma_spec_read_u64_framed_sound(s, pos4, WIRE_TAG_EXPIRES);
    let (_, pos5) = spec_read_u64_framed(s, pos4, WIRE_TAG_EXPIRES).unwrap();
    lemma_spec_read_u64_framed_sound(s, pos5, WIRE_TAG_EXPECTED_REV);
    let (_, pos6) = spec_read_u64_framed(s, pos5, WIRE_TAG_EXPECTED_REV).unwrap();
    lemma_spec_read_bytes_framed_sound(s, pos6, WIRE_TAG_TARGET);
    let (_, pos7) = spec_read_bytes_framed(s, pos6, WIRE_TAG_TARGET).unwrap();
    lemma_spec_read_bytes_framed_sound(s, pos7, WIRE_TAG_CONTENTS);
    let (_, pos8) = spec_read_bytes_framed(s, pos7, WIRE_TAG_CONTENTS).unwrap();
    lemma_spec_read_u64_seq_framed_sound(s, pos8, WIRE_TAG_EPOCHS);
    let (_, pos9) = spec_read_u64_seq_framed(s, pos8, WIRE_TAG_EPOCHS).unwrap();
    lemma_spec_read_bool_seq_framed_sound(s, pos9, WIRE_TAG_REQUIRED);
    let (_, pos10) = spec_read_bool_seq_framed(s, pos9, WIRE_TAG_REQUIRED).unwrap();

    let s0 = s.subrange(0, pos0);
    let s1 = s.subrange(pos0, pos1);
    let s2 = s.subrange(pos1, pos2);
    let s3 = s.subrange(pos2, pos3);
    let s4 = s.subrange(pos3, pos4);
    let s5 = s.subrange(pos4, pos5);
    let s6 = s.subrange(pos5, pos6);
    let s7 = s.subrange(pos6, pos7);
    let s8 = s.subrange(pos7, pos8);
    let s9 = s.subrange(pos8, pos9);
    let s10 = s.subrange(pos9, pos10);

    let t9 = s9 + s10;
    assert(t9 =~= s.subrange(pos8, pos10));
    let t8 = s8 + t9;
    assert(t8 =~= s.subrange(pos7, pos10));
    let t7 = s7 + t8;
    assert(t7 =~= s.subrange(pos6, pos10));
    let t6 = s6 + t7;
    assert(t6 =~= s.subrange(pos5, pos10));
    let t5 = s5 + t6;
    assert(t5 =~= s.subrange(pos4, pos10));
    let t4 = s4 + t5;
    assert(t4 =~= s.subrange(pos3, pos10));
    let t3 = s3 + t4;
    assert(t3 =~= s.subrange(pos2, pos10));
    let t2 = s2 + t3;
    assert(t2 =~= s.subrange(pos1, pos10));
    let t1 = s1 + t2;
    assert(t1 =~= s.subrange(pos0, pos10));
    let t0 = s0 + t1;
    assert(t0 =~= s.subrange(0, pos10));
    assert(s.subrange(0, pos10) =~= s);
}

proof fn lemma_spec_decode_permit_complete(p: SpecPermitWire)
    requires
        spec_permit_fits_u64(p),
        p.epochs.len() == p.required.len(),
    ensures
        spec_decode_permit(encode_permit_fixed_spec(p)) == Some(p),
{
    reveal(encode_permit_fixed_spec);
    let s = encode_permit_fixed_spec(p);
    let r10 = Seq::<u8>::empty();
    let r9 = encode_bool_seq_framed_spec(WIRE_TAG_REQUIRED, p.required);
    assert(r9 + r10 =~= r9);
    let r8 = encode_u64_seq_framed_spec(WIRE_TAG_EPOCHS, p.epochs) + r9;
    let r7 = encode_bytes_fixed_spec(WIRE_TAG_CONTENTS, p.contents) + r8;
    let r6 = encode_bytes_fixed_spec(WIRE_TAG_TARGET, p.target) + r7;
    let r5 = encode_u64_field_spec(WIRE_TAG_EXPECTED_REV, p.expected_rev) + r6;
    let r4 = encode_u64_field_spec(WIRE_TAG_EXPIRES, p.expires) + r5;
    let r3 = encode_u64_field_spec(WIRE_TAG_AMOUNT, p.amount) + r4;
    let r2 = encode_u8_field_spec(WIRE_TAG_EFFECT_KIND, p.effect_kind) + r3;
    let r1 = encode_bytes_fixed_spec(WIRE_TAG_ACTOR, p.actor) + r2;
    let r0 = encode_bytes_fixed_spec(WIRE_TAG_ID, p.id) + r1;
    assert(s.subrange(0, s.len() as int) =~= u64_le_bytes_spec(WIRE_TAG_PERMIT_V1) + r0);

    lemma_spec_read_tag_complete(s, 0, WIRE_TAG_PERMIT_V1, r0);
    let pos0: int = 8;
    lemma_spec_read_bytes_framed_complete(s, pos0, WIRE_TAG_ID, p.id, r1);
    let pos1: int = pos0 + 16 + (p.id.len() as int);
    lemma_spec_read_bytes_framed_complete(s, pos1, WIRE_TAG_ACTOR, p.actor, r2);
    let pos2: int = pos1 + 16 + (p.actor.len() as int);
    lemma_spec_read_u8_framed_complete(s, pos2, WIRE_TAG_EFFECT_KIND, p.effect_kind, r3);
    let pos3: int = pos2 + 9;
    lemma_spec_read_u64_framed_complete(s, pos3, WIRE_TAG_AMOUNT, p.amount, r4);
    let pos4: int = pos3 + 16;
    lemma_spec_read_u64_framed_complete(s, pos4, WIRE_TAG_EXPIRES, p.expires, r5);
    let pos5: int = pos4 + 16;
    lemma_spec_read_u64_framed_complete(s, pos5, WIRE_TAG_EXPECTED_REV, p.expected_rev, r6);
    let pos6: int = pos5 + 16;
    lemma_spec_read_bytes_framed_complete(s, pos6, WIRE_TAG_TARGET, p.target, r7);
    let pos7: int = pos6 + 16 + (p.target.len() as int);
    lemma_spec_read_bytes_framed_complete(s, pos7, WIRE_TAG_CONTENTS, p.contents, r8);
    let pos8: int = pos7 + 16 + (p.contents.len() as int);
    lemma_spec_read_u64_seq_framed_complete(s, pos8, WIRE_TAG_EPOCHS, p.epochs, r9);
    let pos9: int = pos8 + 16 + 8 * (p.epochs.len() as int);
    lemma_spec_read_bool_seq_framed_complete(s, pos9, WIRE_TAG_REQUIRED, p.required, r10);
    let pos10: int = pos9 + 16 + (p.required.len() as int);
    assert(s.subrange(pos10, s.len() as int).len() == 0);
    assert(pos10 == s.len() as int);
}

fn read_u64_le_exec(bytes: &Vec<u8>, pos: usize) -> (res: Option<(u64, usize)>)
    requires
        pos <= bytes.len(),
    ensures
        match res {
            Some((val, next_pos)) => {
                next_pos == pos + 8
                    && next_pos <= bytes.len()
                    && spec_read_u64_le(bytes@, pos as int) == Some((val, next_pos as int))
            }
            None => spec_read_u64_le(bytes@, pos as int) is None,
        },
{
    if bytes.len() - pos < 8 {
        return None;
    }
    let b0 = bytes[pos];
    let b1 = bytes[pos + 1];
    let b2 = bytes[pos + 2];
    let b3 = bytes[pos + 3];
    let b4 = bytes[pos + 4];
    let b5 = bytes[pos + 5];
    let b6 = bytes[pos + 6];
    let b7 = bytes[pos + 7];
    proof {
        lemma_le_bytes_u64_canonical(b0, b1, b2, b3, b4, b5, b6, b7);
    }
    let acc6: u64 = (b6 as u64) + 256u64 * (b7 as u64);
    let acc5: u64 = (b5 as u64) + 256u64 * acc6;
    let acc4: u64 = (b4 as u64) + 256u64 * acc5;
    let acc3: u64 = (b3 as u64) + 256u64 * acc4;
    let acc2: u64 = (b2 as u64) + 256u64 * acc3;
    let acc1: u64 = (b1 as u64) + 256u64 * acc2;
    let val: u64 = (b0 as u64) + 256u64 * acc1;
    let next_pos = pos + 8;
    proof {
        assert(bytes@.subrange(pos as int, next_pos as int)
            =~= seq![b0, b1, b2, b3, b4, b5, b6, b7]);
    }
    Some((val, next_pos))
}

fn read_expected_tag_exec(bytes: &Vec<u8>, pos: usize, expected_tag: u64) -> (res: Option<usize>)
    requires
        pos <= bytes.len(),
    ensures
        match res {
            Some(np) => {
                np <= bytes.len()
                    && spec_read_tag(bytes@, pos as int, expected_tag) == Some(np as int)
            }
            None => spec_read_tag(bytes@, pos as int, expected_tag) is None,
        },
{
    match read_u64_le_exec(bytes, pos) {
        None => None,
        Some((tag, next_pos)) => {
            if tag == expected_tag {
                Some(next_pos)
            } else {
                None
            }
        }
    }
}

fn read_u8_framed_exec(
    bytes: &Vec<u8>,
    pos: usize,
    expected_tag: u64,
) -> (res: Option<(u8, usize)>)
    requires
        pos <= bytes.len(),
    ensures
        match res {
            Some((v, np)) => {
                np <= bytes.len()
                    && spec_read_u8_framed(bytes@, pos as int, expected_tag)
                        == Some((v, np as int))
            }
            None => spec_read_u8_framed(bytes@, pos as int, expected_tag) is None,
        },
{
    let pos1 = match read_expected_tag_exec(bytes, pos, expected_tag) {
        None => return None,
        Some(p) => p,
    };
    if bytes.len() - pos1 < 1 {
        return None;
    }
    Some((bytes[pos1], pos1 + 1))
}

fn read_u64_framed_exec(
    bytes: &Vec<u8>,
    pos: usize,
    expected_tag: u64,
) -> (res: Option<(u64, usize)>)
    requires
        pos <= bytes.len(),
    ensures
        match res {
            Some((v, np)) => {
                np <= bytes.len()
                    && spec_read_u64_framed(bytes@, pos as int, expected_tag)
                        == Some((v, np as int))
            }
            None => spec_read_u64_framed(bytes@, pos as int, expected_tag) is None,
        },
{
    let pos1 = match read_expected_tag_exec(bytes, pos, expected_tag) {
        None => return None,
        Some(p) => p,
    };
    read_u64_le_exec(bytes, pos1)
}

fn read_bytes_framed_exec(
    bytes: &Vec<u8>,
    pos: usize,
    expected_tag: u64,
) -> (res: Option<(Vec<u8>, usize)>)
    requires
        pos <= bytes.len(),
    ensures
        match res {
            Some((out, np)) => {
                np <= bytes.len()
                    && spec_read_bytes_framed(bytes@, pos as int, expected_tag)
                        == Some((out@, np as int))
            }
            None => spec_read_bytes_framed(bytes@, pos as int, expected_tag) is None,
        },
{
    let pos1 = match read_expected_tag_exec(bytes, pos, expected_tag) {
        None => return None,
        Some(p) => p,
    };
    let (len_u64, pos2) = match read_u64_le_exec(bytes, pos1) {
        None => return None,
        Some(pair) => pair,
    };
    let rem_len: usize = bytes.len() - pos2;
    if len_u64 > (rem_len as u64) {
        return None;
    }
    let n: usize = len_u64 as usize;
    let mut out: Vec<u8> = Vec::new();
    let mut i: usize = 0;
    while i < n
        invariant
            pos2 <= bytes.len(),
            n <= bytes.len() - pos2,
            i <= n,
            out@ == bytes@.subrange(pos2 as int, pos2 as int + i as int),
        decreases n - i,
    {
        let byte = bytes[pos2 + i];
        out.push(byte);
        assert(bytes@.subrange(pos2 as int, pos2 as int + i as int + 1)
            =~= bytes@.subrange(pos2 as int, pos2 as int + i as int).push(byte));
        i += 1;
    }
    Some((out, pos2 + n))
}

fn read_u64_seq_framed_exec(
    bytes: &Vec<u8>,
    pos: usize,
    expected_tag: u64,
) -> (res: Option<(Vec<u64>, usize)>)
    requires
        pos <= bytes.len(),
    ensures
        match res {
            Some((out, np)) => {
                np <= bytes.len()
                    && spec_read_u64_seq_framed(bytes@, pos as int, expected_tag)
                        == Some((out@, np as int))
            }
            None => spec_read_u64_seq_framed(bytes@, pos as int, expected_tag) is None,
        },
{
    let pos1 = match read_expected_tag_exec(bytes, pos, expected_tag) {
        None => return None,
        Some(p) => p,
    };
    let (len_u64, pos2) = match read_u64_le_exec(bytes, pos1) {
        None => return None,
        Some(pair) => pair,
    };
    let max_elems: usize = (bytes.len() - pos2) / 8;
    if len_u64 > (max_elems as u64) {
        return None;
    }
    let n: usize = len_u64 as usize;
    let mut out: Vec<u64> = Vec::new();
    let mut i: usize = 0;
    let mut cur: usize = pos2;
    while i < n
        invariant
            pos2 <= bytes.len(),
            n <= (bytes.len() - pos2) / 8,
            i <= n,
            cur == pos2 + 8 * i,
            cur <= bytes.len(),
            out@ == spec_decode_u64_elems(bytes@, pos2 as int, i as nat),
        decreases n - i,
    {
        assert(bytes.len() - cur >= 8);
        let (v, next_cur) = read_u64_le_exec(bytes, cur).unwrap();
        out.push(v);
        assert(out@ =~= spec_decode_u64_elems(bytes@, pos2 as int, (i + 1) as nat));
        cur = next_cur;
        i += 1;
    }
    Some((out, cur))
}

fn read_bool_seq_framed_exec(
    bytes: &Vec<u8>,
    pos: usize,
    expected_tag: u64,
) -> (res: Option<(Vec<bool>, usize)>)
    requires
        pos <= bytes.len(),
    ensures
        match res {
            Some((out, np)) => {
                np <= bytes.len()
                    && spec_read_bool_seq_framed(bytes@, pos as int, expected_tag)
                        == Some((out@, np as int))
            }
            None => spec_read_bool_seq_framed(bytes@, pos as int, expected_tag) is None,
        },
{
    let pos1 = match read_expected_tag_exec(bytes, pos, expected_tag) {
        None => return None,
        Some(p) => p,
    };
    let (len_u64, pos2) = match read_u64_le_exec(bytes, pos1) {
        None => return None,
        Some(pair) => pair,
    };
    let rem_len: usize = bytes.len() - pos2;
    if len_u64 > (rem_len as u64) {
        return None;
    }
    let n: usize = len_u64 as usize;
    let mut out: Vec<bool> = Vec::new();
    let mut i: usize = 0;
    while i < n
        invariant
            pos <= bytes.len(),
            spec_read_tag(bytes@, pos as int, expected_tag) == Some(pos1 as int),
            spec_read_u64_le(bytes@, pos1 as int) == Some((len_u64, pos2 as int)),
            len_u64 <= (bytes.len() - pos2) as u64,
            n == len_u64 as usize,
            pos2 <= bytes.len(),
            n <= bytes.len() - pos2,
            i <= n,
            spec_decode_bool_elems(bytes@, pos2 as int, i as nat) == Some(out@),
        decreases n - i,
    {
        let raw = bytes[pos2 + i];
        if raw > 1u8 {
            proof {
                assert(spec_decode_bool_elems(bytes@, pos2 as int, (i + 1) as nat) is None);
                lemma_spec_decode_bool_elems_none(bytes@, pos2 as int, (i + 1) as nat, n as nat);
            }
            return None;
        }
        let flag = raw == 1u8;
        out.push(flag);
        assert(spec_decode_bool_elems(bytes@, pos2 as int, (i + 1) as nat) == Some(out@));
        i += 1;
    }
    Some((out, pos2 + n))
}

/// Total, panic-free canonical wire decoder for `ExecPermitWire`.
/// Proves both strict canonical non-malleability (`decode(b) == Some(p) ==> encode(p) == b`)
/// and exact roundtrip completeness (`decode(encode(p)) == Some(*p)`), ruling out
/// Broker-Sink parser differentials across arbitrary untrusted byte slices.
pub fn decode_permit_canonical(bytes: &Vec<u8>) -> (res: Option<ExecPermitWire>)
    ensures
        match res {
            Some(out) => {
                spec_decode_permit(bytes@) == Some(out@)
                    && spec_permit_fits_u64(out@)
                    && out@.epochs.len() == out@.required.len()
                    && encode_permit_fixed_spec(out@) == bytes@
                    && encode_permit_spec(out@) == bytes@
            }
            None => spec_decode_permit(bytes@) is None,
        },
{
    let pos0 = match read_expected_tag_exec(bytes, 0, WIRE_TAG_PERMIT_V1) {
        None => return None,
        Some(p) => p,
    };
    let (id, pos1) = match read_bytes_framed_exec(bytes, pos0, WIRE_TAG_ID) {
        None => return None,
        Some(pair) => pair,
    };
    let (actor, pos2) = match read_bytes_framed_exec(bytes, pos1, WIRE_TAG_ACTOR) {
        None => return None,
        Some(pair) => pair,
    };
    let (effect_kind, pos3) = match read_u8_framed_exec(bytes, pos2, WIRE_TAG_EFFECT_KIND) {
        None => return None,
        Some(pair) => pair,
    };
    let (amount, pos4) = match read_u64_framed_exec(bytes, pos3, WIRE_TAG_AMOUNT) {
        None => return None,
        Some(pair) => pair,
    };
    let (expires, pos5) = match read_u64_framed_exec(bytes, pos4, WIRE_TAG_EXPIRES) {
        None => return None,
        Some(pair) => pair,
    };
    let (expected_rev, pos6) = match read_u64_framed_exec(bytes, pos5, WIRE_TAG_EXPECTED_REV) {
        None => return None,
        Some(pair) => pair,
    };
    let (target, pos7) = match read_bytes_framed_exec(bytes, pos6, WIRE_TAG_TARGET) {
        None => return None,
        Some(pair) => pair,
    };
    let (contents, pos8) = match read_bytes_framed_exec(bytes, pos7, WIRE_TAG_CONTENTS) {
        None => return None,
        Some(pair) => pair,
    };
    let (epochs, pos9) = match read_u64_seq_framed_exec(bytes, pos8, WIRE_TAG_EPOCHS) {
        None => return None,
        Some(pair) => pair,
    };
    let (required, pos10) = match read_bool_seq_framed_exec(bytes, pos9, WIRE_TAG_REQUIRED) {
        None => return None,
        Some(pair) => pair,
    };
    if pos10 != bytes.len() {
        return None;
    }
    if epochs.len() != required.len() {
        return None;
    }
    let decoded = ExecPermitWire {
        id,
        actor,
        effect_kind,
        amount,
        expires,
        expected_rev,
        target,
        contents,
        epochs,
        required,
    };
    proof {
        lemma_spec_decode_permit_sound(bytes@);
    }
    Some(decoded)
}

/// Executable roundtrip bijection witness: encodes `p` and decodes via `decode_permit_canonical`,
/// with Z3 statically proving `unwrap()` never panics and `out@ == p@`.
pub fn roundtrip_permit_canonical(p: &ExecPermitWire) -> (out: ExecPermitWire)
    requires
        p.epochs@.len() == p.required@.len(),
    ensures
        out@ == p@,
{
    let wire = encode_permit_canonical(p);
    proof {
        lemma_spec_decode_permit_complete(p@);
    }
    let dec = decode_permit_canonical(&wire);
    dec.unwrap()
}

/// Mechanized two-way canonical wire bijection theorem over arbitrary untrusted byte slices:
/// `spec_decode_permit(b) == Some(p)` holds if and only if `p` is well-formed (`fits_u64` and
/// `epochs.len() == required.len()`) and `encode_permit_fixed_spec(p) == b`.
pub proof fn encode_decode_permit_bijection(b: Seq<u8>, p: SpecPermitWire)
    ensures
        (spec_decode_permit(b) == Some(p))
            <==> (spec_permit_fits_u64(p)
                && p.epochs.len() == p.required.len()
                && encode_permit_fixed_spec(p) == b),
{
    if spec_decode_permit(b) == Some(p) {
        lemma_spec_decode_permit_sound(b);
    }
    if spec_permit_fits_u64(p)
        && p.epochs.len() == p.required.len()
        && encode_permit_fixed_spec(p) == b
    {
        lemma_spec_decode_permit_complete(p);
    }
}

// ============================================================================
// V5 Pillar B: Arbitrary-Depth Delegation & Self-Modification Non-Escalation
// ============================================================================

#[derive(Debug, PartialEq, Eq)]
pub struct ExecScopeGrant {
    pub budget: u64,
    pub expires: u64,
    pub op_mask: u64,
    pub required: Vec<bool>,
    pub issued_epochs: Vec<u64>,
}

pub struct SpecScopeGrant {
    pub budget: u64,
    pub expires: u64,
    pub op_mask: u64,
    pub required: Seq<bool>,
    pub issued_epochs: Seq<u64>,
}

impl View for ExecScopeGrant {
    type V = SpecScopeGrant;

    open spec fn view(&self) -> SpecScopeGrant {
        SpecScopeGrant {
            budget: self.budget,
            expires: self.expires,
            op_mask: self.op_mask,
            required: self.required@,
            issued_epochs: self.issued_epochs@,
        }
    }
}

pub open spec fn bitmask_subset_spec(sub: u64, sup: u64) -> bool {
    (sub & !sup) == 0u64
}

#[verifier::bit_vector]
proof fn lemma_bitmask_subset_refl_trans(a: u64, b: u64, c: u64)
    ensures
        (a & !a) == 0u64,
        ((b & !a) == 0u64 && (c & !b) == 0u64) ==> ((c & !a) == 0u64),
{
}

pub open spec fn grant_wf_spec(g: SpecScopeGrant) -> bool {
    g.required.len() == g.issued_epochs.len()
}

pub open spec fn attenuates_grant_spec(parent: SpecScopeGrant, child: SpecScopeGrant) -> bool {
    grant_wf_spec(parent)
        && grant_wf_spec(child)
        && child.required.len() == parent.required.len()
        && child.budget <= parent.budget
        && child.expires <= parent.expires
        && bitmask_subset_spec(child.op_mask, parent.op_mask)
        && forall|j: int|
            0 <= j < parent.required.len() ==> (
                ((#[trigger] parent.required[j]) ==> child.required[j])
                && child.issued_epochs[j] >= parent.issued_epochs[j]
            )
}

pub open spec fn grant_live_at_spec(
    g: SpecScopeGrant,
    live: Seq<bool>,
    epochs: Seq<u64>,
    now: u64,
) -> bool {
    grant_wf_spec(g)
        && g.required.len() == live.len()
        && g.required.len() == epochs.len()
        && now < g.expires
        && forall|j: int|
            0 <= j < g.required.len() ==> ((#[trigger] g.required[j]) ==> (live[j]
                && epochs[j] == g.issued_epochs[j]))
}

pub open spec fn chain_attenuates_spec(root: SpecScopeGrant, chain: Seq<SpecScopeGrant>) -> bool
    decreases chain.len(),
{
    if chain.len() == 0 {
        grant_wf_spec(root)
    } else if chain.len() == 1 {
        attenuates_grant_spec(root, chain[0])
    } else {
        chain_attenuates_spec(root, chain.drop_last())
            && attenuates_grant_spec(chain[chain.len() - 2], chain.last())
    }
}

pub open spec fn chain_leaf_spec(root: SpecScopeGrant, chain: Seq<SpecScopeGrant>) -> SpecScopeGrant
{
    if chain.len() == 0 {
        root
    } else {
        chain.last()
    }
}

pub open spec fn chain_all_live_spec(
    root: SpecScopeGrant,
    chain: Seq<SpecScopeGrant>,
    live: Seq<bool>,
    epochs: Seq<u64>,
    now: u64,
) -> bool {
    grant_live_at_spec(root, live, epochs, now)
        && chain_attenuates_spec(root, chain)
        && forall|k: int|
            0 <= k < chain.len() ==> grant_live_at_spec(#[trigger] chain[k], live, epochs, now)
}

pub fn attenuates_grant_exec(parent: &ExecScopeGrant, child: &ExecScopeGrant) -> (ok: bool)
    ensures
        ok == attenuates_grant_spec(parent@, child@),
{
    if parent.required.len() != parent.issued_epochs.len()
        || child.required.len() != child.issued_epochs.len()
        || child.required.len() != parent.required.len()
        || child.budget > parent.budget
        || child.expires > parent.expires
        || (child.op_mask & !parent.op_mask) != 0u64
    {
        return false;
    }
    let mut i: usize = 0;
    while i < parent.required.len()
        invariant
            parent.required.len() == parent.issued_epochs.len(),
            child.required.len() == child.issued_epochs.len(),
            child.required.len() == parent.required.len(),
            child.budget <= parent.budget,
            child.expires <= parent.expires,
            bitmask_subset_spec(child.op_mask, parent.op_mask),
            i <= parent.required.len(),
            forall|j: int|
                0 <= j < i ==> (
                    ((#[trigger] parent.required@[j]) ==> child.required@[j])
                    && child.issued_epochs@[j] >= parent.issued_epochs@[j]
                ),
        decreases parent.required.len() - i,
    {
        if (parent.required[i] && !child.required[i])
            || child.issued_epochs[i] < parent.issued_epochs[i]
        {
            return false;
        }
        i += 1;
    }
    true
}

pub fn grant_live_at_exec(
    g: &ExecScopeGrant,
    live: &Vec<bool>,
    epochs: &Vec<u64>,
    now: u64,
) -> (ok: bool)
    ensures
        ok == grant_live_at_spec(g@, live@, epochs@, now),
{
    if g.required.len() != g.issued_epochs.len()
        || g.required.len() != live.len()
        || g.required.len() != epochs.len()
        || now >= g.expires
    {
        return false;
    }
    let mut i: usize = 0;
    while i < g.required.len()
        invariant
            g.required.len() == g.issued_epochs.len(),
            g.required.len() == live.len(),
            g.required.len() == epochs.len(),
            now < g.expires,
            i <= g.required.len(),
            forall|j: int|
                0 <= j < i ==> ((#[trigger] g.required@[j]) ==> (live@[j]
                    && epochs@[j] == g.issued_epochs@[j])),
        decreases g.required.len() - i,
    {
        if g.required[i] && (!live[i] || epochs[i] != g.issued_epochs[i]) {
            return false;
        }
        i += 1;
    }
    true
}

pub open spec fn spec_grant_seq(xs: Seq<ExecScopeGrant>) -> Seq<SpecScopeGrant> {
    Seq::new(xs.len(), |i: int| xs[i]@)
}

proof fn lemma_chain_attenuates_prefix(
    root: SpecScopeGrant,
    sc: Seq<SpecScopeGrant>,
    k: nat,
)
    requires
        1 <= k <= sc.len(),
        chain_attenuates_spec(root, sc),
    ensures
        chain_attenuates_spec(root, sc.subrange(0, k as int)),
    decreases sc.len() - k,
{
    if k == sc.len() {
        assert(sc.subrange(0, k as int) =~= sc);
    } else {
        assert(sc.drop_last() =~= sc.subrange(0, sc.len() - 1));
        lemma_chain_attenuates_prefix(root, sc.drop_last(), k);
        assert(sc.drop_last().subrange(0, k as int) =~= sc.subrange(0, k as int));
    }
}

/// Verifies an arbitrary-depth `d`-hop sub-delegation chain from `root` through `chain`.
pub fn verify_delegation_chain_exec(
    root: &ExecScopeGrant,
    chain: &Vec<ExecScopeGrant>,
    live: &Vec<bool>,
    epochs: &Vec<u64>,
    now: u64,
) -> (ok: bool)
    ensures
        ok == chain_all_live_spec(
            root@,
            spec_grant_seq(chain@),
            live@,
            epochs@,
            now,
        ),
{
    let ghost sc = spec_grant_seq(chain@);
    if !grant_live_at_exec(root, live, epochs, now) {
        return false;
    }
    if chain.len() == 0 {
        assert(sc =~= Seq::<SpecScopeGrant>::empty());
        return true;
    }
    if !attenuates_grant_exec(root, &chain[0]) || !grant_live_at_exec(&chain[0], live, epochs, now)
    {
        proof {
            if chain_all_live_spec(root@, sc, live@, epochs@, now) {
                lemma_chain_attenuates_prefix(root@, sc, 1);
                assert(sc.subrange(0, 1)[0] == sc[0]);
                assert(attenuates_grant_spec(root@, sc[0]));
                assert(grant_live_at_spec(sc[0], live@, epochs@, now));
                assert(false);
            }
        }
        return false;
    }
    let mut k: usize = 1;
    assert(sc.subrange(0, 1).len() == 1);
    assert(sc.subrange(0, 1)[0] == sc[0]);
    while k < chain.len()
        invariant
            1 <= k <= chain.len(),
            sc == spec_grant_seq(chain@),
            grant_live_at_spec(root@, live@, epochs@, now),
            chain_attenuates_spec(root@, sc.subrange(0, k as int)),
            forall|j: int| 0 <= j < k ==> grant_live_at_spec(#[trigger] sc[j], live@, epochs@, now),
        decreases chain.len() - k,
    {
        if !attenuates_grant_exec(&chain[k - 1], &chain[k])
            || !grant_live_at_exec(&chain[k], live, epochs, now)
        {
            proof {
                if chain_all_live_spec(root@, sc, live@, epochs@, now) {
                    lemma_chain_attenuates_prefix(root@, sc, (k + 1) as nat);
                    let sub_k1 = sc.subrange(0, k as int + 1);
                    assert(sub_k1.len() == k + 1);
                    assert(sub_k1[k as int - 1] == sc[k as int - 1]);
                    assert(sub_k1.last() == sc[k as int]);
                    assert(attenuates_grant_spec(sc[k as int - 1], sc[k as int]));
                    assert(grant_live_at_spec(sc[k as int], live@, epochs@, now));
                    assert(false);
                }
            }
            return false;
        }
        proof {
            let next_sub = sc.subrange(0, k as int + 1);
            assert(next_sub.drop_last() =~= sc.subrange(0, k as int));
            assert(next_sub[next_sub.len() - 2] == sc[k as int - 1]);
            assert(next_sub.last() == sc[k as int]);
        }
        k += 1;
    }
    assert(sc.subrange(0, chain.len() as int) =~= sc);
    true
}

/// Verifies a controller self-modification step `curr -> next_ctrl`: absent an explicit fresh
/// human quorum override (`fresh_quorum_override`), the successor controller must strictly
/// attenuate `curr`.
pub fn verify_self_upgrade_exec(
    curr: &ExecScopeGrant,
    next_ctrl: &ExecScopeGrant,
    fresh_quorum_override: bool,
) -> (ok: bool)
    ensures
        ok == (if fresh_quorum_override {
            grant_wf_spec(next_ctrl@)
        } else {
            attenuates_grant_spec(curr@, next_ctrl@)
        }),
{
    if fresh_quorum_override {
        next_ctrl.required.len() == next_ctrl.issued_epochs.len()
    } else {
        attenuates_grant_exec(curr, next_ctrl)
    }
}

pub proof fn attenuates_grant_transitive(
    g0: SpecScopeGrant,
    g1: SpecScopeGrant,
    g2: SpecScopeGrant,
)
    requires
        attenuates_grant_spec(g0, g1),
        attenuates_grant_spec(g1, g2),
    ensures
        attenuates_grant_spec(g0, g2),
{
    lemma_bitmask_subset_refl_trans(g0.op_mask, g1.op_mask, g2.op_mask);
    assert forall|j: int|
        0 <= j < g0.required.len() implies (
            ((#[trigger] g0.required[j]) ==> g2.required[j])
            && g2.issued_epochs[j] >= g0.issued_epochs[j]
        ) by {
        assert(g0.required[j] ==> g1.required[j]);
        assert(g1.required[j] ==> g2.required[j]);
    }
}

/// Mechanized non-amplification theorem across arbitrary-depth `d`-hop delegation or
/// unquorumed self-modification chains: the leaf grant can never expand root budget,
/// expiry, operation bitmask, or drop any required human issuer.
pub proof fn delegation_chain_no_amplification(
    root: SpecScopeGrant,
    chain: Seq<SpecScopeGrant>,
)
    requires
        chain_attenuates_spec(root, chain),
    ensures
        attenuates_grant_spec(root, chain_leaf_spec(root, chain)),
    decreases chain.len(),
{
    if chain.len() == 0 {
        lemma_bitmask_subset_refl_trans(root.op_mask, root.op_mask, root.op_mask);
    } else if chain.len() == 1 {
    } else {
        delegation_chain_no_amplification(root, chain.drop_last());
        assert(chain_leaf_spec(root, chain.drop_last()) == chain[chain.len() - 2]);
        attenuates_grant_transitive(root, chain[chain.len() - 2], chain.last());
    }
}

/// Mechanized subtree-severance theorem: withdrawing *any* issuer `j` required at *any*
/// ancestor hop (`root` or `chain[anc]`) deterministically invalidates the entire
/// descendant delegation subtree.
pub proof fn ancestor_withdrawal_severs_subtree(
    root: SpecScopeGrant,
    chain: Seq<SpecScopeGrant>,
    live: Seq<bool>,
    epochs: Seq<u64>,
    now: u64,
    anc: int,
    issuer: int,
)
    requires
        0 <= issuer < root.required.len(),
        (anc == -1 && root.required[issuer]
            && (issuer >= live.len()
                || issuer >= epochs.len()
                || !live[issuer]
                || epochs[issuer] != root.issued_epochs[issuer]))
            || (0 <= anc < chain.len()
                && issuer < chain[anc].required.len()
                && chain[anc].required[issuer]
                && (issuer >= live.len()
                    || issuer >= epochs.len()
                    || !live[issuer]
                    || epochs[issuer] != chain[anc].issued_epochs[issuer])),
    ensures
        !chain_all_live_spec(root, chain, live, epochs, now),
{
    if chain_all_live_spec(root, chain, live, epochs, now) {
        if anc == -1 {
            assert(grant_live_at_spec(root, live, epochs, now));
            assert(false);
        } else {
            assert(grant_live_at_spec(chain[anc], live, epochs, now));
            assert(false);
        }
    }
}

// ============================================================================
// V5 Pillar C: Multi-Sink Cluster Quorum & Lease Fence Kernel
// ============================================================================

#[derive(Debug, PartialEq, Eq)]
pub struct ExecShardSink {
    pub sink: ExecSinkKernel,
    pub lease_expiry: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ExecClusterFenceKernel {
    pub shards: Vec<ExecShardSink>,
    pub num_issuers: usize,
}

pub open spec fn shard_quiesced_spec(
    shard: ExecShardSink,
    issuer: int,
    floor: u64,
    now: u64,
) -> bool {
    0 <= issuer < shard.sink.floors@.len()
        && (shard.sink.floors@[issuer] >= floor || now >= shard.lease_expiry)
}

pub open spec fn cluster_inv_spec(c: ExecClusterFenceKernel) -> bool {
    c.shards@.len() > 0
        && forall|s: int|
            0 <= s < c.shards@.len() ==> (
                (#[trigger] c.shards@[s]).sink.sink_inv()
                && c.shards@[s].sink.floors@.len() == c.num_issuers as nat
            )
}

pub open spec fn cluster_fence_acked_spec(
    c: ExecClusterFenceKernel,
    issuer: int,
    floor: u64,
    now: u64,
) -> bool {
    cluster_inv_spec(c)
        && 0 <= issuer < c.num_issuers as int
        && forall|s: int|
            0 <= s < c.shards@.len() ==> shard_quiesced_spec(
                #[trigger] c.shards@[s],
                issuer,
                floor,
                now,
            )
}

pub fn cluster_fence_acked_exec(
    c: &ExecClusterFenceKernel,
    issuer: usize,
    floor: u64,
    now: u64,
) -> (acked: bool)
    requires
        cluster_inv_spec(*c),
    ensures
        acked == cluster_fence_acked_spec(*c, issuer as int, floor, now),
{
    if issuer >= c.num_issuers {
        return false;
    }
    let mut s: usize = 0;
    while s < c.shards.len()
        invariant
            cluster_inv_spec(*c),
            issuer < c.num_issuers,
            s <= c.shards.len(),
            forall|k: int|
                0 <= k < s ==> shard_quiesced_spec(
                    #[trigger] c.shards@[k],
                    issuer as int,
                    floor,
                    now,
                ),
        decreases c.shards.len() - s,
    {
        let sh = &c.shards[s];
        assert(sh.sink.floors@.len() == c.num_issuers as nat);
        if sh.sink.floors[issuer] < floor && now < sh.lease_expiry {
            assert(!shard_quiesced_spec(c.shards@[s as int], issuer as int, floor, now));
            return false;
        }
        s += 1;
    }
    true
}

/// Lease-and-fence gated shard completion: rejects commits both when the shard's lease has
/// expired (`now >= shard.lease_expiry`) and when `unfenced` fails on `shard.sink`.
pub fn cluster_shard_complete_exec(
    shard: &mut ExecShardSink,
    id: &Vec<u8>,
    now: u64,
) -> (out: ExecCompleteOutcome)
    requires
        old(shard).sink.sink_inv(),
    ensures
        final(shard).sink.sink_inv(),
        final(shard).lease_expiry == old(shard).lease_expiry,
        final(shard).sink.floors@ == old(shard).sink.floors@,
        now >= old(shard).lease_expiry ==> (
            out == ExecCompleteOutcome::Cancelled
            && final(shard).sink.committed_total == old(shard).sink.committed_total
        ),
        match find_sink_record_spec(old(shard).sink.records@, id@) {
            Some(idx) => {
                let sr = old(shard).sink.records@[idx];
                (now >= old(shard).lease_expiry
                    || !unfenced_spec(sr.permit@.required, sr.permit@.epochs, old(shard).sink.floors@))
                    ==> (!(out is NewlyCommitted)
                        && final(shard).sink.committed_total == old(shard).sink.committed_total)
            }
            None => {
                !(out is NewlyCommitted)
                    && final(shard).sink.committed_total == old(shard).sink.committed_total
            }
        },
{
    if now >= shard.lease_expiry {
        return ExecCompleteOutcome::Cancelled;
    }
    shard.sink.sink_complete(id)
}

/// Mechanized multi-sink cluster post-acknowledgment exclusion theorem: once
/// `cluster_fence_acked_spec(c, issuer, floor, ack_time)` holds, no shard `s` in the
/// cluster can commit any permit requiring `issuer` with `epochs[issuer] < floor` at
/// any timestamp `t >= ack_time`.
pub proof fn cluster_quorum_post_ack_exclusion(
    c: &ExecClusterFenceKernel,
    shard_idx: int,
    permit: SpecPermitWire,
    issuer: int,
    floor: u64,
    ack_time: u64,
    commit_time: u64,
)
    requires
        cluster_fence_acked_spec(*c, issuer, floor, ack_time),
        0 <= shard_idx < c.shards@.len(),
        commit_time >= ack_time,
        permit.required.len() == c.num_issuers as nat,
        permit.epochs.len() == c.num_issuers as nat,
        permit.required[issuer],
        permit.epochs[issuer] < floor,
    ensures
        commit_time >= c.shards@[shard_idx].lease_expiry
            || !unfenced_spec(
                permit.required,
                permit.epochs,
                c.shards@[shard_idx].sink.floors@,
            ),
{
    let sh = c.shards@[shard_idx];
    assert(shard_quiesced_spec(sh, issuer, floor, ack_time));
    if commit_time < sh.lease_expiry {
        assert(ack_time < sh.lease_expiry);
        assert(sh.sink.floors@[issuer] >= floor);
        fenced_zero(permit.required, permit.epochs, sh.sink.floors@, issuer);
    }
}

/// Bounded physical clock-skew quiescence: under physical clock skew bound
/// `|t_shard - t_broker| <= max_skew` (in particular `t_broker - t_shard <= max_skew`),
/// a partitioned shard lacking a signed fence ack is quiesced at broker clock `broker_now`
/// iff `broker_now >= shard.lease_expiry + max_skew`.
pub open spec fn shard_quiesced_bounded_skew_spec(
    shard: ExecShardSink,
    issuer: int,
    floor: u64,
    broker_now: u64,
    max_skew: u64,
) -> bool {
    0 <= issuer < shard.sink.floors@.len()
        && (shard.sink.floors@[issuer] >= floor
            || broker_now as int >= shard.lease_expiry as int + max_skew as int)
}

pub open spec fn cluster_fence_acked_bounded_skew_spec(
    c: ExecClusterFenceKernel,
    issuer: int,
    floor: u64,
    broker_now: u64,
    max_skew: u64,
) -> bool {
    cluster_inv_spec(c)
        && 0 <= issuer < c.num_issuers as int
        && forall|s: int|
            0 <= s < c.shards@.len() ==> shard_quiesced_bounded_skew_spec(
                #[trigger] c.shards@[s],
                issuer,
                floor,
                broker_now,
                max_skew,
            )
}

pub fn cluster_fence_acked_bounded_skew_exec(
    c: &ExecClusterFenceKernel,
    issuer: usize,
    floor: u64,
    broker_now: u64,
    max_skew: u64,
) -> (acked: bool)
    requires
        cluster_inv_spec(*c),
    ensures
        acked == cluster_fence_acked_bounded_skew_spec(*c, issuer as int, floor, broker_now, max_skew),
{
    if issuer >= c.num_issuers {
        return false;
    }
    let mut s: usize = 0;
    while s < c.shards.len()
        invariant
            cluster_inv_spec(*c),
            issuer < c.num_issuers,
            s <= c.shards.len(),
            forall|k: int|
                0 <= k < s ==> shard_quiesced_bounded_skew_spec(
                    #[trigger] c.shards@[k],
                    issuer as int,
                    floor,
                    broker_now,
                    max_skew,
                ),
        decreases c.shards.len() - s,
    {
        let sh = &c.shards[s];
        assert(sh.sink.floors@.len() == c.num_issuers as nat);
        let skew_buffered_expired = broker_now >= max_skew && (broker_now - max_skew) >= sh.lease_expiry;
        if sh.sink.floors[issuer] < floor && !skew_buffered_expired {
            assert(!shard_quiesced_bounded_skew_spec(
                c.shards@[s as int],
                issuer as int,
                floor,
                broker_now,
                max_skew,
            ));
            return false;
        }
        s += 1;
    }
    true
}

/// Mechanized post-acknowledgment exclusion under bounded physical clock skew:
/// once `cluster_fence_acked_bounded_skew_spec(c, issuer, floor, broker_ack_time, max_skew)`
/// holds, at any physical instant with `broker_commit_time >= broker_ack_time` and
/// bounded clock lag `broker_commit_time - shard_commit_time <= max_skew`, no shard `s`
/// can commit a pre-revocation permit (`epochs[issuer] < floor`) on its local clock.
pub proof fn cluster_quorum_skew_bounded_post_ack_exclusion(
    c: &ExecClusterFenceKernel,
    shard_idx: int,
    permit: SpecPermitWire,
    issuer: int,
    floor: u64,
    broker_ack_time: u64,
    broker_commit_time: u64,
    shard_commit_time: u64,
    max_skew: u64,
)
    requires
        cluster_fence_acked_bounded_skew_spec(*c, issuer, floor, broker_ack_time, max_skew),
        0 <= shard_idx < c.shards@.len(),
        broker_commit_time >= broker_ack_time,
        broker_commit_time as int - shard_commit_time as int <= max_skew as int,
        permit.required.len() == c.num_issuers as nat,
        permit.epochs.len() == c.num_issuers as nat,
        permit.required[issuer],
        permit.epochs[issuer] < floor,
    ensures
        shard_commit_time >= c.shards@[shard_idx].lease_expiry
            || !unfenced_spec(
                permit.required,
                permit.epochs,
                c.shards@[shard_idx].sink.floors@,
            ),
{
    let sh = c.shards@[shard_idx];
    assert(shard_quiesced_bounded_skew_spec(sh, issuer, floor, broker_ack_time, max_skew));
    if shard_commit_time < sh.lease_expiry {
        assert((broker_ack_time as int) < sh.lease_expiry as int + max_skew as int);
        assert(sh.sink.floors@[issuer] >= floor);
        fenced_zero(permit.required, permit.epochs, sh.sink.floors@, issuer);
    }
}

// ============================================================================
// V6 Pillar D: Compositional Relational Mutex & Two-Phase Staged Commit Kernel
// ============================================================================

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ExecToxicPair {
    pub mask_a: u64,
    pub mask_b: u64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ExecStagedEffect {
    pub cap_mask: u64,
    pub required_witness_hash: u64,
    pub required_epoch: u64,
    pub committed: bool,
    pub witnessed: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ExecRelationalSinkKernel {
    pub active_cap_mask: u64,
    pub sink_epoch_floor: u64,
    pub toxic_pairs: Vec<ExecToxicPair>,
    pub staged: Vec<ExecStagedEffect>,
    pub committed_staged_count: u64,
}

pub open spec fn pair_conflict_spec(mask: u64, p: ExecToxicPair) -> bool {
    ((mask & p.mask_a) != 0u64) && ((mask & p.mask_b) != 0u64)
}

pub open spec fn relational_mutex_safe_spec(
    active_mask: u64,
    pairs: Seq<ExecToxicPair>,
) -> bool {
    forall|i: int| 0 <= i < pairs.len() ==> !pair_conflict_spec(active_mask, #[trigger] pairs[i])
}

pub open spec fn staged_inv_spec(k: ExecRelationalSinkKernel) -> bool {
    relational_mutex_safe_spec(k.active_cap_mask, k.toxic_pairs@)
        && forall|j: int|
            0 <= j < k.staged@.len() ==> (
                (#[trigger] k.staged@[j]).committed ==> (
                    k.staged@[j].witnessed
                    && bitmask_subset_spec(k.staged@[j].cap_mask, k.active_cap_mask)
                )
            )
}

#[verifier::bit_vector]
proof fn lemma_bitmask_union_monotone(old_m: u64, delta_m: u64, sub_m: u64)
    ensures
        ((sub_m & !old_m) == 0u64) ==> ((sub_m & !(old_m | delta_m)) == 0u64),
        (delta_m & !(old_m | delta_m)) == 0u64,
{
}

/// Checks whether activating `delta_mask` alongside `active_mask` preserves every
/// relational incompatibility pair in `pairs`.
pub fn verify_relational_mutex_exec(
    active_mask: u64,
    delta_mask: u64,
    pairs: &Vec<ExecToxicPair>,
) -> (safe: bool)
    ensures
        safe == relational_mutex_safe_spec(active_mask | delta_mask, pairs@),
{
    let cand = active_mask | delta_mask;
    let mut i: usize = 0;
    while i < pairs.len()
        invariant
            cand == (active_mask | delta_mask),
            i <= pairs.len(),
            forall|k: int| 0 <= k < i ==> !pair_conflict_spec(cand, #[trigger] pairs@[k]),
        decreases pairs.len() - i,
    {
        let p = pairs[i];
        if (cand & p.mask_a) != 0u64 && (cand & p.mask_b) != 0u64 {
            assert(pair_conflict_spec(cand, pairs@[i as int]));
            return false;
        }
        i += 1;
    }
    true
}

/// Commits a two-phase staged effect at `slot` if and only if it has not yet committed,
/// `provided_witness_hash` matches `required_witness_hash`, its `required_epoch` is not
/// fenced by `sink_epoch_floor`, and activating `cap_mask` preserves all relational mutexes.
pub fn commit_staged_postcondition_exec(
    k: &mut ExecRelationalSinkKernel,
    slot: usize,
    provided_witness_hash: u64,
) -> (committed_now: bool)
    requires
        staged_inv_spec(*old(k)),
        old(k).committed_staged_count < u64::MAX,
    ensures
        staged_inv_spec(*final(k)),
        final(k).toxic_pairs@ == old(k).toxic_pairs@,
        final(k).sink_epoch_floor == old(k).sink_epoch_floor,
        final(k).staged@.len() == old(k).staged@.len(),
        committed_now ==> (
            slot < old(k).staged@.len()
            && !old(k).staged@[slot as int].committed
            && provided_witness_hash == old(k).staged@[slot as int].required_witness_hash
            && old(k).staged@[slot as int].required_epoch >= old(k).sink_epoch_floor
            && relational_mutex_safe_spec(
                old(k).active_cap_mask | old(k).staged@[slot as int].cap_mask,
                old(k).toxic_pairs@,
            )
            && final(k).active_cap_mask
                == (old(k).active_cap_mask | old(k).staged@[slot as int].cap_mask)
            && final(k).committed_staged_count == old(k).committed_staged_count + 1
            && final(k).staged@[slot as int].committed
            && final(k).staged@[slot as int].witnessed
        ),
        !committed_now ==> (
            final(k).active_cap_mask == old(k).active_cap_mask
            && final(k).committed_staged_count == old(k).committed_staged_count
            && final(k).staged@ == old(k).staged@
        ),
{
    if slot >= k.staged.len() {
        return false;
    }
    let rec = k.staged[slot];
    if rec.committed {
        return false;
    }
    if provided_witness_hash != rec.required_witness_hash || rec.required_epoch < k.sink_epoch_floor
    {
        return false;
    }
    if !verify_relational_mutex_exec(k.active_cap_mask, rec.cap_mask, &k.toxic_pairs) {
        return false;
    }
    let old_mask = k.active_cap_mask;
    let new_mask = old_mask | rec.cap_mask;
    let updated = ExecStagedEffect {
        cap_mask: rec.cap_mask,
        required_witness_hash: rec.required_witness_hash,
        required_epoch: rec.required_epoch,
        committed: true,
        witnessed: true,
    };
    k.staged.set(slot, updated);
    k.active_cap_mask = new_mask;
    k.committed_staged_count = k.committed_staged_count + 1;
    proof {
        assert forall|j: int|
            0 <= j < k.staged@.len() && (#[trigger] k.staged@[j]).committed implies (
                k.staged@[j].witnessed
                && bitmask_subset_spec(k.staged@[j].cap_mask, k.active_cap_mask)
            ) by {
            if j == slot as int {
                lemma_bitmask_union_monotone(old_mask, rec.cap_mask, rec.cap_mask);
            } else {
                assert(k.staged@[j] == old(k).staged@[j]);
                lemma_bitmask_union_monotone(
                    old_mask,
                    rec.cap_mask,
                    old(k).staged@[j].cap_mask,
                );
            }
        }
    }
    true
}

pub open spec fn run_relational_trace_spec(
    init_mask: u64,
    deltas: Seq<u64>,
    pairs: Seq<ExecToxicPair>,
) -> u64
    decreases deltas.len(),
{
    if deltas.len() == 0 {
        init_mask
    } else {
        let prev = run_relational_trace_spec(init_mask, deltas.drop_last(), pairs);
        let d = deltas.last();
        if relational_mutex_safe_spec(prev | d, pairs) {
            prev | d
        } else {
            prev
        }
    }
}

/// Mechanized compositional relational safety theorem: across any arbitrary interleaved
/// sequence of individual capability deltas `deltas` (even when each delta individually
/// satisfies integer budget and static type rules), gated cumulative composition starting
/// from `init_mask` satisfying `relational_mutex_safe_spec` never activates both halves
/// of any forbidden capability pair `pairs[pair_idx]`.
pub proof fn relational_mutex_prevents_toxic_composition(
    init_mask: u64,
    deltas: Seq<u64>,
    pairs: Seq<ExecToxicPair>,
    pair_idx: int,
)
    requires
        relational_mutex_safe_spec(init_mask, pairs),
        0 <= pair_idx < pairs.len(),
    ensures
        relational_mutex_safe_spec(run_relational_trace_spec(init_mask, deltas, pairs), pairs),
        !pair_conflict_spec(
            run_relational_trace_spec(init_mask, deltas, pairs),
            pairs[pair_idx],
        ),
    decreases deltas.len(),
{
    if deltas.len() > 0 {
        relational_mutex_prevents_toxic_composition(
            init_mask,
            deltas.drop_last(),
            pairs,
            pair_idx,
        );
    }
}

/// Mechanized two-phase staged post-condition theorem: no staged transition can become
/// newly committed if its post-condition witness mismatches, its issuer epoch is fenced,
/// or its capability mask violates any relational mutex with already-committed effects.
pub proof fn staged_commit_requires_verified_postcondition(
    k_pre: ExecRelationalSinkKernel,
    k_post: ExecRelationalSinkKernel,
    slot: int,
    provided_witness_hash: u64,
    committed_now: bool,
)
    requires
        staged_inv_spec(k_pre),
        0 <= slot < k_pre.staged@.len(),
        !k_pre.staged@[slot].committed,
        committed_now ==> (
            provided_witness_hash == k_pre.staged@[slot].required_witness_hash
            && k_pre.staged@[slot].required_epoch >= k_pre.sink_epoch_floor
            && relational_mutex_safe_spec(
                k_pre.active_cap_mask | k_pre.staged@[slot].cap_mask,
                k_pre.toxic_pairs@,
            )
        ),
        !committed_now ==> (k_post.staged@ == k_pre.staged@),
        provided_witness_hash != k_pre.staged@[slot].required_witness_hash
            || k_pre.staged@[slot].required_epoch < k_pre.sink_epoch_floor
            || !relational_mutex_safe_spec(
                k_pre.active_cap_mask | k_pre.staged@[slot].cap_mask,
                k_pre.toxic_pairs@,
            ),
    ensures
        !committed_now,
        !k_post.staged@[slot].committed,
{
}

// ============================================================================
// V6.2 Pillar E: Asymmetric Propose-Verify Safe-Bootstrap Ratchet Kernel
// ============================================================================

#[derive(Debug, PartialEq, Eq)]
pub struct ExecCutCertificate {
    pub severed_routes: Vec<bool>,
    pub proposed_op_mask: u64,
    pub proposed_budget_cap: u64,
    pub proposed_lease_horizon: u64,
    pub polynomial_verifier_passed: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ExecRatchetKernel {
    pub open_routes: Vec<bool>,
    pub open_route_count: u64,
    pub allowed_op_mask: u64,
    pub budget_cap: u64,
    pub lease_horizon: u64,
    pub ratchet_epoch: u64,
}

pub open spec fn count_open_routes_spec(routes: Seq<bool>) -> nat
    decreases routes.len(),
{
    if routes.len() == 0 {
        0
    } else {
        count_open_routes_spec(routes.drop_last()) + if routes.last() { 1nat } else { 0nat }
    }
}

pub proof fn lemma_count_open_routes_upper(routes: Seq<bool>)
    ensures
        count_open_routes_spec(routes) <= routes.len(),
    decreases routes.len(),
{
    if routes.len() > 0 {
        lemma_count_open_routes_upper(routes.drop_last());
    }
}

pub proof fn lemma_count_open_routes_push(prefix: Seq<bool>, b: bool)
    ensures
        count_open_routes_spec(prefix.push(b))
            == count_open_routes_spec(prefix) + if b { 1nat } else { 0nat },
{
    assert(prefix.push(b).drop_last() =~= prefix);
    assert(prefix.push(b).last() == b);
}

#[verifier::bit_vector]
proof fn lemma_bitmask_subset_transitive(a: u64, b: u64, c: u64)
    ensures
        (((a & !b) == 0u64) && ((b & !c) == 0u64)) ==> ((a & !c) == 0u64),
        (a & !a) == 0u64,
{
}

pub open spec fn ratchet_inv_spec(k: ExecRatchetKernel) -> bool {
    k.open_routes@.len() <= u64::MAX as nat
        && k.open_route_count as nat == count_open_routes_spec(k.open_routes@)
}

/// Applies a defensive cut-and-tightening certificate proposed by an untrusted optimizer
/// if and only if an exogenous polynomial verifier validated the cut certificate, the
/// proposed scope/budget/lease strictly attenuate or preserve existing authority, and at
/// least one previously open unmediated route is severed.
pub fn apply_verified_cut_ratchet_exec(
    k: &mut ExecRatchetKernel,
    cert: &ExecCutCertificate,
) -> (accepted: bool)
    requires
        ratchet_inv_spec(*old(k)),
        old(k).ratchet_epoch < u64::MAX,
    ensures
        ratchet_inv_spec(*final(k)),
        final(k).open_routes@.len() == old(k).open_routes@.len(),
        bitmask_subset_spec(final(k).allowed_op_mask, old(k).allowed_op_mask),
        final(k).budget_cap <= old(k).budget_cap,
        final(k).lease_horizon <= old(k).lease_horizon,
        forall|r: int|
            0 <= r < final(k).open_routes@.len() ==> (
                (#[trigger] final(k).open_routes@[r]) ==> old(k).open_routes@[r]
            ),
        accepted ==> (
            cert.polynomial_verifier_passed
            && cert.severed_routes@.len() == old(k).open_routes@.len()
            && final(k).open_route_count < old(k).open_route_count
            && final(k).ratchet_epoch == old(k).ratchet_epoch + 1
        ),
        !accepted ==> (
            final(k).open_routes@ == old(k).open_routes@
            && final(k).open_route_count == old(k).open_route_count
            && final(k).allowed_op_mask == old(k).allowed_op_mask
            && final(k).budget_cap == old(k).budget_cap
            && final(k).lease_horizon == old(k).lease_horizon
            && final(k).ratchet_epoch == old(k).ratchet_epoch
        ),
{
    let _old_mask = k.allowed_op_mask;
    proof {
        lemma_bitmask_subset_transitive(_old_mask, _old_mask, _old_mask);
    }
    if !cert.polynomial_verifier_passed {
        return false;
    }
    if cert.severed_routes.len() != k.open_routes.len() {
        return false;
    }
    if (cert.proposed_op_mask & !k.allowed_op_mask) != 0u64 {
        return false;
    }
    if cert.proposed_budget_cap > k.budget_cap || cert.proposed_lease_horizon > k.lease_horizon {
        return false;
    }

    let n = k.open_routes.len();
    let mut next_routes: Vec<bool> = Vec::with_capacity(n);
    let mut rem_count: u64 = 0;
    let mut newly_severed: u64 = 0;
    let mut i: usize = 0;
    while i < n
        invariant
            n == k.open_routes@.len(),
            n == cert.severed_routes@.len(),
            n <= u64::MAX as nat,
            i <= n,
            next_routes@.len() == i as nat,
            rem_count as nat == count_open_routes_spec(next_routes@),
            (rem_count as nat) + (newly_severed as nat)
                == count_open_routes_spec(k.open_routes@.subrange(0, i as int)),
            forall|r: int|
                0 <= r < i as int ==> (
                    (#[trigger] next_routes@[r])
                        == (k.open_routes@[r] && !cert.severed_routes@[r])
                ),
        decreases n - i,
    {
        let was_open = k.open_routes[i];
        let cut_here = cert.severed_routes[i];
        let still_open = was_open && !cut_here;
        proof {
            assert(k.open_routes@.subrange(0, (i + 1) as int)
                =~= k.open_routes@.subrange(0, i as int).push(was_open));
            lemma_count_open_routes_push(k.open_routes@.subrange(0, i as int), was_open);
            lemma_count_open_routes_upper(k.open_routes@.subrange(0, (i + 1) as int));
            lemma_count_open_routes_push(next_routes@, still_open);
        }
        next_routes.push(still_open);
        if still_open {
            rem_count = rem_count + 1;
        } else if was_open {
            newly_severed = newly_severed + 1;
        }
        i += 1;
    }

    proof {
        assert(k.open_routes@.subrange(0, n as int) =~= k.open_routes@);
    }

    if newly_severed == 0 {
        return false;
    }

    k.open_routes = next_routes;
    k.open_route_count = rem_count;
    k.allowed_op_mask = cert.proposed_op_mask;
    k.budget_cap = cert.proposed_budget_cap;
    k.lease_horizon = cert.proposed_lease_horizon;
    k.ratchet_epoch = k.ratchet_epoch + 1;
    true
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SpecRatchetProposal {
    pub proposed_op_mask: u64,
    pub proposed_budget_cap: u64,
    pub proposed_lease_horizon: u64,
    pub verified: bool,
}

pub open spec fn run_ratchet_authority_spec(
    init_mask: u64,
    init_budget: u64,
    init_lease: u64,
    proposals: Seq<SpecRatchetProposal>,
) -> (u64, u64, u64)
    decreases proposals.len(),
{
    if proposals.len() == 0 {
        (init_mask, init_budget, init_lease)
    } else {
        let (m_prev, b_prev, l_prev) = run_ratchet_authority_spec(
            init_mask,
            init_budget,
            init_lease,
            proposals.drop_last(),
        );
        let p = proposals.last();
        if p.verified
            && bitmask_subset_spec(p.proposed_op_mask, m_prev)
            && p.proposed_budget_cap <= b_prev
            && p.proposed_lease_horizon <= l_prev
        {
            (p.proposed_op_mask, p.proposed_budget_cap, p.proposed_lease_horizon)
        } else {
            (m_prev, b_prev, l_prev)
        }
    }
}

/// Safe-bootstrap non-amplification theorem: across any arbitrary horizon of untrusted
/// optimizer proposals `proposals`, the verified ratchet never widens capability bits
/// beyond `init_mask`, never raises cumulative spend cap `init_budget`, and never extends
/// lease horizon `init_lease`.
pub proof fn ratchet_never_widens_authority(
    init_mask: u64,
    init_budget: u64,
    init_lease: u64,
    proposals: Seq<SpecRatchetProposal>,
)
    ensures
        ({
            let (m_fin, b_fin, l_fin) =
                run_ratchet_authority_spec(init_mask, init_budget, init_lease, proposals);
            bitmask_subset_spec(m_fin, init_mask)
                && b_fin <= init_budget
                && l_fin <= init_lease
        }),
    decreases proposals.len(),
{
    if proposals.len() == 0 {
        lemma_bitmask_subset_transitive(init_mask, init_mask, init_mask);
    } else {
        ratchet_never_widens_authority(init_mask, init_budget, init_lease, proposals.drop_last());
        let (m_prev, _b_prev, _l_prev) =
            run_ratchet_authority_spec(init_mask, init_budget, init_lease, proposals.drop_last());
        let p = proposals.last();
        if p.verified && bitmask_subset_spec(p.proposed_op_mask, m_prev) {
            lemma_bitmask_subset_transitive(p.proposed_op_mask, m_prev, init_mask);
        }
    }
}

pub open spec fn run_ratchet_exposure_spec(u0: nat, accepted_cuts: Seq<nat>) -> nat
    decreases accepted_cuts.len(),
{
    if accepted_cuts.len() == 0 {
        u0
    } else {
        let prev = run_ratchet_exposure_spec(u0, accepted_cuts.drop_last());
        let cut = accepted_cuts.last();
        if cut >= 1 && cut <= prev {
            (prev - cut) as nat
        } else {
            prev
        }
    }
}

/// Monotone safe-bootstrap contraction theorem: every accepted non-trivial verified cut
/// (`1 <= accepted_cuts[t] <= u_t`) reduces residual unmediated route count by at least 1,
/// bounding residual exposure by `u0 - T` after `T` accepted cuts and forcing exact zero
/// residual exposure in at most `u0` accepted steps.
pub proof fn ratchet_monotone_exposure_contraction(u0: nat, accepted_cuts: Seq<nat>)
    requires
        forall|t: int|
            0 <= t < accepted_cuts.len() ==> (
                1 <= #[trigger] accepted_cuts[t]
                && accepted_cuts[t] <= run_ratchet_exposure_spec(u0, accepted_cuts.subrange(0, t))
            ),
    ensures
        accepted_cuts.len() <= u0,
        run_ratchet_exposure_spec(u0, accepted_cuts) + accepted_cuts.len() <= u0,
    decreases accepted_cuts.len(),
{
    if accepted_cuts.len() > 0 {
        let prefix = accepted_cuts.drop_last();
        assert forall|t: int|
            0 <= t < prefix.len() implies (
                1 <= #[trigger] prefix[t]
                && prefix[t] <= run_ratchet_exposure_spec(u0, prefix.subrange(0, t))
            ) by {
            assert(prefix[t] == accepted_cuts[t]);
            assert(prefix.subrange(0, t) =~= accepted_cuts.subrange(0, t));
        }
        ratchet_monotone_exposure_contraction(u0, prefix);
        let last_idx = (accepted_cuts.len() - 1) as int;
        assert(accepted_cuts.subrange(0, last_idx) =~= prefix);
        assert(1 <= accepted_cuts[last_idx]);
        assert(accepted_cuts[last_idx] <= run_ratchet_exposure_spec(u0, prefix));
    }
}

} // verus!

impl Clone for ExecPermitWire {
    fn clone(&self) -> Self {
        clone_permit_wire(self)
    }
}

impl Clone for ExecBrokerRecord {
    fn clone(&self) -> Self {
        Self {
            permit: clone_permit_wire(&self.permit),
            status: self.status,
            inflight_pos: self.inflight_pos,
        }
    }
}

impl Clone for ExecBrokerKernel {
    fn clone(&self) -> Self {
        let mut records = Vec::with_capacity(self.records.len());
        for r in &self.records {
            records.push(r.clone());
        }
        Self {
            budget: self.budget,
            charged: self.charged,
            committed: self.committed,
            pending: self.pending,
            live: clone_vec_bool(&self.live),
            epochs: clone_vec_u64(&self.epochs),
            records,
            inflight_ridxs: clone_vec_usize(&self.inflight_ridxs),
            inflight_amounts: clone_vec_u64(&self.inflight_amounts),
        }
    }
}

impl Clone for ExecSinkRecord {
    fn clone(&self) -> Self {
        Self {
            permit: clone_permit_wire(&self.permit),
            status: self.status,
        }
    }
}

impl Clone for ExecSinkKernel {
    fn clone(&self) -> Self {
        let mut records = Vec::with_capacity(self.records.len());
        for r in &self.records {
            records.push(r.clone());
        }
        Self {
            floors: clone_vec_u64(&self.floors),
            revision: self.revision,
            committed_total: self.committed_total,
            records,
        }
    }
}

impl Clone for ExecAuthorityKernel {
    fn clone(&self) -> Self {
        Self {
            broker: self.broker.clone(),
            sink: self.sink.clone(),
        }
    }
}

impl Clone for ExecScopeGrant {
    fn clone(&self) -> Self {
        Self {
            budget: self.budget,
            expires: self.expires,
            op_mask: self.op_mask,
            required: clone_vec_bool(&self.required),
            issued_epochs: clone_vec_u64(&self.issued_epochs),
        }
    }
}

impl Clone for ExecShardSink {
    fn clone(&self) -> Self {
        Self {
            sink: self.sink.clone(),
            lease_expiry: self.lease_expiry,
        }
    }
}

impl Clone for ExecClusterFenceKernel {
    fn clone(&self) -> Self {
        let mut shards = Vec::with_capacity(self.shards.len());
        for sh in &self.shards {
            shards.push(sh.clone());
        }
        Self {
            shards,
            num_issuers: self.num_issuers,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_permit(
        id: &[u8],
        amount: u64,
        epochs: Vec<u64>,
        required: Vec<bool>,
    ) -> ExecPermitWire {
        ExecPermitWire {
            id: id.to_vec(),
            actor: b"actor-1".to_vec(),
            effect_kind: EFFECT_TRANSFER,
            amount,
            expires: 100,
            expected_rev: 0,
            target: b"approved".to_vec(),
            contents: Vec::new(),
            epochs,
            required,
        }
    }

    #[test]
    fn canonical_wire_framing_prevents_boundary_shift_collision() {
        let mut p1 = sample_permit(b"ab", 5, vec![0, 0], vec![true, false]);
        p1.actor = b"c".to_vec();
        let mut p2 = sample_permit(b"a", 5, vec![0, 0], vec![true, false]);
        p2.actor = b"bc".to_vec();

        assert!(!same_permit_canonical(&p1, &p2));
        let e1 = encode_permit_canonical(&p1);
        let e2 = encode_permit_canonical(&p2);
        assert_ne!(e1, e2);
        assert_eq!(&e1[0..8], &WIRE_TAG_PERMIT_V1.to_le_bytes());
        assert_eq!(e1, encode_permit_canonical(&p1.clone()));
        assert!(same_permit_canonical(&p1, &p1.clone()));
    }

    #[test]
    fn canonical_wire_distinguishes_edit_contents_and_revision() {
        let mut e_a = sample_permit(b"edit-1", 1, vec![2, 3], vec![true, true]);
        e_a.effect_kind = EFFECT_EDIT;
        e_a.expected_rev = 4;
        e_a.target = b"toy.rs".to_vec();
        e_a.contents = b"fn answer() -> u32 { 42 }".to_vec();

        let mut e_b = e_a.clone();
        e_b.contents = b"fn answer() -> u32 { 99 }".to_vec();
        assert!(!same_permit_canonical(&e_a, &e_b));
        assert_ne!(encode_permit_canonical(&e_a), encode_permit_canonical(&e_b));

        let mut e_c = e_a.clone();
        e_c.expected_rev = 5;
        assert!(!same_permit_canonical(&e_a, &e_c));
        assert_ne!(encode_permit_canonical(&e_a), encode_permit_canonical(&e_c));
    }

    #[test]
    fn exec_authority_kernel_conservation_retry_and_settlement() {
        let mut k = ExecAuthorityKernel::exec_new(10, 2);
        let p1 = sample_permit(b"p1", 6, vec![0, 0], vec![true, false]);
        let p2 = sample_permit(b"p2", 6, vec![0, 0], vec![true, false]);
        let p3 = sample_permit(b"p3", 4, vec![0, 0], vec![true, false]);
        let grants = vec![true, true];

        assert_eq!(
            k.exec_reserve(&p1, &p1, &grants, 10),
            ExecReserveOutcome::ReservedNew { slot: 0 }
        );
        assert_eq!(k.broker.charged, 6);
        assert_eq!(k.broker.pending, 6);

        // Duplicate retry / lost-reply idempotence does not double-charge.
        assert_eq!(
            k.exec_reserve(&p1, &p1, &grants, 10),
            ExecReserveOutcome::DuplicateExisting {
                status: ExecSlotStatus::Pending
            }
        );
        assert_eq!(k.broker.charged, 6);

        // Altered payload under same ID is rejected.
        let mut p1_bad = p1.clone();
        p1_bad.amount = 1;
        assert_eq!(
            k.exec_reserve(&p1_bad, &p1_bad, &grants, 10),
            ExecReserveOutcome::PayloadMismatch
        );

        // Over-budget reservation is denied; exact remaining capacity succeeds.
        assert_eq!(
            k.exec_reserve(&p2, &p2, &grants, 10),
            ExecReserveOutcome::DeniedPolicy
        );
        assert_eq!(
            k.exec_reserve(&p3, &p3, &grants, 10),
            ExecReserveOutcome::ReservedNew { slot: 1 }
        );
        assert_eq!(k.broker.charged, 10);

        // Uncertain timeout cannot refund pending charge; settlement without sink authority is rejected.
        k.exec_uncertain_timeout();
        assert_eq!(k.broker.charged, 10);
        assert_eq!(
            k.exec_settle_cancelled(&p1.id),
            ExecSettleOutcome::SinkNotAuthoritative
        );

        // Sink accepts and completes p1 -> broker settles p1 as Committed.
        assert_eq!(k.exec_sink_accept(&p1), ExecAcceptOutcome::AcceptedPending);
        assert_eq!(
            k.exec_sink_complete(&p1.id),
            ExecCompleteOutcome::NewlyCommitted { amount: 6 }
        );
        // Idempotent replay at sink does not double-commit.
        assert_eq!(
            k.exec_sink_complete(&p1.id),
            ExecCompleteOutcome::Idempotent {
                status: ExecSlotStatus::Committed
            }
        );
        assert_eq!(k.sink.committed_total, 6);
        assert_eq!(
            k.exec_settle_committed(&p1.id),
            ExecSettleOutcome::Settled { slot: 0, amount: 6 }
        );
        assert_eq!(k.broker.committed, 6);
        assert_eq!(k.broker.pending, 4);
        assert_eq!(k.broker.charged, 10);

        // Fence issuer 0 before p3 arrives -> sink tombstones p3 -> broker releases p3's 4 units!
        assert!(k.exec_withdraw(0));
        assert!(k.exec_sink_fence(0, 1));
        assert_eq!(
            k.exec_sink_accept(&p3),
            ExecAcceptOutcome::CancelledTombstone
        );
        assert_eq!(
            k.exec_settle_cancelled(&p3.id),
            ExecSettleOutcome::Settled { slot: 0, amount: 4 }
        );
        assert_eq!(k.broker.committed, 6);
        assert_eq!(k.broker.pending, 0);
        assert_eq!(k.broker.charged, 6);
        assert_eq!(k.sink.committed_total, 6);
    }

    #[test]
    fn exec_post_fence_exclusion_and_edit_revision_cas() {
        let mut k = ExecAuthorityKernel::exec_new(10, 2);
        let grants = vec![true, true];

        let mut edit0 = sample_permit(b"edit-0", 1, vec![0, 0], vec![true, true]);
        edit0.effect_kind = EFFECT_EDIT;
        edit0.expected_rev = 0;

        let mut edit_stale = sample_permit(b"edit-stale", 1, vec![0, 0], vec![true, true]);
        edit_stale.effect_kind = EFFECT_EDIT;
        edit_stale.expected_rev = 0;

        let p_fenced = sample_permit(b"in-flight-fenced", 3, vec![0, 0], vec![false, true]);

        assert!(matches!(
            k.exec_reserve(&edit0, &edit0, &grants, 1),
            ExecReserveOutcome::ReservedNew { .. }
        ));
        assert!(matches!(
            k.exec_reserve(&edit_stale, &edit_stale, &grants, 1),
            ExecReserveOutcome::ReservedNew { .. }
        ));
        assert!(matches!(
            k.exec_reserve(&p_fenced, &p_fenced, &grants, 1),
            ExecReserveOutcome::ReservedNew { .. }
        ));

        assert_eq!(
            k.exec_sink_accept(&edit0),
            ExecAcceptOutcome::AcceptedPending
        );
        assert_eq!(
            k.exec_sink_accept(&edit_stale),
            ExecAcceptOutcome::AcceptedPending
        );
        assert_eq!(
            k.exec_sink_accept(&p_fenced),
            ExecAcceptOutcome::AcceptedPending
        );

        // First edit commits and bumps sink revision 0 -> 1; second edit expecting rev 0 cancels!
        assert_eq!(
            k.exec_sink_complete(&edit0.id),
            ExecCompleteOutcome::NewlyCommitted { amount: 1 }
        );
        assert_eq!(k.sink.revision, 1);
        assert_eq!(
            k.exec_sink_complete(&edit_stale.id),
            ExecCompleteOutcome::Cancelled
        );
        assert_eq!(k.sink.revision, 1);

        // Fence issuer 1 to floor 1 while p_fenced is already AcceptedPending at sink:
        assert!(k.exec_sink_fence(1, 1));
        assert_eq!(
            k.exec_sink_complete(&p_fenced.id),
            ExecCompleteOutcome::Idempotent {
                status: ExecSlotStatus::Cancelled
            }
        );
        assert_eq!(k.sink.committed_total, 1);

        // Withdraw/renew issuers 0 and 1 and confirm post-epoch permit succeeds while stale epoch fails.
        assert!(k.exec_withdraw(1));
        assert!(k.exec_renew(1));
        assert!(k.exec_withdraw(0));
        assert!(k.exec_sink_fence(0, 1));
        assert!(k.exec_renew(0));
        let p_stale = sample_permit(b"stale-after-renew", 2, vec![0, 1], vec![true, true]);
        assert_eq!(
            k.exec_reserve(&p_stale, &p_stale, &grants, 1),
            ExecReserveOutcome::DeniedPolicy
        );
        let p_fresh = sample_permit(b"fresh-after-renew", 2, vec![1, 1], vec![true, true]);
        assert!(matches!(
            k.exec_reserve(&p_fresh, &p_fresh, &grants, 1),
            ExecReserveOutcome::ReservedNew { .. }
        ));
        assert_eq!(
            k.exec_sink_accept(&p_fresh),
            ExecAcceptOutcome::AcceptedPending
        );
        assert_eq!(
            k.exec_sink_complete(&p_fresh.id),
            ExecCompleteOutcome::NewlyCommitted { amount: 2 }
        );
        assert_eq!(k.sink.committed_total, 3);
    }

    #[test]
    fn canonical_wire_all_fields_and_length_invariants() {
        let base = ExecPermitWire {
            id: b"permit-42".to_vec(),
            actor: b"worker-a".to_vec(),
            effect_kind: EFFECT_RELEASE,
            amount: 7,
            expires: 500,
            expected_rev: 3,
            target: b"v1.2.3".to_vec(),
            contents: b"payload".to_vec(),
            epochs: vec![10, 20, 30],
            required: vec![true, false, true],
        };
        let encoded = encode_permit_canonical(&base);
        // 8-byte domain tag + 10 field tags (80 bytes) + framed lengths and payloads
        let expected_len = 8
            + (8 + 8 + base.id.len())
            + (8 + 8 + base.actor.len())
            + (8 + 1)
            + (8 + 8)
            + (8 + 8)
            + (8 + 8)
            + (8 + 8 + base.target.len())
            + (8 + 8 + base.contents.len())
            + (8 + 8 + 8 * base.epochs.len())
            + (8 + 8 + base.required.len());
        assert_eq!(encoded.len(), expected_len);

        // Cross-vector shift between target/contents and epochs/required cannot collide.
        let mut shift_tc = base.clone();
        shift_tc.target = b"v1.2.3pay".to_vec();
        shift_tc.contents = b"load".to_vec();
        assert_ne!(encoded, encode_permit_canonical(&shift_tc));
        assert!(!same_permit_canonical(&base, &shift_tc));

        let mut shift_er = base.clone();
        shift_er.epochs = vec![10, 20];
        shift_er.required = vec![true, false, true, false];
        assert_ne!(encoded, encode_permit_canonical(&shift_er));
        assert!(!same_permit_canonical(&base, &shift_er));

        // Every scalar field perturbs the canonical encoding.
        let mut m_kind = base.clone();
        m_kind.effect_kind = EFFECT_TRANSFER;
        assert_ne!(encoded, encode_permit_canonical(&m_kind));

        let mut m_amt = base.clone();
        m_amt.amount = 8;
        assert_ne!(encoded, encode_permit_canonical(&m_amt));

        let mut m_exp = base.clone();
        m_exp.expires = 501;
        assert_ne!(encoded, encode_permit_canonical(&m_exp));

        let mut m_ep = base.clone();
        m_ep.epochs[1] = 21;
        assert_ne!(encoded, encode_permit_canonical(&m_ep));

        let mut m_req = base.clone();
        m_req.required[1] = true;
        assert_ne!(encoded, encode_permit_canonical(&m_req));
    }

    #[test]
    fn standalone_broker_and_sink_out_of_order_inflight_compaction() {
        let mut broker = ExecBrokerKernel::broker_new(20, 2);
        let mut sink = ExecSinkKernel::sink_new(2);
        let grants = vec![true, true];

        let p0 = sample_permit(b"s0", 3, vec![0, 0], vec![true, false]);
        let p1 = sample_permit(b"s1", 4, vec![0, 0], vec![true, false]);
        let p2 = sample_permit(b"s2", 5, vec![0, 0], vec![false, true]);
        let p3 = sample_permit(b"s3", 6, vec![0, 0], vec![true, true]);

        assert_eq!(
            broker.broker_reserve(&p0, &p0, &grants, 0),
            ExecReserveOutcome::ReservedNew { slot: 0 }
        );
        assert_eq!(
            broker.broker_reserve(&p1, &p1, &grants, 0),
            ExecReserveOutcome::ReservedNew { slot: 1 }
        );
        assert_eq!(
            broker.broker_reserve(&p2, &p2, &grants, 0),
            ExecReserveOutcome::ReservedNew { slot: 2 }
        );
        assert_eq!(
            broker.broker_reserve(&p3, &p3, &grants, 0),
            ExecReserveOutcome::ReservedNew { slot: 3 }
        );
        assert_eq!(broker.charged, 18);
        assert_eq!(broker.pending, 18);

        for p in [&p0, &p1, &p2, &p3] {
            assert_eq!(sink.sink_accept(p), ExecAcceptOutcome::AcceptedPending);
        }

        // Complete s1 (middle slot 1), fence issuer 1 so s2 & s3 cancel, complete s0 last!
        assert_eq!(
            sink.sink_complete(&p1.id),
            ExecCompleteOutcome::NewlyCommitted { amount: 4 }
        );
        assert!(sink.sink_fence(1, 1));
        assert_eq!(
            sink.sink_complete(&p0.id),
            ExecCompleteOutcome::NewlyCommitted { amount: 3 }
        );

        // Settle middle slot s1 first -> shifts s2, s3 left in broker inflight vector!
        assert_eq!(
            broker.broker_settle_committed(&p1.id, sink.sink_is_committed(&p1.id)),
            ExecSettleOutcome::Settled { slot: 1, amount: 4 }
        );
        // Settle shifted s2 as cancelled (now at slot 1) -> shifts s3 left to slot 1!
        assert_eq!(
            broker.broker_settle_cancelled(&p2.id, sink.sink_is_cancelled(&p2.id)),
            ExecSettleOutcome::Settled { slot: 1, amount: 5 }
        );
        // Settle s0 at slot 0 -> shifts s3 left to slot 0!
        assert_eq!(
            broker.broker_settle_committed(&p0.id, sink.sink_is_committed(&p0.id)),
            ExecSettleOutcome::Settled { slot: 0, amount: 3 }
        );
        // Settle s3 at slot 0 as cancelled!
        assert_eq!(
            broker.broker_settle_cancelled(&p3.id, sink.sink_is_cancelled(&p3.id)),
            ExecSettleOutcome::Settled { slot: 0, amount: 6 }
        );

        assert_eq!(broker.committed, 7);
        assert_eq!(broker.pending, 0);
        assert_eq!(broker.charged, 7);
        assert_eq!(sink.committed_total, 7);
    }

    #[test]
    fn canonical_wire_decoder_bijection_and_adversarial_frame_rejection() {
        let permit = ExecPermitWire {
            id: b"v5-permit-bijection".to_vec(),
            actor: b"agent-alpha".to_vec(),
            effect_kind: EFFECT_EDIT,
            amount: 42,
            expires: 9_999,
            expected_rev: 17,
            target: b"src/critical.rs".to_vec(),
            contents: b"pub const SAFE: bool = true;".to_vec(),
            epochs: vec![3, 7, 11],
            required: vec![true, false, true],
        };
        let wire = encode_permit_canonical(&permit);
        let decoded = decode_permit_canonical(&wire).expect("valid canonical frame must decode");
        assert_eq!(decoded, permit);
        assert_eq!(encode_permit_canonical(&decoded), wire);

        // Trailing garbage byte must be rejected (strict bijection, no prefix-only parse).
        let mut with_trailing = wire.clone();
        with_trailing.push(0x00);
        assert_eq!(decode_permit_canonical(&with_trailing), None);

        // Every strict prefix truncation must be rejected without panic.
        for cut in [0usize, 1, 7, 8, 15, 24, wire.len() - 1] {
            assert_eq!(decode_permit_canonical(&wire[..cut].to_vec()), None);
        }

        // Corrupted domain header tag must be rejected.
        let mut bad_hdr = wire.clone();
        bad_hdr[0] ^= 0xFF;
        assert_eq!(decode_permit_canonical(&bad_hdr), None);

        // Non-canonical boolean byte (2u8 instead of 0u8/1u8) in `required` must be rejected.
        let mut bad_bool = wire.clone();
        let last_idx = bad_bool.len() - 1;
        bad_bool[last_idx] = 2u8;
        assert_eq!(decode_permit_canonical(&bad_bool), None);

        // Mismatched `epochs.len() != required.len()` must be rejected.
        let mut mismatched = permit.clone();
        mismatched.required = vec![true, false];
        let mismatched_wire = encode_permit_canonical(&mismatched);
        assert_eq!(decode_permit_canonical(&mismatched_wire), None);
    }

    #[test]
    fn delegation_chain_non_amplification_severance_and_self_upgrade_gate() {
        let root = ExecScopeGrant {
            budget: 100,
            expires: 1_000,
            op_mask: 0b0111,
            required: vec![true, false, true],
            issued_epochs: vec![2, 0, 5],
        };
        let hop1 = ExecScopeGrant {
            budget: 60,
            expires: 800,
            op_mask: 0b0011,
            required: vec![true, true, true],
            issued_epochs: vec![2, 4, 5],
        };
        let hop2 = ExecScopeGrant {
            budget: 25,
            expires: 600,
            op_mask: 0b0001,
            required: vec![true, true, true],
            issued_epochs: vec![2, 4, 5],
        };
        let chain = vec![hop1.clone(), hop2.clone()];
        let live = vec![true, true, true];
        let epochs = vec![2, 4, 5];

        assert!(verify_delegation_chain_exec(
            &root, &chain, &live, &epochs, 500
        ));

        // Capability bitmask widening at leaf must be rejected.
        let mut widened_mask = hop2.clone();
        widened_mask.op_mask = 0b0101;
        assert!(!verify_delegation_chain_exec(
            &root,
            &vec![hop1.clone(), widened_mask],
            &live,
            &epochs,
            500,
        ));

        // Dropping a required ancestor human issuer at leaf must be rejected.
        let mut dropped_issuer = hop2.clone();
        dropped_issuer.required = vec![true, false, true];
        assert!(!verify_delegation_chain_exec(
            &root,
            &vec![hop1.clone(), dropped_issuer],
            &live,
            &epochs,
            500,
        ));

        // Withdrawing mid-chain issuer 1 (added at hop1) severs the descendant leaf!
        let live_revoked_mid = vec![true, false, true];
        assert!(!verify_delegation_chain_exec(
            &root,
            &chain,
            &live_revoked_mid,
            &epochs,
            500,
        ));

        // Bumping root issuer 2 epoch (2 -> 6) severs the entire delegation subtree!
        let epochs_bumped = vec![2, 4, 6];
        assert!(!verify_delegation_chain_exec(
            &root,
            &chain,
            &live,
            &epochs_bumped,
            500,
        ));

        // Self-modification without fresh human quorum cannot widen budget or expiry.
        let mut widened_ctrl = hop1.clone();
        widened_ctrl.budget = 61;
        assert!(!verify_self_upgrade_exec(&hop1, &widened_ctrl, false));
        assert!(verify_self_upgrade_exec(&hop1, &hop2, false));
        assert!(verify_self_upgrade_exec(&hop1, &widened_ctrl, true));
    }

    #[test]
    fn cluster_quorum_and_lease_fence_post_ack_exclusion() {
        let p_s0 = sample_permit(b"shard0-p", 4, vec![0, 0], vec![true, false]);
        let p_s1 = sample_permit(b"shard1-p", 4, vec![0, 0], vec![true, false]);
        let p_s2 = sample_permit(b"shard2-p", 4, vec![0, 0], vec![true, false]);

        let mut s0 = ExecShardSink {
            sink: ExecSinkKernel::sink_new(2),
            lease_expiry: 100,
        };
        let mut s1 = ExecShardSink {
            sink: ExecSinkKernel::sink_new(2),
            lease_expiry: 100,
        };
        let mut s2 = ExecShardSink {
            sink: ExecSinkKernel::sink_new(2),
            lease_expiry: 50,
        };

        assert_eq!(
            s0.sink.sink_accept(&p_s0),
            ExecAcceptOutcome::AcceptedPending
        );
        assert_eq!(
            s1.sink.sink_accept(&p_s1),
            ExecAcceptOutcome::AcceptedPending
        );
        assert_eq!(
            s2.sink.sink_accept(&p_s2),
            ExecAcceptOutcome::AcceptedPending
        );

        // Fence shards 0 and 1 to floor 1 on issuer 0; shard 2 is partitioned (un-fenced, lease=50).
        assert!(s0.sink.sink_fence(0, 1));
        assert!(s1.sink.sink_fence(0, 1));

        let cluster = ExecClusterFenceKernel {
            shards: vec![s0, s1, s2],
            num_issuers: 2,
        };

        // At t = 40 (< shard 2's lease_expiry 50), cluster-wide fence ack is NOT yet complete.
        assert!(!cluster_fence_acked_exec(&cluster, 0, 1, 40));
        // Once t = 50 reaches shard 2's bounded lease expiry, cluster fence ack holds!
        assert!(cluster_fence_acked_exec(&cluster, 0, 1, 50));
        // Under bounded physical clock skew max_skew = 15, broker_now = 50 is NOT yet safe
        // (shard 2's clock could lag at 35 < 50), whereas broker_now = 65 IS skew-quiesced!
        assert!(!cluster_fence_acked_bounded_skew_exec(&cluster, 0, 1, 50, 15));
        assert!(!cluster_fence_acked_bounded_skew_exec(&cluster, 0, 1, 64, 15));
        assert!(cluster_fence_acked_bounded_skew_exec(&cluster, 0, 1, 65, 15));

        // Verify zero post-ack commits across all 3 shards at t = 50:
        let mut shards = cluster.shards;
        assert_eq!(
            cluster_shard_complete_exec(&mut shards[0], &p_s0.id, 50),
            ExecCompleteOutcome::Idempotent {
                status: ExecSlotStatus::Cancelled
            }
        );
        assert_eq!(
            cluster_shard_complete_exec(&mut shards[1], &p_s1.id, 50),
            ExecCompleteOutcome::Idempotent {
                status: ExecSlotStatus::Cancelled
            }
        );
        assert_eq!(
            cluster_shard_complete_exec(&mut shards[2], &p_s2.id, 50),
            ExecCompleteOutcome::Cancelled
        );
        assert_eq!(
            shards[0].sink.committed_total
                + shards[1].sink.committed_total
                + shards[2].sink.committed_total,
            0
        );
    }

    #[test]
    fn compositional_relational_mutex_and_staged_postcondition_invariants() {
        let cap_read_secret = 0b0001u64;
        let cap_net_egress = 0b0010u64;
        let cap_local_summarize = 0b0100u64;
        let pairs = vec![ExecToxicPair {
            mask_a: cap_read_secret,
            mask_b: cap_net_egress,
        }];

        let mut rk = ExecRelationalSinkKernel {
            active_cap_mask: 0,
            sink_epoch_floor: 2,
            toxic_pairs: pairs.clone(),
            staged: vec![
                ExecStagedEffect {
                    cap_mask: cap_read_secret,
                    required_witness_hash: 0xAA01,
                    required_epoch: 2,
                    committed: false,
                    witnessed: false,
                },
                ExecStagedEffect {
                    cap_mask: cap_net_egress,
                    required_witness_hash: 0xBB02,
                    required_epoch: 2,
                    committed: false,
                    witnessed: false,
                },
                ExecStagedEffect {
                    cap_mask: cap_local_summarize,
                    required_witness_hash: 0xCC03,
                    required_epoch: 2,
                    committed: false,
                    witnessed: false,
                },
                ExecStagedEffect {
                    cap_mask: cap_local_summarize,
                    required_witness_hash: 0xDD04,
                    required_epoch: 1, // stale epoch < sink_epoch_floor 2
                    committed: false,
                    witnessed: false,
                },
            ],
            committed_staged_count: 0,
        };

        // Wrong postcondition witness hash must be rejected.
        assert!(!commit_staged_postcondition_exec(&mut rk, 0, 0xDEAD));
        assert_eq!(rk.committed_staged_count, 0);

        // Authentic postcondition witness commits slot 0 (secret read).
        assert!(commit_staged_postcondition_exec(&mut rk, 0, 0xAA01));
        assert_eq!(rk.active_cap_mask, cap_read_secret);
        assert_eq!(rk.committed_staged_count, 1);

        // Duplicate commit on already-committed slot 0 is rejected.
        assert!(!commit_staged_postcondition_exec(&mut rk, 0, 0xAA01));

        // Slot 1 (network egress) carries valid witness + epoch, yet composes into toxic pair (secret_read & net_egress) -> rejected!
        assert!(!verify_relational_mutex_exec(
            rk.active_cap_mask,
            cap_net_egress,
            &pairs
        ));
        assert!(!commit_staged_postcondition_exec(&mut rk, 1, 0xBB02));
        assert_eq!(rk.active_cap_mask, cap_read_secret);
        assert_eq!(rk.committed_staged_count, 1);

        // Slot 3 has compatible capability mask, but stale issuer epoch 1 < floor 2 -> rejected!
        assert!(!commit_staged_postcondition_exec(&mut rk, 3, 0xDD04));

        // Slot 2 (local summarize) is relationally compatible, unfenced, and carries valid witness -> commits!
        assert!(commit_staged_postcondition_exec(&mut rk, 2, 0xCC03));
        assert_eq!(rk.active_cap_mask, cap_read_secret | cap_local_summarize);
        assert_eq!(rk.committed_staged_count, 2);
    }

    #[test]
    fn safe_bootstrap_ratchet_monotone_contraction_and_no_widening() {
        let mut rk = ExecRatchetKernel {
            open_routes: vec![true, true, true, true],
            open_route_count: 4,
            allowed_op_mask: 0b0111,
            budget_cap: 100,
            lease_horizon: 8,
            ratchet_epoch: 0,
        };

        // 1. Unverified cut certificate from untrusted proposer -> rejected!
        let unverified = ExecCutCertificate {
            severed_routes: vec![true, true, false, false],
            proposed_op_mask: 0b0011,
            proposed_budget_cap: 80,
            proposed_lease_horizon: 4,
            polynomial_verifier_passed: false,
        };
        assert!(!apply_verified_cut_ratchet_exec(&mut rk, &unverified));
        assert_eq!(rk.open_route_count, 4);

        // 2. Certificate attempting to widen capability scope (bit 3) -> rejected even if cut verified!
        let widened_mask = ExecCutCertificate {
            severed_routes: vec![true, false, false, false],
            proposed_op_mask: 0b1111,
            proposed_budget_cap: 100,
            proposed_lease_horizon: 8,
            polynomial_verifier_passed: true,
        };
        assert!(!apply_verified_cut_ratchet_exec(&mut rk, &widened_mask));
        assert_eq!(rk.allowed_op_mask, 0b0111);

        // 3. Verified cut severing routes 0 and 2 while tightening scope & lease -> accepted!
        let valid_cut_1 = ExecCutCertificate {
            severed_routes: vec![true, false, true, false],
            proposed_op_mask: 0b0011,
            proposed_budget_cap: 80,
            proposed_lease_horizon: 4,
            polynomial_verifier_passed: true,
        };
        assert!(apply_verified_cut_ratchet_exec(&mut rk, &valid_cut_1));
        assert_eq!(rk.open_routes, vec![false, true, false, true]);
        assert_eq!(rk.open_route_count, 2);
        assert_eq!(rk.allowed_op_mask, 0b0011);
        assert_eq!(rk.budget_cap, 80);
        assert_eq!(rk.lease_horizon, 4);
        assert_eq!(rk.ratchet_epoch, 1);

        // 4. Redundant replay of already-severed routes -> rejected (must strictly contract)!
        assert!(!apply_verified_cut_ratchet_exec(&mut rk, &valid_cut_1));
        assert_eq!(rk.ratchet_epoch, 1);

        // 5. Verified cut severing remaining routes 1 and 3 -> reaches 0 unmediated exposure!
        let valid_cut_2 = ExecCutCertificate {
            severed_routes: vec![false, true, false, true],
            proposed_op_mask: 0b0001,
            proposed_budget_cap: 50,
            proposed_lease_horizon: 2,
            polynomial_verifier_passed: true,
        };
        assert!(apply_verified_cut_ratchet_exec(&mut rk, &valid_cut_2));
        assert_eq!(rk.open_route_count, 0);
        assert_eq!(rk.open_routes, vec![false, false, false, false]);
        assert_eq!(rk.ratchet_epoch, 2);
    }
}
