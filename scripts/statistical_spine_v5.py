#!/usr/bin/env python3
"""Pure-stdlib exact unconditional Suissa-Shuster paired multinomial & Ville martingale cross-validator (V5)."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPONENT_ALPHA = 0.05 / 16.0
FAMILY_ALPHA = 0.05


def ln_fact(n: int) -> float:
    return math.lgamma(n + 1.0)


def binom_tail(n: int, p: float, first: int, last: int) -> float:
    if first > last:
        return 0.0
    if p <= 0.0:
        return 1.0 if first == 0 else 0.0
    if p >= 1.0:
        return 1.0 if last == n else 0.0
    terms = []
    largest = -math.inf
    ln_p = math.log(p)
    ln_q = math.log(1.0 - p)
    ln_n = ln_fact(n)
    for k in range(first, last + 1):
        lp = ln_n - ln_fact(k) - ln_fact(n - k) + k * ln_p + (n - k) * ln_q
        if lp > largest:
            largest = lp
        terms.append(lp)
    total = math.exp(largest) * sum(math.exp(x - largest) for x in terms)
    return min(1.0, total)


def clopper_pearson_one_sided(successes: int, trials: int, alpha: float, upper: bool) -> float:
    if upper and successes == trials:
        return 1.0
    if (not upper) and successes == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if upper:
            if binom_tail(trials, mid, 0, successes) > alpha:
                lo = mid
            else:
                hi = mid
        else:
            if binom_tail(trials, mid, successes, trials) < alpha:
                lo = mid
            else:
                hi = mid
    return hi if upper else lo


def paired_score_stat(x10: int, x01: int, n: int, delta0: float) -> float:
    n_f = float(n)
    diff = float(x10) - float(x01) - n_f * delta0
    if diff == 0.0:
        return 0.0
    theta_min = max(0.0, -delta0)
    theta_max = 0.5 * (1.0 - delta0)
    b = float(x10) * (1.0 + delta0) + float(x01) * (1.0 - delta0) - 2.0 * n_f * delta0
    c = float(x01) * delta0 * (1.0 - delta0)
    disc = math.sqrt(max(0.0, b * b + 8.0 * n_f * c))
    theta_hat = min(theta_max, max(theta_min, (b + disc) / (4.0 * n_f)))
    var_rmle = max(1e-12, n_f * (2.0 * theta_hat + delta0 - delta0 * delta0))
    return diff / math.sqrt(var_rmle)


def exact_unconditional_paired_pvalue(n10: int, n01: int, n: int, delta0: float) -> float:
    if delta0 >= 1.0:
        return 1.0
    if delta0 <= -1.0:
        return 0.0
    t_obs = paired_score_stat(n10, n01, n, delta0) + 1e-12
    ln_n = ln_fact(n)
    crit = []
    for x10 in range(n + 1):
        for x01 in range(n - x10 + 1):
            if paired_score_stat(x10, x01, n, delta0) <= t_obs:
                x_rem = n - x10 - x01
                lcoef = ln_n - ln_fact(x10) - ln_fact(x01) - ln_fact(x_rem)
                crit.append((x10, x01, x_rem, lcoef))
    if not crit:
        return 0.0
    if len(crit) == (n + 1) * (n + 2) // 2:
        return 1.0

    theta_min = max(0.0, -delta0)
    theta_max = 0.5 * (1.0 - delta0)
    if theta_max < theta_min:
        return 0.0

    def eval_prob(theta: float) -> float:
        p10 = min(1.0, max(0.0, theta + delta0))
        p01 = min(1.0, max(0.0, theta))
        prem = min(1.0, max(0.0, 1.0 - p10 - p01))
        lp10 = math.log(p10) if p10 > 0.0 else -math.inf
        lp01 = math.log(p01) if p01 > 0.0 else -math.inf
        lprem = math.log(prem) if prem > 0.0 else -math.inf
        total = 0.0
        for x10, x01, x_rem, lcoef in crit:
            if (x10 > 0 and p10 == 0.0) or (x01 > 0 and p01 == 0.0) or (x_rem > 0 and prem == 0.0):
                continue
            lp = lcoef
            if x10 > 0:
                lp += x10 * lp10
            if x01 > 0:
                lp += x01 * lp01
            if x_rem > 0:
                lp += x_rem * lprem
            total += math.exp(lp)
        return min(1.0, total)

    grid_steps = 400
    best_p = 0.0
    best_idx = 0
    for g in range(grid_steps + 1):
        theta = theta_min + (g / grid_steps) * (theta_max - theta_min)
        val = eval_prob(theta)
        if val > best_p:
            best_p = val
            best_idx = g

    step = (theta_max - theta_min) / grid_steps
    a = max(theta_min, theta_min + max(0, best_idx - 1) * step)
    b = min(theta_max, theta_min + (best_idx + 1) * step)
    gr = 0.5 * (math.sqrt(5.0) - 1.0)
    for _ in range(32):
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = eval_prob(c)
        fd = eval_prob(d)
        best_p = max(best_p, fc, fd)
        if fc < fd:
            a = c
        else:
            b = d
    return min(1.0, best_p)


def exact_unconditional_upper_bound(n10: int, n01: int, n: int, alpha: float) -> float:
    lo = (n10 - n01) / float(n)
    hi = 1.0
    for _ in range(42):
        mid = 0.5 * (lo + hi)
        if exact_unconditional_paired_pvalue(n10, n01, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def conditional_mcnemar_one_sided_p(n10: int, n01: int) -> float:
    d = n10 + n01
    if d == 0:
        return 1.0
    return binom_tail(d, 0.5, 0, n10)


def cross_validate_report(report: dict) -> dict:
    assert report["frozen_v2_preserved_untouched"] is True
    max_p_diff = 0.0
    max_ub_diff = 0.0
    for group_key in ("predeclared_holdout_certificates", "pooled_and_benchmark_certificates"):
        for cert in report[group_key]:
            n = cert["total_pairs"]
            n10 = cert["n10_complete_only_harm"]
            n01 = cert["n01_baseline_only_harm"]
            alpha = cert["alpha"]
            py_mc = conditional_mcnemar_one_sided_p(n10, n01)
            py_ss = exact_unconditional_paired_pvalue(n10, n01, n, 0.0)
            py_ub = exact_unconditional_upper_bound(n10, n01, n, alpha)
            assert abs(py_mc - cert["conditional_mcnemar_p"]) < 1e-12
            p_diff = abs(py_ss - cert["unconditional_suissa_shuster_p"])
            ub_diff = abs(py_ub - cert["unconditional_exact_upper_bound"])
            max_p_diff = max(max_p_diff, p_diff)
            max_ub_diff = max(max_ub_diff, ub_diff)
            assert p_diff < 1e-11, f"p-value mismatch on {cert['comparison_id']}: {p_diff}"
            assert ub_diff < 1e-8, f"upper bound mismatch on {cert['comparison_id']}: {ub_diff}"

    # Verify Ville martingale trajectory step-by-step
    e_val = 1.0
    max_e = 1.0
    wins = 0
    for idx, pt in enumerate(report["ville_discordance_martingale"]):
        p_hat = (wins + 0.75) / (idx + 1.0)
        lam = min(0.90, max(0.0, 2.0 * p_hat - 1.0))
        assert abs(lam - pt["predictable_bet_lambda"]) < 1e-12
        y = 1.0 if pt["step_multiplier"] > 1.0 else -1.0
        mult = 1.0 + lam * y
        e_val *= mult
        max_e = max(max_e, e_val)
        if y > 0:
            wins += 1
        assert abs(e_val - pt["cumulative_e_value"]) / max(1.0, e_val) < 1e-12
        assert abs(min(1.0, 1.0 / max_e) - pt["anytime_valid_p_upper"]) < 1e-12

    return {
        "cross_validated": True,
        "max_p_abs_diff": max_p_diff,
        "max_ub_abs_diff": max_ub_diff,
        "ville_steps": len(report["ville_discordance_martingale"]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "certificate_json",
        nargs="?",
        type=Path,
        default=ROOT / "evidence/v3/reference/statistical_certificates.json",
    )
    args = p.parse_args()
    data = json.loads(args.certificate_json.read_text())
    print(json.dumps(cross_validate_report(data), indent=2))


if __name__ == "__main__":
    main()
