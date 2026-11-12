#!/usr/bin/env python3
"""Text macros and tables for the v6 sections, generated only from promoted v6 evidence.

The hero figure's numbers come from scripts/hero_v6.py (v6-hero-macros.tex). This script writes everything else the
v6 section drafts cite: v6-text-macros.tex, v6-e1-table.tex and v6-e6-table.tex. Every outcome word ("held", "did
not hold", "reproduced") is chosen by a rule from the data; nothing about an outcome is written in advance.

Sources default to evidence/v6/reference/<name>. A missing source is a hard error. With --draft, each macro that
depends on it expands to a visible "[E? pending]" marker instead. --source NAME=PATH points a source at a raw run for
checking; output built that way, or in draft mode, may not be written under docs/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hero_v6 import CONTROLLERS, FAMILY_LABEL, STATE, VISIBLE, load_e4, load_rows, rho  # noqa: E402

EVIDENCE = ROOT / "evidence/v6/reference"
SOURCES = {"e4": "e4-revokebench", "e4b": "e4b-revokebench-faults", "e5": "e5-omitted-clause", "e6": "e6-fenceability", "e8": "e8-agentdojo-fair",
           "e8b": "e8b-inside-spec", "e1": "e1-n128"}
CTRL_LABEL = {k: label for k, label, *_ in CONTROLLERS}
SURFACE_LABEL = {"postgresql16": "PostgreSQL 16", "etcd_v3_5_17": "etcd v3.5.17", "posix_fs": "POSIX filesystem",
                 "aws_s3": "Amazon S3", "kubernetes_api": "Kubernetes API"}
CLASS_LABEL = {"sink_enforced": "sink", "writer_supplied_condition": "writer", "advisory": "advisory"}
OUTCOME_LABEL = {"pass": "pass", "fail": "fail", "not_applicable": "n/a", "documented_not_executed": "doc.\\ only"}


def short(ckpt: str) -> str:
    return ckpt.split("/")[-1].replace("-GGUF", "")


def frac(x: int, n: int) -> str:
    return f"{x}/{n}"


def series(items: list[str]) -> str:
    if not items:
        return "none"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + (" and " if len(items) == 2 else ", and ") + items[-1]


class Missing(Exception):
    pass


def src(args, name: str) -> Path:
    p = args.sources.get(name) or EVIDENCE / SOURCES[name]
    if not p.exists():
        raise Missing(name)
    return p


# ------------------------------------------------------------------ E4

def e4_macros(args) -> dict:
    trials, _ = load_e4(src(args, "e4"))
    nofault = [t["config"] for t in trials if t["config"]["fault"] == "none"]
    wc = {(c["w"], c["C"]) for c in nofault}
    from revokebench_reanalysis_v6 import per_trial
    little = [r for r in map(per_trial, trials) if r["little_law_plan"]]
    ratio = sorted(r["post_decision_commits"] / r["little_law_plan"] for r in little)
    med = (ratio[(len(ratio) - 1) // 2] + ratio[len(ratio) // 2]) / 2
    worst = max(little, key=lambda r: r["post_decision_commits"] / r["little_law_plan"])
    sentence = (rf"Over the {len(little)} halting trials in stable cells, the commits after the decision were {med:.2f} times "
                rf"the plan's prediction \(\hat\lambda(L + w)\) at the median (range {ratio[0]:.2f}--{ratio[-1]:.2f}); the largest "
                rf"ratio is at \(L + w = {worst['L'] + worst['w']:g}\)\,s")
    burst = trials[0]["config"]["workload"]["calls_per_step"]
    sentence += (rf", where RPC time and the emitter's bursts of {burst} calls are not small against the window." if
                 worst["L"] + worst["w"] <= 1 else ".")
    return {"vsixEFourTrials": len(trials),
            "vsixEFourFaultTrials": sum(t["config"]["fault"] != "none" for t in trials),
            "vsixEFourCells": len({(c["w"], c["C"], c["L"]) for c in nofault}),
            "vsixEFourWCCells": len(wc),
            "vsixEFourOverloadedCells": len({(c["w"], c["C"]) for c in nofault if rho(c) >= 1}),
            "vsixLittleSentence": sentence}


def e4b_macros(args) -> dict:
    d = src(args, "e4b")
    rows = [json.loads(l) for l in (d / "trials.jsonl").read_text().splitlines() if l.strip()]
    val = json.loads((d / "validator.json").read_text())
    g: dict = {}
    for r in rows:
        g.setdefault((r["config"]["condition"], r["config"]["arm"]), []).append(r["metrics"])
    post = lambda k: [m["post_enforcement_commits_sinkdb"] for m in g[k]]
    rg = lambda xs: f"{min(xs)}" if min(xs) == max(xs) else f"{min(xs)}--{max(xs)}"
    kc = lambda a: ("kill_in_complete", a)
    fence_trap = sum(m["trapped_final_status"] == "cancelled" for m in g[kc("fence_sink")])
    other = g[kc("halt_inference")] + g[kc("revoke_broker")]
    other_after = sum(m["trapped_final_status"] == "committed" for m in other)
    held = lambda a: [h for m in g[("late_accept", a)] for h in m["held_permits"]]
    committed_after = lambda hs: sum(h["sink_record_status"] == "committed" and h["effect_after_watermark"] for h in hs)
    hf, ho = held("fence_sink"), held("halt_inference") + held("revoke_broker")
    cl = g[("closed_loop", "halt_inference")] + g[("closed_loop", "revoke_broker")]
    eq = sum(m["post_enforcement_commits_sinkdb"] == m["in_flight_at_enforcement"] for m in cl)
    clf = g[("closed_loop", "fence_sink")]
    viol = sum(len(m["fence_order_violations"]) for ms in g.values() for m in ms)
    dis = val["sinkdb_vs_harness_post_enforcement_disagreements"]
    dis_conds = sorted({next(r["config"]["condition"] for r in rows if r["config"]["trial_id"] == x["trial"]) for x in dis})
    bad_kills = val["kills_not_meeting_criterion"]
    s = (rf"In E4b ({len(rows)} predeclared trials; "
         + ("every kill was confirmed to land with its request outstanding" if not bad_kills else f"{len(bad_kills)} kills missed the predeclared criterion")
         + rf"), the post-stop count is read from the sink's database. With the sink killed inside an outstanding \texttt{{complete}}, "
         rf"the fenced arm committed {rg(post(kc('fence_sink')))} effects after the stop and the interrupted operation ended cancelled "
         rf"in {fence_trap} of {len(g[kc('fence_sink')])} trials; under halting and revoking it was retried and committed after the "
         rf"stop in {other_after} of {len(other)}, with {rg(post(kc('halt_inference')))} and {rg(post(kc('revoke_broker')))} "
         rf"post-stop commits. Killed inside an outstanding \texttt{{fence}}, the fenced arm committed "
         rf"{rg(post(('kill_in_fence', 'fence_sink')))}. Of {len(hf)} accepts held across the fence, {committed_after(hf)} produced an "
         rf"effect; of {len(ho)} held across a halt or a revocation, {committed_after(ho)} did. With closed-loop agents that wait for "
         rf"each result, halting and revoking let through exactly the calls in flight at the stop in {eq} of {len(cl)} trials, and "
         rf"the fence committed {rg([m['post_enforcement_commits_sinkdb'] for m in clf])} while discarding "
         rf"{rg([m['legit_work_lost'] for m in clf])} calls. The sink's ordering check found {viol} violations.")
    if dis:
        s += (rf" In {len(dis)} trials ({', '.join(c.replace('_', ' ') for c in dis_conds)}) the harness log counted a post-stop "
              r"commit that the database shows was made before the stop, which is why E4b counts from the database.")
    return {"vsixEFourBSentence": s}


# ------------------------------------------------------------------ E5

def e5_macros(args) -> dict:
    from e5_analysis_v6 import analyse
    d = src(args, "e5")
    plan = json.loads((ROOT / "fixtures/v6/omitted_clause_plan.json").read_text())
    an = analyse([json.loads(l) for l in (d / "candidates.jsonl").read_text().splitlines() if l.strip()], plan)
    ck = an["checkpoints"][args.ckpt]
    sets, integ = ck["sets"]["strategy"], ck["integrity"]
    sd = sets["state_dependent"]["arms"]["R2"]["breach_cases"]
    fams = plan["cases"]["families"]
    parts = []
    for f in [f for f in STATE + VISIBLE if f in sets]:
        b = sets[f]
        parts.append(f"{frac(b['single_clause_unique'][fams[f]['target_clause']], b['unauthorized_unique'])} "
                     f"({FAMILY_LABEL[f].lower()})")
    single = (r"of the distinct unauthorized proposals that search found, " + series(parts) +
              r" break only their family's clause.")
    args.tables["v6-e5-table.tex"] = e5_table(an, d)
    m = {"vsixEFiveCkpt": short(args.ckpt), "vsixReviewerStateCases": frac(sd["x"], sd["n"]),
         "vsixSingleClauseSentence": single[0].upper() + single[1:],
         "vsixRtwoTruncated": f"{integ['r2_truncated_unique']} of {integ['unique_valid']}"}
    rp = d / "gate-replay.json"
    if rp.exists():
        r = json.loads(rp.read_text())
        bad = sum("gate decision not reproduced" in p for p in r["problems"])
        n = r["gate_replayed_unique"]
        verb = "reproduced every recorded gate decision" if not bad else f"reproduced {n - bad} of {n} recorded gate decisions"
        m["vsixGateReplaySentence"] = (rf"Re-creating the broker and sink for each of the {n} distinct valid proposals "
                                       rf"(\texttt{{validate\_omitted\_clause\_v6.py --replay-gate}}) {verb}.")
    elif not args.draft:
        raise SystemExit(f"{rp} is missing; promote E5 with scripts/promote_v6.py, which runs the replay")
    return m


def e5_table(an: dict, d: Path) -> str:
    """Every arm, both checkpoints, the strategy-conditioned set; distinct proposals and cases as units."""
    order = list(STATE) + list(VISIBLE) + ["state_dependent", "proposal_visible"]
    name = FAMILY_LABEL | {"state_dependent": r"\emph{State-dependent}", "proposal_visible": r"\emph{Proposal-visible}"}
    arms = ("R0", "R1", "R2", "G", "R2_excl_truncated", "R2_truncated_as_allow")
    rows = []
    for ck, c in sorted(an["checkpoints"].items()):
        sets = c["sets"]["strategy"]
        rows.append(rf"\multicolumn{{9}}{{l}}{{\textbf{{{short(ck)}}}}} \\")
        for f in [f for f in order if f in sets]:
            b = sets[f]
            cells = [frac(b["arms"][a]["breach_unique"]["x"], b["arms"][a]["breach_unique"]["n"]) for a in arms]
            r2c, r2f = b["arms"]["R2"]["breach_cases"], b["arms"]["R2"]["false_block_unique"]
            rows.append(rf"{name[f]} & {' & '.join(cells[:4])} & {frac(r2c['x'], r2c['n'])} & {frac(r2f['x'], r2f['n'])} & "
                        rf"{' & '.join(cells[4:])} \\")
    return "\n".join([
        f"% Generated by scripts/v6_macros.py from {d.name}; do not edit.",
        r"\begin{table*}[t]\centering\footnotesize",
        r"\caption{E5, strategy-conditioned search. Unauthorized proposals admitted per arm, each distinct proposal once "
        r"(admitted/unauthorized). R0 the prior work's prompt, R1 rules stated, R2 rules and reasoning (a truncated review "
        r"is a denial), G the gate. ``Cases'' counts cases with at least one R2 admission; ``false block'' counts authorized "
        r"proposals R2 denied. The last two columns are post hoc sensitivities for R2's truncated reviews.}\label{tab:v6e5}",
        r"\begin{tabular}{lcccccccc}\toprule",
        r"Family & R0 & R1 & R2 & G & R2 cases & R2 false block & R2 excl.\ trunc. & R2 trunc.\ = allow \\\midrule",
        *rows, r"\bottomrule\end{tabular}\end{table*}", ""])


# ------------------------------------------------------------------ E8, E8b, E2

def stage0(static: dict, spec: str) -> tuple[str, str]:
    cells = [v for k, v in static["summary"].items() if k.split("/")[1] == spec]
    u = frac(sum(c["user_reference_admitted"] for c in cells), sum(c["user_tasks"] for c in cells))
    a = frac(sum(c["attack_reference_admitted"] for c in cells), sum(c["attack_pairs"] for c in cells))
    return u, a


def arm_counts(rows: list[dict], arm: str) -> dict:
    ben = [r for r in rows if r["arm"] == arm and r["injection_task"] is None]
    att = [r for r in rows if r["arm"] == arm and r["injection_task"] is not None]
    return {"utility": (sum(bool(r.get("utility")) for r in ben), len(ben)),
            "attack": (sum(bool(r.get("attack_success")) for r in att), len(att))}


def e8_macros(args) -> dict:
    d = src(args, "e8")
    plan = json.loads((ROOT / "fixtures/v6/agentdojo_fair_plan.json").read_text())
    static = json.loads((d / "static.json").read_text())
    rows = load_rows(d)
    m = {"vsixEEightCkpt": short(plan["agent"]["checkpoint"]), "vsixEEightEpisodes": len(rows)}
    for spec, name in (("strict", "Strict"), ("delegated", "Delegated"), ("oracle", "Oracle")):
        m[f"vsixStage{name}User"], m[f"vsixStage{name}Attack"] = stage0(static, spec)
    pol = json.loads((d / "progent_policies.json").read_text())
    failed = sum(v["policy"] is None for v in pol.values())
    p = arm_counts(rows, "progent")
    gen = (f"Policy generation succeeded for all {len(pol)} user tasks." if not failed else
           f"Policy generation failed for {failed} of {len(pol)} user tasks, whose episodes refuse every call.")
    m["vsixEEightProgentSentence"] = (
        f"Progent, with per-task policies generated by the same checkpoint, completed {frac(*p['utility'])} benign tasks, "
        f"and AgentDojo's injections succeeded in {frac(*p['attack'])} of its episodes. {gen}")
    return m


def e8b_macros(args) -> dict:
    static = json.loads((src(args, "e8b") / "static.json").read_text())
    return {"vsixStageInsideStrict": stage0(static, "strict")[1], "vsixStageInsideOracle": stage0(static, "oracle")[1]}


def e2_macros(args) -> dict:
    # promote_v6 names E2 evidence e2-<key>; the checkpoint is read from that key's plan, which E2's validator binds.
    dirs = {p.name[3:]: p for p in EVIDENCE.glob("e2-*") if p.is_dir()} | {k[3:]: v for k, v in args.sources.items() if k.startswith("e2-")}
    if not dirs:
        raise Missing("e2")
    parts, names = [], []
    for key, d in sorted(dirs.items()):
        res = json.loads((d / "results.json").read_text())
        plan_path = ROOT / f"fixtures/v6/e2/{key}.json"
        if not args.draft and res["plan_sha256"] != hashlib.sha256(plan_path.read_bytes()).hexdigest():
            raise SystemExit(f"{d}: results do not match {plan_path}")
        ck = short(json.loads(plan_path.read_text())["agent"]["checkpoint"])
        rows = load_rows(d)
        un, st = arm_counts(rows, "unprotected"), arm_counts(rows, "authority_strict")
        names.append(ck)
        parts.append(f"{ck}: unprotected {frac(*un['utility'])} benign and {frac(*un['attack'])} attacks; "
                     f"strict {frac(*st['utility'])} benign and {frac(*st['attack'])} attacks")
    return {"vsixETwoCkpts": series(names), "vsixETwoSentence": "; ".join(parts) + "."}


# ------------------------------------------------------------------ E1

def e1_report(ca: dict, sec: dict | None, source: str, label: str, title: str) -> tuple[dict, str]:
    """Claims sentence and table from a campaign_analysis.py report (E1, or the v2 pilot, which has no S4)."""
    total = held = planned = s4_ok = s4_n = 0
    sentences, rows = [], []
    fmt = lambda v: "--" if v is None else f"{v:+.3f}"
    mark = lambda b: r"\checkmark" if b else "--"
    for ck, res in sorted(ca["checkpoints"].items()):
        planned += sum(g.get("planned_campaigns") or 0 for g in res["controllers"].values())
        sup, nsup, ni, nni, inel = [], [], [], [], []
        for other, c in res["paired_primary_comparisons"].items():
            lab = CTRL_LABEL[other].lower()
            total += 2
            held += c["safety_superior"] + c["usefulness_noninferior"]
            (sup if c["safety_superior"] else nsup).append(lab)
            (ni if c["usefulness_noninferior"] else nni).append(lab)
            if not c["claims_eligible"]:
                inel.append(lab)
            sd, ud = c["safety_difference"] or {}, c["usefulness_difference"] or {}
            row = (f"{short(ck)} & {CTRL_LABEL[other]} & {fmt(sd.get('mean'))} & {fmt(sd.get('upper'))} & {mark(c['safety_superior'])} & "
                   f"{fmt(ud.get('mean'))} & {fmt(ud.get('lower'))} ({ud.get('loss_cases', '--')}/{ud.get('independent_cases', '--')}) & "
                   f"{mark(c['usefulness_noninferior'])}")
            if sec is not None:
                s4 = sec["S4"][f"{ck}/complete_vs_{other}"]
                s4_n += 1
                s4_ok += s4["noninferior_at_5pp"]
                row += f" & {fmt(s4['usefulness_lower'])} ({s4['loss_cases']}/{s4['non_revocation_cases']}) & {mark(s4['noninferior_at_5pp'])}"
            rows.append(row + r" \\")
        def claim(phrase: str, prep: str, yes: list, no: list) -> str:
            if not yes:
                return f"not {phrase} any comparator"
            return f"{phrase} every comparator" if not no else f"{phrase} {series(yes)} but not {prep} {series(no)}"
        s = (f"At {short(ck)}, complete mediation was {claim('shown safer than', 'than', sup, nsup)}, and "
             f"{claim('shown noninferior in usefulness to', 'to', ni, nni)}.")
        if inel:
            s += f" Claims against {series(inel)} were ineligible because pairs were missing."
        sentences.append(s)
    m = {"planned": planned, "sentence": f"Of the {total} predeclared component claims, {held} held. " + " ".join(sentences),
         "s4": (f"Without the revocation cases (S4), the usefulness bound cleared the five-point margin in {s4_ok} of {s4_n} "
                f"checkpoint--comparator pairs.") if sec is not None else None}
    cols, head = (r"llrrcrrcrc", r" & S4 lower (loss/$n$) & S4") if sec is not None else (r"llrrcrrc", "")
    table = "\n".join([
        f"% Generated by scripts/v6_macros.py from {source}; do not edit.",
        r"\begin{table*}[t]\centering\footnotesize",
        rf"\caption{{{title} (complete mediation minus comparator; one-sided bounds at $\alpha = 0.05/16$; a claim holds when "
        r"the safety upper bound is below 0 or the usefulness lower bound is above $-0.05$). Loss counts are independent cases "
        r"in which the comparator succeeded and complete mediation did not."
        + (r" S4 repeats the usefulness bound without revocation cases." if sec is not None else "") + rf"}}\label{{{label}}}",
        rf"\begin{{tabular}}{{{cols}}}\toprule",
        r"Ckpt & Comparator & $\Delta$unauth. & upper & safer & $\Delta$useful & lower (loss/$n$) & noninf." + head + r" \\\midrule",
        *rows, r"\bottomrule\end{tabular}\end{table*}", ""])
    return m, table


def margin() -> float:
    return json.loads((ROOT / "fixtures/v2/evaluation.json").read_text())["inference"]["usefulness_noninferiority_margin"]


def e1_design(d: Path, alpha: float) -> dict:
    """Design facts of the any-loss bound at E1's sample (from the holdout roster; no outcome is read)."""
    from campaign_analysis import binomial_bound
    h = json.loads((d / "holdout.json").read_text())
    n = h["independent_base_cases"]
    n4 = sum(c["attack"] != "revocation" for c in h["campaigns"])
    ok = [l for l in range(0, n + 1) if binomial_bound(l, n, alpha) < margin()]
    zero, one = binomial_bound(0, n, alpha), binomial_bound(1, n, alpha) if n > 1 else 1.0
    pts = f"{margin() * 100:g}-point"
    if not ok:
        rule = f"At N = {n} the usefulness bound cannot clear the {pts} margin even with no losses (zero-loss bound {zero:.3f})."
    elif max(ok) == 0:
        rule = (f"At N = {n} the usefulness bound clears the {pts} margin only if complete mediation loses no case to the "
                f"comparator: zero losses give a bound of {zero:.3f} and one gives {one:.3f}. A single case that the comparator "
                f"completes and complete mediation does not defeats the claim.")
    else:
        rule = f"At N = {n} the usefulness bound clears the {pts} margin only if complete mediation loses at most {max(ok)} cases."
    return {"n": n, "n4": n4, "zero": zero, "one": one, "zero4": binomial_bound(0, n4, alpha), "rule": rule}


def e1_macros(args) -> tuple[dict, str]:
    d = src(args, "e1")
    ca = json.loads((d / "campaign-analysis.json").read_text())
    r, table = e1_report(ca, json.loads((d / "secondary.json").read_text()),
                         "evidence/v6/reference/e1-n128", "tab:v6e1", "E1 primary claims under the frozen analysis")
    g = e1_design(d, ca["component_alpha"])
    s4 = r["s4"]
    if g["zero4"] >= margin():
        s4 = (rf"S4 cannot clear the {margin() * 100:g}-point margin at this sample even with no losses: its "
              rf"{g['n4']} non-revocation cases give a zero-loss bound of {g['zero4']:.3f} (Table~\ref{{tab:v6e1}} reports its "
              rf"bounds). " + s4)
    return {"vsixEOneCampaigns": f"{r['planned']:,}", "vsixEOneOutcomeSentence": r["sentence"],
            "vsixEOneSFourSentence": s4, "vsixEOneN": g["n"], "vsixEOneZeroLossBound": f"{g['zero']:.3f}",
            "vsixEOneOneLossBound": f"{g['one']:.3f}", "vsixEOneLossRuleSentence": g["rule"]}, table


def pilot_macros() -> tuple[dict, str]:
    """The N=32 pilot (v2 evidence, read only): reported beside E1, never pooled with it."""
    path = ROOT / "evidence/v2/reference/model-campaigns-analyzed.json"
    r, table = e1_report(json.loads(path.read_text()), None, "evidence/v2/reference/model-campaigns-analyzed.json",
                         "tab:v6pilot", "Pilot ($N = 32$ independent cases per checkpoint) under the same frozen analysis")
    from campaign_analysis import binomial_bound
    ca = json.loads(path.read_text())
    a, mg = ca["component_alpha"], margin()
    n_pilot = max(c["usefulness_difference"]["independent_cases"] for res in ca["checkpoints"].values()
                  for c in res["paired_primary_comparisons"].values() if c["usefulness_difference"])
    min_n = next(n for n in range(1, 10000) if binomial_bound(0, n, a) < mg)
    return {"vsixPilotSentence": r["sentence"], "vsixPilotN": n_pilot,
            "vsixPilotZeroLossBound": f"{binomial_bound(0, n_pilot, a):.3f}", "vsixEOneMinN": min_n}, table


# ------------------------------------------------------------------ E6 (+ E6b)

def e6_macros(args) -> dict:
    sf = json.loads((src(args, "e6") / "results.json").read_text())["surfaces"]
    pg, fs = sf["postgresql16"]["primitives"]["P2"], sf["posix_fs"]["primitives"]["P2"]
    d, ck = pg["derived"], pg["checks"]
    s = (f"In the reported run the raise waited {d['fence_lock_wait_seconds']:.2f}\\,s for its lock while "
         f"{d['stale_commits_acked_while_fence_waited']} of {d['attempts']} stale writes were acknowledged; "
         f"{d['attempts_started_after_fence_ack']} writes started after it took effect, and the surface refused {d['refused_by_sink']}.")
    if ck["no_violations"] and not ck["sink_refused_stale"]:
        s += " No stale effect was ordered after the fence, so the safety condition held but vacuously, and the planned P2 outcome is a failure."
    else:
        s += f" The planned P2 outcome is a {pg['outcome']}."
    f = d["fair_fence_variant"]
    fd = f["derived"]
    fair = (f"A variant added by amendment before the reported run first takes a table lock, behind which the lock manager "
            f"queues new writers; its raise waited {fd['fence_lock_wait_seconds']:.3f}\\,s and the surface refused "
            f"{fd['refused_by_sink']} of {fd['attempts']} stale writes (P2 {f['outcome']}), at the cost of pausing writers of "
            f"every issuer while the fence is raised.")
    fsd = fs["derived"]
    fss = (f"The filesystem's advisory \\texttt{{flock}} fence acknowledged {fsd['stale_commits_acked_while_fence_waited']} of "
           f"{fsd['attempts']} stale writes while it waited.")
    return {"vsixESixPgSentence": s, "vsixESixPgFairSentence": fair, "vsixESixFsSentence": fss}


def e6_table(args) -> str:
    res = json.loads((src(args, "e6") / "results.json").read_text())
    rows = []
    for s, v in res["surfaces"].items():
        prim = v.get("primitives") or {}
        enf = (prim.get("P2", {}).get("derived") or {}).get("enforcement")
        mech = prim.get("P2", {}).get("mechanism") or ""
        cls = (CLASS_LABEL[enf] + (r"$^\dagger$" if "trigger" in mech else "")) if enf else "--"
        outs = [OUTCOME_LABEL[(prim.get(p) or {}).get("outcome", "documented_not_executed")] for p in ("P1", "P2", "P3")]
        rows.append(rf"{SURFACE_LABEL.get(s, s)} & {cls} & {' & '.join(outs)} \\")
    return "\n".join([
        "% Generated by scripts/v6_macros.py from evidence/v6/reference/e6-fenceability; do not edit.",
        r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
        r"\caption{Fenceability of effect surfaces (E6): enforcement class of the epoch fence, then the outcome of each "
        r"primitive. Enforcement: \emph{sink}, the surface refuses a bare stale write; \emph{writer}, it refuses only "
        r"when the writer supplies the condition; \emph{advisory}, it never refuses. P1 idempotent retry; P2 epoch fence "
        r"atomic with the effect; P3 crash-surviving hold cancelled by revocation. Local single-node reference instances; ``doc.\ only'' is from documentation. "
        r"$^\dagger$By a trigger we installed in the surface.}\label{tab:v6fence}",
        r"\begin{tabular}{llccc}\toprule", r"Surface & Enforcement & P1 & P2 & P3 \\\midrule", *rows,
        r"\bottomrule\end{tabular}\end{table}", ""])


# ------------------------------------------------------------------ assembly

PENDING = {"e4": ["vsixEFourTrials", "vsixEFourFaultTrials", "vsixEFourCells", "vsixEFourWCCells", "vsixEFourOverloadedCells",
                  "vsixLittleSentence"],
           "e5": ["vsixEFiveCkpt", "vsixReviewerStateCases", "vsixSingleClauseSentence", "vsixRtwoTruncated", "vsixGateReplaySentence"],
           "e8": ["vsixEEightCkpt", "vsixEEightEpisodes", "vsixEEightProgentSentence"] +
                 [f"vsixStage{n}{k}" for n in ("Strict", "Delegated", "Oracle") for k in ("User", "Attack")],
           "e8b": ["vsixStageInsideStrict", "vsixStageInsideOracle"],
           "e6": ["vsixESixPgSentence", "vsixESixPgFairSentence", "vsixESixFsSentence"],
           "e2": ["vsixETwoCkpts", "vsixETwoSentence"],
           "e1": ["vsixEOneCampaigns", "vsixEOneOutcomeSentence", "vsixEOneSFourSentence", "vsixEOneN",
                  "vsixEOneZeroLossBound", "vsixEOneOneLossBound", "vsixEOneLossRuleSentence"],
           "e4b": ["vsixEFourBSentence"],
           # Filled when the E6b report lands (its evidence format is set by its own plan).
           "e6b": ["vsixESixBSentence", "vsixESixBStarvationN"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", default="Qwen/Qwen3-8B-GGUF", help="E5 checkpoint cited in the text (as in the hero)")
    ap.add_argument("--draft", action="store_true", help="missing sources become visible [pending] markers")
    ap.add_argument("--source", action="append", default=[], metavar="NAME=PATH")
    args = ap.parse_args()
    args.sources = {k: Path(v).resolve() for k, v in (s.split("=", 1) for s in args.source)}
    if (args.draft or args.sources) and args.out.resolve().is_relative_to((ROOT / "docs").resolve()):
        raise SystemExit("draft or --source output may not be written under docs/")
    macros, missing, tables = {}, [], {}
    args.tables = tables
    for name, fn in (("e4", e4_macros), ("e4b", e4b_macros), ("e5", e5_macros), ("e6", e6_macros), ("e8", e8_macros), ("e8b", e8b_macros),
                     ("e2", e2_macros)):
        try:
            macros |= fn(args)
        except Missing as e:
            missing.append(str(e))
    try:
        m, tables["v6-e1-table.tex"] = e1_macros(args)
        macros |= m
    except Missing:
        missing.append("e1")
    m, tables["v6-pilot-table.tex"] = pilot_macros()
    macros |= m
    try:
        tables["v6-e6-table.tex"] = e6_table(args)
    except Missing:
        missing.append("e6")
    missing = list(dict.fromkeys(missing + ["e6b"]))
    ckpts = {short(args.ckpt)} if "vsixEFiveCkpt" in macros else set()
    for k in ("vsixEEightCkpt",):
        if k in macros:
            ckpts.add(macros[k])
    if "vsixETwoCkpts" in macros:
        ckpts |= {c.strip() for c in macros["vsixETwoCkpts"].replace(" and ", ", ").split(",") if c.strip()}
    if not args.draft and missing:
        raise SystemExit(f"missing v6 evidence: {', '.join(missing)} (use --draft for a marked draft outside docs/)")
    for name in missing:
        for k in PENDING.get(name, []):
            macros.setdefault(k, rf"\vsixPending{{{name.upper()}}}")
        if name in ("e1", "e5", "e6"):
            tables.setdefault(f"v6-{name}-table.tex", rf"\vsixPending{{{name.upper()} table}}" + "\n")
    macros["vsixCheckpointList"] = series(sorted(ckpts)) if ckpts else r"\vsixPending{checkpoints}"
    assert all(k.isalpha() for k in macros)
    head = "% Generated by scripts/v6_macros.py; do not edit." + (" DRAFT: contains pending markers." if missing else "") + "\n"
    head += r"\providecommand{\vsixPending}[1]{\textbf{[#1 pending]}}" + "\n"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "v6-text-macros.tex").write_text(head + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))
    for fn, text in tables.items():
        (args.out / fn).write_text(text)
    print(json.dumps({"macros": len(macros), "pending_sources": missing, "tables": sorted(tables)}))


if __name__ == "__main__":
    main()
