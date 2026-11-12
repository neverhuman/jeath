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
SOURCES = {"e4": "e4-revokebench", "e5": "e5-omitted-clause", "e6": "e6-fenceability", "e8": "e8-agentdojo-fair",
           "e8b": "e8b-inside-spec", "e1": "e1-n128"}
CTRL_LABEL = {k: label for k, label, *_ in CONTROLLERS}
SURFACE_LABEL = {"postgresql16": "PostgreSQL 16", "etcd_v3_5_17": "etcd v3.5.17", "posix_fs": "POSIX filesystem",
                 "aws_s3": "Amazon S3", "kubernetes_api": "Kubernetes API"}
CLASS_LABEL = {"sink_enforced": "sink-enforced", "writer_supplied_condition": "writer-supplied condition",
               "advisory": "advisory"}
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
    return {"vsixEFourTrials": len(trials),
            "vsixEFourFaultTrials": sum(t["config"]["fault"] != "none" for t in trials),
            "vsixEFourCells": len({(c["w"], c["C"], c["L"]) for c in nofault}),
            "vsixEFourWCCells": len(wc),
            "vsixEFourOverloadedCells": len({(c["w"], c["C"]) for c in nofault if rho(c) >= 1})}


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

def e1_macros(args) -> tuple[dict, str]:
    d = src(args, "e1")
    ca = json.loads((d / "campaign-analysis.json").read_text())
    sec = json.loads((d / "secondary.json").read_text())
    total = held = 0
    sentences, rows, s4_ok, s4_n = [], [], 0, 0
    planned = 0
    for ck, res in sorted(ca["checkpoints"].items()):
        planned += sum(g["planned_campaigns"] or 0 for g in res["controllers"].values())
        sup, nsup, ni, nni, inel = [], [], [], [], []
        for other, c in res["paired_primary_comparisons"].items():
            lab = CTRL_LABEL[other].lower()
            total += 2
            held += c["safety_superior"] + c["usefulness_noninferior"]
            (sup if c["safety_superior"] else nsup).append(lab)
            (ni if c["usefulness_noninferior"] else nni).append(lab)
            if not c["claims_eligible"]:
                inel.append(lab)
            s4 = sec["S4"][f"{ck}/complete_vs_{other}"]
            s4_n += 1
            s4_ok += s4["noninferior_at_5pp"]
            sd, ud = c["safety_difference"] or {}, c["usefulness_difference"] or {}
            fmt = lambda v: "--" if v is None else f"{v:+.3f}"
            mark = lambda b: r"\checkmark" if b else "--"
            rows.append(f"{short(ck)} & {CTRL_LABEL[other]} & {fmt(sd.get('mean'))} & {fmt(sd.get('upper'))} & {mark(c['safety_superior'])} & "
                        f"{fmt(ud.get('mean'))} & {fmt(ud.get('lower'))} ({ud.get('loss_cases', '--')}/{ud.get('independent_cases', '--')}) & "
                        f"{mark(c['usefulness_noninferior'])} & {fmt(s4['usefulness_lower'])} ({s4['loss_cases']}/{s4['non_revocation_cases']}) & "
                        rf"{mark(s4['noninferior_at_5pp'])} \\")
        s = (f"At {short(ck)}, complete mediation was shown safer than {series(sup)}"
             + (f" but not than {series(nsup)}" if nsup else "")
             + f", and noninferior in usefulness to {series(ni)}" + (f" but not to {series(nni)}" if nni else "") + ".")
        if inel:
            s += f" Claims against {series(inel)} were ineligible because pairs were missing."
        sentences.append(s)
    m = {"vsixEOneCampaigns": f"{planned:,}",
         "vsixEOneOutcomeSentence": f"Of the {total} predeclared component claims, {held} held. " + " ".join(sentences),
         "vsixEOneSFourSentence": (f"Without the revocation cases (S4), the usefulness bound cleared the five-point margin in "
                                   f"{s4_ok} of {s4_n} checkpoint--comparator pairs.")}
    table = "\n".join([
        "% Generated by scripts/v6_macros.py from evidence/v6/reference/e1-n128; do not edit.",
        r"\begin{table*}[t]\centering\footnotesize",
        r"\caption{E1 primary claims under the frozen analysis (complete mediation minus comparator; one-sided bounds at "
        r"$\alpha = 0.05/16$; a claim holds when the safety upper bound is below 0 or the usefulness lower bound is above "
        r"$-0.05$). Loss counts are independent cases in which the comparator succeeded and complete mediation did not. S4 "
        r"repeats the usefulness bound without revocation cases.}\label{tab:v6e1}",
        r"\begin{tabular}{llrrcrrcrc}\toprule",
        r"Ckpt & Comparator & $\Delta$unauth. & upper & safer & $\Delta$useful & lower (loss/$n$) & noninf. & S4 lower (loss/$n$) & S4 \\\midrule",
        *rows, r"\bottomrule\end{tabular}\end{table*}", ""])
    return m, table


# ------------------------------------------------------------------ E6 (+ E6b)

def e6_table(args) -> str:
    res = json.loads((src(args, "e6") / "results.json").read_text())
    rows = []
    for s, v in res["surfaces"].items():
        prim = v.get("primitives") or {}
        enf = (prim.get("P2", {}).get("derived") or {}).get("enforcement")
        mech = prim.get("P2", {}).get("mechanism") or ""
        cls = (CLASS_LABEL[enf] + (r"$^\dagger$" if "trigger" in mech else "")) if enf else "not executed"
        outs = [OUTCOME_LABEL[(prim.get(p) or {}).get("outcome", "documented_not_executed")] for p in ("P1", "P2", "P3")]
        rows.append(rf"{SURFACE_LABEL.get(s, s)} & {cls} & {' & '.join(outs)} \\")
    return "\n".join([
        "% Generated by scripts/v6_macros.py from evidence/v6/reference/e6-fenceability; do not edit.",
        r"\begin{table}[t]\centering\footnotesize",
        r"\caption{Fenceability of effect surfaces (E6): enforcement class of the epoch fence, then the outcome of each "
        r"primitive. P1 idempotent retry; P2 epoch fence atomic with the effect; P3 crash-surviving hold cancelled by "
        r"revocation. Local single-node reference instances; ``doc.\ only'' is from documentation. "
        r"$^\dagger$By a trigger we installed in the surface.}\label{tab:v6fence}",
        r"\begin{tabular}{llccc}\toprule", r"Surface & Enforcement & P1 & P2 & P3 \\\midrule", *rows,
        r"\bottomrule\end{tabular}\end{table}", ""])


# ------------------------------------------------------------------ assembly

PENDING = {"e4": ["vsixEFourTrials", "vsixEFourFaultTrials", "vsixEFourCells", "vsixEFourWCCells", "vsixEFourOverloadedCells"],
           "e5": ["vsixEFiveCkpt", "vsixReviewerStateCases", "vsixSingleClauseSentence", "vsixRtwoTruncated", "vsixGateReplaySentence"],
           "e8": ["vsixEEightCkpt", "vsixEEightEpisodes", "vsixEEightProgentSentence"] +
                 [f"vsixStage{n}{k}" for n in ("Strict", "Delegated", "Oracle") for k in ("User", "Attack")],
           "e8b": ["vsixStageInsideStrict", "vsixStageInsideOracle"],
           "e2": ["vsixETwoCkpts", "vsixETwoSentence"],
           "e1": ["vsixEOneCampaigns", "vsixEOneOutcomeSentence", "vsixEOneSFourSentence"],
           # Filled when the E4b / E6b reports land (their evidence formats are set by their own plans).
           "e4b": ["vsixEFourBSentence"], "e6b": ["vsixESixBSentence", "vsixESixBStarvationN"]}


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
    for name, fn in (("e4", e4_macros), ("e5", e5_macros), ("e8", e8_macros), ("e8b", e8b_macros), ("e2", e2_macros)):
        try:
            macros |= fn(args)
        except Missing as e:
            missing.append(str(e))
    try:
        m, tables["v6-e1-table.tex"] = e1_macros(args)
        macros |= m
    except Missing:
        missing.append("e1")
    try:
        tables["v6-e6-table.tex"] = e6_table(args)
    except Missing:
        missing.append("e6")
    missing += ["e4b", "e6b"]
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
        if name in ("e1", "e6"):
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
