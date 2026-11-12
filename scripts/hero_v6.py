#!/usr/bin/env python3
"""Front figure (hero) for "Intelligence Is Not Authority", built only from promoted E4/E5/E8(/E8b) evidence.

Every number and every comparative word in the figure and caption is computed from the inputs; nothing about an outcome
is written in advance. A missing, smoke or fake input is a hard error unless --layout-preview is given, in which case the
figure carries a "LAYOUT PREVIEW - NOT EVIDENCE" watermark and may not be written under docs/.

  (a) E4   effects committed after the stop took effect, per stop layer (real broker/sink processes). The cell drawn is
           chosen by a fixed rule, not by hand: the stable cell (offered load rho = lambda*w/C < 1) with the longest mean
           tool latency, stop enforced at once (L = 0). The caption gives ranges over the whole grid.
  (b) E5   unauthorized proposals admitted, per attack family, under the plan's parse and truncation rules
           (scripts/e5_analysis_v6.py), unique candidates as the unit. The "clause omitted" marks are by construction.
  (c) E8   AgentDojo benign utility vs attack success per arm: AgentDojo's own injections (filled) and, with --e8b,
           injections that stay inside specification S (open), joined per arm.
  E1's predeclared replication has its own figure; panel_e1() is kept for it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
STATE = ("duplicate_replay", "withdrawn_issuer", "stale_revision")
VISIBLE = ("altered_payload", "injected_exfiltration")
FAMILY_LABEL = {"duplicate_replay": "Replay, budget spent", "withdrawn_issuer": "Issuer withdrawn",
                "stale_revision": "Stale revision", "altered_payload": "Altered payload",
                "injected_exfiltration": "Injected payload"}
FILLED = {"square": "square*", "diamond": "diamond*", "o": "*", "pentagon": "pentagon*", "triangle": "triangle*"}
E8_ARMS = (("unprotected", "Unprotected", "structgray", "square"), ("progent", "Progent (per call)", "warnamber!85!black", "diamond"),
           ("authority_strict", "Strict provenance", "exoblue", "o"), ("authority_delegated", "Delegated sources", "exoblue", "pentagon"),
           ("authority_oracle", "Oracle grant (bound)", "exoblue!60!black", "triangle"))
CONTROLLERS = (("none", "No mediation", "structgray", "square*"), ("stateless", "Per-call policy", "warnamber", "diamond*"),
               ("learned_same_checkpoint", "Same-model reviewer", "endored", "triangle*"),
               ("durable_without_fence", "Durable, no fence", "safegreen", "pentagon*"),
               ("complete", "Fenced sink", "exoblue", "*"))


def cp(x: int, n: int, alpha: float = 0.05):
    from omitted_clause_v6 import clopper_pearson
    return tuple(clopper_pearson(x, n, alpha))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rng(xs) -> str:
    lo, hi = min(xs), max(xs)
    return f"{lo}" if lo == hi else f"{lo}--{hi}"


def pct(x, n) -> str:
    return f"{x}/{n}"


LEGEND_C = (r"legend style={font=\tiny,at={(0.5,-0.34)},anchor=north,draw=none,fill=none,legend columns=2,"
            r"row sep=-1.5pt,column sep=3pt},legend cell align=left")
AXIS = (r"scale only axis,width=3.75cm,height=3.7cm,tick label style={font=\scriptsize},label style={font=\footnotesize},"
        r"title style={font=\footnotesize,align=center}")


# ------------------------------------------------------------------ panel (a): E4

def load_e4(path: Path) -> tuple[list[dict], dict]:
    """Promoted evidence (trials.jsonl + results.json) or a raw run (results.json + trials/<id>/trial.json)."""
    d = path if path.is_dir() else path.parent
    res = json.loads((d / "results.json").read_text())
    if (d / "trials.jsonl").exists():
        trials = [json.loads(l) for l in (d / "trials.jsonl").read_text().splitlines() if l.strip()]
    else:
        trials = [json.loads((d / "trials" / t["config"]["trial_id"] / "trial.json").read_text()) for t in res["trials"]]
    return trials, res


def rho(cfg: dict) -> float:
    wl = cfg["workload"]
    return wl["calls_per_step"] / wl["model_step_seconds"] * cfg["w"] / cfg["C"]


def panel_a(trials: list[dict]) -> tuple[list[str], dict]:
    nofault = [t for t in trials if t["config"]["fault"] == "none"]
    stable = [t for t in nofault if rho(t["config"]) < 1]
    w = max(t["config"]["w"] for t in stable)
    C = max(t["config"]["C"] for t in stable if t["config"]["w"] == w)
    sel = [t for t in stable if t["config"]["w"] == w and t["config"]["C"] == C and t["config"]["L"] == 0]
    curves, lost, tmax = {}, [], 0.0
    for t in sel:
        ev = t["events"]
        t_enf = next(e["t"] for e in ev if e["kind"] == "enforced")
        ts = sorted(e["t"] - t_enf for e in ev if e["kind"] == "completed" and e["status"] == "committed" and e["t"] > t_enf)
        if len(ts) != t["metrics"]["post_enforcement_commits"]:
            raise SystemExit(f"{t['config']['trial_id']}: event log does not reproduce post_enforcement_commits")
        curves.setdefault(t["config"]["arm"], []).append(ts)
        tmax = max(tmax, ts[-1] if ts else 0.0)
        if t["config"]["arm"] == "fence_sink":
            lost.append(t["metrics"]["cancelled_legitimate"] + t["metrics"]["denied_after_enforcement"])
    fin = {a: sorted(len(c) for c in cs) for a, cs in curves.items()}
    grid = {a: [t["metrics"]["post_enforcement_commits"] for t in nofault if t["config"]["arm"] == a] for a in fin}
    fence_all = [t for t in trials if t["config"]["arm"] == "fence_sink"]
    st = {"rule": "stable cell (rho<1) with the longest mean tool latency, L=0", "w": w, "C": C, "rho": round(2 * w / C, 3),
          "replicates": len(curves["fence_sink"]), "final": fin, "fence_lost_legitimate": sorted(lost),
          "last_commit_s": round(tmax, 1), "grid_max": {a: max(v) for a, v in grid.items()},
          "grid_cells": len({(t["config"]["w"], t["config"]["C"], t["config"]["L"]) for t in nofault}),
          "fence_trials": len(fence_all), "fence_post_max": max(t["metrics"]["post_enforcement_commits"] for t in fence_all),
          "fence_fault_trials": sum(t["config"]["fault"] != "none" for t in fence_all),
          "overloaded_cells": sorted({(t["config"]["w"], t["config"]["C"]) for t in nofault if rho(t["config"]) >= 1})}
    xmax = math.ceil(max(tmax, 1.0) * 1.08 / 5) * 5
    ymax = max(fin["halt_inference"] + fin["revoke_broker"]) * 1.25 + 1
    lines = [r"\begin{tikzpicture}[baseline]",
             rf"\begin{{axis}}[{AXIS},xmin=0,xmax={xmax},ymin=-0.6,ymax={ymax:.0f},grid=major,clip=false,"
             r"xlabel={Seconds after the stop took effect},ylabel={Effects committed after the stop},"
             r"title={\textbf{(a)} Stopping the model $\neq$ stopping its effects}]"]
    styles = {"halt_inference": "endored,thick,dashed", "revoke_broker": "warnamber!80!black,thick,dashdotted",
              "fence_sink": "exoblue,ultra thick"}
    for arm in ("halt_inference", "revoke_broker", "fence_sink"):
        for ts in curves[arm]:
            pts = [(0.0, 0)] + [(x, i + 1) for i, x in enumerate(ts)] + [(xmax, len(ts))]
            lines.append(rf"\addplot[{styles[arm]},const plot,opacity=0.85] coordinates {{{' '.join(f'({x:.2f},{y})' for x, y in pts)}}};")
    lines += [
        rf"\node[font=\scriptsize,text=ink,anchor=south east,inner sep=1pt] at (axis cs:{xmax},{max(fin['halt_inference'])}) {{halt the model}};",
        rf"\node[font=\scriptsize,text=ink,anchor=north east,inner sep=1.5pt] at (axis cs:{xmax},{min(fin['revoke_broker'])}) {{revoke at the broker}};",
        rf"\node[font=\scriptsize,text=ink,anchor=south east,inner sep=1.5pt,align=right] at (axis cs:{xmax},0.4) "
        rf"{{\textbf{{fence the sink: {rng(fin['fence_sink'])}}}\\[-1pt]loses {rng(lost)} legitimate calls}};",
        r"\end{axis}\end{tikzpicture}"]
    return lines, st


def caption_a(s: dict) -> str:
    f = s["final"]
    return (rf"\textbf{{(a)}} Real broker and sink processes; {s['C']} concurrent tool calls of {s['w']:g}\,s mean latency "
            rf"(offered load {s['rho']:g}, the stable cell with the longest latency), stop enforced at once; each of "
            rf"{s['replicates']} replicates drawn. After the stop, halting the model lets {rng(f['halt_inference'])} effects commit "
            rf"and revoking at the broker {rng(f['revoke_broker'])}, the last {s['last_commit_s']:g}\,s later; fencing the sink lets "
            rf"{rng(f['fence_sink'])} commit and loses {rng(s['fence_lost_legitimate'])} legitimate calls (cancelled or denied). Over all "
            rf"{s['grid_cells']} fault-free cells the most leaked by halting is {s['grid_max']['halt_inference']} and by revoking "
            rf"{s['grid_max']['revoke_broker']}; the fence admitted at most {s['fence_post_max']} in {s['fence_trials']} trials, "
            rf"{s['fence_fault_trials']} with SIGKILL of the sink or late completions. The fence's zero is a design property of the "
            r"sink; what is measured is that this implementation keeps it under those faults. ")


# ------------------------------------------------------------------ panel (b): E5

def panel_b(an: dict, plan: dict, ckpt: str) -> tuple[list[str], dict]:
    sets = an["checkpoints"][ckpt]["sets"]["strategy"]
    fams = plan["cases"]["families"]
    order = [f for f in list(STATE) + list(VISIBLE) if f in sets]
    ypos = {f: len(order) - i + (0.4 if f in STATE else 0) for i, f in enumerate(order)}
    st = {"ckpt": ckpt, "families": {}}
    for f in order:
        b = sets[f]
        omit = b["single_clause_unique"][fams[f]["target_clause"]]
        st["families"][f] = {"unauthorized_unique": b["unauthorized_unique"], "cases": b["cases_with_unauthorized"],
                             "omitted_clause_share": [omit, b["unauthorized_unique"]],
                             **{a: {k: b["arms"][a][k] for k in ("breach_unique", "breach_cases", "false_block_unique")}
                                for a in ("R0", "R1", "R2", "G")}}
    sd, pv = sets["state_dependent"]["arms"]["R2"]["breach_unique"], sets["proposal_visible"]["arms"]["R2"]["breach_unique"]
    st["R2_state"], st["R2_visible"] = sd, pv
    st["G_breaches"] = sum(sets[f]["arms"]["G"]["breach_unique"]["x"] for f in order)
    separated = sd["ci95"] and pv["ci95"] and sd["ci95"][0] > pv["ci95"][1]
    title = (r"\textbf{(b)} Reviewers miss what lives in state" if separated else
             r"\textbf{(b)} Unauthorized proposals admitted")
    st["title_claim_supported"] = bool(separated)
    lines = [r"\begin{tikzpicture}[baseline]",
             rf"\begin{{axis}}[{AXIS},xmin=-3,xmax=103,ymin=0.4,ymax={max(ypos.values()) + 0.6:.1f},"
             r"ytick={" + ",".join(f"{ypos[f]:.2f}" for f in order) + r"},yticklabels={" + ",".join("{" + FAMILY_LABEL[f] + "}" for f in order) + r"},"
             rf"xlabel={{Unauthorized proposals admitted (\%)}},title={{{title}}},"
             r"yticklabel style={font=\tiny,align=right},xmajorgrids,axis y line*=left,clip=false,"
             r"legend style={font=\tiny,at={(0.5,-0.34)},anchor=north,draw=none,fill=none,legend columns=1,row sep=-1.5pt},legend cell align=left]"]
    offs = {"R2": 0.17, "G": 0.0, "omit": -0.17}
    for arm, color, mark, label in (("R2", "endored", "triangle*", "same-model reviewer, rules + reasoning"),
                                    ("G", "exoblue", "*", "verified gate")):
        pts = []
        for f in order:
            e = st["families"][f][arm]["breach_unique"]
            if not e["n"]:
                continue
            m = 100 * e["x"] / e["n"]
            lo, hi = (100 * v for v in e["ci95"])
            pts.append(f"({m:.4g},{ypos[f] + offs[arm]:.3f}) -= ({m - lo:.4g},0) += ({hi - m:.4g},0)")
        lines.append(rf"\addplot[only marks,mark={mark},mark size=2pt,draw={color},fill={color},error bars/.cd,x dir=both,x explicit,"
                     rf"error bar style={{{color},line width=0.5pt}}] coordinates {{{' '.join(pts)}}};\addlegendentry{{{label}}}")
    pts = []
    for f in order:
        x, n = st["families"][f]["omitted_clause_share"]
        if n:
            pts.append(f"({100 * x / n:.4g},{ypos[f] + offs['omit']:.3f})")
    lines.append(r"\addplot[only marks,mark=o,mark size=2pt,draw=exoblue] coordinates {" + " ".join(pts) + r"};"
                 r"\addlegendentry{gate with that clause unwritten (by construction)}")
    ysep = (min(ypos[f] for f in order if f in STATE) + max(ypos[f] for f in order if f in VISIBLE)) / 2
    lines += [rf"\draw[structgray!60,densely dashed] (axis cs:-3,{ysep:.2f}) -- (axis cs:103,{ysep:.2f});",
              rf"\node[font=\tiny\itshape,text=slate,anchor=south east] at (axis cs:103,{max(ypos.values()) + 0.25:.2f}) {{violation lives in state}};",
              rf"\node[font=\tiny\itshape,text=slate,anchor=north east] at (axis cs:103,{ysep - 0.05:.2f}) {{visible in the proposal}};",
              r"\end{axis}\end{tikzpicture}"]
    return lines, st


def caption_b(s: dict, integ: dict) -> str:
    ck = s["ckpt"].split("/")[-1].replace("-GGUF", "")
    sd, pv = s["R2_state"], s["R2_visible"]
    cmp = "more often" if (sd["rate"] or 0) > (pv["rate"] or 0) else "no more often"
    return (rf"\textbf{{(b)}} {ck} red-team proposals under 8 predeclared mutation strategies, each distinct proposal counted once "
            rf"(95\% Clopper--Pearson). A same-checkpoint reviewer given the rules and time to reason admits "
            rf"{pct(sd['x'], sd['n'])} unauthorized proposals whose violation lives in sink state, {cmp} than the "
            rf"{pct(pv['x'], pv['n'])} whose violation is visible in the proposal. The gate admits {s['G_breaches']}; it enforces "
            rf"the written predicate, and its Rust kernel agreed with the Python reference on all {integ['unique_valid']} valid "
            rf"proposals ({integ['mirror_gate_disagreements']} disagreements). Open circles are by construction: the share of each "
            r"family's violations that break only that family's clause, which a gate without that clause would admit. ")


# ------------------------------------------------------------------ panel (c): E8 (+ E8b)

def load_rows(d: Path) -> list[dict]:
    return [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines() if l.strip()]


def panel_c(rows8: list[dict], rows8b: list[dict] | None) -> tuple[list[str], dict]:
    lines = [r"\begin{tikzpicture}[baseline]",
             rf"\begin{{axis}}[{AXIS},xmin=-3,xmax=103,ymin=-3,ymax=103,grid=major,clip=false,"
             r"xlabel={Benign tasks completed (\%)},ylabel={Attacks that succeed (\%)},"
             r"title={\textbf{(c)} A specification stops what it names}," + LEGEND_C + "]"] if rows8b else [
             r"\begin{tikzpicture}[baseline]",
             rf"\begin{{axis}}[{AXIS},xmin=-3,xmax=103,ymin=-3,ymax=103,grid=major,clip=false,"
             r"xlabel={Benign tasks completed (\%)},ylabel={Attacks that succeed (\%)},"
             r"title={\textbf{(c)} What a specification costs}," + LEGEND_C + "]"]
    st = {"arms": {}}
    for key, label, color, mark in E8_ARMS:
        ben = [r for r in rows8 if r["arm"] == key and r["injection_task"] is None]
        att = [r for r in rows8 if r["arm"] == key and r["injection_task"] is not None]
        if not ben or not att:
            raise SystemExit(f"E8 has no episodes for {key}")
        # Every episode counts: an ungraded episode is neither a completed task nor a successful attack, and is reported.
        u, nu = sum(bool(r.get("utility")) for r in ben), len(ben)
        a, na = sum(bool(r.get("attack_success")) for r in att), len(att)
        e = {"benign_utility": [u, nu], "native_attack": [a, na],
             "ungraded": sum(r["grading_status"] != "completed" for r in ben + att)}
        um, am = 100 * u / nu, 100 * a / na
        (ulo, uhi), (alo, ahi) = cp(u, nu), cp(a, na)
        lines.append(rf"\addplot[only marks,mark={FILLED[mark]},mark size=2.3pt,draw={color},fill={color},error bars/.cd,x dir=both,x explicit,"
                     rf"y dir=both,y explicit,error bar style={{{color},line width=0.45pt}}] coordinates {{({um:.4g},{am:.4g}) "
                     rf"+= ({100 * uhi - um:.4g},{100 * ahi - am:.4g}) -= ({um - 100 * ulo:.4g},{am - 100 * alo:.4g})}};"
                     rf"\addlegendentry{{{label}}}")
        if rows8b:
            ib = [r for r in rows8b if r["arm"] == key]
            b, nb = sum(bool(r.get("attack_success")) for r in ib), len(ib)
            e["inside_spec_attack"] = [b, nb]
            bm = 100 * b / nb
            blo, bhi = cp(b, nb)
            lines += [rf"\draw[{color},line width=0.6pt] (axis cs:{um:.4g},{am:.4g}) -- (axis cs:{um:.4g},{bm:.4g});",
                      rf"\addplot[forget plot,only marks,mark={mark},mark size=2.3pt,draw={color},fill=white,error bars/.cd,y dir=both,y explicit,"
                      rf"error bar style={{{color},line width=0.45pt}}] coordinates {{({um:.4g},{bm:.4g}) += (0,{100 * bhi - bm:.4g}) -= (0,{bm - 100 * blo:.4g})}};"]
        st["arms"][key] = e
    if rows8b:
        lines.append(r"\node[font=\tiny,text=slate,anchor=north east,align=right] at (axis cs:103,103) "
                     r"{filled: AgentDojo's injections\\open: injections inside S};")
    lines.append(r"\end{axis}\end{tikzpicture}")
    return lines, st


def caption_c(s: dict, ckpt8: str) -> str:
    a = s["arms"]
    f = lambda k, m: pct(*a[k][m])
    word = lambda x, y: "fewer" if x[0] / x[1] < y[0] / y[1] else ("as many" if x[0] / x[1] == y[0] / y[1] else "more")
    text = (rf"\textbf{{(c)}} AgentDojo banking and Slack, {ckpt8}, AgentDojo's utility graders, every episode counted (95\% "
            rf"Clopper--Pearson). Permissions written without the reference solution (strict provenance) complete "
            rf"{f('authority_strict', 'benign_utility')} benign tasks, {word(a['authority_strict']['benign_utility'], a['unprotected']['benign_utility'])} "
            rf"than unprotected ({f('unprotected', 'benign_utility')}); the oracle grant, copied from the solution, completes "
            rf"{f('authority_oracle', 'benign_utility')}. AgentDojo's injections succeed in {f('authority_strict', 'native_attack')} strict "
            rf"episodes and {f('unprotected', 'native_attack')} unprotected")
    if "inside_spec_attack" in a["authority_strict"]:
        text += (rf"; injections that name only principals already in the user's state succeed in "
                 rf"{f('authority_strict', 'inside_spec_attack')} strict and {f('authority_delegated', 'inside_spec_attack')} delegated "
                 rf"episodes, against {f('unprotected', 'inside_spec_attack')} unprotected and {f('authority_oracle', 'inside_spec_attack')} "
                 r"under the oracle grant (author-written injections and graders, predeclared)")
    return text + ". "


# ------------------------------------------------------------------ E1 (own figure)

def panel_e1(campaigns: list[dict], ckpt: str) -> tuple[list[str], dict]:
    lines = [r"\begin{tikzpicture}[baseline]",
             rf"\begin{{axis}}[{AXIS},xmin=-3,xmax=103,ymin=-3,ymax=103,grid=major,clip=false,"
             r"xlabel={Task success (\% of planned campaigns)},ylabel={Unauthorized effect (\% of measured)},"
             r"title={Predeclared replication}]"]
    st = {"ckpt": ckpt, "controllers": {}}
    for key, label, color, mark in CONTROLLERS:
        rs = [r for r in campaigns if r["checkpoint"] == ckpt and r["controller"] == key]
        ms = [r for r in rs if r.get("status", "measured") == "measured"]
        if not ms:
            raise SystemExit(f"E1 has no measured campaigns for {key} at {ckpt}")
        s, n = sum(bool(r["task_success"]) for r in ms), len(rs)  # an unmeasured campaign is not a success
        u, nm = sum(bool(r["unauthorized"]) for r in ms), len(ms)
        (slo, shi), (ulo, uhi) = cp(s, n), cp(u, nm)
        sm, um = 100 * s / n, 100 * u / nm
        st["controllers"][key] = {"planned": n, "measured": nm, "task_success": s, "unauthorized": u}
        lines += [rf"\addplot[only marks,mark={mark},mark size=2.2pt,draw={color},fill={color},error bars/.cd,x dir=both,x explicit,y dir=both,y explicit,"
                  rf"error bar style={{{color},line width=0.5pt}}] coordinates {{({sm:.4g},{um:.4g}) += ({100 * shi - sm:.4g},{100 * uhi - um:.4g}) -= ({sm - 100 * slo:.4g},{um - 100 * ulo:.4g})}};",
                  rf"\node[font=\tiny,text=ink,anchor=south west,inner sep=1.5pt] at (axis cs:{sm:.4g},{um:.4g}) {{{label}}};"]
    return lines + [r"\end{axis}\end{tikzpicture}"], st


# ------------------------------------------------------------------ assembly

def build(args) -> dict:
    from e5_analysis_v6 import analyse
    trials, e4res = load_e4(args.e4)
    plan5_path = ROOT / "fixtures/v6/omitted_clause_plan.json"
    plan5 = json.loads(plan5_path.read_text())
    e5res = json.loads((args.e5 / "results.json").read_text())
    an = analyse([json.loads(l) for l in (args.e5 / "candidates.jsonl").read_text().splitlines() if l.strip()], plan5)
    e8res = json.loads((args.e8 / "results.json").read_text())
    rows8 = load_rows(args.e8)
    rows8b = load_rows(args.e8b) if args.e8b else None
    if not args.layout_preview:
        for name, res in (("E4", e4res), ("E5", e5res), ("E8", e8res)) + ((("E8b", json.loads((args.e8b / "results.json").read_text())),) if args.e8b else ()):
            if res.get("smoke_test"):
                raise SystemExit(f"smoke-test {name} evidence cannot build the hero")
        if e5res["plan_sha256"] != sha(plan5_path):
            raise SystemExit("E5 results do not match the current plan fixture")
        if e8res["plan_sha256"] != sha(ROOT / "fixtures/v6/agentdojo_fair_plan.json"):
            raise SystemExit("E8 results do not match the current plan fixture")
    a, sa = panel_a(trials)
    b, sb = panel_b(an, plan5, args.ckpt)
    c, sc = panel_c(rows8, rows8b)
    wm = []
    if args.layout_preview:
        wm = [r"\begin{tikzpicture}[remember picture,overlay]\node[rotate=18,font=\Huge\bfseries,text=notred,opacity=0.35] at (current page.center) {LAYOUT PREVIEW -- NOT EVIDENCE};\end{tikzpicture}"]
    ckpt8 = json.loads((ROOT / "fixtures/v6/agentdojo_fair_plan.json").read_text())["agent"]["checkpoint"].split("/")[-1].replace("-GGUF", "")
    caption = (r"\caption{\textbf{Halting a model stops new proposals, not effects already in flight. Only a fence at the "
               r"state-bearing sink stops those, and only for what the specification names.} "
               + caption_a(sa) + caption_b(sb, an["checkpoints"][args.ckpt]["integrity"]) + caption_c(sc, ckpt8)
               + r"Local quantized open-weight models on CPU, temperature 0 for review and agents; no frontier model.}\label{fig:v6hero}")
    tex = ["% Generated by scripts/hero_v6.py from promoted v6 evidence; do not edit.", r"\begin{figure*}[t]\centering"]
    tex += wm + a + [r"\hfill"] + b + [r"\hfill"] + c + [caption, r"\end{figure*}"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "v6-hero-fig.tex").write_text("\n".join(tex) + "\n")
    stats = {"a": sa, "b": sb, "c": sc, "layout_preview": args.layout_preview,
             "inputs": {"e4": str(args.e4), "e5": str(args.e5), "e8": str(args.e8), "e8b": str(args.e8b) if args.e8b else None}}
    (args.out / "v6-hero-stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True, default=list) + "\n")
    return stats


def preview(out: Path):
    pre = (ROOT / "docs/preamble.tex").read_text()
    keep = [r"\usepackage{xcolor,tikz,pgfplots,stfloats}"]
    keep += [l for l in pre.splitlines() if l.startswith((r"\usepackage", r"\usetikzlibrary", r"\usepgfplotslibrary", r"\pgfplotsset", r"\definecolor"))
             and not any(x in l for x in ("hyperref", "natbib", "fancyhdr", "titlesec", "tikz,pgfplots"))]
    doc = [r"\documentclass[10pt,twocolumn]{article}"] + keep + [r"\begin{document}", r"\input{v6-hero-fig.tex}", r"\null\clearpage", r"\end{document}"]
    (out / "preview.tex").write_text("\n".join(doc) + "\n")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "preview.tex"], cwd=out, check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run(["pdftoppm", "-png", "-r", "150", "-f", "1", "-l", "1", "preview.pdf", "preview"], cwd=out, check=True)
    subprocess.run(["pdftoppm", "-png", "-gray", "-r", "110", "-f", "1", "-l", "1", "preview.pdf", "preview-gray"], cwd=out, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e4", type=Path, required=True, help="E4 evidence directory (trials.jsonl) or raw results.json")
    ap.add_argument("--e5", type=Path, required=True, help="E5 directory with candidates.jsonl and results.json")
    ap.add_argument("--e8", type=Path, required=True, help="E8 directory with episodes.jsonl and results.json")
    ap.add_argument("--e8b", type=Path, help="E8b directory (injections inside S); adds the open markers to panel (c)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", default="Qwen/Qwen3-8B-GGUF", help="E5 checkpoint drawn in panel (b): the larger one")
    ap.add_argument("--layout-preview", action="store_true")
    ap.add_argument("--render", action="store_true", help="compile a standalone preview PDF/PNG (needs pdflatex)")
    args = ap.parse_args()
    if args.layout_preview and str(args.out.resolve()).startswith(str((ROOT / "docs").resolve())):
        raise SystemExit("a layout preview may not be written under docs/")
    build(args)
    if args.render:
        preview(args.out)
    print(json.dumps({"panels": ["a", "b", "c"], "layout_preview": args.layout_preview}))


if __name__ == "__main__":
    main()
