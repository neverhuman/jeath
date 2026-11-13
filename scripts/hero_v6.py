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
           ("authority_oracle", "Solution allowlist", "exoblue!60!black", "triangle"))
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
            # The plan's definition: authorized calls "cancelled or denied by the stop", counting every denial, including
            # those made while the enforcement call ran (the promoted cancelled_legitimate counted only cancellations).
            from revokebench_reanalysis_v6 import per_trial
            lost.append(per_trial(t)["legit_lost"])
    fin = {a: sorted(len(c) for c in cs) for a, cs in curves.items()}
    grid = {a: [t["metrics"]["post_enforcement_commits"] for t in nofault if t["config"]["arm"] == a] for a in fin}
    fence_all = [t for t in trials if t["config"]["arm"] == "fence_sink"]
    wl = sel[0]["config"]["workload"]
    st = {"rule": "stable cell (rho<1) with the longest mean tool latency, L=0", "w": w, "C": C, "rho": round(2 * w / C, 3),
          "lambda_offered": wl["calls_per_step"] / wl["model_step_seconds"],
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
    styles = {"halt_inference": "endored,thick,dashed", "revoke_broker": "warnamber!80!black,very thick,densely dotted",
              "fence_sink": "exoblue,ultra thick"}
    for arm in ("halt_inference", "revoke_broker", "fence_sink"):
        for ts in curves[arm]:
            pts = [(0.0, 0)] + [(x, i + 1) for i, x in enumerate(ts)] + [(xmax, len(ts))]
            lines.append(rf"\addplot[{styles[arm]},const plot,opacity=0.85] coordinates {{{' '.join(f'({x:.2f},{y})' for x, y in pts)}}};")
    lines += [
        rf"\node[font=\scriptsize,text=ink,anchor=south east,inner sep=1pt] at (axis cs:{xmax},{max(fin['halt_inference'])}) {{halt the model}};",
        rf"\node[font=\scriptsize,text=ink,anchor=south east,inner sep=1.5pt] at (axis cs:{xmax},{max(fin['revoke_broker'])}) {{revoke at the broker}};",
        rf"\node[font=\scriptsize,text=ink,anchor=south east,inner sep=1.5pt,align=right] at (axis cs:{xmax},0.4) "
        rf"{{\textbf{{fence the sink: {rng(fin['fence_sink'])}}}\\[-1pt]loses {rng(lost)} legitimate calls}};",
        r"\end{axis}\end{tikzpicture}"]
    return lines, st


def banner(s: dict) -> list[str]:
    """The three places a stop can act, with the drawn cell's counts from panel (a). Colours and dash styles match (a)."""
    f = s["final"]
    box = r"line width={lw},rounded corners=2.5pt,minimum width={w}cm,minimum height=1.2cm,align=center,inner sep=2.5pt,font=\scriptsize,text=ink"
    def node(name, x, w, lw, colour, fill, title, line):
        return (rf"\node[draw={colour},fill={fill},{box.format(lw=lw, w=w)}] ({name}) at ({x},0) "
                rf"{{{{\footnotesize\bfseries {title}}}\\[1pt]{line}}};")
    return [r"\begin{tikzpicture}[x=1cm,y=1cm,>=Latex]",
            node("s1", 2.75, 4.9, "0.8pt,dashed", "endored", "endored!6", "1\\enspace Halt the model",
                 rf"new proposals stop\\[-1pt]\textbf{{{rng(f['halt_inference'])}}} dispatched effects still commit"),
            node("s2", 8.95, 4.9, "0.8pt,densely dotted", "warnamber!80!black", "warnamber!8", "2\\enspace Revoke at the broker",
                 rf"new permits stop\\[-1pt]\textbf{{{rng(f['revoke_broker'])}}} admitted effects still commit"),
            node("s3", 15.15, 4.9, "1.2pt", "exoblue", "exoblue!7", "3\\enspace Fence the sink",
                 rf"stale epochs refused at commit\\[-1pt]\textbf{{{rng(f['fence_sink'])}}} commit; "
                 rf"\textbf{{{rng(s['fence_lost_legitimate'])}}} calls in flight discarded"),
            r"\draw[->,line width=1pt,draw=slate] (s1.east) -- node[above,font=\scriptsize,text=ink]{in flight} (s2.west);",
            r"\draw[->,line width=1pt,draw=slate] (s2.east) -- node[above,font=\scriptsize,text=ink]{admitted} (s3.west);",
            r"\end{tikzpicture}"]


def caption_a(s: dict) -> str:
    f = s["final"]
    fence = ("admits none after enforcement in any of its" if s["fence_post_max"] == 0 else f"admits at most {s['fence_post_max']} in its")
    return (rf"\textbf{{(a)}} Real broker and sink processes driven by a synthetic, non-adversarial agent emitting "
            rf"{s['lambda_offered']:g} calls/s to {s['C']} concurrent tool calls of {s['w']:g}\,s mean latency (chosen by a fixed rule: the "
            rf"stable cell with the longest latency), stop enforced at once, {s['replicates']} replicates. After the stop, "
            rf"{rng(f['halt_inference'])} effects commit when the model is halted and {rng(f['revoke_broker'])} when the broker "
            rf"revokes; the fence discards {rng(s['fence_lost_legitimate'])} legitimate calls in flight. Its zero is by design; "
            rf"across all cells it {fence} {s['fence_trials']} trials, {s['fence_fault_trials']} with the sink killed or late completions. ")


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
    for arm in ("R1", "R2"):
        st[f"{arm}_state"] = sets["state_dependent"]["arms"][arm]["breach_unique"]
        st[f"{arm}_visible"] = sets["proposal_visible"]["arms"][arm]["breach_unique"]
        st[f"{arm}_state_cases"] = sets["state_dependent"]["arms"][arm]["breach_cases"]
        st[f"{arm}_visible_cases"] = sets["proposal_visible"]["arms"][arm]["breach_cases"]
    st["G_breaches"] = sum(sets[f]["arms"]["G"]["breach_unique"]["x"] for f in order)
    sd, pv = st["R1_state"], st["R1_visible"]
    # H_E5a's direction, claimed in the title only when the 95% intervals separate.
    separated = sd["ci95"] and pv["ci95"] and sd["ci95"][0] > pv["ci95"][1]
    title = (r"\textbf{(b)} Without reasoning, reviewers miss state" if separated else
             r"\textbf{(b)} Unauthorized proposals admitted")
    st["title_claim_supported"] = bool(separated)
    lines = [r"\begin{tikzpicture}[baseline]",
             rf"\begin{{axis}}[{AXIS},xmin=-3,xmax=103,ymin=0.4,ymax={max(ypos.values()) + 0.6:.1f},"
             r"ytick={" + ",".join(f"{ypos[f]:.2f}" for f in order) + r"},yticklabels={" + ",".join("{" + FAMILY_LABEL[f] + "}" for f in order) + r"},"
             rf"xlabel={{Unauthorized proposals admitted (\%)}},title={{{title}}},"
             r"yticklabel style={font=\tiny,align=right},xmajorgrids,axis y line*=left,clip=false,"
             r"legend style={font=\tiny,at={(0.5,-0.34)},anchor=north,draw=none,fill=none,legend columns=1,row sep=-1.5pt},legend cell align=left]"]
    offs = {"R1": 0.25, "R2": 0.08, "G": -0.08, "omit": -0.25}
    for arm, color, mark, label in (("R1", "endored", "triangle*", "same-model reviewer, rules"),
                                    ("R2", "warnamber", "diamond", "same-model reviewer, rules + reasoning"),
                                    ("G", "exoblue", "*", "gate reading sink state")):
        pts = []
        for f in order:
            e = st["families"][f][arm]["breach_unique"]
            if not e["n"]:
                continue
            m = 100 * e["x"] / e["n"]
            lo, hi = (100 * v for v in e["ci95"]) if arm != "G" else (m, m)  # the gate's admissions are fixed by design: no interval
            pts.append(f"({m:.4g},{ypos[f] + offs[arm]:.3f}) -= ({m - lo:.4g},0) += ({hi - m:.4g},0)")
        fill = "white" if arm == "R2" else color  # open diamond: distinct from the filled triangle in grayscale
        lines.append(rf"\addplot[only marks,mark={mark},mark size=2pt,draw={color},fill={fill},line width=0.6pt,error bars/.cd,x dir=both,x explicit,"
                     rf"error bar style={{{color},line width=0.5pt}}] coordinates {{{' '.join(pts)}}};\addlegendentry{{{label}}}")
    pts = []
    for f in order:
        x, n = st["families"][f]["omitted_clause_share"]
        if n:
            pts.append(f"({100 * x / n:.4g},{ypos[f] + offs['omit']:.3f})")
    lines.append(r"\addplot[only marks,mark=o,mark size=1.4pt,draw=structgray!80] coordinates {" + " ".join(pts) + r"};"
                 r"\addlegendentry{gate with that clause unwritten (by construction)}")
    ysep = (min(ypos[f] for f in order if f in STATE) + max(ypos[f] for f in order if f in VISIBLE)) / 2
    lines += [rf"\draw[structgray!60,densely dashed] (axis cs:-3,{ysep:.2f}) -- (axis cs:103,{ysep:.2f});",
              rf"\node[font=\tiny\itshape,text=slate,anchor=south east] at (axis cs:103,{max(ypos.values()) + 0.25:.2f}) {{violation lives in state}};",
              rf"\node[font=\tiny\itshape,text=slate,anchor=north east] at (axis cs:103,{ysep - 0.05:.2f}) {{visible in the proposal}};",
              r"\end{axis}\end{tikzpicture}"]
    return lines, st


def review_tokens(path: Path, ckpt: str) -> dict:
    """Median generated tokens per review, each distinct valid proposal of the strategy-conditioned set once."""
    seen, toks = set(), {"R1": [], "R2": []}
    for line in path.read_text().splitlines():
        d = json.loads(line) if line.strip() else None
        if not d or d["checkpoint"] != ckpt or not d.get("valid") or d.get("sample_set") != "strategy":
            continue
        key = (d["case_id"], json.dumps(d["rpc"], sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        for arm in toks:
            toks[arm].append(d["reviews"][arm]["completion_tokens"])
    return {arm: median(v) for arm, v in toks.items()}


def exact_copy_tally(e5: Path, ckpt: str) -> dict:
    """State-dependent proposals that are byte-identical to an approved proposal in the case's policy, against the mutated
    ones; each distinct proposal of the strategy-conditioned set once. R2 replies cut off at the token cap are denials (the
    plan's rule). Post hoc: this split was made after the results were read (DISCLOSURES_V6.md)."""
    res = json.loads((e5 / "results.json").read_text())
    approved = {int(k): {json.dumps(p, sort_keys=True) for p in v["policy"]["proposals"]} for k, v in res["visible_states"].items()}
    seen, out = set(), {k: {"R1": 0, "R2": 0, "n": 0} for k in ("exact", "mutated")}
    for line in (e5 / "candidates.jsonl").read_text().splitlines():
        r = json.loads(line) if line.strip() else None
        if (not r or r["checkpoint"] != ckpt or not r["valid"] or r["sample_set"] != "strategy" or not r["violated"]
                or r["family"] not in STATE):
            continue
        key = (r["case_id"], json.dumps(r["rpc"], sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        kind = "exact" if json.dumps(r["rpc"]["proposal"], sort_keys=True) in approved[r["case_id"]] else "mutated"
        out[kind]["n"] += 1
        for arm in ("R1", "R2"):
            rv = r["reviews"][arm]
            out[kind][arm] += bool(rv["allow"]) and not (arm == "R2" and rv["finish_reason"] == "length")
    return out


def caption_b(s: dict, integ: dict) -> str:
    ck = s["ckpt"].split("/")[-1].replace("-GGUF", "")
    sd, pv, sd2, pv2 = s["R1_state"], s["R1_visible"], s["R2_state"], s["R2_visible"]
    # Comparative wording only when the 95% intervals separate (the same rule as the panel title); otherwise neutral.
    cmp = "more often than" if s["title_claim_supported"] else "against"
    tok = s["tokens"]
    return (rf"\textbf{{(b)}} {ck} red-team proposals under 8 predeclared mutation strategies, each distinct proposal counted once "
            rf"(95\% Clopper--Pearson). The reviewer is the same checkpoint, shown the rules and the state the gate reads. "
            rf"Without reasoning it admits {pct(sd['x'], sd['n'])} of unauthorized proposals whose violation lives in sink state, {cmp} "
            rf"{pct(pv['x'], pv['n'])} of those visible in the proposal. With reasoning (median {tok['R2']:g} generated tokens per "
            rf"review, against {tok['R1']:g}) it admits {pct(sd2['x'], sd2['n'])} and {pct(pv2['x'], pv2['n'])}. "
            rf"The gate admits {s['G_breaches']}; it enforces "
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
    # Counts only, no comparative words: 37 tasks at temperature 0 cannot separate most of these arms.
    text = (rf"\textbf{{(c)}} AgentDojo banking and Slack (37 user tasks), {ckpt8}, AgentDojo's graders, every episode counted "
            rf"(95\% Clopper--Pearson, episodes not independent within a task). A provenance specification written without the "
            rf"reference solutions (strict) completes {f('authority_strict', 'benign_utility')} benign tasks; unprotected, "
            rf"{f('unprotected', 'benign_utility')}; an allowlist copied from the solution, whose refusals can steer the agent "
            rf"toward it, {f('authority_oracle', 'benign_utility')}. AgentDojo's injections succeed in {f('authority_strict', 'native_attack')} "
            rf"strict and {f('unprotected', 'native_attack')} unprotected episodes")
    if "inside_spec_attack" in a["authority_strict"]:
        text += (rf"; injections that name only principals already in the user's state (written by us) succeed in "
                 rf"{f('authority_strict', 'inside_spec_attack')} strict, {f('authority_delegated', 'inside_spec_attack')} delegated, "
                 rf"{f('unprotected', 'inside_spec_attack')} unprotected and {f('authority_oracle', 'inside_spec_attack')} allowlist episodes")
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


# ------------------------------------------------------------------ hero panel (b): E4 across enforcement delays

ARM_STYLE = {"halt_inference": ("endored", "*", "halt the model"),
             "revoke_broker": ("warnamber!80!black", "square*", "revoke at the broker"),
             "fence_sink": ("exoblue", "triangle*", "fence the sink")}


def median(xs):
    xs = sorted(xs)
    k = len(xs) // 2
    return xs[k] if len(xs) % 2 else (xs[k - 1] + xs[k]) / 2


def panel_delay(trials: list[dict], w: float, C: int) -> tuple[list[str], dict]:
    """Effects committed after the stop decision, per arm and enforcement delay L, in the cell drawn in (a). The plan's
    own prediction for halting (metrics.little_law_prediction, lambda measured x (L + w)) is drawn beside it; the fence's
    lambda x L is a post hoc reading (the delay term no layer removes) and is labelled so."""
    from revokebench_reanalysis_v6 import per_trial
    cell = [t for t in trials if t["config"]["fault"] == "none" and t["config"]["w"] == w and t["config"]["C"] == C]
    Ls = sorted({t["config"]["L"] for t in cell})
    pts = {a: {L: [] for L in Ls} for a in ARM_STYLE}
    lams, backlog0 = [], []
    for t in cell:
        p = per_trial(t)
        pts[t["config"]["arm"]][t["config"]["L"]].append(p["post_decision_commits"])
        if p["lambda_measured_plan_window"]:
            lams.append(p["lambda_measured_plan_window"])
        if t["config"]["arm"] == "halt_inference" and t["config"]["L"] == min(Ls):
            backlog0.append(p["backlog_at_decision"])
    burst = cell[0]["config"]["workload"]["calls_per_step"]
    lam = sum(lams) / len(lams)
    ratios = []
    for t in trials:
        if t["config"]["fault"] == "none" and t["config"]["arm"] == "halt_inference":
            p = per_trial(t)
            if p["little_law_plan"]:
                ratios.append(p["post_decision_commits"] / p["little_law_plan"])
    ymax = max(max(v) for d in pts.values() for v in d.values()) * 1.12
    n = len(Ls)
    lines = [r"\begin{tikzpicture}[baseline]",
             rf"\begin{{axis}}[{AXIS},xmin=-0.5,xmax={n - 0.5},ymin=0,ymax={ymax:.0f},xtick={{{','.join(str(i) for i in range(n))}}},"
             rf"xticklabels={{{','.join(f'{L:g}' for L in Ls)}}},ymajorgrids,clip=false,"
             r"xlabel={Delay before the stop takes effect, $L$ (s)},ylabel={Effects committed after the decision},"
             r"title={\textbf{(b)} After the decision: $\lambda L$ + $\lambda w$}," + LEGEND_C + "]"]
    pred = {"halt_inference": lambda L: lam * (L + w), "fence_sink": lambda L: lam * L}
    off = {"halt_inference": -0.2, "revoke_broker": 0.0, "fence_sink": 0.2}
    # Predictions are marks at each delay, not lines: L is categorical here. The plan's halting prediction is solid; the
    # fence's lambda*L reading, made after the data, is dotted.
    for arm, f in pred.items():
        col = ARM_STYLE[arm][0]
        dash = "" if arm == "halt_inference" else ",dash pattern=on 0.8pt off 0.8pt"
        lines.append(rf"\addplot[only marks,mark=-,mark size=5pt,mark options={{{col},line width=0.8pt{dash}}},forget plot] "
                     rf"coordinates {{{' '.join(f'({i + off[arm]:.2f},{f(L):.2f})' for i, L in enumerate(Ls))}}};")
    lines += [rf"\node[font=\tiny,text=ink,anchor=east,inner sep=1pt] at (axis cs:{n - 1 + off['halt_inference'] - 0.22},{pred['halt_inference'](Ls[-1]):.2f}) {{$\lambda(L{{+}}w)$}};",
              rf"\node[font=\tiny,text=ink,anchor=west,inner sep=1pt] at (axis cs:{n - 1 + off['fence_sink'] + 0.22},{pred['fence_sink'](Ls[-1]):.2f}) {{$\lambda L$}};"]
    st = {"w": w, "C": C, "L": Ls, "lambda_mean": round(lam, 3), "points": {}}
    for arm, (col, mark, lbl) in ARM_STYLE.items():
        cs = []
        for i, L in enumerate(Ls):
            v = sorted(pts[arm][L])
            m = median(v)
            cs.append(f"({i + off[arm]:.2f},{m:g}) -= (0,{m - v[0]:g}) += (0,{v[-1] - m:g})")
        st["points"][arm] = {f"{L:g}": sorted(pts[arm][L]) for L in Ls}
        lines.append(rf"\addplot[only marks,mark={mark},mark size=1.8pt,draw={col},fill={col},error bars/.cd,y dir=both,y explicit,"
                     rf"error bar style={{{col}}}] coordinates {{{' '.join(cs)}}};\addlegendentry{{{lbl}}}")
    lines.append(r"\end{axis}\end{tikzpicture}")
    st |= {"burst": burst, "burst_at_decision": set(backlog0) == {burst},
           "halt_ratio_median": round(median(ratios), 2), "halt_ratio_range": [round(min(ratios), 2), round(max(ratios), 2)],
           "halt_trials": len(ratios)}
    return lines, st


def caption_delay(s: dict) -> str:
    lo, hi = s["halt_ratio_range"]
    burst = (rf" At \(L = {s['L'][0]:g}\) halting exceeds \(\lambda w\) because the decision falls just after the agent emits a "
             rf"burst of {s['burst']} calls, which a halt cannot recall." if s["burst_at_decision"] else "")
    return (rf"\textbf{{(b)}} The same cell at every delay \(L\) between the decision and enforcement (median and range). Solid "
            rf"bars: \(\lambda(L+w)\), the plan's prediction for halting; dotted: \(\lambda L\), a post hoc reading of the part a "
            rf"fence cannot remove. Over all {s['halt_trials']} stable halting trials the count was "
            rf"{s['halt_ratio_median']:g}\(\times\) the prediction (median; {lo:g}--{hi:g})." + burst + " ")


# ------------------------------------------------------------------ hero panel (c): E6b fenceability

SURF_SHORT = {"postgresql16": "PostgreSQL 16", "etcd_v3_5_17": "etcd 3.5", "posix_fs": "POSIX filesystem"}
HERO_MECH = {"pg_trigger_e6": "epoch trigger", "pg_trigger_e6_fair": "trigger + table lock", "pg_trigger_notnull": "trigger, epoch required",
             "pg_foreign_key": "foreign key to epoch", "pg_revoke": "REVOKE writer role", "etcd_txn_compare": "txn compare on floor",
             "etcd_rbac_revoke": "RBAC revoke", "fs_flock": "flock + floor file", "fs_dir_mode": "directory mode bits"}
FENCE_COLS = ("surface enforces", "forged epoch refused", "no stale write")


def load_e6b(path: Path) -> dict:
    return json.loads((path / "results.json").read_text())["summary"]


def panel_fence(summ: dict) -> tuple[list[str], dict]:
    """Per mechanism, three properties from E6b's predeclared probes: the class assigned by the plan's rule (surface-native
    is filled, author code half), the share of forged-epoch writes refused, and the share of idle races with no stale
    effect ordered after the fence. Combining them is a reading, not a predeclared test."""
    det, idle = summ["deterministic"], summ["race"]["idle"]
    rows = []
    for m, v in det.items():
        r = idle[m]
        c1 = None if v["class"] == "indeterminate" else {"surface_native": 1.0, "author_code_in_surface": 0.5}.get(v["class"], 0.0)
        c2 = v["forge_refused"] / v["runs"] if v["runs"] else None
        ev = r["events"]["violation_free"]
        c3 = None if r.get("fence_failed") or not ev["n"] else ev["k"] / ev["n"]
        rows.append({"mechanism": m, "surface": v["surface"], "cells": [c1, c2, c3], "class": v["class"],
                     "forged": [v["forge_refused"], v["runs"]], "no_stale": [ev["k"], ev["n"]]})
    all3 = [r["mechanism"] for r in rows if all(c == 1.0 for c in r["cells"])]
    h, xs = 0.23, (0.45, 1.15, 1.85)
    items, last = [], None
    for r in rows:
        if r["surface"] != last:
            items.append(("head", SURF_SHORT.get(r["surface"], r["surface"])))
            last = r["surface"]
        items.append(("row", r))
    top = (len(items) - 1) * h
    lines = [r"\begin{tikzpicture}[baseline]"]
    for j, name in enumerate(FENCE_COLS):
        lines.append(rf"\node[font=\tiny,text=ink,rotate=40,anchor=south west,inner sep=0.5pt] at ({xs[j] - 0.08},{top + 0.18:.2f}) {{{name}}};")
    for i, (kind, it) in enumerate(items):
        y = top - i * h
        if kind == "head":
            lines.append(rf"\node[font=\tiny\bfseries,text=ink,anchor=east] at (0.1,{y:.2f}) {{{it}}};")
            continue
        if it["mechanism"] in all3:
            lines.append(rf"\fill[exoblue!12,rounded corners=1pt] (-2.75,{y - h / 2 + 0.02:.2f}) rectangle ({xs[-1] + 0.3},{y + h / 2 - 0.02:.2f});")
        lines.append(rf"\node[font=\tiny,text=ink,anchor=east] at (0.1,{y:.2f}) {{{HERO_MECH.get(it['mechanism'], it['mechanism'])}}};")
        for x, c in zip(xs, it["cells"]):
            if c is None:
                lines.append(rf"\node[font=\tiny,text=slate] at ({x},{y:.2f}) {{--}};")
            elif c == 1.0:
                lines.append(rf"\fill[exoblue] ({x},{y:.2f}) circle (2.1pt);")
            elif c == 0.0:
                lines.append(rf"\draw[structgray,line width=0.5pt] ({x},{y:.2f}) circle (2.1pt);")
            else:
                lines.append(rf"\draw[exoblue,line width=0.5pt] ({x},{y:.2f}) circle (2.1pt);"
                             rf"\fill[exoblue] ({x},{y:.2f}) ++(0,-2.1pt) arc[start angle=-90,end angle=90,radius=2.1pt] -- cycle;")
    # Title at the height pgfplots gives (a) and (b): axis height 3.7 cm plus its 6 pt title shift (all share a baseline).
    lines.append(r"\node[font=\footnotesize,anchor=south,align=center] at (-0.2,3.91) {\textbf{(c)} Where the in-flight term can be closed};")
    lines.append(r"\end{tikzpicture}")
    return lines, {"rows": rows, "all_three": all3}


def caption_fence(s: dict) -> str:
    name = lambda m: next(f"{SURF_SHORT[r['surface']].split(' ')[0]}'s {HERO_MECH[m]}" for r in s["rows"] if r["mechanism"] == m)
    a3 = s["all_three"]
    who = ("No mechanism has all three" if not a3 else
           f"Only {series([name(m) for m in a3])} {'is' if len(a3) == 1 else 'are'} enforced by the surface, refuse"
           f"{'s' if len(a3) == 1 else ''} a forged epoch, and let{'s' if len(a3) == 1 else ''} no stale write commit after the fence")
    runs = sorted({r["forged"][1] for r in s["rows"]})
    races = sorted({r["no_stale"][1] for r in s["rows"] if r["no_stale"][1]})
    span = lambda xs: str(xs[0]) if len(xs) == 1 else f"{xs[0]}--{xs[-1]}"
    bound = (rf" With no stale write in {races[0]} races, the per-race failure rate is below "
             rf"{100 * (1 - 0.05 ** (1 / races[0])):.0f}\% (one-sided 95\%)." if len(races) == 1 and a3 else "")
    return (rf"\textbf{{(c)}} Executable probes on single-node instances ({span(runs)} deterministic runs and {span(races)} races "
            r"per mechanism). Columns: the surface itself enforces the fence, even for a writer that omits every fence field; a "
            r"stale writer claiming the new epoch is refused; no stale write commits after the fence takes effect, in races "
            r"whose writers follow the mechanism's protocol. Filled, in every run; open, in none; half, by a trigger we "
            r"installed; dash, not measurable. " + who + "." + bound + " ")


def series(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


# ------------------------------------------------------------------ assembly

def preview_mark(on: bool) -> list[str]:
    return [r"\begin{tikzpicture}[remember picture,overlay]\node[rotate=18,font=\Huge\bfseries,text=notred,opacity=0.35] at "
            r"(current page.center) {LAYOUT PREVIEW -- NOT EVIDENCE};\end{tikzpicture}"] if on else []


def rel(p: Path) -> str:
    return str(p.resolve().relative_to(ROOT)) if p.resolve().is_relative_to(ROOT) else str(p)


def build_hero(args) -> dict:
    """Page-1 figure: E4 and E6b only, both about stopping."""
    trials, e4res = load_e4(args.e4)
    if not args.layout_preview and e4res.get("smoke_test"):
        raise SystemExit("smoke-test E4 evidence cannot build the hero")
    summ6 = load_e6b(args.e6b)
    a, sa = panel_a(trials)
    b, sb = panel_delay(trials, sa["w"], sa["C"])
    c, sc = panel_fence(summ6)
    caption = (r"\captionof{figure}{\textbf{Stopping a model does not stop its effects.} " + caption_a(sa) + caption_delay(sb)
               + caption_fence(sc) + r"}\label{fig:v6hero}")
    # Not a float: the hero sits under the abstract inside the two-column header, as the previous version's did.
    tex = ["% Generated by scripts/hero_v6.py from promoted v6 evidence; do not edit.",
           r"\begin{center}\begin{minipage}{\textwidth}\centering"]
    tex += preview_mark(args.layout_preview) + banner(sa) + [r"\par\vspace{4pt}"] + a + [r"\hfill"] + b + [r"\hfill"] + c
    tex += [r"\par\vspace{2pt}", caption, r"\end{minipage}\end{center}"]
    stats = {"a": sa, "b": sb, "c": sc, "layout_preview": args.layout_preview, "inputs": {"e4": rel(args.e4), "e6b": rel(args.e6b)}}
    return {"v6-hero-fig.tex": "\n".join(tex) + "\n",
            "v6-hero-stats.json": json.dumps(stats, indent=1, sort_keys=True, default=list) + "\n",
            "v6-hero-macros.tex": hero_macros(sa, sb, sc)}


def build_spec(args) -> dict:
    """Body figure: E5 (reviewers against state) and E8/E8b (what a written specification costs and misses)."""
    from e5_analysis_v6 import analyse
    plan5_path = ROOT / "fixtures/v6/omitted_clause_plan.json"
    plan5 = json.loads(plan5_path.read_text())
    e5res = json.loads((args.e5 / "results.json").read_text())
    an = analyse([json.loads(l) for l in (args.e5 / "candidates.jsonl").read_text().splitlines() if l.strip()], plan5)
    e8res = json.loads((args.e8 / "results.json").read_text())
    rows8 = load_rows(args.e8)
    rows8b = load_rows(args.e8b) if args.e8b else None
    if not args.layout_preview:
        for name, res in (("E5", e5res), ("E8", e8res)) + ((("E8b", json.loads((args.e8b / "results.json").read_text())),) if args.e8b else ()):
            if res.get("smoke_test"):
                raise SystemExit(f"smoke-test {name} evidence cannot build the figure")
        if e5res["plan_sha256"] != sha(plan5_path):
            raise SystemExit("E5 results do not match the current plan fixture")
        if e8res["plan_sha256"] != sha(ROOT / "fixtures/v6/agentdojo_fair_plan.json"):
            raise SystemExit("E8 results do not match the current plan fixture")
    b, sb = panel_b(an, plan5, args.ckpt)
    sb["tokens"] = review_tokens(args.e5 / "candidates.jsonl", args.ckpt)
    sb["exact"] = exact_copy_tally(args.e5, args.ckpt)
    # The text says the unaltered proposal was submitted in "all" state-dependent cases; fail if that stops being true.
    n_state = len({json.loads(l)["case_id"] for l in (args.e5 / "candidates.jsonl").read_text().splitlines()
                   if l.strip() and json.loads(l)["family"] in STATE})
    if sb["exact"]["exact"]["n"] != n_state:
        raise SystemExit(f"unaltered approved proposal submitted in {sb['exact']['exact']['n']} of {n_state} state cases; "
                         "revise the 'all' wording in the abstract and Section 5")
    other = next(c for c in an["checkpoints"] if c != args.ckpt)
    sb["exact_other"], sb["other_ckpt"] = exact_copy_tally(args.e5, other), other
    c, sc = panel_c(rows8, rows8b)
    relabel = lambda xs: [x.replace(r"\textbf{(b)}", r"\textbf{(a)}").replace(r"\textbf{(c)}", r"\textbf{(b)}") for x in xs]
    integ = an["checkpoints"][args.ckpt]["integrity"]
    ckpt8 = json.loads((ROOT / "fixtures/v6/agentdojo_fair_plan.json").read_text())["agent"]["checkpoint"].split("/")[-1].replace("-GGUF", "")
    cap = relabel([caption_b(sb, integ) + caption_c(sc, ckpt8)])[0]
    caption = (r"\caption{\textbf{A same-model reviewer misses violations in state unless it reasons at length; a written specification stops what it names, at a cost.} " + cap
               + r"Local quantized open-weight models on CPU at temperature 0; no frontier model.}\label{fig:v6spec}")
    tex = ["% Generated by scripts/hero_v6.py from promoted v6 evidence; do not edit.", r"\begin{figure*}[t]\centering"]
    tex += preview_mark(args.layout_preview) + relabel(b) + [r"\hspace{1.4cm}"] + relabel(c) + [caption, r"\end{figure*}"]
    stats = {"reviewers": sb, "cost": sc, "layout_preview": args.layout_preview,
             "inputs": {"e5": rel(args.e5), "e8": rel(args.e8), "e8b": rel(args.e8b) if args.e8b else None}}
    return {"v6-spec-fig.tex": "\n".join(tex) + "\n",
            "v6-spec-stats.json": json.dumps(stats, indent=1, sort_keys=True, default=list) + "\n",
            "v6-spec-macros.tex": spec_macros(sb, sc, integ)}


def build(args) -> dict:
    out = build_hero(args)
    if getattr(args, "e5", None) and getattr(args, "e8", None):
        out |= build_spec(args)
    return out


def write_macros(m: dict) -> str:
    assert all(k.isalpha() for k in m)
    return "% Generated by scripts/hero_v6.py; do not edit.\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(m.items()))


def hero_macros(sa: dict, sb: dict, sc: dict) -> str:
    """The text's numbers come from the same statistics the figure draws, so prose and figure cannot disagree."""
    f = sa["final"]
    Lmax = f"{sb['L'][-1]:g}"
    m = {"vsixStopCalls": sa["C"], "vsixStopLatency": f"{sa['w']:g}", "vsixHaltRange": rng(f["halt_inference"]),
         "vsixRevokeRange": rng(f["revoke_broker"]), "vsixFenceRange": rng(f["fence_sink"]),
         "vsixLastCommitSeconds": f"{sa['last_commit_s']:g}", "vsixFenceLost": rng(sa["fence_lost_legitimate"]),
         "vsixGridHaltMax": sa["grid_max"]["halt_inference"], "vsixGridRevokeMax": sa["grid_max"]["revoke_broker"],
         "vsixFenceTrials": sa["fence_trials"], "vsixFencePostMax": sa["fence_post_max"], "vsixFenceFaultTrials": sa["fence_fault_trials"],
         "vsixStopReplicates": sa["replicates"], "vsixDelayMax": Lmax, "vsixDelayLambda": f"{sb['lambda_mean']:.1f}",
         "vsixDelayHaltAtMax": rng(sb["points"]["halt_inference"][Lmax]), "vsixDelayFenceAtMax": rng(sb["points"]["fence_sink"][Lmax]),
         "vsixLittleRatioMedian": f"{sb['halt_ratio_median']:g}", "vsixLittleRatioRange": f"{sb['halt_ratio_range'][0]:g}--{sb['halt_ratio_range'][1]:g}",
         "vsixLittleTrials": sb["halt_trials"], "vsixFenceMechanisms": len(sc["rows"]), "vsixFenceAllThreeCount": len(sc["all_three"]),
         "vsixFenceSurfaces": len({r["surface"] for r in sc["rows"]}),
         "vsixFenceAllThree": series([next(f"{SURF_SHORT[r['surface']].split(' ')[0]}'s {HERO_MECH[x]}" for r in sc["rows"] if r["mechanism"] == x)
                                      for x in sc["all_three"]]) if sc["all_three"] else "none"}
    return write_macros(m)


def spec_macros(sb: dict, sc: dict, integ: dict) -> str:
    a = sc["arms"]
    frac = lambda xn: f"{xn[0]}/{xn[1]}"
    m = {"vsixReviewerState": frac((sb["R1_state"]["x"], sb["R1_state"]["n"])),
         "vsixReviewerVisible": frac((sb["R1_visible"]["x"], sb["R1_visible"]["n"])),
         "vsixReasonerState": frac((sb["R2_state"]["x"], sb["R2_state"]["n"])),
         "vsixReasonerVisible": frac((sb["R2_visible"]["x"], sb["R2_visible"]["n"])),
         "vsixReviewerStateCases": frac((sb["R1_state_cases"]["x"], sb["R1_state_cases"]["n"])),
         "vsixReviewerVisibleCases": frac((sb["R1_visible_cases"]["x"], sb["R1_visible_cases"]["n"])),
         "vsixReasonerStateCases": frac((sb["R2_state_cases"]["x"], sb["R2_state_cases"]["n"])),
         "vsixReasonerTokens": f"{sb['tokens']['R2']:g}", "vsixReviewerTokens": f"{sb['tokens']['R1']:g}",
         "vsixExactCases": sb["exact"]["exact"]["n"],
         "vsixReviewerExact": sb["exact"]["exact"]["R1"], "vsixReasonerExact": sb["exact"]["exact"]["R2"],
         "vsixReviewerMutated": frac((sb["exact"]["mutated"]["R1"], sb["exact"]["mutated"]["n"])),
         "vsixReasonerMutated": frac((sb["exact"]["mutated"]["R2"], sb["exact"]["mutated"]["n"])),
         "vsixOtherCkpt": sb["other_ckpt"].split("/")[-1].replace("-GGUF", ""),
         "vsixReviewerExactOther": frac((sb["exact_other"]["exact"]["R1"], sb["exact_other"]["exact"]["n"])),
         "vsixReasonerExactOther": frac((sb["exact_other"]["exact"]["R2"], sb["exact_other"]["exact"]["n"])),
         "vsixReviewerMutatedOther": frac((sb["exact_other"]["mutated"]["R1"], sb["exact_other"]["mutated"]["n"])),
         "vsixReasonerMutatedOther": frac((sb["exact_other"]["mutated"]["R2"], sb["exact_other"]["mutated"]["n"])),
         "vsixGateBreaches": sb["G_breaches"], "vsixGateUnique": integ["unique_valid"],
         "vsixMirrorDisagreements": integ["mirror_gate_disagreements"],
         "vsixStrictUtility": frac(a["authority_strict"]["benign_utility"]), "vsixUnprotectedUtility": frac(a["unprotected"]["benign_utility"]),
         "vsixOracleUtility": frac(a["authority_oracle"]["benign_utility"]), "vsixStrictNative": frac(a["authority_strict"]["native_attack"]),
         "vsixUnprotectedNative": frac(a["unprotected"]["native_attack"])}
    if "inside_spec_attack" in a["authority_strict"]:
        m |= {"vsixStrictInside": frac(a["authority_strict"]["inside_spec_attack"]),
              "vsixDelegatedInside": frac(a["authority_delegated"]["inside_spec_attack"]),
              "vsixUnprotectedInside": frac(a["unprotected"]["inside_spec_attack"]),
              "vsixOracleInside": frac(a["authority_oracle"]["inside_spec_attack"])}
    return write_macros(m)


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
    ap.add_argument("--e6b", type=Path, required=True, help="E6b evidence directory (results.json)")
    ap.add_argument("--e5", type=Path, help="E5 directory with candidates.jsonl and results.json (body figure)")
    ap.add_argument("--e8", type=Path, help="E8 directory with episodes.jsonl and results.json (body figure)")
    ap.add_argument("--e8b", type=Path, help="E8b directory (injections inside S); adds the open markers to the body figure")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", default="Qwen/Qwen3-8B-GGUF", help="E5 checkpoint drawn in the body figure: the larger one")
    ap.add_argument("--layout-preview", action="store_true")
    ap.add_argument("--render", action="store_true", help="compile a standalone preview PDF/PNG of the hero (needs pdflatex)")
    args = ap.parse_args()
    if args.layout_preview and str(args.out.resolve()).startswith(str((ROOT / "docs").resolve())):
        raise SystemExit("a layout preview may not be written under docs/")
    if bool(args.e5) != bool(args.e8):
        raise SystemExit("--e5 and --e8 go together (the body figure needs both)")
    args.out.mkdir(parents=True, exist_ok=True)
    files = build(args)
    for fn, text in files.items():
        (args.out / fn).write_text(text)
    if args.render:
        preview(args.out)
    print(json.dumps({"files": sorted(files), "layout_preview": args.layout_preview}))


if __name__ == "__main__":
    main()
