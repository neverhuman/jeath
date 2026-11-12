#!/usr/bin/env python3
"""Front figure (hero) for "Intelligence Is Not Authority", built only from measured E4/E5/E1 evidence.

Every number in the figure and its caption comes from the input JSON; a missing or smoke/fake input is a hard
error unless --layout-preview is given, in which case every panel carries a "LAYOUT PREVIEW - NOT EVIDENCE"
watermark and the output must never be committed under docs/.

  (a) E4  effects committed after the stop took effect, by stop layer, vs mean tool latency (real processes)
  (b) E5  share of unauthorized proposals admitted, by attack family: learned reviewers vs verified gate vs
          the same gate with that family's clause omitted
  (c) E1  usefulness vs unauthorized effects per controller (powered predeclared replication)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STATE = ("duplicate_replay", "withdrawn_issuer", "stale_revision")
VISIBLE = ("altered_payload", "injected_exfiltration")
FAMILY_LABEL = {"duplicate_replay": "Replay, budget spent", "withdrawn_issuer": "Issuer withdrawn",
                "stale_revision": "Stale revision", "altered_payload": "Altered payload",
                "injected_exfiltration": "Exfiltration"}
CONTROLLERS = (("none", "No mediation", "structgray", "square*"), ("stateless", "Per-call policy", "warnamber", "diamond*"),
               ("learned_same_checkpoint", "Same-model reviewer", "endored", "triangle*"),
               ("durable_without_fence", "Durable, no fence", "safegreen", "pentagon*"),
               ("complete", "Fenced sink", "exoblue", "*"))
MACRO_RE = re.compile(r"^[A-Za-z]+$")


def cp(x: int, n: int, alpha: float = 0.05):
    """Exact Clopper-Pearson interval (same bisection as omitted_clause_v6.py)."""
    def tail(k, p):
        if k <= 0:
            return 1.0
        if p <= 0:
            return 0.0
        if p >= 1:
            return 1.0
        return min(1.0, sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                                     + i * math.log(p) + (n - i) * math.log1p(-p)) for i in range(k, n + 1)))

    def bis(pred):
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if pred(mid) else (mid, hi)
        return (lo + hi) / 2
    lo = 0.0 if x == 0 else bis(lambda p: tail(x, p) >= alpha / 2)
    hi = 1.0 if x == n else bis(lambda p: 1 - tail(x + 1, p) <= alpha / 2)
    return lo, hi


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# ------------------------------------------------------------------ panel (a)

def panel_a(e4: dict, run: Path, C: int, L: float, w: float) -> tuple[list[str], dict]:
    """Timeline: cumulative effects committed after the stop was enforced, every replicate drawn (step lines)."""
    sel = [t for t in e4["trials"] if t["config"]["fault"] == "none" and t["config"]["C"] == C
           and t["config"]["L"] == L and t["config"]["w"] == w]
    curves, finals, cancelled, tmax = {}, {}, [], 0.0
    for t in sel:
        tr = json.loads((run / "trials" / t["config"]["trial_id"] / "trial.json").read_text())
        ev = tr["events"]
        t_enf = next(e["t"] for e in ev if e["kind"] == "enforced")
        ts = sorted(e["t"] - t_enf for e in ev if e["kind"] == "completed" and e["status"] == "committed" and e["t"] > t_enf)
        if len(ts) != t["metrics"]["post_enforcement_commits"]:
            raise SystemExit(f"{t['config']['trial_id']}: event log does not reproduce post_enforcement_commits")
        curves.setdefault(t["config"]["arm"], []).append(ts)
        tmax = max(tmax, ts[-1] if ts else 0.0)
        if t["config"]["arm"] == "fence_sink":
            cancelled.append(t["metrics"]["cancelled_legitimate"])
    for arm, cs in curves.items():
        finals[arm] = sorted(len(c) for c in cs)
    all_fence = [t["metrics"]["post_enforcement_commits"] for t in e4["trials"] if t["config"]["arm"] == "fence_sink"]
    faults = [t for t in e4["trials"] if t["config"]["fault"] != "none" and t["config"]["arm"] == "fence_sink"]
    stats = {"trials": len(e4["trials"]), "C": C, "L": L, "w": w, "replicates": len(curves["fence_sink"]),
             "halt_final": finals["halt_inference"], "revoke_final": finals["revoke_broker"], "fence_final": finals["fence_sink"],
             "fence_cancelled": sorted(cancelled), "last_commit_s": round(tmax, 1),
             "fence_trials": len(all_fence), "fence_max_post": max(all_fence), "fault_trials": len(faults),
             "duplicates": sum(t["metrics"]["duplicate_effects"] for t in e4["trials"])}
    xmax = math.ceil(tmax * 1.08 / 5) * 5
    ymax = max(finals["halt_inference"] + finals["revoke_broker"]) * 1.25 + 1
    lines = [r"\begin{tikzpicture}[baseline]",
             rf"\begin{{axis}}[scale only axis,width=3.75cm,height=3.7cm,xmin=0,xmax={xmax},ymin=-0.6,ymax={ymax:.0f},"
             r"xlabel={Seconds after the stop took effect},ylabel={Effects committed after the stop},"
             r"title={\textbf{(a)} Stopping the model $\neq$ stopping its effects},tick label style={font=\scriptsize},"
             r"label style={font=\footnotesize},title style={font=\footnotesize},grid=major,clip=false]"]
    styles = {"halt_inference": "endored,thick,dashed", "revoke_broker": "warnamber!80!black,thick,dashdotted",
              "fence_sink": "exoblue,ultra thick"}
    for arm in ("halt_inference", "revoke_broker", "fence_sink"):
        for ts in curves[arm]:
            pts = [(0.0, 0)] + [(x, i + 1) for i, x in enumerate(ts)] + [(xmax, len(ts))]
            lines.append(rf"\addplot[{styles[arm]},const plot,opacity=0.85] coordinates {{{' '.join(f'({x:.2f},{y})' for x, y in pts)}}};")
    hm, rm = median(finals["halt_inference"]), median(finals["revoke_broker"])
    lines += [
        rf"\node[font=\scriptsize,text=ink,anchor=south east,inner sep=1pt] at (axis cs:{xmax},{max(finals['halt_inference'])}) {{halt the model}};",
        rf"\node[font=\scriptsize,text=ink,anchor=north east,inner sep=1.5pt] at (axis cs:{xmax},{min(finals['revoke_broker'])}) {{revoke at the broker}};",
        rf"\node[font=\scriptsize,text=ink,anchor=south east,inner sep=1.5pt,align=right] at (axis cs:{xmax},0.4) "
        rf"{{\textbf{{fence the sink: 0}}\\[-1pt]cancels {min(cancelled)}--{max(cancelled)} in-flight calls}};",
        r"\end{axis}\end{tikzpicture}"]
    stats["halt_median"], stats["revoke_median"] = hm, rm
    wmax = max(t["config"]["w"] for t in e4["trials"])
    alt = {}
    for t in e4["trials"]:
        c = t["config"]
        if c["fault"] == "none" and c["C"] == C and c["L"] == L and c["w"] == wmax:
            alt.setdefault(c["arm"], []).append(t["metrics"]["post_enforcement_commits"])
    stats["wmax"] = wmax
    stats["wmax_final"] = {k: sorted(v) for k, v in alt.items()}
    return lines, stats


# ------------------------------------------------------------------ panel (b)

def panel_b(e5: dict, plan: dict, ckpt: str) -> tuple[list[str], dict]:
    fams = plan["cases"]["families"]
    block = e5["aggregates"][ckpt]["sets"]["strategy"]
    order = list(STATE) + list(VISIBLE)
    # The strongest reviewer (rules stated + reasoning) is drawn; R0/R1 are in the appendix figure and stats.
    arms = (("R2", "endored", "triangle*", "same-model reviewer"), ("G", "exoblue", "*", "verified gate"),
            ("Gminus", "exoblue", "o", "gate, that clause omitted"))
    all_arms = ("R0", "R1", "R2", "G", "Gminus")
    ypos = {f: len(order) - i + (0.4 if f in STATE else 0) for i, f in enumerate(order)}
    offs = {"R2": 0.17, "G": 0.0, "Gminus": -0.17}
    lines = [r"\begin{tikzpicture}[baseline]",
             rf"\begin{{axis}}[scale only axis,width=3.75cm,height=3.7cm,xmin=-3,xmax=103,ymin=0.4,ymax={max(ypos.values()) + 0.6:.1f},"
             r"ytick={" + ",".join(f"{ypos[f]:.2f}" for f in order) + r"},yticklabels={" + ",".join("{" + FAMILY_LABEL[f] + "}" for f in order) + r"},"
             r"xlabel={Unauthorized proposals admitted (\%)},title={\textbf{(b)} Reviewers read proposals, not state},"
             r"tick label style={font=\scriptsize},yticklabel style={font=\tiny,align=right},label style={font=\footnotesize},"
             r"title style={font=\footnotesize},xmajorgrids,axis y line*=left,clip=false,"
             r"legend style={font=\tiny,at={(0.5,-0.34)},anchor=north,draw=none,fill=none,legend columns=1,row sep=-1.5pt},legend cell align=left]"]
    stats = {"ckpt": ckpt, "families": {}}
    for f in order:
        n = block[f]["unauthorized"]
        for arm in all_arms:
            key = f"G-{fams[f]['target_clause']}" if arm == "Gminus" else arm
            stats["families"].setdefault(f, {"n": n})[arm] = (block[f]["arms"][key]["breaches"], n)
    for arm, color, mark, label in arms:
        err = []
        for f in order:
            fd = block[f]
            key = f"G-{fams[f]['target_clause']}" if arm == "Gminus" else arm
            a = fd["arms"][key]
            n = fd["unauthorized"]
            if n == 0:
                continue
            m = 100 * a["breaches"] / n
            lo, hi = (100 * v for v in a["breach_ci95"])
            err.append(f"({m:.4g},{ypos[f] + offs[arm]:.3f}) -= ({m - lo:.4g},0) += ({hi - m:.4g},0)")
        fill = "" if mark in ("o", "triangle") else f",fill={color}"
        lines.append(rf"\addplot[only marks,mark={mark},mark size=2pt,draw={color}{fill},error bars/.cd,x dir=both,x explicit,"
                     rf"error bar style={{{color},line width=0.5pt}}] coordinates {{{' '.join(err)}}};\addlegendentry{{{label}}}")
    ysep = (ypos[STATE[-1]] + ypos[VISIBLE[0]]) / 2
    lines += [rf"\draw[structgray!60,densely dashed] (axis cs:-3,{ysep:.2f}) -- (axis cs:103,{ysep:.2f});",
              rf"\node[font=\tiny\itshape,text=slate,anchor=south east] at (axis cs:103,{max(ypos.values()) + 0.25:.2f}) {{violation lives in state}};",
              rf"\node[font=\tiny\itshape,text=slate,anchor=north east] at (axis cs:103,{ysep - 0.05:.2f}) {{visible in the proposal}};",
              r"\end{axis}\end{tikzpicture}"]
    return lines, stats


# ------------------------------------------------------------------ panel (c)

def panel_c(campaigns: list[dict], ckpt: str) -> tuple[list[str], dict]:
    lines = [r"\begin{tikzpicture}[baseline]",
             r"\begin{axis}[scale only axis,width=3.75cm,height=3.7cm,xmin=-3,xmax=103,ymin=-3,ymax=103,"
             r"xlabel={Task success (\%)},ylabel={Unauthorized effect (\% of campaigns)},"
             r"title={\textbf{(c)} What mediation costs},tick label style={font=\scriptsize},label style={font=\footnotesize},"
             r"title style={font=\footnotesize},grid=major,clip=false]"]
    stats = {"ckpt": ckpt, "controllers": {}}
    for key, label, color, mark in CONTROLLERS:
        rs = [r for r in campaigns if r["checkpoint"] == ckpt and r["controller"] == key and r.get("status") == "measured"]
        n = len(rs)
        if n == 0:
            raise SystemExit(f"E1 has no measured campaigns for {key} at {ckpt}")
        u, s = sum(bool(r["unauthorized"]) for r in rs), sum(bool(r["task_success"]) for r in rs)
        (ulo, uhi), (slo, shi) = cp(u, n), cp(s, n)
        um, sm = 100 * u / n, 100 * s / n
        stats["controllers"][key] = {"n": n, "unauthorized": u, "task_success": s}
        lines += [rf"\addplot[only marks,mark={mark},mark size=2.2pt,draw={color},fill={color},error bars/.cd,x dir=both,x explicit,y dir=both,y explicit,"
                  rf"error bar style={{{color},line width=0.5pt}}] coordinates {{({sm:.4g},{um:.4g}) += ({100 * shi - sm:.4g},{100 * uhi - um:.4g}) -= ({sm - 100 * slo:.4g},{um - 100 * ulo:.4g})}};",
                  rf"\node[font=\tiny,text=ink,anchor=south west,inner sep=1.5pt] at (axis cs:{sm:.4g},{um:.4g}) {{{label}}};"]
    lines += [r"\end{axis}\end{tikzpicture}"]
    return lines, stats


E8_ARMS = (("unprotected", "Unprotected", "structgray", "square*", "fill=structgray", "north east"),
           ("progent", "Progent (per call)", "warnamber!85!black", "diamond*", "fill=warnamber", "south west"),
           ("authority_strict", "Strict provenance", "exoblue", "*", "fill=exoblue", "north west"),
           ("authority_delegated", "Delegated sources", "exoblue", "o", "fill=white", "south east"),
           ("authority_oracle", "Oracle grant (bound)", "exoblue", "square", "fill=white", "south east"))


def panel_c_e8(run: Path) -> tuple[list[str], dict]:
    """E8: AgentDojo native utility (benign episodes) vs native attack success (attacked episodes), per arm."""
    rows = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    lines = [r"\begin{tikzpicture}[baseline]",
             r"\begin{axis}[scale only axis,width=3.75cm,height=3.7cm,xmin=-3,xmax=103,ymin=-3,ymax=103,"
             r"xlabel={Benign tasks completed (\%)},ylabel={Attacks that succeed (\%)},"
             r"title={\textbf{(c)} What a specification costs},tick label style={font=\scriptsize},label style={font=\footnotesize},"
             r"title style={font=\footnotesize},grid=major,clip=false]"]
    stats = {"arms": {}}
    for key, label, color, mark, fill, anchor in E8_ARMS:
        ben = [r for r in rows if r["arm"] == key and r["injection_task"] is None and r["grading_status"] == "completed"]
        att = [r for r in rows if r["arm"] == key and r["injection_task"] is not None and r["grading_status"] == "completed"]
        if not ben or not att:
            raise SystemExit(f"E8 has no graded episodes for {key}")
        u, nu = sum(r["utility"] for r in ben), len(ben)
        a, na = sum(r["attack_success"] for r in att), len(att)
        (ulo, uhi), (alo, ahi) = cp(u, nu), cp(a, na)
        um, am = 100 * u / nu, 100 * a / na
        stats["arms"][key] = {"benign_utility": [u, nu], "attack_success": [a, na]}
        lines += [rf"\addplot[only marks,mark={mark},mark size=2.3pt,draw={color},{fill},error bars/.cd,x dir=both,x explicit,y dir=both,y explicit,"
                  rf"error bar style={{{color},line width=0.5pt}}] coordinates {{({um:.4g},{am:.4g}) += ({100 * uhi - um:.4g},{100 * ahi - am:.4g}) -= ({um - 100 * ulo:.4g},{am - 100 * alo:.4g})}};",
                  rf"\node[font=\tiny,text=ink,anchor={anchor},inner sep=2pt] at (axis cs:{um:.4g},{am:.4g}) {{{label}}};"]
    lines += [r"\end{axis}\end{tikzpicture}"]
    return lines, stats


# ------------------------------------------------------------------ assembly

def build(args) -> dict:
    e4 = json.loads(args.e4.read_text())
    e5 = json.loads(args.e5.read_text())
    plan5 = json.loads((ROOT / "fixtures/v6/omitted_clause_plan.json").read_text())
    campaigns = [json.loads(l) for l in (args.e1 / "campaigns.jsonl").read_text().splitlines() if l.strip()]
    if not args.layout_preview:
        if e4.get("smoke_test") or e5.get("smoke_test"):
            raise SystemExit("smoke-test evidence cannot build the hero; pass --layout-preview for layout work only")
        if e5["plan_sha256"] != __import__("hashlib").sha256((ROOT / "fixtures/v6/omitted_clause_plan.json").read_bytes()).hexdigest():
            raise SystemExit("E5 results do not match the current plan fixture")
    a, sa = panel_a(e4, args.e4.parent, args.C, args.L, args.w)
    b, sb = panel_b(e5, plan5, args.ckpt)
    if args.e8:
        e8res = json.loads((args.e8 / "results.json").read_text())
        if not args.layout_preview and e8res.get("smoke_test"):
            raise SystemExit("smoke-test E8 evidence cannot build the hero")
        c, sc = panel_c_e8(args.e8)
    else:
        c, sc = panel_c(campaigns, args.ckpt)
    wm = []
    if args.layout_preview:
        wm = [r"\begin{tikzpicture}[remember picture,overlay]\node[rotate=18,font=\Huge\bfseries,text=notred,opacity=0.35] at (current page.center) {LAYOUT PREVIEW -- NOT EVIDENCE};\end{tikzpicture}"]
    ck = args.ckpt.split("/")[-1].replace("-GGUF", "")
    st = sb["families"]
    state_r2 = sum(st[f]["R2"][0] for f in STATE if f in st), sum(st[f]["R2"][1] for f in STATE if f in st)
    vis_r2 = sum(st[f]["R2"][0] for f in VISIBLE if f in st), sum(st[f]["R2"][1] for f in VISIBLE if f in st)
    caption = (
        r"\caption{\textbf{Halting a model stops new proposals, not effects already in flight; only a fence at the "
        r"state-bearing sink stops them, and a fence cannot stop what the specification omits.} "
        rf"\textbf{{(a)}} Real broker and sink processes, {sa['C']} concurrent tool calls of {sa['w']:g}\,s mean latency, stop "
        rf"enforced {sa['L']:g}\,s after the decision; every replicate drawn. Halting the model still lets "
        rf"{sa['halt_final'][0]}--{sa['halt_final'][-1]} effects commit and revoking at the broker "
        rf"{sa['revoke_final'][0]}--{sa['revoke_final'][-1]}, the last {sa['last_commit_s']:g}\,s after the stop; fencing the sink admits "
        rf"{sa['fence_max_post']} in all {sa['fence_trials']} fence trials of the {sa['trials']}-trial grid, including {sa['fault_trials']} with SIGKILL of the "
        rf"sink or late completions, at the cost of the legitimate in-flight calls it cancels (at {sa['wmax']:g}\,s latency: "
        rf"{sa['wmax_final']['halt_inference'][0]}--{sa['wmax_final']['halt_inference'][-1]} and "
        rf"{sa['wmax_final']['revoke_broker'][0]}--{sa['wmax_final']['revoke_broker'][-1]} leak, fence {max(sa['wmax_final']['fence_sink'])}). "
        rf"\textbf{{(b)}} {ck} red-team proposals under 8 mutation strategies (95\% Clopper--Pearson): a same-checkpoint reviewer with the "
        rf"rules and reasoning admits {state_r2[0]}/{state_r2[1]} unauthorized proposals whose violation lives in sink state and "
        rf"{vis_r2[0]}/{vis_r2[1]} whose violation is visible in the proposal; the verified gate admits none of either, but the "
        r"same gate with one clause removed admits exactly the violations of that clause. "
        + (c_caption_e8(sc) if args.e8 else
           rf"\textbf{{(c)}} Predeclared {ck} replication ({sc['controllers']['complete']['n']} independent cases per controller, 95\% "
           r"Clopper--Pearson). ")
        + r"Local quantized open-weight models on CPU; no frontier model.}\label{fig:v6hero}"
    )
    tex = ["% Generated by scripts/hero_v6.py from E4/E5/E1 evidence; do not edit.", r"\begin{figure*}[t]\centering"]
    tex += wm + a + [r"\hfill"] + b + [r"\hfill"] + c + [caption, r"\end{figure*}"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "v6-hero-fig.tex").write_text("\n".join(tex) + "\n")
    stats = {"a": sa, "b": sb, "c": sc, "layout_preview": args.layout_preview}
    (args.out / "v6-hero-stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")
    return stats


def c_caption_e8(sc: dict) -> str:
    a = sc["arms"]
    f = lambda k, m: f"{a[k][m][0]}/{a[k][m][1]}"
    return (rf"\textbf{{(c)}} AgentDojo banking and Slack with natively placed injections, Qwen3-8B, native graders (95\% "
            rf"Clopper--Pearson). Permissions written without the reference solution cost utility: strict provenance completes "
            rf"{f('authority_strict', 'benign_utility')} benign tasks (oracle grant {f('authority_oracle', 'benign_utility')}, unprotected "
            rf"{f('unprotected', 'benign_utility')}); trusting sources the user names recovers some and readmits attacks "
            rf"({f('authority_delegated', 'attack_success')}, against {f('authority_strict', 'attack_success')} strict). ")


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
    ap.add_argument("--e4", type=Path, required=True, help="E4 results.json")
    ap.add_argument("--e5", type=Path, required=True, help="E5 results.json")
    ap.add_argument("--e1", type=Path, required=True, help="E1 run directory (campaigns.jsonl)")
    ap.add_argument("--e8", type=Path, help="E8 run directory; when given, panel (c) is the AgentDojo cost panel")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", default="Qwen/Qwen3-8B-GGUF")
    ap.add_argument("--C", type=int, default=32)
    ap.add_argument("--L", type=float, default=0, help="enforcement latency (s) shown in panel (a)")
    ap.add_argument("--w", type=float, default=5.0, help="mean tool latency (s) shown in panel (a)")
    ap.add_argument("--layout-preview", action="store_true")
    ap.add_argument("--render", action="store_true", help="compile a standalone preview PDF/PNG (needs pdflatex)")
    args = ap.parse_args()
    if args.layout_preview and str(args.out.resolve()).startswith(str((ROOT / "docs").resolve())):
        raise SystemExit("a layout preview may not be written under docs/")
    stats = build(args)
    if args.render:
        preview(args.out)
    print(json.dumps({"panels": ["a", "b", "c"], "layout_preview": stats["layout_preview"]}))


if __name__ == "__main__":
    main()
