#!/usr/bin/env python3
"""Build all figures from the results files and the open fossil datasets.
Run with python -I (the fossil spreadsheets are third-party files)."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)
for f in glob.glob("/usr/share/fonts/opentype/inter/Inter-*.otf"):
    fm.fontManager.addfont(f)
HAVE_INTER = any(f.name == "Inter" for f in fm.fontManager.ttflist)

INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e4e3df"; BAND = "#f1f0ec"
BLUE, ORANGE, AQUA, VIOLET, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100", "#e34948"
B250, B300, B400, B550, B700 = "#86b6ef", "#6da7ec", "#3987e5", "#1c5cab", "#0d366b"
GREY = "#9a9993"

plt.rcParams.update({
    "font.family": "Inter" if HAVE_INTER else "DejaVu Sans", "font.size": 7.2,
    "mathtext.fontset": "custom", "mathtext.rm": "Inter" if HAVE_INTER else "DejaVu Sans",
    "mathtext.it": ("Inter:italic" if HAVE_INTER else "DejaVu Sans:italic"),
    "mathtext.bf": ("Inter:bold" if HAVE_INTER else "DejaVu Sans:bold"), "mathtext.fallback": "stixsans",
    "axes.edgecolor": INK3, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "axes.labelsize": 7.4,
    "axes.titlesize": 7.8, "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.titlepad": 5, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False, "legend.fontsize": 6.8,
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "pdf.fonttype": 3, "savefig.dpi": 300,
    "figure.dpi": 150, "text.color": INK,
})
W2, W1 = 7.1, 3.45

def J(name):
    with open(os.path.join(RES, name + ".json")) as f:
        return json.load(f)

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)

def panel(ax, letter, title):
    ax.set_title(f"{letter}   {title}")

def dot(ax, x, y, color, marker="o", s=14, z=3, alpha=1.0, **kw):
    return ax.scatter(x, y, s=s, marker=marker, facecolor=color, edgecolor="white", linewidth=0.5, zorder=z, alpha=alpha, **kw)

# ------------------------------------------------------------------ fossil helpers
def fossil_groups():
    from load_fossils import du, will_full
    early = [u for u in du() if u[1] >= 0.95]
    g = {}
    g["aus"] = [(u[1], u[3]) for u in early if u[2].startswith(("Au.", "P."))]
    g["eh"] = [(u[1], u[3]) for u in early if u[2].startswith("H.")]
    W = will_full()
    g["mp"] = [(w["age"], w["ecv"]) for w in W if w["taxon"] == "Mid-Pleistocene Homo"]
    g["nea"] = [(w["age"], w["ecv"]) for w in W if w["taxon"] == "Neanderthals"]
    g["sap"] = [(w["age"], w["ecv"]) for w in W if w["taxon"] == "Pleistocene Homo sapiens"]
    return {k: np.array(v) for k, v in g.items()}, W

GSTYLE = {"aus": (GREY, "o", "Australopithecus, Paranthropus"), "eh": (B300, "s", "early Homo, H. erectus"),
          "mp": (B550, "D", "Mid-Pleistocene Homo"), "nea": (ORANGE, "^", "Neanderthals"),
          "sap": (AQUA, "o", "Pleistocene H. sapiens")}

def step_curve(F, x):
    st, bp = F["step"], F["models"]["stasis_step"]["bps"]
    l1 = np.log(st["level_I"]); l3 = np.log(st["level_III"]); le = np.log(st["end"])
    y = np.where(x >= bp[0], l1, np.where(x >= bp[1], l1 + (l3 - l1) * (bp[0] - x) / (bp[0] - bp[1]),
                 np.where(x >= bp[2], l3, l3 + (le - l3) * (bp[2] - x) / (bp[2] - 0.01))))
    return np.exp(y)

# ------------------------------------------------------------------ Figure 1: main graphic
def box(ax, x, y, w, h, fc, ec="none", r=0.012, lw=0.0, z=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", facecolor=fc, edgecolor=ec, linewidth=lw,
                       transform=ax.transAxes, zorder=z, clip_on=False)
    ax.add_patch(p); return p

def fig_main():
    """First-page graphic: the argument in six pictures (no panel is repeated in a later figure)."""
    F = J("e13_fossils"); G, _ = fossil_groups(); S = J("e17_speed"); P = J("e15_polymath"); V = J("e16_verify"); GS = J("gss"); VG = J("e19_village")
    from exp_verify import r_capacity, r_delivery
    from exp_polymath import c_max
    fig = plt.figure(figsize=(W2, 3.62))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.62, 1.0], hspace=0.50, left=0.062, right=0.992, top=0.945, bottom=0.085)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0.22)
    bot = outer[1].subgridspec(1, 4, wspace=0.62)

    # ---- A: the fossil record
    ax = fig.add_subplot(top[0, 0])
    bp = F["models"]["stasis_step"]["bps"]
    for lo, hi in ((3.35, bp[0]), (bp[1], bp[2])):
        ax.axvspan(lo, hi, color=BAND, zorder=0, lw=0)
    for k in ("aus", "eh", "mp", "sap", "nea"):
        c, m, _ = GSTYLE[k]
        dot(ax, G[k][:, 0], G[k][:, 1], c, m, s=12 if m != "D" else 9, z=4 if k == "nea" else 3)
    xx = np.linspace(3.3, 0.01, 800)
    ax.plot(xx, step_curve(F, xx), color=INK, lw=1.7, zorder=5)
    ax.set_yscale("log"); ax.set_xlim(3.35, -0.33); ax.set_ylim(300, 2150)
    ax.set_yticks([400, 600, 800, 1000, 1400, 2000]); ax.set_yticklabels(["400", "600", "800", "1,000", "1,400", "2,000"])
    ax.minorticks_off(); ax.grid(axis="y", zorder=0)
    ax.set_xticks([3, 2.5, 2, 1.5, 1, 0.5, 0]); ax.set_xlabel("Million years ago", labelpad=2); ax.set_ylabel("Endocranial volume (cm$^3$, log scale)")
    panel(ax, "A", "Brains tripled, slowly and in steps")
    for x, t in ((2.75, "stasis"), (1.80, "increase"), (1.05, "stasis"), (0.36, "increase")):
        ax.text(x, 1960, t, ha="center", va="center", fontsize=6.4, color=INK2, fontweight="medium")
    lead = dict(arrowstyle="-", color=INK3, lw=0.5, shrinkA=1, shrinkB=3)
    ax.plot([0.285, 0.285], [465, 560], color=VIOLET, lw=2.2, solid_capstyle="round", zorder=4)
    ax.annotate("H. naledi", (0.285, 520), (0.70, 575), fontsize=6.1, color=INK2, ha="center", va="center", arrowprops=lead)
    dot(ax, [0.08], [426], VIOLET, "o", s=20, z=4)
    ax.annotate("H. floresiensis", (0.08, 426), (0.50, 388), fontsize=6.1, color=INK2, ha="center", va="center", arrowprops=lead)
    ax.plot([-0.045], [1349], marker="<", ms=4.0, color=INK, zorder=6, clip_on=False)
    ax.text(-0.105, 1349, "recent\nmean", fontsize=5.9, color=INK2, ha="left", va="center", linespacing=1.05)
    for x in (1.0, 0.78):
        ax.plot([x], [311], marker="^", ms=3.8, color=YELLOW, zorder=6, clip_on=False)
    ax.text(0.89, 350, "fire, cooking", fontsize=6.1, color=INK2, ha="center", va="center")
    h = [Line2D([], [], marker=GSTYLE[k][1], ls="", ms=4.0, mfc=GSTYLE[k][0], mec="white", mew=0.4, label=GSTYLE[k][2]) for k in ("aus", "eh", "mp", "nea", "sap")]
    h.append(Line2D([], [], color=INK, lw=1.7, label="stasis–step model"))
    ax.legend(handles=h, loc="upper left", bbox_to_anchor=(0.003, 0.905), handletextpad=0.4, labelspacing=0.28, borderpad=0.2, fontsize=6.1)

    # ---- B: the jury law, measured
    ax = fig.add_subplot(top[0, 1])
    kk = np.linspace(1, 40, 300)
    for a, col, mk, lab, dy in (("attr", YELLOW, "o", "looks", 0.0), ("fun", RED, "s", "fun", 0.0), ("intel", AQUA, "D", "intelligence", 0.035), ("sinc", VIOLET, "^", "sincerity", -0.045)):
        r1 = S["jury"][a]["curve"][0]
        ax.plot(kk, kk * r1 / (1 + (kk - 1) * r1), color=col, lw=1.5, zorder=2)
        dot(ax, np.arange(1, 9), S["jury"][a]["curve"], col, mk, s=15 if mk in ("o", "s") else 13)
    ax.axhline(0.8, color=INK3, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.text(1.05, 0.815, "dependable (0.8)", fontsize=6.0, color=INK2, ha="left", va="bottom")
    hb = [Line2D([], [], marker=mk, ls="-", lw=1.5, color=col, ms=3.8, mfc=col, mec="white", mew=0.4, label=lab)
          for col, mk, lab in ((YELLOW, "o", "looks"), (RED, "s", "fun"), (AQUA, "D", "intelligence"), (VIOLET, "^", "sincerity"))]
    ax.legend(handles=hb, loc="lower right", bbox_to_anchor=(1.0, 0.0), handlelength=1.7, handletextpad=0.4, labelspacing=0.28, borderpad=0.2, fontsize=6.2)
    ax.set_xscale("log"); ax.set_xlim(0.9, 40); ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32"]); ax.minorticks_off()
    ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.grid(axis="y")
    ax.set_xlabel("Observers who met the person", labelpad=2); ax.set_ylabel("Agreement between two panels")
    panel(ax, "B", "One observer misreads; a camp reads")

    # ---- C: breadth aims selection at the generalist
    ax = fig.add_subplot(bot[0, 0])
    K = np.linspace(1, 20, 200); Kd = np.arange(1, 21)
    ax.plot(K, K, color=INK, lw=1.6, zorder=2)
    ax.plot(Kd, [1 / c_max(int(k)) for k in Kd], color=GREY, lw=1.4, zorder=2)
    for k, v in P["aim_common"].items():
        dot(ax, [int(k)], [v["ratio"][0]], INK, "o", s=17)
    for k, v in P["aim_division"].items():
        dot(ax, [int(k)], [v["ratio"][0]], GREY, "D", s=13)
    ax.set_xlim(0, 21); ax.set_ylim(0, 21); ax.set_xticks([1, 5, 10, 15, 20]); ax.set_yticks([1, 5, 10, 15, 20]); ax.grid(axis="y")
    ax.set_xlabel("Domains compared", labelpad=2); ax.set_ylabel("Selection on general ability\nrelative to one specialty")
    ax.text(11.2, 17.0, "all judged on\nthe same\ndomains", fontsize=5.8, color=INK2, ha="right", va="center", linespacing=1.05)
    ax.text(20.6, 6.6, "each judged in\none specialty", fontsize=5.8, color=INK2, ha="right", va="center", linespacing=1.05)
    panel(ax, "C", "Breadth selects generality"); ax.title.set_fontsize(7.3)

    # ---- D: a staged test measures capacity, not delivery
    ax = fig.add_subplot(bot[0, 1])
    d = np.linspace(0, 1, 401)
    ax.plot(d, [r_capacity(40, x) for x in d], color=AQUA, lw=1.6, zorder=2)
    ax.plot(d, [r_delivery(40, x) for x in d], color=VIOLET, lw=1.6, zorder=2)
    St = V["staging"]
    dot(ax, [float(k) for k in St], [v["R_b"][0] for v in St.values()], AQUA, "o", s=15)
    dot(ax, [float(k) for k in St], [v["R_p"][0] for v in St.values()], VIOLET, "s", s=13)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(0, 1.0); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "0.5", "1"]); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.grid(axis="y")
    ax.set_xlabel("Share of occasions staged", labelpad=2); ax.set_ylabel("Validity of the assessment")
    ax.text(0.97, 0.965, "can do", fontsize=6.3, color=INK2, ha="right", va="center")
    ax.text(0.03, 0.915, "will do", fontsize=6.3, color=INK2, ha="left", va="center")
    panel(ax, "D", "Tests show can, not will"); ax.title.set_fontsize(7.3)

    # ---- E: three episodes of selection on income, by sex
    ax = fig.add_subplot(bot[0, 2]); T3 = GS["three_episodes"]["income"]
    ax.axhline(0, color=INK3, lw=0.7, zorder=1)
    for sname, col, mk, off in (("men", BLUE, "o", -0.09), ("women", ORANGE, "D", 0.09)):
        v = np.array([T3[sname][k] for k in ("pairing", "conversion", "number")])
        x = np.arange(3) + off
        ax.vlines(x, v[:, 1], v[:, 2], color=col, lw=1.3, zorder=2)
        ax.scatter(x, v[:, 0], s=22 if mk == "o" else 17, marker=mk, facecolor=col, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["marrying", "parent,\nif married", "family\nsize"], fontsize=5.7, linespacing=0.95)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.115, 0.075)
    ax.set_yticks([-0.10, -0.05, 0, 0.05]); ax.set_yticklabels(["−0.10", "−0.05", "0", "+0.05"]); ax.grid(axis="y")
    ax.set_ylabel("Selection on income (SD)", fontsize=6.9)
    ax.text(-0.02, 0.061, "men", fontsize=6.2, color=INK2, ha="left", va="center"); ax.text(0.09, -0.020, "women", fontsize=6.2, color=INK2, ha="center", va="center")
    panel(ax, "E", "Men at the door"); ax.title.set_fontsize(7.3)

    # ---- F: the door is closing: never married at ages 25-39, United States
    ax = fig.add_subplot(bot[0, 3]); BP = VG["births"]["periods"]
    pers = list(BP)
    y = [100 * BP[p]["all"]["never_married"] for p in pers]
    ax.plot(range(len(pers)), y, color=INK, lw=1.6, zorder=2)
    dot(ax, range(len(pers)), y, INK, "o", s=17)
    ax.set_xticks(range(len(pers))); ax.set_xticklabels(["1972\n–84", "1985\n–99", "2000\n–11", "2012\n–18", "2021\n–24"], fontsize=5.7, linespacing=0.95)
    ax.set_xlim(-0.35, len(pers) - 0.65); ax.set_ylim(0, 60); ax.set_yticks([0, 20, 40, 60]); ax.grid(axis="y")
    ax.set_ylabel("Never married, ages 25–39 (%)")
    ax.text(0.42, y[0] - 4.5, f"{y[0]:.0f}%", fontsize=6.4, color=INK2, ha="center", va="center")
    ax.text(len(pers) - 1.0, y[-1] + 6.5, f"{y[-1]:.0f}%", fontsize=6.4, color=INK2, ha="center", va="center")
    ax.text(len(pers) - 0.72, 9.5, "children per adult\n%.2f → %.2f" % (BP[pers[0]]["all"]["children"], BP[pers[-1]]["all"]["children"]),
            fontsize=5.8, color=INK2, ha="right", va="center", linespacing=1.1)
    panel(ax, "F", "The door is closing"); ax.title.set_fontsize(7.3)
    save(fig, "fig1_main")

# ------------------------------------------------------------------ Figure 2: fossil reanalysis
def fig_fossils():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from load_fossils import gingerich
    F = J("e13_fossils"); G, W = fossil_groups(); Gi = gingerich()
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.55), gridspec_kw=dict(width_ratios=[1.25, 1.1, 0.9], wspace=0.30, left=0.07, right=0.955, top=0.88, bottom=0.17))
    ax = axs[0]
    age = np.array([g[2] for g in Gi]); ecv = np.array([g[3] for g in Gi])
    dot(ax, age, ecv, GREY, "o", s=7, alpha=0.75)
    xx = np.linspace(3.2, 0.01, 600)
    ax.plot(xx, np.exp(np.log(ecv).mean() + F["slope_specimen"] * (-(xx) + age.mean())), color=INK3, lw=1.1, ls=(0, (4, 2)), label="constant trend")
    ax.plot(xx, step_curve(F, xx), color=BLUE, lw=1.8, label="stasis–step")
    for (lo, hi), yb, est in zip(F["bp_ci"], (318, 338, 360), F["models"]["stasis_step"]["bps"]):
        ax.plot([lo, hi], [yb, yb], color=BLUE, lw=2.2, solid_capstyle="butt", alpha=0.45)
        ax.plot([est], [yb], "o", ms=2.6, color=BLUE)
    ax.set_yscale("log"); ax.set_xlim(3.3, -0.05); ax.set_ylim(300, 2100)
    ax.set_yticks([400, 600, 1000, 1500, 2000]); ax.set_yticklabels(["400", "600", "1,000", "1,500", "2,000"]); ax.minorticks_off()
    ax.set_xlabel("Million years ago"); ax.set_ylabel("Endocranial volume (cm$^3$)"); ax.grid(axis="y")
    ax.legend(loc="upper left", handlelength=1.8, borderpad=0.1)
    panel(ax, "A", "A constant trend is rejected")
    m = F["models"]
    ax.text(3.22, 1000, f"$\\Delta$AICc\nconstant trend {m['linear']['daicc']:.0f}\nstasis–step {m['stasis_step']['daicc']:.0f}\nfree 3-break {m['pw3']['daicc']:.0f}",
            ha="left", va="center", fontsize=6.0, color=INK2, linespacing=1.25)
    ax = axs[1]
    ser = [("africa_to_sapiens", AQUA, "o", "Africa → H. sapiens", lambda w: (w["taxon"] == "Mid-Pleistocene Homo" and w["region"] == "Africa") or w["taxon"] == "Pleistocene Homo sapiens"),
           ("europe_w_asia", ORANGE, "^", "W Eurasia → Neanderthals", lambda w: w["region"] == "Europe/W Asia" and w["taxon"] != "Pleistocene Homo sapiens"),
           ("east_asia_archaic", B550, "D", "E/SE Asia, archaic", lambda w: w["taxon"] == "Mid-Pleistocene Homo" and w["region"] == "E/SE Asia")]
    for key, col, mk, lab, f in ser:
        a = np.array([w["age"] for w in W if f(w)]); e = np.array([w["ecv"] for w in W if f(w)])
        dot(ax, a, e, col, mk, s=9, alpha=0.85)
        sl = F["series"][key]["slope"]; x2 = np.linspace(a.max(), a.min(), 50)
        ax.plot(x2, np.exp(np.log(e).mean() + sl * (-(x2) + a.mean())), color=col, lw=1.7, label=f"{lab}  {sl:.2f}")
    ax.set_yscale("log"); ax.set_xlim(0.95, -0.02); ax.set_ylim(780, 1950)
    ax.set_yticks([800, 1000, 1200, 1500, 1800]); ax.set_yticklabels(["800", "1,000", "1,200", "1,500", "1,800"]); ax.minorticks_off()
    ax.set_xlabel("Million years ago"); ax.grid(axis="y")
    ax.legend(loc="upper left", handlelength=1.4, borderpad=0.1, title="slope of ln volume per Myr", title_fontsize=6.2, fontsize=6.1, labelspacing=0.3, alignment="left")
    panel(ax, "B", "Three regional series rise in parallel")
    ax = axs[2]
    rng = np.random.default_rng(3)
    order = [("mp", "Mid-Pleist.\nHomo"), ("nea", "Neander-\nthals"), ("sap", "Pleist.\nH. sapiens")]
    for i, (k, lab) in enumerate(order):
        v = G[k][:, 1]; c = GSTYLE[k][0]
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=6, facecolor=c, edgecolor="white", linewidth=0.3, alpha=0.8, zorder=3)
        ax.plot([i - 0.27, i + 0.27], [v.mean(), v.mean()], color=INK, lw=1.4, zorder=4)
        ax.text(i, v.max() + 45, f"{v.mean():,.0f}", ha="center", va="bottom", fontsize=6.4, color=INK)
    ax.axhline(1349, color=INK3, lw=0.8, ls=(0, (4, 2)), zorder=1); ax.text(1.025, 1349, "recent\n1,349", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.9, color=INK2, linespacing=1.05, clip_on=False)
    ax.axhspan(465, 560, xmin=0.02, xmax=0.98, color=VIOLET, alpha=0.16, lw=0, zorder=1); ax.text(1.0, 512, "H. naledi", ha="center", va="center", fontsize=6.0, color=INK2)
    ax.axhline(426, color=VIOLET, lw=1.0, zorder=1); ax.text(1.0, 426 - 26, "H. floresiensis", ha="center", va="top", fontsize=6.0, color=INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels([o[1] for o in order], fontsize=6.3); ax.set_xlim(-0.5, 2.5); ax.set_ylim(280, 2000)
    ax.set_yticks([400, 800, 1200, 1600, 2000]); ax.set_yticklabels(["400", "800", "1,200", "1,600", "2,000"]); ax.grid(axis="y")
    ax.tick_params(axis="x", length=0)
    panel(ax, "C", "Late Homo: large and small brains")
    save(fig, "fig2_fossils")

# ------------------------------------------------------------------ Figure S1: routes and portfolio
def fig_routes():
    E2, E3 = J("e2_routes"), J("e3_portfolio")
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.45), gridspec_kw=dict(wspace=0.24, left=0.07, right=0.995, top=0.88, bottom=0.17))
    ax = axs[0]; R = np.linspace(0, 1, 200)
    spec = [("pair", BLUE, "o", "partner quality only", lambda r: 0.4 * (0.1 + 0.1 * r)),
            ("pair_standing", ORANGE, "s", "partner quality + standing", lambda r: 0.4 * (0.1 + 0.1 * r + 0.1 * np.sqrt(r))),
            ("men_standing", AQUA, "D", "standing in men, no assortment", lambda r: 0.04 + 0.04 * np.sqrt(r))]
    for k, c, mk, lab, f in spec:
        ax.plot(R, f(R), color=c, lw=1.7, label=lab)
        rows = [x for x in E2["rows"] if x["scenario"] == k]
        ax.errorbar([x["R"] for x in rows], [x["mc"] for x in rows], yerr=[2 * x["mcse"] for x in rows], fmt=mk, ms=3.6, mfc=c, mec="white", mew=0.5, ecolor=c, elinewidth=0.6, zorder=4)
    ax.axhline(0.04, color=INK3, lw=0.7, ls=(0, (4, 2))); ax.text(0.99, 0.0413, "no information", ha="right", va="bottom", fontsize=6.0, color=INK2)
    ax.set_xlabel("Signal validity $R$"); ax.set_ylabel(r"One-generation response $\Delta\bar{A}$"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.035, 0.125); ax.grid(axis="y")
    ax.legend(loc="upper left", handlelength=1.6, borderpad=0.1); panel(ax, "A", "Three routes from information to selection")
    ax = axs[1]
    for mode, c, mk, lab in (("diverse", BLUE, "o", "each event in a new domain"), ("repeat", ORANGE, "s", "all events in one domain")):
        rows = [x for x in E3["validity"] if x["mode"] == mode]
        ax.plot([x["T"] for x in rows], [x["exact"] for x in rows], color=c, lw=1.7, label=lab)
        ax.plot([x["T"] for x in rows], [x["mc"] for x in rows], mk, ms=3.6, mfc=c, mec="white", mew=0.5, zorder=4)
    ax.axhline(E3["ceiling_portfolio"], color=INK3, lw=0.7, ls=(0, (4, 2))); ax.text(1.05, E3["ceiling_portfolio"] + 0.012, "portfolio ceiling (general nuisance only)", fontsize=6.0, color=INK2)
    ax.axhline(E3["ceiling_single"], color=INK3, lw=0.7, ls=(0, (4, 2))); ax.text(64, E3["ceiling_single"] + 0.012, "single-domain ceiling", fontsize=6.0, color=INK2, ha="right")
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32, 64]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"]); ax.minorticks_off()
    ax.set_xlabel("Total observed events"); ax.set_ylabel("Validity for general capacity $R_G$"); ax.set_ylim(0, 0.92); ax.grid(axis="y")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), handlelength=1.6, borderpad=0.1); panel(ax, "B", "Diversity beats repetition")
    save(fig, "figS1_routes")

# ------------------------------------------------------------------ Figure S2: multigeneration
def fig_multigen():
    E = J("e4_multigen")
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.4), gridspec_kw=dict(wspace=0.24, left=0.07, right=0.995, top=0.88, bottom=0.17))
    cols = [GREY, B250, B400, B550, B700]; labs = ["random pairing", "$R_0$ = 0.40", "$R_0$ = 0.67", "$R_0$ = 0.91", "$R_0$ = 1"]
    t = np.arange(E["gens"] + 1)
    for ax, key, dkey in ((axs[0], "mean", "det_mean"), (axs[1], "va", "det_va")):
        for c, lab, cd in zip(cols, labs, E["conds"]):
            ax.plot(t, cd[key], color=c, lw=1.6)
            ax.plot(t, cd[dkey], color=INK, lw=0.6, ls=(0, (2, 2)))
            ax.text(153, cd[key][-1], lab, fontsize=6.2, color=INK2, va="center")
        ax.set_xlim(0, 190); ax.set_xticks([0, 50, 100, 150]); ax.set_xlabel("Generation"); ax.grid(axis="y")
    axs[0].set_ylabel(r"Mean breeding value $\bar{A}_t$"); axs[1].set_ylabel("Additive variance $V_{A,t}$")
    axs[0].plot([], [], color=INK, lw=0.6, ls=(0, (2, 2)), label="deterministic recursion"); axs[0].plot([], [], color=B550, lw=1.6, label="simulation mean (10 replicates)")
    axs[0].legend(loc="upper left", handlelength=1.8, borderpad=0.1)
    panel(axs[0], "A", "Cumulative response"); panel(axs[1], "B", "Assortment inflates heritable variance")
    save(fig, "figS2_multigen")

# ------------------------------------------------------------------ Figure S3: persistence and tissue
def fig_persist():
    E = J("e6_persistence")
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.45), gridspec_kw=dict(wspace=0.24, left=0.07, right=0.995, top=0.88, bottom=0.17))
    ax = axs[0]; L = E["phase_len"]; tt = np.arange(len(E["tissue_positional"])) * 20 / 1000
    for lo, hi in ((0, 3), (6, 9), (12, 15)):
        ax.axvspan(lo, hi, color=BAND, lw=0, zorder=0)
    ax.plot(tt, E["tissue_positional"], color=BLUE, lw=1.8, label="positional (rank-based) benefit")
    ax.plot(tt, E["tissue_ecological"], color=ORANGE, lw=1.8, label="saturating ecological benefit")
    ax.plot(tt, E["tissue_both"], color=INK3, lw=1.0, ls=(0, (4, 2)), label="half of each")
    for x, s in ((1.5, r"$\kappa_B$"), (4.5, r"$\kappa_B/2$"), (7.5, r"$\kappa_B/4$"), (10.5, r"$\kappa_B/8$"), (13.5, "decoupling")):
        ax.text(x, 8.75, s, ha="center", va="center", fontsize=6.3, color=INK2)
    ax.set_xlim(0, 15); ax.set_ylim(0, 9.3); ax.set_xlabel("Generation (thousands)"); ax.set_ylabel(r"Mean tissue investment $\bar{B}$ (SD)"); ax.grid(axis="y")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), handlelength=1.8, borderpad=0.1); panel(ax, "A", "Budget steps become tissue under positional demand")
    ax = axs[1]; t = np.arange(len(E["sim_mean"]))
    ax.plot(t, E["sim_mean"], color=BLUE, lw=1.6, label="individual-based simulation (6 replicates)")
    ax.plot(t, E["det"], color=INK, lw=0.7, ls=(0, (2, 2)), label="exact recursion")
    ax.axhline(E["eq_pred"], color=ORANGE, lw=0.9); ax.text(40, E["eq_pred"] + 0.04, r"$\mu^* = \beta_S\sqrt{R/V_P}\,/\,c$", fontsize=6.6, color=INK2)
    ax.set_xlim(0, 1500); ax.set_ylim(0, 2.25); ax.set_xlabel("Generation"); ax.set_ylabel(r"Mean capacity $\bar{A}_t$ (SD)"); ax.grid(axis="y")
    ax.legend(loc="lower right", handlelength=1.8, borderpad=0.1); panel(ax, "B", "Exact equilibrium check")
    save(fig, "figS3_persist")

# ------------------------------------------------------------------ Figure S4: contests, coalitions, aim
def fig_contest():
    A, Rp, D = J("e10_aim"), J("e11_repeat"), J("e12_despot")
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.5), gridspec_kw=dict(wspace=0.36, left=0.065, right=0.995, top=0.88, bottom=0.18))
    ax = axs[0]
    regs = ["dominance", "mixed", "prestige"]; rows = {r["regime"]: r for r in A["rows"] if r["rule"] == "exponential"}
    x = np.arange(3); bw = 0.30
    for j, (key, c, lab) in enumerate((("exF", ORANGE, "formidability"), ("exG", BLUE, "general capacity"))):
        v = [rows[r][key] for r in regs]
        ax.bar(x + (j - 0.5) * (bw + 0.03), v, bw, color=c, label=lab, zorder=3)
        for xi, vi in zip(x, v):
            ax.text(xi + (j - 0.5) * (bw + 0.03), vi + 0.002, f"{vi:.3f}", ha="center", va="bottom", fontsize=5.9, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(["contest of\nforce", "mixed\nportfolio", "skill and\nprestige"], fontsize=6.4); ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Selection differential (SD)"); ax.set_ylim(0, 0.092); ax.grid(axis="y")
    ax.legend(loc="upper center", ncol=2, handlelength=1.0, columnspacing=1.0, borderpad=0.1, bbox_to_anchor=(0.5, 1.0)); panel(ax, "A", "The score sets the aim")
    ax = axs[1]
    o = Rp["optimum"]; d = [q["d"] for q in o]
    ax.plot(d, [q["R"] for q in o], color=BLUE, lw=1.8, marker="o", ms=3.6, mfc=BLUE, mec="white", mew=0.5)
    ax.axhline(Rp["ceiling"], color=INK3, lw=0.7, ls=(0, (4, 2))); ax.text(0.2, Rp["ceiling"] + 0.006, "ceiling", ha="left", va="bottom", fontsize=6.0, color=INK2)
    for q, off, ha in ((o[0], (7, -9), "left"), (o[2], (7, -9), "left"), (o[-1], (-2, -12), "right")):
        ax.annotate(f"$n^*$ = {q['n_eff']:.0f}", (q["d"], q["R"]), off, textcoords="offset points", ha=ha, va="center", fontsize=6.1, color=INK2)
    ax.set_xscale("log"); ax.set_xlim(0.3, 0.0007); ax.set_xticks([0.2, 0.05, 0.01, 0.002]); ax.set_xticklabels(["0.2", "0.05", "0.01", "0.002"]); ax.minorticks_off()
    ax.set_ylim(0.12, 0.43); ax.set_xlabel("Cost per event relative to the value\nof accurate assessment, $\\delta/B$"); ax.set_ylabel("Validity at the optimal number of events"); ax.grid(axis="y")
    ax.text(0.26, 0.135, "lethal", fontsize=6.0, color=INK2, ha="left"); ax.text(0.0008, 0.135, "ritualised", fontsize=6.0, color=INK2, ha="right")
    panel(ax, "B", "Safer contests inform more")
    ax = axs[2]
    for th, c, lab in ((1.0, AQUA, r"linear law, $\theta$ = 1"), (2.0, VIOLET, r"square law, $\theta$ = 2")):
        rows = [r for r in D["rows"] if r["theta"] == th and r["q"] == 0.05]
        ax.plot([r["M"] for r in rows], [r["share"] for r in rows], color=c, lw=1.8, marker="o", ms=3.4, mfc=c, mec="white", mew=0.5, label=lab)
        ax.axhline((1 + 0.05) / (1 + th), color=c, lw=0.7, ls=(0, (4, 2)))
    ax.text(0.27, 0.535, r"limit $(1+q)/(1+\theta)$", fontsize=6.0, color=INK2, va="bottom");
    ax.set_xscale("log", base=2); ax.set_xticks([0.25, 1, 4, 16, 64]); ax.set_xticklabels(["¼", "1", "4", "16", "64"]); ax.minorticks_off()
    ax.set_ylim(0.3, 1.02); ax.set_xlabel("Strength of rival coalitions, $M$\n(fighter equivalents)"); ax.set_ylabel("Dominant's optimal share of reproduction"); ax.grid(axis="y")
    ax.legend(loc="upper right", handlelength=1.5, borderpad=0.1); panel(ax, "C", "Strong rivals level the group")
    save(fig, "figS4_contest")

# ------------------------------------------------------------------ Figure S5: power
def fig_power():
    E = J("e8_power")
    fig, ax = plt.subplots(figsize=(W1, 2.45), gridspec_kw=dict(left=0.14, right=0.985, top=0.88, bottom=0.17))
    for clips, c in ((4, BLUE), (1, ORANGE)):
        for obs, ls, mk in ((40, "-", "o"), (20, (0, (4, 2)), "s")):
            rows = [r for r in E["rows"] if r["clips"] == clips and r["observers"] == obs]
            ax.plot([r["targets"] for r in rows], [r["power"] for r in rows], color=c, lw=1.6, ls=ls, marker=mk, ms=3.4, mfc=c, mec="white", mew=0.5,
                    label=f"{clips} clip{'s' if clips > 1 else ''} per cell, {obs} observers")
    ax.axhline(0.9, color=INK3, lw=0.7, ls=(0, (2, 2))); ax.text(1210, 0.885, "90%", ha="right", va="top", fontsize=6.2, color=INK2)
    ax.set_xlim(60, 1240); ax.set_ylim(0, 1.03); ax.set_xlabel("Targets"); ax.set_ylabel("Power (two-sided $\\alpha$ = 0.05)"); ax.grid(axis="y")
    ax.legend(loc="lower right", handlelength=2.0, borderpad=0.1, fontsize=6.3); ax.set_title("Stimulus sampling buys more power than observers")
    save(fig, "figS5_power")

# ------------------------------------------------------------------ Figure 3: said, did, and what four minutes can read
ATT_LAB = dict(attr="looks", fun="fun", intel="intelligence", amb="ambition", sinc="sincerity", shar="shared\ninterests")

def fig_jury():
    S = J("e17_speed")
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.42), gridspec_kw=dict(left=0.062, right=0.992, top=0.865, bottom=0.165, wspace=0.30))
    order = ["attr", "shar", "fun", "intel", "sinc", "amb"]
    # A: said and did, as shares of 100
    ax = axs[0]
    rows = ["attr", "shar", "fun", "intel", "sinc", "amb"]
    for i, a in enumerate(rows):
        y0 = len(rows) - 1 - i
        for n, col, mk, dy in (("men", BLUE, "o", 0.17), ("women", ORANGE, "D", -0.17)):
            pos = {b: max(S["did"]["own"][n]["coef"][b][0], 0.0) for b in rows}
            did = 100 * pos[a] / sum(pos.values()); said = S["said"][n][a]
            ax.annotate("", xy=(did, y0 + dy), xytext=(said, y0 + dy), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, mutation_scale=6, shrinkA=2.2, shrinkB=2.6), zorder=2)
            ax.scatter([said], [y0 + dy], s=17 if mk == "o" else 13, marker=mk, facecolor="white", edgecolor=col, linewidth=1.0, zorder=3)
            ax.scatter([did], [y0 + dy], s=20 if mk == "o" else 15, marker=mk, facecolor=col, edgecolor="white", linewidth=0.5, zorder=4)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(["looks", "shared\ninterests", "fun", "intelligence", "sincerity", "ambition"][::-1], fontsize=6.5, linespacing=0.95)
    ax.set_ylim(-0.6, len(rows) - 0.4); ax.set_xlim(-2, 56); ax.set_xticks([0, 10, 20, 30, 40, 50]); ax.grid(axis="x"); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Share of weight (%): said → did")
    h = [Line2D([], [], marker="o", ls="", ms=4.2, mfc=BLUE, mec="white", mew=0.4, label="men"),
         Line2D([], [], marker="D", ls="", ms=3.6, mfc=ORANGE, mec="white", mew=0.4, label="women"),
         Line2D([], [], marker="o", ls="", ms=3.8, mfc="white", mec=INK2, mew=0.9, label="said"),
         Line2D([], [], marker="o", ls="", ms=4.2, mfc=INK2, mec="white", mew=0.4, label="did")]
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=2, columnspacing=0.8, handletextpad=0.2, labelspacing=0.25, borderpad=0.15, fontsize=6.2)
    panel(ax, "A", "What people say is not what they do")
    # B: legibility and weight
    ax = axs[1]
    ax.axhline(0, color=INK3, lw=0.7, zorder=1)
    r1 = {a: S["jury"][a]["r1"] for a in ("attr", "fun", "intel", "amb", "sinc")}
    for n, col, mk in (("men", BLUE, "o"), ("women", ORANGE, "D")):
        for a in r1:
            c = S["did"]["panel"][n]["coef"][a]
            ax.vlines(r1[a], c[1], c[2], color=col, lw=1.0, zorder=2)
            ax.scatter([r1[a]], [c[0]], s=20 if mk == "o" else 15, marker=mk, facecolor=col, edgecolor="white", linewidth=0.5, zorder=3)
    for a, (x, y, t, ha) in dict(attr=(0.39, 7.2, "looks", "center"), fun=(0.264, 9.4, "fun", "center"), intel=(0.150, 7.4, "intelligence,\nambition", "center"),
                                 sinc=(0.060, -5.6, "sincerity", "center")).items():
        ax.text(x, y, t, fontsize=6.2, color=INK2, ha=ha, va="center", linespacing=1.05)
    ax.set_xlim(0, 0.46); ax.set_ylim(-7.5, 21); ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4]); ax.set_yticks([-5, 0, 5, 10, 15, 20]); ax.grid(axis="y")
    ax.set_xlabel("Agreement between two observers"); ax.set_ylabel("Change in chance of a yes per SD\nof the panel's rating (points)")
    panel(ax, "B", "People act on what they can read")
    # C: expected and actual interest
    ax = axs[2]; I = S["interest"]
    for n, col, mk, off in (("men", BLUE, "o", -0.03), ("women", ORANGE, "D", 0.03)):
        e, a = I[n]["expected"], I[n]["actual"]
        x = np.array([0, 1]) + off
        ax.plot(x, [100 * e[0], 100 * a[0]], color=col, lw=1.5, zorder=2)
        ax.vlines(x, [100 * e[1], 100 * a[1]], [100 * e[2], 100 * a[2]], color=col, lw=1.1, zorder=2)
        ax.scatter(x, [100 * e[0], 100 * a[0]], s=22 if mk == "o" else 17, marker=mk, facecolor=col, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xlim(-0.45, 1.45); ax.set_xticks([0, 1]); ax.set_xticklabels(["expected\na yes", "received\na yes"], linespacing=1.0)
    ax.set_ylim(30, 55); ax.set_yticks([30, 35, 40, 45, 50, 55]); ax.grid(axis="y"); ax.set_ylabel("Share of meetings (%)")
    ax.text(1.09, 100 * I["women"]["actual"][0] + 0.2, "women", fontsize=6.4, color=INK2, ha="left", va="center")
    ax.text(1.09, 100 * I["men"]["actual"][0], "men", fontsize=6.4, color=INK2, ha="left", va="center")
    panel(ax, "C", "Men overestimate their welcome")
    save(fig, "fig3_jury")

# ------------------------------------------------------------------ Figure 4: the generalist target
def fig_polymath():
    P = J("e15_polymath")
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.42), gridspec_kw=dict(left=0.062, right=0.992, top=0.865, bottom=0.165, wspace=0.30))
    # A: breadth buys time
    ax = axs[0]
    ramp = {1: B250, 3: B300, 6: B400, 12: B550, 24: B700}
    n = np.arange(1, 41)
    for k, col in ramp.items():
        yv = P["breadth"]["curves"][str(k)]
        ax.plot(n, yv, color=col, lw=1.6)
        ax.text(38, yv[-1] + 0.012, f"K = {k}", fontsize=6.3, color=INK2, va="bottom", ha="right")
    ax.axhline(0.5, color=INK3, lw=0.7, ls=(0, (2, 2)))
    ax.text(2.25, 0.507, "target R* = 0.5", fontsize=6.2, color=INK2, va="bottom", ha="left")
    ax.set_xscale("log"); ax.set_xlim(1, 40); ax.set_xticks([1, 2, 5, 10, 20, 40]); ax.set_xticklabels(["1", "2", "5", "10", "20", "40"]); ax.minorticks_off()
    ax.set_ylim(0, 0.85); ax.grid(axis="y"); ax.set_xlabel("Events observed per domain"); ax.set_ylabel("Validity for general capacity")
    panel(ax, "A", "Breadth buys time")
    # B: expenditure share
    ax = axs[1]
    Kc = np.linspace(2, 32, 300)
    for (ratio, col) in (("4.0", B300), ("8.0", B550), ("16.0", B700)):
        rr = float(ratio)
        ax.plot(Kc, Kc / (Kc + rr), color=col, lw=1.6, zorder=2)
        d = P["expenditure"][ratio]
        dot(ax, [int(k) for k in d], [d[k]["share"][0] for k in d], col, "o", s=17)
        ax.scatter([rr], [0.5], s=26, facecolor="white", edgecolor=col, linewidth=1.1, zorder=5)
        ax.text(2.25, 2 / (2 + rr) + 0.055, f"{int(rr)}", fontsize=6.3, color=INK2, va="center", ha="center")
    ax.text(33, 0.07, "numbers: cost ratio $c_G/c_S$", fontsize=6.1, color=INK2, va="center", ha="right")
    ax.axhline(0.5, color=INK3, lw=0.7, ls=(0, (2, 2)))
    ax.set_xscale("log", base=2); ax.set_xlim(1.8, 35); ax.set_xticks([2, 4, 8, 16, 32]); ax.set_xticklabels(["2", "4", "8", "16", "32"]); ax.minorticks_off()
    ax.set_ylim(0, 1.0); ax.grid(axis="y"); ax.set_xlabel("Number of domains, K"); ax.set_ylabel("Share of investment in\ngeneral capacity")
    panel(ax, "B", "Expenditure follows breadth")
    # C: what breadth dilutes (specific skill, and exaggeration, per domain relative to general investment)
    ax = axs[2]
    for (ratio, col) in (("4.0", B300), ("8.0", B550), ("16.0", B700)):
        rr = float(ratio)
        ax.plot(Kc, rr / Kc, color=col, lw=1.6, zorder=2)
        d = P["expenditure"][ratio]
        dot(ax, [int(k) for k in d], [d[k]["s"][0] / d[k]["g"][0] for k in d], col, "o", s=17)
        ax.text(2.3, rr / 2 * 1.32, f"{int(rr)}", fontsize=6.3, color=INK2, va="center", ha="center")
    ax.axhline(1.0, color=INK3, lw=0.7, ls=(0, (2, 2)))
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2); ax.set_xlim(1.8, 35); ax.set_ylim(0.1, 11)
    ax.set_xticks([2, 4, 8, 16, 32]); ax.set_xticklabels(["2", "4", "8", "16", "32"]); ax.set_yticks([0.125, 0.25, 0.5, 1, 2, 4, 8]); ax.set_yticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    ax.minorticks_off(); ax.grid(axis="y"); ax.set_xlabel("Number of domains, K"); ax.set_ylabel("Per-domain investment in specialty\nor show, relative to general")
    panel(ax, "C", "Breadth dilutes specialty and show")
    save(fig, "fig4_polymath")

# ------------------------------------------------------------------ Figure 5: enforcement, crowding out and the gate
def fig_verify():
    V = J("e16_verify"); G = J("gss")["gate"]
    from exp_verify import equilibrium, ENF
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.42), gridspec_kw=dict(left=0.062, right=0.992, top=0.865, bottom=0.165, wspace=0.30))
    # A: enforcement below the take-off threshold
    ax = axs[0]
    E = V["enforcement"]; g = np.array(E["grid"]); bmax = E["params"]["bmax"]
    ax.plot(g, np.array(E["b"]) / bmax, color=AQUA, lw=1.7, zorder=2)
    ax.plot(g, E["p"], color=VIOLET, lw=1.7, zorder=2)
    I = E["ibm"]
    dot(ax, [float(k) for k in I], [v["b"][0] / bmax for v in I.values()], AQUA, "o", s=18)
    dot(ax, [float(k) for k in I], [v["p"][0] for v in I.values()], VIOLET, "s", s=15)
    ax.scatter([E["e_star"]], [E["p_max"]], s=30, facecolor="white", edgecolor=VIOLET, linewidth=1.1, zorder=5)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.02, 1.04); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.grid(axis="y")
    ax.set_xlabel("Share of product delivered by enforcement"); ax.set_ylabel("Equilibrium level")
    ax.text(0.56, 0.80, "capacity\n(share of its maximum)", fontsize=6.3, color=INK2, ha="right", va="center", linespacing=1.1)
    ax.text(0.70, 0.245, "evolved\ncommitment", fontsize=6.3, color=INK2, ha="left", va="center", linespacing=1.1)
    panel(ax, "A", "Enforcement before commitment")
    # B: delivered share below and above the threshold
    ax = axs[1]
    fine = np.linspace(0, 1, 401)
    low = [equilibrium(float(x))[2] for x in fine]
    high = [equilibrium(float(x), dict(ENF, Rp=0.75))[2] for x in fine]
    ax.plot(fine, low, color=B300, lw=1.7, zorder=2); ax.plot(fine, high, color=B700, lw=1.7, zorder=2)
    cr = E["crowding"]
    ax.scatter([cr["at"]], [cr["min_delivered"]], s=30, facecolor="white", edgecolor=B700, linewidth=1.1, zorder=5)
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.02, 1.04); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.grid(axis="y")
    ax.set_xlabel("Share of product delivered by enforcement"); ax.set_ylabel("Share of product delivered in all")
    ax.text(0.16, 0.885, "choice alone\nsuffices (T = 1.5)", fontsize=6.2, color=INK2, ha="center", va="center", linespacing=1.1)
    ax.text(0.60, 0.46, "choice alone\nfails (T = 0.6)", fontsize=6.2, color=INK2, ha="left", va="center", linespacing=1.1)
    panel(ax, "B", "Where it helps, where it crowds out")
    # C: the pairing gate
    ax = axs[2]
    coh = ["1900-1919", "1920-1939", "1940-1959", "1960-1979"]
    men = [[100 * v for v in G["men"][c]["parents_nm"]] for c in coh]
    women = [[100 * v for v in G["women"][c]["parents_nm"]] for c in coh]
    _two(ax, range(4), men, women)
    ax.set_xlim(-0.35, 3.35); ax.set_xticks(range(4)); ax.set_xticklabels(["1900–19", "1920–39", "1940–59", "1960–79"])
    ax.set_ylim(0, 75); ax.set_yticks([0, 25, 50, 75]); ax.grid(axis="y")
    ax.set_xlabel("Birth cohort"); ax.set_ylabel("Parents among people\nwho never married (%)")
    ax.text(2.82, women[3][0] + 5.5, "women", fontsize=6.5, color=INK2, ha="right", va="center")
    ax.text(3.05, men[3][0] - 9.0, "men", fontsize=6.5, color=INK2, ha="center", va="center")
    panel(ax, "C", "The pairing gate has opened")
    save(fig, "fig5_verify")

# ------------------------------------------------------------------ Figure 6: selection today (GSS)
def _two(ax, xs, men, women, labels=True, ms=4.2):
    """Men and women series with 95% intervals; values are [point, lo, hi]."""
    for series, col, mk, off in ((men, BLUE, "o", -0.045), (women, ORANGE, "D", 0.045)):
        x = np.array(xs, float) + off
        pt = np.array([v[0] for v in series]); lo = np.array([v[1] for v in series]); hi = np.array([v[2] for v in series])
        ax.plot(x, pt, color=col, lw=1.5, zorder=2)
        ax.vlines(x, lo, hi, color=col, lw=1.1, zorder=2)
        ax.scatter(x, pt, s=19 if mk == "o" else 15, marker=mk, facecolor=col, edgecolor="white", linewidth=0.5, zorder=3)

def fig_today():
    G = J("gss")
    fig, axs = plt.subplots(2, 3, figsize=(W2, 4.35), gridspec_kw=dict(left=0.062, right=0.992, top=0.935, bottom=0.085, wspace=0.30, hspace=0.50))
    pc = lambda v: [100 * v[0], 100 * v[1], 100 * v[2]]
    thirds = ["lowest", "middle", "highest"]
    # A childless by fifth of income (the shape of the floor)
    ax = axs[0, 0]; Fq = G["fifths"]["income"]
    _two(ax, [0, 1, 2, 3, 4], [pc(v) for v in Fq["men"]["childless"]], [pc(v) for v in Fq["women"]["childless"]])
    ax.set_xticks([0, 1, 2, 3, 4]); ax.set_xticklabels(["lowest", "2nd", "3rd", "4th", "highest"], fontsize=6.4); ax.set_xlim(-0.4, 4.4); ax.set_ylim(0, 27); ax.grid(axis="y")
    ax.set_xlabel("Fifth of personal income"); ax.set_ylabel("Childless (%)")
    ax.text(0.22, 24.2, "men", fontsize=6.6, color=INK2, ha="left", va="center"); ax.text(0.2, 6.4, "women", fontsize=6.6, color=INK2, ha="left", va="center")
    panel(ax, "A", "Access: a floor for men")
    # B kids among parents by schooling third
    ax = axs[0, 1]; T = G["thirds"]["schooling"]
    _two(ax, [0, 1, 2], [r["kids_parents"] for r in T["men"]], [r["kids_parents"] for r in T["women"]])
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(thirds); ax.set_xlim(-0.35, 2.35); ax.set_ylim(2.0, 3.6); ax.grid(axis="y")
    ax.set_xlabel("Third of schooling"); ax.set_ylabel("Children per parent")
    ax.text(0.14, 3.47, "women", fontsize=6.6, color=INK2, ha="left", va="center"); ax.text(-0.22, 3.04, "men", fontsize=6.6, color=INK2, ha="center", va="center")
    panel(ax, "B", "Number: schooling and family size")
    # C Bateman curve
    ax = axs[0, 2]; B = G["bateman"]
    rows_m = [r for r in B["men"]["rows"] if r["lo"] >= 1]; rows_w = [r for r in B["women"]["rows"] if r["lo"] >= 1]
    xs = list(range(len(rows_m)))
    _two(ax, xs, [r["children"] for r in rows_m], [r["children"] for r in rows_w])
    ax.set_xticks(xs); ax.set_xticklabels(["1", "2", "3–4", "5–9", "10–19", "20+"]); ax.set_xlim(-0.4, 5.4); ax.set_ylim(1.4, 3.0); ax.grid(axis="y")
    ax.set_xlabel("Lifetime partners of the other sex"); ax.set_ylabel("Mean number of children")
    ax.text(5.3, 2.36, "men", fontsize=6.6, color=INK2, ha="right", va="center"); ax.text(4.72, 1.66, "women", fontsize=6.6, color=INK2, ha="right", va="center")
    panel(ax, "C", "Partners are not offspring")
    # D partnerless by period
    ax = axs[1, 0]; Pm, Pw = G["partnerless"]["men"], G["partnerless"]["women"]
    per = list(Pm.keys())
    _two(ax, [0, 1, 2], [pc(Pm[p]["all"]) for p in per], [pc(Pw[p]["all"]) for p in per])
    lo3 = [100 * Pm[p]["low"][0] for p in per]; hi3 = [100 * Pm[p]["high"][0] for p in per]
    ax.plot([0, 1, 2], lo3, color=B300, lw=1.2, ls=(0, (4, 2)), zorder=1); ax.plot([0, 1, 2], hi3, color=B700, lw=1.2, ls=(0, (1, 1.6)), zorder=1)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels([p.replace("-", "–") for p in per], fontsize=6.3); ax.set_xlim(-0.3, 3.25); ax.set_ylim(0, 21); ax.grid(axis="y")
    ax.set_xlabel("Survey years"); ax.set_ylabel("No partner in past year (%)")
    for yv, lab in ((lo3[-1] + 1.1, "men, lowest\nincome third"), (100 * Pm[per[-1]]["all"][0] - 0.2, "all men"), (100 * Pw[per[-1]]["all"][0] + 1.3, "all women"), (hi3[-1] - 1.6, "men, highest\nincome third")):
        ax.text(2.13, yv, lab, fontsize=6.0, color=INK2, ha="left", va="center", linespacing=1.0)
    panel(ax, "D", "Partnerless at ages 22–34")
    # E share of opportunity due to childlessness
    ax = axs[1, 1]; O = G["opportunity"]; coh = ["1900-1919", "1920-1939", "1940-1959", "1960-1979"]
    _two(ax, [0, 1, 2, 3], [pc(O["men"][c]["share_childless"]) for c in coh], [pc(O["women"][c]["share_childless"]) for c in coh])
    ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels([c.replace("-", "–") for c in coh], fontsize=6.1); ax.set_xlim(-0.4, 3.4); ax.set_ylim(0, 50); ax.grid(axis="y")
    ax.set_xlabel("Birth cohort"); ax.set_ylabel("Share of opportunity for\nselection from childlessness (%)")
    ax.text(3.3, 46.5, "men", fontsize=6.6, color=INK2, ha="right", va="center"); ax.text(3.3, 31.0, "women", fontsize=6.6, color=INK2, ha="right", va="center")
    panel(ax, "E", "The floor's share of the variance")
    # F family size by marital history, relative to one's sex and birth decade
    ax = axs[1, 2]; H = G["marital_history"]
    ax.axhline(1.0, color=INK3, lw=0.7, zorder=1)
    keys = ["intact", "remarried", "divorced"]
    for sname, col, mk, off in (("men", BLUE, "o", -0.09), ("women", ORANGE, "D", 0.09)):
        v = np.array([H[sname][k]["rel"] for k in keys])
        x = np.arange(3) + off
        ax.vlines(x, v[:, 1], v[:, 2], color=col, lw=1.3, zorder=2)
        ax.scatter(x, v[:, 0], s=22 if mk == "o" else 17, marker=mk, facecolor=col, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["one\nmarriage", "remarried\nafter divorce", "divorced,\nnot remarried"], fontsize=6.1, linespacing=1.0)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(0.92, 1.17); ax.set_yticks([0.95, 1.0, 1.05, 1.10, 1.15]); ax.grid(axis="y")
    ax.set_ylabel("Family size relative to\nsex and birth-decade mean")
    ax.text(0.80, 1.150, "men", fontsize=6.6, color=INK2, ha="right", va="center"); ax.text(1.22, 1.060, "women", fontsize=6.6, color=INK2, ha="left", va="center")
    panel(ax, "F", "A second marriage pays men only")
    save(fig, "fig6_today")


# ------------------------------------------------------------------ Figure 7: who decides, and who becomes an ancestor
def fig_choice():
    C = J("e18_cultures"); N = J("e21_notes"); V = J("e19_village")
    fig = plt.figure(figsize=(W2, 4.75))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.62, left=0.125, right=0.985, top=0.94, bottom=0.085, width_ratios=[1.0, 1.0])
    # A who chooses a spouse
    ax = fig.add_subplot(gs[0, 0]); A = C["arrangement"]
    rows = [("Sons", "men"), ("Daughters", "women"), ("Sons, foragers", "men_foragers"), ("Daughters, foragers", "women_foragers")]
    segs = [("alone", "chooses alone", B250), ("with_approval", "chooses, elders approve", B400), ("either", "either custom", B550), ("parents", "parents choose", B700)]
    for i, (lab, k) in enumerate(rows):
        left = 0.0
        for sk, _, col in segs:
            v = A[k]["share"][sk]
            ax.barh(i, v - 0.8, left=left + 0.4, color=col, height=0.62, zorder=2)
            if v >= 9:
                ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=6.3, color="white" if col in (B400, B550, B700) else INK)
            left += v
        ax.text(101.5, i, f"{A[k]['n']}", fontsize=6.0, color=INK3, va="center")
    ax.set_yticks(range(4)); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlim(0, 111); ax.set_xticks([0, 25, 50, 75, 100])
    ax.text(101.5, -0.62, "n", fontsize=6.0, color=INK3, va="center", style="italic")
    ax.set_xlabel("Societies (%)", labelpad=2); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=c) for _, _, c in segs], labels=[l for _, l, _ in segs], loc="lower left",
              bbox_to_anchor=(-0.30, 1.0), ncol=4, fontsize=5.7, handlelength=0.9, handletextpad=0.4, columnspacing=0.8, borderaxespad=0.2)
    ax.set_title("A   A jury decides, and more firmly for daughters", pad=17)
    # B constraint, coercion and exit
    ax = fig.add_subplot(gs[0, 1])
    cw = C["consent"]["women"]; rp = C["rape"]
    items = [("Bride's consent not required", cw["share"]["2"] + cw["share"]["3"], cw["n"], VIOLET),
             ("Daughter cannot easily object", 100 * A["women_cannot_object"] / A["women"]["n"], A["women"]["n"], VIOLET),
             ("Wife-beating present", C["wife_beating"]["share"]["2"], C["wife_beating"]["n"], VIOLET),
             ("Rape reported", rp["sanday"]["share"]["2"], rp["sanday"]["n"], VIOLET),
             ("Women taken captive in war", C["captives"]["share"]["1"], C["captives"]["n"], VIOLET),
             ("Divorce open equally to both", C["divorce_access"]["share"]["3"], C["divorce_access"]["n"], AQUA),
             ("Divorce common", C["divorce_frequency"]["share"]["common"], C["divorce_frequency"]["n"], AQUA)]
    for i, (lab, v, n, col) in enumerate(items):
        y = i + (0.45 if i >= 5 else 0)
        ax.barh(y, v, color=col, height=0.6, zorder=2)
        ax.text(v + 1.5, y, f"{v:.0f}%  (n = {n})", fontsize=6.2, color=INK2, va="center")
    ax.set_yticks([i + (0.45 if i >= 5 else 0) for i in range(len(items))]); ax.set_yticklabels([it[0] for it in items], fontsize=6.4)
    ax.invert_yaxis(); ax.set_xlim(0, 118); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xlabel("Societies with data (%)", labelpad=2)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.text(117, 1.0, "constraint", fontsize=6.2, color=VIOLET, ha="right", va="center", fontweight="medium")
    ax.text(117, 6.45, "exit", fontsize=6.2, color="#12805a", ha="right", va="center", fontweight="medium")
    panel(ax, "B", "Constraint is common, and so is exit")
    # C men wait
    ax = fig.add_subplot(gs[1, 0]); P = np.array(C["binford_age"]["pairs"])
    ax.plot([8, 36], [8, 36], color=INK3, lw=0.8, zorder=1)
    jit = np.random.default_rng(5).normal(0, 0.12, P.shape)
    ax.scatter(P[:, 1] + jit[:, 1], P[:, 0] + jit[:, 0], s=10, facecolor=B400, edgecolor="white", linewidth=0.4, alpha=0.85, zorder=3)
    ax.set_xlim(8, 26); ax.set_ylim(8, 36); ax.grid(zorder=0); ax.set_xlabel("Women's age at first marriage (years)", labelpad=2)
    ax.set_ylabel("Men's age at first marriage (years)")
    ax.text(21.3, 22.6, "same age", fontsize=6.2, color=INK2, ha="right", va="center", rotation=37)
    ax.text(25.6, 11.6, f"{C['binford_age']['n']} forager groups;\nmen marry {C['binford_age']['gap']:.1f} years later", fontsize=6.4, color=INK2, ha="right", va="center", linespacing=1.15)
    panel(ax, "C", "Men wait")
    # D who becomes an ancestor
    ax = fig.add_subplot(gs[1, 1]); cu = N["ancestors"]["curve"]
    qm = np.array(cu["qm"]); share = 100 * (1 - qm)
    for a, col, lab in ((0.0, B250, "fathers alike"), (0.5, B400, ""), (1.0, B550, ""), (2.0, B700, "fathers very unequal")):
        ax.plot(share, cu[f"alpha_{a}"], color=col, lw=1.5, zorder=2)
    ax.axhline(2.0, color=INK3, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(19.5, 1.50, "ratio read from\nmitochondrial and\nY lineages", fontsize=5.9, color=INK2, ha="left", va="center", linespacing=1.05)
    dot(ax, [40], [2.0], INK, "o", s=22, z=5)
    ax.annotate("80% of women,\n40% of men", (40, 2.0), (47, 1.22), fontsize=6.0, color=INK2, ha="center", va="center", linespacing=1.1,
                arrowprops=dict(arrowstyle="-", color=INK3, lw=0.5, shrinkA=1, shrinkB=3))
    an = V["ancestors"]
    xs = [100 * (1 - an[k]["men"]["q"]) for k in ("1900-1939", "1940-1959", "1960-1979")]
    ys = [an[k]["ratio"] for k in ("1900-1939", "1940-1959", "1960-1979")]
    dot(ax, xs, ys, ORANGE, "D", s=20, z=5)
    ax.annotate("United States,\nborn 1900–1979", (xs[1], ys[1]), (63, 0.86), fontsize=6.0, color=INK2, ha="center", va="center", linespacing=1.1,
                arrowprops=dict(arrowstyle="-", color=INK3, lw=0.5, shrinkA=1, shrinkB=3))
    ax.set_yscale("log"); ax.set_ylim(0.72, 8); ax.set_yticks([1, 2, 4, 8]); ax.set_yticklabels(["1", "2", "4", "8"]); ax.minorticks_off()
    ax.set_xlim(18, 100); ax.grid(axis="y", zorder=0)
    ax.set_xlabel("Men who leave descendants (%)", labelpad=2); ax.set_ylabel("Women's effective number\nrelative to men's")
    ax.legend(handles=[Line2D([0], [0], color=B700, lw=1.5), Line2D([0], [0], color=B250, lw=1.5)], labels=["fathers very unequal", "fathers alike"],
              loc="upper right", fontsize=6.0, handlelength=1.4, borderaxespad=0.2)
    panel(ax, "D", "Twice as many mothers is not half the men")
    save(fig, "fig7_choice")


# ------------------------------------------------------------------ Figure 8: the world, said and paired
def fig_world():
    W = J("e20_world"); WT = W["walter"]; C = J("e18_cultures")
    fig = plt.figure(figsize=(W2, 4.75))
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.34, left=0.125, right=0.985, top=0.94, bottom=0.085)
    # A stated ideals in 45 countries
    ax = fig.add_subplot(gs[0, 0])
    traits = [("kindness", "Kindness"), ("health", "Health"), ("intelligence", "Intelligence"), ("physatt", "Looks"), ("resources", "Financial\nprospects")]
    rng = np.random.default_rng(3)
    for i, (k, lab) in enumerate(traits):
        for sname, col, off in (("men", BLUE, -0.17), ("women", ORANGE, 0.17)):
            v = np.array(WT["country_means"][k][sname])
            ax.scatter(v, i + off + rng.normal(0, 0.035, len(v)), s=5, facecolor=col, edgecolor="none", alpha=0.28, zorder=2)
        dot(ax, [WT["means"][k]["men"]], [i - 0.17], BLUE, "o", s=26, z=4)
        dot(ax, [WT["means"][k]["women"]], [i + 0.17], ORANGE, "D", s=21, z=4)
    ax.set_yticks(range(5)); ax.set_yticklabels([t[1] for t in traits], linespacing=1.0); ax.invert_yaxis(); ax.set_xlim(3.9, 7.0); ax.grid(axis="x", zorder=0)
    ax.set_xlabel("How much of it an ideal partner should have (1–7)", labelpad=2); ax.tick_params(axis="y", length=0)
    dot(ax, [6.58], [3.80], BLUE, "o", s=20, z=4); ax.text(6.66, 3.80, "men", fontsize=6.3, color=INK2, ha="left", va="center")
    dot(ax, [6.58], [4.18], ORANGE, "D", s=16, z=4); ax.text(6.66, 4.18, "women", fontsize=6.3, color=INK2, ha="left", va="center")
    ax.text(3.95, 0.0, "small points:\ncountries", fontsize=5.9, color=INK3, ha="left", va="center", linespacing=1.05)
    panel(ax, "A", "Said, in 45 countries: kindness, health, mind")
    # B direction shared, size not
    ax = fig.add_subplot(gs[0, 1]); I = W["ideals"]
    rows = [("Women want more\nmoney in a partner", I["resources"], 70), ("Women want\nmore kindness", I["kindness"], 70),
            ("Men want\nbetter looks", I["looks"], 70), ("Women want more\nintelligence", WT["d"]["intelligence"], 45)]
    ax.axvline(0, color=INK3, lw=0.8, zorder=1)
    for i, (lab, d, n) in enumerate(rows):
        v = np.array(d["values"])
        ax.scatter(v, i + rng.normal(0, 0.10, len(v)), s=8, facecolor=B300, edgecolor="white", linewidth=0.3, alpha=0.9, zorder=2)
        ax.plot([d["median"]] * 2, [i - 0.30, i + 0.30], color=INK, lw=1.8, zorder=4)
        ax.text(1.46, i, f"{d['positive']} of {d['countries']}", fontsize=6.3, color=INK2, ha="right", va="center")
    ax.set_yticks(range(4)); ax.set_yticklabels([r[0] for r in rows], fontsize=6.3, linespacing=1.0); ax.invert_yaxis()
    ax.set_xlim(-0.62, 1.48); ax.grid(axis="x", zorder=0); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Sex difference in the ideal, by country (standard deviations)", labelpad=2)
    panel(ax, "B", "The direction is shared; the size is not")
    # C the door in 70 countries
    ax = fig.add_subplot(gs[1, 0]); O = W["door"]
    keys = [("resources", "Financial\nprospects"), ("looks", "Looks"), ("kindness", "Kindness"), ("health", "Health")]
    ax.axhline(0, color=INK3, lw=0.8, zorder=1)
    for sname, col, mk, off in (("men", BLUE, "o", -0.10), ("women", ORANGE, "D", 0.10)):
        for i, (k, _) in enumerate(keys):
            lo, hi = O["ci"][sname][k]
            ax.vlines(i + off, lo, hi, color=col, lw=1.3, zorder=2)
            ax.scatter([i + off], [O["coef"][sname][k]], s=24 if mk == "o" else 19, marker=mk, facecolor=col, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels([k[1] for k in keys], fontsize=6.4, linespacing=1.0); ax.set_xlim(-0.5, 3.5); ax.set_ylim(-2.4, 6.6)
    ax.set_yticks([-2, 0, 2, 4, 6]); ax.set_yticklabels(["−2", "0", "+2", "+4", "+6"]); ax.grid(axis="y", zorder=0)
    ax.set_ylabel("Having a partner: points per\nstandard deviation of own rating")
    ax.text(0.90, 5.95, "men", fontsize=6.4, color=INK2, ha="center", va="center"); ax.text(0.22, 2.35, "women", fontsize=6.4, color=INK2, ha="left", va="center")
    panel(ax, "C", "The door again, in 70 countries")
    # D age gaps
    ax = fig.add_subplot(gs[1, 1]); P = np.array(C["binford_age"]["pairs"])
    sets = [("Forager groups\n(ethnographic)", P[:, 0] - P[:, 1], B700), ("45 countries\n(2016–2018)", np.array(WT["age_gap"]["values"]), B400),
            ("175-country survey\n(2021)", np.array(W["ages"]["values"]), B250)]
    ax.axvline(0, color=INK3, lw=0.8, zorder=1)
    for i, (lab, v, col) in enumerate(sets):
        ax.scatter(v, i + rng.normal(0, 0.11, len(v)), s=8, facecolor=col, edgecolor="white", linewidth=0.3, alpha=0.9, zorder=2)
        ax.plot([np.median(v)] * 2, [i - 0.32, i + 0.32], color=INK, lw=1.8, zorder=4)
        ax.text(21.6, i - 0.40, f"median {np.median(v):.1f}", fontsize=6.2, color=INK2, ha="right", va="center")
    ax.set_yticks(range(3)); ax.set_yticklabels([s_[0] for s_ in sets], fontsize=6.3, linespacing=1.0); ax.invert_yaxis(); ax.set_xlim(-6, 22); ax.set_ylim(2.55, -0.62)
    ax.grid(axis="x", zorder=0); ax.tick_params(axis="y", length=0); ax.set_xlabel("Years by which the man is older", labelpad=2)
    panel(ax, "D", "Men are the older partner almost everywhere")
    save(fig, "fig8_world")


# ------------------------------------------------------------------ Figure 9: the village
def fig_village():
    V = J("e19_village"); W = J("e20_world")
    pers = ["1972-1984", "1985-1999", "2000-2011", "2012-2018", "2021-2024"]
    xl = ["1972\n–84", "1985\n–99", "2000\n–11", "2012\n–18", "2021\n–24"]
    fig = plt.figure(figsize=(W2, 4.75))
    gs = fig.add_gridspec(2, 2, hspace=0.52, wspace=0.34, left=0.085, right=0.985, top=0.94, bottom=0.09)
    # A wanted and had
    ax = fig.add_subplot(gs[0, 0]); ID = V["ideals"]
    ideal = [ID[p]["ideal"] for p in pers]; act = [ID[p]["actual"] for p in pers]
    for y, lo_hi, col, mk in ((ideal, "ideal_ci", AQUA, "o"), (act, "actual_ci", INK, "s")):
        ax.plot(range(5), y, color=col, lw=1.6, zorder=2)
        ax.vlines(range(5), [ID[p][lo_hi][0] for p in pers], [ID[p][lo_hi][1] for p in pers], color=col, lw=1.0, zorder=2)
        dot(ax, range(5), y, col, mk, s=17)
    ax.set_xticks(range(5)); ax.set_xticklabels(xl, fontsize=6.3, linespacing=0.95); ax.set_xlim(-0.3, 4.3); ax.set_ylim(1.5, 3.4); ax.grid(axis="y", zorder=0)
    ax.set_ylabel("Children")
    ax.text(4.2, 2.69, "ideal number, ages 18–44", fontsize=6.3, color=INK2, ha="right", va="center")
    ax.text(4.2, 1.93, "children had by ages 40–49", fontsize=6.3, color=INK2, ha="right", va="center")
    panel(ax, "A", "People want as many children as in 1972")
    # B where the births went
    ax = fig.add_subplot(gs[0, 1]); L = V["births"]["long"]["all"]
    bars = [("Change in children\nper adult aged 25–39", L["change"], INK), ("from fewer people\nmarrying", L["marriage"], VIOLET),
            ("from fewer children\namong the married", L["within_ever"], B400), ("from more children\noutside marriage", L["within_never"], AQUA)]
    ax.axvline(0, color=INK3, lw=0.8, zorder=1)
    for i, (lab, v, col) in enumerate(bars):
        y = i + (0.35 if i else 0)
        ax.barh(y, v, color=col, height=0.6, zorder=2)
        ax.text(v + (0.012 if v > 0 else -0.012), y, f"{v:+.2f}".replace("-", "−"), fontsize=6.4, color=INK2, va="center", ha="left" if v > 0 else "right")
    ax.set_yticks([i + (0.35 if i else 0) for i in range(4)]); ax.set_yticklabels([b[0] for b in bars], fontsize=6.3, linespacing=1.0); ax.invert_yaxis()
    ax.set_xlim(-0.56, 0.30); ax.set_xticks([-0.4, -0.2, 0, 0.2]); ax.set_xticklabels(["−0.4", "−0.2", "0", "+0.2"]); ax.grid(axis="x", zorder=0)
    ax.tick_params(axis="y", length=0); ax.set_xlabel("Children per adult, 1972–84 to 2021–24", labelpad=2)
    panel(ax, "B", "The shortfall is at the door")
    # C fewer witnesses
    ax = fig.add_subplot(gs[1, 0]); T = V["village"]["trend"]
    series = [("friends_monthly", "friends", YELLOW, "o"), ("relatives_monthly", "relatives", RED, "s"),
              ("neighbours_monthly", "neighbours", AQUA, "D"), ("worship_weekly", "worship, weekly", VIOLET, "^")]
    for k, lab, col, mk in series:
        y = [T[p][k]["all"]["pct"] for p in pers]
        ax.plot(range(5), y, color=col, lw=1.5, zorder=2)
        dot(ax, range(5), y, col, mk, s=15)
        ax.text(4.12, y[-1] + {"relatives_monthly": 2.6, "friends_monthly": -2.6}.get(k, 0), lab, fontsize=6.3, color=INK2, ha="left", va="center")
    ax.set_xticks(range(5)); ax.set_xticklabels(xl, fontsize=6.3, linespacing=0.95); ax.set_xlim(-0.3, 5.45); ax.set_ylim(0, 90); ax.grid(axis="y", zorder=0)
    ax.set_ylabel("Adults aged 22–34 seeing them\nat least monthly (%)")
    panel(ax, "C", "Fewer witnesses")
    # D ties and having a partner
    ax = fig.add_subplot(gs[1, 1]); A = V["village"]["assoc"]; S = W["screens"]
    rows = [("Sees relatives monthly", A["relatives_monthly"]["points"], A["relatives_monthly"]["ci"], RED),
            ("Attends worship weekly", A["worship_weekly"]["points"], A["worship_weekly"]["ci"], VIOLET),
            ("Sees neighbours monthly", A["neighbours_monthly"]["points"], A["neighbours_monthly"]["ci"], AQUA),
            ("Sees friends monthly", A["friends_monthly"]["points"], A["friends_monthly"]["ci"], YELLOW),
            ("Hour a day on social\nmedia, men (70 countries)", -S["men"]["points_per_hour"], [-S["men"]["ci"][1], -S["men"]["ci"][0]], GREY),
            ("the same, women", -S["women"]["points_per_hour"], [-S["women"]["ci"][1], -S["women"]["ci"][0]], GREY)]
    ax.axvline(0, color=INK3, lw=0.8, zorder=1)
    for i, (lab, v, c, col) in enumerate(rows):
        y = i + (0.45 if i >= 4 else 0)
        ax.hlines(y, c[0], c[1], color=col, lw=1.4, zorder=2)
        dot(ax, [v], [y], col, "o", s=22, z=3)
    ax.set_yticks([i + (0.45 if i >= 4 else 0) for i in range(6)]); ax.set_yticklabels([r[0] for r in rows], fontsize=6.3, linespacing=1.0); ax.invert_yaxis()
    ax.set_xlim(-15, 15); ax.grid(axis="x", zorder=0); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Difference in having a partner (percentage points)", labelpad=2)
    panel(ax, "D", "Kin go with pairing; friends and screens do not")
    save(fig, "fig9_village")

if __name__ == "__main__":
    todo = sys.argv[1:] or ["main", "fossils", "jury", "polymath", "verify", "routes", "multigen", "persist", "contest", "today", "power", "choice", "world", "village"]
    for k in todo:
        globals()["fig_" + k](); print("built", k, flush=True)
