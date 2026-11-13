"""Macros, tables and figures for the v17 confidence audit layer and the high-resolution facial evidence maps.

Kept apart from ``performance_paper.py`` so the two can evolve without merge conflicts. Every number is read from
the validated reference JSON (``data["confidence"]`` and ``data["face_highres"]``); nothing here is hand-typed.
"""
from __future__ import annotations

import numpy as np

# ledger id -> (macro stem, scale, decimals). Values print in math mode with a true minus sign.
LEDGER_MACROS = (
    ("hu2025_pitch_funding", "HuFund", 100, 1),
    ("hu2025_pitch_exit", "HuExit", 100, 1),
    ("hu2025_gender", "HuWomen", 1, 3),
    ("bernstein2017_team", "BernTeam", 100, 1),
    ("bernstein2017_novices", "BernNovice", 100, 1),
    ("gompers2020_selection", "GompSelect", 100, 0),
    ("gompers2020_attribution", "GompSuccess", 100, 0),
    ("ewens2020_entry", "EwensEntry", 100, 1),
    ("ewens2020_outcome_test", "EwensExit", 100, 1),
    ("kanze2018_prevention", "KanzePrev", 1, 1),
    ("gornall2025_cold_email", "Gornall", 1, 2),
    ("brooks2014_voice", "BrooksVoice", 100, 1),
    ("graham2017_pay", "GrahamPay", 1, 1),
    ("malmendier2008_acquisitions", "MalmOdds", 1, 2),
    ("malmendier2008_market", "MalmCar", 1, 2),
    ("tenney2007_exp2_before", "TenneyBefore", 1, 2),
    ("tenney2007_exp2_after", "TenneyAfter", 1, 2),
    ("kennedy2013_revealed", "KennedyEta", 1, 2),
    ("anderson2012_mba7wk", "AndersonWeeks", 1, 2),
    ("anderson2012_dyad", "AndersonDyad", 1, 2),
    ("todorov2005_senate", "Todorov", 100, 1),
    ("olivola2010_validity", "OlivolaTraits", 1, 0),
    ("jacquart2015_ambiguity", "Jacquart", 1, 2),
    ("vonrueden2016_humans", "VrHumans", 1, 2),
    ("vonrueden2016_primates", "VrPrimates", 1, 2),
    ("wiessner2005_bigshot", "WiessBig", 100, 0),
    ("wiessner2014_day_criticism", "WiessDay", 100, 0),
    ("wiessner2014_night_stories", "WiessNight", 100, 0),
    ("williams2016_hireability", "WilliamsHire", 1, 2),
    ("williams2016_likability", "WilliamsLike", 1, 2),
    ("exley2022_self_promotion", "Exley", 1, 1),
    ("folke2020_promotion_divorce", "Folke", 1, 0),
    ("bertrand2015_happy", "BertHappy", 1, 0),
    ("bertrand2015_divorce", "BertDivorce", 100, 1),
    ("rogers2023_hero_meaning", "Rogers", 1, 2),
    ("koenig2011_men_leaders", "Koenig", 1, 2),
    ("paustian2014_other_ratings", "PaustOther", 1, 2),
    ("paustian2014_self_ratings", "PaustSelf", 1, 2),
    ("johnson_fowler2011_binomial", "JohnsonFowler", 1, 1),
    ("cheng2013_prestige", "ChengPrestige", 1, 2),
    ("cheng2013_dominance", "ChengDominance", 1, 2),
    ("adams2018_ceo_traits", "Adams", 100, 0),
    ("lu2020_assertiveness", "Lu", 1, 2),
    ("fisman2006_intel_kink", "FismanKink", 100, 1),
)


def num(x: float, d: int = 1, plus: bool = False) -> str:
    """Math-mode number with a true minus sign; ``plus`` forces an explicit sign."""
    v = round(float(x), d)
    if v == 0:
        v = 0.0
    s = f"{abs(v):.{d}f}"
    if v < 0:
        return "-" + s
    return ("+" + s) if plus else s


def ci(lo: float, hi: float, d: int = 1) -> str:
    return f"[{num(lo, d)},\\,{num(hi, d)}]"


def pval(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _ledger(conf):
    return {r["id"]: r for r in conf["ledger"]}


def confidence_macros(data) -> dict[str, str]:
    conf = data["confidence"]
    v: dict[str, str] = {}
    g = conf["gss_audience_women_leaders"]
    comp = g["audience_composition"]
    early, recent = comp["early_1975_1985"], comp["recent_2016_2024"]
    v["PboConfGssN"] = f"{g['n_answered']:,}".replace(",", "{,}")
    v["PboConfGssWaves"] = str(len(g["years"]))
    for tag, cell in (("Early", early), ("Recent", recent)):
        v[f"PboConfGss{tag}Men"] = num(100 * cell["men"]["agree_share"])
        v[f"PboConfGss{tag}Women"] = num(100 * cell["women"]["agree_share"])
        v[f"PboConfGss{tag}Young"] = num(100 * cell["age_18_34"]["agree_share"])
        v[f"PboConfGss{tag}Old"] = num(100 * cell["age_55_plus"]["agree_share"])
        v[f"PboConfGss{tag}BA"] = num(100 * cell["bachelors_plus"]["agree_share"])
        v[f"PboConfGss{tag}NoBA"] = num(100 * cell["below_bachelors"]["agree_share"])
    gap = g["recent_audience_sex_gap"]
    v["PboConfGssGap"] = num(gap["estimate_pp"], 1, plus=True)
    v["PboConfGssGapCI"] = ci(*gap["ci_95_pp"])
    lpm = g["recent_adjusted_lpm"]["coefficients_pp"]
    v["PboConfGssBACoef"] = num(lpm["bachelors_plus"]["estimate"], 1, plus=True)
    v["PboConfGssBACI"] = ci(*lpm["bachelors_plus"]["ci_95"])
    v["PboConfGssFemaleCoef"] = num(lpm["female"]["estimate"], 1, plus=True)
    v["PboConfGssFemaleCI"] = ci(*lpm["female"]["ci_95"])
    fb = g["firebaugh_decomposition"]
    for tag, key in (("", "pooled"), ("Women", "women"), ("Men", "men")):
        c = fb[key]["components_pp"]
        v[f"PboConfFb{tag}Observed"] = num(c["observed_change"]["estimate"])
        v[f"PboConfFb{tag}Within"] = num(c["within_cohort_change"]["estimate"])
        v[f"PboConfFb{tag}Replace"] = num(c["cohort_replacement"]["estimate"])
        v[f"PboConfFb{tag}ReplaceCI"] = ci(*c["cohort_replacement"]["ci_95"])
        v[f"PboConfFb{tag}Residual"] = num(c["nonlinearity_residual"]["estimate"])
        v[f"PboConfFb{tag}Share"] = num(100 * fb[key]["replacement_share_of_fitted"], 0)
    sens = fb["pooled_pre_mode_change_sensitivity"]
    v["PboConfFbPreModeShare"] = num(100 * sens["replacement_share_of_fitted"], 0)
    v["PboConfFbN"] = f"{fb['pooled']['n']:,}".replace(",", "{,}")

    h = conf["hcmst_relative_income"]
    hc = h["confirmatory"]
    v["PboConfHcmstN"] = f"{h['sample']['n']:,}".replace(",", "{,}")
    v["PboConfHcmstSheMore"] = num(hc["estimate_pp"], 1, plus=True)
    v["PboConfHcmstCI"] = ci(*hc["ci_95_pp"])
    cells = h["excellent_share_cells"]
    v["PboConfHcmstWomenHe"] = num(100 * cells["he_earns_more"]["women_respondents"]["excellent_share"])
    v["PboConfHcmstWomenShe"] = num(100 * cells["she_earns_more"]["women_respondents"]["excellent_share"])
    v["PboConfHcmstMenHe"] = num(100 * cells["he_earns_more"]["men_respondents"]["excellent_share"])
    v["PboConfHcmstMenShe"] = num(100 * cells["she_earns_more"]["men_respondents"]["excellent_share"])
    mult = conf["multiplicity"]
    for tag, key in (("Hcmst", "h1_she_earns_more_vs_he_earns_more_excellent_pp"),
                     ("GssGap", "g1_recent_audience_men_minus_women_agree_pp"),
                     ("Replace", "g1_cohort_replacement_component_pp")):
        v[f"PboConf{tag}RawP"] = pval(mult["raw_p"][key])
        v[f"PboConf{tag}HolmP"] = pval(mult["holm_adjusted_p"][key])

    w = conf["speed_dating_warm_confidence"]
    for src, s_tag in (("own", "Own"), ("panel", "Panel")):
        for d, d_tag in (("male_chooser_female_target", "FT"), ("female_chooser_male_target", "MT")):
            cf = w[src][d]["coefficients_pp_per_sd"]
            for term, t_tag in (("agency", "Agency"), ("warmth", "Warmth"), ("agency_x_warmth", "Inter"), ("attractiveness", "Attr")):
                v[f"PboConfWarm{s_tag}{t_tag}{d_tag}"] = num(cf[term]["estimate"], 1, plus=True)
                v[f"PboConfWarm{s_tag}{t_tag}{d_tag}CI"] = ci(*cf[term]["interval_95"])
            v[f"PboConfWarm{s_tag}N{d_tag}"] = f"{w[src][d]['n_decisions']:,}".replace(",", "{,}")
        df = w[src]["female_minus_male_target_pp_per_sd"]
        for term, t_tag in (("warmth", "Warmth"), ("agency_x_warmth", "Inter")):
            v[f"PboConfWarm{s_tag}{t_tag}Diff"] = num(df[term]["estimate"], 1, plus=True)
            v[f"PboConfWarm{s_tag}{t_tag}DiffCI"] = ci(*df[term]["interval_95"])

    sat = conf["speed_dating_sat_proxy"]
    v["PboConfSatValues"] = str(sat["n_distinct_mn_sat_values"])
    v["PboConfSatPeople"] = str(sat["n_participants_with_mn_sat"])
    v["PboConfSatBlock"] = str(sat["largest_shared_value_block"])
    cb = sat["cluster_bootstrap"]["institution_sat_value"]
    v["PboConfSatIntelCI"] = ci(*cb["r_stranger_intel_vs_partner_institution_sat_ci_95"], d=3)
    v["PboConfSatYesCI"] = ci(*cb["r_partner_institution_sat_vs_yes_ci_95"], d=3)

    e25 = conf["proposition_25_e25"]
    v["PboConfKStar"] = num(e25["witness_threshold_k_star"], 0)
    by_k = {row["k_contexts"]: row for row in e25["by_k"]}
    v["PboConfEInwardZero"] = num(by_k[0]["selection_differential_inward_dominance"]["closed_form"], 2, plus=True)
    kmax = max(by_k)
    v["PboConfEInwardMaxK"] = str(kmax)
    v["PboConfEInwardMax"] = num(by_k[kmax]["selection_differential_inward_dominance"]["closed_form"], 2, plus=True)
    v["PboConfEMaxErr"] = num(max(e25["max_abs_error"], conf["proposition_26_e26"]["max_abs_error"]), 3)
    e26 = conf["proposition_26_e26"]
    rows26 = {row["raters_m"]: row for row in e26["by_raters"]}
    m_lo, m_hi = min(rows26), max(rows26)
    v["PboConfEMHi"] = str(m_hi)
    v["PboConfEGap"] = num(rows26[m_lo]["rating_gap_at_equal_quality"]["closed_form"], 2)
    for m, tag in ((m_lo, "Lo"), (m_hi, "Hi")):
        r = rows26[m]["closed_form"]
        v[f"PboConfEFundRatio{tag}"] = num(r["penalized_group"]["funded_rate"] / r["reference_group"]["funded_rate"], 2)
        v[f"PboConfEOutcomeGap{tag}"] = num(rows26[m]["outcome_test_quality_gap_funded"]["closed_form"], 2, plus=True)
    sym = e26["symmetric_protocol_m16"]["closed_form"] if "closed_form" in e26["symmetric_protocol_m16"] else e26["symmetric_protocol_m16"]
    v["PboConfESymRatio"] = num(_ratio(sym), 2)

    led = _ledger(conf)
    for lid, stem, scale, d in LEDGER_MACROS:
        est = led[lid]["estimate"]
        v[f"PboLed{stem}"] = num(scale * est, d)
        if led[lid]["n"]:
            v[f"PboLed{stem}N"] = f"{led[lid]['n']:,}".replace(",", "{,}")
    summ = conf["ledger_summary"]
    v["PboConfLedgerN"] = str(summ["rows"])
    v["PboConfLedgerFull"] = str(summ["by_status"]["verified-fulltext"])
    v["PboConfLedgerPreprint"] = str(summ["by_status"]["verified-fulltext-preprint"])
    v["PboConfLedgerAbstract"] = str(summ["by_status"]["verified-abstract-only"])
    for stage, tag in (("entry", "Entry"), ("outcome", "Outcome"), ("benchmark", "Bench"), ("boundary", "Boundary")):
        v[f"PboConfLedger{tag}"] = str(summ["by_stage"][stage])
    tot = conf["scorecard"]["totals"]
    v["PboConfScoreRows"] = str(len(conf["scorecard"]["rows"]))
    for theory, tag in (("PBO", "Pbo"), ("SRT", "Srt"), ("SST", "Sst"), ("Becker", "Becker"), ("CST", "Cst"), ("SET", "Set")):
        v[f"PboConfScore{tag}P"] = str(tot[theory]["predicted"])
        v[f"PboConfScore{tag}C"] = str(tot[theory]["contradicted"])
        v[f"PboConfScore{tag}S"] = str(tot[theory]["silent"])
    v.update(v16_audit_macros(data))
    return v


def _ratio(sym) -> float:
    if isinstance(sym, dict) and "penalized_group" in sym:
        return sym["penalized_group"]["funded_rate"] / sym["reference_group"]["funded_rate"]
    if isinstance(sym, dict) and "funded_rate_ratio" in sym:
        return sym["funded_rate_ratio"]
    raise KeyError("symmetric protocol cell has no funded rates")


# ---------------------------------------------------------------------------------------------------------- tables
def _row_cell(led, lid, text, scale=1, d=2, unit=""):
    est = led[lid]["estimate"]
    return text.replace("{x}", f"${num(scale * est, d, plus=False)}${unit}")


def confidence_ledger_table(data) -> str:
    """Main-text table: entry versus outcome across arenas, every cell bound to a verified ledger row."""
    conf = data["confidence"]
    led = _ledger(conf)
    w = conf["speed_dating_warm_confidence"]
    g = conf["gss_audience_women_leaders"]["recent_audience_sex_gap"]
    hc = conf["hcmst_relative_income"]["confirmatory"]
    holm = conf["multiplicity"]["holm_adjusted_p"]

    def e(lid, scale=1, d=2, plus=False):
        return f"${num(scale * led[lid]['estimate'], d, plus=plus)}$"

    own_ft = w["own"]["male_chooser_female_target"]["coefficients_pp_per_sd"]
    pan_ft = w["panel"]["male_chooser_female_target"]["coefficients_pp_per_sd"]
    pan_mt = w["panel"]["female_chooser_male_target"]["coefficients_pp_per_sd"]
    rows = [
        ("Small groups \\& witnesses",
         f"Self-enhancers liked after one meeting~\\cite{{paulhus1998}}; confident witness $d={num(led['tenney2007_exp2_before']['estimate'], 2)}$~\\cite{{tenney2007}}",
         f"Disliked by meeting~7; after one revealed error $d={num(led['tenney2007_exp2_after']['estimate'], 2)}$",
         f"\\emph{{Loss:}} revealed overconfidence kept status ($\\eta_p^2={num(led['kennedy2013_revealed']['estimate'], 2)}$, $n={led['kennedy2013_revealed']['n']}$)~\\cite{{kennedy2013}}"),
        ("Romance (Tier~1 + 2)",
         f"Own-rated agency {'$' + num(own_ft['agency']['estimate'], 1, True) + '$'}\\,pp/SD; other raters {'$' + num(pan_ft['agency']['estimate'], 1, True) + '$'}/{'$' + num(pan_mt['agency']['estimate'], 1, True) + '$'}",
         f"Rivalry erodes unions~\\cite{{wurst2017}}; she-earns-more ``excellent'' ${num(hc['estimate_pp'], 1, True)}$\\,pp (Holm $p={pval(holm['h1_she_earns_more_vs_he_earns_more_excellent_pp'])}$)",
         f"Other-rated warmth adds ${num(pan_ft['warmth']['estimate'], 1, True)}$/${num(pan_mt['warmth']['estimate'], 1, True)}$\\,pp/SD at first meeting"),
        ("Leaders",
         "Narcissism predicts emergence~\\cite{grijalva2015}",
         "No linear effectiveness gain; curvilinear peak",
         f"Other-rated effectiveness, men$-$women {e('paustian2014_other_ratings')}; self-rated {e('paustian2014_self_ratings', plus=True)}~\\cite{{paustian2014}}"),
        ("CEOs",
         f"Competent face {e('graham2017_pay', d=1, plus=True)}\\% pay per point~\\cite{{graham2017}}; overconfident bidders OR {e('malmendier2008_acquisitions')}~\\cite{{malmendier2008}}",
         f"ROA unrelated to face; bid returns {e('malmendier2008_market')}\\% over 3 days",
         "Hired for people skills; execution pays~\\cite{kaplan2012}; overconfident CEOs innovate more~\\cite{hirshleifer2012}"),
        ("Founders \\& investors",
         f"Pitch delivery {e('hu2025_pitch_funding', 100, 1, True)}\\,pp funding/SD (women-only teams {e('hu2025_gender', d=3)} vs.\\ men-only)~\\cite{{hu2025}}; team news {e('bernstein2017_team', 100, 1, True)}\\,pp~\\cite{{bernstein2017}}",
         f"Same delivery {e('hu2025_pitch_exit', 100, 1)}\\,pp exit/SD; {e('gompers2020_attribution', 100, 0)}\\% of VCs credit the team for success~\\cite{{gompers2020}}",
         f"\\emph{{Loss:}} preparedness, not passion, funded~\\cite{{chen2009}}; novices chase traction {e('bernstein2017_novices', 100, 1, True)}\\,pp"),
        ("Gender protocol",
         f"Identical cold email: women {e('gornall2025_cold_email', d=2, plus=True)}\\,pp replies~\\cite{{gornall2025}}; live Q\\&A: each prevention question \\${num(abs(led['kanze2018_prevention']['estimate']), 1)}M less~\\cite{{kanze2018}}",
         f"Male investors: female-led interest {e('ewens2020_entry', 100, 1)}\\,pp, yet the men they pick exit {e('ewens2020_outcome_test', 100, 1)}\\,pp less~\\cite{{ewens2020}}",
         f"GSS audience sex gap ${num(g['estimate_pp'], 1, True)}$\\,pp ${ci(*g['ci_95_pp'])}$: a shared, not male-only, stereotype"),
        ("Voters",
         f"Face-inferred competence calls {e('todorov2005_senate', 100, 1)}\\% of Senate races~\\cite{{todorov2005}}",
         f"Appearance inferences beat base rates on {e('olivola2010_validity', d=0)} of 11 traits~\\cite{{olivola2010}}",
         "Charisma decides only when performance is ambiguous~\\cite{jacquart2015}"),
        ("Evolutionary benchmark",
         f"Status and reproductive success: men $r={num(led['vonrueden2016_humans']['estimate'], 2)}$ vs.\\ primates ${num(led['vonrueden2016_primates']['estimate'], 2)}$~\\cite{{vonrueden2016}}",
         f"Big-shot behaviour {e('wiessner2005_bigshot', 100, 0)}\\% of punished issues~\\cite{{wiessner2005}}",
         f"Forager talk: {e('wiessner2014_day_criticism', 100, 0)}\\% criticism by day, {e('wiessner2014_night_stories', 100, 0)}\\% stories by firelight~\\cite{{wiessner2014}}"),
    ]
    return "\n".join(" & ".join(r) + r" \\\addlinespace[0.8pt]" for r in rows) + "\n"


SCORE_SYMBOL = {"P": r"$\bullet$", "C": r"$\times$", "S": r"$\cdot$"}


def confidence_scorecard_table(data) -> str:
    card = data["confidence"]["scorecard"]
    theories = card["theories"]
    lines = []
    for i, row in enumerate(card["rows"], start=1):
        label = row["regularity"].replace("&", r"\&").replace("%", r"\%")
        lines.append(f"{i}. {label} & " + " & ".join(SCORE_SYMBOL[row["codes"][t]] for t in theories) + r" \\")
    tot = card["totals"]
    lines.append(r"\midrule")
    lines.append(r"Predicted / contradicted / silent & " + " & ".join(
        f"{tot[t]['predicted']}/{tot[t]['contradicted']}/{tot[t]['silent']}" for t in theories) + r" \\")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------- figure
def confidence_audit_plot(data) -> bytes:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    import performance_paper as pp

    conf = data["confidence"]
    g = conf["gss_audience_women_leaders"]
    fig = plt.figure(figsize=(7.16, 4.55))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.42, left=0.065, right=0.99, top=0.90, bottom=0.085)

    # A. GSS trend by respondent sex
    ax = fig.add_subplot(gs[0, 0])
    years = sorted(int(y) for y in g["trend_by_year_sex"])
    for key, color, marker, label in (("women", pp.WOMEN, "D", "Women respondents"), ("men", pp.MEN, "o", "Men respondents")):
        est = np.array([100 * g["trend_by_year_sex"][str(y)][key]["agree_share"] for y in years])
        lo = np.array([100 * g["trend_by_year_sex"][str(y)][key]["ci_95"][0] for y in years])
        hi = np.array([100 * g["trend_by_year_sex"][str(y)][key]["ci_95"][1] for y in years])
        ax.fill_between(years, lo, hi, color=color, alpha=0.13, lw=0)
        ax.plot(years, est, color=color, lw=1.15, marker=marker, ms=2.3, label=label)
    ax.set_xlim(1972, 2026)
    ax.set_ylim(0, 60)
    ax.set_xticks([1975, 1990, 2005, 2024])
    pp.panel(ax, "A", "Audience: men better suited for politics?", ylabel="Agree (%)", fontsize=7.6, pad=4)
    ax.legend(loc="upper right", fontsize=5.6, handlelength=1.4)
    gap = g["recent_audience_sex_gap"]
    ax.text(2023.5, 3.0, f"2016–24 men−women {gap['estimate_pp']:+.1f} pp\n[{gap['ci_95_pp'][0]:.1f}, {gap['ci_95_pp'][1]:.1f}]",
            ha="right", va="bottom", fontsize=5.3, color=pp.INK2)

    # B. Firebaugh decomposition
    ax = fig.add_subplot(gs[0, 1])
    fb = g["firebaugh_decomposition"]
    groups = (("pooled", "All"), ("women", "Women"), ("men", "Men"))
    comps = (("within_cohort_change", "Within", pp.AQUA), ("cohort_replacement", "Replacement", pp.VIOLET),
             ("nonlinearity_residual", "Residual", pp.GOLD))
    x = np.arange(len(groups))
    width = 0.24
    for j, (ck, clabel, color) in enumerate(comps):
        est = [fb[k]["components_pp"][ck]["estimate"] for k, _ in groups]
        lo = [fb[k]["components_pp"][ck]["ci_95"][0] for k, _ in groups]
        hi = [fb[k]["components_pp"][ck]["ci_95"][1] for k, _ in groups]
        xx = x + (j - 1) * width
        ax.bar(xx, est, width=width, color=color, alpha=0.88, label=clabel, hatch=("", "///", "..")[j], edgecolor="white", lw=0.3)
        ax.errorbar(xx, est, yerr=[np.subtract(est, lo), np.subtract(hi, est)], fmt="none", ecolor=pp.INK, elinewidth=0.6, capsize=1.5)
    for i, (k, _) in enumerate(groups):
        ax.text(x[i], -26.5, f"repl. {100 * fb[k]['replacement_share_of_fitted']:.0f}%", ha="center", va="bottom", fontsize=5.2, color=pp.VIOLET)
    ax.axhline(0, color=pp.MUTED, lw=0.6)
    ax.set_xticks(x, [lbl for _, lbl in groups])
    ax.set_ylim(-27, 7)
    pp.panel(ax, "B", "1975–78 → 2018–24 decline split", ylabel="Change (pp)", fontsize=7.6, pad=4)
    ax.legend(loc="upper center", fontsize=5.0, handlelength=1.1, ncol=3, columnspacing=0.6, borderaxespad=0.1)

    # C. HCMST relative income
    ax = fig.add_subplot(gs[0, 2])
    h = conf["hcmst_relative_income"]["excellent_share_cells"]
    cats = (("he_earns_more", "He earns\nmore"), ("about_the_same", "About\nsame"), ("she_earns_more", "She earns\nmore"))
    for key, color, marker, off, label in (("women_respondents", pp.WOMEN, "D", -0.09, "Women report"), ("men_respondents", pp.MEN, "o", 0.09, "Men report")):
        ys, ns = [], []
        for c, _ in cats:
            cell = h[c][key]
            ys.append(100 * cell["excellent_share"])
            ns.append(cell["n"])
        xs = np.arange(len(cats)) + off
        ax.plot(xs, ys, color=color, lw=0.9, marker=marker, ms=3.4, label=label)
        for xi, yi, ni in zip(xs, ys, ns):
            ax.text(xi, yi + 1.6, f"n={ni}", ha="center", fontsize=4.6, color=color)
    hc = conf["hcmst_relative_income"]["confirmatory"]
    holm = conf["multiplicity"]["holm_adjusted_p"]["h1_she_earns_more_vs_he_earns_more_excellent_pp"]
    ax.text(0.02, 0.04, f"She vs he more: {hc['estimate_pp']:+.1f} pp\n[{hc['ci_95_pp'][0]:.1f}, {hc['ci_95_pp'][1]:.1f}], Holm p={holm:.3f}",
            transform=ax.transAxes, fontsize=5.3, color=pp.INK2, va="bottom")
    ax.set_xticks(range(len(cats)), [lbl for _, lbl in cats], fontsize=5.8)
    ax.set_ylim(30, 80)
    pp.panel(ax, "C", "HCMST: union rated “excellent”", ylabel="Excellent (%)", fontsize=7.6, pad=4)
    ax.legend(loc="lower right", fontsize=5.4, handlelength=1.2)

    # D. Warm confidence: own vs other raters
    ax = fig.add_subplot(gs[1, 0])
    w = conf["speed_dating_warm_confidence"]
    terms = (("agency", "Agency"), ("warmth", "Warmth"), ("agency_x_warmth", "Agency×warmth"))
    yb = np.arange(len(terms))[::-1] * 1.0
    combos = (("own", "male_chooser_female_target", pp.WOMEN, "D", 0.27, False), ("panel", "male_chooser_female_target", pp.WOMEN, "D", 0.09, True),
              ("own", "female_chooser_male_target", pp.MEN, "o", -0.09, False), ("panel", "female_chooser_male_target", pp.MEN, "o", -0.27, True))
    for src, d, color, marker, off, hollow in combos:
        for yi, (t, _) in zip(yb, terms):
            c = w[src][d]["coefficients_pp_per_sd"][t]
            ax.errorbar(c["estimate"], yi + off, xerr=[[c["estimate"] - c["interval_95"][0]], [c["interval_95"][1] - c["estimate"]]],
                        fmt=marker, ms=3.2, color=color, mfc="white" if hollow else color, elinewidth=0.75, capsize=1.4)
    ax.axvline(0, color=pp.MUTED, lw=0.6, ls=":")
    ax.set_yticks(yb, [lbl for _, lbl in terms], fontsize=6.0)
    ax.set_xlabel("Yes decision, pp per SD", labelpad=2.5)
    ax.grid(axis="x", color=pp.GRID, lw=0.5)
    pp.panel(ax, "D", "Speed dating: own vs. other raters", fontsize=7.6, pad=4)
    ax.grid(axis="y", visible=False)
    handles = [Line2D([], [], color=pp.WOMEN, marker="D", ls="", ms=3.2, label="Female target"),
               Line2D([], [], color=pp.MEN, marker="o", ls="", ms=3.2, label="Male target"),
               Line2D([], [], color=pp.INK2, marker="s", ls="", ms=3.2, mfc="white", label="Other raters (hollow)")]
    ax.legend(handles=handles, loc="lower right", fontsize=5.0, handlelength=0.9)

    # E. Proposition 25 / E25
    ax = fig.add_subplot(gs[1, 1])
    e25 = conf["proposition_25_e25"]
    ks = [row["k_contexts"] for row in e25["by_k"]]
    kx = np.array(ks, dtype=float)
    kx_plot = np.where(kx == 0, 0.5, kx)
    for key, color, marker, label in (("selection_differential_outward_formidability", pp.AQUA, "s", "Outward (rivals)"),
                                      ("selection_differential_inward_dominance", pp.RED, "^", "Inward (partner/team)")):
        cf = [row[key]["closed_form"] for row in e25["by_k"]]
        mc = [row[key]["monte_carlo"] for row in e25["by_k"]]
        ax.plot(kx_plot, cf, color=color, lw=1.1, label=label)
        ax.scatter(kx_plot, mc, color=color, marker=marker, s=11, zorder=3, edgecolor="white", lw=0.4)
    ax.axhline(0, color=pp.MUTED, lw=0.6)
    ax.axvline(e25["witness_threshold_k_star"], color=pp.INK2, lw=0.7, ls="--")
    ax.text(e25["witness_threshold_k_star"] * 1.08, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 0.5, f"$k^*={e25['witness_threshold_k_star']:.0f}$",
            fontsize=5.8, color=pp.INK2, va="top")
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32], ["0", "1", "2", "4", "8", "16", "32"])
    ax.set_xlabel("Witnessed contexts $k$", labelpad=2.5)
    pp.panel(ax, "E", "Prop. 25: dominance direction", ylabel="Selection differential", fontsize=7.6, pad=4)
    ax.legend(loc="center right", fontsize=5.2, handlelength=1.2)

    # F. Proposition 26 / E26
    ax = fig.add_subplot(gs[1, 2])
    e26 = conf["proposition_26_e26"]
    ms = [row["raters_m"] for row in e26["by_raters"]]
    ratio_cf = [row["closed_form"]["penalized_group"]["funded_rate"] / row["closed_form"]["reference_group"]["funded_rate"] for row in e26["by_raters"]]
    ratio_mc = [row["monte_carlo"]["penalized_group"]["funded_rate"] / row["monte_carlo"]["reference_group"]["funded_rate"] for row in e26["by_raters"]]
    gap_cf = [row["outcome_test_quality_gap_funded"]["closed_form"] for row in e26["by_raters"]]
    gap_mc = [row["outcome_test_quality_gap_funded"]["monte_carlo"] for row in e26["by_raters"]]
    ax.plot(ms, ratio_cf, color=pp.VIOLET, lw=1.1, label="Funding-rate ratio")
    ax.scatter(ms, ratio_mc, color=pp.VIOLET, marker="s", s=11, zorder=3, edgecolor="white", lw=0.4)
    ax.axhline(1, color=pp.MUTED, lw=0.6, ls=":")
    ax.set_xscale("log", base=4)
    ax.set_xticks(ms, [str(m) for m in ms])
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Raters $m$ sharing the same stereotype", labelpad=2.5)
    pp.panel(ax, "F", "Prop. 26: shared-stereotype ceiling", ylabel="Penalized / reference funded", fontsize=7.6, pad=4)
    ax2 = ax.twinx()
    ax2.plot(ms, gap_cf, color=pp.GOLD, lw=1.0, ls="--", label="Outcome-test gap")
    ax2.scatter(ms, gap_mc, color=pp.GOLD, marker="^", s=11, zorder=3, edgecolor="white", lw=0.4)
    ax2.set_ylabel("Funded quality gap (SD)", fontsize=6.2, color=pp.GOLD, labelpad=2)
    ax2.tick_params(axis="y", colors=pp.GOLD, labelsize=6.0)
    ax2.spines["right"].set_visible(True)
    ax2.set_ylim(0, max(gap_cf + gap_mc) * 1.25)
    sym = e26["symmetric_protocol_m16"]
    try:
        sr = _ratio(sym["closed_form"] if "closed_form" in sym else sym)
        ax.scatter([16], [sr], marker="*", s=42, color=pp.AQUA, zorder=4, label="Symmetric protocol")
    except KeyError:
        pass
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=5.0, handlelength=1.2)
    return pp.pdf(fig)


FACE_ZONE_SHORT = {
    "eyes_iris": "eyes and iris", "brows": "brows", "tear_trough": "tear trough", "lips_vermilion": "lips",
    "nose_alar_tip": "nasal tip", "nose_bridge": "nasal bridge", "malar_cheek": "cheekbones", "nasolabial": "nasolabial fold",
    "jaw_chin": "jaw and chin", "philtrum": "philtrum", "forehead_glabella": "forehead", "lateral_skin": "lateral cheek skin",
}


def face_highres_macros(data) -> dict[str, str]:
    """Macros for the high-resolution facial evidence maps (filled once the frozen reference exists)."""
    fh = data["face_highres"]
    v: dict[str, str] = {}
    ex = fh["extraction"]
    v["PboFhGrid"] = str(ex["grid"])
    v["PboFhAtlas"] = str(ex["atlas_px"])
    v["PboFhPortrait"] = str(ex["portrait_px"])
    v["PboFhN"] = f"{ex['n_ok']:,}".replace(",", "{,}")
    v["PboFhIod"] = num(ex["source_interocular_px"]["median"], 0)
    v["PboFhCells"] = str(fh["modelled_cells"]["face_oval_cells"])
    v["PboFhLegacyCells"] = str(fh["modelled_cells"]["legacy_face_oval_cells"])
    v["PboFhPcaVar"] = num(100 * fh["token_pca"]["explained_variance"], 1)
    v["PboFhMinTierN"] = str(min(fh["by_sex"][g]["tiers"][t]["n_faces"] for g in ("Female", "Male") for t in ("top15", "bot15")))
    for g, tag in (("Female", "W"), ("Male", "M")):
        r = fh["by_sex"][g]
        pm = r["protocol_matched_test"]
        v[f"PboFhRhoGrid{tag}"] = num(pm["vitb16_672_grid42"]["spearman_r"], 3)
        v[f"PboFhRhoPooled{tag}"] = num(pm["vitb16_672_pooled_grid14"]["spearman_r"], 3)
        v[f"PboFhRhoCls{tag}"] = num(pm["vitb16_672_global_pool_only"]["spearman_r"], 3)
        if "published_vits16_224_grid14" in pm:
            v[f"PboFhRhoLegacy{tag}"] = num(pm["published_vits16_224_grid14"]["spearman_r"], 3)
            d = r["paired_bootstrap_grid42_minus"]["published_vits16_224_grid14"]
            v[f"PboFhDeltaLegacy{tag}"] = num(d["delta_spearman"], 3, plus=True)
            v[f"PboFhDeltaLegacy{tag}CI"] = ci(*d["ci_95"], d=3)
        d = r["paired_bootstrap_grid42_minus"]["vitb16_672_pooled_grid14"]
        v[f"PboFhDeltaPooled{tag}"] = num(d["delta_spearman"], 3, plus=True)
        v[f"PboFhDeltaPooled{tag}CI"] = ci(*d["ci_95"], d=3)
        v[f"PboFhTestN{tag}"] = str(r["n_test_protocol"])
        v[f"PboFhTrainN{tag}"] = str(r["n_train_protocol"])
        v[f"PboFhCrossRho{tag}"] = num(r["cross_fitted_all_faces"]["spearman_r"], 3)
        v[f"PboFhPubRho{tag}"] = num(r["published_multiblock_reference"]["spearman_r"], 3)
        for tier, t_tag in (("top15", "Top"), ("bot15", "Bot")):
            c = r["tiers"][tier]
            v[f"PboFh{t_tag}N{tag}"] = str(c["n_faces"])
            if c["recovered_share_of_actual"] is not None:
                v[f"PboFh{t_tag}Recovered{tag}"] = num(100 * c["recovered_share_of_actual"], 0)
            v[f"PboFh{t_tag}SigShare{tag}"] = num(100 * c["cells_ci_excluding_zero_share"], 0)
            if c["reliability"]["spearman_brown_full"] is not None:
                v[f"PboFh{t_tag}Reliab{tag}"] = num(c["reliability"]["spearman_brown_full"], 2)
            v[f"PboFh{t_tag}NullCorr{tag}"] = num(c["null_map_corr_with_real"], 2, plus=True)
            dz = c.get("densest_zone")
            if dz and dz["per_cell_over_oval_mean"] is not None:
                v[f"PboFh{t_tag}DenseZone{tag}"] = FACE_ZONE_SHORT[dz["zone_key"]]
                v[f"PboFh{t_tag}DenseRatio{tag}"] = num(dz["per_cell_over_oval_mean"], 1)
        if r["tail_legibility"]["bottom_over_top"] is not None:
            v[f"PboFhTailRatio{tag}"] = num(r["tail_legibility"]["bottom_over_top"], 2)
        za = r["published_zone_agreement"]
        v[f"PboFhZoneTop{tag}"] = num(za["top15_rank_spearman"], 2, plus=True)
        v[f"PboFhZoneBot{tag}"] = num(za["bot15_rank_spearman"], 2, plus=True)
        dt = fh.get("deletion_test")
        if dt:
            c = dt[g]
            v[f"PboFhDelDrv{tag}"] = num(c["drivers_deleted_delta_pct"]["estimate"], 1, plus=True)
            v[f"PboFhDelRej{tag}"] = num(c["rejectors_deleted_delta_pct"]["estimate"], 1, plus=True)
            v[f"PboFhDelRnd{tag}"] = num(c["random_deleted_delta_pct"]["estimate"], 1, plus=True)
            v[f"PboFhDelDrvMinusRnd{tag}"] = num(c["drivers_minus_random_pct"]["estimate"], 1, plus=True)
            v[f"PboFhDelDrvMinusRnd{tag}CI"] = ci(*c["drivers_minus_random_pct"]["ci_95"])
            v[f"PboFhDelRejMinusRnd{tag}"] = num(c["rejectors_minus_random_pct"]["estimate"], 1, plus=True)
            v[f"PboFhDelRejMinusRnd{tag}CI"] = ci(*c["rejectors_minus_random_pct"]["ci_95"])
            v[f"PboFhDelN{tag}"] = str(c["drivers_deleted_delta_pct"]["n"])
    return v


def v16_audit_macros(data) -> dict[str, str]:
    """Uncertainty and cell sizes for the v16 speed-dating and E24 statements the v17 text qualifies."""
    cb = data["speed_dating"]["confidence_and_backlash"]
    aw = cb["agency_x_warmth_dyadic"]
    v: dict[str, str] = {}
    v["PboAuditColdNW"] = str(aw["male_chooser_to_female_target"]["cold_arrogant_high_agency"]["n"])
    v["PboAuditColdNM"] = str(aw["female_chooser_to_male_target"]["cold_arrogant_high_agency"]["n"])
    race = cb["audience_race_modulation"]
    for key, tag in (("black", "Black"), ("white_euro", "White"), ("latino", "Latino"), ("asian", "Asian")):
        v[f"PboAuditRace{tag}N"] = f"{race[key]['n']:,}".replace(",", "{,}")
        for coef, ctag in (("amb", "Amb"), ("fun", "Fun"), ("target_overclaim", "Over")):
            v[f"PboAuditRace{tag}{ctag}CI"] = ci(*race[key]["coefficients"][coef]["ci_95_pp"])
    age = cb["audience_age_modulation"]
    for key, tag in (("age_le_24", "Young"), ("age_25_28", "Mid"), ("age_ge_29", "Mature")):
        v[f"PboAuditAge{tag}N"] = f"{age[key]['n']:,}".replace(",", "{,}")
        v[f"PboAuditOver{tag}CI"] = ci(*age[key]["coefficients"]["target_overclaim"]["ci_95_pp"])
        v[f"PboAuditAmb{tag}CI"] = ci(*age[key]["coefficients"]["amb"]["ci_95_pp"])
    st = cb["status_threat"]
    v["PboAuditOutsmartWCI"] = ci(*st["male_chooser_evaluating_woman"]["coefficients"]["target_outsmarts_chooser"]["ci_95_pp"])
    v["PboAuditOutsmartMCI"] = ci(*st["female_chooser_evaluating_man"]["coefficients"]["target_outsmarts_chooser"]["ci_95_pp"])
    e3 = data["open_synthesis"]["proposition_24_e24"]["part_iii_alpha_double_bind_escape"]
    v["PboAuditEscapeW"] = num(e3["alpha_female_dual_signal_high_caliber_pool"]["closed_form"], 2)
    v["PboAuditSmallW"] = num(e3["alpha_female_playing_small_insecure_pool"]["closed_form"], 2)
    v["PboAuditUncoupledW"] = num(e3["alpha_female_uncoupled_insecure_pool"]["closed_form"], 2)
    v["PboAuditAlphaM"] = num(e3["alpha_male_standard"]["closed_form"], 2)
    kw = data["open_synthesis"]["influence_and_prestige"]["kowal_175_countries"]["archetypes_by_sex"]
    v["PboAuditKowColdPPM"] = num(kw["men"]["warm_outlier_minus_cold_partnered_pp"], 1, plus=True)
    v["PboAuditKowColdPPW"] = num(kw["women"]["warm_outlier_minus_cold_partnered_pp"], 1, plus=True)
    v["PboAuditKowHumblePPW"] = num(kw["women"]["warm_outlier_minus_humble_partnered_pp"], 1, plus=True)
    return v


# ----------------------------------------------------------------------------- high-resolution facial evidence
FACE_HR_NPZ = "evidence/reference/performance-face-highres.npz"
_PORTRAIT_X0, _PORTRAIT_Y0, _PORTRAIT_SPAN = -10.0, -2.0, 244.0  # performance_face_attribution.portrait_template


def load_face_highres_npz(data, root):
    import hashlib
    from io import BytesIO

    raw = (root / FACE_HR_NPZ).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != data["face_highres"]["npz_sha256"]:
        raise ValueError(f"{FACE_HR_NPZ}: sha256 {got} != reference {data['face_highres']['npz_sha256']}")
    return np.load(BytesIO(raw))


def atlas_grid_to_portrait(grid, size, order=3):
    """Resample a GRIDxGRID map on the 224-unit atlas frame into the padded size x size portrait frame."""
    from scipy.ndimage import map_coordinates

    n = grid.shape[0]
    px = (np.arange(size, dtype=np.float64) + 0.5) * _PORTRAIT_SPAN / size
    x224, y224 = px + _PORTRAIT_X0, px + _PORTRAIT_Y0
    gx = x224 * n / 224.0 - 0.5
    gy = y224 * n / 224.0 - 0.5
    yy, xx = np.meshgrid(gy, gx, indexing="ij")
    return map_coordinates(np.asarray(grid, dtype=np.float64), [yy, xx], order=order, mode="constant", cval=0.0)


def _hr_overlay(rgb_u8, map42, sig42, cells42, sil_u8, vmax):
    """Diverging cell map over a luminance relief of the composite; non-significant cells are faded."""
    from scipy.ndimage import gaussian_filter

    import performance_paper as pp

    size = rgb_u8.shape[0]
    rgb = rgb_u8.astype(np.float32) / 255.0
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    lum = lum + 0.35 * (lum - gaussian_filter(lum, sigma=4.0))
    relief = np.clip(0.30 + 0.66 * lum, 0.0, 1.0)
    base = np.stack([relief * 0.965, relief * 0.972, relief * 0.988], axis=-1)
    m = atlas_grid_to_portrait(map42, size, order=3)
    keep = atlas_grid_to_portrait(cells42.astype(np.float64), size, order=1)
    sig = atlas_grid_to_portrait(sig42.astype(np.float64), size, order=1)
    sil = gaussian_filter(sil_u8.astype(np.float32) / 255.0, sigma=3.0)
    mask = np.clip(np.minimum(keep * 1.4, sil * 1.6), 0.0, 1.0)
    norm = np.clip(m / vmax, -1.0, 1.0) * mask
    heat = pp.FACE_CMAP((norm + 1.0) * 0.5)[..., :3]
    alpha = (np.abs(norm) ** 0.76) * 0.86 * mask * (0.30 + 0.70 * np.clip(sig, 0.0, 1.0))
    out = (1.0 - alpha[..., None]) * base + alpha[..., None] * heat
    bg = np.array([0.969, 0.973, 0.980])[None, None, :]
    frame = np.clip(sil * 1.25, 0.0, 1.0)[..., None]
    return np.clip(frame * out + (1.0 - frame) * bg, 0.0, 1.0)


def face_highres_plot(data, root) -> bytes:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    import performance_paper as pp

    fh = data["face_highres"]
    z = load_face_highres_npz(data, root)
    cells = z["cells_mask_42"]
    sil = z["silhouette_768"]
    vmax = max(float(np.percentile(np.abs(np.concatenate([z[f"map42__{g}__{t}"][cells > 0]
                                                           for g in ("Female", "Male") for t in ("top15", "bot15")])), 98.5)), 1e-3)
    fig = plt.figure(figsize=(7.16, 4.30))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.80, 1.80], wspace=0.30, left=0.035, right=0.995, top=0.93, bottom=0.10)
    maps = outer[0].subgridspec(3, 3, height_ratios=[1, 1, 0.075], hspace=0.20, wspace=0.05)
    side = outer[1].subgridspec(2, 1, hspace=0.62)
    cols = (("top15", "map42", "Top 15%: drivers"), ("bot15", "map42", "Bottom 15%: rejectors"), ("top15", "null42", "Shuffled labels: null"))
    for r, (g, label, color) in enumerate((("Female", "A   Women", pp.WOMEN), ("Male", "B   Men", pp.MEN))):
        t = fh["by_sex"][g]["tiers"]
        for c, (tier, kind, title) in enumerate(cols):
            ax = fig.add_subplot(maps[r, c])
            comp = z[f"comp768__tier__{g}__{'mid' if kind == 'null42' else tier}"]
            sig = z[f"sig42__{g}__{tier}"] if kind == "map42" else np.zeros_like(cells)
            ax.imshow(_hr_overlay(comp, z[f"{kind}__{g}__{tier}"], sig, cells, sil, vmax), interpolation="bicubic")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#D6D5CE")
                sp.set_linewidth(0.6)
            if r == 0:
                ax.set_title(title, fontsize=6.9, fontweight="bold", pad=3.0, color=pp.INK)
            if kind == "map42":
                badge = (f"n = {t[tier]['n_faces']:,}; {100 * t[tier]['cells_ci_excluding_zero_share']:.0f}% cells CI$\\neq$0\n"
                         f"split-half r = {t[tier]['reliability']['spearman_brown_full']:.2f}")
            else:
                badge = f"r(null, real) = {t[tier]['null_map_corr_with_real']:+.2f}"
            ax.text(0.5, 0.025, badge, transform=ax.transAxes, ha="center", va="bottom", fontsize=5.2, color=pp.INK,
                    linespacing=1.0, bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "#D8D7D2", "alpha": 0.9, "lw": 0.4})
            if c == 0:
                ax.text(-0.07, 0.5, label.split()[-1], transform=ax.transAxes, rotation=90, ha="center", va="center",
                        fontsize=7.4, fontweight="bold", color=color)
                ax.text(0.0, 1.13 if r == 0 else 1.02, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.0,
                        fontweight="bold", color=pp.INK)
    cax = fig.add_subplot(maps[2, :])
    cb = fig.colorbar(ScalarMappable(Normalize(-vmax, vmax), cmap=pp.FACE_CMAP), cax=cax, orientation="horizontal")
    cb.set_label("Cross-fitted contribution of a 42$\\times$42 cell to the predicted percentile (pp)", fontsize=6.0, labelpad=2)
    cb.ax.tick_params(labelsize=5.6, length=2)
    cb.outline.set_linewidth(0.4)

    arms = (("published_vits16_224_grid14", "v16 encoder\n224 px, 14$\\times$14"),
            ("vitb16_672_pooled_grid14", "Same 672 px\npooled 14$\\times$14"),
            ("vitb16_672_global_pool_only", "Same 672 px\none face vector"))
    ax = fig.add_subplot(side[0])
    for k, (g, color, marker, dy) in enumerate((("Female", pp.WOMEN, "D", -0.13), ("Male", pp.MEN, "o", 0.13))):
        pb = fh["by_sex"][g]["paired_bootstrap_grid42_minus"]
        for i, (arm, _) in enumerate(arms):
            if arm not in pb:
                continue
            d = pb[arm]
            ax.errorbar(d["delta_spearman"], i + dy, xerr=[[d["delta_spearman"] - d["ci_95"][0]], [d["ci_95"][1] - d["delta_spearman"]]],
                        fmt=marker, color=color, ms=3.4, elinewidth=0.85, capsize=1.6, mfc=color if marker == "D" else "white",
                        label=("Women" if g == "Female" else "Men") if i == 0 else None)
    ax.axvline(0.0, color=pp.MUTED, lw=0.7, ls="--")
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([a[1] for a in arms], fontsize=5.6, linespacing=1.0)
    ax.invert_yaxis()
    ax.grid(axis="x", color=pp.GRID, lw=0.5)
    # One shared sex legend in the free top-right corner, so it never sits on top of an interval in C or D.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 0.999), ncol=2, fontsize=5.8,
               handlelength=1.0, columnspacing=0.9, frameon=False)
    nt = fh["by_sex"]["Female"]["n_test_protocol"] + fh["by_sex"]["Male"]["n_test_protocol"]
    ax.text(0.98, 0.60, f"{nt} held-out faces\npaired bootstrap", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.4, color=pp.MUTED, linespacing=1.0)
    pp.panel(ax, "C", "More pixels, same accuracy", xlabel="$\\Delta$ Spearman, 42$\\times$42 minus row (95% CI)", fontsize=7.4, pad=4)
    ax.grid(axis="y", visible=False)

    ax = fig.add_subplot(side[1])
    dt = fh.get("deletion_test")
    if dt:
        keys = (("drivers_deleted_delta_pct", "Drivers"), ("random_deleted_delta_pct", "Random"), ("rejectors_deleted_delta_pct", "Rejectors"))
        for g, color, marker, dx in (("Female", pp.WOMEN, "D", -0.13), ("Male", pp.MEN, "o", 0.13)):
            for i, (k, _) in enumerate(keys):
                c = dt[g][k]
                ax.errorbar(i + dx, c["estimate"], yerr=[[c["estimate"] - c["ci_95"][0]], [c["ci_95"][1] - c["estimate"]]],
                            fmt=marker, color=color, ms=3.4, elinewidth=0.85, capsize=1.6, mfc=color if marker == "D" else "white")
        ax.axhline(0.0, color=pp.MUTED, lw=0.7, ls="--")
        ax.set_xticks(range(3))
        ax.set_xticklabels([k[1] for k in keys], fontsize=5.8)
        n = dt["Female"]["drivers_deleted_delta_pct"]["n"] + dt["Male"]["drivers_deleted_delta_pct"]["n"]
        ax.text(0.02, 0.97, "cells replaced by the mean face,\nre-encoded, re-scored", transform=ax.transAxes, ha="left", va="top",
                fontsize=5.4, color=pp.MUTED, linespacing=1.0)
        pp.panel(ax, "D", f"Deleting mapped cells moves scores (n = {n})", ylabel="$\\Delta$ predicted percentile (pp)", fontsize=7.4, pad=4)
    else:
        ax.axis("off")
    return pp.pdf(fig, dpi=300)
