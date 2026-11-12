#!/usr/bin/env python3
"""Write paper/generated/numbers.tex (macros) and table bodies from the results
files, so that no numerical value in the manuscript is transcribed by hand."""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); GEN = os.path.join(ROOT, "paper", "generated")
os.makedirs(GEN, exist_ok=True)
J = lambda n: json.load(open(os.path.join(RES, n + ".json")))
M = {}

def sci(x, d=1):
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    if round(m, d) >= 10:
        m /= 10; e += 1
    return f"{m:.{d}f}\\times10^{{{e}}}"

def th(x):
    return f"{x:,.0f}".replace(",", "{,}")

def rng_(lo, hi, d=1):
    return f"{lo:.{d}f}--{hi:.{d}f}"

# ---------------- fossils
F = J("e13_fossils"); m = F["models"]; st = F["step"]; g = F["groups"]; w = F["welch"]; se = F["series"]
r = [v * 1e4 for v in F["rates"].values()]
M.update(FosN=F["n"], FosSpan=f"{F['span']:.2f}", FosSlopeSpec=f"{F['slope_specimen']:.3f}", FosSlopeSpecSE=f"{F['se_specimen']:.3f}",
         FosSlopeBin=f"{F['slope_binned']:.3f}", FosSlopeBinSE=f"{F['se_binned']:.3f}", FosBins=F["nbins"], FosSDw=f"{F['sd_within']:.3f}",
         FosRateLo=f"{min(r):.1f}", FosRateHi=f"{max(r):.1f}",
         FosAICLin=f"{m['linear']['daicc']:.0f}", FosAICStep=f"{m['stasis_step']['daicc']:.1f}", FosAICPub=f"{m['published_phases']['daicc']:.1f}",
         FosAICOne=f"{m['pw1']['daicc']:.0f}", FosBICLin=f"{m['linear']['dbic']:.0f}", FosBICStep=f"{m['stasis_step']['dbic']:.1f}", FosBICFree=f"{m['pw3']['dbic']:.1f}",
         FosBpA=f"{m['stasis_step']['bps'][0]:.2f}", FosBpB=f"{m['stasis_step']['bps'][1]:.2f}", FosBpC=f"{m['stasis_step']['bps'][2]:.2f}",
         FosCIA=rng_(*F["bp_ci"][0]), FosCIB=rng_(*F["bp_ci"][1]), FosCIC=rng_(*F["bp_ci"][2]), FosBoot=F["nboot"],
         FosLevI=th(st["level_I"]), FosLevIII=th(st["level_III"]), FosLevEnd=th(st["end"]), FosSlII=f"{st['slope_II']:.2f}", FosSlIV=f"{st['slope_IV']:.2f}",
         FosRateII=f"{st['rate_II'] * 1e4:.1f}", FosRateIV=f"{st['rate_IV'] * 1e4:.1f}",
         FosFreeSlA=f"{m['pw3']['coef'][1]:.2f}", FosFreeSlB=f"{m['pw3']['coef'][2]:.2f}", FosFreeSlC=f"{m['pw3']['coef'][3]:.2f}", FosFreeSlD=f"{m['pw3']['coef'][4]:.2f}",
         FosFreeBpA=f"{m['pw3']['bps'][0]:.1f}", FosFreeBpB=f"{m['pw3']['bps'][1]:.1f}", FosFreeBpC=f"{m['pw3']['bps'][2]:.2f}",
         FosStI=f"{F['stasis']['I']['slope']:.2f}", FosStISE=f"{F['stasis']['I']['se']:.2f}", FosStIN=F["stasis"]["I"]["n"],
         FosStIII=f"{F['stasis']['III']['slope']:.2f}", FosStIIISE=f"{F['stasis']['III']['se']:.2f}", FosStIIIN=F["stasis"]["III"]["n"],
         NeaN=g["neanderthal"]["n"], NeaMean=th(g["neanderthal"]["mean"]), NeaSD=f"{g['neanderthal']['sd']:.0f}",
         SapN=g["sapiens_pleistocene"]["n"], SapMean=th(g["sapiens_pleistocene"]["mean"]), SapSD=f"{g['sapiens_pleistocene']['sd']:.0f}",
         MPN=g["mid_pleistocene"]["n"], MPMean=th(g["mid_pleistocene"]["mean"]), MPSD=f"{g['mid_pleistocene']['sd']:.0f}",
         WelchDiff=f"{w['diff_cc']:.0f}", WelchPct=f"{(w['ratio'] - 1) * 100:.0f}", WelchP=f"{w['p']:.2f}", WelchD=f"{w['cohen_d']:.1f}",
         SapWinN=w["sapiens_same_window_n"], SapWinMean=th(w["sapiens_same_window_mean"]),
         SapOld=th(g["sapiens_over_100ka"]["mean"]), SapOldN=g["sapiens_over_100ka"]["n"], SapMid=th(g["sapiens_100_30ka"]["mean"]), SapMidN=g["sapiens_100_30ka"]["n"],
         SapLate=th(g["sapiens_30_10ka"]["mean"]), SapLateN=g["sapiens_30_10ka"]["n"],
         RecentDrop=f"{(1 - 1349 / g['sapiens_30_10ka']['mean']) * 100:.0f}",
         MPEurMean=th(g["mp_europe"]["mean"]), MPEurN=g["mp_europe"]["n"], MPAsiaMean=th(g["mp_asia"]["mean"]), MPAsiaN=g["mp_asia"]["n"],
         SerEur=f"{se['europe_w_asia']['slope']:.2f}", SerEurCI=rng_(se["europe_w_asia"]["lo"], se["europe_w_asia"]["hi"], 2), SerEurN=se["europe_w_asia"]["n"],
         SerAfr=f"{se['africa_to_sapiens']['slope']:.2f}", SerAfrCI=rng_(se["africa_to_sapiens"]["lo"], se["africa_to_sapiens"]["hi"], 2), SerAfrN=se["africa_to_sapiens"]["n"],
         SerAsia=f"{se['east_asia_archaic']['slope']:.2f}", SerAsiaCI=rng_(se["east_asia_archaic"]["lo"], se["east_asia_archaic"]["hi"], 2), SerAsiaN=se["east_asia_archaic"]["n"])

# ---------------- rates
E7 = J("e7_rates")
M.update(RateTotalSD=f"{E7['total_sd']:.1f}", RateGens=th(E7["generations"]), RateBase=sci(E7["rate"], 2), RatePhase=sci(E7["phase_rate"]),
         RateGridMax=sci(max(max(v) for v in E7["table2"].values())), ReqBase=sci(E7["base_required"]),
         ReqLo=sci(min(min(v) for v in E7["required"].values())), ReqHi=sci(max(max(v) for v in E7["required"].values())),
         MutZLo=f"{min(E7['mutational_z']):.1f}", MutZHi=f"{max(E7['mutational_z']):.1f}",
         MultLoA=f"{E7['multiples']['0.3'][0]:.0f}", MultLoB=f"{E7['multiples']['1.0'][0]:.0f}", MultHiA=f"{E7['multiples']['0.3'][2]:.0f}", MultHiB=f"{E7['multiples']['1.0'][2]:.0f}")
rows = []
for gt in ("20", "25", "30"):
    rows.append(f"{gt} & " + " & ".join(f"${sci(v)}$" for v in E7["table2"][gt]) + r" \\")
open(os.path.join(GEN, "tab_rates.tex"), "w").write("\n".join(rows))
rows = []
for h in ("0.3", "0.5", "0.8"):
    rows.append(f"{h} & " + " & ".join(f"{v:.1f}" for v in E7["table3"][h]) + r" \\")
open(os.path.join(GEN, "tab_drift.tex"), "w").write("\n".join(rows))
rows = [r"\multicolumn{4}{@{}l}{\textit{(a) Required differential per generation}}\\",
        r" & $r_g=0.1$ & $r_g=0.24$ & $r_g=0.5$ \\ \midrule"]
for k, lab in (("0.3_0.5", r"$h^2_c{=}0.3,\ h^2_b{=}0.5$"), ("0.5_0.8", r"$h^2_c{=}0.5,\ h^2_b{=}0.8$")):
    rows.append(lab + " & " + " & ".join(f"${sci(v)}$" for v in E7["required"][k]) + r" \\")
rows += [r"\addlinespace\multicolumn{4}{@{}l}{\textit{(b) Available through men's standing (multiple of $" + sci(E7["base_required"]) + r"$)}}\\",
         r" & $\rho_{ZC}=0.05$ & $\rho_{ZC}=0.1$ & $\rho_{ZC}=0.3$ \\ \midrule"]
for I in ("0.3", "0.6", "1.0"):
    rows.append(f"$I={I}$ & " + " & ".join(f"${sci(a)}$ ({b:.0f})" if b >= 9.5 else f"${sci(a)}$ ({b:.1f})" for a, b in zip(E7["available"][I], E7["multiples"][I])) + r" \\")
open(os.path.join(GEN, "tab_envelope.tex"), "w").write("\n".join(rows))

# ---------------- core experiments
E1, E2, E3, E4, E5, E6, E8, E9 = (J(n) for n in ("e1_baseline", "e2_routes", "e3_portfolio", "e4_multigen", "e5_matching", "e6_persistence", "e8_power", "e9_identity"))
r32 = [x for x in E1["rows"] if x["n"] == 32 and x["rho"] == 0.0][0]
M.update(EOneMaxDiff=sci(E1["max_abs_diff"]), EOneExact=f"{r32['exact']:.6f}", EOneMC=f"{r32['mc']:.6f}",
         ETwoMaxZ=f"{E2['max_abs_z']:.1f}", ETwoN=len(E2["rows"]))
v = {(x["T"], x["mode"]): x for x in E3["validity"]}; s = {x["K"]: x for x in E3["selection"]}
M.update(EThreeMaxZ=f"{E3['max_abs_z']:.1f}", EThreeN=E3["n_comparisons"], PortDiv=f"{v[(64, 'diverse')]['exact']:.2f}", PortRep=f"{v[(64, 'repeat')]['exact']:.2f}",
         PortCeilS=f"{E3['ceiling_single']:.2f}", PortCeilP=f"{E3['ceiling_portfolio']:.2f}", PortSGa=f"{s[1]['ex_G']:.3f}", PortSGb=f"{s[24]['ex_G']:.3f}",
         PortSGratio=f"{s[24]['ex_G'] / s[1]['ex_G']:.2f}", PortSSa=f"{s[1]['ex_S']:.3f}", PortSSb=f"{s[24]['ex_S']:.4f}", PortSSratio=f"{s[1]['ex_S'] / s[24]['ex_S']:.0f}")
c = E4["conds"]
M.update(MGva=", ".join(f"{x['final_va']:.2f}" for x in c[1:4]) + f", and {c[4]['final_va']:.2f}",
         MGratio=", ".join(f"{x['ratio_vs_random']:.2f}$\\times$" for x in c[1:4]) + f", and {c[4]['ratio_vs_random']:.2f}$\\times$",
         MGF=", ".join(f"{x['F_eq']:.2f}" for x in c[1:4]) + f", and {c[4]['F_eq']:.1f}",
         MGerr=f"{E4['max_rel_err'] * 100:.0f}", MGinflLo=f"{c[1]['final_va'] / 0.4:.1f}", MGinflHi=f"{c[4]['final_va'] / 0.4:.1f}",
         MGFmod=rng_(*E4["F_m044"], 1), MGFhiLo=f"{E4['F_table']['0.2'][3]:.1f}", MGFhiHi=f"{E4['F_table']['0.8'][3]:.1f}")
rows = []
for h in ("0.2", "0.4", "0.6", "0.8"):
    rows.append(f"{h} & " + " & ".join((f"{x:.2f}" if np.isfinite(x) else r"$\infty$") for x in E4["F_table"][h]) + r" \\")
open(os.path.join(GEN, "tab_F.tex"), "w").write("\n".join(rows))
mm = E5["matching"]; hh = E5["hiding"]
M.update(DAmZero=f"{mm[0]['m']:.2f}", DAmOne=f"{mm[3]['m']:.2f}", DAmTwo=f"{mm[4]['m']:.2f}", DAdiff=sci(E5["max_resp_diff"]),
         HideThr=f"{hh[2]['threshold']:.2f}", HideRand=f"{hh[2]['random']:.2f}", HideThrMid=f"{hh[1]['threshold']:.2f}")
rows = [r"\multicolumn{4}{@{}l}{\textit{(a) Deferred acceptance with idiosyncratic taste ($b=0.2$, $V_A=0.4$)}}\\",
        r"Taste SD $s$ & Partner corr.\ $m$ & Response & $bV_A(1+m)/2$ \\ \midrule"]
for x in mm:
    rows.append(f"{x['s']:g} & {x['m']:.3f} & {x['resp']:.4f} & {x['pred']:.4f} \\\\")
rows += [r"\addlinespace\multicolumn{4}{@{}l}{\textit{(b) Effective validity when a fraction $q$ does not display}}\\",
         r"Hiding $q$ & $(1-q)R$ & Random hiding & Threshold hiding \\ \midrule"]
for x in hh:
    rows.append(f"{x['q']:g} & {x['naive']:.2f} & {x['random']:.3f} & {x['threshold']:.3f} \\\\")
open(os.path.join(GEN, "tab_robust.tex"), "w").write("\n".join(rows))
M.update(PersSim=f"{E6['eq_sim_mean']:.3f}", PersSD=f"{E6['eq_sim_sd']:.3f}", PersPred=f"{E6['eq_pred']:.3f}",
         TisPos=", ".join(f"{x:.0f}" for x in E6["phase_ends"]["positional"][:4]), TisBack=f"{E6['phase_ends']['positional'][4]:.1f}",
         TisEco=", ".join(f"{x:.2f}" for x in E6["phase_ends"]["ecological"][:3]) + f", and {E6['phase_ends']['ecological'][3]:.2f}",
         TisApproachLo=f"{E6['approach_time'][0]:.0f}", TisApproachHi=f"{E6['approach_time'][3]:.0f}")
p = {(x["clips"], x["observers"], x["targets"]): x for x in E8["rows"]}
M.update(PowSlope=f"{E8['mean_slope']:.3f}", PowOneLo=f"{round(E8['targets_for_90']['1_40'], -1):.0f}", PowOneHi=f"{round(E8['targets_for_90']['1_20'], -1):.0f}",
         PowFourA=f"{p[(4, 40, 400)]['power'] * 100:.0f}", PowFourB=f"{p[(4, 20, 600)]['power'] * 100:.0f}", PowOneAtFour=f"{p[(1, 40, 400)]['power'] * 100:.0f}")
M.update(ENineMaxZ=f"{E9['max_abs_z']:.1f}", ENineN=E9["n_comparisons"], PctEff=f"{E9['percentile_efficiency'] * 100:.1f}",
         LapWorst=f"{E9['laplace_worst']['rel_err'] * 100:.0f}", LapWorstR=f"{E9['laplace_worst']['R']:g}",
         FixLo=f"{E9['fixed_opportunity'][0]['mc']:.4f}", FixHi=f"{E9['fixed_opportunity'][1]['mc']:.4f}", FixRatio=f"{E9['fixed_ratio']:.2f}")
names = {"exponential": r"Exponential, $e^{0.3Z}$", "percentile": "Percentile acceptance", "truncation": "Truncation at median", "top_sixth_x4": r"Top sixth $\times4$"}
rows = []
for k, lab in names.items():
    rr = {x["R"]: x for x in E9["rows"] if x["rule"] == k}
    rows.append(f"{lab} & {rr[0.5]['cv']:.3f} & " + " & ".join(f"{rr[R]['resp_over_sqrtR']:.4f}" for R in (0.1, 0.5, 1.0)) + r" \\")
open(os.path.join(GEN, "tab_identity.tex"), "w").write("\n".join(rows))

# ---------------- Model V
A, Rp, D = J("e10_aim"), J("e11_repeat"), J("e12_despot")
ar = {x["regime"]: x for x in A["rows"] if x["rule"] == "exponential"}
o = {x["d"]: x for x in Rp["optimum"]}
dr = {(x["theta"], x["M"]): x for x in D["rows"] if x["q"] == 0.05}
Id, Ia, Ib = D["Npool"] - 1, dr[(1.0, 64)]["I_pool"], dr[(2.0, 64)]["I_pool"]
ia, ib = np.sqrt(Id / Ia), np.sqrt(Id / Ib)
M.update(AimMaxZ=f"{A['max_abs_z']:.1f}", AimN=A["n_comparisons"], AimGain=f"{A['aim_gain_G']:.1f}", AimLoss=f"{A['aim_loss_F']:.1f}",
         AimDomF=f"{ar['dominance']['exF']:.3f}", AimDomG=f"{ar['dominance']['exG']:.3f}", AimPreF=f"{ar['prestige']['exF']:.3f}", AimPreG=f"{ar['prestige']['exG']:.3f}",
         AimMixF=f"{ar['mixed']['exF']:.3f}", AimMixG=f"{ar['mixed']['exG']:.3f}",
         AimRatDom=f"{ar['dominance']['ratio_exact']:.2f}", AimRatPre=f"{ar['prestige']['ratio_exact']:.1f}",
         CueSQ=f"{A['cue_SQ']:.3f}", CueSG=sci(abs(A["cue_SG"])), CueSGse=sci(A["cue_SG_se"]),
         RepROne=f"{Rp['R1']:.2f}", RepCeil=f"{Rp['ceiling']:.2f}", RepMaxZ=f"{Rp['max_abs_z']:.1f}", RepNsafe=f"{o[0.001]['n_eff']:.0f}", RepRsafe=f"{o[0.001]['R']:.2f}",
         RepFrac=f"{o[0.001]['frac_ceiling'] * 100:.0f}", RepGain=f"{o[0.001]['gain_vs_single']:.2f}", RepNmid=f"{o[0.02]['n_eff']:.0f}", RepRmid=f"{o[0.02]['R']:.2f}",
         DesShareLin=f"{D['asymptote']['1.0_0.05']['share']:.2f}", DesShareSq=f"{D['asymptote']['2.0_0.05']['share']:.2f}",
         DesKLin=f"{D['asymptote']['1.0_0.05']['k']:.1f}", DesKSq=f"{D['asymptote']['2.0_0.05']['k']:.0f}",
         DesThrLin=f"{D['tolerance_threshold_M']['1.0_0.05']:.2f}", DesThrSq=f"{D['tolerance_threshold_M']['2.0_0.05']:.2f}",
         DesIdespot=f"{Id:.0f}", DesILin=f"{Ia:.1f}", DesISq=f"{Ib:.1f}", DesIntLo=f"{ia:.1f}", DesIntHi=f"{ib:.1f}",
         NetGLo=f"{A['aim_gain_G'] / ib:.1f}", NetGHi=f"{A['aim_gain_G'] / ia:.1f}", NetFLo=f"{A['aim_loss_F'] * ia:.0f}", NetFHi=f"{A['aim_loss_F'] * ib:.0f}")
rows = []
for Mv in (0.25, 1, 4, 16, 64):
    a, b = dr[(1.0, Mv)], dr[(2.0, Mv)]
    lab = "1/4" if Mv == 0.25 else f"{Mv:g}"
    rows.append(f"{lab} & {a['k_int']} & {1 - a['k_int'] * 0.05:.2f} & {a['I_pool']:.1f} & {b['k_int']} & {1 - b['k_int'] * 0.05:.2f} & {b['I_pool']:.1f} \\\\")
open(os.path.join(GEN, "tab_despot.tex"), "w").write("\n".join(rows))
rows = []
for k, lab in (("dominance", "Contest of force"), ("mixed", "Mixed portfolio"), ("prestige", "Skill and prestige")):
    x = ar[k]
    rows.append(f"{lab} & {x['lF']:.2f} & {x['lG']:.2f} & {x['exF']:.3f} & {x['exG']:.3f} & {x['ratio_exact']:.2f} \\\\")
open(os.path.join(GEN, "tab_aim.tex"), "w").write("\n".join(rows))

# ---------------- audit
Au = J("audit")
M.update(AudZ=f"{Au['A1_prop2_max_abs_z']:.1f}", AudQuad=sci(max(Au["A2_prop6_max_abs_err"], 1e-17), 0), AudRoot=sci(Au["A3_prop5_max_abs_err"], 0),
         AudCov=sci(max(Au["A4_prop3_max_abs_err"], Au["A5_prop8_max_abs_err"]), 0), AudFOC=sci(Au["A6_prop9_foc_rel_err"], 0),
         AudAsym=sci(Au["A7_prop10_asymptote_err"], 0), AudEq=sci(max(Au["A8_prop7_eq_err"], Au["A8_prop7_a0_tissue"]), 0))

# ---------------- present-day decomposition (GSS)
G = J("gss"); Dg = G["decomposition"]
sg = lambda x, d=3: ("%.*f" % (d, 0.0)) if round(x, d) == 0 else "\\ensuremath{%s%.*f}" % ("+" if x >= 0 else "-", d, abs(x))
ci = lambda v, d=3: "%s to %s" % (sg(v[1], d), sg(v[2], d))
def _sn(x, d):
    return ("%.*f" % (d, 0.0)) if round(x, d) == 0 else "$%s%.*f$" % ("+" if x >= 0 else "-", d, abs(x))
cell = lambda v: "%s [%s, %s]" % tuple(_sn(x, 3) for x in v)
pct = lambda x, d=1: f"{100 * x:.{d}f}"
M.update(GssN=th(G["n_base"]), GssBoot=th(G["nboot"]),
         GssCheck=sci(max(max(abs(e["check"]) for e in v.values()) for v in Dg.values()), 0),
         GssHomOld=f"{G['homogamy']['1972-1989']['r'][0]:.2f}", GssHomNew=f"{G['homogamy']['2008-2024']['r'][0]:.2f}")
for tk, tp in (("schooling", "Sch"), ("vocabulary", "Voc"), ("prestige", "Pre"), ("income", "Inc")):
    for sk, sp in (("men", "M"), ("women", "W")):
        e = Dg[tk][sk]
        for ck, cp in (("access", "Acc"), ("number", "Num"), ("total", "Tot")):
            M[f"Gss{tp}{sp}{cp}"] = sg(e[ck][0]); M[f"Gss{tp}{sp}{cp}CI"] = ci(e[ck])
    dd = G["sex_difference"][tk]["access"]
    M[f"Gss{tp}DiffAcc"] = sg(dd[0]); M[f"Gss{tp}DiffAccCI"] = ci(dd)
T3 = G["thirds"]
M.update(GssChlMIncLo=pct(T3["income"]["men"][0]["childless"][0]), GssChlMIncHi=pct(T3["income"]["men"][2]["childless"][0]),
         GssChlWIncLo=pct(T3["income"]["women"][0]["childless"][0]), GssChlWIncHi=pct(T3["income"]["women"][2]["childless"][0]))
TA, TM, Op, Ba, Pl = G["timing_account"], G["timing"], G["opportunity"], G["bateman"], G["partnerless"]
row = lambda sx, a, b: next(r for r in Ba[sx]["rows"] if r["lo"] == a and r["hi"] == b)
M.update(GssTimShareW=pct(TA["women"]["share"][0], 0), GssTimShareM=pct(TA["men"]["share"][0], 0),
         GssTimSlopeW=pct(-TA["women"]["slope_per_year"][0]), GssTimSlopeM=pct(-TA["men"]["slope_per_year"][0]),
         GssAfbRW=f"{TM['women']['r_school_agefirst'][0]:.2f}", GssAfbRM=f"{TM['men']['r_school_agefirst'][0]:.2f}",
         GssIM=f"{Op['men']['1900-1979']['I'][0]:.2f}", GssIW=f"{Op['women']['1900-1979']['I'][0]:.2f}",
         GssShareM=pct(Op["men"]["1900-1979"]["share_childless"][0], 0), GssShareW=pct(Op["women"]["1900-1979"]["share_childless"][0], 0),
         GssShareMOld=pct(Op["men"]["1920-1939"]["share_childless"][0], 0), GssShareMNew=pct(Op["men"]["1960-1979"]["share_childless"][0], 0),
         GssShareWOld=pct(Op["women"]["1920-1939"]["share_childless"][0], 0), GssShareWNew=pct(Op["women"]["1960-1979"]["share_childless"][0], 0),
         GssBatMOne=f"{row('men', 1, 1)['children'][0]:.2f}", GssBatWOne=f"{row('women', 1, 1)['children'][0]:.2f}",
         GssBatMTwenty=f"{row('men', 20, 988)['children'][0]:.2f}", GssBatWTwenty=f"{row('women', 20, 988)['children'][0]:.2f}",
         GssBatMGrad=sg(Ba["men"]["gradient_all"][0]), GssBatMGradCI=ci(Ba["men"]["gradient_all"]),
         GssBatWGrad=sg(Ba["women"]["gradient_all"][0]), GssBatWGradCI=ci(Ba["women"]["gradient_all"]),
         GssBatMMed=f"{Ba['men']['median']:.0f}", GssBatWMed=f"{Ba['women']['median']:.0f}",
         GssPlMOld=pct(Pl["men"]["1989-2002"]["all"][0]), GssPlMNew=pct(Pl["men"]["2016-2024"]["all"][0]),
         GssPlWOld=pct(Pl["women"]["1989-2002"]["all"][0]), GssPlWNew=pct(Pl["women"]["2016-2024"]["all"][0]),
         GssPlMLo=pct(Pl["men"]["2016-2024"]["low"][0]), GssPlMHi=pct(Pl["men"]["2016-2024"]["high"][0]),
         GssPlMNone=pct(Pl["men"]["2016-2024"]["no_earnings"][0]))
TN = {"schooling": "Years of schooling", "vocabulary": "Vocabulary (10 words)", "prestige": "Occupational prestige", "income": "Personal income (log)"}
with open(os.path.join(GEN, "tab_gss.tex"), "w") as f:
    f.write("Trait & Sex & $n$ & Access & Number & Total\\\\\n\\midrule\n")
    for tk in TN:
        for i, sk in enumerate(("men", "women")):
            e = Dg[tk][sk]
            f.write("%s & %s & %s & %s & %s & %s\\\\\n" % (TN[tk] if i == 0 else "", sk.capitalize(), th(e["n"]), cell(e["access"]), cell(e["number"]), cell(e["total"])))
        if tk != "income":
            f.write("\\addlinespace\n")
with open(os.path.join(GEN, "tab_gss_cohort.tex"), "w") as f:
    f.write("Trait & Sex & Born & $n$ & Access & Number & Total\\\\\n\\midrule\n")
    for tk in TN:
        for sk in ("men", "women"):
            for j, (ck, e) in enumerate(Dg[tk][sk]["by_cohort"].items()):
                f.write("%s & %s & %s & %s & %s & %s & %s\\\\\n" % (TN[tk] if (j == 0 and sk == "men") else "", sk.capitalize() if j == 0 else "",
                        ck.replace("-", "--"), th(e["n"]), cell(e["access"]), cell(e["number"]), cell(e["total"])))
        if tk != "income":
            f.write("\\addlinespace\n")
with open(os.path.join(GEN, "tab_gss_robust.tex"), "w") as f:
    f.write("Variant & Trait, sex & Access & Number & Total\\\\\n\\midrule\n")
    names = {"main": "Main analysis", "unweighted": "Unweighted", "ages 50-74": "Ages 50--74 only", "surveys 1990-2024": "Surveys 1990--2024",
             "surveys 1972-1989": "Surveys 1972--1989"}
    for vi, (vk, vv) in enumerate(G["robustness"].items()):
        for j, (kk, e) in enumerate(vv.items()):
            tr, sx = kk.split("|")
            f.write("%s & %s, %s & %s & %s & %s\\\\\n" % (names[vk] if j == 0 else "", "Schooling" if tr == "schooling" else "Income", sx,
                    cell(e["access"]), cell(e["number"]), cell(e["total"])))
        if vi < len(G["robustness"]) - 1:
            f.write("\\addlinespace\n")

TE, FF = G["three_episodes"], G["fifths"]
for tk, tp in (("schooling", "Sch"), ("vocabulary", "Voc"), ("prestige", "Pre"), ("income", "Inc")):
    for sk, sp in (("men", "M"), ("women", "W")):
        e = TE[tk][sk]
        M[f"Gss{tp}{sp}Pair"] = sg(e["pairing"][0]); M[f"Gss{tp}{sp}PairCI"] = ci(e["pairing"])
        M[f"Gss{tp}{sp}Conv"] = sg(e["conversion"][0]); M[f"Gss{tp}{sp}ConvCI"] = ci(e["conversion"])
fm, fw = FF["income"]["men"], FF["income"]["women"]
M.update(GssQfMOne=pct(fm["childless"][0][0]), GssQfMTwo=pct(fm["childless"][1][0]), GssQfMThree=pct(fm["childless"][2][0]),
         GssQfMFive=pct(fm["childless"][4][0]), GssQfWOne=pct(fw["childless"][0][0]), GssQfWFive=pct(fw["childless"][4][0]),
         GssNmMOne=pct(fm["never_married"][0][0]), GssNmMFive=pct(fm["never_married"][4][0]),
         GssFloorM=pct(fm["concentration"][0]), GssFloorMCI="%s to %s" % (pct(fm["concentration"][1]), pct(fm["concentration"][2])))
with open(os.path.join(GEN, "tab_gss_three.tex"), "w") as f:
    f.write("Trait & Sex & Pairing & Conversion & Number & Total\\\\\n\\midrule\n")
    for tk in TN:
        for i, sk in enumerate(("men", "women")):
            e = TE[tk][sk]
            f.write("%s & %s & %s & %s & %s & %s\\\\\n" % (TN[tk] if i == 0 else "", sk.capitalize(), cell(e["pairing"]), cell(e["conversion"]),
                                                     cell(e["number"]), cell(e["total"])))
        if tk != "income":
            f.write("\\addlinespace\n")

# ---------------- the generalist target (E15)
P = J("e15_polymath"); ac, ad = P["aim_common"], P["aim_division"]; br = P["breadth"]
errs = [abs(v["ratio"][0] / v["ratio_exact"] - 1) * 100 for v in ac.values()]
sherr = max(abs(v["share"][0] - v["share_exact"]) for d in P["expenditure"].values() for v in d.values())
M.update(PolyNSix=f"{br['n_star'][5]:.1f}", PolyNEight=f"{br['n_star'][7]:.1f}",
         PolyDivFive=f"{ad['5']['common_over_division']:.1f}", PolyDivTen=f"{ad['10']['common_over_division']:.1f}",
         PolyDivTwenty=f"{ad['20']['common_over_division']:.1f}", PolyAimErr=f"{max(errs):.1f}", PolyShareErr=f"{sherr:.3f}",
         PolyRatioOne=f"{ac['1']['ratio'][0]:.2f}", PolyRatioFour=f"{ac['4']['ratio'][0]:.2f}", PolyRatioSixteen=f"{ac['16']['ratio'][0]:.2f}",
         PolyValOne=f"{ac['1']['validity'][0]:.2f}", PolyValSixteen=f"{ac['16']['validity'][0]:.2f}",
         PolyDivSimTen=f"{ad['10']['ratio'][0]:.2f}", PolyDivExTen=f"{ad['10']['ratio_exact'][0] if isinstance(ad['10']['ratio_exact'], list) else ad['10']['ratio_exact']:.2f}",
         PolySteinMean=f"{P['stein']['10']['validity_mean'][0]:.2f}", PolySteinMax=f"{P['stein']['10']['validity_max'][0]:.2f}",
         NPred=35, NProp=19, NObj=51)

M.update(AudPoly=sci(max(Au["A9_generalist_aim_err"], Au["A9_share_err"], 1e-17), 0), AudCK=sci(Au["A9_cK_mc_err"], 0),
         AudGss=sci(max(Au["A10_gss_recompute_err"] or 0.0, 1e-17), 0), AudDelZ=f"{Au['A11_delivery_z']:.1f}")

# ---------------- tests, enforcement and the pairing gate (E16, A12, E14 section 13)
Vf = J("e16_verify"); Sg, En, Cl = Vf["staging"], Vf["enforcement"], Vf["clean"]; bmx = En["params"]["bmax"]
stdiff = max(max(abs(v["R_b"][0] - v["R_b_exact"]), abs(v["R_p"][0] - v["R_p_exact"])) for v in Sg.values())
ibmdiff = max(max(abs(v["b"][0] - v["b_exact"]) / bmx, abs(v["p"][0] - v["p_exact"])) for v in En["ibm"].values())
M.update(VerN=Vf["params"]["n"], VerRbLo=f"{Sg['0.1']['R_b_exact']:.2f}", VerRbHi=f"{Sg['0.9']['R_b_exact']:.2f}",
         VerRpLo=f"{Sg['0.1']['R_p_exact']:.2f}", VerRpHi=f"{Sg['0.9']['R_p_exact']:.2f}", VerRpAll=f"{Sg['0.0']['R_p_exact']:.2f}",
         VerDstar=f"{Vf['optimum']['40']['closed']:.2f}", VerMaxDiff=f"{stdiff:.4f}",
         VerCleanHalf=pct(Cl["0.5"]["exact"]), VerCleanNine=pct(Cl["0.9"]["exact"], 0),
         VerUpperZero=pct(Cl["0.0"]["upper95"], 0), VerUpperNine=pct(Cl["0.9"]["upper95"], 0),
         VerHalfZero=f"{Cl['0.0']['n_half']:.1f}", VerHalfNine=f"{Cl['0.9']['n_half']:.0f}",
         EnfT=f"{En['params']['T']:.1f}", EnfCapFifth=f"{En['b_at_fifth'] / bmx:.2f}", EnfCapHalf=f"{En['b_at_half'] / bmx:.2f}",
         EnfPmax=f"{En['p_max']:.2f}", EnfEstar=f"{En['e_star']:.2f}", EnfIbmDiff=f"{ibmdiff:.3f}",
         EnfRecErr=sci(max(En["recursion_max_err"], En["recursion_max_err_T15"], 1e-17), 0),
         EnfCrowdT=f"{En['crowding']['T']:.1f}", EnfCrowdMin=f"{En['crowding']['min_delivered']:.2f}",
         EnfCrowdAt=f"{En['crowding']['at']:.2f}")
GT = G["gate"]
gci = lambda v, d=0: "%s to %s" % (pct(v[1], d), pct(v[2], d))
for sk, sp in (("men", "M"), ("women", "W")):
    for ck, cp in (("1900-1919", "First"), ("1920-1939", "Old"), ("1960-1979", "New")):
        e = GT[sk][ck]
        M[f"GssGate{sp}{cp}"] = pct(e["parents_nm"][0], 0); M[f"GssGate{sp}{cp}CI"] = gci(e["parents_nm"])
        M[f"GssOut{sp}{cp}"] = pct(e["outside"][0]); M[f"GssOut{sp}{cp}CI"] = gci(e["outside"], 1)
        M[f"GssIn{sp}{cp}"] = pct(1 - e["outside"][0])
TC = G["three_by_cohort"]
for sk, sp in (("men", "M"), ("women", "W")):
    for ck, cp in (("1920-1939", "Old"), ("1960-1979", "New")):
        M[f"GssPair{sp}{cp}"] = sg(TC[sk][ck]["pairing"][0]); M[f"GssPair{sp}{cp}CI"] = ci(TC[sk][ck]["pairing"])
        M[f"GssNev{sp}{cp}"] = pct(GT[sk][ck]["never"][0])
    M[f"GssPair{sp}Change"] = sg(TC[sk]["pairing_change"][0]); M[f"GssPair{sp}ChangeCI"] = ci(TC[sk]["pairing_change"])
pc = lambda v, d=1: "%s [%s, %s]" % (pct(v[0], d), pct(v[1], d), pct(v[2], d))
with open(os.path.join(GEN, "tab_gss_gate.tex"), "w") as f:
    f.write("Sex & Born & Never married (\\%) & Parents among the never married (\\%) & Children of the never married "
            "(\\% of all children) & Pairing differential, income\\\\\n\\midrule\n")
    for i, sk in enumerate(("men", "women")):
        for j, ck in enumerate(("1920-1939", "1940-1959", "1960-1979")):
            e = GT[sk][ck]
            f.write("%s & %s & %s & %s & %s & %s\\\\\n" % (sk.capitalize() if j == 0 else "", ck.replace("-", "--"), pc(e["never"]),
                                                       pc(e["parents_nm"]), pc(e["outside"]), cell(TC[sk][ck]["pairing"])))
        if i == 0:
            f.write("\\addlinespace\n")
M.update(AudStage=sci(max(Au["A12_staging_validity_err"], 1e-17), 0), AudStageOpt=sci(Au["A12_staging_optimum_err"], 0),
         AudEnfZ=f"{Au['A12_enforcement_z']:.1f}", AudEnfEq=sci(max(Au["A12_enforcement_equilibrium_err"], 1e-17), 0),
         AudEnfCases=Au["A12_enforcement_interior_cases"])
M.update(AudSpeedGap=f"{Au['A13_agreement_gap']:.3f}", AudSpeedLpm=sci(max(Au["A13_lpm_err"], 1e-17), 0))
M.update(AudCeil=f"{Au['A13_ceiling_err']:.3f}")

# ---------------- speed dating (E17), further survey analyses (E14 sections 15-17)
S = J("e17_speed"); JU = S["jury"]; DD = S["did"]; AC = S["access"]; IN = S["interest"]; CK = S["checks"]
sgn1 = lambda x, d=1: "\\ensuremath{%s%.*f}" % ("+" if x >= 0 else "-", d, abs(x))
tag = dict(attr="Att", fun="Fun", intel="Int", amb="Amb", sinc="Sin", shar="Sha")
M.update(SdRows=th(S["rows"]), SdN=S["people"], SdW=S["women"], SdM=S["men"], SdSessions=S["sessions"], SdAge=f"{S['mean_age']:.1f}",
         JurGapMax=f"{max(v['max_gap'] for v in JU.values()):.2f}")
for a, t in tag.items():
    if a in JU:
        M[f"JurOne{t}"] = f"{JU[a]['r1']:.2f}"; M[f"JurK{t}"] = f"{round(JU[a]['k_for_08']):d}"
        M[f"SdSelf{t}"] = f"{CK['self'][a]['r']:.2f}"
    for sk, sp in (("women", "W"), ("men", "M")):
        M[f"SdSt{t}{sp}"] = f"{S['said'][sk][a]:.1f}"
        M[f"SdRev{t}{sp}"] = sgn1(DD["own"][sk]["coef"][a][0]) if DD["own"][sk]["coef"][a][0] < 0 else f"{DD['own'][sk]['coef'][a][0]:.1f}"
        M[f"SdPan{t}{sp}"] = sgn1(DD["panel"][sk]["coef"][a][0]) if DD["panel"][sk]["coef"][a][0] < 0 else f"{DD['panel'][sk]['coef'][a][0]:.1f}"
dm = DD["own"]["men_minus_women"]
M.update(SdRevDiffAtt=f"{dm['attr'][0]:.1f}", SdRevDiffAttCI="%.1f to %.1f" % (dm["attr"][1], dm["attr"][2]),
         SdRevDiffInt=sgn1(dm["intel"][0]), SdRevDiffIntCI="%s to %s" % (sgn1(dm["intel"][1]), sgn1(dm["intel"][2])),
         SdLegPear=f"{S['legibility_panel']['pearson']:.2f}", SdLegSpear=f"{S['legibility_panel']['spearman']:.1f}",
         SdYesW=f"{AC['women']['says_yes']:.0f}", SdYesM=f"{AC['men']['says_yes']:.0f}", SdMatch=f"{AC['men']['match']:.0f}",
         SdPopCVM=f"{AC['men']['cv']:.2f}", SdPopCVW=f"{AC['women']['cv']:.2f}", SdPopZeroM=f"{AC['men']['none']:.1f}", SdPopZeroW=f"{AC['women']['none']:.1f}",
         SdExpM=pct(IN["men"]["expected"][0], 0), SdActM=pct(IN["men"]["actual"][0], 0), SdExpW=pct(IN["women"]["expected"][0], 0), SdActW=pct(IN["women"]["actual"][0], 0),
         SdGapM=pct(IN["men"]["gap"][0], 0), SdGapMCI="%s to %s" % (pct(IN["men"]["gap"][1], 0), pct(IN["men"]["gap"][2], 0)),
         SdGapW=sgn1(100 * IN["women"]["gap"][0], 0), SdGapWCI="%s to %s" % (sgn1(100 * IN["women"]["gap"][1], 0), sgn1(100 * IN["women"]["gap"][2], 0)),
         SdGapDiff=pct(IN["gap_men_minus_women"][0], 0), SdGapDiffCI="%s to %s" % (pct(IN["gap_men_minus_women"][1], 0), pct(IN["gap_men_minus_women"][2], 0)),
         SdAccM=f"{IN['men']['accuracy'][0]:.2f}", SdAccW=f"{IN['women']['accuracy'][0]:.2f}",
         SdSat=f"{CK['sat']['intel']['r']:.2f}", SdSatCI="%.2f to %.2f" % tuple(CK["sat"]["intel"]["ci"]), SdSatN=CK["sat"]["intel"]["n"],
         SdSatAtt=f"{CK['sat']['attr']['r']:.2f}", SdAftAttW=f"{S['after']['women']['attr']:.0f}", SdAftAttM=f"{S['after']['men']['attr']:.0f}",
         SdAftN=S["after"]["women"]["n"] + S["after"]["men"]["n"],
         JurFunMen=f"{JU['fun']['by_sex']['men']['r1']:.2f}", JurFunWomen=f"{JU['fun']['by_sex']['women']['r1']:.2f}",
         JurAttMen=f"{JU['attr']['by_sex']['men']['r1']:.2f}", JurAttWomen=f"{JU['attr']['by_sex']['women']['r1']:.2f}")
_r1 = JU["intel"]["r1"]
M.update(JurCeil=f"{_r1 / (_r1 + 0.3 * (1 - _r1)):.2f}")
with open(os.path.join(GEN, "tab_speed.tex"), "w") as f:
    f.write("Attribute & \\multicolumn{2}{c}{Said} & \\multicolumn{2}{c}{Did, own rating} & \\multicolumn{2}{c}{Did, panel rating} & Agreement & Observers\\\\\n")
    f.write(" & W & M & W & M & W & M & of two & for 0.8\\\\\n\\midrule\n")
    num = lambda v: ("$-$%.1f" % abs(v)) if v < 0 else ("%.1f" % v)
    for a, lab in (("attr", "Looks"), ("shar", "Shared interests"), ("fun", "Fun"), ("intel", "Intelligence"), ("sinc", "Sincerity"), ("amb", "Ambition")):
        f.write("%s & %.1f & %.1f & %s & %s & %s & %s & %s & %s\\\\\n" % (
            lab, S["said"]["women"][a], S["said"]["men"][a], num(DD["own"]["women"]["coef"][a][0]), num(DD["own"]["men"]["coef"][a][0]),
            num(DD["panel"]["women"]["coef"][a][0]), num(DD["panel"]["men"]["coef"][a][0]),
            ("%.2f" % JU[a]["r1"]) if a in JU else "--", ("%d" % round(JU[a]["k_for_08"])) if a in JU else "--"))

HY, MH, PT = G["hypergamy"], G["marital_history"], G["partners_by_trait"]
pp = lambda v, d=0: "%s to %s" % (pct(v[1], d), pct(v[2], d))
for ck, cp in (("1920-1939", "Old"), ("1940-1959", "Mid"), ("1960-1979", "New"), ("1980-2004", "Young")):
    M[f"GssHypMore{cp}"] = pct(HY[ck]["wife_more"][0], 0); M[f"GssHypLess{cp}"] = pct(HY[ck]["wife_less"][0], 0)
M.update(GssHypDiffYoung=pct(HY["1980-2004"]["more_minus_less"][0], 0), GssHypDiffYoungCI=pp(HY["1980-2004"]["more_minus_less"]),
         GssHypR=f"{HY['1960-1979']['r'][0]:.2f}")
for sk, sp in (("men", "M"), ("women", "W")):
    M[f"GssRem{sp}"] = sg(MH[sk]["remarried_minus_intact"][0], 2); M[f"GssRem{sp}CI"] = ci(MH[sk]["remarried_minus_intact"], 2)
    M[f"GssDiv{sp}"] = sg(MH[sk]["divorced_minus_intact"][0], 2); M[f"GssDiv{sp}CI"] = ci(MH[sk]["divorced_minus_intact"], 2)
    M[f"GssDivChl{sp}"] = pct(MH[sk]["divorced"]["childless"][0]); M[f"GssIntChl{sp}"] = pct(MH[sk]["intact"]["childless"][0])
    for tk, tp in (("income", "Inc"), ("vocabulary", "Voc"), ("schooling", "Sch"), ("prestige", "Pre")):
        M[f"GssPart{tp}{sp}"] = sg(PT[sk][tk]["rho"][0], 2); M[f"GssPart{tp}{sp}CI"] = ci(PT[sk][tk]["rho"], 2)
    M[f"GssPartMed{sp}"] = f"{PT[sk]['schooling']['median']:.0f}"
    M[f"GssConv{sp}Old"] = sg(TC[sk]["1920-1939"]["conversion"][0]); M[f"GssConv{sp}New"] = sg(TC[sk]["1960-1979"]["conversion"][0])
    M[f"GssConv{sp}NewCI"] = ci(TC[sk]["1960-1979"]["conversion"])
M.update(GssRemDiff=sg(MH["sex_difference_remarried"][0], 2), GssRemDiffCI=ci(MH["sex_difference_remarried"], 2),
         GssDivDiff=sg(MH["sex_difference_divorced"][0], 2), GssDivDiffCI=ci(MH["sex_difference_divorced"], 2))
PLn = G["partnerless"]; last = "2016-2024"
for sk, sp in (("men", "M"), ("women", "W")):
    M[f"GssPlLow{sp}"] = pct(PLn[sk][last]["low"][0]); M[f"GssPlHigh{sp}"] = pct(PLn[sk][last]["high"][0])
c2 = lambda v: "%s [%s, %s]" % tuple(_sn(x, 2) for x in v)
with open(os.path.join(GEN, "tab_funnel.tex"), "w") as f:
    f.write("Step & Measure & Men & Women\\\\\n\\midrule\n")
    rowsF = [
        ("Sexual partners", "Rank correlation of income with lifetime number of partners", c2(PT["men"]["income"]["rho"]), c2(PT["women"]["income"]["rho"])),
        ("A partner at all", "No partner in the past year at ages 22--34, 2016--2024: lowest against highest third of income (\\%)",
         "%s against %s" % (pct(PLn["men"][last]["low"][0]), pct(PLn["men"][last]["high"][0])), "%s against %s" % (pct(PLn["women"][last]["low"][0]), pct(PLn["women"][last]["high"][0]))),
        ("Marrying", "Pairing differential on income (SD)", cell(TE["income"]["men"]["pairing"]), cell(TE["income"]["women"]["pairing"])),
        ("Parenthood if married", "Conversion differential on income (SD)", cell(TE["income"]["men"]["conversion"]), cell(TE["income"]["women"]["conversion"])),
        ("Family size", "Number differential on income among parents (SD)", cell(TE["income"]["men"]["number"]), cell(TE["income"]["women"]["number"])),
        ("A second marriage", "Family size of the remarried relative to people in a first marriage", c2(MH["men"]["remarried_minus_intact"]), c2(MH["women"]["remarried_minus_intact"])),
        ("A marriage ended", "Family size of the divorced and not remarried, relative to people in a first marriage", c2(MH["men"]["divorced_minus_intact"]), c2(MH["women"]["divorced_minus_intact"])),
    ]
    for i, r in enumerate(rowsF):
        f.write("%s & %s & %s & %s\\\\\n" % r)
        if i == 4:
            f.write("\\addlinespace\n")
with open(os.path.join(GEN, "tab_gss_hyp.tex"), "w") as f:
    f.write("Born & $n$ & Wife has more schooling (\\%) & Wife has less schooling (\\%) & More minus less (points) & Correlation\\\\\n\\midrule\n")
    p3 = lambda v: "%s [%s, %s]" % (pct(v[0]), pct(v[1]), pct(v[2]))
    for ck in ("1900-1919", "1920-1939", "1940-1959", "1960-1979", "1980-2004"):
        e = HY[ck]
        f.write("%s & %s & %s & %s & %s & %.2f\\\\\n" % (ck.replace("-", "--"), th(e["n"]), p3(e["wife_more"]), p3(e["wife_less"]),
                                                        "$%s%.1f$ [$%s%.1f$, $%s%.1f$]" % tuple(z for x in e["more_minus_less"] for z in ("+" if x >= 0 else "-", abs(100 * x))), e["r"][0]))
with open(os.path.join(GEN, "tab_gss_hist.tex"), "w") as f:
    f.write("Sex & Marital history & $n$ & Children & Childless (\\%) & Family size relative to sex and birth decade\\\\\n\\midrule\n")
    labs = dict(never="Never married", intact="One marriage, not divorced", remarried="Remarried after divorce", divorced="Divorced or separated, not remarried")
    for i, sk in enumerate(("men", "women")):
        for j, g in enumerate(("never", "intact", "remarried", "divorced")):
            e = MH[sk][g]
            f.write("%s & %s & %s & %.2f & %s & %.2f [%.2f, %.2f]\\\\\n" % (sk.capitalize() if j == 0 else "", labs[g], th(e["n"]), e["children"], pct(e["childless"][0]), *e["rel"]))
        if i == 0:
            f.write("\\addlinespace\n")

# ---------------- version 9: cultures (E18), village (E19), world (E20), new results (E21), audit A14-A17
def n1(x, d=1):
    """Number with a typeset minus sign; values that round to zero print unsigned."""
    t = f"{abs(x):.{d}f}"
    return ("$-$" + t) if (x < 0 and float(t) != 0) else t

def sp(x, d=1):
    t = f"{abs(x):.{d}f}"
    return t if float(t) == 0 else (("+" if x > 0 else "$-$") + t)

def ci_(v, d=1, scale=1.0):
    return f"{n1(scale * v[0], d)} to {n1(scale * v[1], d)}"

P0 = lambda x: f"{x:.0f}"
CU = J("e18_cultures"); AR = CU["arrangement"]; RP = CU["rape"]
M.update(CcSccsN=CU["n"]["sccs"], CcEaN=th(CU["n"]["ea"]), CcBinN=CU["n"]["binford"], CcForN=CU["n"]["sccs_foragers"],
         CcArrWomenN=AR["women"]["n"], CcArrMenN=AR["men"]["n"], CcAloneW=P0(AR["women"]["share"]["alone"]), CcAloneM=P0(AR["men"]["share"]["alone"]),
         CcAloneWCount=AR["women"]["groups"]["alone"], CcApprW=P0(AR["women"]["share"]["with_approval"]), CcApprM=P0(AR["men"]["share"]["with_approval"]),
         CcEitherW=P0(AR["women"]["share"]["either"]), CcEitherM=P0(AR["men"]["share"]["either"]),
         CcParW=P0(AR["women"]["share"]["parents"]), CcParM=P0(AR["men"]["share"]["parents"]),
         CcNoObjW=AR["women_cannot_object"], CcNoObjWPct=P0(100 * AR["women_cannot_object"] / AR["women"]["n"]), CcNoObjM=AR["men_cannot_object"],
         CcPairN=AR["paired"]["n"], CcPairDau=AR["paired"]["daughter_less_free"], CcPairSame=AR["paired"]["same"], CcPairSon=AR["paired"]["son_less_free"],
         CcForArrN=AR["women_foragers"]["n"], CcAloneWFor=P0(AR["women_foragers"]["share"]["alone"]), CcParWFor=P0(AR["women_foragers"]["share"]["parents"]),
         CcAloneMFor=P0(AR["men_foragers"]["share"]["alone"]), CcParMFor=P0(AR["men_foragers"]["share"]["parents"]),
         CcConsN=CU["consent"]["women"]["n"], CcConsNec=P0(CU["consent"]["women"]["share"]["1"]), CcConsAsk=P0(CU["consent"]["women"]["share"]["2"]),
         CcConsNot=P0(CU["consent"]["women"]["share"]["3"]), CcConsMN=CU["consent"]["men"]["n"], CcConsMNec=P0(CU["consent"]["men"]["share"]["1"]),
         CcConsMNot=P0(CU["consent"]["men"]["share"]["3"]),
         CcRapeN=RP["frequency"]["n"], CcRapeAbs=RP["frequency"]["counts"]["1"], CcRapeRare=RP["frequency"]["counts"]["2"],
         CcRapeCom=RP["frequency"]["counts"]["3"], CcRapeComPct=P0(RP["frequency"]["share"]["3"]),
         CcSanN=RP["sanday"]["n"], CcSanAbs=RP["sanday"]["counts"]["1"], CcSanPres=RP["sanday"]["counts"]["2"], CcSanPresPct=P0(RP["sanday"]["share"]["2"]),
         CcSanForN=RP["sanday_foragers"]["n"], CcSanForPres=RP["sanday_foragers"]["counts"]["2"],
         CcRapeParN=sum(RP["by_arrangement"]["parents"].values()), CcRapeParCom=RP["by_arrangement"]["parents"].get("3", 0),
         CcRapeOwnN=sum(RP["by_arrangement"]["alone"].values()) + sum(RP["by_arrangement"]["with_approval"].values()),
         CcRapeOwnCom=RP["by_arrangement"]["alone"].get("3", 0) + RP["by_arrangement"]["with_approval"].get("3", 0),
         CcCapN=CU["captives"]["n"], CcCap=CU["captives"]["counts"]["1"], CcCapPct=P0(CU["captives"]["share"]["1"]),
         CcBeatN=CU["wife_beating"]["n"], CcBeat=CU["wife_beating"]["counts"]["2"], CcBeatPct=P0(CU["wife_beating"]["share"]["2"]),
         CcDivN=CU["divorce_access"]["n"], CcDivEq=CU["divorce_access"]["counts"]["3"], CcDivEqPct=P0(CU["divorce_access"]["share"]["3"]),
         CcDivMaleOnly=CU["divorce_access"]["counts"]["1"], CcDivFreqN=CU["divorce_frequency"]["n"],
         CcDivCommon=CU["divorce_frequency"]["groups"]["common"], CcDivCommonPct=P0(CU["divorce_frequency"]["share"]["common"]),
         CcDivRare=CU["divorce_frequency"]["groups"]["rare"],
         CcAgeN=CU["binford_age"]["n"], CcAgeM=f"{CU['binford_age']['men']:.1f}", CcAgeW=f"{CU['binford_age']['women']:.1f}",
         CcGap=f"{CU['binford_age']['gap']:.1f}", CcGapCI=ci_(CU["binford_age"]["gap_ci"]), CcMenOlderPct=P0(CU["binford_age"]["men_older_pct"]),
         CcRelN=CU["relative_age"]["n"], CcRelMenOlder=CU["relative_age"]["groups"]["men_older"], CcRelWomenOlder=CU["relative_age"]["groups"]["women_older"],
         CcPolyN=CU["binford_polygyny"]["n"], CcPolyMed=P0(CU["binford_polygyny"]["median_pct"]), CcGapPolyR=f"{CU['binford_polygyny']['r_gap']:.2f}",
         CcGapPolyCI=ci_(CU["binford_polygyny"]["r_gap_ci"], 2), CcGapLow=f"{CU['binford_polygyny']['gap_low']:.1f}", CcGapHigh=f"{CU['binford_polygyny']['gap_high']:.1f}",
         CcGamesN=CU["games"]["all"]["n"], CcGamesAny=CU["games"]["all"]["n"] - CU["games"]["all"]["groups"]["none"],
         CcGamesAnyPct=P0(100 - CU["games"]["all"]["share"]["none"]), CcGamesForN=CU["games"]["foragers"]["n"],
         CcGamesForAny=CU["games"]["foragers"]["n"] - CU["games"]["foragers"]["groups"]["none"],
         CcTrN=th(CU["transfers"]["all"]["n"]), CcBridewealth=P0(CU["transfers"]["all"]["share"]["bridewealth"]),
         CcBrideService=P0(CU["transfers"]["all"]["share"]["bride_service"]), CcDowry=P0(CU["transfers"]["all"]["share"]["dowry"]),
         CcForService=P0(CU["transfers"]["foragers"]["share"]["bride_service"]), CcProdService=P0(CU["transfers"]["others"]["share"]["bride_service"]),
         CcMonoN=th(CU["marriage_system"]["n"]), CcMono=P0(CU["marriage_system"]["share"]["monogamous"]),
         CcPolyWomenMed=P0(CU["polygyny_women_pct"]["median"]), CcPolyWomenN=CU["polygyny_women_pct"]["n"],
         CcWarN=CU["warrior_prestige"]["n"], CcWarGreat=CU["warrior_prestige"]["counts"]["1"], CcWarGreatPct=P0(CU["warrior_prestige"]["share"]["1"]))

VG = J("e19_village"); AN = VG["ancestors"]; BR = VG["births"]; ID = VG["ideals"]; VT = VG["village"]["trend"]; VA = VG["village"]["assoc"]
PA, PB, PC, PD, PE = "1972-1984", "1985-1999", "2000-2011", "2012-2018", "2021-2024"
tp = lambda per, k, s="all": VT[per][k][s]["pct"]
M.update(VgNeM=f"{AN['all']['men']['ne_direct']:.2f}", VgNeW=f"{AN['all']['women']['ne_direct']:.2f}", VgNeRatio=f"{AN['all']['ratio']:.2f}",
         VgNeRatioCI=ci_(AN["all"]["ratio_ci"], 2), VgNeRatioNew=f"{AN['1960-1979']['ratio']:.2f}", VgNeRatioNewCI=ci_(AN["1960-1979"]["ratio_ci"], 2),
         VgNeRatioOld=f"{AN['1900-1939']['ratio']:.2f}", VgNeRatioOldCI=ci_(AN["1900-1939"]["ratio_ci"], 2),
         VgQM=P0(100 * AN["all"]["men"]["q"]), VgQW=P0(100 * AN["all"]["women"]["q"]), VgQMNew=P0(100 * AN["1960-1979"]["men"]["q"]),
         VgQWNew=P0(100 * AN["1960-1979"]["women"]["q"]), VgAlphaM=n1(AN["all"]["men"]["alpha"], 2), VgAlphaW=n1(AN["all"]["women"]["alpha"], 2),
         VgIm=f"{AN['all']['men']['I']:.2f}", VgIf=f"{AN['all']['women']['I']:.2f}", VgAncN=th(AN["all"]["men"]["n"] + AN["all"]["women"]["n"]),
         VgKidsOld=f"{BR['periods'][PA]['all']['children']:.2f}", VgKidsNew=f"{BR['periods'][PE]['all']['children']:.2f}",
         VgNevOld=P0(100 * BR["periods"][PA]["all"]["never_married"]), VgNevNew=P0(100 * BR["periods"][PE]["all"]["never_married"]),
         VgKidsNevOld=f"{BR['periods'][PA]['all']['children_never']:.2f}", VgKidsNevNew=f"{BR['periods'][PE]['all']['children_never']:.2f}",
         VgKidsEverOld=f"{BR['periods'][PA]['all']['children_ever']:.2f}", VgKidsEverNew=f"{BR['periods'][PE]['all']['children_ever']:.2f}",
         VgParNevOld=P0(100 * BR["periods"][PA]["all"]["parent_never"]), VgParNevNew=P0(100 * BR["periods"][PE]["all"]["parent_never"]),
         VgBirthN=th(sum(v["all"]["n"] for v in BR["periods"].values())),
         VgChange=n1(BR["long"]["all"]["change"], 2), VgMarr=n1(BR["long"]["all"]["marriage"], 2), VgWithinEver=n1(BR["long"]["all"]["within_ever"], 2),
         VgWithinNever=sp(BR["long"]["all"]["within_never"], 2), VgMarrShare=P0(100 * BR["long"]["all"]["marriage_share"]),
         VgMarrShareCI=ci_(BR["long"]["all"]["marriage_share_ci"], 0, 100), VgRecChange=n1(BR["recent"]["all"]["change"], 2),
         VgRecMarr=n1(BR["recent"]["all"]["marriage"], 2), VgMarrShareM=P0(100 * BR["long"]["men"]["marriage_share"]),
         VgMarrShareW=P0(100 * BR["long"]["women"]["marriage_share"]),
         VgIdealOld=f"{ID[PA]['ideal']:.2f}", VgIdealNew=f"{ID[PE]['ideal']:.2f}", VgIdealNever=f"{ID[PE]['ideal_never']:.2f}",
         VgIdealEver=f"{ID[PE]['ideal_ever']:.2f}", VgActOld=f"{ID[PA]['actual']:.2f}", VgActNew=f"{ID[PE]['actual']:.2f}",
         VgActNever=f"{ID[PE]['actual_never']:.2f}", VgActEver=f"{ID[PE]['actual_ever']:.2f}", VgIdealGap=f"{ID[PE]['gap']:.2f}",
         VgIdealN=th(sum(v["n_ideal"] for v in ID.values())),
         VgNeighOld=P0(tp(PA, "neighbours_monthly")), VgNeighNew=P0(tp(PE, "neighbours_monthly")), VgNeighMid=P0(tp(PD, "neighbours_monthly")),
         VgFriOld=P0(tp(PA, "friends_monthly")), VgFriNew=P0(tp(PE, "friends_monthly")), VgRelOld=P0(tp(PA, "relatives_monthly")),
         VgRelNew=P0(tp(PE, "relatives_monthly")), VgBarOld=P0(tp(PA, "bar_monthly")), VgBarNew=P0(tp(PE, "bar_monthly")),
         VgWorOld=P0(tp(PA, "worship_weekly")), VgWorNew=P0(tp(PE, "worship_weekly")),
         VgNevYOld=P0(tp(PA, "never_married")), VgNevYNew=P0(tp(PE, "never_married")),
         VgNoPartOld=f"{tp(PB, 'no_partner_year'):.0f}", VgNoPartNew=f"{tp(PE, 'no_partner_year'):.0f}",
         VgNoPartMOld=f"{tp(PB, 'no_partner_year', 'men'):.0f}", VgNoPartMNew=f"{tp(PE, 'no_partner_year', 'men'):.0f}",
         VgNoPartWOld=f"{tp(PB, 'no_partner_year', 'women'):.0f}", VgNoPartWNew=f"{tp(PE, 'no_partner_year', 'women'):.0f}",
         VgOnlineOld=f"{VT[PC]['hours_online']['all']['mean']:.0f}", VgOnlineNew=f"{VT[PE]['hours_online']['all']['mean']:.0f}",
         VgAsN=th(VA["relatives_monthly"]["n"]), VgAsRel=sp(VA["relatives_monthly"]["points"]), VgAsRelCI=ci_(VA["relatives_monthly"]["ci"]),
         VgAsFri=sp(VA["friends_monthly"]["points"]), VgAsFriCI=ci_(VA["friends_monthly"]["ci"]),
         VgAsNeigh=sp(VA["neighbours_monthly"]["points"]), VgAsNeighCI=ci_(VA["neighbours_monthly"]["ci"]),
         VgAsWor=sp(VA["worship_weekly"]["points"]), VgAsWorCI=ci_(VA["worship_weekly"]["ci"]))

WD = J("e20_world"); WI = WD["ideals"]; WB = WD["budget"]; WO = WD["door"]; WS = WD["screens"]; WE = WD["equality"]; WT = WD["walter"]
M.update(WdN=th(WD["rows"]), WdCtry=WI["resources"]["countries"], WdPeople=th(WI["resources"]["people"]),
         WdResPos=WI["resources"]["positive"], WdResD=f"{WI['resources']['median']:.2f}", WdResLo=f"{WI['resources']['min']:.2f}",
         WdResHi=f"{WI['resources']['max']:.2f}", WdLooksPos=WI["looks"]["positive"], WdLooksD=f"{WI['looks']['median']:.2f}",
         WdLooksMin=n1(WI["looks"]["min"], 2), WdLooksMax=f"{WI['looks']['max']:.2f}", WdKindPos=WI["kindness"]["positive"],
         WdKindD=f"{WI['kindness']['median']:.2f}", WdHealthPos=WI["health"]["positive"],
         WdBudN=th(WB["n_men"] + WB["n_women"]), WdKindTopM=WB["kindness_top"]["men"], WdKindTopW=WB["kindness_top"]["women"],
         WdKindTopN=WB["kindness_top"]["countries"],
         **{f"WdBud{a}{s[0].upper()}": f"{WB['points'][k][s]:.1f}" for a, k in (("Kind", "kindness"), ("Health", "health"), ("Looks", "looks"),
                                                                              ("Res", "resources"), ("Age", "age"), ("Rel", "religion")) for s in ("men", "women")},
         WdDoorN=th(WO["n"]), WdDoorResM=sp(WO["coef"]["men"]["resources"]), WdDoorResMCI=ci_(WO["ci"]["men"]["resources"]),
         WdDoorResW=sp(WO["coef"]["women"]["resources"]), WdDoorResWCI=ci_(WO["ci"]["women"]["resources"]),
         WdDoorResDiff=f"{WO['diff']['resources']['value']:.1f}", WdDoorResDiffCI=ci_(WO["diff"]["resources"]["ci"]),
         WdDoorLooksM=sp(WO["coef"]["men"]["looks"]), WdDoorLooksMCI=ci_(WO["ci"]["men"]["looks"]), WdDoorLooksW=sp(WO["coef"]["women"]["looks"]),
         WdDoorLooksWCI=ci_(WO["ci"]["women"]["looks"]), WdDoorKindM=sp(WO["coef"]["men"]["kindness"]), WdDoorKindMCI=ci_(WO["ci"]["men"]["kindness"]),
         WdDoorKindW=sp(WO["coef"]["women"]["kindness"]), WdDoorKindWCI=ci_(WO["ci"]["women"]["kindness"]),
         WdDoorHealthW=sp(WO["coef"]["women"]["health"]), WdDoorHealthM=sp(WO["coef"]["men"]["health"]),
         WdDoorCtry=WO["by_country"]["countries"], WdDoorCtryPos=WO["by_country"]["positive"], WdDoorCtryMed=f"{WO['by_country']['median']:.1f}",
         WdDoorCtryCI=ci_(WO["by_country"]["median_ci"]),
         WdAgeCtry=WD["ages"]["countries"], WdAgePos=WD["ages"]["positive"], WdAgeGap=f"{WD['ages']['median']:.1f}", WdAgePeople=th(WD["ages"]["people"]),
         WdScrN=th(WS["n"]), WdScrHours=f"{WS['mean_hours']:.1f}", WdScrM=sp(WS["men"]["points_per_hour"]), WdScrMCI=ci_(WS["men"]["ci"]),
         WdScrW=sp(WS["women"]["points_per_hour"]), WdScrWCI=ci_(WS["women"]["ci"]), WdScrY=sp(WS["young"]["points_per_hour"]), WdScrYCI=ci_(WS["young"]["ci"]),
         WdEqRes=n1(WE["resources"]["rho"], 2), WdEqResCI=ci_(WE["resources"]["ci"], 2), WdEqLooks=sp(WE["looks"]["rho"], 2), WdEqLooksCI=ci_(WE["looks"]["ci"], 2),
         WtN=th(WT["n"]), WtCtry=WT["countries"], WtIntAboveW=WT["intelligence_above_looks_and_money"]["women"],
         WtIntAboveM=WT["intelligence_above_looks_and_money"]["men"],
         **{f"Wt{a}{s[0].upper()}": f"{WT['means'][k][s]:.2f}" for a, k in (("Kind", "kindness"), ("Int", "intelligence"), ("Health", "health"),
                                                                           ("Looks", "physatt"), ("Res", "resources")) for s in ("men", "women")},
         WtResPos=WT["d"]["resources"]["positive"], WtResD=f"{WT['d']['resources']['median']:.2f}", WtLooksPos=WT["d"]["physatt"]["positive"],
         WtLooksD=f"{WT['d']['physatt']['median']:.2f}", WtIntPos=WT["d"]["intelligence"]["positive"], WtIntD=f"{WT['d']['intelligence']['median']:.2f}",
         WtAgeCtry=WT["age_gap"]["countries"], WtAgePos=WT["age_gap"]["positive"], WtAgeGap=f"{WT['age_gap']['median']:.1f}",
         WtAgeEq=n1(WT["age_gap_equality"]["rho"], 2), WtAgeEqCI=ci_(WT["age_gap_equality"]["ci"], 2))

NT = J("e21_notes")
from scipy.stats import norm as _norm
_iq = lambda q: float(_norm.pdf(_norm.ppf(q)) / (1 - q))
_w = {(r["k"], r["d"]): r["periods"] for r in NT["waiting"]["rows"]}
_rt = {(r["r_safe"], r["r_danger"]): r["n_needed"] for r in NT["ritual"]["rows"]}
M.update(NtAncErr=f"{100 * NT['ancestors']['max_rel_err']:.0f}", NtWaitErr=f"{NT['waiting']['max_abs_err']:.3f}",
         NtRitErr=f"{NT['ritual']['max_abs_err']:.3f}", NtNarErr=f"{NT['narrow']['max_abs_err']:.3f}", NtCoeErr=f"{NT['coercion']['max_abs_err']:.3f}",
         NtWaitCamp=f"{_w[(30, 0.2)]:.1f}", NtWaitFive=f"{_w[(5, 0.2)]:.1f}", NtWaitOne=f"{_w[(1, 0.2)]:.0f}", NtWaitOneStaged=f"{_w[(1, 0.8)]:.0f}",
         NtRitA=f"{_rt[(0.1, 0.5)]:.0f}", NtRitB=f"{np.ceil(_rt[(0.05, 0.6)]):.0f}",
         NtIqUS=f"{_iq(AN['all']['men']['q']):.2f}", NtIqForty=f"{_iq(0.4):.2f}", NtIqSixty=f"{_iq(0.6):.2f}",
         NtNarHurt=n1(NT["narrow"]["half_that_hurts"]["slope_on_x"], 2), NtNarBroad=n1(NT["narrow"]["broad"]["slope_on_x"], 2))
M.update(AudAnc=sci(max(Au["A14_ancestors_identity"], 1e-17), 0), AudBirths=sci(max(Au["A15_births_sum"], 1e-17), 0),
         AudCultures=Au["A16_cultures_recount"], AudWorldSign=Au["A17_world_sign"])

# table bodies
with open(os.path.join(GEN, "tab_cultures.tex"), "w") as f:
    f.write("What is coded & Source of the codes & Societies & Finding\\\\\n\\midrule\n")
    rows = [
        ("Who chooses a daughter's spouse", "broude1983", AR["women"]["n"], "She alone %s\\%%; she chooses, elders approve %s\\%%; either custom %s\\%%; parents %s\\%% (she cannot easily object in %d societies)" % (M["CcAloneW"], M["CcApprW"], M["CcEitherW"], M["CcParW"], AR["women_cannot_object"])),
        ("Who chooses a son's spouse", "broude1983", AR["men"]["n"], "He alone %s\\%%; with approval %s\\%%; either %s\\%%; parents %s\\%%" % (M["CcAloneM"], M["CcApprM"], M["CcEitherM"], M["CcParM"])),
        ("Daughters against sons, same society", "broude1983", AR["paired"]["n"], "Daughter less free in %d, equally free in %d, son less free in %d" % (AR["paired"]["daughter_less_free"], AR["paired"]["same"], AR["paired"]["son_less_free"])),
        ("Is the bride's consent needed", "frayser1985", CU["consent"]["women"]["n"], "Necessary %s\\%%; she is consulted %s\\%%; not consulted %s\\%%" % (M["CcConsNec"], M["CcConsAsk"], M["CcConsNot"])),
        ("Is the groom's consent needed", "frayser1985", CU["consent"]["men"]["n"], "Necessary %s\\%%; consulted %s\\%%; not consulted %s\\%%" % (M["CcConsMNec"], P0(CU["consent"]["men"]["share"]["2"]), M["CcConsMNot"])),
        ("Frequency of rape", "broude1976", RP["frequency"]["n"], "Absent in %d, rare in %d, common in %d" % (M["CcRapeAbs"], M["CcRapeRare"], M["CcRapeCom"])),
        ("Rape reported or institutionalized", "sanday1981book", RP["sanday"]["n"], "Absent in %d, present in %d (%s\\%%)" % (M["CcSanAbs"], M["CcSanPres"], M["CcSanPresPct"])),
        ("Women taken captive in war", "white1988", CU["captives"]["n"], "In %d (%s\\%%)" % (M["CcCap"], M["CcCapPct"])),
        ("Wife-beating", "broude1983", CU["wife_beating"]["n"], "Present in %d (%s\\%%)" % (M["CcBeat"], M["CcBeatPct"])),
        ("Who can end a marriage", "whyte1978", CU["divorce_access"]["n"], "Equally open to both in %d (%s\\%%); husband only in %d" % (M["CcDivEq"], M["CcDivEqPct"], M["CcDivMaleOnly"])),
        ("How often marriages end", "broude1983", CU["divorce_frequency"]["n"], "Common in %d (%s\\%%); rare in %d" % (M["CcDivCommon"], M["CcDivCommonPct"], M["CcDivRare"])),
        ("Age at first marriage, forager groups", "binford2001", CU["binford_age"]["n"], "Men %s, women %s years; men older in %s\\%% of groups" % (M["CcAgeM"], M["CcAgeW"], M["CcMenOlderPct"])),
    ]
    for a, src, n, txt in rows:
        f.write("%s & \\cite{%s} & %d & %s\\\\\n" % (a, src, n, txt))

with open(os.path.join(GEN, "tab_births.tex"), "w") as f:
    f.write("Survey years & $n$ & Never married (\\%) & Children per adult & of the never married & of the ever married & Ideal number (ages 18--44)\\\\\n\\midrule\n")
    for per in (PA, PB, PC, PD, PE):
        e = BR["periods"][per]["all"]
        f.write("%s & %s & %.0f & %.2f & %.2f & %.2f & %s\\\\\n" % (per.replace("-", "--"), th(e["n"]), 100 * e["never_married"], e["children"],
                                                                   e["children_never"], e["children_ever"], ("%.2f" % ID[per]["ideal"]) if per in ID else "--"))

with open(os.path.join(GEN, "tab_village.tex"), "w") as f:
    f.write("Among adults aged 22--34 (\\%) & 1972--84 & 1985--99 & 2000--11 & 2012--18 & 2021--24\\\\\n\\midrule\n")
    labs = (("neighbours_monthly", "Social evening with a neighbour at least monthly"), ("friends_monthly", "With friends outside the neighbourhood, at least monthly"),
            ("relatives_monthly", "With relatives, at least monthly"), ("bar_monthly", "At a bar or tavern, at least monthly"),
            ("worship_weekly", "Religious service nearly every week or more"), ("never_married", "Never married"),
            ("no_steady_partner", "No steady partner"), ("no_partner_year", "No sexual partner in the past year"))
    for k, lab in labs:
        f.write(lab + " & " + " & ".join(("%.0f" % VT[per][k]["all"]["pct"]) if k in VT[per] else "--" for per in (PA, PB, PC, PD, PE)) + "\\\\\n")
    f.write("Hours online per week (mean) & " + " & ".join(("%.0f" % VT[per]["hours_online"]["all"]["mean"]) if "hours_online" in VT[per] else "--" for per in (PA, PB, PC, PD, PE)) + "\\\\\n")

with open(os.path.join(GEN, "tab_world.tex"), "w") as f:
    f.write("Quantity & Data & Countries & Result\\\\\n\\midrule\n")
    rows = [
        ("Women want better financial prospects in a partner than men do", "K", WI["resources"]["countries"], "In %d; median $d=%s$ (range %s to %s)" % (M["WdResPos"], M["WdResD"], M["WdResLo"], M["WdResHi"])),
        ("Men want better looks than women do", "K", WI["looks"]["countries"], "In %d; median $d=%s$ (range %s to %s)" % (M["WdLooksPos"], M["WdLooksD"], M["WdLooksMin"], M["WdLooksMax"])),
        ("Women want more kindness than men do", "K", WI["kindness"]["countries"], "In %d; median $d=%s$" % (M["WdKindPos"], M["WdKindD"])),
        ("Kindness is given the most of thirty points", "K", WB["kindness_top"]["countries"], "By men in %d, by women in %d" % (M["WdKindTopM"], M["WdKindTopW"])),
        ("Intelligence rated above both looks and money", "W", WT["countries"], "By women in %d, by men in %d" % (M["WtIntAboveW"], M["WtIntAboveM"])),
        ("Women want more intelligence than men do", "W", WT["d"]["intelligence"]["countries"], "In %d; median $d=%s$" % (M["WtIntPos"], M["WtIntD"])),
        ("Own financial prospects and having a partner: men's effect exceeds women's", "K", WO["by_country"]["countries"], "In %d; median difference %s points per SD" % (M["WdDoorCtryPos"], M["WdDoorCtryMed"])),
        ("The man is the older partner on average", "K", WD["ages"]["countries"], "In %d; median gap %s years" % (M["WdAgePos"], M["WdAgeGap"])),
        ("The man is the older partner on average", "W", WT["age_gap"]["countries"], "In %d; median gap %s years" % (M["WtAgePos"], M["WtAgeGap"])),
    ]
    for a, src, n, txt in rows:
        f.write("%s & %s & %d & %s\\\\\n" % (a, src, n, txt))

# ---------------- references
V = J("ref_verification")
import re, glob
cited = set()
for f in glob.glob(os.path.join(ROOT, "paper", "sections", "*.tex")) + [os.path.join(ROOT, "paper", "main.tex")]:
    for grp in re.findall(r"\\cite\{([^}]*)\}", open(f).read()):
        cited.update(k.strip() for k in grp.split(","))
M.update(RefResolved=sum(1 for k, x in V.items() if x.get("doi") and k in cited), RefCited=len(cited))

with open(os.path.join(GEN, "numbers.tex"), "w") as f:
    f.write("% Generated by code/make_tex.py from results/*.json. Do not edit by hand.\n")
    for k in sorted(M):
        f.write("\\newcommand{\\%s}{%s}\n" % (k, M[k]))
print(len(M), "macros written")
