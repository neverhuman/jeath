"""Render only human-study evidence into reproducible vector figures and tables."""
from io import BytesIO, StringIO
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COLORS = {"women": "#A44262", "men": "#236F88", "Female": "#A44262", "Male": "#236F88"}
MARKERS = {"women": "o", "men": "^", "Female": "o", "Male": "^"}
PAIRS = (("women", "Female"), ("men", "Male"))
ITEMS = ["kindness", "health", "appearance", "resources", "appropriate_age", "religion"]
LABELS = {"kindness": "Kindness", "health": "Health", "appearance": "Appearance",
          "resources": "Financial prospects", "appropriate_age": "Appropriate age", "religion": "Religion",
          "age_import": "Age", "attractive_import": "Attractiveness", "physicalbuild_import": "Physical build",
          "education_import": "Education", "intelligence_import": "Intelligence", "income_import": "Income",
          "openness_import": "Openness", "trust_import": "Trust", "emotionalconn_import": "Emotional connection"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 10,
                     "axes.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "svg.hashsalt": "minds-we-choose",
                     "savefig.facecolor": "white", "axes.titlelocation": "left"})


def tex(value):
    value = str(value).replace("−", "-").replace("–", "--").replace("’", "'")
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return "".join({"&": r"\&", "%": r"\%", "_": r"\_\allowbreak{}", "#": r"\#", "$": r"\$",
                    "^": r"\textasciicircum{}", "~": r"\textasciitilde{}", "{": r"\{", "}": r"\}",
                    "\\": r"\textbackslash{}", "/": r"/\allowbreak{}"}.get(c, c) for c in value)


def fnum(value, digits=2):
    return "--" if value is None else f"{value:.{digits}f}"


def ci(cell, scale=1, digits=1):
    return f"{cell['mean']*scale:.{digits}f} [{cell['interval_95'][0]*scale:.{digits}f}, {cell['interval_95'][1]*scale:.{digits}f}]"


def table(headers, rows, widths=None):
    columns = widths or ("l" + "r" * (len(headers) - 1))
    lines = [r"{\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{longtable}{@{}" + columns + r"@{}}", r"\toprule",
             " & ".join(headers) + r"\\", r"\midrule\endhead"]
    lines.extend(" & ".join(map(str, row)) + r"\\" for row in rows)
    return "\n".join(lines + [r"\bottomrule\end{longtable}}", ""])


def csv_bytes(headers, rows):
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(headers)
    writer.writerows(rows)
    return stream.getvalue()


def title(ax, letter, text, ylabel=None, xlabel=None):
    ax.set_title(f"{letter}  {text}", fontweight="bold", pad=10)
    if ylabel: ax.set_ylabel(ylabel)
    if xlabel: ax.set_xlabel(xlabel)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="#E4E7E9", linewidth=.5, zorder=0)
    ax.set_axisbelow(True)


def errorbar(ax, x, cells, sex, scale=1, label=None, **kwargs):
    means = np.array([c["mean"] for c in cells]) * scale
    limits = np.array([c["interval_95"] for c in cells]).T * scale
    ax.errorbar(x, means, yerr=np.maximum(0, np.stack((means-limits[0], limits[1]-means))),
                marker=MARKERS[sex], color=COLORS[sex], linestyle="-" if sex in ("women", "Female") else "--",
                markersize=4, linewidth=1.4, capsize=2, label=label, **kwargs)


def country_means(data, gender, band, key, countries=None):
    countries = countries or data["hon"]["common_support_countries"]
    values = [data["hon"]["thresholds"]["20"][c][gender]["bands"][band][key] for c in countries]
    return np.mean(values, axis=0)


def overview(data):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.25), layout="constrained")
    k = data["surveys"]["kowal"]
    for offset, sex in zip((-.13, .13), ("women", "men")):
        x = np.arange(3) + offset
        axes[0, 0].bar(x, [k["equal_country"][sex][i]["mean"] for i in ("kindness", "appearance", "resources")],
                       width=.25, color=COLORS[sex], label=sex.capitalize(), hatch="" if sex == "women" else "//")
    axes[0, 0].set_xticks(range(3), ["Kindness", "Looks", "Resources"], fontsize=7)
    title(axes[0, 0], "A", "Shared priorities", "Budget points (of 30)")
    axes[0, 0].set_ylim(0, 8)
    axes[0, 0].legend(fontsize=6.5, frameon=False)
    a = data["hon"]["age_analysis"]
    for sex, gender in PAIRS:
        curve = a["curves"][gender]["mean_percentile"]
        axes[0, 1].plot(a["ages"], np.array(curve["mean"])*100, color=COLORS[sex],
                        linestyle="-" if sex == "women" else "--", label=sex)
        axes[0, 1].fill_between(a["ages"], * (np.array(curve["interval_95"])*100), color=COLORS[sex], alpha=.12)
    title(axes[0, 1], "B", "Appearance across age", "Mean percentile", "Profile age")
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].set_xticks([20, 40, 60])
    s = data["speed_dating"]
    for sex, _ in PAIRS:
        cells = s["yes_by_panel_quintile"][sex]["quintiles"]
        axes[0, 2].plot(range(1, 6), [100*x["estimate"] for x in cells], color=COLORS[sex], marker=MARKERS[sex],
                        linestyle="-" if sex == "women" else "--")
    title(axes[0, 2], "C", "Decisions after a date", "Yes (%)", "Within-chooser rank band")
    axes[0, 2].set_xticks(range(1, 6)); axes[0, 2].set_ylim(0, 100)
    w = data["surveys"]["whyte"]["analyses"]["all"]
    bands = ["18-29", "30-39", "40-49", "50-64" if "50-64" in w["women"] else "50-65"]
    for item, style, marker in (("trust_import", "-", "o"), ("intelligence_import", "--", "s"), ("attractive_import", ":", "^")):
        vals = [(w["women"][b]["items"][item]["mean"] + w["men"][b]["items"][item]["mean"])/2 for b in bands]
        axes[1, 0].plot(range(len(bands)), vals, linestyle=style, marker=marker, color="#344F63", label=LABELS[item])
    title(axes[1, 0], "D", "Relational expectations", "Importance (0–100)", "Age band")
    axes[1, 0].set_xticks(range(len(bands)), [b.replace("-", "–") for b in bands], fontsize=6.5)
    axes[1, 0].set_ylim(0, 100);axes[1, 0].legend(fontsize=6.0, frameon=False, loc="lower right")
    axes[1, 0].text(.03,.12,"Equal gender means",fontsize=6,transform=axes[1, 0].transAxes)
    p = data["pov"]["country_balanced"]["all"]
    for sex, gender in PAIRS:
        for key, style in (("serious_intent_share", "-"), ("want_children_share", "--")):
            axes[1, 1].plot([0, 1], [100*p["by_band"][b][gender]["outcomes"][key]["mean"] for b in ("25-29", "45-49")],
                            color=COLORS[sex], marker=MARKERS[sex], linestyle=style, linewidth=1.3)
    title(axes[1, 1], "E", "Commitment and children", "Reporting outcome (%)", "Profile age band")
    axes[1, 1].set_xticks([0, 1], ["25–29", "45–49"]);axes[1, 1].set_ylim(0, 100)
    axes[1, 1].text(.03, .08, "Solid: commitment\nDashed: want children", transform=axes[1, 1].transAxes, fontsize=6.5)
    contrasts = a["country_contrasts"]
    for sex, gender in PAIRS:
        values = [100*c[gender]["contrasts"]["mean_percentile"]["young_minus_older"] for c in contrasts]
        axes[1, 2].hist(values, bins=np.arange(-40, 71, 10), histtype="step", linewidth=1.6,
                        color=COLORS[sex], linestyle="-" if sex == "women" else "--")
    title(axes[1, 2], "F", "Country variation", "Countries", "25–29 minus 45–49 (pp)")
    axes[1, 2].axvline(0, color="#777777", linewidth=.7)
    for ax in axes.flat:
        ax.title.set_fontsize(8.1); ax.xaxis.label.set_fontsize(7); ax.yaxis.label.set_fontsize(7)
        ax.tick_params(labelsize=6.5)
    return fig


def preferences(data):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), layout="constrained")
    k = data["surveys"]["kowal"]
    for j, sex in enumerate(("women", "men")):
        x = np.arange(6)+(j-.5)*.14
        errorbar(axes[0, 0], x, [k["equal_country"][sex][i] for i in ITEMS], sex, label=sex.capitalize())
        axes[0, 0].scatter(x, [k["pooled"][sex][i]["mean"] for i in ITEMS], marker="x", color=COLORS[sex], s=22)
    title(axes[0, 0], "A", "All budget traits", "Points (of 30)")
    axes[0, 0].set_xticks(range(6), ["Kindness", "Health", "Looks", "Resources", "Age", "Religion"], rotation=35, ha="right")
    axes[0, 0].legend(fontsize=7, frameon=False);axes[0, 0].set_ylim(0, 8)
    w = data["surveys"]["whyte"]["analyses"]["all"]
    full = "18-64" if "18-64" in w["women"] else "18-65"
    items = data["surveys"]["whyte"]["items"]
    for j, sex in enumerate(("women", "men")):
        means = [w[sex][full]["items"][i]["mean"] for i in items]
        limits = np.array([w[sex][full]["items"][i]["interval_95"] for i in items]).T
        axes[0, 1].errorbar(means, np.arange(len(items))+(j-.5)*.20, xerr=[means-limits[0], limits[1]-means],
                            fmt=MARKERS[sex], markersize=3, color=COLORS[sex], capsize=2)
    title(axes[0, 1], "B", "All importance items", xlabel="Importance (0–100)")
    axes[0, 1].set_yticks(range(len(items)), [LABELS[i] for i in items], fontsize=7)
    axes[0, 1].invert_yaxis();axes[0, 1].set_xlim(0, 100)
    bands = ["18-29", "30-39", "40-49", "50-64" if "50-64" in w["women"] else "50-65"]
    for ax, item, letter in ((axes[1, 0], "trust_import", "C"), (axes[1, 1], "openness_import", "D")):
        for sex, _ in PAIRS:
            errorbar(ax, range(len(bands)), [w[sex][b]["items"][item] for b in bands], sex, label=sex.capitalize())
        title(ax, letter, LABELS[item]+" across age", "Importance (0–100)")
        ax.set_xticks(range(len(bands)), bands);ax.set_ylim(0, 100)
    fig.supxlabel("Kowal: country bootstrap intervals; crosses are respondent-weighted means. Whyte: respondent intervals.", fontsize=7)
    return fig


def countries(data):
    rows = sorted(data["hon"]["age_analysis"]["country_contrasts"], key=lambda r: r["country"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 7.3), layout="constrained")
    parts = (rows[:(len(rows)+1)//2], rows[(len(rows)+1)//2:])
    for ax, records, letter in zip(axes, parts, ("A", "B")):
        y = np.arange(len(records))
        for sex, gender in PAIRS:
            vals = [r[gender]["contrasts"]["mean_percentile"]["young_minus_older"]*100 for r in records]
            ax.scatter(vals, y+(-.12 if sex == "women" else .12), color=COLORS[sex], marker=MARKERS[sex], s=17, label=sex.capitalize())
        labels = [r["country"].replace("United States of America", "United States").replace("Bosnia and Herzegovina", "Bosnia & Herz.").replace("Republic of Serbia", "Serbia") for r in records]
        ax.set_yticks(y, labels, fontsize=6.7);ax.invert_yaxis();ax.axvline(0, color="#777777", linewidth=.7)
        title(ax, letter, "HON country age contrasts", xlabel="25–29 minus 45–49 percentile (pp)")
        ax.set_xlim(-35, 70);ax.tick_params(axis="y",labelsize=6.7)
    axes[0].legend(loc="lower right", fontsize=7, frameon=False)
    return fig


def beauty(data):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), layout="constrained")
    a = data["hon"]["age_analysis"]
    quantile_countries = [c for c in data["hon"]["common_support_countries"]
                          if all(b in data["hon"]["thresholds"]["20"][c][g]["bands"]
                                 for g in ("Female", "Male") for b in ("25-29", "45-49"))]
    for sex, gender in PAIRS:
        for ax, key, scale, letter, heading, ylabel in (
            (axes[0, 0], "mean_score", 1, "A", "Supplied scores", "Mean score (0–10)"),
            (axes[0, 1], "mean_percentile", 100, "B", "Fixed-reference percentile", "Mean percentile")):
            curve = a["curves"][gender][key]
            ax.plot(a["ages"], np.array(curve["mean"])*scale, color=COLORS[sex], linestyle="-" if sex=="women" else "--", label=sex.capitalize())
            ax.fill_between(a["ages"], *(np.array(curve["interval_95"])*scale), color=COLORS[sex], alpha=.15)
            observed = [country_means(data, gender, b, key) * scale for b in ("18-29", "30-44", "45-64")]
            ax.scatter([23.5, 37, 54.5], observed, color=COLORS[sex], marker=MARKERS[sex], s=23)
            title(ax, letter, heading, ylabel, "Profile age")
        quant = [country_means(data, gender, b, "score_quantiles", quantile_countries) for b in ("25-29", "45-49")]
        for j, q in enumerate(quant):
            y = j +(-.12 if sex=="women" else .12)
            axes[1, 0].plot([q[0],q[4]], [y,y], color=COLORS[sex], alpha=.45, linewidth=2)
            axes[1, 0].plot([q[1],q[3]], [y,y], color=COLORS[sex], linewidth=5)
            axes[1, 0].scatter(q[2], y, marker=MARKERS[sex], color=COLORS[sex], s=25)
        values=[country_means(data, gender, b, "upper_decile_share")*100 for b in ("18-29","30-44","45-64")]
        axes[1, 1].plot(range(3), values, marker=MARKERS[sex], color=COLORS[sex], linestyle="-" if sex=="women" else "--")
    axes[0,0].legend(fontsize=7, frameon=False);axes[0,1].set_ylim(0,100)
    title(axes[1,0], "C", f"Distribution overlap ({len(quantile_countries)} countries)", xlabel="Mean country score quantiles")
    axes[1,0].set_yticks([0,1],["25–29","45–49"]);axes[1,0].set_xlabel("Mean country score quantiles")
    title(axes[1,1], "D", "Upper-tail share", "Above reference 90th percentile (%)")
    axes[1,1].set_xticks(range(3),["18–29","30–44","45–64"])
    return fig


def attention(data):
    fig, axes = plt.subplots(2,2, figsize=(7.2,5.0), layout="constrained")
    bands=["18-29","30-44","45-64"]
    for sex,gender in PAIRS:
        median=[country_means(data,gender,b,"vote_quantiles")[2] for b in bands]
        axes[0,0].plot(range(len(bands)),median,color=COLORS[sex],marker=MARKERS[sex],linestyle="-" if sex=="women" else "--",label=sex.capitalize())
        for key,style in (("mean_like_rate","-"),("vote_pooled_like_rate","--")):
            axes[0,1].plot(range(len(bands)),[100*country_means(data,gender,b,key) for b in bands],color=COLORS[sex],marker=MARKERS[sex],linestyle=style)
        coverage=[r for r in data["hon"]["coverage"] if r["gender"]==gender]
        grouped={"US":[],"Other":[]}
        for r in coverage:grouped["US" if r["country"] in ("United States","United States of America") else "Other"].append(r)
        vals=[100*sum(r["inventoried"] for r in group)/sum(r["eligible"] for r in group) for group in grouped.values()]
        axes[1,0].bar(np.arange(2)+(-.15 if sex=="women" else .15),vals,width=.28,color=COLORS[sex],hatch="" if sex=="women" else "//")
        port=data["hon"]["portfolio"]["reported"][gender]
        cats=sorted(port)
        axes[1,1].plot(range(len(cats)),[port[c]["vote_quantiles"][2] for c in cats],color=COLORS[sex],marker=MARKERS[sex],linestyle="-" if sex=="women" else "--")
        axes[1,1].set_xticks(range(len(cats)),cats)
    title(axes[0,0],"A","Voting exposure","Mean country median votes (log)");axes[0,0].set_yscale("log");axes[0,0].legend(fontsize=7,frameon=False)
    title(axes[0,1],"B","Profile and vote weighting","Mean country like rate (%)")
    axes[0,1].text(.04,.06,"Solid: equal profiles\nDashed: vote-pooled",transform=axes[0,1].transAxes,fontsize=7)
    for ax in axes[0]:ax.set_xticks(range(len(bands)),bands)
    title(axes[1,0],"C","Saved-image coverage","Eligible profiles inventoried (%)");axes[1,0].set_xticks([0,1],["US","Other recorded groups"])
    title(axes[1,1],"D","Reported photos and exposure","Median votes (log)","Reported photo category");axes[1,1].set_yscale("log")
    return fig


def choice(data):
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.85),layout="constrained")
    s=data["speed_dating"]
    for sex,_ in PAIRS:
        cells=s["yes_by_panel_quintile"][sex]["quintiles"]
        errorbar(axes[0],range(1,6),[{"mean":c["estimate"],"interval_95":c["interval_95"]} for c in cells],sex,scale=100,label=sex.capitalize())
        cells=[s["adjusted_slopes"][method][sex]["coefficients_pp"]["attr"] for method in ("own","panel")]
        errorbar(axes[1],np.arange(2)+(-.07 if sex=="women" else .07),[{"mean":c["estimate"],"interval_95":c["interval_95"]} for c in cells],sex)
    title(axes[0],"A","Requests for contact","Yes (%)","Within-chooser rank band");axes[0].set_xticks(range(1,6));axes[0].set_ylim(0,100);axes[0].legend(fontsize=6.5,frameon=False)
    title(axes[1],"B","Adjusted association","Yes difference (pp per rating SD)");axes[1].set_xticks([0,1],["Own rating","Peer panel"],rotation=15);axes[1].axhline(0,color="#777",linewidth=.7)
    j=data["legacy_v9"]["speed"]["jury"]
    for key,style in (("attr","-"),("intel","--")):
        axes[2].plot(range(1,9),j[key]["curve"],linestyle=style,marker="o" if key=="attr" else "^",color="#344F63",label="Appearance" if key=="attr" else "Perceived intelligence",markersize=3)
    title(axes[2],"C","Panel agreement","Correlation of panel means","Observers per panel");axes[2].set_ylim(0,1);axes[2].legend(fontsize=6,frameon=False,loc="lower right")
    return fig


def goals(data):
    fig,axes=plt.subplots(2,2,figsize=(7.2,4.8),layout="constrained")
    p=data["pov"]["country_balanced"]["all"]
    for ax,key,letter,heading in zip(axes.flat,("serious_intent_share","want_children_share","for_long_term_share","parenthood_share"),"ABCD",("Serious relationship / marriage (intent)","Want children","Long Term (separate for question)","Reported parenthood")):
        for sex,gender in PAIRS:
            errorbar(ax,[0,1],[p["by_band"][b][gender]["outcomes"][key] for b in ("25-29","45-49")],sex,scale=100,label=sex.capitalize())
        title(ax,letter,heading,"Country-balanced share (%)","Profile age band");ax.set_xticks([0,1],["25–29","45–49"]);ax.set_ylim(0,100)
    axes[0,0].legend(fontsize=7,frameon=False)
    fig.supxlabel(f"Fixed support: {p['country_count']} countries; intervals resample countries.\nIntention nonresponse retained; parenthood uses known reports.",fontsize=7)
    return fig


def assessment(data):
    fig=plt.figure(figsize=(7.2,5.5),layout="constrained")
    grid=fig.add_gridspec(2,2,height_ratios=[1.1,1])
    ax=fig.add_subplot(grid[0,:]);ax.axis("off")
    ax.set_title("A  Three routes from activity to consequence",loc="left",fontweight="bold")
    for j,(head,obs,rival) in enumerate((("REVEAL","Better prediction of later contribution","Rival: familiarity or endorsement"),("PRODUCE","A recipient or group gains a benefit","Rival: resources, access or luck"),("ALLOCATE","A decision changes an opportunity","Rival: authority or institutional rule"))):
        x=.015+j*.335
        ax.add_patch(plt.Rectangle((x,.12),.315,.70,facecolor="#F0F3F5",edgecolor="#BFC9D0",transform=ax.transAxes))
        ax.text(x+.015,.68,head,fontweight="bold",color="#245B82",transform=ax.transAxes)
        ax.text(x+.015,.45,obs.replace("later contribution","later\ncontribution").replace("gains a benefit","gains\na benefit").replace("an opportunity","an\nopportunity"),fontsize=8,transform=ax.transAxes)
        ax.text(x+.015,.22,rival.replace(" or "," or\n"),fontsize=7,transform=ax.transAxes)
    ax=fig.add_subplot(grid[1,0]);m=np.arange(1,31)
    for k,style in ((1,"-"),(4,"--"),(12,":")):
        value=1/(1+.5+1/m+1/k+1/(m*k))
        ax.plot(m,value,linestyle=style,label=f"{k} context" + ("s" if k != 1 else ""))
    title(ax,"B","Illustrative criterion validity",r"$R^2$ with independent target","Observers");ax.set_ylim(0,1);ax.legend(fontsize=7,frameon=False)
    ax.text(.02,.03,"Variances: target 1; halo 0.5;\nobserver, context, residual each 1.",transform=ax.transAxes,fontsize=6.8)
    ax=fig.add_subplot(grid[1,1]);ax.axis("off")
    ax.set_title("C  Matched-cost prospective test",loc="left",fontweight="bold")
    steps=["Randomize relevant observations","Separate confidence from prediction","Record recipient benefit and authority","Test independent later contribution"]
    for j,line in enumerate(steps):
        y=.84-j*.22;ax.text(.03,y,line,fontsize=8,transform=ax.transAxes)
        if j<3:ax.annotate("",xy=(.08,y-.15),xytext=(.08,y-.03),xycoords="axes fraction",arrowprops={"arrowstyle":"->","color":"#74828D"})
    return fig


def outcomes(data):
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),layout="constrained")
    g=data["legacy_v9"]["gss"]["decomposition"]["income"]
    for j,(sex,_) in enumerate(PAIRS):
        if "gss_design" in data:
            design=data["gss_design"]["income"][sex]
            cells=[[design["estimate"][c],*design["interval_95"][c]] for c in ("access","number","total")]
            axes[0].scatter([g[sex][c][0] for c in ("access","number","total")],np.arange(3)+(j-.5)*.17,
                            marker=MARKERS[sex],s=45,facecolors="none",edgecolors=COLORS[sex],linewidths=.6)
        else:
            cells=[g[sex][c] for c in ("access","number","total")]
        means=np.array([c[0] for c in cells]);limits=np.array([c[1:] for c in cells]).T
        axes[0].errorbar(means,np.arange(3)+(j-.5)*.17,xerr=np.maximum(0,[means-limits[0],limits[1]-means]),fmt=MARKERS[sex],color=COLORS[sex],markersize=4,label=sex.capitalize())
    title(axes[0],"A","GSS income association",xlabel="Standardized differential")
    axes[0].set_yticks(range(3),["Parenthood","Family size","Total"]);axes[0].axvline(0,color="#777",linewidth=.7);axes[0].legend(fontsize=6.5,frameon=False)
    axes[1].axis("off");axes[1].set_title("B  Evolutionary links",loc="left",fontweight="bold")
    lines=["Informative behavior","Consequential decision","Reproductive covariance","Inherited response","Development and anatomy"]
    for i,line in enumerate(lines):
        y=.88-i*.18;axes[1].text(.03,y,line,fontsize=8.2,transform=axes[1].transAxes)
        if i<4:axes[1].annotate("",xy=(.09,y-.12),xytext=(.09,y-.025),xycoords="axes fraction",arrowprops={"arrowstyle":"->","color":"#74828D"})
    return fig


def bibliography_entries(texts, allow_legacy_doi_duplicates=False):
    entries = {}
    pattern = r"\\bibitem(?:\[[\s\S]*?\])?\{([^}]+)\}([\s\S]*?)(?=\\bibitem|\\end\{thebibliography\})"
    for source_index, text in enumerate(texts):
        for key, body in re.findall(pattern, text):
            body = body.replace(r"\newpage", "").strip()
            if key in entries:
                previous = entries[key]
                old_doi = re.findall(r"\\doi\{([^}]+)\}", previous)
                new_doi = re.findall(r"\\doi\{([^}]+)\}", body)
                # Historical resources format the same DOI differently. A different
                # DOI or differing text without a shared DOI is a conflicting key.
                if not (allow_legacy_doi_duplicates and source_index == 1 and old_doi and old_doi == new_doi) and re.sub(r"\s+", "", previous) != re.sub(r"\s+", "", body):
                    raise ValueError("Conflicting bibliography key: " + key)
            else:
                entries[key] = body
    return entries


def bibliography():
    archive=ROOT/"project/brain/tips"
    texts=[(archive/"Performance_Before_Outcomes_v9/paper/main.bbl").read_text(),
           (archive/"The_Minds_We_Choose_Final.tex").read_text(),
           (ROOT/"docs/sections/brain/minds-environment-references.tex").read_text()]
    entries={key: r"\bibitem{"+key+"}"+body+"\n" for key,body in bibliography_entries(texts, allow_legacy_doi_duplicates=True).items()}
    sources=[p.read_text() for p in sorted((ROOT/"docs/sections/brain").glob("*.tex"))]
    sources.append((archive/"Performance_Before_Outcomes_v9/paper/generated/tab_cultures.tex").read_text())
    keys=[]
    for text in sources:
        for match in re.findall(r"\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}",text):
            for key in match.split(","):
                if key not in keys:keys.append(key)
    missing=set(keys)-set(entries)
    if missing:raise ValueError(f"Missing primary bibliography entries: {sorted(missing)}")
    return r"\begin{thebibliography}{99}"+"\n"+"\n".join(entries[k] for k in keys)+r"\end{thebibliography}"+"\n"


def numerical_text(data):
    h=data["hon"];a=h["age_analysis"];p=data["pov"];k=data["surveys"]["kowal"];w=data["surveys"]["whyte"]
    s=data["speed_dating"];g=data["legacy_v9"]["gss"]
    distribution_countries=[c for c in h['common_support_countries'] if all(b in h['thresholds']['20'][c][g]['bands'] for g in ('Female','Male') for b in ('25-29','45-49'))]
    macros={"MwcHonN":f"{h['flow']['primary']:,}","MwcPovN":f"{p['records']:,}","MwcHonDistributionCountries":str(len(distribution_countries)),
            "MwcHonCountries":str(a["countries"]),"MwcKowalN":f"{sum(k['gender_counts'].values()):,}"}
    out={"minds-numbers.tex":"\n".join("\\newcommand{\\"+key+"}{"+value+"}" for key,value in macros.items())+"\n"}
    n_c=len(k["countries"])
    km=k["equal_country"]["men"];kw=k["equal_country"]["women"]
    abstract_gap=a["women_minus_men_decline"]["mean"]*100
    out["minds-abstract-results.tex"]=(f"Kindness leads country-balanced budgets in both genders across {n_c} samples. "
        f"Across {a['countries']} HON country groups, women's 25--29 versus 45--49 percentile decline exceeds men's by {abstract_gap:.1f} points. "
        f"Across {p['country_balanced']['all']['country_count']} POV countries, women more often report commitment at both ages while the gender gap in wanting children reverses. ")
    app=k["country_gender_contrasts"]["appearance"];res=k["country_gender_contrasts"]["resources"]
    out["minds-kowal-results.tex"]=(f"The reproduced budget cohort contains {sum(k['gender_counts'].values()):,} respondents; {n_c} countries meet the threshold of 100 respondents in each gender group. "
        f"With equal country weights, kindness receives {kw['kindness']['mean']:.2f} points among women and {km['kindness']['mean']:.2f} among men. "
        f"Appearance receives {kw['appearance']['mean']:.2f} and {km['appearance']['mean']:.2f}; financial prospects receive {kw['resources']['mean']:.2f} and {km['resources']['mean']:.2f}, respectively. "
        f"The women-minus-men appearance difference is {ci(app,1,2)} points and the resources difference {ci(res,1,2)} points, with country-bootstrap 95\\% intervals in brackets. "
        f"The appearance difference is negative in {app['negative']} of {n_c} country point estimates; the resources difference is positive in {res['positive']}. These sign counts do not count individually significant effects.\n")
    correlations=k['country_gender_mean_correlations']
    out['minds-kowal-context.tex']=(f"Across these {n_c} samples, gender-specific country means correlate at $r={correlations['religion']:.2f}$ for religion and $r={correlations['kindness']:.2f}$ for kindness. "
        "These country-level associations do not describe resemblance within couples.\n")
    wa=w["analyses"]["all"];full="18-64" if "18-64" in wa["women"] else "18-65"
    vals=[]
    for sex in ("women","men"):
        cell=wa[sex][full]
        vals.append(f"{sex}: trust {cell['items']['trust_import']['mean']:.1f}, intelligence {cell['items']['intelligence_import']['mean']:.1f}, attractiveness {cell['items']['attractive_import']['mean']:.1f}, openness {cell['items']['openness_import']['mean']:.1f}")
    out["minds-whyte-results.tex"]=(f"The deposited file has {w['records']:,} records, compared with the article's {w['article_headline_records']:,} headline sample. "
        "On the deposited 0--100 importance scale, the broad adult summaries are "+"; ".join(vals)+". "
        "Trust is therefore interpreted alongside openness and emotional connection, rather than as the only relational item. Complete paired contrasts and sensitivities are in the supplement.\n")
    walter=data["surveys"]["walter"]
    wc=walter.get("country_gender_contrasts_count_reconciled")
    if wc:
        appearance=wc["physatt"];resources=wc["resources"]
        out["minds-walter-results.tex"]=(f"Across all 45 country samples, the equal-country women-minus-men differences are {ci(appearance,1,2)} points for appearance and {ci(resources,1,2)} for resources. "
            f"Appearance reverses the average direction in {appearance['positive']} country point estimates and resources in {resources['negative']}; complete results retain those reversals. ")
    else:
        out["minds-walter-results.tex"]="The coding reconciliation and conditional reanalysis are documented in the supplement. "
    female=a["contrasts"]["Female"]["mean_percentile"];male=a["contrasts"]["Male"]["mean_percentile"];gap=a["women_minus_men_decline"]
    if "mean_percentile" in gap:gap=gap["mean_percentile"]
    out["minds-hon-results.tex"]=(f"Of {h['flow']['all']:,} canonical profiles, {h['flow']['primary']:,} meet primary eligibility. "
        f"The fixed-support age comparison includes {a['countries']} normalized country or territory groups and {a['profile_n_age18_64']:,} profiles aged 18--64. "
        f"The fitted 25--29 minus 45--49 percentile contrast is {ci(female,100)} points for women's profiles and {ci(male,100)} for men's profiles. "
        f"The difference in declines is {ci(gap,100)} points (women minus men). Country-bootstrap 95\\% intervals are in brackets; their width does not describe the spread across countries. "
        f"The men's contrast reverses sign in {male['negative']} countries; the full atlas preserves every estimate.\n")
    q=s["quintile_contrast_men_minus_women"]
    vals=[]
    for sex in ("women","men"):
        x=s["yes_by_panel_quintile"][sex]["quintiles"]
        vals.append(f"{sex}, {100*x[0]['estimate']:.1f}\\% in the lowest band and {100*x[-1]['estimate']:.1f}\\% in the highest")
    out["minds-speed-results.tex"]=("Yes rates increase across these bands in both groups: "+"; ".join(vals)+". "
        f"The men-minus-women difference in the top-minus-bottom gradient is {100*q['estimate']:.1f} percentage points "
        f"(session-bootstrap 95\\% interval {100*q['interval_95'][0]:.1f} to {100*q['interval_95'][1]:.1f}). "
        "The raw contrast is kept beside the adjusted models, including its uncertainty, rather than being replaced by the most favorable specification.\n")
    pb=p["country_balanced"]["all"]
    gaps=[pb["by_band"][b]["women_minus_men_pp"] for b in ("25-29","45-49")]
    out["minds-pov-results.tex"]=(f"The principal two-band all-profile comparison supports {pb['country_count']} countries: "+
        ", ".join(tex(c) for c in pb["countries"])+". "
        f"The equal-country women-minus-men difference in wanting children is {ci(gaps[0]['want_children_share'])} percentage points at ages 25--29 and {ci(gaps[1]['want_children_share'])} at 45--49. "
        f"The corresponding serious-relationship or marriage gaps are {ci(gaps[0]['serious_intent_share'])} and {ci(gaps[1]['serious_intent_share'])} points. "
        "Country-bootstrap 95\\% intervals are in brackets. The childless and coordinate-supported comparisons have their own fixed support sets and denominators.\n")
    inc=g["decomposition"]["income"]
    out["minds-gss-results.tex"]=(f"The archived GSS base contains {g['n_base']:,} respondents. In the income-specific cohorts ({inc['men']['n']:,} men; {inc['women']['n']:,} women), "
        f"men's parenthood and family-size components are {inc['men']['access'][0]:+.3f} and {inc['men']['number'][0]:+.3f}; "
        f"women's are {inc['women']['access'][0]:+.3f} and {inc['women']['number'][0]:+.3f}. "
        "These are weighted phenotypic associations in standardized-trait units. Opposed components in men illustrate why an access association alone cannot stand for the total.\n")
    if "gss_design" in data:
        gd=data["gss_design"];men=gd["income"]["men"];women=gd["income"]["women"]
        out["minds-gss-results.tex"]=(f"The reproduced GSS base contains {g['n_base']:,} adults; {gd['base_with_supported_design_n']:,} have supported released design information. "
            f"The income sensitivity retains {men['used_n']:,} men and {women['used_n']:,} women. "
            f"Men's parenthood and family-size components are {men['estimate']['access']:+.3f} and {men['estimate']['number']:+.3f}, summing to {men['estimate']['total']:+.3f} "
            f"(95\\% design-bootstrap interval {men['interval_95']['total'][0]:+.3f} to {men['interval_95']['total'][1]:+.3f}). "
            f"Women's corresponding components are {women['estimate']['access']:+.3f} and {women['estimate']['number']:+.3f}. "
            "These are weighted phenotypic associations, with the full archived baseline retained alongside the design-supported subset.\n")
    return out


def supplement_tables(data):
    out={};h=data["hon"];p=data["pov"];sur=data["surveys"]
    rows=[[tex(k),f"{v:,}"] for k,v in h["flow"].items()]
    rows += [["POV source accounts",f"{p['records']:,}"],["POV primary eligible",f"{p['eligible']:,}"]]
    rows += [["Kowal: "+tex(k),str(v)] for k,v in sur["kowal"]["flow"].items()]
    rows += [["Whyte deposit",str(sur["whyte"]["records"])],["Whyte article headline",str(sur["whyte"]["article_headline_records"])]]
    out["minds-flow.tex"]=table(["Sequential source-specific criterion","Records"],rows,"p{11cm}r")
    rows=[]
    for r in h["coverage"]:
        rows.append([tex(r["country"]),"W" if r["gender"]=="Female" else "M",f"{r['eligible']:,}",f"{r['inventoried']:,}",fnum(100*r['inventoried']/r['eligible'] if r['eligible'] else None)])
    out["minds-coverage.tex"]=table(["Recorded country / territory","Holder","Eligible","Inventoried","Coverage (\\%)"],rows,"p{6.6cm}lrrr")
    out["minds-coverage.csv"]=csv_bytes(["country","recorded_gender","eligible_profiles","inventoried_profiles","percent"],[[r["country"],r["gender"],r["eligible"],r["inventoried"],100*r["inventoried"]/r["eligible"] if r["eligible"] else None] for r in h["coverage"]])
    rows=[];csvrows=[]
    for c in h["age_analysis"]["country_contrasts"]:
        row=[tex(c["country"])]
        for gender in ("Female","Male"):
            row.extend([f"{c[gender]['n_age18_64']:,}",f"{100*c[gender]['contrasts']['mean_percentile']['young_minus_older']:.2f}"])
        row.append(f"{100*c['women_minus_men_decline']:.2f}");rows.append(row)
        csvrows.append([c["country"],c["Female"]["n_age18_64"],c["Female"]["contrasts"]["mean_percentile"]["young_minus_older"],c["Male"]["n_age18_64"],c["Male"]["contrasts"]["mean_percentile"]["young_minus_older"],c["women_minus_men_decline"]])
    out["minds-hon-contrasts.tex"]=table(["Country / territory","W $n$","W pp","M $n$","M pp","W--M pp"],rows,"p{5.5cm}rrrrr")
    out["minds-hon-contrasts.csv"]=csv_bytes(["country","women_n18_64","women_young_minus_old_fraction","men_n18_64","men_young_minus_old_fraction","women_minus_men_decline_fraction"],csvrows)
    rows=[];csvrows=[]
    for threshold,countries_ in h["thresholds"].items():
        for country,cells in countries_.items():
            for gender,holder in cells.items():
                for band,cell in holder["bands"].items():
                    csvrows.append([threshold,country,gender,band,cell["n"],cell["score_n"],cell["mean_score"],cell["mean_percentile"],cell["mean_like_rate"],cell["vote_pooled_like_rate"],cell["vote_quantiles"][2],cell["inventoried_n"]])
    out["minds-hon-age-cells.csv"]=csv_bytes(["minimum_votes","country","holder_gender","age_band","n","score_n","mean_score","mean_fixed_country_gender_percentile","equal_profile_like_rate","vote_pooled_like_rate","median_votes","inventoried_n"],csvrows)
    rows=[[tex(k),f"{v:,}",f"{100*v/p['records']:.2f}"] for k,v in p["mapping_status_all"].items()]
    out["minds-geography.tex"]=table(["Mutually exclusive mapping status","Accounts","Share (\\%)"],rows,"p{9cm}rr")
    rows=[]
    for support,cells in p["country_balanced"].items():
        for band,groups in cells["by_band"].items():
            for gender in ("Female","Male"):
                x=groups[gender];o=x["outcomes"]
                rows.append([tex(support),band,"W" if gender=="Female" else "M",str(cells["country_count"]),f"{x['n']:,}",ci(o["serious_intent_share"],100),ci(o["want_children_share"],100)])
    out["minds-pov-balanced.tex"]=table(["Cohort","Age","Holder","Countries","$n$","Commitment \\% [95\\%]","Want children \\% [95\\%]"],rows,"p{2.4cm}llrrp{3.1cm}p{3.1cm}")
    rows=[];csvrows=[]
    for country,groups in p["countries"].items():
        for gender,bands in groups.items():
            for band,cell in bands.items():
                if cell["n"]<100:continue
                wc=cell["want_counts"]
                csvrows.append([country,gender,band,cell["n"],cell["parenthood_share"],cell["serious_intent_share"],cell["for_long_term_share"],*[wc.get(k,0) for k in ("want","do_not_want","undecided_open","nonresponse")]])
                if country in p["common_support_countries"]["all"] and band in ("25-29","45-49"):
                    rows.append([tex(country),band,"W" if gender=="Female" else "M",f"{cell['n']:,}",f"{100*cell['serious_intent_share']:.1f}",f"{100*cell['want_children_share']:.1f}",f"{100*wc['nonresponse']/cell['n']:.1f}"])
    out["minds-pov-country.tex"]=table(["Derived country","Age","Holder","$n$","Commit \\%","Want \\%","Missing \\%"],rows,"p{5cm}llrrrr")
    out["minds-pov-age-cells.csv"]=csv_bytes(["derived_country","holder_gender","age_band","n","known_parenthood_share","serious_intent_share","for_long_term_share","want_children_n","do_not_want_n","undecided_open_n","nonresponse_n"],csvrows)
    rows=[]
    for name,attribute in p["attributes"].items():
        for gender,categories in attribute["unadjusted"].items():
            for category,cell in categories.items():
                if cell["n"]<100:continue
                rows.append([tex("Duration field" if name=="relationshipage_id" else name),tex(category),"W" if gender=="Female" else "M",f"{cell['n']:,}",fnum(100*cell["parenthood_share"],1),fnum(100*cell["serious_intent_share"],1),fnum(100*cell["want_children_share"],1)])
    out["minds-pov-attributes.tex"]=table(["Attribute","Reported category","Holder","$n$","Parent \\%","Commit \\%","Want \\%"],rows,"p{2.2cm}p{5cm}lrrrr")
    rows=[]
    for name,record in sur["kowal"]["countries"].items():
        for item in ITEMS:
            wm=record["by_gender"]["women"][item];mm=record["by_gender"]["men"][item];c=record["contrasts"][item]
            rows.append([tex(name),LABELS[item],str(wm["n"]),f"{wm['mean']:.2f}",str(mm["n"]),f"{mm['mean']:.2f}",f"{c['women_minus_men']:.2f}"])
    out["minds-kowal-atlas.tex"]=table(["Country sample","Budget item","W $n$","W mean","M $n$","M mean","W--M"],rows,"p{4.4cm}p{3cm}rrrrr")
    rows=[]
    for method,groups in sur["whyte"]["analyses"].items():
        for sex,bands in groups.items():
            for band,cell in bands.items():
                for item,estimate in cell["items"].items():
                    rows.append([tex(method),sex[0].upper(),band,LABELS[item],str(estimate["n"]),ci(estimate,1,2)])
    out["minds-whyte-complete.tex"]=table(["Analysis","Respondent","Age","Item","$n$","Mean [95\\%]"],rows,"p{3.2cm}llp{3.6cm}rp{3.6cm}")
    rows=[]
    for r in sur["eastwick"]["attributes"]:
        rows.append([tex(r["attribute"]),str(r["n"]),f"{r['stated_mean']:.2f}",f"{r['stated_sd']:.2f}",str(r["stated_rank"]),f"{r['evaluation_beta']:.2f}",str(r["evaluation_rank"])])
    out["minds-eastwick.tex"]=table(["Attribute","$n$","Ideal mean","SD","Rank","Evaluation $\\beta$","Rank"],rows,"p{6cm}rrrrrr")
    rows=[]
    s=data["speed_dating"]
    for method in ("own","panel"):
        for sex in ("women","men"):
            group=s["adjusted_slopes"][method][sex]
            for attr,c in group["coefficients_pp"].items():
                rows.append([method,sex,attr,str(group["n_decisions"]),f"{c['estimate']:.2f} [{c['interval_95'][0]:.2f}, {c['interval_95'][1]:.2f}]"])
    out["minds-speed-complete.tex"]=table(["Ratings","Chooser","Attribute","Decisions","pp per SD [95\\%]"],rows,"lllrp{5cm}")
    rows=[];csvrows=[]
    waves=data["experiments"]["waves"]
    for r in waves:
        m=r.get("metrics") or {};test=m.get("profile_test",{});track=m.get("track",(r.get("config") or {}).get("track",""))
        row=[r["wave"],tex(track),str(test.get("n","--")),fnum(test.get("beauty_mae"),4),fnum(test.get("spearman_r"),4),fnum(test.get("r2"),4),"Complete" if r["complete"] else "Unfinished"]
        rows.append(row);csvrows.append([r["wave"],track,test.get("n"),test.get("beauty_mae"),test.get("spearman_r"),test.get("r2"),r["complete"],r.get("metrics_sha256")])
    out["minds-experiments.tex"]=table(["Wave","Inputs / track","Test $n$","MAE","Spearman","Score $R^2$","Cutoff status"],rows,"lp{4cm}rrrrl")
    out["minds-experiments.csv"]=csv_bytes(["wave","input_track","profile_test_n","beauty_mae","spearman_rank_correlation","beauty_regression_r2","complete_at_cutoff","metrics_sha256"],csvrows)
    rows=[]
    for name,groups in data["legacy_v9"]["gss"]["decomposition"].items():
        for sex,r in groups.items():
            rows.append([name,sex,f"{r['n']:,}",*[f"{r[k][0]:+.3f} [{r[k][1]:+.3f}, {r[k][2]:+.3f}]" for k in ("access","number","total")]])
    out["minds-gss-complete.tex"]=table(["Measured trait","Group","$n$","Parenthood","Family size","Total"],rows,"llrp{3.0cm}p{3.0cm}p{3.0cm}")
    return out


def hon_atlas(data):
    stream=BytesIO()
    records=data["hon"]["age_analysis"]["country_contrasts"]
    with PdfPages(stream,metadata={"CreationDate":None,"ModDate":None,"Title":"HON country atlas: observed country-gender referenced percentiles"}) as pdf:
        for start in range(0,len(records),6):
            fig,axes=plt.subplots(3,2,figsize=(7.2,8.4),layout="constrained")
            for ax,r in zip(axes.flat,records[start:start+6]):
                for sex,gender in PAIRS:
                    exact=data["hon"]["thresholds"]["20"][r["country"]][gender]["exact_age"]
                    ax.plot([x["age"] for x in exact],[100*x["mean_percentile"] if x["n"]>=100 else np.nan for x in exact],color=COLORS[sex],linestyle="-" if sex=="women" else "--",marker=MARKERS[sex],markersize=2,label=f"{sex.capitalize()} n={r[gender]['n_age18_64']:,}")
                ax.set_title(r["country"],fontweight="bold");ax.set_ylim(0,100);ax.set_xlim(18,64);ax.set_xlabel("Profile age");ax.set_ylabel("Mean percentile");ax.legend(fontsize=6,frameon=False)
            for ax in list(axes.flat)[len(records[start:start+6]):]:ax.axis("off")
            fig.suptitle("Observed HON country age profiles: fixed adult country/gender reference",fontsize=10)
            pdf.savefig(fig);plt.close(fig)
    return stream.getvalue()


def pov_atlas(data):
    stream=BytesIO();names=data["pov"]["country_balanced"]["all"]["countries"]
    bands=["18-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-79"]
    centers=[sum(map(int,b.split("-")))/2 for b in bands]
    outcomes_=[("serious_intent_share","Serious relationship / marriage"),("want_children_share","Want children"),
               ("for_long_term_share","Long Term (separate for question)"),("parenthood_share","Parenthood among known reports")]
    with PdfPages(stream,metadata={"CreationDate":None,"ModDate":None,"Title":"POV supported country age atlas"}) as pdf:
        for start in range(0,len(names),2):
            fig,axes=plt.subplots(4,2,figsize=(7.2,9.2),layout="constrained")
            for column,name in enumerate(names[start:start+2]):
                for row,(key,label) in enumerate(outcomes_):
                    ax=axes[row,column]
                    for sex,gender in PAIRS:
                        cells=data["pov"]["countries"][name][gender]
                        values=[cells[b][key]*100 if b in cells and cells[b][key] is not None else np.nan for b in bands]
                        ax.plot(centers,values,color=COLORS[sex],linestyle="-" if sex=="women" else "--",marker=MARKERS[sex],markersize=3,label=sex.capitalize())
                    ax.set_title((name+"\n" if row==0 else "")+label,fontsize=8.5,fontweight="bold")
                    ax.set_ylim(0,100);ax.set_ylabel("Share (%)");ax.set_xticks([20,40,60,80]);ax.set_xlabel("Profile age (band midpoint)")
                    if row==0:ax.legend(fontsize=6.5,frameon=False)
            fig.suptitle("POV: observed age patterns within validated derived countries; cells n ≥ 100",fontsize=9.5)
            pdf.savefig(fig);plt.close(fig)
    return stream.getvalue()


def detail_outputs(data):
    """Release complete scalar result families, with long vectors kept in the reference."""
    def leaves(obj,path="",depth=0):
        if isinstance(obj,dict):
            for key,value in obj.items():yield from leaves(value,path+" / "+str(key) if path else str(key),depth+1)
        elif isinstance(obj,list):
            if not obj:yield path,"Empty"
            elif all(isinstance(v,(str,int,float,bool)) or v is None for v in obj) and len(obj)<=12:
                yield path,", ".join(fnum(v,4) if isinstance(v,float) else str(v) for v in obj)
            elif all(isinstance(v,str) for v in obj):yield path,", ".join(obj)
            else:
                for j,value in enumerate(obj):yield from leaves(value,path+f" / {j+1}",depth+1)
        else:
            if isinstance(obj,float):obj=fnum(obj,5)
            yield path,str(obj)
    def describe(obj):
        rows=[]
        for key,value in leaves(obj):
            if re.fullmatch(r"[0-9a-f]{64}",value):value=value[:16]+"... (full hash in reference)"
            left=tex(key).replace(" / ",r" /\allowbreak{} ")
            right=tex(value)
            rows.append([left,right])
        return table(["Quantity / definition","Recorded result"],rows,"p{5.8cm}p{10.1cm}")
    h=data["hon"];a=h["age_analysis"];p=data["pov"];s=data["surveys"]
    provenance={"platform":{k:v for k,v in data["provenance"].items() if k not in ("exports",)},
                "survey_sources":s["sources"],"measurement_register":data.get("source_register",{}).get("measurement_register",{}),
                "analysis_sha256":data.get("analysis_sha256",{})}
    portfolio={k:v for k,v in h["portfolio"].items() if k not in ("reported","inventoried")}
    details={
        "minds-provenance.tex":provenance,
        "minds-hon-sensitivities.tex":{"threshold":a["threshold_sensitivity"],"geography":a["geographic_sensitivity"],"rank_references":a["contrasts"]},
        "minds-hon-coverage-notes.tex":{"text_field_coverage":h["field_coverage"],"portfolio_adjustment":portfolio,"processing":data["experiments"]["image_processing_summary"],"finalization":data["experiments"]["finalization_artifacts"]},
        "minds-geography-detail.tex":{"boundary_sources":data.get("geography",{}).get("sources",[]),"selection":p["geography_selection"],"support_sets":p["common_support_countries"]},
        "minds-pov-sensitivities.tex":p.get("country_balanced_sensitivities",p.get("sensitivity_analysis",{})),
        "minds-pov-adjusted.tex":{"attribute_standardization":p["attribute_standardization"],"height_audit":p["height_audit"],"profession":p["profession"]},
        "minds-kowal-additional.tex":{"age_standardized_contrasts":s["kowal"]["age_standardized_contrasts"],"shared_country_variation":s["kowal"]["country_gender_mean_correlations"],"self_quality_partnering":s["kowal"]["self_quality_partnering"]},
        "minds-walter.tex":{k:v for k,v in s["walter"].items() if k not in ("countries_by_recorded_code","countries_count_reconciled","pooled_count_reconciled","equal_country_count_reconciled","country_gender_contrasts_count_reconciled")},
        "minds-eastwick-gender.tex":{"gender_composites":s["eastwick"]["gender_composites"],"interpretation":s["eastwick"]["interpretation"]},
        "minds-speed-robustness.tex":{"accounting":data["speed_dating"].get("additional_accounting",{}),"session_deletion":data["speed_dating"].get("session_deletion_models",{})},
        "minds-experiment-audit.tex":{"interpretation":data["experiments"].get("interpretation",{}),"comparison_audit":data["experiments"].get("comparison_audit",[])},
        "minds-fossils.tex":{k:v for k,v in data["legacy_v9"]["fossils"].items() if k in ("n","span","nboot","models","bp_ci","rates")},
        "minds-gss-design.tex":data.get("gss_design",{"status":"Design refit in progress in this draft"}),
    }
    out={key:describe(obj) for key,obj in details.items()}
    if "countries_count_reconciled" in s["walter"]:
        rows=[]
        for country,record in s["walter"]["countries_count_reconciled"].items():
            for trait,contrast in record["contrasts"].items():
                w=record["by_gender"]["women"][trait];m=record["by_gender"]["men"][trait]
                rows.append([tex(country),tex(trait),str(w["n"]),f"{w['mean']:.2f}",str(m["n"]),f"{m['mean']:.2f}",f"{contrast['women_minus_men']:.2f}"])
        out["minds-walter.tex"]+=table(["Country sample","Desired quality","W $n$","W mean","M $n$","M mean","W--M"],rows,"p{4.7cm}p{2.8cm}rrrrr")
    # Keep the PDF readable; full scalar/vector families remain in JSON/CSV.
    rows=[]
    for threshold,record in a["threshold_sensitivity"].items():
        for sex in ("Female","Male"):
            rows.append(["Observed, votes " + threshold,"W" if sex=="Female" else "M",str(len(record["countries"])),ci(record[sex]["mean_percentile"],100)])
    for name,record in a["geographic_sensitivity"].items():
        for sex in ("Female","Male"):
            rows.append([tex(name),"W" if sex=="Female" else "M",str(len(record["countries"])),ci(record[sex],100)])
    for sex in ("Female","Male"):
        for key in ("mean_percentile","mean_global_gender_percentile","mean_pooled_percentile"):
            rows.append([tex(key),"W" if sex=="Female" else "M",str(a["countries"]),ci(a["contrasts"][sex][key],100)])
    out["minds-hon-sensitivities.tex"]=("Observed threshold contrasts use only countries with at least 100 profiles in both narrow bands and genders; they differ from fitted contrasts on the principal broad-band support set. Country sets and all source cells are retained in the JSON and CSV.\n"+
        table(["Specification / reference","Holder","Countries","25--29 minus 45--49 pp [95\\%]"],rows,"p{7cm}lrp{5.5cm}"))
    rows=[]
    for gender,record in a.get('profile_size_weighted',{}).get('by_gender',{}).items():
        for outcome,cell in record['outcomes'].items():
            scale=1 if outcome=='mean_score' else 100
            rows.append(["W" if gender=='Female' else "M",tex(outcome),str(record['profiles']),ci(cell,scale,2)])
    out['minds-profile-weighted.tex']=(
        "This sensitivity weights fitted country contrasts by each gender's country profile count aged 18--64, held fixed across age. It differs from allowing country composition to vary at each age. Scores are in their 0--10 units; rank and rate contrasts are in percentage points.\n"+
        table(['Holder','Outcome','Profile-weight sum','Weighted contrast [95\\%]'],rows,'lp{6.7cm}rp{4cm}'))
    rows=[]
    global_baseline=h.get('profile_weighted_baseline',{})
    for band in ['all_ages','25-29','45-49','65-69','75-79']:
        for gender,subsets in global_baseline.get('ratings',{}).get('votes_20',{}).get(band,{}).items():
            for subset,cell in subsets.items():
                rows.append([tex(band),"W" if gender=='Female' else "M",subset,str(cell['n']),f"{100*cell['mean_profile_like_rate']:.2f}",f"{100*cell['vote_pooled_like_rate']:.2f}",f"{cell['median_votes']:.0f}"])
    out['minds-profile-weighted.tex']+=(
        "The exactly reproduced legacy global profile-weighted baseline includes all eligible recorded countries, including cells too small for country-level comparisons. It provides source-wide equal-profile and vote-pooled like rates and the separate inventoried-image subset.\n"+
        table(['Age','Holder','Subset','$n$','Mean rate \\%','Vote-pooled \\%','Median votes'],rows,'lllr rrr'))
    rows=[];csvrows=[]
    for method,selection in data["speed_dating"].get("session_deletion_models",{}).items():
        for complete,groups in selection.items():
            for sex in ("women","men"):
                r=groups[sex]
                for attr,value in r["coefficients_pp"].items():
                    interval=r["leave_one_session_out_range_pp"][attr]
                    rows.append([method,tex(complete.replace("_complete","")),sex[0].upper(),attr,str(r["n_decisions"]),f"{value:.2f}",f"{interval[0]:.2f} to {interval[1]:.2f}"])
                    for session,estimate in enumerate(r["leave_one_session_out_pp"][attr],1):csvrows.append([method,complete,sex,attr,session,estimate])
    out["minds-speed-robustness.tex"]=(describe(data["speed_dating"].get("additional_accounting",{}))+
        "Session deletion ranges below are sensitivity ranges, not confidence intervals. Each deletion estimate is released in the companion CSV.\n"+
        table(["Ratings","Cases","Chooser","Trait","Decisions","pp/SD","Deletion range"],rows,"lp{2.9cm}llrrp{3cm}"))
    out["minds-session-deletions.csv"]=csv_bytes(["ratings","complete_case_rule","chooser_gender","trait","deleted_session_index","coefficient_pp_per_sd"],csvrows)
    rows=[]
    for r in data["experiments"].get("comparison_audit",[]):
        rows.append([r["wave"],r["reference"],r["unit"],f"{r['recomputed_delta_mae']:+.6f}",f"{r['stored_paired_delta_mae']:+.6f}","Yes" if r["matching_metric_delta"] else "No"])
    out["minds-experiment-audit.tex"]=(describe(data["experiments"].get("interpretation",{}))+
        "The table compares subtraction of frozen metric files against reported paired deltas. A matching point difference alone does not independently verify a paired interval without full matching prediction vectors.\n"+
        table(["Wave","Reference","Unit","Metric difference","Stored paired delta","Match"],rows,"lllrrl"))
    baseline=data["experiments"].get("age_gender_baseline")
    if baseline:
        rows=[]
        for name,record in baseline['models'].items():
            m=record['test_metrics'];rows.append([tex(name),str(m['n']),fnum(m['beauty_mae'],4),fnum(m['beauty_rmse'],4),fnum(m['r2'],4),fnum(m['spearman_r'],4)])
        for name,record in baseline['matching_frozen_wave_metrics'].items():
            m=record['saved_profile_test'];rows.append([name+' (frozen)',str(m['n']),fnum(m['beauty_mae'],4),fnum(m['beauty_rmse'],4),fnum(m['r2'],4),fnum(m['spearman_r'],4)])
        meta={key:baseline[key] for key in ['protocol','split_counts','eligibility','source_sha256','cohort_fingerprints','limitations']}
        out['minds-model-baseline.tex']=(
            "A new metadata-only diagnostic uses the inherited frozen benchmark, including ages 18--99 rather than the main HON cohort's 18--79. Natural-age spline coefficients, means and scales are fitted within recorded gender using only 8,000 training profiles; validation is diagnostic without tuning. Sample metadata and target-variance checks support comparison with the listed frozen waves, while complete paired prediction vectors remain unavailable. Algorithms differ, so these differences are not controlled input ablations.\n"+
            table(['Model / diagnostic','Test $n$','MAE','RMSE','Score $R^2$','Spearman'],rows,'p{6cm}rrrrr')+describe(meta))
    else:
        out['minds-model-baseline.tex']=''
    rows=[];subgroup_csv=[]
    for wave in data['experiments']['waves']:
        if not wave['complete']:continue
        for family,categories in wave['metrics'].get('subgroups_profile_test',{}).items():
            for category,m in categories.items():
                if not isinstance(m,dict) or 'n' not in m:continue
                rows.append([wave['wave'],tex(family),tex(category),str(m['n']),fnum(m.get('beauty_mae'),4),fnum(m.get('bias'),4),fnum(m.get('r2'),4),fnum(m.get('spearman_r'),4)])
                subgroup_csv.append([wave['wave'],family,category,m['n'],m.get('beauty_mae'),m.get('bias'),m.get('r2'),m.get('spearman_r')])
    if baseline:
        for name,model in baseline['models'].items():
            for family in ['test_by_gender','test_by_age_band']:
                for category,m in model[family].items():
                    rows.append([tex(name),tex(family.removeprefix('test_by_')),tex(category),str(m['n']),fnum(m.get('beauty_mae'),4),fnum(m.get('bias'),4),fnum(m.get('r2'),4),fnum(m.get('spearman_r'),4)])
                    subgroup_csv.append([name,family.removeprefix('test_by_'),category,m['n'],m.get('beauty_mae'),m.get('bias'),m.get('r2'),m.get('spearman_r')])
    out['minds-model-subgroups.tex']=table(['Model','Group variable','Profile group','$n$','MAE','Bias','Score $R^2$','Spearman'],rows,'p{3.5cm}p{2.7cm}p{2cm}rrrrr')
    out['minds-model-subgroups.csv']=csv_bytes(['model','group_variable','profile_group','n','mae','bias','score_r2','spearman'],subgroup_csv)
    if "gss_design" in data:
        gd=data["gss_design"];rows=[]
        for sex in ("women","men"):
            r=gd["income"][sex]
            for key in ("access","number","total"):
                limits=r["interval_95"][key]
                rows.append([sex,key,str(r["used_n"]),f"{r['estimate'][key]:+.4f}",f"[{limits[0]:+.4f}, {limits[1]:+.4f}]",str(r["topcoded_children_n"])])
        meta={"source":gd["source"],"base_count":gd["n_base"],"design_supported_base":gd["base_with_supported_design_n"],
              "design":{k:v for k,v in gd["design"].items() if k not in ("stratum_psu_counts",)},
              "measurement":gd["measurement"],"topcoding":gd["topcoding"],"weight_policy":gd["weight_policy"],"documentation":gd["documentation"]}
        weights=gd["weight_sensitivity"]
        out["minds-gss-design.tex"]=(describe(meta)+table(["Group","Component","$n$","Estimate","PSU-bootstrap 95\\%","8+ children $n$"],rows,"llrrlr")+describe({"alternative_weight_sensitivity":weights}))
    rows=[]
    for scenario,record in p.get("country_balanced_sensitivities",{}).items():
        if not record["country_count"]:continue
        for band,groups in record["by_band"].items():
            gap=groups["women_minus_men_pp"]
            rows.append([tex(scenario),band,str(record["country_count"]),str(groups["Female"]["n"]),str(groups["Male"]["n"]),ci(gap["want_children_share"]),ci(gap["serious_intent_share"])])
    out["minds-pov-sensitivities.tex"]=table(["Childless scenario","Age","Countries","W $n$","M $n$","Want W--M pp [95\\%]","Commit W--M pp [95\\%]"],rows,"p{3cm}lrrrp{3.5cm}p{3.5cm}")
    rows=[];category_rows=[]
    for country,groups in p["countries"].items():
        for gender,bands in groups.items():
            for band,cell in bands.items():
                if country in p["country_balanced"]["all"]["countries"]:
                    c=cell["want_counts"]
                    rows.append([tex(country),band,"W" if gender=="Female" else "M",str(cell["n"]),*[str(c[k]) for k in ("want","do_not_want","undecided_open","nonresponse")]])
                for question,key in (("wantchildren","want_counts"),("intent","intent_counts"),("for","for_counts")):
                    for response,count in cell[key].items():
                        if count:category_rows.append([country,gender,band,question,response,count,cell["n"]])
    out["minds-pov-response-counts.tex"]=table(["Derived country","Age","Holder","$n$","Want","Do not want","Undecided","Missing"],rows,"p{4.4cm}llrrrrr")
    out["minds-pov-response-counts.csv"]=csv_bytes(["derived_country","holder_gender","age_band","question","response","response_n","eligible_n"],category_rows)
    rows=[]
    for method,groups in s["whyte"]["analyses"].items():
        for sex,bands in groups.items():
            for band,cell in bands.items():
                for key,estimate in cell["paired_contrasts"].items():
                    rows.append([tex(method),sex,band,tex(key),str(estimate["n"]),ci(estimate,1,2)])
    out["minds-whyte-paired.tex"]=table(["Analysis","Group","Age","Paired difference","$n$","Mean [95\\%]"],rows,"p{3cm}llp{4.2cm}rp{3.1cm}")
    rows=[]
    for name,groups in data["legacy_v9"]["gss"]["decomposition"].items():
        for sex,record in groups.items():
            for cohort,c in record["by_cohort"].items():
                rows.append([name,sex,cohort,str(c["n"]),*[f"{c[k][0]:+.3f}" for k in ("access","number","total")]])
    out["minds-gss-cohorts.tex"]=table(["Trait","Group","Birth cohort","$n$","Parenthood","Family size","Total"],rows,"lllr rrr")
    cultures=data["legacy_v9"]["cultures"]
    # This archived generated table is checked against the original ZIP before reuse.
    folder=ROOT/"project/brain/tips/Performance_Before_Outcomes_v9"
    content=(folder/"paper/generated/tab_cultures.tex").read_bytes()
    from zipfile import ZipFile
    with ZipFile(folder.parent/"Performance_Before_Outcomes_v9_ALL.zip") as archive:
        assert archive.read("Performance_Before_Outcomes_v9/paper/generated/tab_cultures.tex")==content
    out["minds-ethnography.tex"]=(r"{\footnotesize\begin{longtable}{@{}p{4cm}p{2.3cm}rp{8.5cm}@{}}\toprule"+"\n"+content.decode()+r"\bottomrule\end{longtable}}"+"\n")
    out["minds-result-families.json"]=json.dumps(details,sort_keys=True,indent=2)+"\n"
    return out


def generate(data):
    outputs={}
    for name,fn in (("overview",overview),("preferences",preferences),("countries",countries),("beauty",beauty),
                    ("attention",attention),("choice",choice),("goals",goals),("assessment",assessment),("outcomes",outcomes)):
        fig=fn(data)
        for extension in ("pdf","svg"):
            stream=BytesIO();fig.savefig(stream,format=extension,metadata={"CreationDate":None,"ModDate":None} if extension=="pdf" else {"Date":None})
            outputs[f"minds-{name}.{extension}"]=stream.getvalue()
        plt.close(fig)
    from brain.minds_environment_paper import outputs as environment_outputs
    outputs.update(environment_outputs(data["fossil_environment"]))
    outputs.update(numerical_text(data))
    outputs.update(supplement_tables(data))
    outputs.update(detail_outputs(data))
    outputs["minds-references.tex"]=bibliography()
    outputs["minds-hon-atlas.pdf"]=hon_atlas(data)
    outputs["minds-pov-atlas.pdf"]=pov_atlas(data)
    outputs["minds-artifacts.json"]=json.dumps({"study":"minds-we-choose","paper":"human-brain-and-behavior",
        "reference_sha256":hashlib.sha256((json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()).hexdigest(),
        "generated_sha256":{name:hashlib.sha256(value.encode() if isinstance(value,str) else value).hexdigest() for name,value in outputs.items()},
        "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted([*(ROOT/"docs/sections/brain").glob("*.tex"),Path(__file__)])},
        "files":sorted(outputs),"figure_population":"Distinct source-specific populations; no SAFE AI experimental evidence",
        "illustration_variances":{"target":1,"halo":.5,"observer":1,"context":1,"residual":1}},indent=2)+"\n"
    return outputs
