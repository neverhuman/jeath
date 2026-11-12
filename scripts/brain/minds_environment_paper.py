"""Render environmental context exclusively from the validated human reference."""
from io import BytesIO, StringIO
import csv
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def fossil_panel(ax, data, name, limits, inset=False):
    records = data['fossils'][name]
    rows = data['bins'][name]
    ax.axvspan(data['published_stasis']['younger_ma'], data['published_stasis']['older_ma'],
               color='.91', zorder=0)
    ax.scatter([r['age_ma'] for r in records], [r['cranial_volume_cm3'] for r in records],
               color='.67', s=9 if not inset else 2.5, linewidths=0, zorder=2)
    means = [np.nan if r['n'] == 0 else r['geometric_mean_cm3'] for r in rows]
    ax.plot([r['midpoint_ma'] for r in rows], means, '-o', color='black',
            markersize=4 if not inset else 2, linewidth=.7, zorder=3)
    ax.set_xlim(*limits)
    ax.set_ylim(200, 1900)
    if not inset:
        ax.set_ylabel('Cranial volume (cm³)')
    ax.tick_params(labelsize=7 if not inset else 5.5)


def environment_figure(data):
    events = {r['id']: r for r in data['events']}
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7.2, 6.7),
                             gridspec_kw={'height_ratios': [2, 1.45, 1.65]}, layout='constrained')
    fig.suptitle('Brain-volume trajectories and environmental context', fontsize=11, fontweight='bold')
    a, b, c = axes
    fossil_panel(a, data, 'gingerich', data['display_range_ma'])
    a.set_title(f"A  Fossil cranial volume ({data['displayed_consensus_n']} of {data['counts']['gingerich']} observations)", fontsize=9)
    a.text(1.28, 1810, 'Published interval of relative stasis', fontsize=7.1, va='top')
    a.text(1.68, 280, 'Gray: specimens   Black: 100-kyr geometric means', fontsize=6.7)
    small = a.inset_axes([.06, .51, .30, .39])
    fossil_panel(small, data, 'gingerich', [3.3, 0], inset=True)
    small.axvspan(.5, 1.7, facecolor='none', edgecolor='.3', linewidth=.8)
    small.set_xticks([3, 2, 1, 0]);small.set_yticks([500, 1500])
    small.set_title('Complete trajectory (Ma)', fontsize=6, pad=1)
    climate = data['climate']
    b.plot([r['age_ma'] for r in climate], [r['benthic_d18o_per_mil'] for r in climate],
           color='#304f60', linewidth=.8)
    mpt = events['mpt'];mis = events['mis34'];terminal = events['mis34_terminal']
    b.axvspan(mpt['younger_ma'], mpt['older_ma'], facecolor='.9', zorder=0)
    b.axvspan(mis['younger_ma'], mis['older_ma'], facecolor='.65', zorder=0)
    b.set_ylim(3, 5.1);b.invert_yaxis()
    b.set_ylabel('Benthic δ¹⁸O (‰)')
    b.set_title(f"B  LR04 stack ({data['climate_source']['records_averaged']} records; published age model)", fontsize=9)
    b.text(1.23, 3.14, 'Middle Pleistocene Transition', fontsize=7)
    b.annotate('MIS 34', xy=(1.14, 4.65), xytext=(1.38, 4.95), fontsize=7,
               arrowprops={'arrowstyle': '-', 'linewidth': .6, 'color': '.3'})
    c.set_title('C  Dated evidence and hypotheses', fontsize=9)
    c.set_ylim(-.55, 3.65)
    c.set_yticks([3, 2, 1, 0], ['Global climate', 'N Atlantic / SW Europe', 'Australasian impact', 'Ancestral demography'], fontsize=7)
    for e, y in [(mpt, 3), (mis, 2), (terminal, 2)]:
        c.add_patch(Rectangle((e['younger_ma'], y-.13), e['older_ma']-e['younger_ma'], .26,
                              facecolor='.3' if e['id'].endswith('terminal') else '.7', edgecolor='.2', linewidth=.6))
    c.text(1.24, 3.24, 'MPT', fontsize=7)
    c.annotate('MIS 34; dark tip = terminal stadial', xy=(1.125, 2), xytext=(1.67, 2.36),
               fontsize=6.6, arrowprops={'arrowstyle': '-', 'linewidth': .6})
    impact = events['impact']
    c.plot(impact['age_ma'], 1, marker='D', color='black', markersize=5)
    c.text(impact['age_ma']+.025, 1.02, f"≈{impact['age_ma']:.2f} Ma", fontsize=7, ha='right')
    bottleneck = events['bottleneck']
    c.add_patch(Rectangle((bottleneck['younger_ma'], -.13), bottleneck['older_ma']-bottleneck['younger_ma'], .26,
                          facecolor='white', edgecolor='.2', linewidth=1, linestyle='--', hatch='////'))
    c.text(bottleneck['older_ma']+.025, .03, 'Bottleneck: disputed', fontsize=7, ha='right')
    c.set_xlabel('Millions of years ago (older at left)')
    c.set_xticks(np.arange(.5, 1.71, .1))
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.set_xlim(*data['display_range_ma'])
    return fig


def comparison_figure(data):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7.2, 7.2), layout='constrained')
    names = [('gingerich', 'Gingerich consensus'), ('du', 'Du compilation'), ('will', 'Will compilation')]
    for ax, (name, label) in zip(axes, names):
        fossil_panel(ax, data, name, [3.3, 0])
        if name == 'du':
            # Only actual supplied numeric bounds; these are not generic 95% CIs.
            for r in data['fossils'][name]:
                bounds = r['dating_bounds_ma']
                if bounds is not None:
                    ax.plot(bounds, [r['cranial_volume_cm3']]*2, color='.65', linewidth=.5, zorder=1)
        ax.set_title(f"{label}: {data['counts'][name]} specimens", fontsize=9)
    axes[-1].set_xlabel('Millions of years ago (older at left)')
    fig.suptitle('Separate, overlapping fossil compilations; no pooling', fontsize=10)
    return fig


def csv_text(rows):
    out = StringIO()
    keys = list(dict.fromkeys(k for row in rows for k in row))
    writer = csv.DictWriter(out, fieldnames=keys, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    return out.getvalue()


def outputs(data):
    from brain.minds_paper import table, tex
    from brain.minds_environment import check
    check(data)
    result = {}
    for name, fig in [('environment', environment_figure(data)), ('fossil-comparisons', comparison_figure(data))]:
        for ext in ['pdf', 'svg']:
            stream = BytesIO()
            fig.savefig(stream, format=ext, metadata={'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {'Date': None})
            result[f'minds-{name}.{ext}'] = stream.getvalue()
        plt.close(fig)
    macros = {'MwcFossilN': data['counts']['gingerich'], 'MwcFossilZoomN': data['displayed_consensus_n'],
              'MwcDuN': data['counts']['du'], 'MwcWillN': data['counts']['will'],
              'MwcClimateN': len(data['climate']), 'MwcClimateRecords': data['climate_source']['records_averaged']}
    events = {e['id']: e for e in data['events']}
    for name, event in [('Stasis', data['published_stasis']), ('Mpt', events['mpt']), ('Mis', events['mis34']), ('Bottleneck', events['bottleneck'])]:
        macros['Mwc'+name+'Older'] = f"{event['older_ma']:g}"
        macros['Mwc'+name+'Younger'] = f"{event['younger_ma']:g}"
    macros['MwcEffectiveSize'] = f"{events['bottleneck']['effective_population_size']:,}"
    macros['MwcImpactAge'] = f"{events['impact']['age_ma']:.2f}"
    result['minds-environment-numbers.tex'] = '\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in macros.items())+'\n'
    for name, rows in data['fossils'].items():
        result[f'minds-fossils-{name}.csv'] = csv_text(rows)
    bin_rows = [{'compilation': name, **r} for name, rows in data['bins'].items() for r in rows]
    result['minds-fossil-bins.csv'] = csv_text(bin_rows)
    result['minds-climate-lr04.csv'] = csv_text(data['climate'])
    result['minds-environment-events.csv'] = csv_text(data['events'])
    result['minds-environment-register.json'] = json.dumps({k: data[k] for k in ['climate_source','published_stasis','events','source_sha256','register_sha256','archived_models','limits']}, indent=2)+'\n'
    rows = []
    for e in data['events']:
        date = f"{e['age_ma']:.3f}" if e['kind']=='point' else f"{e['older_ma']:g}--{e['younger_ma']:g}"
        cite = r'\citep{'+','.join(e['citations'])+'}'
        rows.append([tex(e['label'])+' '+cite, date, tex(e['geography']), tex(e['evidence_status'])+'. '+tex(e['date_meaning'])+'. '+tex(e['interpretation_limit'])])
    result['minds-environment-chronology.tex'] = table(['Evidence','Ma','Scope','Status and interpretation limits'], rows, 'p{3.0cm}p{1.65cm}p{2.6cm}p{8cm}')
    source_rows = [{'path_or_url': k, 'sha256': v, 'version_note': 'Supplied archive; preserved bytes'} for k,v in data['source_sha256'].items()]
    source_rows += [{'path_or_url': data['climate_source']['url'], 'sha256': data['climate_source']['sha256'], 'version_note': data['climate_source']['version']},
                    {'path_or_url':'evidence/brain/environment-events.json','sha256':data['register_sha256'],'version_note':'Curated event register schema 1'}]
    for event in data['events']:
        if 'dating_source_audit' in event:
            audit = event['dating_source_audit']
            source_rows.append({'path_or_url':audit['url'], 'sha256':audit['sha256'],
                                'version_note':'Sieh et al. 2023 SI: Table S1 and tektite-dating discussion'})
    result['minds-environment-sources.csv'] = csv_text(source_rows)
    result['minds-environment-sources.tex'] = table(['Source (full SHA-256 in public CSV)','SHA-256 prefix','Version'],
        [[tex(r['path_or_url']),r['sha256'][:16],tex(r['version_note'])] for r in source_rows], 'p{8cm}p{3cm}p{4.3cm}')
    rows = []
    for index in range(1 + max(r['index'] for rows_ in data['bins'].values() for r in rows_)):
        row = {'younger_ma': index * .1, 'older_ma': (index+1) * .1}
        cells = []
        for name in ['gingerich', 'du', 'will']:
            other = next((b for b in data['bins'][name] if b['index']==index), None)
            cells += [str(other['n'] if other else 0), f"{other['geometric_mean_cm3']:.1f}" if other and other['n'] else '--']
        rows.append([f"{row['younger_ma']:.1f}--{row['older_ma']:.1f}", *cells])
    result['minds-environment-bins.tex'] = table(['Nominal Ma bin','G: n','G: cm$^3$','Du: n','Du: cm$^3$','Will: n','Will: cm$^3$'],rows,'lrrrrrr')
    models = data['archived_models']['models']
    result['minds-environment-models.tex'] = table(['Archived model','k','AICc','Delta AICc','BIC','Delta BIC','Breakpoints (Ma)'],
        [[tex(k),str(m['k']),f"{m['aicc']:.2f}",f"{m['daicc']:.2f}",f"{m['bic']:.2f}",f"{m['dbic']:.2f}",', '.join(f'{v:g}' for v in m['bps']) or '--'] for k,m in models.items()], 'lrrrrrl')
    rows = [[tex(u['meaning']),f"{u['age_ka']:g} $\\pm$ {u['plus_minus_ka']:g}",tex(u['convention'])] for u in events['impact']['reported_uncertainty']]
    result['minds-environment-impact-dates.tex'] = table(['Published date comparison','ka','Uncertainty convention / meaning'],rows,'p{4cm}p{3cm}p{8.3cm}')
    return result
