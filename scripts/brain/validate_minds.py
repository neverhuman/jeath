#!/usr/bin/env python3
"""Reconcile aggregate denominators, support, weights and principal country estimates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def close(a,b):assert abs(a-b)<1e-8,(a,b)


def check(data):
    assert data['study']=='minds-we-choose' and data['schema_version']==1
    if 'fossil_environment' in data:
        from brain.minds_environment import check as check_environment
        check_environment(data['fossil_environment'])
    assert data['provenance']['source_host'].split('.')[0]=='xbabe3'
    assert data['provenance']['architecture']=='x86_64' and data['provenance']['input_stat_unchanged']
    h=data['hon'];p=data['pov'];s=data['surveys'];a=h['age_analysis']
    assert h['flow']['all']==6727811 and h['flow']['primary']==6677772
    assert h['flow']['all']>=h['flow']['age_gender']>=h['flow']['valid_unconflicted']>=h['flow']['primary']==h['flow']['score_primary']
    assert sum(r['eligible'] for r in h['coverage'])==h['flow']['primary']
    assert sum(h['image_country_counts'].values())==sum(r['inventoried'] for r in h['coverage'])
    assert a['countries']==len(a['country_names'])==len(set(a['country_names']))==len(a['country_contrasts'])
    for name in a['country_names']:
        for sex in ('Female','Male'):
            cell=h['thresholds']['20'][name][sex]
            assert all(cell['bands'][b]['n']>=100 for b in ['18-29','30-44','45-64'])
            assert sum(r['n'] for r in cell['exact_age'])==sum(cell['bands'][b]['n'] for b in ['18-29','30-44','45-64'])
            for key in ['mean_percentile','mean_score','mean_like_rate']:
                values=[r[sex]['contrasts'][key]['young_minus_older'] for r in a['country_contrasts']]
                close(float(np.mean(values)),a['contrasts'][sex][key]['mean'])
    for sex in ('Female','Male'):
        atlas=h['thresholds']['20']
        for name,rows in atlas.items():
            if sex not in rows:continue
            c=rows[sex]['all'];assert 0<=c['mean_percentile']<=1;assert abs(c['mean_percentile']-.5)<1e-6  # remote midranks stored asfloat32
            assert c['inventoried_n']<=c['n'] and c['n']>=100
    close(a['women_minus_men_decline']['mean'],a['contrasts']['Female']['mean_percentile']['mean']-a['contrasts']['Male']['mean_percentile']['mean'])
    if 'profile_weighted_baseline' in h:
        assert sum(h['profile_weighted_baseline']['ratings']['votes_20']['all_ages'][sex]['all']['n'] for sex in ('Female','Male'))==h['flow']['primary']
        for sex in ('Female','Male'):
            assert a['profile_size_weighted']['by_gender'][sex]['profiles']==sum(r[sex]['n_age18_64'] for r in a['country_contrasts'])
    assert p['records']==1208771 and p['eligible']==1137309
    assert sum(p['mapping_status_all'].values())==p['records']
    assert p['derived_country_profiles']+p['mapping_status_all']['conflict']+p['mapping_status_all']['unresolved']==p['records']
    assert p['derived_country_profiles']==sum(p['mapping_status_all'][k] for k in ['agreement','coordinates_only','state_only'])
    assert p['coordinate_assigned_unconflicted_profiles']==p['mapping_status_all']['agreement']+p['mapping_status_all']['coordinates_only']
    assert sum(p['height_audit'].values())==p['records']
    for setname,summary in p['country_balanced'].items():
        names=summary['countries'];assert len(names)==len(set(names))==summary['country_count']
        if not names:continue
        for band,bysex in summary['by_band'].items():
            for sex in ('Female','Male'):
                assert bysex[sex]['n']>=100*len(names)
            for key,gap in bysex['women_minus_men_pp'].items():
                close(gap['mean'],100*(bysex['Female']['outcomes'][key]['mean']-bysex['Male']['outcomes'][key]['mean']))
    for scenario,summary in p.get('country_balanced_sensitivities',{}).items():
        assert summary['country_count']==len(summary['countries'])
        raw=p['headline_sensitivities'][scenario]
        for name in summary['countries']:
            assert all(raw[name][band][sex]['n']>=100 for band in ['25-29','45-49'] for sex in ['Female','Male'])
        for band,rows in summary.get('by_band',{}).items():
            for key,gap in rows['women_minus_men_pp'].items():
                close(gap['mean'],100*(rows['Female']['outcomes'][key]['mean']-rows['Male']['outcomes'][key]['mean']))
    if 'gss_design' in data:
        g=data['gss_design'];assert g['legacy_baseline_reproduced'] and g['n_base']==34092
        assert 0<g['base_with_supported_design_n']<=g['n_base']
        assert sum(int(m)*n for m,n in g['design']['stratum_psu_counts'].items())==g['design']['psus']
        assert g['design']['replicates']==2000
        for sex in ['men','women']:
            row=g['income'][sex];assert row['used_n']==sum(c['n'] for c in row['cells'])
            close(row['estimate']['total'],row['estimate']['access']+row['estimate']['number'])
            assert row['identity_max_abs_error']<1e-12
            assert 0<=row['topcoded_children_n']<=row['used_n']
            assert 0<row['share_of_original_income_domain']<=1
    k=s['kowal'];assert k['flow']==dict(source=117293,age_18_90=107539,gender_man_woman=105963,heterosexual=86744,complete_six=67817,valid_range=67817,sum_30=67743,country_observed=67615,country_min100_each_gender=62185)
    assert len(k['countries'])==53 and len(k['age_standardized_countries'])==32
    assert k['gender_counts']=={'men':23059,'women':44684}
    for sex in ('women','men'):
        close(sum(v['mean'] for v in k['pooled'][sex].values()),30)
        close(sum(v['mean'] for v in k['equal_country'][sex].values()),30)
        for item in k['items']:close(k['equal_country'][sex][item]['mean'],float(np.mean([r['by_gender'][sex][item]['mean'] for r in k['countries'].values()])))
    w=s['whyte'];assert w['records']==7060 and w['unavailable_difference']==265 and w['age_range']==[18,64] and len(w['items'])==9
    for sex in ('women','men'):
        overall=w['analyses']['all'][sex]['18-64'];assert overall['n']==sum(w['analyses']['all'][sex][b]['n'] for b in ['18-29','30-39','40-49','50-64'])
        close(sum(v['mean'] for v in w['analyses']['centered'][sex]['18-64']['items'].values()),0)
        for key,item in [('trust_minus_intelligence','intelligence_import'),('trust_minus_attractiveness','attractive_import')]:close(overall['paired_contrasts'][key]['mean'],overall['items']['trust_import']['mean']-overall['items'][item]['mean'])
    assert len(s['eastwick']['attributes'])==35 and len({r['attribute'] for r in s['eastwick']['attributes']})==35
    speed=data['speed_dating'];assert speed['records']==8378 and speed['participants']==551 and speed['sessions']==21
    assert speed['additional_accounting']['reciprocal_pairs']*2<=speed['counts']['valid_binary_decisions']
    assert all(v for v in data['baseline_reproduction']['checks'].values())
    assert len(data['experiments']['waves'])==40
    if 'age_gender_baseline' in data['experiments']:
        from brain.minds_demographic import check as check_baseline
        baseline=data['experiments']['age_gender_baseline'];check_baseline(baseline)
        waves={r['wave']:r for r in data['experiments']['waves'] if r['complete']}
        for wave,row in baseline['matching_frozen_wave_metrics'].items():
            assert baseline['source_sha256']['experiment/'+wave+'/metrics.json']==waves[wave]['metrics_sha256']
            for key,value in row['saved_profile_test'].items():
                close(value,waves[wave]['metrics']['profile_test'][key])
    for r in data['experiments']['comparison_audit']:
        close(r['recomputed_delta_mae'],r['current_mae']-r['reference_mae'])
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('path',type=Path);args=parser.parse_args();path=args.path/'minds-we-choose.json' if args.path.is_dir() else args.path
    check(json.loads(path.read_text()));print('minds-we-choose evidence validated')

if __name__=='__main__':main()
