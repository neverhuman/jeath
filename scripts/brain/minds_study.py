#!/usr/bin/env python3
"""Run the human assessment study; never consumes SAFE AI experimental evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.minds_analysis import hon_analysis,pov_analysis,pov_sensitivity_analysis,profile_weighted_analysis
from brain.minds_geography import prepare
from brain.minds_surveys import analyze as surveys
from brain.minds_choice import extend
from performance_choice import analyze as choice

ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'target/minds-we-choose'


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def remote_scan(payload):
    script=ROOT/'scripts/brain/minds_remote.py'
    scanner_source=script.read_text()
    program=('import base64,json,zlib\nGEO=json.loads(zlib.decompress(base64.b64decode('+repr(payload)+')))\n'+scanner_source).encode()
    command=['ssh','-o','BatchMode=yes','xbabe3','cd /home/ubuntu/HON_INDEX && PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/home/ubuntu/.cache/matplotlib .review-venv/bin/python -']
    result=subprocess.run(command,input=program,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')[-6000:])
    data=json.loads(result.stdout);data['provenance']['scanner_sha256']=hashlib.sha256(scanner_source.encode()).hexdigest();data['provenance']['stdin_program_sha256']=hashlib.sha256(program).hexdigest()
    return data


def reproduce_baseline(speed_data):
    scanner=ROOT/'scripts/performance_before_outcomes.py'
    result=subprocess.run(['ssh','-o','BatchMode=yes','xbabe3','cd /home/ubuntu/HON_INDEX && PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .review-venv/bin/python -'],input=scanner.read_bytes(),stdout=subprocess.PIPE,check=True)
    data=json.loads(result.stdout);data['speed_dating']=choice(speed_data)
    baseline=ROOT/'evidence/reference/performance-before-outcomes.json';reference=json.loads(baseline.read_text())
    checks={key:data[key]==reference[key] for key in ['hon','pov','speed_dating','inventory','input_sha256']}
    if not all(checks.values()):raise ValueError('Existing platform reference changed: '+repr(checks))
    return {'checks':checks,'reference_sha256':digest(baseline),'scanner_sha256':digest(scanner),'interpretation':'Exact reproduction of frozen legacy human platform evidence; this is separate from SAFE AI study evidence.'},data['speed_dating']


def audit_experiments(experiments):
    waves={r['wave']:r['metrics'] for r in experiments['waves'] if r['complete']};comparisons=[]
    for wave,current in waves.items():
        for key,reported in current.get('paired_bootstrap_comparisons',{}).items():
            import re
            match=re.fullmatch(r'vs_(H\d+)_(profile|image)_test',key)
            if not match or match.group(1) not in waves:continue
            ref,unit=match.groups();a=current[unit+'_test']['beauty_mae'];b=waves[ref][unit+'_test']['beauty_mae'];delta=a-b
            comparisons.append({'wave':wave,'reference':ref,'unit':unit,'current_mae':a,'reference_mae':b,'recomputed_delta_mae':delta,'stored_paired_delta_mae':reported['delta_mae'],'matching_metric_delta':abs(delta-reported['delta_mae'])<1e-8,'inference_allowed':abs(delta-reported['delta_mae'])<1e-8,'stored_interval_95':[reported['ci95_low'],reported['ci95_high']]})
    experiments['comparison_audit']=comparisons
    experiments['interpretation']={'target':'supplied beauty score, source0-10 scale, benchmark10000 profiles not fullHON','rank':'Spearman is rank association; never substitute1-rho^2 for residual variance or predictive R2','r2':'Use saved profile_test.r2 for target beauty-score predictive skill, not percentile-rank variance','configurations':'H1-H7 pixels at prediction time (H7 auxiliary age/gender training targets);H8 visual plus pre-UV landmarks;H9+ demographic/geographic/portfolio context and ensembles. Later models cannot be described as pixel-only.','exploratory':'Repeatedly inspected experiment progression and sequential choices; no untouched confirmatory final test','duplicate_scope':'Source dataset.py unions non-sampled bridge accounts in older766174-image audit plus selected image hashes; new3685764-image global audit components not available in finalization paths. Source-level grouping is supported, full-new-audit split safety is unverified.','delta_mismatch':'H11-H20 code reconstructs phase-I reference vectors; stored paired deltas that differ from subtraction of frozen metric files are flagged and their inference suppressed. Full matching frozen prediction vectors were not supplied in per-wave artifacts; sample predictions cannot repair this.'}


def legacy():
    folder=ROOT/'project/brain/tips/Performance_Before_Outcomes_v9/results';files={'gss':'gss.json','fossils':'e13_fossils.json','speed':'e17_speed.json','cultures':'e18_cultures.json'}
    return {**{key:json.loads((folder/name).read_text()) for key,name in files.items()},'sha256':{name:digest(folder/name) for name in files.values()},'status':'Archived human-study analyses; source algorithms and uncertainty retain original limitations. Not newly reanalysed national-survey designs. SAFE AI experiments are excluded.'}


def run(output,speed_data=None,remote_snapshot=None,demographic_frozen=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);inputs=CACHE/'inputs';inputs.mkdir(parents=True,exist_ok=True)
    speed_data=Path(speed_data) if speed_data else ROOT/'target/performance-before-outcomes/inputs/speed_dating.csv'
    # All remote operations sequential and read-only; no model/cache mutation or fleet allocation.
    payload,geography=prepare(inputs)
    if remote_snapshot:
        data=json.loads(Path(remote_snapshot).read_text())
        baseline=data['baseline_reproduction'];speed=data['speed_dating']
        if not all(baseline['checks'].values()):raise ValueError('Recorded baseline is not validated')
    else:
        baseline,speed=reproduce_baseline(speed_data)
        data=remote_scan(payload)
    data['execution']={'mode':'aggregate-replay' if remote_snapshot else 'fresh-source-scan',
                       'aggregate_snapshot_sha256':digest(Path(remote_snapshot)) if remote_snapshot else None,
                       'baseline':'reused recorded successful baseline verification' if remote_snapshot else 'fresh raw-source reproduction'}
    data['schema_version']=1;data['study']='minds-we-choose'
    data['specification']=json.loads((ROOT/'evidence/brain/specification.json').read_text())
    data['specification_amendments']=json.loads((ROOT/'evidence/brain/specification-amendments.json').read_text())
    data['hon']['age_analysis']=hon_analysis(data['hon']);data['pov']['country_balanced']=pov_analysis(data['pov']);data['pov']['country_balanced_sensitivities']=pov_sensitivity_analysis(data['pov'])
    data['hon']['age_analysis']['profile_size_weighted']=profile_weighted_analysis(data['hon']['age_analysis'])
    data['hon']['profile_weighted_baseline']=json.loads((ROOT/'evidence/reference/performance-before-outcomes.json').read_text())['hon']
    data['surveys']=surveys(inputs);data['speed_dating']=extend(speed,speed_data);data['baseline_reproduction']=baseline
    from brain.minds_gss import analyze as gss_design
    data['gss_design']=gss_design(ROOT,inputs)
    data['geography']=geography;data['legacy_v9']=legacy();audit_experiments(data['experiments'])
    from brain.minds_demographic import analyze as demographic_baseline
    data['experiments']['age_gender_baseline']=demographic_baseline(ROOT,demographic_frozen)
    from brain.minds_environment import analyze as environment
    data['fossil_environment']=environment(inputs)
    data['analysis_sha256']={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'scripts/brain').glob('minds_*.py')) if p.name != 'minds_paper.py'}
    data['analysis_sha256']['scripts/brain/validate_minds.py']=digest(ROOT/'scripts/brain/validate_minds.py')
    for name in ['specification.json','specification-amendments.json']:
        data['analysis_sha256']['evidence/brain/'+name]=digest(ROOT/'evidence/brain'/name)
    data['analysis_sha256']['evidence/brain/environment-events.json']=digest(ROOT/'evidence/brain/environment-events.json')
    data['provenance']['analysis_run_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    data['source_register']={'platform_exports':data['provenance'],'public_surveys':data['surveys']['sources'],'geography':geography['sources'],'archived_human_results':data['legacy_v9']['sha256'],'measurement_register':{'HON':'canonical profile; received ratings/attention, recorded residence and gender; collection period unknown in export','POV':'account-unique source record; self-reported intentions/attributes, derived residence; period unknown','Kowal':'respondent in convenience survey; current-residence country; separate six-trait fixed30-point budget','Whyte':'respondent in deposited Australian sexual-attraction survey; nine0-100importance ratings, not choices','Columbia':'directed decision per four-minute meeting; chooser vs target recorded gender distinguished','GSS':'U.S. respondent retrospective reported children and relationship history; archivedcase-study; no new design-aware uncertainty','image_models':'10000profile benchmark,20000images; separate image and profile targets; no inference to globalHONpopulation'}}
    data['limitations']=['Platform collection periods unavailable in supplied exports; cross-sectional age comparisons combine selection and cohort effects.','Geography is derived residence, not nationality; contradictory/ambiguous geography remains unresolved. Country-balanced estimates are supported convenience samples, not world-population estimates.','Walter0women/1men mapping reconciles exact primary final sample counts; original numeric questionnaire key still unavailable. Raw codes retained; mapping never inferred from outcomes.','POV relationshipage_id exact original questionnaire wording not recovered; retain source categories without asserting it is longest prior relationship.','POV attribute standardization is associational, exact common strata only; sparse support and unmeasured selection remain.','GSS new uncertainty is a released-design sensitivity under independent survey-year treatment; archived full-domain baseline remains separate.','Prediction audit lacks full matching frozen vectors and final all-image duplicate components; mismatching paired intervals not promoted.','Current expanded run does not implement every prospective sensitivity requested in planning; missing analyses are identified in evidence/minds/analysis-coverage.json.']
    data['limitations']=[s.replace('evidence/minds/analysis-coverage.json','evidence/brain/analysis-coverage.json') for s in data['limitations']]
    data['limitations'].extend(data['experiments']['age_gender_baseline']['limitations'])
    from brain.validate_minds import check
    check(data)
    destination=output/'minds-we-choose.json';destination.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print('wrote '+str(destination),flush=True)
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--speed-data',type=Path);p.add_argument('--remote-snapshot',type=Path,help='Replay a validated full-study aggregate snapshot without remote scans')
    p.add_argument('--demographic-frozen',type=Path,help='Replay the separate metadata-only benchmark baseline without a remote call')
    args=p.parse_args();run(args.output,args.speed_data,args.remote_snapshot,args.demographic_frozen)
