"""Read frozen benchmark metadata and fit fixed age/gender OLS; no remote writes."""
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

MANIFEST=Path('/tmp/hon_exp_scratch_v2/manifest.json')
MANIFEST_SHA256='6c5733bb6eb3dad0caf1476243c5b42744fba474840f5a8b29cea54003974205'
ROOT=Path('/home/ubuntu/HON_INDEX')
SEXES=('Female','Male')
SPLITS=('train','validation','test')


def natural_basis(age):
    age=np.asarray(age,float);knots=np.array([18,25,35,45,55,100],float)
    t=(age-18)/82;k=(knots-18)/82
    def d(j):return (np.maximum(t-k[j],0)**3-np.maximum(t-k[-1],0)**3)/(k[-1]-k[j])
    return np.column_stack([np.ones(len(age)),t,*[d(j)-d(len(k)-2) for j in range(len(k)-2)]])


def midrank(x):
    _,inverse,n=np.unique(x,return_inverse=True,return_counts=True)
    return ((n.cumsum()-.5*n)/len(x))[inverse]


def correlation(a,b):
    if np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


def metrics(y,pred):
    residual=pred-y;sse=float(np.dot(residual,residual));den=float(np.sum((y-y.mean())**2))
    return {'n':len(y),'beauty_mae':float(np.abs(residual).mean()),'beauty_rmse':float(np.sqrt(np.mean(residual**2))),'r2':1-sse/den if den>0 else None,'spearman_r':correlation(midrank(y),midrank(pred)),'pearson_r':correlation(y,pred),'bias':float(residual.mean()),'target_mean':float(y.mean()),'target_variance':float(np.var(y)),'prediction_range':[float(pred.min()),float(pred.max())]}


def fit_baselines(profiles):
    age=np.array([p['age'] for p in profiles],float);gender=np.array([p['gender'] for p in profiles]);split=np.array([p['split'] for p in profiles]);y=np.array([p['beauty_score'] for p in profiles],float)
    assert np.isfinite(age).all() and np.isfinite(y).all() and ((age>=18)&(age<=100)).all() and ((y>=0)&(y<=10)).all() and np.isin(gender,SEXES).all()
    masks={key:split==key for key in SPLITS};assert {k:int(v.sum()) for k,v in masks.items()}=={'train':8000,'validation':1000,'test':1000}
    train_mean=float(y[masks['train']].mean());pred_mean=np.full(len(y),train_mean);pred_gender=np.empty(len(y));pred_spline=np.empty(len(y));params={}
    basis=natural_basis(age)
    for sex in SEXES:
        rows=gender==sex;train=rows&masks['train'];mu=basis[train,1:].mean(axis=0);scale=basis[train,1:].std(axis=0);scale=np.where(scale>0,scale,1.)
        design=np.column_stack([np.ones(len(y)),(basis[:,1:]-mu)/scale]);coef=np.linalg.lstsq(design[train],y[train],rcond=None)[0]
        pred_gender[rows]=y[train].mean();pred_spline[rows]=design[rows]@coef
        lo,hi=float(age[train].min()),float(age[train].max());params[sex]={'train_n':int(train.sum()),'train_target_mean':float(y[train].mean()),'train_age_range':[lo,hi],'basis_means_train_only':mu.tolist(),'basis_scales_train_only':scale.tolist(),'coefficients':coef.tolist(),'test_age_extrapolation_n':int((rows&masks['test']&((age<lo)|(age>hi))).sum())}
    result={}
    for name,pred in [('train_mean',pred_mean),('gender_only',pred_gender),('age_gender_spline',pred_spline)]:
        result[name]={key+'_metrics':metrics(y[m],pred[m]) for key,m in masks.items()}
        test=masks['test'];result[name]['test_by_gender']={sex:metrics(y[test&(gender==sex)],pred[test&(gender==sex)]) for sex in SEXES}
        result[name]['test_by_age_band']={}
        for tag,lo,hi in [('18-24',18,24),('25-34',25,34),('35-44',35,44),('45-54',45,54),('55+',55,100)]:
            m=test&(age>=lo)&(age<=hi);result[name]['test_by_age_band'][tag]=metrics(y[m],pred[m]) if m.sum()>=20 else {'n':int(m.sum()),'suppressed_metrics':True}
    return result,params


def fingerprint(profiles,split):
    # One whole-cohort digest, never per-account hashes or identifiers.
    values=sorted((str(p['profile_id']),int(p['age']),str(p['gender']),float(p['beauty_score'])) for p in profiles if p['split']==split)
    return hashlib.sha256(json.dumps(values,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()


def main():
    assert platform.node().split('.')[0]=='xbabe3' and platform.machine()=='x86_64'
    before=MANIFEST.stat();raw=MANIFEST.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest!=MANIFEST_SHA256:raise ValueError('Frozen H-wave cohort manifest differs')
    manifest=json.loads(raw);profiles=manifest['profiles'];assert len(profiles)==10000
    byid={str(p['profile_id']):p for p in profiles};assert len(byid)==10000
    groups={}
    for p in profiles:groups.setdefault(str(p['duplicate_group']),set()).add(p['split'])
    assert all(len(v)==1 for v in groups.values())
    models,parameters=fit_baselines(profiles);target_variance=models['age_gender_spline']['test_metrics']['target_variance'];matching={};source_hashes={str(MANIFEST):digest}
    for wave in ['H1','H7','H8','H10','H20']:
        path=ROOT/'experiment'/wave/'metrics.json';sample=path.with_name('sample_predictions.csv')
        raw_metrics=path.read_bytes();source_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(raw_metrics).hexdigest();m=json.loads(raw_metrics)['profile_test'];variance=float(m['beauty_rmse']**2/(1-m['r2']))
        audit={'test_n_match':m['n']==1000,'target_variance_match':abs(variance-target_variance)<1e-8,'implied_test_target_variance':variance,'sampled_rows_checked':0,'sampled_metadata_matches':True,'saved_profile_test':{k:m[k] for k in ['n','beauty_mae','beauty_rmse','r2','spearman_r']},'full_prediction_pairing_available':False}
        if sample.exists():
            content=sample.read_bytes();source_hashes[str(sample.relative_to(ROOT))]=hashlib.sha256(content).hexdigest()
            for row in csv.DictReader(content.decode().splitlines()):
                p=byid.get(row['profile_id']);ok=p is not None and row['split']==p['split'] and row['recorded_gender']==p['gender'] and int(row['recorded_age'])==p['age'] and abs(float(row['beauty_score'])-float(p['beauty_score']))<.00011
                audit['sampled_rows_checked']+=1;audit['sampled_metadata_matches']&=ok
        assert audit['test_n_match'] and audit['target_variance_match'] and audit['sampled_metadata_matches'],(wave,audit)
        matching[wave]=audit
    for name in ['experiment/lib/dataset.py','experiment/lib/waves.py']:
        source_hashes[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    after=MANIFEST.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    eligibility={'profiles':len(profiles),'ages':[min(p['age'] for p in profiles),max(p['age'] for p in profiles)],'minimum_votes':min(p['votes'] for p in profiles),'finite_in_range_beauty_scores':all(np.isfinite(p['beauty_score']) and 0<=p['beauty_score']<=10 for p in profiles),'numeric_age_gender_missing_n':0,'duplicate_groups':len(groups),'duplicate_groups_disjoint_across_splits':True,'label_status':'Inherited frozen H-wave manifest, not freshly joined canonical label status; manifest numeric fields verified only','age_bins':'Same18-24,25-34,35-44,45-54,55+ as frozen H-wave metrics; benchmark observedages18-99 differ from main adultHON18-79'}
    result={'protocol':{'status':'exploratory supplement; specified after inspecting H-wave progression','model':'separate natural cubic age OLS in each recorded-gender group, equivalent to full age-basis by gender interaction','boundary_knots':[18,100],'interior_knots':[25,35,45,55],'training':'8000 fixed train profiles, equal profile weights, original beauty_score target; fit means/scales and coefficients on train only','validation':'1000fixed validation profiles diagnostic only, no tuning or model selection','test':'1000fixed test profiles; one prediction per profile; no repeated-image weighting','missingness':'No imputation needed: all frozen manifest ages, genders and beauty scores valid','outcome':'received supplied beauty score0-10, not an independent criterion of attractiveness or contribution','uncertainty':'Point metrics only; full matching frozen H-wave prediction vectors absent, so no paired CI or significance claim','r2':'1-SSE/SST relative to test-target mean, not squared Spearman'},'source_host':platform.node(),'architecture':platform.machine(),'snapshot_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'source_sha256':source_hashes,'manifest_stat_unchanged':True,'split_counts':{key:sum(p['split']==key for p in profiles) for key in SPLITS},'cohort_fingerprints':{key:fingerprint(profiles,key) for key in SPLITS},'eligibility':eligibility,'models':models,'fitted_parameters':parameters,'matching_frozen_wave_metrics':matching,'limitations':['All H-wave benchmark selection and old duplicate-audit coverage limits remain.','Matching supplied samples and targetvariance supports commonbenchmark metadata; full1000 frozen prediction vectors were not independently paired.','Age/gender baseline predicts rating variation associated with recorded context; differences from visual models do not identify causal appearance effects.','Algorithms and tuning differ across benchmark waves; this is a diagnostic comparison, not a controlled input ablation.','No images, embeddings, pickle files, tensors or model caches loaded; no remote file writes.']}
    json.dump(result,sys.stdout,allow_nan=False,sort_keys=True,indent=2)

if __name__=='__main__':main()
