"""GSS income/parenthood decomposition with a released-design sensitivity."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import numpy as np

SOURCE_COMMIT='9ed92cf9e065396fedb7cc60a4590ddb25b64179'
SOURCE_SHA256='38712b98fe23598856f757c096e5d4be908f6a9f644f695fce62ba0db4e4627f'
SOURCE_URL=f'https://raw.githubusercontent.com/kjhealy/gssr/{SOURCE_COMMIT}/data/gss_all.rda'
SEED=20261111
REPLICATES=2000


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def load(root,inputs):
    path=inputs/'gss_all.rda'
    if not path.exists():
        with urllib.request.urlopen(SOURCE_URL,timeout=180) as response,path.open('wb') as f:
            import shutil;shutil.copyfileobj(response,f)
    if digest(path)!=SOURCE_SHA256:raise ValueError('GSS source hash differs')
    sys.path.insert(0,str(root/'project/brain/tips/Performance_Before_Outcomes_v9/code'))
    import rda_reader
    wanted='year id age sex cohort childs educ realrinc wtssps wtssnrps vstrat vpsu sample'.split()
    data,labels,missing=rda_reader.read_columns(path,wanted)
    if missing:raise ValueError('Missing GSS variables '+repr(missing))
    return data,{key:{str(v):label for v,label in value.items() if np.isfinite(v)} for key,value in labels.items()}


def sufficient(x,c,w,group,n_groups):
    vectors=[w,w*x,w*x*x,w*c,w*x*c,w*(c>0),w*x*(c>0)]
    return np.stack([np.bincount(group,weights=y,minlength=n_groups) for y in vectors],axis=1)


def decomposition_from_sums(sums):
    """Seven weighted moments; last two axes are decade cell and moments."""
    w,wx,wx2,wc,wxc,wp,wxp=np.moveaxis(sums,-1,0)
    mx=wx/w;sd=np.sqrt(np.maximum(wx2/w-mx*mx,0));total=(wxc/wc-mx)/sd;access=(wxp/wp-mx)/sd;number=total-access
    weights=w/w.sum(axis=-1,keepdims=True)
    return np.stack([(total*weights).sum(axis=-1),(access*weights).sum(axis=-1),(number*weights).sum(axis=-1)],axis=-1)


def design_multiplier(data,weight):
    valid=np.isfinite(data['year'])&np.isfinite(data['vstrat'])&np.isfinite(data['vpsu'])&(data['vstrat']>0)&(data['vpsu']>0)&np.isfinite(weight)&(weight>0)
    idx=np.flatnonzero(valid);keys=np.stack([data[k][idx] for k in ['year','vstrat','vpsu']],axis=1)
    psus,inverse=np.unique(keys,axis=0,return_inverse=True);strata,stratum=np.unique(psus[:,:2],axis=0,return_inverse=True)
    counts=np.bincount(stratum);supported=counts>=2;valid[idx[~supported[stratum[inverse]]]]=False
    # Build bootstrap design from the full supported survey, before subsetting the age/income domains.
    rng=np.random.default_rng(SEED);multiplier=np.zeros((REPLICATES,len(psus)),np.float32)
    for h in np.flatnonzero(supported):
        ii=np.flatnonzero(stratum==h);m=len(ii);draws=rng.integers(m,size=(REPLICATES,m-1))
        rows=np.repeat(np.arange(REPLICATES),m-1);selected=ii[draws.ravel()]
        np.add.at(multiplier,(rows,selected),m/(m-1))
    group=np.full(len(weight),-1,int);group[idx]=inverse
    info={'full_source_n':len(weight),'design_positive_weight_records':len(idx),'supported_records':int(valid.sum()),'psus':int(supported[stratum].sum()),'strata':int(supported.sum()),'singleton_strata_excluded':int((~supported).sum()),'stratum_psu_counts':{str(int(m)):int((counts[supported]==m).sum()) for m in np.unique(counts[supported])},'stratification':'surveyyear by releasedVSTRAT; PSU is releasedVPSU nested inside each year-stratum','bootstrap':'Rao-Wu style rescaled cluster bootstrap: samplem_h-1 PSUs with replacement per supported stratum; multiply original caseweights bym_h/(m_h-1) times PSU multiplicity','replicates':REPLICATES,'seed':SEED,'domain_handling':'PSU support determined on full positive-weight survey, before restricting age,gender,children,income; zero domain contributions retained','limitations':'Sensitivity under independent survey-year treatment. Does not model possible cross-year PSU dependence, nonresponse not captured by weights, topcoded outcome uncertainty or unmeasured confounding. Singleton strata excluded rather than imputed.'}
    return valid,group,multiplier,info


def point(data,base,x,weight,sexval,group=None,n_groups=1,multiplier=None):
    domain=base&np.isfinite(x)&(data['sex']==sexval);decade=np.floor(data['cohort']/10);cells=[];moments=[];used=np.zeros(len(x),bool)
    for d in np.unique(decade[domain]):
        mask=domain&(decade==d);ii=np.flatnonzero(mask)
        if len(ii)<60 or int((data['childs'][ii]>0).sum())<5 or int((data['childs'][ii]==0).sum())<1:continue
        used[ii]=True;cells.append({'birth_decade':int(d*10),'n':len(ii)})
        groups=np.zeros(len(ii),int) if group is None else group[ii]
        moments.append(sufficient(x[ii],data['childs'][ii],weight[ii],groups,n_groups))
    statistics=np.stack(moments,axis=1)
    estimate=decomposition_from_sums(statistics.sum(axis=0));result={'domain_n':int(domain.sum()),'used_n':int(used.sum()),'age_range':[45,64],'cells':cells,'topcoded_children_n':int((used&(data['childs']==8)).sum()),'estimate':dict(zip(['total','access','number'],estimate.tolist()))}
    if multiplier is not None:
        # BxPSU times PSUx(cell*moment) keeps the replication local and bounded.
        rep=(multiplier@statistics.reshape(n_groups,-1)).reshape(REPLICATES,len(cells),7)
        values=decomposition_from_sums(rep);valid=np.isfinite(values).all(axis=1)
        if not valid.all():raise ValueError('Degenerate GSS replicates; fixed cell set not identified')
        result['interval_95']={key:np.quantile(values[:,j],[.025,.975]).tolist() for j,key in enumerate(['total','access','number'])}
        result['identity_max_abs_error']=float(np.abs(values[:,0]-values[:,1]-values[:,2]).max())
        return result,values
    return result,None


def analyze(root,inputs):
    root=Path(root);inputs=Path(inputs);data,labels=load(root,inputs)
    wt=data['wtssps'];base=np.isfinite(data['age'])&np.isfinite(data['sex'])&np.isfinite(data['cohort'])&np.isfinite(data['childs'])&np.isfinite(wt)&(wt>0)&(data['age']>=45)&(data['age']<=79)&(data['cohort']>=1900)&(data['cohort']<=1979)
    assert int(base.sum())==34092
    with np.errstate(divide='ignore',invalid='ignore'):x=np.log(data['realrinc'])
    x[data['age']>64]=np.nan
    old=json.loads((root/'project/brain/tips/Performance_Before_Outcomes_v9/results/gss.json').read_text())
    baseline={}
    for name,sv in [('men',1),('women',2)]:
        result,_=point(data,base,x,wt,sv);reference=old['decomposition']['income'][name]
        for key,value in result['estimate'].items():
            if abs(value-reference[key][0])>1e-10:raise ValueError('GSS legacy point mismatch '+name+key)
        baseline[name]=result
    design,group,multiplier,design_info=design_multiplier(data,wt);design_base=base&design;results={};boot={}
    for name,sv in [('men',1),('women',2)]:
        result,values=point(data,design_base,x,wt,sv,group,multiplier.shape[1],multiplier);results[name]=result;boot[name]=values
        result['share_of_original_income_domain']=result['domain_n']/baseline[name]['domain_n']
    difference=boot['men']-boot['women'];results['men_minus_women']={'estimate':{key:results['men']['estimate'][key]-results['women']['estimate'][key] for key in ['total','access','number']},'interval_95':{key:np.quantile(difference[:,j],[.025,.975]).tolist() for j,key in enumerate(['total','access','number'])},'bootstrap':'same stratum/PSU draws for both gender domains'}
    # Alternative nonresponse-poststratification weights where available; retain WTSSPS elsewhere.
    alt=np.where((data['year']>=2004)&np.isfinite(data['wtssnrps'])&(data['wtssnrps']>0),data['wtssnrps'],wt)
    sensitivity={name:point(data,design_base,x,alt,sv)[0] for name,sv in [('men',1),('women',2)]}
    return {'source':{'url':SOURCE_URL,'git_commit':SOURCE_COMMIT,'sha256':SOURCE_SHA256,'bytes':(inputs/'gss_all.rda').stat().st_size,'rows':len(wt),'years':[int(data['year'].min()),int(data['year'].max())],'reader_sha256':digest(root/'project/brain/tips/Performance_Before_Outcomes_v9/code/rda_reader.py')},'labels':labels,'n_base':int(base.sum()),'base_with_supported_design_n':int(design_base.sum()),'legacy_baseline_reproduced':True,'baseline_income':baseline,'design':design_info,'income':results,'weight_sensitivity':{'definition':'WTSSNRPS when observed and positive from2004 onward, WTSSPS otherwise; point estimates on same design-supported domain','points':sensitivity},'measurement':'logrespondent income REALRINC; within sex by birth-decade weighted traitstandardization; reportedCHILDS outcome with8or more scored8; ages45-64 for income; no claim of completed reproduction or genetic selection coefficient','topcoding':'CHILDS code8 means8or more; differences involving high-family-size tails are conservative representations, not recovered uncensored births','weight_policy':'Main maintains archivedWTSSPS for direct comparison; currentNORC recommendsWTSSNRPS where available; this alternative-weight point sensitivity is kept distinct from main fixed-WTSSPS estimates. Weighting does not make retrospective income associations causal.','documentation':[{'url':'https://gss.norc.org/content/dam/gss/get-documentation/pdf/reports/methodological-reports/MR135%20poststratified%20weights.pdf','supports':'poststratified weights and svysetstrataVSTRAT PSU VPSU'},{'url':'https://gss.norc.org/content/dam/gss/get-documentation/pdf/codebook/GSS%202024%20Codebook%20R3.pdf','supports':'currentweight guidance; rawvariable interpretation'}]}

if __name__=='__main__':
    root=Path.cwd();result=analyze(root,root/'target/minds-we-choose/inputs');(root/'target/minds-we-choose/gss-design.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result['income'],indent=2));print(result['design'])
