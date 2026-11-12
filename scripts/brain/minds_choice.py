"""Additional accounting and session-deletion checks for the frozen Columbia cohort."""
import numpy as np
from performance_choice import ATTRIBUTES, load_data


def extend(result,path):
    d=load_data(path);wave,sex,chooser,partner,yes=d[:,:5].T;own=d[:,5:11].copy();own[(own<1)|(own>10)]=np.nan
    valid=np.isin(yes,[0,1]);panel=np.full_like(own,np.nan)
    for w in np.unique(wave):
        for p in np.unique(partner[wave==w]):
            ii=np.flatnonzero((wave==w)&(partner==p));v=own[ii];nn=np.isfinite(v).sum(axis=0)-np.isfinite(v);ss=np.nansum(v,axis=0)-np.nan_to_num(v)
            panel[ii]=np.divide(ss,nn,out=np.full_like(ss,np.nan),where=nn>=3)
    directed={(int(w),int(i),int(p)):int(y) for w,i,p,y in zip(wave[valid],chooser[valid],partner[valid],yes[valid]) if np.isfinite(p)}
    pairs={tuple([w,min(i,p),max(i,p)]) for w,i,p in directed if (w,p,i) in directed}
    reciprocals={'observed_directed_keys':len(directed),'reciprocal_pairs':len(pairs),'mutual_yes_pairs':sum(directed[(w,a,b)]==directed[(w,b,a)]==1 for w,a,b in pairs),'missing_partner_rows':int((valid&~np.isfinite(partner)).sum()),'duplicate_directed_rows':int((valid&np.isfinite(partner)).sum())-len(directed)}
    common=np.isfinite(own).all(axis=1)&np.isfinite(panel).all(axis=1)&valid
    sessions=np.unique(wave)
    models={}
    for method,ratings in [('own',own),('panel',panel)]:
        models[method]={}
        for selection_name,mask in [('method_complete',np.isfinite(ratings).all(axis=1)&valid),('common_complete',common)]:
            scale=ratings[mask].std(axis=0);x=ratings/scale;bysex={}
            matrices={}
            for s,sname in enumerate(['women','men']):
                xx=np.zeros((len(sessions),6,6));xy=np.zeros((len(sessions),6));used=0;people=0
                for person in np.unique(chooser[mask&(sex==s)]):
                    ii=np.flatnonzero(mask&(sex==s)&(chooser==person))
                    if len(ii)<2:continue
                    event=np.searchsorted(sessions,wave[ii[0]]);xc=x[ii]-x[ii].mean(axis=0);yc=yes[ii]-yes[ii].mean();xx[event]+=xc.T@xc;xy[event]+=xc.T@yc;used+=len(ii);people+=1
                coef=np.linalg.solve(xx.sum(axis=0),xy.sum(axis=0));loo=np.stack([np.linalg.solve(xx.sum(axis=0)-xx[j],xy.sum(axis=0)-xy[j]) for j in range(len(sessions))])
                bysex[sname]={'n_decisions':used,'n_choosers':people,'coefficients_pp':dict(zip(ATTRIBUTES,(100*coef).tolist())),'leave_one_session_out_pp':{key:(100*loo[:,j]).tolist() for j,key in enumerate(ATTRIBUTES)},'leave_one_session_out_range_pp':{key:[float(100*loo[:,j].min()),float(100*loo[:,j].max())] for j,key in enumerate(ATTRIBUTES)}}
                matrices[sname]=loo
            bysex['men_minus_women_session_deletion_pp']={key:(100*(matrices['men'][:,j]-matrices['women'][:,j])).tolist() for j,key in enumerate(ATTRIBUTES)}
            models[method][selection_name]=bysex
    result['additional_accounting']=reciprocals;result['common_complete_case_decisions']=int(common.sum());result['session_deletion_models']=models
    result['agreement_interpretation']='Agreement estimates in archived v9 concern consistency of impressions and do not validate intelligence, commitment or future performance against an independent criterion.'
    return result
