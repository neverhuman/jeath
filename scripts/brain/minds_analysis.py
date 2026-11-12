"""Local fitting and paired country resampling of aggregate platform evidence."""
import numpy as np

SEXES=('Female','Male')
SEED=20261111
BOOTSTRAPS=2000


def natural_basis(ages,knots=(18,25,35,45,55,64)):
    """Natural cubic regression basis; linear outside the two boundary knots."""
    ages=np.asarray(ages,float);t=(ages-knots[0])/(knots[-1]-knots[0]);k=(np.asarray(knots,float)-knots[0])/(knots[-1]-knots[0])
    def d(j):return (np.maximum(t-k[j],0)**3-np.maximum(t-k[-1],0)**3)/(k[-1]-k[j])
    return np.column_stack([np.ones(len(t)),t,*[d(j)-d(len(k)-2) for j in range(len(k)-2)]])


def summarize(values,draws=None):
    x=np.asarray(values,float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('Finite, nonempty country vector required')
    if draws is None:draws=np.random.default_rng(SEED).integers(len(x),size=(BOOTSTRAPS,len(x)))
    boot=x[draws].mean(axis=1)
    return {'countries':len(x),'mean':float(x.mean()),'interval_95':np.quantile(boot,[.025,.975]).tolist(),'between_country_quantiles':np.quantile(x,[0,.1,.25,.5,.75,.9,1]).tolist(),'positive':int((x>0).sum()),'negative':int((x<0).sum()),'zero':int((x==0).sum())}


def hon_analysis(hon):
    names=hon['common_support_countries'];data=hon['thresholds']['20'];ages=np.arange(18,65);basis=natural_basis(ages)
    rng=np.random.default_rng(SEED);draws=rng.integers(len(names),size=(BOOTSTRAPS,len(names)))
    outcomes=['mean_percentile','mean_score','mean_like_rate','mean_global_gender_percentile','mean_pooled_percentile']
    fits={s:{key:[] for key in outcomes} for s in SEXES};country_contrasts=[]
    for name in names:
        row={'country':name}
        for sex in SEXES:
            rows=data[name][sex]['exact_age'];assert [r['age'] for r in rows]==ages.tolist()
            contrasts={};counts={}
            for key in outcomes:
                weights=np.array([r['n'] if key=='mean_like_rate' else r['score_n'] for r in rows],float)
                y=np.array([r[key] if r[key] is not None else np.nan for r in rows],float);ok=np.isfinite(y)&(weights>0)
                coef=np.linalg.lstsq(basis[ok]*np.sqrt(weights[ok,None]),y[ok]*np.sqrt(weights[ok]),rcond=None)[0];pred=basis@coef;fits[sex][key].append(pred)
                young=float(pred[(ages>=25)&(ages<=29)].mean());older=float(pred[(ages>=45)&(ages<=49)].mean())
                contrasts[key]={'young':young,'older':older,'young_minus_older':young-older}
            row[sex]={'n_age18_64':sum(r['n'] for r in rows),'contrasts':contrasts}
        row['women_minus_men_decline']=row['Female']['contrasts']['mean_percentile']['young_minus_older']-row['Male']['contrasts']['mean_percentile']['young_minus_older']
        country_contrasts.append(row)
    curves={};contrasts={}
    for sex in SEXES:
        curves[sex]={};contrasts[sex]={}
        for key in outcomes:
            matrix=np.asarray(fits[sex][key]);boot=matrix[draws].mean(axis=1)
            curves[sex][key]={'mean':matrix.mean(axis=0).tolist(),'interval_95':np.quantile(boot,[.025,.975],axis=0).tolist(),'between_country_10_90':np.quantile(matrix,[.1,.9],axis=0).tolist()}
            contrasts[sex][key]=summarize([r[sex]['contrasts'][key]['young_minus_older'] for r in country_contrasts],draws)
    gap=summarize([r['women_minus_men_decline'] for r in country_contrasts],draws)
    sensitivity={}
    for threshold,atlas in hon['thresholds'].items():
        selected=[name for name in names if all(sex in atlas.get(name,{}) and all(b in atlas[name][sex]['bands'] for b in ['25-29','45-49']) for sex in SEXES)]
        sensitivity[threshold]={'countries':selected,'estimand':'observed25-29 minus45-49 means, countries from principal fixed set with>=100 in both bands/genders; percentile reference recalculated at threshold'}
        for sex in SEXES:
            sensitivity[threshold][sex]={key:summarize([atlas[n][sex]['bands']['25-29'][key]-atlas[n][sex]['bands']['45-49'][key] for n in selected]) for key in ['mean_percentile','mean_score','mean_like_rate']}
    subsets={}
    for label,subset in [('excluding_US',[r for r in country_contrasts if r['country']!='United States of America']),('US_only',[r for r in country_contrasts if r['country']=='United States of America'])]:
        subsets[label]={'countries':[r['country'] for r in subset],**{sex:summarize([r[sex]['contrasts']['mean_percentile']['young_minus_older'] for r in subset]) for sex in SEXES}}
    return {'ages':ages.tolist(),'knots':[25,35,45,55],'boundary_knots':[18,64],'countries':len(names),'country_names':names,'profile_n_age18_64':sum(r[s]['n_age18_64'] for r in country_contrasts for s in SEXES),'curves':curves,'contrasts':contrasts,'women_minus_men_decline':gap,'country_contrasts':country_contrasts,'threshold_sensitivity':sensitivity,'geographic_sensitivity':subsets,'uncertainty':'2000 paired country bootstrap draws, seed20261111; pointwise intervals conditional on included convenience samples; separate between-country10th-90th bands. Within-country sampling error not resampled.','fit':'natural cubic least squares from exact-age sufficient means weighted by eligible score counts (profile counts for like rate), no extrapolation beyond18-64','contrast_definition':'equal weight over exact ages25,26,27,28,29 minus45,46,47,48,49, then equal weight per included country'}


def pov_analysis(pov):
    outcomes=['parenthood_share','want_children_share','serious_intent_share','for_long_term_share'];result={}
    for setname,names in pov['common_support_countries'].items():
        tag={'all':None,'childless':'childless','coordinate_all':'coordinate','coordinate_childless':'coordinate_childless'}[setname]
        def cell(name,sex,band):
            row=pov['countries'][name][sex][band];return row if tag is None else row[tag]
        if not names:result[setname]={'countries':[],'country_count':0};continue
        draws=np.random.default_rng(SEED).integers(len(names),size=(BOOTSTRAPS,len(names)))
        row={'countries':names,'country_count':len(names),'by_band':{},'country_contrasts':[]}
        for band in ['25-29','45-49']:
            row['by_band'][band]={}
            for sex in SEXES:
                row['by_band'][band][sex]={'n':sum(cell(n,sex,band)['n'] for n in names),'outcomes':{key:summarize([cell(n,sex,band)[key] for n in names],draws) for key in outcomes}}
            row['by_band'][band]['women_minus_men_pp']={key:summarize([100*(cell(n,'Female',band)[key]-cell(n,'Male',band)[key]) for n in names],draws) for key in outcomes}
        for name in names:
            row['country_contrasts'].append({'country':name,'women_minus_men_pp':{band:{key:100*(cell(name,'Female',band)[key]-cell(name,'Male',band)[key]) for key in outcomes} for band in ['25-29','45-49']}})
        row['older_minus_young']={sex:{key:summarize([cell(n,sex,'45-49')[key]-cell(n,sex,'25-29')[key] for n in names],draws) for key in outcomes} for sex in SEXES}
        result[setname]=row
    return result


def profile_weighted_analysis(age_analysis):
    """Fixed country-size sensitivity; weights stay constant across the age contrast."""
    records=age_analysis['country_contrasts'];n=len(records)
    draws=np.random.default_rng(SEED).integers(n,size=(BOOTSTRAPS,n))
    result={'definition':'Country fitted contrasts weighted by each gender-specific country profile count ages18-64; country weights held fixed across compared ages, same country set as principal; differs from pooled-within-age country composition',
            'countries':n,'bootstrap':'2000 country resamples, each retaining its original profile count; conditional convenience-sample sensitivity','by_gender':{}}
    for sex in SEXES:
        weights=np.array([r[sex]['n_age18_64'] for r in records],float)
        outcomes={}
        for key in records[0][sex]['contrasts']:
            values=np.array([r[sex]['contrasts'][key]['young_minus_older'] for r in records])
            boot=(values[draws]*weights[draws]).sum(axis=1)/weights[draws].sum(axis=1)
            outcomes[key]={'mean':float(np.average(values,weights=weights)),'interval_95':np.quantile(boot,[.025,.975]).tolist()}
        result['by_gender'][sex]={'profiles':int(weights.sum()),'outcomes':outcomes}
    return result


def pov_sensitivity_analysis(pov):
    """Freeze equal-country estimates for every retained childless sensitivity scenario."""
    result={}
    for scenario,countries in pov['headline_sensitivities'].items():
        names=sorted(name for name,row in countries.items() if all(row.get(band,{}).get(sex) and row[band][sex]['n']>=100 for band in ['25-29','45-49'] for sex in SEXES))
        # Reuse exactly the same paired-country estimand as principal comparisons.
        atlas={name:{sex:{band:countries[name][band][sex] for band in ['25-29','45-49']} for sex in SEXES} for name in names}
        result[scenario]=pov_analysis({'countries':atlas,'common_support_countries':{'all':names}})['all']
        result[scenario]['cohort']='childless; scenario-specific seeking, single, answered and/or exact-age standardization; fixed country set across both genders and headline age bands'
    return result
