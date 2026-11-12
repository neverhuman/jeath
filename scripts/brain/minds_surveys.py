"""Exploratory reproductions from hash-pinned public survey files."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILES = {
 'Final_dataset.csv':('https://osf.io/download/eyn65/','00f3292fa94280ad126724700fb439570fbada271595c4235968d00aa295120a'),
 'Codebook.xlsx':('https://osf.io/download/4e5nr/','5150b30737bd6ae3484e30fbe908ee06bd1750001da4a7e3b8397f5a1470ade7'),
 'walter-primary.pdf':('https://labs.la.utexas.edu/buss/files/2020/03/Sex-Differences-in-Mate-Preferences-Across-45-Nations-2020.pdf','ebf657cc47d184d7364c2bd93ce13f0bdc5898fec34ccc876b92098eed700d84'),
 'processed_data.csv':('https://osf.io/download/5fmyu/','0509eb7bbaae75c08533d19e1e6fa3a343a4775d825d644068be63824dff5933'),
 'whyte.dta':('https://osf.io/download/f84pw/','a7c621f9aa6e68719f92010a354500d38bd3a26013368e6a87a2c87070f87f00'),
 'whyte-analysis.do':('https://osf.io/download/cykas/','fd0066cc5f22abacc35afc052b73fb6d1bb861ad237ca4ce3148db698c599671'),
 'walter-analysis.R':('https://osf.io/download/yn965/','e664d8e374deb60fc4d9ad933df7c4400f0a38fc641a9fad4b7b2e7f5f9c415f'),
}
ITEMS={'kindness':'preference_imp_kind','health':'preference_imp_health','appearance':'preference_imp_physat','resources':'preference_imp_financ','appropriate_age':'preference_imp_age','religion':'preference_imp_relig'}
SEXES={0:'men',1:'women'}


def fetch(folder):
    folder.mkdir(parents=True,exist_ok=True);sources=[]
    for name,(url,want) in FILES.items():
        p=folder/name
        if not p.exists():
            with urllib.request.urlopen(url,timeout=180) as r:p.write_bytes(r.read())
        got=hashlib.sha256(p.read_bytes()).hexdigest()
        if got!=want:raise ValueError('Survey hash mismatch '+name)
        sources.append({'file':name,'url':url,'sha256':got,'bytes':p.stat().st_size})
    return sources


def estimate(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)];n=len(x)
    if not n:return None
    mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(n)) if n>1 else 0.
    return {'n':n,'mean':mean,'sd':float(x.std(ddof=1)) if n>1 else 0.,'interval_95':[mean-1.96*se,mean+1.96*se]}


def country_summary(values):
    x=np.asarray(values,float);rng=np.random.default_rng(20261111);draws=x[rng.integers(len(x),size=(2000,len(x)))].mean(axis=1)
    return {'countries':len(x),'mean':float(x.mean()),'interval_95':np.quantile(draws,[.025,.975]).tolist(),'between_country_quantiles':np.quantile(x,[0,.1,.25,.5,.75,.9,1]).tolist(),'positive':int((x>0).sum()),'negative':int((x<0).sum()),'zero':int((x==0).sum()),'bootstrap_unit':'country','bootstrap_draws':2000}


def kowal(folder):
    columns=['Age','Sex','Sex01','Sexual_Orientation','Country_live','Children','Relationship_status','Relationship_status01','cot_educ','self_physatt','self_resources','self_kind','self_health',*ITEMS.values()]
    d=pd.read_csv(folder/'Final_dataset.csv',sep=';',encoding='latin1',usecols=columns,low_memory=False)
    for col in columns:
        if col!='Country_live':d[col]=pd.to_numeric(d[col],errors='coerce')
    x=d[list(ITEMS.values())].to_numpy();flow={'source':len(d)}
    ok=d.Age.between(18,90).to_numpy();flow['age_18_90']=int(ok.sum())
    ok&=d.Sex.isin([0,1]).to_numpy();flow['gender_man_woman']=int(ok.sum())
    ok&=(d.Sexual_Orientation==1).to_numpy();flow['heterosexual']=int(ok.sum())
    ok&=np.isfinite(x).all(axis=1);flow['complete_six']=int(ok.sum())
    ok&=((x>=0)&(x<=30)).all(axis=1);flow['valid_range']=int(ok.sum())
    ok&=np.isclose(x.sum(axis=1),30,rtol=0,atol=1e-6);flow['sum_30']=int(ok.sum())
    q=d.loc[ok].copy();q['age_band']=np.select([q.Age<=29,q.Age<=44],[0,1],default=2)
    flow['country_observed']=int(q.Country_live.notna().sum())
    pooled={s:{item:estimate(q.loc[q.Sex==code,col]) for item,col in ITEMS.items()} for code,s in SEXES.items()}
    countries={};standardized={}
    for name,g in q.groupby('Country_live'):
        if min(int((g.Sex==v).sum()) for v in SEXES)<100:continue
        bysex={s:{item:estimate(g.loc[g.Sex==code,col]) for item,col in ITEMS.items()} for code,s in SEXES.items()}
        diffs={item:{'women_minus_men':bysex['women'][item]['mean']-bysex['men'][item]['mean'],'interval_95':[bysex['women'][item]['mean']-bysex['men'][item]['mean']+sign*1.96*np.sqrt(bysex['women'][item]['sd']**2/bysex['women'][item]['n']+bysex['men'][item]['sd']**2/bysex['men'][item]['n']) for sign in [-1,1]]} for item in ITEMS}
        countries[name]={'n':len(g),'by_gender':bysex,'contrasts':diffs,'age_mean':{s:float(g.loc[g.Sex==c,'Age'].mean()) for c,s in SEXES.items()}}
        if all(int(((g.Sex==c)&(g.age_band==b)).sum())>=20 for c in SEXES for b in range(3)):
            weights=np.array([(g.age_band==b).mean() for b in range(3)])
            standardized[name]={'n':len(g),'age_weights':weights.tolist(),'by_gender':{s:{item:float(sum(weights[b]*g.loc[(g.Sex==c)&(g.age_band==b),col].mean() for b in range(3))) for item,col in ITEMS.items()} for c,s in SEXES.items()}}
    flow['country_min100_each_gender']=sum(r['n'] for r in countries.values())
    equal={s:{item:country_summary([r['by_gender'][s][item]['mean'] for r in countries.values()]) for item in ITEMS} for s in SEXES.values()}
    differences={item:country_summary([r['contrasts'][item]['women_minus_men'] for r in countries.values()]) for item in ITEMS}
    standardized_summary={item:country_summary([r['by_gender']['women'][item]-r['by_gender']['men'][item] for r in standardized.values()]) for item in ITEMS}
    correlations={item:float(np.corrcoef([r['by_gender']['women'][item]['mean'] for r in countries.values()],[r['by_gender']['men'][item]['mean'] for r in countries.values()])[0,1]) for item in ITEMS}
    compositional={}
    for field in ['age_band','Relationship_status','cot_educ','Children']:
        compositional[field]={}
        for category,g in q.groupby(field):
            if len(g)<100:continue
            compositional[field][str(category)]={s:{item:estimate(g.loc[g.Sex==c,col]) for item,col in ITEMS.items()} for c,s in SEXES.items() if (g.Sex==c).sum()>=100}
    # Within-country, age-adjusted linear probability models of current non-single status; all four self-ratings together.
    partner={}
    traits=['self_physatt','self_resources','self_kind','self_health']
    for code,s in SEXES.items():
        g=d.loc[d.Age.between(18,60)&(d.Sex==code)&d.Country_live.notna()&d.Relationship_status01.isin([0,1]),['Age','Country_live','Relationship_status01',*traits]].dropna().copy()
        country_n=g.groupby('Country_live').size();g=g[g.Country_live.isin(country_n[country_n>=100].index)]
        a=g[traits].to_numpy();a=(a-a.mean(axis=0))/a.std(axis=0);a=np.column_stack([a,g.Age.to_numpy(),g.Age.to_numpy()**2]);y=g.Relationship_status01.to_numpy(float)
        for _,idx in g.groupby('Country_live').indices.items():a[idx]-=a[idx].mean(axis=0);y[idx]-=y[idx].mean()
        coef=np.linalg.lstsq(a,y,rcond=None)[0];partner[s]={'n':len(g),'countries':int(g.Country_live.nunique()),'coefficients_pp_per_pooled_sd':dict(zip(traits,(100*coef[:4]).tolist())),'adjustment':'country fixed effects, age and age squared, simultaneous four self-ratings','uncertainty':'point estimates only; no population design known','estimand':'cross-sectional association with reporting non-single status, including dating'}
    return {'flow':flow,'gender_counts':{s:int((q.Sex==c).sum()) for c,s in SEXES.items()},'items':list(ITEMS),'pooled':pooled,'countries':countries,'equal_country':equal,'country_gender_contrasts':differences,'age_standardized_countries':standardized,'age_standardized_contrasts':standardized_summary,'country_gender_mean_correlations':correlations,'composition':compositional,'self_quality_partnering':partner,'source_coding':{'Sex':'0 men,1 women; codebook What is your gender?','orientation':'Sexual_Orientation==1 heterosexual or straight','country':'Country_live current self-reported country of residence','budget':'six0-30 point allocations, fixed total30','children':'0,1,2,3,4,5or more; not uncensored completed fertility'},'recruitment':'Nonprobability social-media and snowball samples; outsourcing and paper-questionnaire modes vary by country; Algeria invalid totals retained in exclusion audit. Country estimates are sample contrasts, not national parameters.'}


def whyte(folder):
    reader=pd.io.stata.StataReader(folder/'whyte.dta');labels=reader.variable_labels();d=pd.read_stata(folder/'whyte.dta',convert_categoricals=False)
    items=[c for c in d.columns if c.endswith('_import')];bands=[(18,29),(30,39),(40,49),(50,65)];sexes={1:'men',0:'women'}
    x=d[items].to_numpy(float);assert len(d)==7060 and d.id.nunique()==7060 and np.isfinite(x).all() and ((x>=0)&(x<=100)).all()
    results={}
    for name,mask,center in [('all',np.ones(len(d),bool),False),('heterosexual',(d.hetero_dummy==1).to_numpy(),False),('centered',np.ones(len(d),bool),True),('heterosexual_centered',(d.hetero_dummy==1).to_numpy(),True)]:
        matrix=x-x.mean(axis=1,keepdims=True) if center else x
        row={}
        for code,s in sexes.items():
            row[s]={}
            for lo,hi in [(18,65),*bands]:
                m=mask&(d.male_dummy==code).to_numpy()&d.age.between(lo,hi).to_numpy()
                values=matrix[m];record={'n':int(m.sum()),'items':{item:estimate(values[:,i]) for i,item in enumerate(items)}}
                record['paired_contrasts']={name:estimate(x[m,items.index('trust_import')]-x[m,items.index(other)]) for name,other in [('trust_minus_intelligence','intelligence_import'),('trust_minus_attractiveness','attractive_import')]}
                row[s][f'{lo}-{hi}']=record
        for values in row.values():
            values['18-64']=values['18-65']
            values['50-64']=values['50-65']
        results[name]=row
    return {'records':len(d),'article_headline_records':7325,'unavailable_difference':265,'age_range':[int(d.age.min()),int(d.age.max())],'items':items,'variable_labels':labels,'gender_coding':'male_dummy1 men,0 women; variable label Male dummy and source Stata lines93-97 use mc_ for1 and fc_ for0','heterosexual_definition':'hetero_dummy==1; zero is not one homogeneous orientation','analyses':results,'intervals':'mean plus/minus1.96 standard errors conditional on independent respondents; paired differences use within-person differences','intelligence_wording':'deciding to be sexually intimate, unlike general sexual-attractiveness wording of several other items','sample_discrepancy':'265-row source/article difference unresolved; no reconstruction or imputation'}


def walter(folder):
    d=pd.read_csv(folder/'processed_data.csv',encoding='latin1');traits=['kindness','intelligence','health','physatt','resources']
    counts={str(c):int((d.sex==c).sum()) for c in [0,1]}
    # Primary final-paper Methods reports14399 respondents,7909female. This
    # independently reconciles code0 withwomen andcode1 withmen, without using outcomes.
    # No literal original numeric questionnaire-key has been recovered.
    if len(d)!=14399 or counts!={'0':7909,'1':6490}:raise ValueError('Walter published-count reconciliation differs')
    raw={str(c):{t:estimate(pd.to_numeric(d.loc[d.sex==c,'ideal_'+t],errors='coerce')) for t in traits} for c in [0,1]}
    countries_raw={name:{str(c):{t:estimate(pd.to_numeric(g.loc[g.sex==c,'ideal_'+t],errors='coerce')) for t in traits} for c in [0,1]} for name,g in d.groupby('country')}
    labelled={}
    for name,row in countries_raw.items():
        bysex={'women':row['0'],'men':row['1']};contrasts={}
        for trait in traits:
            w,m=bysex['women'][trait],bysex['men'][trait];delta=w['mean']-m['mean'];se=np.sqrt(w['sd']**2/w['n']+m['sd']**2/m['n'])
            contrasts[trait]={'women_minus_men':delta,'interval_95':[delta-1.96*se,delta+1.96*se]}
        labelled[name]={'by_gender':bysex,'contrasts':contrasts}
    return {'records':len(d),'countries':int(d.country.nunique()),'coding_status':'count-reconciled mapping, conditional on exact final published sample composition; original numeric questionnaire key not recovered','recorded_code_counts':counts,'coding_reconciliation':{'published_final_n':14399,'published_female_n':7909,'source_code0_n':counts['0'],'source_code1_n':counts['1'],'mapping':{'0':'women','1':'men'},'basis':'Final primary paper Methods,p4 reports14399total and7909female; only code0 has7909 rows, and processedtotal matches. This is an explicit count-based inference, not an outcome-based age-preference inference.','primary_pdf_sha256':'ebf657cc47d184d7364c2bd93ce13f0bdc5898fec34ccc876b92098eed700d84','primary_url':FILES['walter-primary.pdf'][0],'original_numeric_key_available':False},'pooled_by_recorded_code':raw,'countries_by_recorded_code':countries_raw,'pooled_count_reconciled':{'women':raw['0'],'men':raw['1']},'countries_count_reconciled':labelled,'equal_country_count_reconciled':{sex:{trait:country_summary([r['by_gender'][sex][trait]['mean'] for r in labelled.values()]) for trait in traits} for sex in ['women','men']},'country_gender_contrasts_count_reconciled':{trait:country_summary([r['contrasts'][trait]['women_minus_men'] for r in labelled.values()]) for trait in traits},'scale':'source ideal_ variables,1-7 bipolar desiredlevel, not six-item fixedbudget','recruitment':'Collected2016 inperson, university/community convenience samples. Finalarticle knownsite records6604respondents,3118community47.21%; about halfsite compositionrecords missing.','intervals':'Country within-gender conditional mean intervals; countrybalanced2000paired-country bootstrap summaries exploratory. These descriptive comparisons do not reproduce original multilevel modelling or national sampling.'}


def eastwick(folder):
    path=folder/'eastwick.html'
    if not path.exists():path.write_bytes(urllib.request.urlopen('https://pmc.ncbi.nlm.nih.gov/articles/PMC12622239/',timeout=45).read())
    table=pd.read_html(io.StringIO(path.read_text()))[4]
    def number(v):return float(str(v).replace('−','-').replace('*',''))
    attributes=[]
    for _,row in table.iloc[1:36].iterrows():
        v=row.tolist();attributes.append({'attribute':v[0],'n':int(v[1]),'stated_mean':number(v[2]),'stated_sd':number(v[3]),'stated_rank':int(v[4]),'evaluation_beta':number(v[6]),'evaluation_rank':int(v[7])})
    composites={};current=None
    for _,r in table.iterrows():
        v=r.tolist()
        if v[0] in ['Attractiveness Composite','Earning Potential Composite']:current=v[0];composites[current]={}
        elif v[0] in ['Heterosexual Men','Heterosexual Women'] and current:
            composites[current][v[0]]={'n':int(v[1]),'stated_mean':number(v[2]),'stated_sd':number(v[3]),'evaluation_beta':number(v[6])}
    return {'source_url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC12622239/','table':'Table5','html_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'attributes':attributes,'gender_composites':composites,'interpretation':'Published descriptive estimates, not a new respondent-level reanalysis. Stated means on source scale; separate standardized mixed-model evaluation slopes, not mutually adjusted weights or correlations. Source table provides no coefficient intervals.'}


def analyze(folder):
    sources=fetch(folder)
    return {'sources':sources,'kowal':kowal(folder),'whyte':whyte(folder),'walter':walter(folder),'eastwick':eastwick(folder)}
