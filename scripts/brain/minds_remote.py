#!/usr/bin/env python3
"""Read-only, aggregate-only scan executed through stdin on xbabe3."""
import base64
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SEXES = ('Female', 'Male')
BANDS = [(18,29),(30,44),(45,64),(65,79),(25,29),(45,49)]
FIVE = [(18,19)]+[(x,x+4) for x in range(20,80,5)]


def read(name, columns):
    return pq.read_table(Path('exports')/(name+'.parquet'),columns=columns,use_threads=False)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def array(col, fallback=-1):return pc.fill_null(col,fallback).to_numpy()


def dictionary(col):
    x=pc.dictionary_encode(pc.fill_null(col,'' )).combine_chunks()
    return x.indices.to_numpy(), x.dictionary.to_pylist()


def midranks(values):
    _, inverse, counts=np.unique(values,return_inverse=True,return_counts=True)
    return ((counts.cumsum()-.5*counts)/len(values))[inverse]


def gini(x):
    x=np.sort(x.astype(float));return float((2*np.dot(np.arange(1,len(x)+1),x)/x.sum()-len(x)-1)/len(x)) if len(x) and x.sum() else None


def stats(idx,score,rank,global_rank,pooled,likes,votes,images):
    if len(idx)==0:return None
    r=rank[idx];s=score[idx];v=votes[idx];l=likes[idx];ok=np.isfinite(s)&(s>=0)&(s<=10)
    return {'n':len(idx),'score_n':int(ok.sum()),'mean_score':float(s[ok].mean()) if ok.any() else None,'median_score':float(np.median(s[ok])) if ok.any() else None,
            'score_quantiles':np.quantile(s[ok],[.1,.25,.5,.75,.9]).tolist() if ok.any() else None,
            'mean_percentile':float(np.nanmean(r)),'mean_global_gender_percentile':float(np.nanmean(global_rank[idx])),
            'mean_pooled_percentile':float(np.nanmean(pooled[idx])), 'percentile_quantiles':np.nanquantile(r,[.1,.25,.5,.75,.9]).tolist(),
            'upper_decile_share':float(np.nanmean(r>=.9)),'upper_quartile_share':float(np.nanmean(r>=.75)),
            'mean_like_rate':float((l/v).mean()),'vote_pooled_like_rate':float(l.sum()/v.sum()),
            'votes':int(v.sum()),'mean_votes':float(v.mean()),'vote_quantiles':np.quantile(v,[.1,.25,.5,.75,.9,.99]).tolist(),
            'votes_gini':gini(v),'inventoried_n':int(images[idx].sum())}


def hon():
    p=read('profiles',['profile_id','age','gender','country','metadata_conflict','label_conflict','image_count','decoded_count'])
    l=read('labels',['profile_id','likes','votes','beauty_score','status'])
    assert pc.all(pc.equal(p['profile_id'],l['profile_id'])).as_py()
    n=len(p);age=array(p['age']);sex,sexnames=dictionary(p['gender']);country,cnames=dictionary(p['country'])
    aliases=GEO['country_aliases'];normal=[aliases.get(x.strip().casefold(),x if x else 'Unresolved blank') for x in cnames]
    normnames=sorted(set(normal));normindex={x:i for i,x in enumerate(normnames)};country=np.array([normindex[x] for x in normal],dtype='int16')[country]
    sexcode=np.array([SEXES.index(x) if x in SEXES else -1 for x in sexnames],dtype='int8')[sex]
    likes=array(l['likes']);votes=array(l['votes']);score=array(l['beauty_score'],np.nan)
    images=array(p['image_count'])>0;decoded=array(p['decoded_count']);icount=array(p['image_count'])
    base=(age>=18)&(age<=79)&(sexcode>=0)&(array(p['metadata_conflict'])==0)&(array(p['label_conflict'])==0)&(votes>0)&(likes>=0)&(likes<=votes)
    del p,l
    gc.collect()
    primary=base&(votes>=20);score_ok=np.isfinite(score)&(score>=0)&(score<=10)
    rank=np.full(n,np.nan,dtype='float32');glob=rank.copy();pool=rank.copy()
    for s in range(2):
        ii=np.flatnonzero(primary&score_ok&(sexcode==s));glob[ii]=midranks(score[ii])
    ii=np.flatnonzero(primary&score_ok);pool[ii]=midranks(score[ii])
    order=np.flatnonzero(primary);order=order[np.argsort(country[order],kind='stable')]
    countries={};coverage=[];common=[];source_country_counts=[]
    for c,cname in enumerate(normnames):
        ii=order[np.searchsorted(country[order],c,'left'):np.searchsorted(country[order],c,'right')]
        sexes={}
        for s,sname in enumerate(SEXES):
            jj=ii[sexcode[ii]==s];kk=jj[score_ok[jj]]
            if len(kk):rank[kk]=midranks(score[kk])
            broad={f'{lo}-{hi}':int(((age[jj]>=lo)&(age[jj]<=hi)).sum()) for lo,hi in BANDS}
            sexes[sname]={'n':len(jj),'band_n':broad}
        if all(all(sexes[s]['band_n'][f'{lo}-{hi}']>=100 for lo,hi in BANDS[:3]) for s in SEXES):common.append(cname)
        countries[cname]=sexes
    # Mean/quantile distributions across each country's fixed adult reference.
    thresholds={}
    for threshold in [1,20,100]:
        bycountry={}
        for c,cname in enumerate(normnames):
            ii=np.flatnonzero(base&(votes>=threshold)&(country==c));sexes={}
            for s,sname in enumerate(SEXES):
                jj=ii[sexcode[ii]==s]
                if threshold==20:
                    coverage.append({'country':cname,'gender':sname,'eligible':len(jj),'inventoried':int(images[jj].sum()),'decoded_count':int(decoded[jj].clip(0).sum()),'inventoried_count':int(icount[jj].clip(0).sum())})
                if len(jj)<100:continue
                # Sensitivity ranks are recomputed against this threshold's adult reference.
                rr=np.full(n,np.nan,dtype='float32') if threshold!=20 else rank
                if threshold!=20:
                    kk=jj[score_ok[jj]];rr[kk]=midranks(score[kk])
                sexrow={'all':stats(jj,score,rr,glob,pool,likes,votes,images),'bands':{}}
                for lo,hi in BANDS+FIVE:
                    kk=jj[(age[jj]>=lo)&(age[jj]<=hi)]
                    if len(kk)>=100:sexrow['bands'][f'{lo}-{hi}']=stats(kk,score,rr,glob,pool,likes,votes,images)
                if threshold==20 and cname in common:
                    exact=[]
                    for a in range(18,65):
                        kk=jj[age[jj]==a];ok=kk[score_ok[kk]]
                        exact.append({'age':a,'n':len(kk),'score_n':len(ok),'mean_score':float(score[ok].mean()) if len(ok) else None,'mean_percentile':float(rank[ok].mean()) if len(ok) else None,'mean_global_gender_percentile':float(glob[ok].mean()) if len(ok) else None,'mean_pooled_percentile':float(pool[ok].mean()) if len(ok) else None,'mean_like_rate':float((likes[kk]/votes[kk]).mean()) if len(kk) else None})
                    sexrow['exact_age']=exact
                if threshold==20:
                    for tag,mask in [('inventoried',images[jj]),('not_inventoried',~images[jj])]:
                        kk=jj[mask];sexrow[tag]=stats(kk,score,rr,glob,pool,likes,votes,images) if len(kk)>=100 else None

                sexes[sname]=sexrow
            if sexes:bycountry[cname]=sexes
        thresholds[str(threshold)]=bycountry
    source=read('HON_source_fields',['image_count','why_here','work_and_education','languages','interests','about_me'])
    fields={}
    for field in source.column_names:
        col=pc.fill_null(source[field],'');present=pc.not_equal(pc.utf8_trim_whitespace(col),'').to_numpy()
        fields[field]={'populated_all':int(present.sum()),'overlap_primary':int((present&primary).sum())}
    reported=source['image_count'].to_pylist();rphoto=np.array([int(x) if x and x.isdigit() else -1 for x in reported],dtype='int16')
    del source,reported
    gc.collect()
    portfolios={}
    for name,photos in [('reported',rphoto),('inventoried',icount)]:
        groups={}
        for s,sname in enumerate(SEXES):
            groups[sname]={}
            for tag,pm in [('0',photos==0),('1',photos==1),('2',photos==2),('3-4',(photos>=3)&(photos<=4)),('5+',photos>=5)]:
                jj=np.flatnonzero(primary&(sexcode==s)&pm)
                if len(jj)>=100:groups[sname][tag]=stats(jj,score,rank,glob,pool,likes,votes,images)
        portfolios[name]=groups
    portfolio_adjusted={}
    for name,photos in [('reported',rphoto),('inventoried',icount)]:
        portfolio_adjusted[name]={}
        for sex,sname in enumerate(SEXES):
            portfolio_adjusted[name][sname]={}
            for outcome,y,adjust_votes in [('percentile',rank,True),('like_rate',likes/np.maximum(votes,1),True),('log1p_votes',np.log1p(np.maximum(votes,0)),False)]:
                ii=np.flatnonzero(primary&(sexcode==sex)&(age<=64)&((photos==1)|(photos>=5))&np.isfinite(y))
                values=[country[ii],age[ii]]
                if adjust_votes:values.append(np.digitize(votes[ii],[100,500,2000,10000]))
                _,inv=np.unique(np.stack(values,axis=1),axis=0,return_inverse=True);k=int(inv.max()+1)
                nn=np.zeros((2,k));yy=np.zeros((2,k))
                for j,mask in enumerate([photos[ii]==1,photos[ii]>=5]):
                    nn[j]=np.bincount(inv[mask],minlength=k);yy[j]=np.bincount(inv[mask],weights=y[ii[mask]],minlength=k)
                support=(nn>=5).all(axis=0);weights=nn[:,support].sum(axis=0);rates=(yy[:,support]/nn[:,support])@weights/weights.sum() if weights.sum() else None
                portfolio_adjusted[name][sname][outcome]={'categories':['1','5+'],'common_strata':int(support.sum()),'supported_profiles':int(weights.sum()),'rates':list(map(float,rates)) if rates is not None else None,'five_plus_minus_one':float(rates[1]-rates[0]) if rates is not None else None,'adjustment':['country','exact_age']+(['vote_history_bins20-99,100-499,500-1999,2000-9999,10000+'] if adjust_votes else []),'minimum_per_category_stratum':5,'interpretation':'cross-sectional standardized association; no causal portfolio effect'}
    exclusions={'all':n,'age_gender':int(((age>=18)&(age<=79)&(sexcode>=0)).sum()),'valid_unconflicted':int(base.sum()),'primary':int(primary.sum()),'score_primary':int((primary&score_ok).sum())}
    image_country={cname:int((primary&images&(country==c)).sum()) for c,cname in enumerate(normnames)}
    return {'flow':exclusions,'common_support_countries':common,'countries_before_normalization':len(cnames),'countries_after_normalization':len(normnames),'decoded_count_meaning':'Decoded_count in canonical profiles export is the earlier index/pilot checkpoint; completed current faceflow processing counts are separately recorded under experiments.image_processing_summary. No final merged per-profile decode inventory is assumed.','country_normalization':[{'source':x,'normalized':y,'status':'exact_alias' if x.strip().casefold() in aliases else 'unresolved_label'} for x,y in zip(cnames,normal)],'thresholds':thresholds,'coverage':coverage,'image_country_counts':image_country,'field_coverage':fields,'portfolio':portfolios,'portfolio_standardization':portfolio_adjusted}


def map_coordinates(strings):
    from matplotlib.path import Path as Polygon
    unique=list(dict.fromkeys(strings));points=np.full((len(unique),2),np.nan);valid=np.zeros(len(unique),bool)
    for i,x in enumerate(unique):
        try:
            pair=json.loads(x);lat,lon=pair
            if isinstance(lat,(float,int)) and isinstance(lon,(float,int)) and np.isfinite(lat) and np.isfinite(lon) and -90<=lat<=90 and -180<=lon<=180:points[i]=[lon,lat];valid[i]=True
        except (TypeError,ValueError):pass
    found=[set() for _ in unique]
    for feature in GEO['geography']:
        inside=np.zeros(len(unique),bool)
        for ring in feature['rings']:
            vertices=np.asarray(ring);xmin,ymin=vertices.min(axis=0);xmax,ymax=vertices.max(axis=0)
            ii=np.flatnonzero(valid&(points[:,0]>=xmin)&(points[:,0]<=xmax)&(points[:,1]>=ymin)&(points[:,1]<=ymax))
            if len(ii):inside[ii]^=Polygon(vertices).contains_points(points[ii])
        for i in np.flatnonzero(inside):found[i].add(feature['name'])
    result={x:(next(iter(found[i])) if len(found[i])==1 else '',bool(valid[i]),'multiple_boundary_hits' if len(found[i])>1 else 'mapped' if found[i] else 'valid_unassigned' if valid[i] else 'invalid_or_missing') for i,x in enumerate(unique)}
    return result


def resolve_country(state, coordinate):
    if state and coordinate and state != coordinate:return '', 'conflict'
    if state and coordinate:return state, 'agreement'
    if coordinate:return coordinate, 'coordinates_only'
    if state:return state, 'state_only'
    return '', 'unresolved'


def pov():
    columns=['age','gender','seeking','haschildren','wantchildren','intent','for','ambition','college_id','maritalstatus','state_id','location','relationshipage_id','height','body','religion','profession','country']
    t=read('POV_source_fields',columns);n=len(t)
    codes={};labels={}
    for k in columns:
        codes[k],labels[k]=dictionary(t[k])
    del t;gc.collect()
    ages=np.array([int(x) if x.isdigit() else -1 for x in labels['age']],dtype='int16')[codes['age']]
    sexes=np.array([SEXES.index(x) if x in SEXES else -1 for x in labels['gender']],dtype='int8')[codes['gender']]
    seeking=np.array([{'Man':0,'Woman':1}.get(x,-1) for x in labels['seeking']],dtype='int8')[codes['seeking']]
    eligible=(ages>=18)&(ages<=79)&(sexes>=0)&(sexes==seeking)
    states=labels['state_id'];aliases=GEO['aliases'];state_country=[aliases.get(x.strip().casefold(),'') for x in states]
    locations=labels['location'];mapping=map_coordinates(locations)
    assigned=[];statuses=[];coord_country=[];valid_location=[]
    # Joint state/location categories avoid any exported coordinates.
    pairs=np.unique(np.stack([codes['state_id'],codes['location']],axis=1),axis=0)
    pairmap={}
    for si,li in pairs:
        st=state_country[si];co,valid,cs=mapping[locations[li]]
        name,status=resolve_country(st,co)
        pairmap[(si,li)]=(name,status,co,valid)
    country_names=sorted({v[0] for v in pairmap.values()}|{''});ci={x:i for i,x in enumerate(country_names)}
    status_names=['agreement','coordinates_only','state_only','conflict','unresolved'];sti={x:i for i,x in enumerate(status_names)}
    country=np.empty(n,dtype='int16');status=np.empty(n,dtype='int8');coordinate=np.zeros(n,bool);validcoords=np.zeros(n,bool)
    for i,(s,l) in enumerate(zip(codes['state_id'],codes['location'])):
        name,st,co,va=pairmap[(s,l)];country[i]=ci[name];status[i]=sti[st];coordinate[i]=bool(co) and st!='conflict';validcoords[i]=va
    parent=np.array([0 if x=='No' else 1 if x in ('Yes','All my kids are over 18') else -1 for x in labels['haschildren']],dtype='int8')[codes['haschildren']]
    wants=np.array([{'Want children':1,'Does not want children':0,'Undecided/Open':2}.get(x,3) for x in labels['wantchildren']],dtype='int8')[codes['wantchildren']]
    intentlabels=labels['intent'];serious=np.isin(codes['intent'],[i for i,x in enumerate(intentlabels) if x in ('is looking for a relationship','is actively seeking a relationship','wants to find someone to marry')])
    longterm=np.array([x=='Long Term' for x in labels['for']])[codes['for']]
    def outcome(ii):
        if len(ii)<100:return None
        known=parent[ii]>=0
        return {'n':len(ii),'parenthood_known':int(known.sum()),'parents':int((parent[ii]==1).sum()),'parenthood_share':float((parent[ii][known]==1).mean()) if known.any() else None,'want_children_share':float((wants[ii]==1).mean()),'serious_intent_share':float(serious[ii].mean()),'for_long_term_share':float(longterm[ii].mean()),'want_counts':{key:int((wants[ii]==v).sum()) for key,v in [('want',1),('do_not_want',0),('undecided_open',2),('nonresponse',3)]},'intent_counts':{key:int((codes['intent'][ii]==j).sum()) for j,key in enumerate(intentlabels)},'for_counts':{key:int((codes['for'][ii]==j).sum()) for j,key in enumerate(labels['for'])}}
    bins=[(18,24),(25,29),(30,34),(35,39),(40,44),(45,49),(50,54),(55,59),(60,64),(65,69),(70,79)]
    countries={};common={key:[] for key in ['all','childless','coordinate_all','coordinate_childless']}
    for c,cname in enumerate(country_names):
        if not cname:continue
        row={}
        for s,sname in enumerate(SEXES):
            row[sname]={}
            for lo,hi in bins:
                ii=np.flatnonzero(eligible&(country==c)&(sexes==s)&(ages>=lo)&(ages<=hi))
                result=outcome(ii)
                if result:
                    result['childless']=outcome(ii[parent[ii]==0]);result['coordinate']=outcome(ii[coordinate[ii]]);result['coordinate_childless']=outcome(ii[coordinate[ii]&(parent[ii]==0)])
                    row[sname][f'{lo}-{hi}']=result
        for name,tag in [('all',None),('childless','childless'),('coordinate_all','coordinate'),('coordinate_childless','coordinate_childless')]:
            if all((row[s].get(b) if tag is None else (row[s].get(b) or {}).get(tag)) for s in SEXES for b in ['25-29','45-49']):common[name].append(cname)
        if any(row.values()):countries[cname]=row
    headline_sensitivities={}
    all_eligible=(ages>=18)&(ages<=79)&(sexes>=0)
    single=np.array([x=='Single' for x in labels['maritalstatus']])[codes['maritalstatus']]
    for name, options in [('main',{}),('single',{'single':True}),('answered',{'answered':True}),('all_seeking',{'all_seeking':True}),('exact_age',{'exact_age':True}),('single_answered_exact_age',{'single':True,'answered':True,'exact_age':True})]:
        bycountry={}
        for cname in common['childless']:
            c=ci[cname];byband={}
            for lo,hi in [(25,29),(45,49)]:
                mask=(all_eligible if options.get('all_seeking') else eligible)&(country==c)&(parent==0)&(ages>=lo)&(ages<=hi)
                if options.get('single'):mask&=single
                if options.get('answered'):mask&=wants!=3
                bysex={sname:outcome(np.flatnonzero(mask&(sexes==s))) for s,sname in enumerate(SEXES)}
                if not all(bysex.values()):continue
                if options.get('exact_age'):
                    weights=np.array([int((mask&(ages==a)).sum()) for a in range(lo,hi+1)],float)
                    means={};valid=True
                    for s,sname in enumerate(SEXES):
                        cells=[np.flatnonzero(mask&(sexes==s)&(ages==a)) for a in range(lo,hi+1)]
                        if any(len(ii)==0 for ii in cells):valid=False;break
                        means[sname]={key:float(np.dot(weights,[float(y[ii].mean()) for ii in cells])/weights.sum()) for key,y in [('want_children_share',wants==1),('serious_intent_share',serious),('for_long_term_share',longterm)]}
                    if not valid:continue
                    for sname in SEXES:
                        for key,value in means[sname].items():bysex[sname][key]=value
                    bysex['exact_age_weights']=(weights/weights.sum()).tolist()
                byband[f'{lo}-{hi}']=bysex
            if len(byband)==2:bycountry[cname]=byband
        headline_sensitivities[name]=bycountry
    pooled={sname:{f'{lo}-{hi}':outcome(np.flatnonzero(eligible&(sexes==s)&(ages>=lo)&(ages<=hi))) for lo,hi in bins} for s,sname in enumerate(SEXES)}
    mapping_coverage={}
    for s,sname in enumerate(SEXES):
        mapping_coverage[sname]={}
        for lo,hi in bins:
            ii=np.flatnonzero(eligible&(sexes==s)&(ages>=lo)&(ages<=hi))
            mapping_coverage[sname][f'{lo}-{hi}']={name:int((status[ii]==j).sum()) for j,name in enumerate(status_names)}
    # Attribute comparisons: exact common strata for the two end categories, weighting pooled stratum sizes.
    attributes={}
    for field in ['ambition','college_id','relationshipage_id','body','religion']:
        bysex={}
        for s,sname in enumerate(SEXES):
            ii=np.flatnonzero(eligible&(sexes==s)&(ages>=25)&(ages<=49))
            bysex[sname]={key:outcome(ii[codes[field][ii]==j]) for j,key in enumerate(labels[field]) if int((codes[field][ii]==j).sum())>=100}
        attributes[field]={'population':'opposite-gender-seeking profiles25-49','unadjusted':bysex}
    standard={}
    for field,ends in [('ambition',('Not Ambitious','Very Ambitious')),('college_id',('High school','Bachelors Degree'))]:
        standard[field]={}
        for s,sname in enumerate(SEXES):
            standard[field][sname]={}
            for outname,y in [('parenthood',parent==1),('want_children',wants==1),('serious_intent',serious)]:
                selected=eligible&(sexes==s)&(ages>=25)&(ages<=49)&(country!=ci[''])
                if outname=='parenthood':selected&=parent>=0
                categories=np.array([labels[field].index(v) for v in ends]);ii=np.flatnonzero(selected&np.isin(codes[field],categories))
                keys=np.stack([ages[ii],country[ii],codes['maritalstatus'][ii],np.zeros(len(ii),int) if outname=='parenthood' else parent[ii]],axis=1)
                _,inv=np.unique(keys,axis=0,return_inverse=True);k=inv.max()+1 if len(inv) else 0
                nn=np.zeros((2,k));yy=np.zeros((2,k))
                for j,cval in enumerate(categories):
                    m=codes[field][ii]==cval;nn[j]=np.bincount(inv[m],minlength=k);yy[j]=np.bincount(inv[m],weights=y[ii[m]],minlength=k)
                support=(nn>=5).all(axis=0);weights=nn[:,support].sum(axis=0)
                rates=(yy[:,support]/nn[:,support])@weights/weights.sum() if weights.sum() else [np.nan,np.nan]
                standard[field][sname][outname]={'categories':list(ends),'common_strata':int(support.sum()),'supported_profiles':int(weights.sum()),'rates':list(map(float,rates)) if weights.sum() else None,'high_minus_low_pp':float(100*(rates[1]-rates[0])) if weights.sum() else None,'adjustment':['exact_age','country','maritalstatus']+([] if outname=='parenthood' else ['existing_parenthood']),'min_per_category_stratum':5}
    height={};categories=Counter()
    for j,key in enumerate(labels['height']):
        count=int((codes['height']==j).sum());m=re.search(r'(?<![<\d])(\d{3})\s*cm',key)
        if '<' in key:categories['censored']+=count
        elif not m:categories['malformed_or_missing']+=count
        elif not 120<=int(m.group(1))<=230:categories['implausible']+=count
        else:categories['valid_explicit_cm']+=count
    profession=Counter({key:int((codes['profession']==j).sum()) for j,key in enumerate(labels['profession'])})
    return {'definitions':{'serious_intent':'Exact English is looking for a relationship OR is actively seeking a relationship OR wants to find someone to marry; denominator all eligible profiles, rare other-language/nonresponse retained in rawcounts','for_long_term':'Exact Long Term in separate for question, denominator all eligible profiles','want_children':'Exact Want children; denominator includes all response categories','parenthood':'Yes or All my kids are over 18 vs No, denominator known parenthood responses','residence':'Derived current residence; no nationality inference'},'records':n,'eligible':int(eligible.sum()),'country_column_nonblank':sum(int((codes['country']==j).sum()) for j,key in enumerate(labels['country']) if key.strip()),'mapping_status_all':{key:int((status==j).sum()) for j,key in enumerate(status_names)},'valid_coordinate_profiles':int(validcoords.sum()),'coordinate_assigned_unconflicted_profiles':int(coordinate.sum()),'derived_country_profiles':int((country!=ci['']).sum()),'mapping_coverage':mapping_coverage,'common_support_countries':common,'countries':countries,'pooled':pooled,'attributes':attributes,'attribute_standardization':standard,'height_audit':dict(categories),'profession':{'nonblank':sum(v for k,v in profession.items() if k.strip()),'top_categories':profession.most_common(30),'grouping':'none; source strings retained'},'headline_sensitivities':headline_sensitivities,'geography_selection':{'mapped':outcome(np.flatnonzero(eligible&(country!=ci['']))),'unresolved':outcome(np.flatnonzero(eligible&(country==ci['']))),'coordinate':outcome(np.flatnonzero(eligible&coordinate))},'state_mapping_audit':[{'label':key,'n':int((codes['state_id']==j).sum()),'country':state_country[j] or None,'ambiguous':key.strip().casefold() in GEO['ambiguous_aliases']} for j,key in enumerate(states)]}


def experiments():
    leaderboard=Path('experiment/LEADERBOARD.json');leader=json.loads(leaderboard.read_text());waves=[]
    for j in range(1,41):
        folder=Path('experiment')/f'H{j}';path=folder/'metrics.json'
        if path.exists():
            data=json.loads(path.read_text());waves.append({'wave':f'H{j}','complete':True,'metrics_sha256':sha(path),'metrics':data,'config':json.loads((folder/'config.json').read_text()) if (folder/'config.json').exists() else None})
        else:waves.append({'wave':f'H{j}','complete':False})
    summary=Path('faceflow/output/summary.json')
    return {'leaderboard':leader,'leaderboard_sha256':sha(leaderboard),'waves':waves,'image_processing_summary':json.loads(summary.read_text()),'image_processing_summary_sha256':sha(summary),'finalization_artifacts':{str(p):p.exists() for p in [Path('faceflow/output/exports/manifest.json'),Path('faceflow/reports/global_duplicates.json'),Path('faceflow/output/duplicate_components.parquet')]},'audit_scope':'Metric/config snapshot only. Full-account global duplicate-component artifacts unavailable at inspected paths; no new model training or account-level predictions exported.'}


def main():
    assert platform.node().split('.')[0]=='xbabe3' and platform.machine()=='x86_64'
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    files=[Path('exports')/(x+'.parquet') for x in ['profiles','labels','HON_source_fields','POV_source_fields']]
    before={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in files};hashes={str(p):sha(p) for p in files}
    cutoff=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    h=hon();gc.collect();p=pov();gc.collect();e=experiments()
    assert before=={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in files}
    result={'hon':h,'pov':p,'experiments':e,'provenance':{'source_host':platform.node(),'architecture':platform.machine(),'snapshot_utc':cutoff,'sha256':hashes,'input_sizes_bytes':{k:v[0] for k,v in before.items()},'input_stat_unchanged':True,'exports':'aggregate statistics only; no account identifiers or coordinates','threads':1}}
    json.dump(result,sys.stdout,allow_nan=False,separators=(',',':'))

if __name__=='__main__':main()
