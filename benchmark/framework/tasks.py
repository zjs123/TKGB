"""Evaluation contracts accept predictions, never import a model."""
import csv
import math
from collections import defaultdict
from pathlib import Path
from ..core.schema import TimestampFact
from ..core.evaluator import evaluate_tail_ranking
from ..core.experiment_protocols import horizon_cutoff


def iter_facts(root,split):
    for kind in ['facts','points']:
        path=Path(root)/f'{kind}_{split}.csv'
        if not path.exists():continue
        with path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                yield TimestampFact(split=split,s_id=int(row['s_id']),r_id=int(row['r_id']),o_id=int(row['o_id']),
                                    t=row['t'],source_name=row.get('source_name',''),relation_label=row.get('relation_label',''),
                                    target_name=row.get('target_name',''),timestamp_source=kind,fact_hash=row.get('hash',''))


def evaluate_tail_file(root,records,split='test',scope='all',horizon=None,granularity='day'):
    queries=[];scores={};keys=set();identities=set()
    for index,row in enumerate(records):
        s,r,o,t=int(row['s_id']),int(row['r_id']),int(row['o_id']),str(row['t'])
        identity=(s,r,o,t)
        if identity in identities:raise ValueError('Duplicate query quadruple')
        identities.add(identity);keys.add((s,r,t))
        if horizon is not None and row.get('history_cutoff')!=horizon_cutoff(t,horizon,granularity):
            raise ValueError('Horizon predictions must declare the expected history_cutoff')
        scores[index]={int(k):float(v) for k,v in row['scores'].items()}
        if any(not math.isfinite(v) for v in scores[index].values()):raise ValueError('Nonfinite prediction score')
        queries.append(TimestampFact(split,s,r,o,'','','',t,'external_prediction'))
    truths=[];history=defaultdict(list);seen=set();found=set()
    sr_keys={(q.s_id,q.r_id) for q in queries}
    for part in ['train','valid','test']:
        for fact in iter_facts(root,part):
            if part=='train':seen.update([fact.s_id,fact.o_id])
            if (fact.s_id,fact.r_id,fact.t) in keys:truths.append(fact)
            if part==split and fact.id_quad() in identities:found.add(fact.id_quad())
            if part in ['train','valid'] and (fact.s_id,fact.r_id) in sr_keys:
                history[fact.s_id,fact.r_id].append((fact.t,fact.o_id))
    if found!=identities:raise ValueError('A prediction query is absent from the requested dataset split')
    with (Path(root)/'entities.csv').open(encoding='utf-8-sig',newline='') as f:
        candidates=[int(row['entity_id']) for row in csv.DictReader(f)]
    if scope=='transductive':
        if any(q.s_id not in seen or q.o_id not in seen for q in queries):raise ValueError('Non-transductive query in transductive evaluation')
        candidates=sorted(seen)
    elif scope=='inductive':
        if any(q.s_id in seen and q.o_id in seen for q in queries):raise ValueError('Train-seen query in inductive evaluation')
    result=evaluate_tail_ranking(queries,scores,truths,candidates).to_dict()
    groups=defaultdict(list)
    for i,q in enumerate(queries):
        past={o for t,o in history[q.s_id,q.r_id] if t<q.t}
        groups['C10' if q.o_id in past and len(past)<=10 else 'Other'].append(i)
    grouped={}
    for name,indices in groups.items():
        qs=[queries[i] for i in indices];sc={j:scores[i] for j,i in enumerate(indices)}
        grouped[name]=evaluate_tail_ranking(qs,sc,truths,candidates).to_dict()
    result.update(groups=grouped,scope=scope,candidate_count=len(candidates),horizon=horizon,
                  ties='optimistic among supplied finite scores; missing gold is worst filtered rank',
                  C10_history='full train+valid with t_history < t_query')
    return result


def binary_metrics(labels,scores,threshold=.5):
    if len(labels)!=len(scores) or not labels:raise ValueError('Need equally sized nonempty labels and scores')
    if any(y not in [0,1] for y in labels) or any(not math.isfinite(s) for s in scores):raise ValueError('Invalid labels/scores')
    predictions=[int(s>=threshold) for s in scores]
    tp=sum(y==p==1 for y,p in zip(labels,predictions));fp=sum(y==0 and p==1 for y,p in zip(labels,predictions))
    fn=sum(y==1 and p==0 for y,p in zip(labels,predictions));pos=sum(labels);neg=len(labels)-pos
    ordered=sorted(zip(scores,labels));rank_sum=0.;i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
        rank_sum+=(i+1+j)/2*sum(y for _,y in ordered[i:j]);i=j
    auc=(rank_sum-pos*(pos+1)/2)/(pos*neg) if pos and neg else None
    return dict(count=len(labels),auc=auc,accuracy=sum(y==p for y,p in zip(labels,predictions))/len(labels),
                precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,threshold=threshold)


def evaluate_activation_file(root,records,split='test',threshold=.5):
    keys={(int(q['s_id']),str(q['t'])) for q in records};active=set()
    if len(keys)!=len(records):raise ValueError('Duplicate source-time query')
    for fact in iter_facts(root,split):
        if (fact.s_id,fact.t) in keys:active.add((fact.s_id,fact.t))
    labels=[int((int(q['s_id']),str(q['t'])) in active) for q in records]
    scores=[float(q['score']) for q in records]
    return binary_metrics(labels,scores,threshold)


def evaluate_generation_file(root,records,split='test',ks=(100,1000,5000,10000)):
    keys={(int(q['s_id']),str(q['t'])) for q in records};gold=defaultdict(set)
    if len(keys)!=len(records):raise ValueError('Duplicate source-time query')
    for fact in iter_facts(root,split):
        if (fact.s_id,fact.t) in keys:gold[fact.s_id,fact.t].add((fact.r_id,fact.o_id))
    per_query=[]
    for q in records:
        key=int(q['s_id']),str(q['t']);pairs=[tuple(map(int,p)) for p in q['pairs']]
        if len(pairs)!=len(set(pairs)):raise ValueError('Duplicate generated pair')
        row=dict(s_id=key[0],t=key[1],gold_count=len(gold[key]),prediction_count=len(pairs))
        for k in ks:row[f'recall_at_{k}']=len(set(pairs[:k])&gold[key])/len(gold[key]) if gold[key] else None
        per_query.append(row)
    positive=[x for x in per_query if x['gold_count']]
    return dict(query_count=len(records),positive_query_count=len(positive),negative_query_count=len(records)-len(positive),
                **{f'macro_recall_at_{k}':sum(x[f'recall_at_{k}'] for x in positive)/len(positive) if positive else None for k in ks},by_query=per_query)
