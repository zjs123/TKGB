"""Model-independent horizon and context protocols with blind score callbacks."""
from dataclasses import dataclass
import random
from ..core.evaluator import evaluate_tail_ranking
from ..core.experiment_protocols import horizon_cutoff,visible_history_for_query
from ..core.context_transforms import transform_dataset_for_context,condition_uses_text


@dataclass(frozen=True)
class TailQuery:
    s_id:int
    r_id:int
    t:str
    candidate_entity_ids:tuple[int,...]
    history_cutoff:str|None=None


def evaluate_horizons(dataset,score_query,granularity,horizons=(1,3,10)):
    """score_query(TailQuery, visible_history) returns finite object scores.

    Gold objects are never passed to this callback. Past test events can become
    visible strictly before the declared horizon cutoff, never at query time.
    """
    ids=tuple(dataset.entity_ids());results={}
    for h in horizons:
        if h<1:raise ValueError('Forecast horizons must be positive')
        rows={};facts=[]
        for fact in dataset.test:
            cutoff=horizon_cutoff(fact.t,h,granularity)
            visible=visible_history_for_query(dataset,fact.t,h,granularity)
            if not visible:continue
            query=TailQuery(fact.s_id,fact.r_id,fact.t,ids,cutoff)
            rows[len(facts)]=score_query(query,tuple(visible));facts.append(fact)
        results[str(h)]=evaluate_tail_ranking(facts,rows,dataset.all_timestamp_facts(),ids).to_dict()
    return results


def evaluate_contexts(dataset,fit_callback,granularity,conditions=('timestamp_only','description','interval','description+interval')):
    """fit_callback receives train/valid/context only and returns score_query.

    No model is bundled. fit_callback(train=..., valid=..., entities=...,
    relations=..., condition=...) controls whether/how training uses descriptions.
    Test labels are held by the evaluator and never passed to fitting/scoring.
    """
    result={}
    for condition in conditions:
        supported={'timestamp_only','description','interval','description+interval','shuffled_description','expired_interval','shuffled_interval'}
        if condition not in supported:raise NotImplementedError('Context condition requires an explicit external transform: '+condition)
        transformed=transform_dataset_for_context(dataset,condition,granularity)
        view=transformed.dataset
        entities={i:dict(r) for i,r in view.entities.items()}
        if condition=='shuffled_description':
            ids=sorted(entities);values=[entities[i].get('description',entities[i].get('short_description','')) for i in ids]
            random.Random(0).shuffle(values)
            for i,value in zip(ids,values):entities[i].update(description=value,short_description=value)
        if not condition_uses_text(condition):
            for row in entities.values():
                for k in ['description','short_description']:row.pop(k,None)
        scorer=fit_callback(train=tuple(view.train),valid=tuple(view.valid),entities=entities,
                            relations={i:dict(r) for i,r in view.relations.items()},condition=condition)
        ids=tuple(view.entity_ids())
        scores={i:scorer(TailQuery(f.s_id,f.r_id,f.t,ids)) for i,f in enumerate(view.test)}
        result[condition]=dict(context=transformed.report,
                              evaluation=evaluate_tail_ranking(view.test,scores,view.all_timestamp_facts(),ids).to_dict())
    return result
