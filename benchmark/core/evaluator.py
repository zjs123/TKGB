from __future__ import annotations
import math
from collections import defaultdict
from typing import Dict, Iterable, Mapping, Sequence
from .schema import EvaluationResult, TimestampFact

def evaluate_tail_ranking(test_facts: Sequence[TimestampFact], scores_by_index: Mapping[int, Mapping[int, float]], all_true_facts: Iterable[TimestampFact], candidate_entity_ids: Iterable[int]) -> EvaluationResult:
    candidates = sorted({int(x) for x in candidate_entity_ids})
    true_objects = defaultdict(set)
    for fact in all_true_facts:
        true_objects[fact.s_id, fact.r_id, fact.t].add(fact.o_id)
    ranks = []
    for (index, fact) in enumerate(test_facts):
        scores = scores_by_index.get(index, {})
        if fact.o_id not in candidates:
            raise ValueError('Gold object is outside the declared candidate space')
        if any(not math.isfinite(float(value)) for value in scores.values()):
            raise ValueError('Scores must be finite')
        target_score = float(scores.get(fact.o_id, float('-inf')))
        filtered = set(true_objects.get((fact.s_id, fact.r_id, fact.t), set()))
        filtered.discard(fact.o_id)
        if fact.o_id not in scores:
            ranks.append(len(set(candidates) - filtered))
            continue
        better_count = 0
        for candidate in candidates:
            if candidate == fact.o_id or candidate in filtered:
                continue
            if float(scores.get(candidate, float('-inf'))) > target_score:
                better_count += 1
        ranks.append(better_count + 1)
    count = len(ranks)
    if count == 0:
        return EvaluationResult(count=0, ranks=[], mrr=0.0, hits_at_1=0.0, hits_at_3=0.0, hits_at_10=0.0)
    return EvaluationResult(count=count, ranks=ranks, mrr=sum((1.0 / rank for rank in ranks)) / count, hits_at_1=sum((rank <= 1 for rank in ranks)) / count, hits_at_3=sum((rank <= 3 for rank in ranks)) / count, hits_at_10=sum((rank <= 10 for rank in ranks)) / count)
