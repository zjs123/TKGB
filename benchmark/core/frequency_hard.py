from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from .schema import TimestampFact

@dataclass(frozen=True)
class FrequencyHardSubset:
    facts: list[TimestampFact]
    original_indices: list[int]
    summary: dict[str, Any]

def build_frequency_hard_subset(facts: Sequence[TimestampFact], history_facts: Sequence[TimestampFact], top_k: int=10) -> FrequencyHardSubset:
    """Keep facts whose gold object is outside historical (s,r) and s top-k sets."""
    k = max(0, int(top_k))
    (sr_top, s_top) = _history_top_sets(history_facts, k)
    kept: list[TimestampFact] = []
    original_indices: list[int] = []
    easy_by_sr = 0
    easy_by_s = 0
    for (index, row) in enumerate(facts):
        sr_hit = row.o_id in sr_top.get((row.s_id, row.r_id), set())
        s_hit = row.o_id in s_top.get(row.s_id, set())
        if sr_hit:
            easy_by_sr += 1
        if s_hit:
            easy_by_s += 1
        if not sr_hit and (not s_hit):
            kept.append(row)
            original_indices.append(index)
    total = len(facts)
    hard_count = len(kept)
    return FrequencyHardSubset(facts=kept, original_indices=original_indices, summary={'frequency_hard_protocol': 'gold_object_not_in_history_sr_topk_or_s_topk_union', 'history_top_k': k, 'history_fact_count': len(history_facts), 'frequency_hard_count': hard_count, 'frequency_easy_count': total - hard_count, 'frequency_hard_ratio': hard_count / total if total else 0.0, 'frequency_easy_by_sr_topk_count': easy_by_sr, 'frequency_easy_by_s_topk_count': easy_by_s})

def remap_scores_for_subset(scores_by_index: Mapping[int, Mapping[int, float]], original_indices: Sequence[int]) -> dict[int, dict[int, float]]:
    """Re-index score rows so `evaluate_tail_ranking` can consume a subset."""
    return {new_index: {int(candidate): float(score) for (candidate, score) in scores_by_index.get(old_index, {}).items()} for (new_index, old_index) in enumerate(original_indices)}

def _history_top_sets(history_facts: Sequence[TimestampFact], top_k: int) -> tuple[dict[tuple[int, int], set[int]], dict[int, set[int]]]:
    if top_k <= 0:
        return ({}, {})
    sr_counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    s_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for row in history_facts:
        sr_counts[int(row.s_id), int(row.r_id)][int(row.o_id)] += 1
        s_counts[int(row.s_id)][int(row.o_id)] += 1
    sr_top = {key: set(_top_counter_keys(counter, top_k)) for (key, counter) in sr_counts.items()}
    s_top = {key: set(_top_counter_keys(counter, top_k)) for (key, counter) in s_counts.items()}
    return (sr_top, s_top)

def _top_counter_keys(counter: Counter[int], top_k: int) -> list[int]:
    return [key for (key, _count) in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:top_k]]
