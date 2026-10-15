from __future__ import annotations
import hashlib
from collections import defaultdict
from datetime import date
from typing import Any, Dict, Iterable, List, Mapping, Sequence
GENERATION_RECALL_KS: tuple[int, ...] = (100, 1000, 5000, 10000)
DEFAULT_GENERATION_TIMESTAMPS = 10
DEFAULT_ACTIVE_SOURCES_PER_TIMESTAMP = 50

def normalize_recall_ks(recall_ks: Sequence[int] | None=None) -> tuple[int, ...]:
    values = GENERATION_RECALL_KS if recall_ks is None else tuple((int(value) for value in recall_ks))
    cleaned = sorted({value for value in values if value > 0})
    if not cleaned:
        raise ValueError('At least one positive generation recall K is required.')
    return tuple(cleaned)

def summarize_generation_recall(gold_by_t: Mapping[str, Iterable[Sequence[int] | tuple[int, int, int]]], predictions_by_t: Mapping[str, Sequence[Mapping[str, Any]]], recall_ks: Sequence[int] | None=None) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compute timestamp-level Recall@K for generated (s,r,o) facts.

    For each timestamp t, Recall@K is |TopK(t) intersect G_t| / |G_t|.
    The public `generation_recall_at_K` fields are macro-averages over
    timestamps, which keeps each forecast timestamp equally weighted.
    """
    ks = normalize_recall_ks(recall_ks)
    timestamps = sorted((str(timestamp) for timestamp in gold_by_t))
    total_gold = 0
    total_hits = defaultdict(int)
    macro_recall = defaultdict(float)
    by_time_rows: List[Dict[str, Any]] = []
    for timestamp in timestamps:
        gold = {_normalize_triple(value) for value in gold_by_t.get(timestamp, [])}
        predictions = [_normalize_triple(row.get('fact', ())) for row in predictions_by_t.get(timestamp, [])]
        gold_count = len(gold)
        total_gold += gold_count
        row: Dict[str, Any] = {'t': timestamp, 'gold_count': gold_count, 'prediction_count': len(predictions)}
        for k in ks:
            predicted_top_k = set(predictions[:k])
            hits = len(predicted_top_k & gold)
            recall = hits / gold_count if gold_count else 0.0
            total_hits[k] += hits
            macro_recall[k] += recall
            row[f'hits_at_{k}'] = hits
            row[f'recall_at_{k}'] = recall
        by_time_rows.append(row)
    timestamp_count = len(timestamps)
    summary: Dict[str, Any] = {'generation_metric': 'macro_recall_at_k_by_timestamp', 'recall_ks': list(ks), 'timestamp_count': timestamp_count, 'total_gold_count': total_gold, 'total_hits_at': {str(k): int(total_hits[k]) for k in ks}, 'micro_recall_at': {str(k): total_hits[k] / total_gold if total_gold else 0.0 for k in ks}}
    for k in ks:
        summary[f'generation_recall_at_{k}'] = macro_recall[k] / timestamp_count if timestamp_count else 0.0
    return (summary, by_time_rows)

def generation_by_time_fieldnames(recall_ks: Sequence[int] | None=None) -> List[str]:
    fields = ['t', 'gold_count', 'prediction_count']
    for k in normalize_recall_ks(recall_ks):
        fields.extend([f'hits_at_{k}', f'recall_at_{k}'])
    return fields

def build_source_conditioned_generation_queries(test_facts: Sequence[Any], entity_ids: Sequence[int], seed: int=0, max_timestamps: int=DEFAULT_GENERATION_TIMESTAMPS, max_active_sources_per_timestamp: int=DEFAULT_ACTIVE_SOURCES_PER_TIMESTAMP, history_facts: Sequence[Any] | None=None, negative_sampling: str='hard_inactive', inactivity_lookback_steps: int | None=None, granularity: str='day') -> tuple[Dict[tuple[int, str], set[tuple[int, int]]], Dict[tuple[int, str], str], Dict[str, Any]]:
    selected_timestamps = set(select_evenly_spaced_timestamps([str(fact.t) for fact in test_facts], max_timestamps=max_timestamps))
    all_facts_by_t: Dict[str, List[Any]] = defaultdict(list)
    for fact in test_facts:
        all_facts_by_t[str(fact.t)].append(fact)
    granularity = str(granularity or 'day').lower()
    lookback_steps = int(inactivity_lookback_steps) if inactivity_lookback_steps is not None else default_inactivity_lookback_steps(granularity)
    history_activity = _build_source_activity_index(test_facts if history_facts is None else history_facts, granularity)
    gold_by_query: Dict[tuple[int, str], set[tuple[int, int]]] = defaultdict(set)
    source_type: Dict[tuple[int, str], str] = {}
    selected_positive_sources_by_t: Dict[str, set[int]] = {}
    all_positive_sources_by_t: Dict[str, set[int]] = {}
    evaluated_fact_count = 0
    total_test_fact_count = len(test_facts)
    for timestamp in sorted(selected_timestamps):
        facts_at_t = all_facts_by_t.get(timestamp, [])
        source_counts: Dict[int, int] = defaultdict(int)
        for fact in facts_at_t:
            source_counts[int(fact.s_id)] += 1
        all_positive_sources_by_t[timestamp] = set(source_counts)
        selected_sources = {source for (source, _count) in sorted(source_counts.items(), key=lambda item: (-item[1], item[0]))[:max(0, int(max_active_sources_per_timestamp))]}
        selected_positive_sources_by_t[timestamp] = selected_sources
        for fact in facts_at_t:
            s_id = int(fact.s_id)
            if s_id not in selected_sources:
                continue
            gold_by_query[s_id, timestamp].add((int(fact.r_id), int(fact.o_id)))
            source_type[s_id, timestamp] = 'positive'
            evaluated_fact_count += 1
    all_entities = sorted({int(value) for value in entity_ids})
    hard_negative_shortage_count = 0
    selected_negative_count = 0
    for (timestamp, positive_sources) in sorted(selected_positive_sources_by_t.items()):
        if negative_sampling == 'hard_inactive':
            (negative_sources, shortage) = _sample_hard_inactive_negative_sources(timestamp=timestamp, positive_sources=positive_sources, all_gt_sources=all_positive_sources_by_t.get(timestamp, set()), all_entities=all_entities, history_activity=history_activity, lookback_steps=lookback_steps, granularity=granularity)
            hard_negative_shortage_count += shortage
        else:
            raise ValueError(f'Unsupported negative_sampling: {negative_sampling}')
        selected_negative_count += len(negative_sources)
        for s_id in negative_sources:
            key = (int(s_id), timestamp)
            gold_by_query.setdefault(key, set())
            source_type[key] = 'negative'
    selected_source_count = sum((len(values) for values in selected_positive_sources_by_t.values()))
    available_source_count = sum((len(values) for values in all_positive_sources_by_t.values()))
    selection_summary = {'timestamp_selection': 'evenly_spaced', 'max_generation_timestamps': int(max_timestamps), 'evaluated_timestamp_count': len(selected_timestamps), 'active_sources_per_timestamp': int(max_active_sources_per_timestamp), 'selected_positive_source_count': selected_source_count, 'available_positive_source_count_in_selected_timestamps': available_source_count, 'source_coverage_fact_rate': selected_source_count / available_source_count if available_source_count else 0.0, 'evaluated_fact_count': evaluated_fact_count, 'total_test_fact_count': total_test_fact_count, 'evaluated_fact_coverage_rate': evaluated_fact_count / total_test_fact_count if total_test_fact_count else 0.0, 'negative_sampling_protocol': 'hard_inactive_activity_matched', 'negative_inactivity_lookback_steps': lookback_steps, 'hard_negative_shortage_count': hard_negative_shortage_count, 'selected_negative_source_count': selected_negative_count}
    return (dict(gold_by_query), source_type, selection_summary)

def summarize_source_conditioned_generation_recall(gold_by_query: Mapping[tuple[int, str], Iterable[Sequence[int] | tuple[int, int]]], predictions_by_query: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]], source_type_by_query: Mapping[tuple[int, str], str], recall_ks: Sequence[int] | None=None) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    ks = normalize_recall_ks(recall_ks)
    total_gold = 0
    total_hits = defaultdict(int)
    macro_recall = defaultdict(float)
    positive_count = 0
    negative_count = 0
    by_query_rows: List[Dict[str, Any]] = []
    for (s_id, timestamp) in sorted(gold_by_query, key=lambda key: (key[1], key[0])):
        query_key = (int(s_id), str(timestamp))
        source_type = str(source_type_by_query.get(query_key, 'positive'))
        gold = {_normalize_pair(value) for value in gold_by_query.get(query_key, [])}
        predictions = [_normalize_prediction_pair(row.get('fact', ())) for row in predictions_by_query.get(query_key, [])]
        gold_count = len(gold)
        if source_type == 'negative':
            negative_count += 1
        else:
            positive_count += 1
            total_gold += gold_count
        row: Dict[str, Any] = {'t': str(timestamp), 's_id': int(s_id), 'source_type': source_type, 'gold_count': gold_count, 'prediction_count': len(predictions)}
        scores = [float(row.get('score', 0.0)) for row in predictions_by_query.get(query_key, [])]
        row['top1_score'] = scores[0] if scores else 0.0
        row['top10_score_mean'] = sum(scores[:10]) / min(10, len(scores)) if scores else 0.0
        row['top100_score_mean'] = sum(scores[:100]) / min(100, len(scores)) if scores else 0.0
        for k in ks:
            hits = len(set(predictions[:k]) & gold)
            recall = hits / gold_count if gold_count else 0.0
            row[f'hits_at_{k}'] = hits
            row[f'recall_at_{k}'] = recall
            if source_type != 'negative':
                total_hits[k] += hits
                macro_recall[k] += recall
        by_query_rows.append(row)
    summary: Dict[str, Any] = {'generation_metric': 'source_conditioned_macro_recall_at_k', 'recall_ks': list(ks), 'source_query_count': positive_count + negative_count, 'positive_source_query_count': positive_count, 'negative_source_query_count': negative_count, 'total_gold_count': total_gold, 'total_hits_at': {str(k): int(total_hits[k]) for k in ks}, 'micro_recall_at': {str(k): total_hits[k] / total_gold if total_gold else 0.0 for k in ks}}
    positive_top1 = [float(row['top1_score']) for row in by_query_rows if row['source_type'] != 'negative']
    negative_top1 = [float(row['top1_score']) for row in by_query_rows if row['source_type'] == 'negative']
    positive_top10 = [float(row['top10_score_mean']) for row in by_query_rows if row['source_type'] != 'negative']
    negative_top10 = [float(row['top10_score_mean']) for row in by_query_rows if row['source_type'] == 'negative']
    positive_top100 = [float(row['top100_score_mean']) for row in by_query_rows if row['source_type'] != 'negative']
    negative_top100 = [float(row['top100_score_mean']) for row in by_query_rows if row['source_type'] == 'negative']
    summary.update({'source_uncertainty_score': 'top10_score_mean', 'positive_top1_score_mean': _mean(positive_top1), 'negative_top1_score_mean': _mean(negative_top1), 'positive_top10_score_mean': _mean(positive_top10), 'negative_top10_score_mean': _mean(negative_top10), 'positive_top100_score_mean': _mean(positive_top100), 'negative_top100_score_mean': _mean(negative_top100), 'positive_negative_top1_gap': _mean(positive_top1) - _mean(negative_top1), 'positive_negative_top10_gap': _mean(positive_top10) - _mean(negative_top10), 'positive_negative_top100_gap': _mean(positive_top100) - _mean(negative_top100), 'source_uncertainty_auc': _binary_auc(positive_top10, negative_top10), 'hard_negative_source_uncertainty_auc_top10': _binary_auc(positive_top10, negative_top10), 'source_uncertainty_auc_top1_diagnostic': _binary_auc(positive_top1, negative_top1), 'source_uncertainty_auc_top100_diagnostic': _binary_auc(positive_top100, negative_top100)})
    for k in ks:
        summary[f'generation_recall_at_{k}'] = macro_recall[k] / positive_count if positive_count else 0.0
    return (summary, by_query_rows)

def generation_by_query_fieldnames(recall_ks: Sequence[int] | None=None) -> List[str]:
    fields = ['t', 's_id', 'source_type', 'gold_count', 'prediction_count', 'top1_score', 'top10_score_mean', 'top100_score_mean']
    for k in normalize_recall_ks(recall_ks):
        fields.extend([f'hits_at_{k}', f'recall_at_{k}'])
    return fields

def source_score_distribution_fieldnames() -> List[str]:
    return ['t', 's_id', 'source_type', 'negative_sampling_type', 'rank', 'score', 'r_id', 'o_id']

def source_score_distribution_rows(predictions_by_query: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]], source_type_by_query: Mapping[tuple[int, str], str], top_n: int=10, negative_sampling_type: str='hard_inactive_activity_matched') -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    limit = max(0, int(top_n))
    for (s_id, timestamp) in sorted(predictions_by_query, key=lambda key: (key[1], key[0])):
        source_type = str(source_type_by_query.get((int(s_id), str(timestamp)), 'positive'))
        for (rank, row) in enumerate(list(predictions_by_query.get((s_id, timestamp), []))[:limit], start=1):
            fact = _normalize_prediction_pair(row.get('fact', ()))
            out.append({'t': str(timestamp), 's_id': int(s_id), 'source_type': source_type, 'negative_sampling_type': negative_sampling_type, 'rank': rank, 'score': float(row.get('score', 0.0)), 'r_id': int(fact[0]), 'o_id': int(fact[1])})
    return out

def default_inactivity_lookback_steps(granularity: str) -> int:
    value = str(granularity or 'day').lower()
    if value == 'year':
        return 2
    if value == 'month':
        return 6
    return 180

def select_evenly_spaced_timestamps(timestamps: Sequence[str], max_timestamps: int=DEFAULT_GENERATION_TIMESTAMPS) -> List[str]:
    values = sorted({str(value) for value in timestamps})
    limit = max(0, int(max_timestamps))
    if limit <= 0 or len(values) <= limit:
        return values
    if limit == 1:
        return [values[0]]
    selected = []
    last = len(values) - 1
    for index in range(limit):
        pos = round(index * last / (limit - 1))
        selected.append(values[pos])
    return selected

def _sample_hard_inactive_negative_sources(timestamp: str, positive_sources: set[int], all_gt_sources: set[int], all_entities: Sequence[int], history_activity: Mapping[int, Mapping[int, int]], lookback_steps: int, granularity: str) -> tuple[List[int], int]:
    current_step = _time_step_index(timestamp, granularity)
    if current_step is None or not positive_sources:
        return ([], len(positive_sources))
    historical_counts: Dict[int, int] = defaultdict(int)
    recent_sources: set[int] = set()
    recent_floor = current_step - max(0, int(lookback_steps))
    for (source, counts_by_step) in history_activity.items():
        for (fact_step, count) in counts_by_step.items():
            if fact_step >= current_step:
                continue
            historical_counts[int(source)] += int(count)
            if fact_step >= recent_floor:
                recent_sources.add(int(source))
    idle_candidates = [entity_id for entity_id in all_entities if entity_id not in all_gt_sources and entity_id not in recent_sources and (historical_counts.get(entity_id, 0) > 0)]
    unused = set(idle_candidates)
    selected: List[int] = []
    for source in sorted(positive_sources):
        if not unused:
            break
        source_activity = historical_counts.get(source, 0)
        candidate = min(unused, key=lambda entity_id: (abs(historical_counts.get(entity_id, 0) - source_activity), -historical_counts.get(entity_id, 0), entity_id))
        unused.remove(candidate)
        selected.append(candidate)
    return (selected, max(0, len(positive_sources) - len(selected)))

def _build_source_activity_index(history_facts: Iterable[Any], granularity: str) -> Dict[int, Dict[int, int]]:
    activity: Dict[int, Dict[int, int]] = defaultdict(lambda : defaultdict(int))
    for fact in history_facts:
        fact_step = _time_step_index(str(fact.t), granularity)
        if fact_step is None:
            continue
        activity[int(fact.s_id)][fact_step] += 1
    return {source: dict(counts_by_step) for (source, counts_by_step) in activity.items()}

def _time_step_index(value: str, granularity: str) -> int | None:
    parsed = _parse_date(value)
    if parsed is None:
        return None
    unit = str(granularity or 'day').lower()
    if unit == 'year':
        return parsed.year
    if unit == 'month':
        return parsed.year * 12 + parsed.month
    return parsed.toordinal()

def _parse_date(value: str) -> date | None:
    text = str(value or '').strip()
    if not text:
        return None
    parts = text.split('-')
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 and parts[1] else 1
        day = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        return date(year, max(1, min(12, month)), max(1, min(31, day)))
    except Exception:
        return None

def _normalize_triple(value: Sequence[int] | tuple[int, int, int]) -> tuple[int, int, int]:
    if len(value) < 3:
        raise ValueError(f'Generated or gold fact must contain at least three ids: {value!r}')
    return (int(value[0]), int(value[1]), int(value[2]))

def _normalize_pair(value: Sequence[int] | tuple[int, int]) -> tuple[int, int]:
    if len(value) < 2:
        raise ValueError(f'Gold relation-object pair must contain at least two ids: {value!r}')
    return (int(value[0]), int(value[1]))

def _normalize_prediction_pair(value: Sequence[int] | tuple[int, ...]) -> tuple[int, int]:
    if len(value) >= 3:
        return (int(value[1]), int(value[2]))
    return _normalize_pair(value)

def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f'{int(seed)}|{value}'.encode('utf-8')).hexdigest()
    return int(digest[:16], 16)

def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def _binary_auc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    if not positive_scores or not negative_scores:
        return 0.0
    wins = 0.0
    total = len(positive_scores) * len(negative_scores)
    for pos in positive_scores:
        for neg in negative_scores:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total if total else 0.0
