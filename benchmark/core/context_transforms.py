from __future__ import annotations
import copy
import hashlib
import math
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Mapping, Sequence
import numpy as np
from .schema import BenchmarkDataset, IntervalFact, TimestampFact, entity_name, relation_name
TEXT_CONTEXT_CONDITIONS = {'description', 'description+interval', 'shuffled_description', 'wrong_domain_description'}
INTERVAL_CONTEXT_CONDITIONS = {'interval', 'description+interval', 'expired_interval', 'shuffled_interval'}

@dataclass(frozen=True)
class ContextTransformedDataset:
    dataset: BenchmarkDataset
    report: Dict[str, Any]

def condition_uses_text(condition: str) -> bool:
    return str(condition) in TEXT_CONTEXT_CONDITIONS

def condition_uses_interval_expansion(condition: str) -> bool:
    return str(condition) in INTERVAL_CONTEXT_CONDITIONS

def transform_dataset_for_context(dataset: BenchmarkDataset, condition: str, granularity: str) -> ContextTransformedDataset:
    condition = str(condition)
    train_before = len(dataset.train)
    train_rows = list(dataset.train)
    expansion_report = {'status': 'not_requested', 'mode': 'none', 'expanded_fact_count': 0, 'source_interval_count': 0, 'deduplicated_fact_count': 0, 'granularity': granularity}
    if condition_uses_interval_expansion(condition):
        (expanded, expansion_report) = expand_train_intervals(dataset, condition, granularity)
        existing = {fact.id_quad() for fact in train_rows}
        deduplicated = []
        for fact in expanded:
            if fact.id_quad() in existing:
                continue
            existing.add(fact.id_quad())
            deduplicated.append(fact)
        expansion_report['deduplicated_fact_count'] = len(deduplicated)
        train_rows.extend(deduplicated)
    transformed = BenchmarkDataset(root=dataset.root, train=train_rows, valid=list(dataset.valid), test=list(dataset.test), intervals={split: list(rows) for (split, rows) in dataset.intervals.items()}, entities=copy.deepcopy(dataset.entities), relations=copy.deepcopy(dataset.relations))
    report = {'condition': condition, 'train_fact_count_before_context': train_before, 'train_fact_count_after_context': len(transformed.train), 'text_initialization': {'status': 'requested' if condition_uses_text(condition) else 'not_requested'}, 'interval_expansion': expansion_report}
    return ContextTransformedDataset(dataset=transformed, report=report)

def expand_train_intervals(dataset: BenchmarkDataset, condition: str, granularity: str) -> tuple[List[TimestampFact], Dict[str, Any]]:
    mode = _interval_mode(condition)
    train_bounds = _split_time_bounds(dataset.train)
    source_intervals = list(dataset.intervals.get('train', []))
    if train_bounds is None or not source_intervals:
        return ([], {'status': 'skipped_empty_train_or_intervals', 'mode': mode, 'expanded_fact_count': 0, 'source_interval_count': len(source_intervals), 'deduplicated_fact_count': 0, 'granularity': granularity})
    (train_start, train_end) = train_bounds
    entity_ids = dataset.entity_ids()
    expanded: List[TimestampFact] = []
    for (index, interval) in enumerate(source_intervals):
        current = _interval_for_mode(interval, mode, entity_ids, dataset)
        if current is None:
            continue
        (start, end) = _active_window_for_mode(current, mode, train_start, train_end, granularity)
        if start is None or end is None or start > end:
            continue
        for timestamp in _time_values_between(start, end, granularity):
            expanded.append(TimestampFact(split='train', s_id=current.s_id, r_id=current.r_id, o_id=current.o_id, source_name=current.source_name, relation_label=current.relation_label, target_name=current.target_name, t=timestamp, timestamp_source=f'interval_expanded:{mode}', fact_hash=f'interval_expanded:{mode}:{index}:{current.s_id}:{current.r_id}:{current.o_id}:{timestamp}', metadata={'timestamp_source': 'interval_expanded', 'interval_mode': mode, 'source_interval_hash': current.fact_hash, 'source_t_start': current.t_start, 'source_t_end': current.t_end}))
    return (expanded, {'status': 'applied', 'mode': mode, 'expanded_fact_count': len(expanded), 'source_interval_count': len(source_intervals), 'deduplicated_fact_count': 0, 'granularity': granularity})

def _interval_mode(condition: str) -> str:
    if condition == 'expired_interval':
        return 'expired'
    if condition == 'shuffled_interval':
        return 'shuffled'
    return 'active'

def _interval_for_mode(interval: IntervalFact, mode: str, entity_ids: Sequence[int], dataset: BenchmarkDataset) -> IntervalFact | None:
    if mode != 'shuffled':
        return interval
    if not entity_ids:
        return interval
    candidates = sorted((int(x) for x in entity_ids))
    try:
        pos = candidates.index(interval.o_id)
        replacement = candidates[(pos + 1) % len(candidates)]
    except ValueError:
        replacement = candidates[0]
    if len(candidates) > 1 and replacement == interval.o_id:
        replacement = candidates[1]
    return IntervalFact(split=interval.split, s_id=interval.s_id, r_id=interval.r_id, o_id=replacement, source_name=interval.source_name, relation_label=interval.relation_label, target_name=entity_name(replacement, dataset.entities), t_start=interval.t_start, t_end=interval.t_end, fact_hash=interval.fact_hash)

def _active_window_for_mode(interval: IntervalFact, mode: str, train_start: date, train_end: date, granularity: str) -> tuple[date | None, date | None]:
    if mode == 'expired':
        if not str(interval.t_end).strip():
            return (None, None)
        end = _parse_date(str(interval.t_end))
        if end is None:
            return (None, None)
        start = _step_date(end, granularity)
        return (max(start, train_start), train_end)
    start = _parse_date(str(interval.t_start))
    end = _parse_date(str(interval.t_end)) if str(interval.t_end).strip() else train_end
    if start is None:
        return (None, None)
    if end is None:
        end = train_end
    return (max(_period_floor(start, granularity), train_start), min(_period_floor(end, granularity), train_end))

def _time_values_between(start: date, end: date, granularity: str) -> List[str]:
    values = []
    current = _period_floor(start, granularity)
    end = _period_floor(end, granularity)
    while current <= end:
        values.append(_format_date(current, granularity))
        current = _step_date(current, granularity)
    return values

def _split_time_bounds(facts: Sequence[TimestampFact]) -> tuple[date, date] | None:
    values = [_parse_date(fact.t) for fact in facts]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return (min(values), max(values))

def _parse_date(value: str) -> date | None:
    text = str(value or '').strip()
    if not text:
        return None
    parts = text.split('-')
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 and parts[1] else 1
        day = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        month = min(max(month, 1), 12)
        day = min(max(day, 1), 28 if month == 2 else 31)
        return date(year, month, day)
    except Exception:
        return None

def _period_floor(value: date, granularity: str) -> date:
    if granularity == 'year':
        return date(value.year, 1, 1)
    if granularity == 'month':
        return date(value.year, value.month, 1)
    return value

def _step_date(value: date, granularity: str) -> date:
    value = _period_floor(value, granularity)
    if granularity == 'year':
        return date(value.year + 1, 1, 1)
    if granularity == 'month':
        year = value.year + (1 if value.month == 12 else 0)
        month = 1 if value.month == 12 else value.month + 1
        return date(year, month, 1)
    return value + timedelta(days=1)

def _format_date(value: date, granularity: str) -> str:
    if granularity == 'year':
        return f'{value.year:04d}'
    if granularity == 'month':
        return f'{value.year:04d}-{value.month:02d}'
    return f'{value.year:04d}-{value.month:02d}-{value.day:02d}'

def _entity_text(entity_id: int, dataset: BenchmarkDataset) -> str:
    row = dataset.entities.get(entity_id, {})
    values = [entity_name(entity_id, dataset.entities)]
    for key in ('description', 'short_description', 'wikidata_label', 'qid'):
        value = str(row.get(key, '')).strip()
        if value and value not in values:
            values.append(value)
    return ' '.join(values)

def _relation_text(relation_id: int, dataset: BenchmarkDataset) -> str:
    row = dataset.relations.get(relation_id, {})
    values = [relation_name(relation_id, dataset.relations)]
    for key in ('description', 'label', 'name'):
        value = str(row.get(key, '')).strip()
        if value and value not in values:
            values.append(value)
    return ' '.join(values)
