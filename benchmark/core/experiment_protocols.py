from __future__ import annotations
from datetime import date, timedelta
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence
from .evaluator import evaluate_tail_ranking
from .schema import BenchmarkDataset, TimestampFact
ScoreRows = Mapping[int, Mapping[int, float]]
MutableScoreRows = Dict[int, Dict[int, float]]

def visible_history_for_query(dataset: BenchmarkDataset, query_time: str, horizon: int, granularity: str) -> List[TimestampFact]:
    cutoff = horizon_cutoff(query_time, horizon, granularity)
    return [fact for fact in dataset.train + dataset.valid + dataset.test if _date_tuple(fact.t) <= _date_tuple(cutoff)]

def horizon_cutoff(query_time: str, horizon: int, granularity: str) -> str:
    granularity = granularity.strip().lower()
    if horizon < 0:
        raise ValueError('horizon must be non-negative')
    value = _parse_date(query_time)
    if granularity == 'day':
        return _format_date(value - timedelta(days=horizon))
    if granularity == 'month':
        year = value.year
        month = value.month - horizon
        while month <= 0:
            month += 12
            year -= 1
        day = min(value.day, _days_in_month(year, month))
        return _format_date(date(year, month, day))
    if granularity == 'year':
        year = value.year - horizon
        day = min(value.day, _days_in_month(year, value.month))
        return _format_date(date(year, value.month, day))
    raise ValueError(f'Unknown granularity: {granularity}')

def exhaustive_timestamp_generation(entity_ids: Sequence[int], relation_ids: Sequence[int], timestamps: Sequence[str], score_triple: Callable[[int, int, int, str], float], top_k: int) -> Dict[str, List[Dict[str, Any]]]:
    if top_k <= 0:
        raise ValueError('top_k must be positive')
    out: Dict[str, List[Dict[str, Any]]] = {}
    for timestamp in timestamps:
        rows: List[Dict[str, Any]] = []
        for s_id in entity_ids:
            for r_id in relation_ids:
                for o_id in entity_ids:
                    rows.append({'fact': [int(s_id), int(r_id), int(o_id), timestamp], 'score': float(score_triple(int(s_id), int(r_id), int(o_id), timestamp))})
        rows.sort(key=lambda row: (-float(row['score']), row['fact']))
        out[timestamp] = rows[:top_k]
    return out

def _parse_date(value: str) -> date:
    parts = str(value).split('-')
    year = int(parts[0])
    month = int(parts[1]) if len(parts) > 1 else 1
    day = int(parts[2]) if len(parts) > 2 else 1
    return date(year, month, day)

def _format_date(value: date) -> str:
    return value.isoformat()

def _date_tuple(value: str) -> tuple[int, int, int]:
    parsed = _parse_date(value)
    return (parsed.year, parsed.month, parsed.day)

def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - timedelta(days=1)).day
