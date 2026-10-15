from __future__ import annotations
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from .schema import IntervalFact
PROTOCOL_VERSION = 'causal_interval_context_v1'
DEFAULT_SOURCE_SPLITS = ('train', 'valid', 'test')

def dataset_name(dataset_path: Path) -> str:
    return f'{dataset_path.parent.name}_{dataset_path.name}'

def safe_timestamp_filename(timestamp: str) -> str:
    return str(timestamp).replace(':', '-').replace('/', '-').replace('\\', '-')

def normalize_time(value: str, granularity: str) -> str:
    raw = str(value or '').strip()
    if not raw:
        return ''
    granularity = str(granularity or 'day').lower()
    if granularity == 'year':
        return raw[:4] if len(raw) >= 4 else ''
    if granularity == 'month':
        return raw[:7] if len(raw) >= 7 else ''
    return raw[:10] if len(raw) >= 10 else ''

def _pick(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key, '')).strip()
        if value:
            return value
    return ''

def read_intervals(path: Path, split: str) -> list[IntervalFact]:
    out: list[IntervalFact] = []
    if not path.exists():
        return out
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if not str(row.get('s_id', '')).strip() or not str(row.get('r_id', '')).strip() or (not str(row.get('o_id', '')).strip()):
                continue
            out.append(IntervalFact(split=split, s_id=int(str(row['s_id']).strip()), r_id=int(str(row['r_id']).strip()), o_id=int(str(row['o_id']).strip()), source_name=_pick(row, 'source_name', 'head_name', 's_name'), relation_label=_pick(row, 'relation_label', 'property_label', 'r_label'), target_name=_pick(row, 'target_name', 'tail_name', 'o_name'), t_start=str(row.get('t_start', '')).strip(), t_end=str(row.get('t_end', '')).strip(), fact_hash=str(row.get('hash', '')).strip()))
    return out

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(dict(payload), f, ensure_ascii=False, indent=2, sort_keys=True)

def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + '\n')
            count += 1
    return count

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda : f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()

def load_interval_splits(dataset_path: Path, source_splits: Sequence[str]=DEFAULT_SOURCE_SPLITS) -> list[IntervalFact]:
    intervals: list[IntervalFact] = []
    for split in source_splits:
        intervals.extend(read_intervals(dataset_path / f'intervals_{split}.csv', split))
    return intervals

def visible_interval_context_rows(intervals: Sequence[IntervalFact], timestamp: str, granularity: str, target_fact_hash: str | None=None) -> list[dict[str, Any]]:
    tau = normalize_time(timestamp, granularity)
    if not tau:
        return []
    target_hash = str(target_fact_hash or '').strip()
    rows: list[dict[str, Any]] = []
    for interval in intervals:
        start = normalize_time(interval.t_start, granularity)
        if not start or start > tau:
            continue
        end = normalize_time(interval.t_end, granularity)
        is_target = bool(target_hash and target_hash == str(interval.fact_hash or '').strip())
        end_visible = bool(end and end < tau and (not is_target))
        rows.append({'split': interval.split, 's_id': int(interval.s_id), 'r_id': int(interval.r_id), 'o_id': int(interval.o_id), 'source_name': interval.source_name, 'relation_label': interval.relation_label, 'target_name': interval.target_name, 't_start': start, 't_end_visible': end if end_visible else '?', 't_end_is_visible': end_visible, 'fact_hash': interval.fact_hash})
    rows.sort(key=lambda row: (str(row['t_start']), int(row['s_id']), int(row['r_id']), int(row['o_id']), str(row['split']), str(row['fact_hash'])))
    return rows

def interval_table_rows(intervals: Sequence[IntervalFact], granularity: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (index, interval) in enumerate(intervals):
        rows.append({'interval_index': int(index), 'split': interval.split, 's_id': int(interval.s_id), 'r_id': int(interval.r_id), 'o_id': int(interval.o_id), 'source_name': interval.source_name, 'relation_label': interval.relation_label, 'target_name': interval.target_name, 't_start': normalize_time(interval.t_start, granularity), 't_end': normalize_time(interval.t_end, granularity), 'fact_hash': interval.fact_hash})
    rows.sort(key=lambda row: (str(row['t_start']), int(row['s_id']), int(row['r_id']), int(row['o_id']), str(row['split']), str(row['fact_hash'])))
    for (index, row) in enumerate(rows):
        row['interval_index'] = int(index)
    return rows

def visible_interval_context_rows_from_table(interval_rows: Sequence[Mapping[str, Any]], timestamp: str, granularity: str, target_fact_hash: str | None=None) -> list[dict[str, Any]]:
    tau = normalize_time(timestamp, granularity)
    if not tau:
        return []
    target_hash = str(target_fact_hash or '').strip()
    out: list[dict[str, Any]] = []
    for row in interval_rows:
        start = str(row.get('t_start', '')).strip()
        if not start or start > tau:
            continue
        end = str(row.get('t_end', '')).strip()
        is_target = bool(target_hash and target_hash == str(row.get('fact_hash', '')).strip())
        end_visible = bool(end and end < tau and (not is_target))
        out.append({'split': str(row.get('split', '')), 's_id': int(row['s_id']), 'r_id': int(row['r_id']), 'o_id': int(row['o_id']), 'source_name': str(row.get('source_name', '')), 'relation_label': str(row.get('relation_label', '')), 'target_name': str(row.get('target_name', '')), 't_start': start, 't_end_visible': end if end_visible else '?', 't_end_is_visible': end_visible, 'fact_hash': str(row.get('fact_hash', '')), 'interval_index': int(row.get('interval_index', -1))})
    return out

def collect_shared_context_timestamps(shared_sample_dir: Path) -> tuple[list[str], dict[str, int]]:
    timestamps: set[str] = set()
    counts: Counter[str] = Counter()
    generation_path = shared_sample_dir / 'exp4_generation_queries.jsonl'
    for row in read_jsonl(generation_path):
        t = str(row.get('t', '')).strip()
        if t:
            timestamps.add(t)
            counts['generation_query_timestamps'] += 1
    obsolescence_path = shared_sample_dir / 'obsolescence_samples.jsonl'
    for row in read_jsonl(obsolescence_path):
        for key in ('pre_times', 'post_times'):
            for t in row.get(key, []) or []:
                t = str(t).strip()
                if t:
                    timestamps.add(t)
                    counts[f'obsolescence_{key}'] += 1
    return (sorted(timestamps), dict(counts))

def build_causal_interval_context_cache(dataset_path: Path, output_root: Path, query_timestamps: Sequence[str], granularity: str, source_splits: Sequence[str]=DEFAULT_SOURCE_SPLITS, overwrite: bool=False) -> dict[str, Any]:
    dataset_key = dataset_name(dataset_path)
    cache_root = output_root / dataset_key
    manifest_path = cache_root / 'causal_interval_context_manifest.json'
    query_timestamps = sorted({normalize_time(t, granularity) for t in query_timestamps if normalize_time(t, granularity)})
    interval_files = [dataset_path / f'intervals_{split}.csv' for split in source_splits]
    hash_input = {'protocol': PROTOCOL_VERSION, 'dataset_key': dataset_key, 'dataset_path': str(dataset_path), 'time_granularity': granularity, 'source_splits': list(source_splits), 'query_timestamps': query_timestamps, 'interval_file_hashes': {path.name: file_sha256(path) for path in interval_files if path.exists()}}
    context_hash = payload_hash(hash_input)
    if manifest_path.exists() and (not overwrite):
        existing = json.loads(manifest_path.read_text(encoding='utf-8'))
        if existing.get('interval_context_hash') == context_hash:
            return existing
    intervals = load_interval_splits(dataset_path, source_splits)
    table_rows = interval_table_rows(intervals, granularity)
    interval_table_path = cache_root / 'interval_table.jsonl'
    write_jsonl(interval_table_path, table_rows)
    timestamp_rows: list[dict[str, Any]] = []
    total_context_rows = 0
    max_context_rows = 0
    min_context_rows = None
    for timestamp in query_timestamps:
        rows = visible_interval_context_rows_from_table(table_rows, timestamp, granularity)
        count = len(rows)
        timestamp_rows.append({'t': timestamp, 'context_row_count': int(count)})
        total_context_rows += count
        max_context_rows = max(max_context_rows, count)
        min_context_rows = count if min_context_rows is None else min(min_context_rows, count)
    manifest = {'protocol': PROTOCOL_VERSION, 'interval_context_hash': context_hash, 'manifest_path': str(manifest_path), 'dataset_key': dataset_key, 'dataset_path': str(dataset_path), 'time_granularity': granularity, 'source_splits': list(source_splits), 'target_interval_t_end_policy': 'always_hidden', 'non_target_t_end_policy': 'visible_only_if_t_end_before_scored_timestamp', 'cache_storage': 'single_interval_table_plus_timestamp_counts', 'interval_table_path': str(interval_table_path), 'interval_table_hash': file_sha256(interval_table_path), 'query_timestamp_count': len(query_timestamps), 'interval_count': len(intervals), 'total_context_rows': int(total_context_rows), 'min_context_rows_per_timestamp': int(min_context_rows or 0), 'max_context_rows_per_timestamp': int(max_context_rows), 'timestamp_contexts': timestamp_rows, 'hash_input': hash_input}
    write_json(manifest_path, manifest)
    return manifest

def load_context_for_timestamp(manifest: Mapping[str, Any], timestamp: str) -> list[dict[str, Any]]:
    interval_rows = read_jsonl(Path(str(manifest['interval_table_path'])))
    return visible_interval_context_rows_from_table(interval_rows, str(timestamp), str(manifest.get('time_granularity', 'day')))

def context_metadata(report_manifest: Mapping[str, Any], support: str) -> dict[str, Any]:
    return {'interval_context_enabled': True, 'interval_context_support': support, 'interval_context_protocol': PROTOCOL_VERSION, 'interval_context_source_splits': list(report_manifest.get('source_splits', [])), 'interval_context_hash': str(report_manifest.get('interval_context_hash', '')), 'interval_context_manifest_path': str(report_manifest.get('manifest_path', ''))}
