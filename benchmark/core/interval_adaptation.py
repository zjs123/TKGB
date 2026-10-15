from __future__ import annotations
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from .generation_metrics import build_source_conditioned_generation_queries
from .schema import IntervalFact, TimestampFact
from .sample_io import read_eval_sample, write_eval_sample
DEFAULT_DELTA_UNITS = {'day': 10, 'month': 5, 'year': 2}
DEFAULT_INTERVAL_FLATTEN_CAPS = {'day': 16, 'month': 8, 'year': 4}

def dataset_name(dataset_path: Path) -> str:
    return f'{dataset_path.parent.name}_{dataset_path.name}'

def read_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)

def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + '\n')
            count += 1
    return count

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda : f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def manifest_hash(paths: Sequence[Path], extra: Mapping[str, Any] | None=None) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path.name).encode('utf-8'))
        if path.exists():
            digest.update(file_sha256(path).encode('utf-8'))
    if extra:
        digest.update(json.dumps(dict(extra), ensure_ascii=False, sort_keys=True).encode('utf-8'))
    return digest.hexdigest()

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

def time_leq(left: str, right: str) -> bool:
    return str(left) <= str(right)

def sample_evenly(values: Sequence[str], cap: int) -> list[str]:
    unique = sorted({str(value) for value in values if str(value).strip()})
    limit = max(0, int(cap))
    if limit <= 0 or len(unique) <= limit:
        return unique
    if limit == 1:
        return [unique[0]]
    last = len(unique) - 1
    return [unique[round(i * last / (limit - 1))] for i in range(limit)]

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

def build_flattened_interval_train_facts(dataset_path: Path, train_timestamps: Sequence[str], granularity: str, cap_by_granularity: Mapping[str, int] | None=None) -> tuple[list[TimestampFact], dict[str, Any]]:
    cap_map = {**DEFAULT_INTERVAL_FLATTEN_CAPS, **dict(cap_by_granularity or {})}
    cap = int(cap_map.get(str(granularity).lower(), DEFAULT_INTERVAL_FLATTEN_CAPS['day']))
    timeline = sorted({normalize_time(t, granularity) for t in train_timestamps if normalize_time(t, granularity)})
    if not timeline:
        return ([], {'flattened_fact_count': 0, 'interval_count': 0, 'skip_reason_counts': {'empty_train_timeline': 1}})
    max_train_t = timeline[-1]
    facts: list[TimestampFact] = []
    skip_counts: Counter[str] = Counter()
    intervals = read_intervals(dataset_path / 'intervals_train.csv', 'train')
    for interval in intervals:
        start = normalize_time(interval.t_start, granularity)
        end = normalize_time(interval.t_end, granularity) or max_train_t
        if not start:
            skip_counts['missing_start'] += 1
            continue
        if end < start:
            skip_counts['end_before_start'] += 1
            continue
        candidate_times = [t for t in timeline if start <= t <= end]
        selected_times = sample_evenly(candidate_times, cap)
        if not selected_times:
            skip_counts['no_timestamp_in_train_vocab'] += 1
            continue
        for timestamp in selected_times:
            facts.append(TimestampFact(split='train', s_id=int(interval.s_id), r_id=int(interval.r_id), o_id=int(interval.o_id), source_name=interval.source_name, relation_label=interval.relation_label, target_name=interval.target_name, t=timestamp, timestamp_source='flattened_interval_train', fact_hash=interval.fact_hash, metadata={'interval_t_start': interval.t_start, 'interval_t_end': interval.t_end}))
    summary = {'interval_count': len(intervals), 'flattened_fact_count': len(facts), 'cap_per_interval': cap, 'skip_reason_counts': dict(skip_counts)}
    return (facts, summary)

def ensure_shared_samples(dataset_path: Path, shared_root: Path, valid_facts: Sequence[TimestampFact], test_facts: Sequence[TimestampFact], history_facts: Sequence[TimestampFact], candidate_entity_ids: Sequence[int], granularity: str, delta_units: int, seed: int, overwrite: bool=False, max_obsolescence_samples: int | None=None, max_generation_queries: int | None=None) -> dict[str, Any]:
    sample_dir = shared_root / dataset_name(dataset_path)
    sample_dir.mkdir(parents=True, exist_ok=True)
    valid_path = sample_dir / 'valid_eval.csv'
    test_path = sample_dir / 'test_eval.csv'
    if overwrite or not valid_path.exists():
        write_eval_sample(valid_path, list(valid_facts))
    if overwrite or not test_path.exists():
        write_eval_sample(test_path, list(test_facts))
    generation_path = sample_dir / 'exp4_generation_queries.jsonl'
    if overwrite or not generation_path.exists():
        write_generation_queries(generation_path, test_facts, candidate_entity_ids, history_facts, granularity, seed, max_queries=max_generation_queries)
    obsolescence_path = sample_dir / 'obsolescence_samples.jsonl'
    obsolescence_skip_path = sample_dir / 'obsolescence_skips.json'
    if overwrite or not obsolescence_path.exists():
        (samples, skips) = build_obsolescence_samples(dataset_path=dataset_path, timestamps=sorted({fact.t for fact in list(history_facts) + list(test_facts)}), granularity=granularity, delta_units=delta_units, seed=seed, max_samples=max_obsolescence_samples)
        write_jsonl(obsolescence_path, samples)
        write_json(obsolescence_skip_path, {'skip_reason_counts': skips, 'sample_count': len(samples)})
    hard_indices_path = sample_dir / 'exp1_hard_indices.json'
    if overwrite or not hard_indices_path.exists():
        write_json(hard_indices_path, {'note': 'frequency-hard subset is computed from the shared test_eval.csv and persisted in each method report'})
    files = [valid_path, test_path, generation_path, obsolescence_path, hard_indices_path]
    extra = {'granularity': granularity, 'delta_units': int(delta_units), 'seed': int(seed)}
    digest = manifest_hash(files, extra)
    manifest = {'dataset_name': dataset_name(dataset_path), 'dataset_path': str(dataset_path), 'seed': int(seed), 'time_granularity': granularity, 'delta_units': int(delta_units), 'valid_eval_path': str(valid_path), 'test_eval_path': str(test_path), 'exp1_hard_indices_path': str(hard_indices_path), 'exp4_generation_queries_path': str(generation_path), 'obsolescence_samples_path': str(obsolescence_path), 'sample_manifest_hash': digest, 'file_hashes': {path.name: file_sha256(path) for path in files if path.exists()}}
    write_json(sample_dir / 'sample_manifest.json', manifest)
    return manifest

def load_shared_eval_facts(manifest: Mapping[str, Any]) -> tuple[list[TimestampFact], list[TimestampFact]]:
    return (read_eval_sample(Path(str(manifest['valid_eval_path'])), 'valid'), read_eval_sample(Path(str(manifest['test_eval_path'])), 'test'))

def write_generation_queries(path: Path, test_facts: Sequence[TimestampFact], candidate_entity_ids: Sequence[int], history_facts: Sequence[TimestampFact], granularity: str, seed: int, max_queries: int | None=None) -> dict[str, Any]:
    (gold_by_query, source_type_by_query, summary) = build_source_conditioned_generation_queries(test_facts, candidate_entity_ids, seed=int(seed), history_facts=history_facts, negative_sampling='hard_inactive', granularity=granularity)
    rows = []
    for (s_id, timestamp) in sorted(gold_by_query, key=lambda key: (key[1], key[0])):
        rows.append({'s_id': int(s_id), 't': str(timestamp), 'source_type': str(source_type_by_query.get((int(s_id), str(timestamp)), 'positive')), 'gold_pairs': [[int(r_id), int(o_id)] for (r_id, o_id) in sorted(gold_by_query[int(s_id), str(timestamp)])]})
    if max_queries is not None and int(max_queries) > 0 and (len(rows) > int(max_queries)):
        rng = random.Random(int(seed))
        rows = sorted(rng.sample(rows, int(max_queries)), key=lambda row: (str(row['t']), int(row['s_id'])))
        summary['query_cap'] = int(max_queries)
        summary['query_count_before_cap'] = len(gold_by_query)
    write_jsonl(path, rows)
    return summary

def read_generation_queries(path: Path) -> tuple[dict[tuple[int, str], set[tuple[int, int]]], dict[tuple[int, str], str]]:
    gold: dict[tuple[int, str], set[tuple[int, int]]] = {}
    source_type: dict[tuple[int, str], str] = {}
    for row in read_jsonl(path):
        key = (int(row['s_id']), str(row['t']))
        gold[key] = {(int(pair[0]), int(pair[1])) for pair in row.get('gold_pairs', [])}
        source_type[key] = str(row.get('source_type', 'positive'))
    return (gold, source_type)

def build_obsolescence_samples(dataset_path: Path, timestamps: Sequence[str], granularity: str, delta_units: int, seed: int=0, max_samples: int | None=None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    granularity = str(granularity or 'day').lower()
    timeline = sorted({normalize_time(t, granularity) for t in timestamps if normalize_time(t, granularity)})
    timeline_index = {timestamp: index for (index, timestamp) in enumerate(timeline)}
    delta = max(1, int(delta_units))
    samples: list[dict[str, Any]] = []
    skips: Counter[str] = Counter()
    for interval in read_intervals(dataset_path / 'intervals_test.csv', 'test'):
        t_end = normalize_time(interval.t_end, granularity)
        t_start = normalize_time(interval.t_start, granularity)
        if not t_end:
            skips['missing_t_end'] += 1
            continue
        if t_end not in timeline_index:
            skips['timestamp_not_in_vocab'] += 1
            continue
        end_index = timeline_index[t_end]
        pre = timeline[max(0, end_index - delta):end_index]
        post = timeline[end_index:min(len(timeline), end_index + delta + 1)]
        if not pre:
            skips['empty_pre_window'] += 1
            continue
        if not post:
            skips['empty_post_window'] += 1
            continue
        if t_start and (not t_start < pre[0]):
            skips['interval_not_active_before_pre_window'] += 1
            continue
        samples.append({'sample_id': len(samples), 's_id': int(interval.s_id), 'r_id': int(interval.r_id), 'o_id': int(interval.o_id), 'source_name': interval.source_name, 'relation_label': interval.relation_label, 'target_name': interval.target_name, 't_start': interval.t_start, 't_end': interval.t_end, 'normalized_t_start': t_start, 'normalized_t_end': t_end, 'pre_times': pre, 'post_times': post, 'fact_hash': interval.fact_hash})
    if max_samples is not None and int(max_samples) > 0 and (len(samples) > int(max_samples)):
        rng = random.Random(int(seed))
        samples = sorted(rng.sample(samples, int(max_samples)), key=lambda row: int(row['sample_id']))
        for (new_id, row) in enumerate(samples):
            row['sample_id'] = new_id
    return (samples, dict(skips))

def _pick(row: Mapping[str, str], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key, '')).strip()
        if value:
            return value
    return ''
