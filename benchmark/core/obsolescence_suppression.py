from __future__ import annotations
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from .schema import IntervalFact, TimestampFact

def normalize_time(value: str, granularity: str) -> str:
    value = str(value or '').strip()
    if not value:
        return ''
    granularity = str(granularity or 'day').lower()
    if granularity == 'year':
        return value[:4]
    if granularity == 'month':
        return value[:7]
    return value[:10]

def build_suppression_queries(dataset_path: str | Path, output_dir: str | Path, max_queries: int=500, seed: int=0) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _read_json(dataset_path / 'dataset_config.json')
    granularity = str(config.get('time_granularity', _granularity_from_dataset_name(dataset_path))).lower()
    test_times = sorted(_read_test_times(dataset_path, granularity))
    all_entities = _read_entity_ids(dataset_path / 'entities.csv')
    train_visible_entities = _read_train_visible_entities(dataset_path)
    candidate_entities = sorted(all_entities & train_visible_entities)
    intervals = _read_all_intervals(dataset_path)
    intervals_by_sr: dict[tuple[int, int], list[IntervalFact]] = defaultdict(list)
    for row in intervals:
        intervals_by_sr[int(row.s_id), int(row.r_id)].append(row)
    rows: list[dict[str, Any]] = []
    candidate_set = set(candidate_entities)
    for ((s_id, r_id), group) in intervals_by_sr.items():
        rows.extend(_queries_for_sr(int(s_id), int(r_id), group, test_times, granularity, candidate_set, len(rows)))
    original_count = len(rows)
    if max_queries and len(rows) > int(max_queries):
        rows = _sample_evenly_by_time(rows, int(max_queries), int(seed))
        for (index, row) in enumerate(rows):
            row['query_id'] = index
    queries_path = output_dir / 'queries.jsonl'
    with queries_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    candidate_path = output_dir / 'candidate_entities.json'
    _write_json(candidate_path, {'candidate_entity_ids': candidate_entities, 'candidate_scope': 'train_visible_entities', 'train_visible_entity_definition': 'entities appearing in train timestamp facts, train point facts, or train interval facts'})
    digest = _hash_files([queries_path, candidate_path])
    manifest = {'task': 'obsolete_knowledge_suppression', 'protocol': 'train_visible_entity_candidate_ranking', 'dataset_path': str(dataset_path), 'dataset_name': _dataset_name(dataset_path), 'semantic_domain': config.get('semantic_domain', ''), 'time_granularity': granularity, 'query_count': len(rows), 'pre_sample_query_count': original_count, 'max_queries': int(max_queries), 'seed': int(seed), 'candidate_entity_count': len(candidate_entities), 'all_entity_count': len(all_entities), 'train_visible_entity_count': len(train_visible_entities), 'queries_path': str(queries_path), 'candidate_entities_path': str(candidate_path), 'manifest_hash': digest}
    _write_json(output_dir / 'manifest.json', manifest)
    return manifest

def _queries_for_sr(s_id: int, r_id: int, group: Sequence[IntervalFact], test_times: Sequence[str], granularity: str, candidate_set: set[int], start_query_id: int) -> list[dict[str, Any]]:
    starts: list[tuple[str, int, str]] = []
    ends: list[tuple[str, int]] = []
    for interval in group:
        o_id = int(interval.o_id)
        if o_id not in candidate_set:
            continue
        start = normalize_time(interval.t_start, granularity)
        end = normalize_time(interval.t_end, granularity)
        if not start:
            continue
        starts.append((start, o_id, end))
        if end:
            ends.append((end, o_id))
    if not starts or not ends:
        return []
    starts.sort(key=lambda row: row[0])
    ends.sort(key=lambda row: row[0])
    min_start = starts[0][0]
    relevant_times = [t for t in test_times if t >= min_start]
    if not relevant_times:
        return []
    active_counts: defaultdict[int, int] = defaultdict(int)
    obsolete: set[int] = set()
    rows: list[dict[str, Any]] = []
    start_pos = 0
    end_pos = 0
    example = group[0]
    for t in relevant_times:
        while start_pos < len(starts) and starts[start_pos][0] <= t:
            (_start, o_id, end) = starts[start_pos]
            if not end or end >= t:
                active_counts[o_id] += 1
            elif end < t:
                obsolete.add(o_id)
            start_pos += 1
        while end_pos < len(ends) and ends[end_pos][0] < t:
            (_end, o_id) = ends[end_pos]
            if active_counts.get(o_id, 0) > 0:
                active_counts[o_id] -= 1
                if active_counts[o_id] <= 0:
                    active_counts.pop(o_id, None)
            obsolete.add(o_id)
            end_pos += 1
        valid = set(active_counts)
        obs = obsolete - valid
        if valid and obs:
            rows.append({'query_id': start_query_id + len(rows), 's_id': int(s_id), 'r_id': int(r_id), 't': t, 'valid_objects': sorted(valid), 'obsolete_objects': sorted(obs), 'source_name': example.source_name, 'relation_label': example.relation_label, 'candidate_scope': 'train_visible_entities'})
        if start_pos >= len(starts) and (not active_counts):
            break
    return rows

def load_suppression_manifest(path: str | Path) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    path = Path(path)
    manifest = _read_json(path / 'manifest.json')
    queries = [_parse_query(row) for row in _read_jsonl(Path(str(manifest['queries_path'])))]
    candidates = [int(x) for x in _read_json(Path(str(manifest['candidate_entities_path'])))['candidate_entity_ids']]
    return (queries, candidates, manifest)

def evaluate_suppression_queries(queries: Sequence[Mapping[str, Any]], candidate_entity_ids: Sequence[int], score_query_fn: Callable[[int, int, str, Sequence[int]], Mapping[int, float]], output_dir: str | Path, model_name: str, dataset_name: str, manifest_hash: str, progress_every: int=25) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_entity_ids = [int(x) for x in candidate_entity_ids]
    obsolete_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for (index, query) in enumerate(queries, start=1):
        (s_id, r_id, t) = (int(query['s_id']), int(query['r_id']), str(query['t']))
        valid_objects = [int(x) for x in query.get('valid_objects', []) if int(x) in candidate_entity_ids]
        obsolete_objects = [int(x) for x in query.get('obsolete_objects', []) if int(x) in candidate_entity_ids]
        if not valid_objects or not obsolete_objects:
            continue
        scores = dict(score_query_fn(s_id, r_id, t, candidate_entity_ids))
        ranks = _ranks_from_scores(scores, candidate_entity_ids)
        valid_ranks = [ranks[o] for o in valid_objects if o in ranks]
        obsolete_ranks = [ranks[o] for o in obsolete_objects if o in ranks]
        if not valid_ranks or not obsolete_ranks:
            continue
        for (o_id, rank) in zip(valid_objects, valid_ranks):
            valid_rows.append(_rank_row(query, int(o_id), int(rank), 'valid', float(scores.get(int(o_id), 0.0))))
        for (o_id, rank) in zip(obsolete_objects, obsolete_ranks):
            obsolete_rows.append(_rank_row(query, int(o_id), int(rank), 'obsolete', float(scores.get(int(o_id), 0.0))))
        pair_correct = 0
        pair_count = 0
        for v in valid_objects:
            for obs in obsolete_objects:
                if v not in ranks or obs not in ranks:
                    continue
                correct = int(ranks[v] < ranks[obs])
                pair_correct += correct
                pair_count += 1
                pair_rows.append({'query_id': int(query['query_id']), 's_id': s_id, 'r_id': r_id, 't': t, 'valid_o_id': int(v), 'obsolete_o_id': int(obs), 'valid_rank': int(ranks[v]), 'obsolete_rank': int(ranks[obs]), 'valid_score': float(scores.get(int(v), 0.0)), 'obsolete_score': float(scores.get(int(obs), 0.0)), 'correct': correct})
        query_rows.append({'query_id': int(query['query_id']), 's_id': s_id, 'r_id': r_id, 't': t, 'valid_count': len(valid_ranks), 'obsolete_count': len(obsolete_ranks), 'pair_count': pair_count, 'voa': float(pair_correct / pair_count) if pair_count else '', 'min_valid_rank': min(valid_ranks), 'min_obsolete_rank': min(obsolete_ranks)})
        if progress_every and (index % progress_every == 0 or index == len(queries)):
            print(f'[obsolescence_suppression] {model_name}/{dataset_name}: {index}/{len(queries)} queries', flush=True)
    obsolete_summary = _rank_summary([int(row['rank']) for row in obsolete_rows], prefix='Obs')
    valid_summary = _rank_summary([int(row['rank']) for row in valid_rows], prefix='Valid')
    voa = _mean([float(row['correct']) for row in pair_rows])
    report = {'task': 'obsolete_knowledge_suppression', 'protocol': 'train_visible_entity_candidate_ranking', 'model': model_name, 'dataset_name': dataset_name, 'manifest_hash': manifest_hash, 'candidate_entity_count': len(candidate_entity_ids), 'query_count': len(queries), 'covered_query_count': len(query_rows), 'obsolete_item_count': len(obsolete_rows), 'valid_item_count': len(valid_rows), 'pair_count': len(pair_rows), 'VOA': voa, **obsolete_summary, **valid_summary}
    _write_csv(output_dir / 'obsolete_ranks.csv', obsolete_rows)
    _write_csv(output_dir / 'valid_ranks.csv', valid_rows)
    _write_csv(output_dir / 'valid_obsolete_pairs.csv', pair_rows)
    _write_csv(output_dir / 'query_level_metrics.csv', query_rows)
    _write_json(output_dir / 'method_report.json', report)
    return report

def _rank_summary(ranks: Sequence[int], prefix: str) -> dict[str, Any]:
    ranks = [int(x) for x in ranks]
    if not ranks:
        return {f'{prefix}MRR': None, f'{prefix}Hits@10': None, f'{prefix}Hits@100': None, f'{prefix}Hits@1000': None}
    return {f'{prefix}MRR': _mean([1.0 / rank for rank in ranks]), f'{prefix}Hits@10': _mean([float(rank <= 10) for rank in ranks]), f'{prefix}Hits@100': _mean([float(rank <= 100) for rank in ranks]), f'{prefix}Hits@1000': _mean([float(rank <= 1000) for rank in ranks]), f'{prefix}MeanRank': _mean([float(rank) for rank in ranks])}

def _ranks_from_scores(scores: Mapping[int, float], candidate_entity_ids: Sequence[int]) -> dict[int, int]:
    if any(not math.isfinite(float(v)) for v in scores.values()):
        raise ValueError('Nonfinite object score')
    ids = sorted(set(map(int, candidate_entity_ids)))
    ranked = sorted(((float(scores[e]), e) for e in ids if e in scores), key=lambda item: (-item[0], item[1]))
    ranks = {entity_id: rank for (rank, (_score, entity_id)) in enumerate(ranked, start=1)}
    ranks.update({e: len(ids) for e in ids if e not in scores})
    return ranks

def _rank_row(query: Mapping[str, Any], o_id: int, rank: int, item_type: str, score: float) -> dict[str, Any]:
    return {'query_id': int(query['query_id']), 's_id': int(query['s_id']), 'r_id': int(query['r_id']), 't': str(query['t']), 'o_id': int(o_id), 'rank': int(rank), 'score': float(score), 'item_type': item_type}

def _sample_evenly_by_time(rows: list[dict[str, Any]], max_queries: int, seed: int) -> list[dict[str, Any]]:
    by_time: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_time[str(row['t'])].append(row)
    rng = random.Random(seed)
    times = sorted(by_time)
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < max_queries and times:
        t = times[cursor % len(times)]
        bucket = by_time[t]
        if bucket:
            selected.append(rng.choice(bucket))
            bucket.remove(selected[-1])
        times = [time for time in times if by_time[time]]
        cursor += 1
    return sorted(selected, key=lambda row: (str(row['t']), int(row['s_id']), int(row['r_id'])))[:max_queries]

def _parse_query(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out['valid_objects'] = [int(x) for x in out.get('valid_objects', [])]
    out['obsolete_objects'] = [int(x) for x in out.get('obsolete_objects', [])]
    return out

def _read_train_visible_entities(dataset_path: Path) -> set[int]:
    ids: set[int] = set()
    for split_file in ('facts_train.csv', 'points_train.csv', 'intervals_train.csv'):
        path = dataset_path / split_file
        if not path.exists():
            continue
        with path.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                try:
                    ids.add(int(str(row.get('s_id', '')).strip()))
                    ids.add(int(str(row.get('o_id', '')).strip()))
                except Exception:
                    continue
    return ids

def _read_entity_ids(path: Path) -> set[int]:
    ids: set[int] = set()
    if not path.exists():
        return ids
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            raw = str(row.get('entity_id', row.get('id', ''))).strip()
            if not raw:
                raw = str(row.get('id', '')).strip()
            try:
                ids.add(int(raw))
            except Exception:
                continue
    return ids

def _read_test_times(dataset_path: Path, granularity: str) -> set[str]:
    times: set[str] = set()
    for split_file in ('facts_test.csv', 'points_test.csv'):
        path = dataset_path / split_file
        if not path.exists():
            continue
        with path.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                t = normalize_time(str(row.get('t', '')).strip(), granularity)
                if t:
                    times.add(t)
    return times

def _read_all_intervals(dataset_path: Path) -> list[IntervalFact]:
    intervals: list[IntervalFact] = []
    for split in ('train', 'valid', 'test'):
        path = dataset_path / f'intervals_{split}.csv'
        if not path.exists():
            continue
        with path.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                try:
                    intervals.append(IntervalFact(split=split, s_id=int(str(row.get('s_id', '')).strip()), r_id=int(str(row.get('r_id', '')).strip()), o_id=int(str(row.get('o_id', '')).strip()), source_name=str(row.get('source_name', '')).strip(), relation_label=str(row.get('relation_label', '')).strip(), target_name=str(row.get('target_name', '')).strip(), t_start=str(row.get('t_start', '')).strip(), t_end=str(row.get('t_end', '')).strip(), fact_hash=str(row.get('hash', '')).strip()))
                except Exception:
                    continue
    return intervals

def _read_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def _hash_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode('utf-8'))
        digest.update(path.read_bytes())
    return digest.hexdigest()

def _mean(values: Sequence[float]) -> float | None:
    values = [float(x) for x in values]
    return sum(values) / len(values) if values else None

def _dataset_name(path: Path) -> str:
    return f'{path.parent.name}_{path.name}'

def _granularity_from_dataset_name(path: Path) -> str:
    name = _dataset_name(path)
    if '-year' in name:
        return 'year'
    if '-month' in name:
        return 'month'
    return 'day'
