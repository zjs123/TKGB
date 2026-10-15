from __future__ import annotations
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
ScoreBatchFn = Callable[[Sequence[tuple[int, int, int, str]]], list[float]]

def evaluate_obsolescence_samples(samples: Sequence[Mapping[str, Any]], score_batch_fn: ScoreBatchFn, output_dir: Path, dataset_name: str, dataset_hash: str, model_name: str, granularity: str, delta_units: int, sample_manifest_hash: str, score_batch_size: int=4096, progress_label: str='', progress_interval: int=0) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_interval_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    score_batch_size = max(1, int(score_batch_size))
    skipped = 0
    skip_reason_counts: dict[str, int] = {}
    total_samples = len(samples)
    started = time.time()
    last_progress = started
    for (sample_index, sample) in enumerate(samples, start=1):
        pre_times = [str(t) for t in sample.get('pre_times', [])]
        post_times = [str(t) for t in sample.get('post_times', [])]
        if not pre_times or not post_times:
            skipped += 1
            reason = 'empty_window'
            skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
            continue
        queries = [(int(sample['s_id']), int(sample['r_id']), int(sample['o_id']), t) for t in pre_times + post_times]
        scores: list[float] = []
        for start in range(0, len(queries), score_batch_size):
            scores.extend(score_batch_fn(queries[start:start + score_batch_size]))
        if len(scores) != len(queries) or any((not math.isfinite(float(value)) for value in scores)):
            skipped += 1
            reason = 'invalid_score'
            skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
            continue
        pre_scores = [float(value) for value in scores[:len(pre_times)]]
        post_scores = [float(value) for value in scores[len(pre_times):]]
        pre_mean = _mean(pre_scores)
        post_mean = _mean(post_scores)
        delta = pre_mean - post_mean
        correct = delta > 0.0
        interval_row = {'dataset': dataset_name, 'dataset_hash': dataset_hash, 'model': model_name, 'sample_id': int(sample.get('sample_id', len(by_interval_rows))), 's_id': int(sample['s_id']), 'r_id': int(sample['r_id']), 'o_id': int(sample['o_id']), 'source_name': str(sample.get('source_name', '')), 'relation_label': str(sample.get('relation_label', '')), 'target_name': str(sample.get('target_name', '')), 't_start': str(sample.get('t_start', '')), 't_end': str(sample.get('t_end', '')), 'normalized_t_end': str(sample.get('normalized_t_end', '')), 'pre_count': len(pre_scores), 'post_count': len(post_scores), 'pre_score_mean': pre_mean, 'post_score_mean': post_mean, 'delta': delta, 'correct': int(correct), 'time_granularity': granularity, 'sample_manifest_hash': sample_manifest_hash}
        by_interval_rows.append(interval_row)
        for (side, times, side_scores) in (('pre', pre_times, pre_scores), ('post', post_times, post_scores)):
            for (timestamp, score) in zip(times, side_scores):
                distribution_rows.append({'dataset': dataset_name, 'dataset_hash': dataset_hash, 'model': model_name, 'sample_id': interval_row['sample_id'], 's_id': interval_row['s_id'], 'r_id': interval_row['r_id'], 'o_id': interval_row['o_id'], 't_start': interval_row['t_start'], 't_end': interval_row['t_end'], 't': timestamp, 'window_side': side, 'score': score, 'delta': delta, 'correct': int(correct), 'time_granularity': granularity, 'sample_manifest_hash': sample_manifest_hash})
        if progress_label and progress_interval > 0 and (sample_index % progress_interval == 0 or sample_index == total_samples):
            elapsed = max(1e-09, time.time() - started)
            recent_elapsed = max(1e-09, time.time() - last_progress)
            rate = sample_index / elapsed
            remaining = (total_samples - sample_index) / rate if rate > 0 else 0.0
            print(f'[obsolescence] {progress_label}: {sample_index}/{total_samples} interval(s), evaluated={len(by_interval_rows)}, skipped={skipped}, elapsed={elapsed / 60:.1f}m, eta={remaining / 60:.1f}m', flush=True)
            last_progress = time.time()
    by_interval_path = output_dir / 'obsolescence_by_interval.csv'
    distribution_path = output_dir / 'obsolescence_score_distribution.csv'
    _write_csv(by_interval_path, by_interval_rows, _interval_fields())
    _write_csv(distribution_path, distribution_rows, _distribution_fields())
    deltas = [float(row['delta']) for row in by_interval_rows]
    pre_scores_all = [float(row['score']) for row in distribution_rows if row['window_side'] == 'pre']
    post_scores_all = [float(row['score']) for row in distribution_rows if row['window_side'] == 'post']
    evaluated = len(by_interval_rows)
    summary = {'task': 'knowledge_obsolescence_prediction', 'dataset': dataset_name, 'dataset_hash': dataset_hash, 'model': model_name, 'delta_units': int(delta_units), 'delta_units_by_granularity': {'day': 10, 'month': 5, 'year': 2}, 'time_granularity': granularity, 'timestamp_mode': 'strict_existing_vocab', 'sample_manifest_hash': sample_manifest_hash, 'evaluated_interval_count': evaluated, 'skipped_interval_count': int(skipped), 'accuracy': sum((1 for row in by_interval_rows if int(row['correct']) == 1)) / evaluated if evaluated else None, 'mean_delta': _mean(deltas) if deltas else None, 'median_delta': statistics.median(deltas) if deltas else None, 'pre_score_mean': _mean(pre_scores_all) if pre_scores_all else None, 'post_score_mean': _mean(post_scores_all) if post_scores_all else None, 'pre_score_std': _std(pre_scores_all) if pre_scores_all else None, 'post_score_std': _std(post_scores_all) if post_scores_all else None, 'skip_reason_counts': skip_reason_counts, 'obsolescence_by_interval_path': str(by_interval_path), 'obsolescence_score_distribution_path': str(distribution_path)}
    with (output_dir / 'obsolescence_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    return summary

def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})

def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0

def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))

def _interval_fields() -> list[str]:
    return ['dataset', 'dataset_hash', 'model', 'sample_id', 's_id', 'r_id', 'o_id', 'source_name', 'relation_label', 'target_name', 't_start', 't_end', 'normalized_t_end', 'pre_count', 'post_count', 'pre_score_mean', 'post_score_mean', 'delta', 'correct', 'time_granularity', 'sample_manifest_hash']

def _distribution_fields() -> list[str]:
    return ['dataset', 'dataset_hash', 'model', 'sample_id', 's_id', 'r_id', 'o_id', 't_start', 't_end', 't', 'window_side', 'score', 'delta', 'correct', 'time_granularity', 'sample_manifest_hash']
