from __future__ import annotations
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List
from .schema import BenchmarkDataset, IntervalFact, TimestampFact
SPLITS = ('train', 'valid', 'test')

def load_dataset(path: str | Path, test_source: str='offline', max_facts_per_split: int | None=None, max_intervals_per_split: int | None=None) -> BenchmarkDataset:
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f'Dataset path does not exist: {root}')
    if test_source not in {'offline', 'online'}:
        raise ValueError(f'Unknown test_source: {test_source}')
    dataset_root = root.parent if root.name == 'online_test' and test_source == 'online' else root
    online_root = root if root.name == 'online_test' else dataset_root / 'online_test'
    if test_source == 'online' and (not online_root.exists()):
        raise FileNotFoundError(f'Online test artifact does not exist: {online_root}')
    entities = _read_id_table(dataset_root / 'entities.csv', id_columns=('entity_id', 'id'))
    relations = _read_id_table(dataset_root / 'relations.csv', id_columns=('relation_id', 'id'))
    if test_source == 'online':
        entities.update(_read_id_table(online_root / 'entities_online.csv', id_columns=('entity_id', 'id')))
        relations.update(_read_id_table(online_root / 'relations_online.csv', id_columns=('relation_id', 'id')))
    dataset = BenchmarkDataset(root=str(dataset_root), entities=entities, relations=relations)
    fact_limit = max_facts_per_split if max_facts_per_split and max_facts_per_split > 0 else None
    interval_limit = max_intervals_per_split if max_intervals_per_split and max_intervals_per_split > 0 else None
    dataset.train = _read_timestamp_split(dataset_root, 'train', fact_limit)
    dataset.valid = _read_timestamp_split(dataset_root, 'valid', fact_limit)
    if test_source == 'online':
        dataset.test = _read_timestamp_files(online_root, 'test', (('facts', 'facts_online_test.csv'), ('points', 'points_online_test.csv')), fact_limit)
        dataset.intervals = {'train': _read_interval_file(dataset_root / 'intervals_train.csv', 'train', interval_limit), 'valid': _read_interval_file(dataset_root / 'intervals_valid.csv', 'valid', interval_limit), 'test': _read_interval_file(online_root / 'intervals_online_test.csv', 'test', interval_limit)}
    else:
        dataset.test = _read_timestamp_split(dataset_root, 'test', fact_limit)
        dataset.intervals = {split: _read_interval_file(dataset_root / f'intervals_{split}.csv', split, interval_limit) for split in SPLITS}
    return dataset

def _iter_csv_rows(path: Path) -> Iterator[Dict[str, str]]:
    if not path.exists():
        return
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            yield dict(row)

def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    return list(_iter_csv_rows(path))

def _read_id_table(path: Path, id_columns: tuple[str, ...]) -> Dict[int, Dict[str, Any]]:
    rows = _read_csv_rows(path)
    table: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        raw_id = ''
        for col in id_columns:
            raw_id = str(row.get(col, '')).strip()
            if raw_id:
                break
        if not raw_id:
            continue
        try:
            row_id = int(raw_id)
        except ValueError:
            continue
        table[row_id] = row
    return table

def _read_timestamp_split(root: Path, split: str, row_limit: int | None=None) -> List[TimestampFact]:
    return _read_timestamp_files(root, split, (('facts', f'facts_{split}.csv'), ('points', f'points_{split}.csv')), row_limit)

def _read_timestamp_files(root: Path, split: str, files: Iterable[tuple[str, str]], row_limit: int | None=None) -> List[TimestampFact]:
    facts: List[TimestampFact] = []
    for (source_name, filename) in files:
        for row in _iter_csv_rows(root / filename):
            if not _has_required_timestamp_fields(row):
                continue
            facts.append(TimestampFact(split=split, s_id=int(str(row['s_id']).strip()), r_id=int(str(row['r_id']).strip()), o_id=int(str(row['o_id']).strip()), source_name=_pick(row, 'source_name', 'head_name', 's_name'), relation_label=_pick(row, 'relation_label', 'property_label', 'r_label'), target_name=_pick(row, 'target_name', 'tail_name', 'o_name'), t=str(row.get('t', '')).strip(), timestamp_source=source_name, fact_hash=str(row.get('hash', '')).strip(), metadata=dict(row)))
            if row_limit is not None and len(facts) >= row_limit:
                return facts
    return facts

def _read_interval_file(path: Path, split: str, row_limit: int | None=None) -> List[IntervalFact]:
    intervals: List[IntervalFact] = []
    for row in _iter_csv_rows(path):
        if not _has_required_interval_fields(row):
            continue
        intervals.append(IntervalFact(split=split, s_id=int(str(row['s_id']).strip()), r_id=int(str(row['r_id']).strip()), o_id=int(str(row['o_id']).strip()), source_name=_pick(row, 'source_name', 'head_name', 's_name'), relation_label=_pick(row, 'relation_label', 'property_label', 'r_label'), target_name=_pick(row, 'target_name', 'tail_name', 'o_name'), t_start=str(row.get('t_start', '')).strip(), t_end=str(row.get('t_end', '')).strip(), fact_hash=str(row.get('hash', '')).strip()))
        if row_limit is not None and len(intervals) >= row_limit:
            return intervals
    return intervals

def _has_required_timestamp_fields(row: Dict[str, str]) -> bool:
    required = ('s_id', 'r_id', 'o_id', 't')
    return all((str(row.get(col, '')).strip() for col in required))

def _has_required_interval_fields(row: Dict[str, str]) -> bool:
    required = ('s_id', 'r_id', 'o_id', 't_start')
    return all((str(row.get(col, '')).strip() for col in required))

def _pick(row: Dict[str, str], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key, '')).strip()
        if value:
            return value
    return ''
