from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence

@dataclass(frozen=True)
class TimestampFact:
    split: str
    s_id: int
    r_id: int
    o_id: int
    source_name: str
    relation_label: str
    target_name: str
    t: str
    timestamp_source: str
    fact_hash: str = ''
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def name_quad(self) -> tuple[str, str, str, str]:
        return (self.source_name, self.relation_label, self.target_name, self.t)

    def id_quad(self) -> tuple[int, int, int, str]:
        return (self.s_id, self.r_id, self.o_id, self.t)

@dataclass(frozen=True)
class IntervalFact:
    split: str
    s_id: int
    r_id: int
    o_id: int
    source_name: str
    relation_label: str
    target_name: str
    t_start: str
    t_end: str
    fact_hash: str = ''

@dataclass(frozen=True)
class EntityRef:
    entity_id: int
    name: str = ''

    @property
    def s_id(self) -> int:
        return self.entity_id

@dataclass
class BenchmarkDataset:
    root: str
    train: List[TimestampFact] = field(default_factory=list)
    valid: List[TimestampFact] = field(default_factory=list)
    test: List[TimestampFact] = field(default_factory=list)
    intervals: Dict[str, List[IntervalFact]] = field(default_factory=lambda : {'train': [], 'valid': [], 'test': []})
    entities: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    relations: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def split_facts(self, split: str) -> List[TimestampFact]:
        if split == 'train':
            return self.train
        if split == 'valid':
            return self.valid
        if split == 'test':
            return self.test
        raise ValueError(f'Unknown split: {split}')

    def all_timestamp_facts(self) -> List[TimestampFact]:
        return list(self.train) + list(self.valid) + list(self.test)

    def entity_ids(self) -> List[int]:
        ids = set(self.entities)
        for fact in self.all_timestamp_facts():
            ids.add(fact.s_id)
            ids.add(fact.o_id)
        for rows in self.intervals.values():
            for fact in rows:
                ids.add(fact.s_id)
                ids.add(fact.o_id)
        return sorted(ids)

    def relation_ids(self) -> List[int]:
        ids = set(self.relations)
        for fact in self.all_timestamp_facts():
            ids.add(fact.r_id)
        for rows in self.intervals.values():
            for fact in rows:
                ids.add(fact.r_id)
        return sorted(ids)

    def train_entities(self) -> List[EntityRef]:
        ids = set()
        for fact in self.train:
            ids.add(fact.s_id)
            ids.add(fact.o_id)
        return [EntityRef(entity_id=i, name=entity_name(i, self.entities)) for i in sorted(ids)]

@dataclass(frozen=True)
class EvaluationResult:
    count: int
    ranks: List[int]
    mrr: float
    hits_at_1: float
    hits_at_3: float
    hits_at_10: float

    def to_dict(self) -> Dict[str, Any]:
        return {'count': self.count, 'ranks': list(self.ranks), 'mrr': self.mrr, 'hits_at_1': self.hits_at_1, 'hits_at_3': self.hits_at_3, 'hits_at_10': self.hits_at_10}

def entity_name(entity_id: int, entities: Mapping[int, Mapping[str, Any]]) -> str:
    row = entities.get(entity_id, {})
    for key in ('entity_name', 'name', 'label', 'wikidata_label', 'original_text'):
        value = str(row.get(key, '')).strip()
        if value:
            return value
    return f'entity_{entity_id}'

def relation_name(relation_id: int, relations: Mapping[int, Mapping[str, Any]]) -> str:
    row = relations.get(relation_id, {})
    for key in ('relation_label', 'label', 'name'):
        value = str(row.get(key, '')).strip()
        if value:
            return value
    return f'relation_{relation_id}'

def split_counts(dataset: BenchmarkDataset) -> Dict[str, int]:
    return {split: len(dataset.split_facts(split)) for split in ('train', 'valid', 'test')}
