from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

@dataclass(frozen=True)
class TransductiveScope:
    train_entity_ids: list[int]

    @property
    def train_entity_set(self) -> set[int]:
        return set(self.train_entity_ids)

    def is_transductive_fact(self, fact: Any) -> bool:
        train_entities = self.train_entity_set
        return int(fact.s_id) in train_entities and int(fact.o_id) in train_entities

    def candidate_entity_ids(self, entity_ids: Iterable[int]) -> list[int]:
        train_entities = self.train_entity_set
        return sorted({int(entity_id) for entity_id in entity_ids if int(entity_id) in train_entities})

def build_transductive_scope(train_facts: Sequence[Any]) -> TransductiveScope:
    train_entities = set()
    for fact in train_facts:
        train_entities.add(int(fact.s_id))
        train_entities.add(int(fact.o_id))
    return TransductiveScope(train_entity_ids=sorted(train_entities))

def filter_transductive_facts(facts: Sequence[Any], scope: TransductiveScope) -> list[Any]:
    return [fact for fact in facts if scope.is_transductive_fact(fact)]

def transductive_protocol_summary(raw_facts: Sequence[Any], transductive_facts: Sequence[Any], scope: TransductiveScope) -> dict[str, Any]:
    raw_count = len(raw_facts)
    transductive_count = len(transductive_facts)
    return {'fact_scope': 'transductive_only', 'entity_scope': 'train_seen_entities_only', 'raw_fact_count': raw_count, 'transductive_fact_count': transductive_count, 'filtered_inductive_fact_count': max(0, raw_count - transductive_count), 'candidate_entity_count': len(scope.train_entity_ids)}

def add_transductive_protocol_metadata(prefix: str, raw_facts: Sequence[Any], transductive_facts: Sequence[Any], scope: TransductiveScope) -> dict[str, Any]:
    summary = transductive_protocol_summary(raw_facts, transductive_facts, scope)
    return {f'{prefix}_{key}': value for (key, value) in summary.items()}
