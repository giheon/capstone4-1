from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.schemas.response import ConceptResponse, FlowStepResponse


CATALOG_FILENAMES = [
    "all_math_concept_dict_merged.json",
    "math1_concept_dict_merged.json",
    "math2_concept_dict_merged.json",
    "calculus_concept_dict_merged.json",
    "geometry_concept_dict_merged.json",
    "probstat_concept_dict_merged.json",
]


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    concept_id: str
    label_ko: str
    subject: str
    chapter_id: str
    section_id: str
    type: str
    aliases: tuple[str, ...]
    latex_patterns: tuple[str, ...]


class ConceptCatalog:
    def __init__(self, entries_by_id: dict[str, CatalogEntry], loaded_files: list[str]) -> None:
        self._entries_by_id = entries_by_id
        self._loaded_files = loaded_files
        self._concepts_by_id = {
            concept_id: entry
            for concept_id, entry in entries_by_id.items()
            if entry.type == "concept"
        }
        self._actions_by_id = {
            concept_id: entry
            for concept_id, entry in entries_by_id.items()
            if entry.type == "action"
        }
        self._prompt_reference = self._build_prompt_reference()

    @property
    def entry_count(self) -> int:
        return len(self._entries_by_id)

    @property
    def concept_count(self) -> int:
        return len(self._concepts_by_id)

    @property
    def action_count(self) -> int:
        return len(self._actions_by_id)

    @property
    def loaded_files(self) -> list[str]:
        return list(self._loaded_files)

    @property
    def prompt_reference(self) -> str:
        return self._prompt_reference

    @classmethod
    def from_project_root(cls, project_root: Path) -> "ConceptCatalog":
        entries_by_id: dict[str, CatalogEntry] = {}
        loaded_files: list[str] = []

        for filename in CATALOG_FILENAMES:
            file_path = project_root / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Concept catalog file not found: {file_path}")
            loaded_files.append(filename)
            with file_path.open("r", encoding="utf-8") as file:
                raw_data = json.load(file)
            for entry in cls._iter_entries(raw_data):
                entries_by_id.setdefault(entry.concept_id, entry)

        return cls(entries_by_id=entries_by_id, loaded_files=loaded_files)

    @staticmethod
    def _iter_entries(raw_data: dict[str, Any]) -> Iterable[CatalogEntry]:
        subjects = raw_data.get("subjects", [raw_data])
        for subject_data in subjects:
            subject_name = subject_data["subject"]
            for chapter in subject_data.get("chapters", []):
                chapter_id = chapter["chapter_id"]
                for section in chapter.get("sections", []):
                    section_id = section["section_id"]
                    for entry in section.get("entries", []):
                        yield CatalogEntry(
                            concept_id=entry["concept_id"],
                            label_ko=entry["canonical_ko"],
                            subject=subject_name,
                            chapter_id=chapter_id,
                            section_id=section_id,
                            type=entry["type"],
                            aliases=tuple(entry.get("aliases", [])),
                            latex_patterns=tuple(entry.get("latex_patterns", [])),
                        )

    def get(self, concept_id: str) -> CatalogEntry | None:
        return self._entries_by_id.get(concept_id)

    def get_concept(self, concept_id: str) -> CatalogEntry | None:
        return self._concepts_by_id.get(concept_id)

    def validate_concept_ids(self, concept_ids: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        valid_ids: list[str] = []
        for raw_concept_id in concept_ids:
            concept_id = raw_concept_id.strip()
            if not concept_id or concept_id in seen:
                continue
            if concept_id in self._concepts_by_id:
                seen.add(concept_id)
                valid_ids.append(concept_id)
        return valid_ids

    def merge_valid_concept_ids(
        self,
        concept_ids: Iterable[str],
        flow_items: Iterable[dict[str, Any]],
    ) -> list[str]:
        merged = self.validate_concept_ids(concept_ids)
        existing = set(merged)
        for flow_item in flow_items:
            concept_id = self._extract_flow_concept_id(flow_item)
            if concept_id in self._concepts_by_id and concept_id not in existing:
                merged.append(concept_id)
                existing.add(concept_id)
        return merged

    def build_public_concepts(self, concept_ids: Iterable[str]) -> list[ConceptResponse]:
        public_concepts: list[ConceptResponse] = []
        for concept_id in concept_ids:
            entry = self._concepts_by_id[concept_id]
            public_concepts.append(
                ConceptResponse(
                    concept_id=entry.concept_id,
                    label_ko=entry.label_ko,
                    subject=entry.subject,
                    chapter_id=entry.chapter_id,
                    type="concept",
                )
            )
        return public_concepts

    def build_public_flow(
        self,
        flow_items: Iterable[dict[str, Any]],
        allowed_concept_ids: Iterable[str],
    ) -> list[FlowStepResponse]:
        allowed = list(allowed_concept_ids)
        allowed_set = set(allowed)
        seen: set[str] = set()
        public_flow: list[FlowStepResponse] = []

        for flow_item in flow_items:
            concept_id = self._extract_flow_concept_id(flow_item)
            if concept_id in allowed_set and concept_id not in seen:
                entry = self._concepts_by_id[concept_id]
                public_flow.append(
                    FlowStepResponse(
                        order=len(public_flow) + 1,
                        concept_id=entry.concept_id,
                        label_ko=entry.label_ko,
                    )
                )
                seen.add(concept_id)

        if public_flow:
            return public_flow

        for concept_id in allowed:
            entry = self._concepts_by_id[concept_id]
            public_flow.append(
                FlowStepResponse(
                    order=len(public_flow) + 1,
                    concept_id=entry.concept_id,
                    label_ko=entry.label_ko,
                )
            )
        return public_flow

    def _build_prompt_reference(self) -> str:
        lines = []
        for entry in sorted(self._concepts_by_id.values(), key=lambda item: item.concept_id):
            lines.append(
                f"{entry.concept_id} | {entry.label_ko} | {entry.subject} | {entry.chapter_id}"
            )
        return "\n".join(lines)

    @staticmethod
    def _extract_flow_concept_id(flow_item: dict[str, Any]) -> str:
        concept_id = flow_item.get("concept_id", "")
        return concept_id.strip() if isinstance(concept_id, str) else ""
