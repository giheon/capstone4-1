from __future__ import annotations

from pathlib import Path
import unittest

from app.domain.concepts.catalog import ConceptCatalog


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConceptCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = ConceptCatalog.from_project_root(PROJECT_ROOT)

    def test_loads_all_catalog_files(self) -> None:
        self.assertEqual(len(self.catalog.loaded_files), 6)
        self.assertEqual(self.catalog.entry_count, 343)
        self.assertEqual(self.catalog.concept_count, 212)
        self.assertEqual(self.catalog.action_count, 131)

    def test_known_concept_lookup(self) -> None:
        concept = self.catalog.get_concept("MATH1_CH01_CONCEPT_ROOT_NTH")
        self.assertIsNotNone(concept)
        self.assertEqual(concept.label_ko, "n제곱근")
        self.assertEqual(concept.subject, "수학 I")

    def test_flow_falls_back_to_validated_concepts(self) -> None:
        flow = self.catalog.build_public_flow([], ["MATH1_CH01_CONCEPT_ROOT_NTH"])
        self.assertEqual(len(flow), 1)
        self.assertEqual(flow[0].order, 1)
        self.assertEqual(flow[0].concept_id, "MATH1_CH01_CONCEPT_ROOT_NTH")
