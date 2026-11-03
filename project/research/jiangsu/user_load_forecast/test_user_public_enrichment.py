"""Tests for the public-information enrichment guardrails."""

from __future__ import annotations

import unittest

import pandas as pd

from project.research.jiangsu.user_load_forecast.user_public_enrichment import (
    apply_curated_enrichment,
    build_private_queue,
)
from project.research.jiangsu.user_load_forecast.user_segmentation import UNKNOWN


class PublicEnrichmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.profiles = pd.DataFrame(
            [
                {
                    "user_id": "unknown_user",
                    "enterprise_name": "待核验企业",
                    "history_days": 20,
                    "positive_energy": 60.0,
                    "positive_energy_share": 0.6,
                    "baseline_city": UNKNOWN,
                    "baseline_city_confidence": "none",
                    "baseline_gbt_industry": UNKNOWN,
                    "baseline_industry_confidence": "none",
                },
                {
                    "user_id": "known_user",
                    "enterprise_name": "已有规则企业",
                    "history_days": 20,
                    "positive_energy": 40.0,
                    "positive_energy_share": 0.4,
                    "baseline_city": "苏州",
                    "baseline_city_confidence": "high",
                    "baseline_gbt_industry": "C31 黑色金属冶炼和压延加工业",
                    "baseline_industry_confidence": "high",
                },
            ]
        )

    @staticmethod
    def curated() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "user_id": "unknown_user",
                    "city": "南通",
                    "city_confidence": "high",
                    "gbt_industry": "C26 化学原料和化学制品制造业",
                    "industry_confidence": "high",
                    "evidence_tier": "A",
                    "source_kind": "government",
                    "source_url": "https://example.invalid/unknown",
                    "retrieved_at": "2026-07-30",
                    "notes": "",
                },
                {
                    "user_id": "known_user",
                    "city": "无锡",
                    "city_confidence": "high",
                    "gbt_industry": "C26 化学原料和化学制品制造业",
                    "industry_confidence": "high",
                    "evidence_tier": "A",
                    "source_kind": "government",
                    "source_url": "https://example.invalid/known",
                    "retrieved_at": "2026-07-30",
                    "notes": "",
                },
            ]
        )

    def test_curated_labels_only_fill_unresolved_values(self) -> None:
        enriched, conflicts = apply_curated_enrichment(
            self.profiles,
            self.curated(),
        )
        rows = enriched.set_index("user_id")

        self.assertEqual(rows.loc["unknown_user", "final_city"], "南通")
        self.assertEqual(
            rows.loc["unknown_user", "final_gbt_industry"],
            "C26 化学原料和化学制品制造业",
        )
        self.assertEqual(rows.loc["unknown_user", "city_source"], "curated_public")
        self.assertEqual(rows.loc["known_user", "final_city"], "苏州")
        self.assertEqual(
            rows.loc["known_user", "final_gbt_industry"],
            "C31 黑色金属冶炼和压延加工业",
        )
        self.assertEqual(rows.loc["known_user", "city_source"], "name_rule")
        self.assertEqual(
            set(conflicts["dimension"]),
            {"city", "gbt_industry"},
        )

    def test_private_queue_prioritizes_dual_missing_high_load_user(self) -> None:
        enriched, _ = apply_curated_enrichment(
            self.profiles,
            self.curated().iloc[0:0],
        )
        queue = build_private_queue(enriched, queue_size=1)

        self.assertEqual(queue.iloc[0]["user_id"], "unknown_user")
        self.assertEqual(queue.iloc[0]["lookup_need"], "city_and_industry")
        self.assertIn("enterprise_name", queue.columns)
        self.assertIn("suggested_query", queue.columns)


if __name__ == "__main__":
    unittest.main()
