from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.regional_weather import (
    CityWeatherPanel,
    weighted_weather_store,
)
from project.research.jiangsu.user_load_forecast.industrial_behavior_research import (
    build_user_behavior_features,
    read_visible_market_outcomes,
)
from project.research.jiangsu.user_load_forecast.user_segmentation import (
    JIANGSU_CITIES,
    UNKNOWN,
    ProxyLabel,
    SegmentedUserPanel,
    city_load_weights,
    infer_city,
    infer_gbt_industry,
    infer_industry,
)
from project.research.jiangsu.user_load_forecast.weather_features import (
    read_weather_snapshot,
)


class RegionalWeatherTest(unittest.TestCase):
    def test_name_proxy_does_not_treat_generic_technology_as_electronics(
        self,
    ) -> None:
        self.assertEqual(infer_city("某某（南通）环保有限公司").value, "南通")
        self.assertEqual(infer_city("江阴某某机械有限公司").value, "无锡")
        self.assertEqual(infer_industry("某某科技有限公司").value, UNKNOWN)
        self.assertEqual(infer_industry("某某纺织有限公司").value, "纺织服装")
        self.assertTrue(infer_gbt_industry("某某纺织有限公司").value.startswith("C17 "))
        self.assertTrue(
            infer_gbt_industry("某某汽车零部件有限公司").value.startswith("C36 ")
        )

    def test_city_weights_use_visible_energy_and_spread_unknown(self) -> None:
        visible_day = date(2026, 4, 28)
        future_day = date(2026, 5, 1)
        panel = SegmentedUserPanel(
            curves={
                "known": {
                    visible_day: np.full(96, 80.0 / 96),
                    future_day: np.full(96, 10_000.0 / 96),
                },
                "unknown": {visible_day: np.full(96, 20.0 / 96)},
            },
            city={
                "known": ProxyLabel("南通", "HIGH_NAME_EXPLICIT"),
                "unknown": ProxyLabel(UNKNOWN, "NONE"),
            },
            industry={
                "known": ProxyLabel(UNKNOWN, "NONE"),
                "unknown": ProxyLabel(UNKNOWN, "NONE"),
            },
            gbt_industry={
                "known": ProxyLabel(UNKNOWN, "NONE"),
                "unknown": ProxyLabel(UNKNOWN, "NONE"),
            },
            negative_users=set(),
        )
        weights, metadata = city_load_weights(panel, cutoff=visible_day)
        unknown_each = 0.2 / len(JIANGSU_CITIES)
        self.assertTrue(np.isclose(weights["南通"], 0.8 + unknown_each))
        self.assertTrue(np.isclose(weights["南京"], unknown_each))
        self.assertTrue(np.isclose(sum(weights.values()), 1.0))
        self.assertTrue(np.isclose(metadata["known_city_energy_share"], 0.8))
        behavior = build_user_behavior_features(
            panel,
            cutoff=visible_day,
            recent_days=1,
            minimum_observed_days=1,
        )
        self.assertTrue(
            np.isclose(behavior.iloc[0]["average_daily_energy"], 80.0)
        )

    def test_weighted_weather_store_respects_city_weights(self) -> None:
        timestamp = pd.Timestamp("2026-05-01 01:00:00")
        rows = []
        for index, city in enumerate(JIANGSU_CITIES):
            rows.append(
                {
                    "city": city,
                    "source": "test",
                    "timestamp": timestamp,
                    "temp_2m": float(index),
                    "humidity_2m": 50.0,
                    "rain": 0.0,
                    "radiation": 0.0,
                    "wind_10m": 1.0,
                }
            )
        panel = CityWeatherPanel(
            frame=pd.DataFrame(rows),
            source="test",
            source_path="/tmp/not-read",
            source_sha256="0" * 64,
        )
        weights = {city: 0.0 for city in JIANGSU_CITIES}
        weights["南通"] = 1.0
        store = weighted_weather_store(
            panel,
            city_weights=weights,
            variant="nantong_only",
        )
        expected = float(JIANGSU_CITIES.index("南通"))
        self.assertTrue(
            np.isclose(store.hourly.loc[timestamp, "temp_2m"], expected)
        )

    def test_proxy_city_mean_is_invariant_to_csv_chunk_boundaries(self) -> None:
        rows = []
        for hour in (1, 2):
            for city, temperature in (("南京", 10.0), ("南通", 30.0)):
                rows.append(
                    {
                        "city": city,
                        "source": "Ts_weather_1",
                        "timestamp": f"2026-05-01 {hour:02d}:00:00",
                        "temp_2m": temperature,
                        "humidity_2m": 50.0,
                        "rain": 0.0,
                        "radiation": 0.0,
                        "wind_10m": 1.0,
                    }
                )
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weather.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            store = read_weather_snapshot(path, chunksize=3)
        self.assertTrue(np.allclose(store.hourly["temp_2m"], 20.0))

    def test_market_outcome_reader_enforces_cutoff_and_rt_only(self) -> None:
        cutoff = date(2026, 7, 22)
        rows = [
            {
                "trade_date": cutoff,
                "trade_interval": 1,
                "data_name": "SYSTEM_LOAD_POWER",
                "value": 100.0,
                "data_type": "RT",
            },
            {
                "trade_date": date(2026, 7, 23),
                "trade_interval": 1,
                "data_name": "SYSTEM_LOAD_POWER",
                "value": 999.0,
                "data_type": "RT",
            },
            {
                "trade_date": cutoff,
                "trade_interval": 1,
                "data_name": "SYSTEM_LOAD_POWER",
                "value": 888.0,
                "data_type": "DA",
            },
        ]
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "market.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            matrix, _ = read_visible_market_outcomes(
                path,
                start=cutoff,
                cutoff=cutoff,
            )
        self.assertEqual(set(matrix.index.get_level_values("trade_date")), {cutoff})
        self.assertEqual(matrix.iloc[0]["SYSTEM_LOAD_POWER"], 100.0)


if __name__ == "__main__":
    unittest.main()
