from datetime import date

import numpy as np

from project.research.jiangsu.user_load_forecast.chronos_foundation_delay_stress_backtest import (
    local_model_snapshot,
    same_weekday_mean,
)


def test_same_weekday_mean_respects_cutoff():
    curves = {
        date(2026, 4, 1): np.ones(96),
        date(2026, 4, 8): np.ones(96) * 2,
        date(2026, 4, 15): np.ones(96) * 3,
        date(2026, 4, 22): np.ones(96) * 4,
        date(2026, 4, 29): np.ones(96) * 100,
    }
    result = same_weekday_mean(
        curves,
        target_date=date(2026, 5, 6),
        cutoff=date(2026, 4, 28),
    )
    assert np.all(result == 2.5)


def test_model_snapshots_resolve_from_local_cache():
    assert local_model_snapshot("amazon/chronos-2")
    assert local_model_snapshot("amazon/chronos-bolt-small")
