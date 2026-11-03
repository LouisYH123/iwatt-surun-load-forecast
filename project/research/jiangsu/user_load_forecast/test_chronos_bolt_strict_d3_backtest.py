from datetime import date

import numpy as np

from project.research.jiangsu.user_load_forecast.chronos_bolt_strict_d3_backtest import (
    complete_matrix,
)


def test_complete_matrix_ends_at_d_minus_three_and_preserves_missing_days() -> None:
    curves = {
        date(2026, 4, 1): np.full(96, 1.0),
        date(2026, 4, 3): np.full(96, 3.0),
        date(2026, 4, 6): np.full(96, 999999.0),
    }

    dates, matrix = complete_matrix(
        curves,
        cutoff=date(2026, 4, 3),
        context_days=None,
    )

    assert dates == [date(2026, 4, 1), date(2026, 4, 2), date(2026, 4, 3)]
    np.testing.assert_allclose(matrix[0], 1.0)
    assert np.isnan(matrix[1]).all()
    np.testing.assert_allclose(matrix[2], 3.0)
    assert np.nanmax(matrix) < 999999.0
