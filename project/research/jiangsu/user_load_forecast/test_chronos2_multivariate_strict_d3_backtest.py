from datetime import date

import numpy as np

from project.research.jiangsu.user_load_forecast.chronos2_multivariate_strict_d3_backtest import (
    forecast_target_contexts,
)


class _FakePipeline:
    def predict_quantiles(self, inputs, **_kwargs):
        assert len(inputs) == 3
        # All contexts must end at D-3 and preserve a missing natural day.
        assert inputs[0].shape[0] == 96
        assert np.isnan(inputs[0][:, 2]).all()
        outputs = []
        for values in inputs:
            forecast = np.zeros((96, 3, 1), dtype=np.float32)
            forecast[:, 2, 0] = values[:, -1]
            outputs.append(_FakeTensor(forecast))
        return outputs, outputs


class _FakeTensor:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, item):
        return _FakeTensor(self.values[item])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def test_multivariate_forecast_uses_d3_and_preserves_calendar_gap():
    curves = {
        date(2026, 4, 1): np.ones(96),
        date(2026, 4, 2): np.ones(96) * 2,
        # 2026-04-03 deliberately missing.
        date(2026, 4, 4): np.ones(96) * 4,
        date(2026, 4, 5): np.ones(96) * 5,
    }
    outputs = forecast_target_contexts(
        _FakePipeline(),
        curves,
        target_date=date(2026, 4, 8),
    )
    prediction, context_start, cutoff, missing = outputs[
        "chronos2__multivariate96_daily_ctxall"
    ]
    assert context_start == date(2026, 4, 1)
    assert cutoff == date(2026, 4, 5)
    assert missing == 96
    assert np.all(prediction == 5)
