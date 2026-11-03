"""Rescore immutable prediction artifacts against a newer portfolio snapshot.

This utility never refits a model or changes a prediction.  It replaces only
the observed load/error columns where a newer portfolio curve is available and
writes a separate, auditable output directory.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.model_backtest import (
    prepare_output_dir,
)
from project.research.jiangsu.user_load_forecast.pv_sensitive_segment_backtest import (
    read_portfolio_curves,
)
from project.research.jiangsu.user_load_forecast.pv_weather_challenger_audit import (
    score_window,
)
from project.research.jiangsu.user_load_forecast.strict_operational_weather_backtest import (
    daily_metrics,
    sha256_file,
)


VERSION = "surun_prediction_artifact_rescore_v1"


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    predictions = pd.read_csv(args.predictions_input)
    predictions["target_date"] = pd.to_datetime(predictions["target_date"]).dt.date
    predictions["trade_interval"] = pd.to_numeric(
        predictions["trade_interval"], errors="raise"
    ).astype(int)
    curves = read_portfolio_curves(args.portfolio_input)
    old_actual = predictions["actual_load"].to_numpy(dtype=float).copy()
    refreshed = []
    for row in predictions.itertuples(index=False):
        curve = curves.get(row.target_date)
        refreshed.append(
            float(curve[row.trade_interval - 1])
            if curve is not None
            else float(row.actual_load)
        )
    predictions["actual_load"] = refreshed
    predictions["error"] = predictions["predicted_load"] - predictions["actual_load"]
    predictions["absolute_error"] = predictions["error"].abs()
    changed = ~np.isclose(old_actual, predictions["actual_load"].to_numpy(dtype=float))
    changed_dates = sorted(
        {value.isoformat() for value in predictions.loc[changed, "target_date"]}
    )

    rows = []
    for model in sorted(predictions["model"].astype(str).unique()):
        model_frame = predictions.loc[predictions["model"].eq(model)]
        for window, start, end in (
            ("legacy_holdout", args.holdout_start, args.legacy_holdout_end),
            ("new_diagnostic", args.new_diagnostic_start, args.new_diagnostic_end),
        ):
            score = score_window(model_frame, window=window, start=start, end=end)
            if score:
                rows.append(score)
    summary = pd.DataFrame(rows).sort_values(["window", "wmape_pct", "model"])
    daily = daily_metrics(predictions)

    predictions.to_csv(
        args.output_dir / "predictions_96.csv.gz", index=False, compression="gzip"
    )
    daily.to_csv(
        args.output_dir / "daily_metrics.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        args.output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "version": VERSION,
        "built_at": datetime.now().astimezone().isoformat(),
        "rescore_only_no_refit": True,
        "prediction_values_unchanged": True,
        "predictions_input": str(args.predictions_input.resolve()),
        "predictions_input_sha256": sha256_file(args.predictions_input),
        "portfolio_input": str(args.portfolio_input.resolve()),
        "portfolio_sha256": sha256_file(args.portfolio_input),
        "changed_point_rows": int(changed.sum()),
        "changed_target_dates": changed_dates,
        "windows": {
            "legacy_holdout": [
                args.holdout_start.isoformat(),
                args.legacy_holdout_end.isoformat(),
            ],
            "new_diagnostic": [
                args.new_diagnostic_start.isoformat(),
                args.new_diagnostic_end.isoformat(),
            ],
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "run_summary.md").write_text(
        "# 江苏用户负荷预测：最新实绩重评分\n\n"
        "- 只刷新真实值和误差，预测值未改变，模型未重训。\n"
        f"- 发生变化的日期：{', '.join(changed_dates) or '无'}。\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(summary[["model", "window", "wmape_pct"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-input", type=Path, required=True)
    parser.add_argument("--portfolio-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--holdout-start", type=date.fromisoformat, default=date(2026, 6, 15)
    )
    parser.add_argument(
        "--legacy-holdout-end", type=date.fromisoformat, default=date(2026, 7, 25)
    )
    parser.add_argument(
        "--new-diagnostic-start", type=date.fromisoformat, default=date(2026, 8, 1)
    )
    parser.add_argument(
        "--new-diagnostic-end", type=date.fromisoformat, default=date(2026, 8, 4)
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
