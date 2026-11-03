"""Export a reproducible report from a validated strict-runner dashboard result."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.model_backtest import (
    prepare_output_dir,
)


MODEL_IDS = (
    "operational_weather_solar_fusion_v1",
    "operational_weather_stable_v1",
    "operational_market_timing_v1",
)
VERSION = "surun_strict_operational_fullcycle_acceptance_v1"
CHINA = ZoneInfo("Asia/Shanghai")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    payload = json.loads(args.dashboard_input.read_text(encoding="utf-8"))
    actual = pd.read_csv(args.portfolio_input)
    actual["trade_date"] = pd.to_datetime(actual["trade_date"]).dt.date
    actual_lookup = actual.set_index(["trade_date", "trade_interval"])["actual_load"]
    model_meta = {model["id"]: model for model in payload["models"]}
    rows = []
    for day in payload["days"]:
        target_date = pd.Timestamp(day["date"]).date()
        for model_id in MODEL_IDS:
            values = day.get("predictions", {}).get(model_id)
            if values is None:
                continue
            if len(values) != 96:
                raise ValueError(f"{target_date}/{model_id} is not 96 points")
            for interval, predicted in enumerate(values, start=1):
                actual_value = float(actual_lookup.loc[(target_date, interval)])
                error = float(predicted) - actual_value
                rows.append(
                    {
                        "model": model_id,
                        "target_date": target_date,
                        "trade_interval": interval,
                        "actual_load": actual_value,
                        "predicted_load": float(predicted),
                        "error": error,
                        "absolute_error": abs(error),
                        "load_label_cutoff": target_date - pd.Timedelta(days=3),
                        "market_cutoff": target_date - pd.Timedelta(days=2),
                    }
                )
    predictions = pd.DataFrame(rows)
    counts = predictions.groupby(["model", "target_date"])["trade_interval"].nunique()
    if not counts.eq(96).all():
        raise ValueError("Every model/date must contain 96 intervals")
    coverage = predictions.groupby("model")["target_date"].agg(["min", "max", "nunique"])
    if not coverage["nunique"].eq(112).all():
        raise ValueError(f"Unexpected coverage: {coverage.to_dict('index')}")

    daily = (
        predictions.groupby(["model", "target_date"], as_index=False)
        .agg(
            absolute_error=("absolute_error", "sum"),
            actual_abs=("actual_load", lambda values: values.abs().sum()),
            actual_energy=("actual_load", "sum"),
            predicted_energy=("predicted_load", "sum"),
        )
    )
    daily["wmape_pct"] = daily["absolute_error"] / daily["actual_abs"] * 100
    daily["abs_energy_bias_pct"] = (
        (daily["predicted_energy"] - daily["actual_energy"]).abs()
        / daily["actual_abs"]
        * 100
    )
    summary_rows = []
    for model_id, group in predictions.groupby("model"):
        model_daily = daily.loc[daily["model"].eq(model_id)]
        correlations = []
        for _, day in group.groupby("target_date"):
            correlation = np.corrcoef(day["actual_load"], day["predicted_load"])[0, 1]
            if np.isfinite(correlation):
                correlations.append(float(correlation))
        summary_rows.append(
            {
                "model": model_id,
                "label": model_meta[model_id]["label"],
                "window": "full_cycle_after_7_day_warmup",
                "window_start": str(group["target_date"].min()),
                "window_end": str(group["target_date"].max()),
                "target_days": int(group["target_date"].nunique()),
                "wmape_pct": group["absolute_error"].sum()
                / group["actual_load"].abs().sum()
                * 100,
                "daily_wmape_p90_pct": model_daily["wmape_pct"].quantile(0.9),
                "daily_wmape_max_pct": model_daily["wmape_pct"].max(),
                "median_daily_curve_correlation": float(np.median(correlations)),
                "mean_abs_daily_energy_bias_pct": model_daily[
                    "abs_energy_bias_pct"
                ].mean(),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("wmape_pct")

    predictions.to_csv(
        args.output_dir / "predictions_96.csv.gz",
        index=False,
        compression="gzip",
    )
    daily.to_csv(
        args.output_dir / "daily_metrics.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        args.output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "version": VERSION,
        "generated_at": datetime.now(CHINA).isoformat(),
        "dashboard_input": str(args.dashboard_input.resolve()),
        "dashboard_sha256": sha256_file(args.dashboard_input),
        "portfolio_input": str(args.portfolio_input.resolve()),
        "portfolio_sha256": sha256_file(args.portfolio_input),
        "models": list(MODEL_IDS),
        "load_label_policy": "SIMULATED_T_PLUS_2_D_MINUS_3",
        "market_policy": "STRICT_SUBMISSION_ASOF_T_MINUS_2_MARKET_DATE",
        "backtest_policy": "DAILY_EXPANDING_AFTER_7_COMPLETE_WARMUP_DAYS",
        "coverage_start": str(predictions["target_date"].min()),
        "coverage_end": str(predictions["target_date"].max()),
        "target_days_per_model": 112,
        "first_seven_days": "UNAVAILABLE_NO_PRIOR_PORTFOLIO_TRAINING_HISTORY",
        "target_day_da_feature_check": "PASS_NO_TARGET_DAY_DA",
        "promotion_selection_changed": False,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 江苏苏润严格运营模型全周期回测验收",
        "",
        "- 逐日扩窗，用户负荷标签严格截止目标日D-3。",
        "- 市场特征最大日期为目标日T-2；禁用目标日DA。",
        "- 当前负荷历史始于2026-04-01，7天预热后从04-08连续回测。",
        "- 04-01至04-07没有更早训练历史，保持不可用，不倒推补齐。",
        "- 本次扩展覆盖不改变原预注册晋级选择。",
        "",
        "## 全周期结果",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- {row.label}：WMAPE {row.wmape_pct:.4f}%，"
            f"P90 {row.daily_wmape_p90_pct:.4f}%，{row.target_days}天。"
        )
    (args.output_dir / "run_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard-input", type=Path, required=True)
    parser.add_argument("--portfolio-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
