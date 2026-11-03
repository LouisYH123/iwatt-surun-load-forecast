"""Run strict T+2 walk-forward baselines for the Surun 96-point load target."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.contracts import (
    LATENCY_POLICY,
    build_availability_ledger,
    validate_visible_source_dates,
)
from project.research.jiangsu.user_load_forecast.load_data import (
    AggregatedLoadDataset,
    interval_end_label,
    read_aggregated_load_snapshot,
)


METHOD_VERSION = "surun_baseline_v1.0"
BASELINES = {
    "d7": (7,),
    "same_weekday_mean_4": (7, 14, 21, 28),
}
DEFAULT_OUTPUT_DIR = Path(
    "project/reports/provinces/jiangsu/user_load_forecast/"
    "backtests/baseline_v1_20260501_20260725"
)


def parse_clock(value: str) -> time:
    try:
        hour, minute = (int(item) for item in value.split(":", maxsplit=1))
        return time(hour, minute)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid clock {value!r}; expected HH:MM"
        ) from exc


def date_range(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("target_end must not be earlier than target_start")
    return [
        start + timedelta(days=offset)
        for offset in range((end - start).days + 1)
    ]


def run_backtest(
    dataset: AggregatedLoadDataset,
    *,
    target_start: date,
    target_end: date,
    forecast_as_of_time: time = time(8, 45),
    extra_delay_days: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate point-level out-of-sample predictions and the availability ledger."""
    target_dates = date_range(target_start, target_end)
    missing_targets = sorted(set(target_dates) - set(dataset.curves))
    if missing_targets:
        raise ValueError(f"Missing target labels for evaluation: {missing_targets}")

    ledger = build_availability_ledger(
        target_dates,
        set(dataset.curves),
        forecast_as_of_time=forecast_as_of_time,
        extra_delay_days=extra_delay_days,
    )
    ledger_by_target = ledger.set_index("target_date").to_dict(orient="index")
    rows: list[dict] = []

    for target_date in target_dates:
        contract = ledger_by_target[target_date]
        latest_available = contract["latest_available_label_date"]
        actual = dataset.curves[target_date]
        for model_name, lags in BASELINES.items():
            source_dates = [
                target_date - timedelta(days=lag_days) for lag_days in lags
            ]
            unavailable = [item for item in source_dates if item not in dataset.curves]
            if unavailable:
                raise ValueError(
                    f"{model_name} cannot forecast {target_date}; "
                    f"missing sources={unavailable}"
                )
            validate_visible_source_dates(
                target_date=target_date,
                latest_available_label_date=latest_available,
                source_dates=source_dates,
            )
            prediction = np.mean(
                np.vstack([dataset.curves[item] for item in source_dates]),
                axis=0,
            )
            for interval in range(1, 97):
                actual_value = float(actual[interval - 1])
                prediction_value = float(prediction[interval - 1])
                rows.append(
                    {
                        "model": model_name,
                        "run_date": contract["run_date"],
                        "target_date": target_date,
                        "forecast_as_of": contract["forecast_as_of"],
                        "latest_available_label_date": latest_available,
                        "label_age_days": contract["label_age_days"],
                        "latency_level": contract["latency_level"],
                        "trade_interval": interval,
                        "interval_end": interval_end_label(interval),
                        "source_dates": "|".join(item.isoformat() for item in source_dates),
                        "source_min_date": min(source_dates),
                        "source_max_date": max(source_dates),
                        "actual_load": actual_value,
                        "predicted_load": prediction_value,
                        "error": prediction_value - actual_value,
                        "absolute_error": abs(prediction_value - actual_value),
                    }
                )
    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise ValueError("Backtest generated no predictions")
    if not (
        pd.to_datetime(predictions["source_max_date"]).dt.date
        <= pd.to_datetime(predictions["latest_available_label_date"]).dt.date
    ).all():
        raise AssertionError("Leakage audit failed after prediction generation")
    return predictions, ledger


def daily_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, target_date), group in predictions.groupby(
        ["model", "target_date"],
        sort=True,
    ):
        denominator = group["actual_load"].abs().sum()
        if denominator <= 0:
            raise ValueError(f"Non-positive WMAPE denominator for {target_date}")
        actual_total = group["actual_load"].sum()
        predicted_total = group["predicted_load"].sum()
        rows.append(
            {
                "model": model,
                "target_date": target_date,
                "points": len(group),
                "wmape_pct": group["absolute_error"].sum() / denominator * 100,
                "mae": group["absolute_error"].mean(),
                "actual_total": actual_total,
                "predicted_total": predicted_total,
                "total_bias_pct": (
                    (predicted_total - actual_total) / actual_total * 100
                    if actual_total
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def summary_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, model_frame in predictions.groupby("model", sort=True):
        segments = [("overall", "all", model_frame)]
        segments.extend(
            ("month", str(month), frame)
            for month, frame in model_frame.groupby(
                pd.to_datetime(model_frame["target_date"]).dt.month,
                sort=True,
            )
        )
        for segment_type, segment, frame in segments:
            denominator = frame["actual_load"].abs().sum()
            rows.append(
                {
                    "model": model,
                    "segment_type": segment_type,
                    "segment": segment,
                    "target_days": frame["target_date"].nunique(),
                    "points": len(frame),
                    "absolute_error_sum": frame["absolute_error"].sum(),
                    "actual_abs_sum": denominator,
                    "wmape_pct": frame["absolute_error"].sum() / denominator * 100,
                    "mae": frame["absolute_error"].mean(),
                }
            )
    return pd.DataFrame(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty backtest directory: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def write_outputs(
    *,
    output_dir: Path,
    dataset: AggregatedLoadDataset,
    predictions: pd.DataFrame,
    ledger: pd.DataFrame,
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    target_start: date,
    target_end: date,
    forecast_as_of_time: time,
    extra_delay_days: int,
) -> dict:
    prepare_output_dir(output_dir)
    frames = {
        "aggregated_load_96.csv": dataset.to_long_frame(),
        "availability_ledger.csv": ledger,
        "predictions_96.csv": predictions,
        "daily_metrics.csv": daily,
        "summary_metrics.csv": summary,
    }
    for name, frame in frames.items():
        frame.to_csv(output_dir / name, index=False, encoding="utf-8-sig")

    overall = summary.loc[summary["segment_type"].eq("overall")].copy()
    leakage_pass = bool(
        (
            pd.to_datetime(predictions["source_max_date"]).dt.date
            <= pd.to_datetime(
                predictions["latest_available_label_date"]
            ).dt.date
        ).all()
    )
    metadata = {
        "method_version": METHOD_VERSION,
        "province_display": "江苏",
        "province_slug": "jiangsu",
        "raw_schema": "jiangsu",
        "target": "苏润组合96点实际负荷聚合值",
        "target_unit": (
            dataset.value_units[0]
            if len(dataset.value_units) == 1
            else list(dataset.value_units)
        ),
        "timezone": "Asia/Shanghai",
        "latency_policy": LATENCY_POLICY,
        "target_start": target_start.isoformat(),
        "target_end": target_end.isoformat(),
        "forecast_as_of_time": forecast_as_of_time.isoformat(timespec="minutes"),
        "extra_delay_days": extra_delay_days,
        "source_path": dataset.source_path,
        "source_sha256": dataset.source_sha256,
        "source_start_date": dataset.dates[0].isoformat(),
        "source_end_date": dataset.dates[-1].isoformat(),
        "source_calendar_days": len(dataset.dates),
        "source_account_day_rows": dataset.row_count,
        "source_distinct_accounts": dataset.distinct_account_count,
        "source_distinct_users": dataset.distinct_user_count,
        "source_distinct_retailers": dataset.distinct_retailer_count,
        "source_negative_values": dataset.negative_value_count,
        "models": {key: list(value) for key, value in BASELINES.items()},
        "prediction_rows": len(predictions),
        "leakage_check": {
            "status": "PASS" if leakage_pass else "FAIL",
            "rule": "every actual-load source_date <= latest_available_label_date",
        },
        "overall_wmape_pct": {
            row["model"]: row["wmape_pct"]
            for row in overall.to_dict(orient="records")
        },
    }

    output_files = {}
    for name in frames:
        path = output_dir / name
        output_files[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    metadata["output_files"] = output_files
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result_lines = [
        "# 苏润组合负荷强基线回放",
        "",
        f"- 方法版本：`{METHOD_VERSION}`",
        f"- 回放日期：{target_start} 至 {target_end}",
        f"- 可见性：`{LATENCY_POLICY}`，额外延迟 {extra_delay_days} 天",
        f"- 输入校验值：`{dataset.source_sha256}`",
        f"- 防泄漏检查：{'PASS' if leakage_pass else 'FAIL'}",
        "",
        "## 全周期结果",
        "",
        "| 模型 | WMAPE |",
        "|---|---:|",
    ]
    for row in overall.sort_values("model").itertuples(index=False):
        result_lines.append(f"| {row.model} | {row.wmape_pct:.3f}% |")
    result_lines.extend(
        [
            "",
            "所有预测仅使用目标日前、且不晚于该次回放最新可用标签日的实际负荷。",
            "本版本未使用天气、市场边界、用户状态或目标日实际值作为输入。",
            "",
        ]
    )
    (output_dir / "run_summary.md").write_text(
        "\n".join(result_lines),
        encoding="utf-8",
    )
    return metadata


def run(args: argparse.Namespace) -> dict:
    dataset = read_aggregated_load_snapshot(args.input)
    if args.expected_input_sha256:
        expected = args.expected_input_sha256.lower()
        if dataset.source_sha256.lower() != expected:
            raise ValueError(
                f"Input SHA256 mismatch: expected={expected}, "
                f"actual={dataset.source_sha256}"
            )
    predictions, ledger = run_backtest(
        dataset,
        target_start=args.target_start,
        target_end=args.target_end,
        forecast_as_of_time=args.forecast_as_of_time,
        extra_delay_days=args.extra_delay_days,
    )
    daily = daily_metrics(predictions)
    summary = summary_metrics(predictions)
    return write_outputs(
        output_dir=args.output_dir,
        dataset=dataset,
        predictions=predictions,
        ledger=ledger,
        daily=daily,
        summary=summary,
        target_start=args.target_start,
        target_end=args.target_end,
        forecast_as_of_time=args.forecast_as_of_time,
        extra_delay_days=args.extra_delay_days,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--target-start",
        type=date.fromisoformat,
        default=date(2026, 5, 1),
    )
    parser.add_argument(
        "--target-end",
        type=date.fromisoformat,
        default=date(2026, 7, 25),
    )
    parser.add_argument(
        "--forecast-as-of-time",
        type=parse_clock,
        default=time(8, 45),
    )
    parser.add_argument("--extra-delay-days", type=int, default=0)
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run(args)
    print(json.dumps(metadata["overall_wmape_pct"], ensure_ascii=False, indent=2))
    print(f"Saved baseline backtest to: {args.output_dir}")


if __name__ == "__main__":
    main()
