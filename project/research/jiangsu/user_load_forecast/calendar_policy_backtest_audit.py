"""Compare the legacy weekday policy with the official four-class calendar.

This audit rebuilds only the five operational fallback/market-fusion curves.
Both policies receive the same actual-load and market snapshots, so the
reported delta isolates calendar policy changes. Strict GBDT models are
validated from the separately supplied full operational backtest payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LOCAL_AGENT = (
    REPOSITORY_ROOT
    / "apps/jiangsu-load-forecast-dashboard/product-design-prototype/local_agent"
)
if str(LOCAL_AGENT) not in sys.path:
    sys.path.insert(0, str(LOCAL_AGENT))

from operational_runner import (  # noqa: E402
    FOUR_CLASS_CALENDAR_POLICY,
    LEGACY_CALENDAR_POLICY,
    build_payload,
    calendar_source_dates,
    market_lookup,
    read_actual,
    recent_source_dates,
)
from project.research.jiangsu.user_load_forecast.calendar_features import (  # noqa: E402
    day_type,
)


MODEL_IDS = (
    "operational_calendar_d3",
    "operational_recent_trend",
    "operational_d2_adjustment",
    "operational_deadline_fusion",
    "operational_da_d2_fusion",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_row(group: pd.DataFrame) -> dict[str, float | int]:
    actual = group["actual_load"].to_numpy(dtype=float)
    predicted = group["predicted_load"].to_numpy(dtype=float)
    denominator = np.abs(actual).sum()
    daily = []
    correlations = []
    energy_bias = []
    for _, day in group.groupby("target_date", sort=True):
        truth = day["actual_load"].to_numpy(dtype=float)
        estimate = day["predicted_load"].to_numpy(dtype=float)
        daily_denominator = np.abs(truth).sum()
        daily.append(np.abs(estimate - truth).sum() / daily_denominator * 100)
        correlation = float(np.corrcoef(truth, estimate)[0, 1])
        if math.isfinite(correlation):
            correlations.append(correlation)
        energy_bias.append(
            abs(estimate.sum() - truth.sum()) / daily_denominator * 100
        )
    return {
        "day_count": int(group["target_date"].nunique()),
        "wmape_pct": float(np.abs(predicted - actual).sum() / denominator * 100),
        "daily_wmape_p90_pct": float(np.quantile(daily, 0.9)),
        "daily_wmape_max_pct": float(max(daily)),
        "median_daily_curve_correlation": (
            float(np.median(correlations)) if correlations else math.nan
        ),
        "mean_abs_daily_energy_bias_pct": float(np.mean(energy_bias)),
    }


def prediction_rows(payload: dict, policy: str) -> list[dict]:
    rows = []
    for day in payload["days"]:
        if day["actual"] is None:
            continue
        for model_id in MODEL_IDS:
            prediction = day["predictions"].get(model_id)
            if prediction is None:
                continue
            for interval, (actual, predicted) in enumerate(
                zip(day["actual"], prediction, strict=True), start=1
            ):
                rows.append(
                    {
                        "calendar_policy": policy,
                        "model": model_id,
                        "target_date": day["date"],
                        # Use one canonical classifier for both policies. The
                        # legacy payload itself exposes only workday/nonwork.
                        "day_type": day_type(date.fromisoformat(day["date"])),
                        "trade_interval": interval,
                        "actual_load": actual,
                        "predicted_load": predicted,
                        "absolute_error": abs(predicted - actual),
                    }
                )
    return rows


def validate_full_payload(payload: dict) -> dict:
    expected_types = {"workday", "weekend", "holiday", "adjusted_workday"}
    observed_types = {day["dayType"] for day in payload["days"]}
    if observed_types != expected_types:
        raise ValueError(
            f"full backtest day types {sorted(observed_types)} != "
            f"{sorted(expected_types)}"
        )
    curve_count = 0
    for day in payload["days"]:
        for model_id, points in day["predictions"].items():
            if len(points) != 96 or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in points
            ):
                raise ValueError(f"invalid curve {day['date']}/{model_id}")
            curve_count += 1
    strict = payload.get("source", {}).get("strictModels", {})
    if not strict.get("available"):
        raise ValueError(f"strict models unavailable: {strict.get('reason')}")
    return {
        "day_count": len(payload["days"]),
        "curve_count": curve_count,
        "model_count": len(payload["models"]),
        "strict_available": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-input", type=Path, required=True)
    parser.add_argument("--market-input", type=Path, required=True)
    parser.add_argument("--full-v2-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "legacy": args.output_dir / "legacy-fallback-dashboard.json",
        "v2": args.output_dir / "calendar-aware-fallback-dashboard.json",
        "points": args.output_dir / "point_comparison.csv.gz",
        "daily": args.output_dir / "daily_metrics.csv",
        "summary": args.output_dir / "summary_metrics.csv",
        "metadata": args.output_dir / "run_metadata.json",
        "report": args.output_dir / "run_summary.md",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite audit outputs: {existing}")

    frame = read_actual(args.actual_input)
    market = market_lookup(args.market_input)
    payloads = {
        LEGACY_CALENDAR_POLICY: build_payload(
            frame,
            market,
            mode="backtest",
            target_date=None,
            data_readiness="ready",
            calendar_policy=LEGACY_CALENDAR_POLICY,
        ),
        FOUR_CLASS_CALENDAR_POLICY: build_payload(
            frame,
            market,
            mode="backtest",
            target_date=None,
            data_readiness="ready",
            calendar_policy=FOUR_CLASS_CALENDAR_POLICY,
        ),
    }
    outputs["legacy"].write_text(
        json.dumps(payloads[LEGACY_CALENDAR_POLICY], ensure_ascii=False),
        encoding="utf-8",
    )
    outputs["v2"].write_text(
        json.dumps(payloads[FOUR_CLASS_CALENDAR_POLICY], ensure_ascii=False),
        encoding="utf-8",
    )

    full_payload = json.loads(args.full_v2_result.read_text(encoding="utf-8"))
    validation = validate_full_payload(full_payload)
    full_days = {day["date"]: day for day in full_payload["days"]}
    for day in payloads[FOUR_CLASS_CALENDAR_POLICY]["days"]:
        for model_id, points in day["predictions"].items():
            full_points = full_days[day["date"]]["predictions"].get(model_id)
            if full_points != points:
                raise ValueError(
                    f"full/fallback v2 mismatch at {day['date']}/{model_id}"
                )

    rows = []
    for policy, payload in payloads.items():
        rows.extend(prediction_rows(payload, policy))
    points = pd.DataFrame(rows)
    points.to_csv(outputs["points"], index=False, compression="gzip")

    daily = (
        points.groupby(
            ["calendar_policy", "model", "target_date", "day_type"],
            as_index=False,
        )
        .agg(
            absolute_error_sum=("absolute_error", "sum"),
            actual_abs_sum=("actual_load", lambda value: value.abs().sum()),
            actual_total=("actual_load", "sum"),
            predicted_total=("predicted_load", "sum"),
        )
    )
    daily["wmape_pct"] = daily["absolute_error_sum"] / daily["actual_abs_sum"] * 100
    daily["abs_energy_bias_pct"] = (
        (daily["predicted_total"] - daily["actual_total"]).abs()
        / daily["actual_abs_sum"]
        * 100
    )
    daily.to_csv(outputs["daily"], index=False)

    summary_rows = []
    for (policy, model_id), group in points.groupby(
        ["calendar_policy", "model"], sort=True
    ):
        summary_rows.append(
            {
                "calendar_policy": policy,
                "model": model_id,
                "day_type": "all",
                **metric_row(group),
            }
        )
        for kind, typed in group.groupby("day_type", sort=True):
            summary_rows.append(
                {
                    "calendar_policy": policy,
                    "model": model_id,
                    "day_type": kind,
                    **metric_row(typed),
                }
            )
    summary = pd.DataFrame(summary_rows)
    legacy = summary[
        summary["calendar_policy"] == LEGACY_CALENDAR_POLICY
    ].set_index(["model", "day_type"])
    v2 = summary[
        summary["calendar_policy"] == FOUR_CLASS_CALENDAR_POLICY
    ].set_index(["model", "day_type"])
    common = legacy.index.intersection(v2.index)
    deltas = []
    for model_id, kind in common:
        deltas.append(
            {
                "calendar_policy": "v2_minus_legacy",
                "model": model_id,
                "day_type": kind,
                "day_count": int(v2.loc[(model_id, kind), "day_count"]),
                **{
                    column: float(
                        v2.loc[(model_id, kind), column]
                        - legacy.loc[(model_id, kind), column]
                    )
                    for column in (
                        "wmape_pct",
                        "daily_wmape_p90_pct",
                        "daily_wmape_max_pct",
                        "median_daily_curve_correlation",
                        "mean_abs_daily_energy_bias_pct",
                    )
                },
            }
        )
    summary = pd.concat([summary, pd.DataFrame(deltas)], ignore_index=True)
    summary.to_csv(outputs["summary"], index=False)

    history = {
        target: group.sort_values("trade_interval")["actual_load"].to_numpy()
        for target, group in frame.groupby("trade_date")
    }
    visibility_violations = []
    for target in sorted(history)[7:]:
        cutoff = target - timedelta(days=3)
        for source in (
            calendar_source_dates(history, target, FOUR_CLASS_CALENDAR_POLICY)
            + recent_source_dates(history, target, FOUR_CLASS_CALENDAR_POLICY)
        ):
            if source > cutoff:
                visibility_violations.append(
                    {"target": target.isoformat(), "source": source.isoformat()}
                )
    if visibility_violations:
        raise ValueError(f"D-3 visibility violations: {visibility_violations[:3]}")

    all_delta = pd.DataFrame(deltas)
    all_delta = all_delta[all_delta["day_type"] == "all"].sort_values("model")
    metadata = {
        "audit": "calendar_policy_backtest_audit_v1",
        "actual_input": str(args.actual_input),
        "actual_sha256": sha256(args.actual_input),
        "market_input": str(args.market_input),
        "market_sha256": sha256(args.market_input),
        "full_v2_result": str(args.full_v2_result),
        "full_v2_result_sha256": sha256(args.full_v2_result),
        "history_start": min(history).isoformat(),
        "history_end": max(history).isoformat(),
        "complete_actual_days": len(history),
        "forecast_visibility": "D_MINUS_3_OR_EARLIER",
        "visibility_violation_count": 0,
        "policies": [LEGACY_CALENDAR_POLICY, FOUR_CLASS_CALENDAR_POLICY],
        "full_payload_validation": validation,
        "selection_note": (
            "No weights or models were selected from this audit; it only "
            "compares a pre-specified calendar policy change."
        ),
    }
    outputs["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 江苏用户负荷四分类日历策略离线回测",
        "",
        f"- 实际负荷：{metadata['history_start']} 至 {metadata['history_end']}，"
        f"{metadata['complete_actual_days']} 个完整日。",
        "- 可见性：每个目标日仅使用 D-3 及更早用户实际负荷；自动检查无违规。",
        "- 对照：旧星期策略与中国官方四分类日历使用完全相同的负荷和市场快照。",
        "- 四分类：普通工作日、周末、法定节假日、调休工作日。",
        "",
        "## 全周期 WMAPE 变化（新版减旧版，负数为改善）",
        "",
        "| 模型 | WMAPE变化（百分点） | P90变化（百分点） | 最差日变化（百分点） |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in all_delta.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.wmape_pct:+.4f} | "
            f"{row.daily_wmape_p90_pct:+.4f} | {row.daily_wmape_max_pct:+.4f} |"
        )
    lines.extend(
        [
            "",
            "完整逐点预测、逐日指标、分日型汇总、输入哈希和全量严格模型产物均保存在本目录。",
        ]
    )
    outputs["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
