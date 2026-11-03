"""Audit fixed Jiangsu Chronos candidates across time and promotion gates.

This script performs no tuning.  It compares the previously frozen weather
anchor, Chronos-Bolt blend, and Chronos-2 blend over disjoint time segments,
day types, and time blocks, then applies conservative promotion gates.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.model_backtest import prepare_output_dir
from project.research.jiangsu.user_load_forecast.strict_operational_weather_backtest import (
    sha256_file,
)


VERSION = "surun_chronos_candidate_stability_audit_v1"
ANCHOR = "anchor__anchor_core_weather"
BOLT = "blend_a0.25__chronos_bolt_small__daily96_ctxall__anchor_core_weather"
CHRONOS2 = "blend_a0.5__chronos2__multivariate96_daily_ctx42__anchor_core_weather"
PERIODS = (
    ("jun15_jun30", "2026-06-15", "2026-06-30"),
    ("jul01_jul15", "2026-07-01", "2026-07-15"),
    ("jul16_jul25", "2026-07-16", "2026-07-25"),
    ("jul26_aug05", "2026-07-26", "2026-08-05"),
)


def wmape(frame: pd.DataFrame) -> float:
    return float(100.0 * frame["absolute_error"].sum() / frame["actual_load"].abs().sum())


def metric_row(frame: pd.DataFrame, *, group_kind: str, group: str, model: str) -> dict[str, object]:
    daily = frame.groupby("target_date").agg(
        absolute_error=("absolute_error", "sum"),
        actual_load=("actual_load", lambda values: values.abs().sum()),
    )
    daily_wmape = 100.0 * daily["absolute_error"] / daily["actual_load"]
    return {
        "group_kind": group_kind,
        "group": group,
        "model": model,
        "target_days": frame["target_date"].nunique(),
        "wmape_pct": wmape(frame),
        "daily_wmape_p90_pct": float(daily_wmape.quantile(0.9)),
        "daily_wmape_max_pct": float(daily_wmape.max()),
        "bias_pct": float(100.0 * frame["error"].sum() / frame["actual_load"].abs().sum()),
    }


def paired_bootstrap(frame: pd.DataFrame, model: str, comparator: str) -> dict[str, object]:
    grouped = frame.loc[frame["model"].isin([model, comparator])].groupby(
        ["model", "target_date"]
    ).agg(
        absolute_error=("absolute_error", "sum"),
        actual_load=("actual_load", lambda values: values.abs().sum()),
    )
    ae = grouped["absolute_error"].unstack("model")
    actual = grouped["actual_load"].unstack("model")
    n_days = len(ae)
    rng = np.random.default_rng(20260819)
    indices = rng.integers(0, n_days, size=(50_000, n_days))
    ae_values = ae[[model, comparator]].to_numpy()
    actual_values = actual[[model, comparator]].to_numpy()
    estimates = 100.0 * np.take(ae_values, indices, axis=0).sum(axis=1) / np.take(
        actual_values, indices, axis=0
    ).sum(axis=1)
    delta = estimates[:, 0] - estimates[:, 1]
    return {
        "model": model,
        "comparator": comparator,
        "target_days": n_days,
        "delta_wmape_pp": wmape(frame.loc[frame["model"].eq(model)])
        - wmape(frame.loc[frame["model"].eq(comparator)]),
        "delta_ci025_pp": float(np.quantile(delta, 0.025)),
        "delta_ci975_pp": float(np.quantile(delta, 0.975)),
        "probability_better": float((delta < 0).mean()),
    }


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    new = pd.read_csv(args.chronos2_predictions, low_memory=False)
    old = pd.read_csv(args.bolt_predictions, low_memory=False)
    new = new.loc[new["model"].isin([CHRONOS2, ANCHOR])].copy()
    old = old.loc[old["model"].eq(BOLT)].copy()
    frame = pd.concat([new, old], ignore_index=True)
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    day_type_map = (
        frame.loc[frame["model"].eq(ANCHOR)]
        .dropna(subset=["day_type"])
        .groupby("target_date")["day_type"]
        .first()
    )
    frame["audit_day_type"] = frame["target_date"].map(day_type_map)
    frame["time_block"] = pd.cut(
        frame["trade_interval"],
        [0, 24, 48, 72, 96],
        labels=["00_06", "06_12", "12_18", "18_24"],
    )

    rows = []
    for model, part in frame.groupby("model"):
        rows.append(metric_row(part, group_kind="all", group="all", model=model))
        for period, start, end in PERIODS:
            scoped = part.loc[part["target_date"].between(start, end)]
            rows.append(metric_row(scoped, group_kind="period", group=period, model=model))
        for day_type, scoped in part.groupby("audit_day_type"):
            rows.append(metric_row(scoped, group_kind="day_type", group=str(day_type), model=model))
        for block, scoped in part.groupby("time_block", observed=True):
            rows.append(metric_row(scoped, group_kind="time_block", group=str(block), model=model))
        iso = part["target_date"].dt.isocalendar()
        weekly = part.assign(week=iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2))
        for week, scoped in weekly.groupby("week"):
            rows.append(metric_row(scoped, group_kind="week", group=str(week), model=model))
    metrics = pd.DataFrame(rows).sort_values(["group_kind", "group", "wmape_pct", "model"])

    bootstraps = pd.DataFrame(
        [
            paired_bootstrap(frame, CHRONOS2, BOLT),
            paired_bootstrap(frame, CHRONOS2, ANCHOR),
            paired_bootstrap(frame, BOLT, ANCHOR),
        ]
    )
    overall = metrics.loc[metrics["group_kind"].eq("all")].set_index("model")
    period = metrics.loc[metrics["group_kind"].eq("period")].pivot(
        index="group", columns="model", values="wmape_pct"
    )
    chronos2_beats_anchor_periods = int((period[CHRONOS2] < period[ANCHOR]).sum())
    relative_improvement_vs_bolt = float(
        100.0 * (overall.loc[BOLT, "wmape_pct"] - overall.loc[CHRONOS2, "wmape_pct"])
        / overall.loc[BOLT, "wmape_pct"]
    )
    absolute_improvement_vs_bolt = float(
        overall.loc[BOLT, "wmape_pct"] - overall.loc[CHRONOS2, "wmape_pct"]
    )
    gates = {
        "minimum_new_diagnostic_days_28": False,
        "improvement_vs_current_at_least_2pct_or_0_10pp": bool(
            relative_improvement_vs_bolt >= 2.0 or absolute_improvement_vs_bolt >= 0.10
        ),
        "p90_not_worse_than_current": bool(
            overall.loc[CHRONOS2, "daily_wmape_p90_pct"]
            <= overall.loc[BOLT, "daily_wmape_p90_pct"]
        ),
        "holiday_not_worse_than_current": bool(
            metrics.loc[
                metrics["group_kind"].eq("day_type")
                & metrics["group"].eq("holiday")
                & metrics["model"].eq(CHRONOS2),
                "wmape_pct",
            ].iloc[0]
            <= metrics.loc[
                metrics["group_kind"].eq("day_type")
                & metrics["group"].eq("holiday")
                & metrics["model"].eq(BOLT),
                "wmape_pct",
            ].iloc[0]
        ),
        "beats_anchor_in_at_least_3_of_4_periods": chronos2_beats_anchor_periods >= 3,
    }
    promotion = all(gates.values())
    metadata = {
        "version": VERSION,
        "built_at": datetime.now().astimezone().isoformat(),
        "status": "PROMOTION_CANDIDATE" if promotion else "NO_PROMOTION",
        "network_mode": "offline",
        "selection_policy": "NO_TUNING_FIXED_CANDIDATE_AUDIT",
        "models": [ANCHOR, BOLT, CHRONOS2],
        "relative_improvement_vs_bolt_pct": relative_improvement_vs_bolt,
        "absolute_improvement_vs_bolt_pp": absolute_improvement_vs_bolt,
        "chronos2_beats_anchor_periods": chronos2_beats_anchor_periods,
        "promotion_gates": gates,
        "promotion_recommended": promotion,
        "chronos2_predictions": str(args.chronos2_predictions.resolve()),
        "chronos2_predictions_sha256": sha256_file(args.chronos2_predictions),
        "bolt_predictions": str(args.bolt_predictions.resolve()),
        "bolt_predictions_sha256": sha256_file(args.bolt_predictions),
        "delay_stress_metrics": str(args.delay_stress_metrics.resolve()),
        "delay_stress_metrics_sha256": sha256_file(args.delay_stress_metrics),
    }
    metrics.to_csv(args.output_dir / "stability_metrics.csv", index=False, encoding="utf-8-sig")
    bootstraps.to_csv(args.output_dir / "paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "run_summary.md").write_text(
        "# Chronos 候选扩展稳定性审计\n\n"
        f"- Chronos-2 全49日 WMAPE：{overall.loc[CHRONOS2, 'wmape_pct']:.4f}%。\n"
        f"- Bolt 全49日 WMAPE：{overall.loc[BOLT, 'wmape_pct']:.4f}%。\n"
        f"- Chronos-2 相对 Bolt 改善：{relative_improvement_vs_bolt:.2f}%，"
        f"绝对 {absolute_improvement_vs_bolt:.4f}pp。\n"
        f"- Chronos-2 优于气象锚点的时间段：{chronos2_beats_anchor_periods}/4。\n"
        f"- 晋级结论：{'通过' if promotion else '不通过'}。\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(metrics.loc[metrics["group_kind"].isin(["all", "period"])].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chronos2-predictions", type=Path, required=True)
    parser.add_argument("--bolt-predictions", type=Path, required=True)
    parser.add_argument("--delay-stress-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
