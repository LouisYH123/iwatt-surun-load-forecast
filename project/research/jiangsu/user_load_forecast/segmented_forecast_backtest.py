"""Backtest whether industrial and operational segmentation improves load forecasts.

Segment definitions are frozen using only data visible at the first target
date's D-3 cutoff.  Each segment is forecast independently and then summed.
Optional public-market features contain RT outcomes from D-3 or earlier only.
No target-period behavior is used to redefine the historical segments.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.calendar_baseline import (
    run_calendar_baseline,
    select_similar_dates,
)
from project.research.jiangsu.user_load_forecast.features import (
    build_feature_frame,
)
from project.research.jiangsu.user_load_forecast.industrial_behavior_research import (
    build_user_behavior_features,
    cluster_user_behavior,
    read_visible_market_outcomes,
)
from project.research.jiangsu.user_load_forecast.load_data import (
    AggregatedLoadDataset,
    read_aggregated_load_snapshot,
)
from project.research.jiangsu.user_load_forecast.model_backtest import (
    prepare_output_dir,
    walk_forward_model,
)
from project.research.jiangsu.user_load_forecast.user_segmentation import (
    SegmentedUserPanel,
    read_segmented_user_panel,
)


VERSION = "surun_segmented_forecast_backtest_v1"
CALENDAR_ANCHOR = "baseline_calendar_similar_day_mean4"


@dataclass(frozen=True)
class Segmentation:
    name: str
    assignment: dict[str, str]
    profile: pd.DataFrame


def _visible_user_energy(
    panel: SegmentedUserPanel,
    cutoff: date,
) -> dict[str, float]:
    values = {}
    for user_id, curves in panel.curves.items():
        value = sum(
            max(float(curve.sum()), 0.0)
            for trade_date, curve in curves.items()
            if trade_date <= cutoff
        )
        if value > 0:
            values[user_id] = value
    return values


def _collapse_small_groups(
    raw: dict[str, str],
    energy: dict[str, float],
    *,
    minimum_share: float,
    other_label: str,
) -> dict[str, str]:
    totals: dict[str, float] = defaultdict(float)
    for user_id, label in raw.items():
        totals[label] += energy.get(user_id, 0.0)
    total = sum(totals.values())
    keep = {
        label
        for label, value in totals.items()
        if total and value / total >= minimum_share
    }
    return {
        user_id: label if label in keep else other_label
        for user_id, label in raw.items()
    }


def _profile_assignment(
    name: str,
    assignment: dict[str, str],
    energy: dict[str, float],
) -> pd.DataFrame:
    total = sum(energy.values())
    rows = []
    for segment in sorted(set(assignment.values())):
        users = [user_id for user_id, label in assignment.items() if label == segment]
        visible = [user_id for user_id in users if user_id in energy]
        segment_energy = sum(energy[user_id] for user_id in visible)
        rows.append(
            {
                "segmentation": name,
                "segment": segment,
                "user_count": len(users),
                "visible_user_count": len(visible),
                "pretest_positive_energy": segment_energy,
                "pretest_load_share": segment_energy / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "pretest_load_share",
        ascending=False,
    )


def build_segmentations(
    panel: SegmentedUserPanel,
    *,
    cutoff: date,
    recent_days: int = 28,
    clusters: int = 6,
) -> list[Segmentation]:
    energy = _visible_user_energy(panel, cutoff)
    visible_features = build_user_behavior_features(
        panel,
        cutoff=cutoff,
        recent_days=recent_days,
        minimum_observed_days=14,
    )
    assigned_behavior, _ = cluster_user_behavior(
        visible_features,
        clusters=clusters,
    )
    behavior_visible = dict(
        zip(
            assigned_behavior["user_id"],
            assigned_behavior["behavior_type"],
            strict=True,
        )
    )

    behavior_raw = {
        user_id: behavior_visible.get(user_id, "B_UNSEEN_OR_SHORT_HISTORY")
        for user_id in panel.curves
    }
    behavior_assignment = _collapse_small_groups(
        behavior_raw,
        energy,
        minimum_share=0.01,
        other_label="B_OTHER_RARE_OR_UNSEEN",
    )

    industry_raw = {
        user_id: (
            panel.gbt_industry[user_id].value
            if user_id in energy
            else "I_UNSEEN_AT_FREEZE"
        )
        for user_id in panel.curves
    }
    industry_assignment = _collapse_small_groups(
        industry_raw,
        energy,
        minimum_share=0.02,
        other_label="I_OTHER",
    )

    combination_raw = {
        user_id: (
            f"{industry_assignment[user_id]} | {behavior_assignment[user_id]}"
            if user_id in energy
            else "IB_UNSEEN_AT_FREEZE"
        )
        for user_id in panel.curves
    }
    combination_assignment = _collapse_small_groups(
        combination_raw,
        energy,
        minimum_share=0.02,
        other_label="IB_OTHER",
    )
    definitions = (
        ("behavior", behavior_assignment),
        ("gbt_industry", industry_assignment),
        ("industry_behavior", combination_assignment),
    )
    return [
        Segmentation(
            name=name,
            assignment=assignment,
            profile=_profile_assignment(name, assignment, energy),
        )
        for name, assignment in definitions
    ]


def _segment_datasets(
    panel: SegmentedUserPanel,
    aggregate: AggregatedLoadDataset,
    segmentation: Segmentation,
) -> dict[str, AggregatedLoadDataset]:
    segment_curves = {
        segment: {
            trade_date: np.zeros(96, dtype=float)
            for trade_date in aggregate.dates
        }
        for segment in set(segmentation.assignment.values())
    }
    for user_id, curves in panel.curves.items():
        segment = segmentation.assignment[user_id]
        for trade_date, curve in curves.items():
            segment_curves[segment][trade_date] += curve
    datasets = {}
    for segment, curves in segment_curves.items():
        counts = {trade_date: 1 for trade_date in curves}
        datasets[segment] = AggregatedLoadDataset(
            curves=curves,
            daily_account_count=counts,
            daily_user_count=counts,
            row_count=len(curves),
            distinct_account_count=1,
            distinct_user_count=1,
            distinct_retailer_count=1,
            negative_value_count=0,
            value_units=aggregate.value_units,
            source_sha256=aggregate.source_sha256,
            source_path=aggregate.source_path,
        )
    for trade_date in aggregate.dates:
        reconstructed = sum(
            (dataset.curves[trade_date] for dataset in datasets.values()),
            start=np.zeros(96),
        )
        if not np.allclose(reconstructed, aggregate.curves[trade_date]):
            raise AssertionError(
                f"Segment curves do not reconstruct aggregate on {trade_date}"
            )
    return datasets


def build_market_lag_features(
    market: pd.DataFrame,
    dates: list[date],
) -> pd.DataFrame:
    rows = []
    for trade_date in dates:
        for interval in range(1, 97):
            row = {
                "trade_date": trade_date,
                "trade_interval": interval,
            }
            for lag in (3, 7, 14):
                source_date = trade_date - timedelta(days=lag)
                key = (source_date, interval)
                if key not in market.index:
                    continue
                values = market.loc[key]
                system = float(values["SYSTEM_LOAD_POWER"])
                solar = float(values["NEW_ENERGY_PHOTOVOLTAIC_POWER"])
                wind = float(values["NEW_ENERGY_WIND_POWER"])
                row[f"market_system_load_lag{lag}"] = system
                row[f"market_net_load_lag{lag}"] = system - solar - wind
                row[f"market_solar_lag{lag}"] = solar
                row[f"market_wind_lag{lag}"] = wind
                row[f"market_import_lag{lag}"] = float(
                    values["CALL_WIRE_POWER_huadong"]
                )
                row[f"market_rt_price_lag{lag}"] = float(
                    values["AREA_CLEARING_PRICE"]
                )
            visible_system = []
            visible_net = []
            for lag in range(3, 10):
                key = (trade_date - timedelta(days=lag), interval)
                if key not in market.index:
                    continue
                values = market.loc[key]
                system = float(values["SYSTEM_LOAD_POWER"])
                visible_system.append(system)
                visible_net.append(
                    system
                    - float(values["NEW_ENERGY_PHOTOVOLTAIC_POWER"])
                    - float(values["NEW_ENERGY_WIND_POWER"])
                )
            row["market_system_load_visible7_mean"] = (
                float(np.mean(visible_system)) if visible_system else np.nan
            )
            row["market_net_load_visible7_mean"] = (
                float(np.mean(visible_net)) if visible_net else np.nan
            )
            rows.append(row)
    return pd.DataFrame(rows)


def add_calendar_anchor(
    features: pd.DataFrame,
    dataset: AggregatedLoadDataset,
) -> pd.DataFrame:
    anchor_rows = []
    for trade_date in dataset.dates:
        try:
            source_dates = select_similar_dates(trade_date, dataset.dates)
        except ValueError:
            anchor = np.full(96, np.nan)
        else:
            anchor = np.mean(
                np.vstack([dataset.curves[item] for item in source_dates]),
                axis=0,
            )
        for interval, value in enumerate(anchor, start=1):
            anchor_rows.append(
                {
                    "trade_date": trade_date,
                    "trade_interval": interval,
                    CALENDAR_ANCHOR: float(value),
                }
            )
    return features.merge(
        pd.DataFrame(anchor_rows),
        on=["trade_date", "trade_interval"],
        how="left",
        validate="one_to_one",
    )


def recent_trend_prediction(
    dataset: AggregatedLoadDataset,
    *,
    target_date: date,
    shrinkage: float = 0.5,
) -> tuple[np.ndarray, dict]:
    source_dates = select_similar_dates(target_date, dataset.dates)
    base = np.mean(
        np.vstack([dataset.curves[item] for item in source_dates]),
        axis=0,
    )
    cutoff = target_date - timedelta(days=3)
    ratios = []
    for history_date in [
        cutoff - timedelta(days=offset) for offset in range(14)
    ]:
        if history_date not in dataset.curves:
            continue
        try:
            history_sources = select_similar_dates(history_date, dataset.dates)
        except ValueError:
            continue
        reference_total = float(
            np.mean(
                np.vstack([dataset.curves[item] for item in history_sources]),
                axis=0,
            ).sum()
        )
        if abs(reference_total) < 1e-9:
            continue
        ratios.append(float(dataset.curves[history_date].sum()) / reference_total)
    raw_factor = float(np.median(ratios[-7:])) if ratios else 1.0
    factor = float(np.clip(1.0 + shrinkage * (raw_factor - 1.0), 0.85, 1.15))
    return np.clip(base * factor, 0.0, None), {
        "source_dates": "|".join(item.isoformat() for item in source_dates),
        "trend_history_days": len(ratios),
        "raw_trend_factor": raw_factor,
        "applied_trend_factor": factor,
    }


def _sum_segment_predictions(
    frames: list[pd.DataFrame],
    *,
    method: str,
) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    summed = (
        combined.groupby(["target_date", "trade_interval"], as_index=False)
        .agg(
            actual_load=("actual_load", "sum"),
            predicted_load=("predicted_load", "sum"),
        )
    )
    summed.insert(0, "model", method)
    summed["error"] = summed["predicted_load"] - summed["actual_load"]
    summed["absolute_error"] = summed["error"].abs()
    return summed


def _summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = (
        predictions.groupby(["model", "target_date"], as_index=False)
        .agg(
            absolute_error_sum=("absolute_error", "sum"),
            actual_abs_sum=("actual_load", lambda value: value.abs().sum()),
            actual_total=("actual_load", "sum"),
            predicted_total=("predicted_load", "sum"),
        )
    )
    daily["wmape_pct"] = daily["absolute_error_sum"] / daily["actual_abs_sum"] * 100
    daily["total_bias_pct"] = (
        (daily["predicted_total"] - daily["actual_total"])
        / daily["actual_total"]
        * 100
    )
    rows = []
    for model, group in daily.groupby("model"):
        points = predictions.loc[predictions["model"].eq(model)]
        rows.append(
            {
                "model": model,
                "target_days": len(group),
                "wmape_pct": (
                    points["absolute_error"].sum()
                    / points["actual_load"].abs().sum()
                    * 100
                ),
                "daily_wmape_mean_pct": group["wmape_pct"].mean(),
                "daily_wmape_p90_pct": group["wmape_pct"].quantile(0.9),
                "daily_wmape_max_pct": group["wmape_pct"].max(),
                "mean_abs_total_bias_pct": group["total_bias_pct"].abs().mean(),
                "worst_date": group.loc[group["wmape_pct"].idxmax(), "target_date"],
            }
        )
    return daily, pd.DataFrame(rows).sort_values("wmape_pct")


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    aggregate = read_aggregated_load_snapshot(args.load_input)
    panel = read_segmented_user_panel(args.load_input)
    freeze_cutoff = args.target_start - timedelta(days=3)
    segmentations = build_segmentations(panel, cutoff=freeze_cutoff)
    requested_segmentations = {
        item.strip()
        for item in args.segmentations.split(",")
        if item.strip()
    }
    if requested_segmentations:
        segmentations = [
            item for item in segmentations if item.name in requested_segmentations
        ]
    if not segmentations:
        raise ValueError(
            f"No requested segmentations found: {sorted(requested_segmentations)}"
        )
    market, market_coverage = read_visible_market_outcomes(
        args.market_input,
        start=aggregate.dates[0] - timedelta(days=30),
        cutoff=args.target_end - timedelta(days=3),
    )
    market_features = build_market_lag_features(market, aggregate.dates)
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())

    aggregate_baseline, _, _ = run_calendar_baseline(
        args.load_input,
        target_start=args.target_start,
        target_end=args.target_end,
    )
    all_predictions = [
        aggregate_baseline[
            [
                "model",
                "target_date",
                "trade_interval",
                "actual_load",
                "predicted_load",
                "error",
                "absolute_error",
            ]
        ].copy()
    ]
    segment_prediction_outputs = []
    profiles = []
    failures = []

    for segmentation in segmentations:
        print(f"[segmented] start segmentation={segmentation.name}", flush=True)
        profiles.append(segmentation.profile)
        datasets = _segment_datasets(panel, aggregate, segmentation)

        trend_frames = []
        for segment, dataset in datasets.items():
            rows = []
            target_date = args.target_start
            while target_date <= args.target_end:
                prediction, metadata = recent_trend_prediction(
                    dataset,
                    target_date=target_date,
                )
                actual = dataset.curves[target_date]
                for interval in range(1, 97):
                    rows.append(
                        {
                            "segmentation": segmentation.name,
                            "segment": segment,
                            "component_model": "recent_trend",
                            "target_date": target_date,
                            "trade_interval": interval,
                            "train_label_cutoff": target_date - timedelta(days=3),
                            **metadata,
                            "actual_load": float(actual[interval - 1]),
                            "predicted_load": float(prediction[interval - 1]),
                        }
                    )
                target_date += timedelta(days=1)
            trend_frames.append(pd.DataFrame(rows))
        trend_sum = _sum_segment_predictions(
            trend_frames,
            method=f"{segmentation.name}__recent_trend",
        )
        all_predictions.append(trend_sum)
        segment_prediction_outputs.extend(trend_frames)

        for segment, dataset in datasets.items():
            features = build_feature_frame(dataset, weather=None)
            features = add_calendar_anchor(features, dataset)
            features = features.merge(
                market_features,
                on=["trade_date", "trade_interval"],
                how="left",
                validate="one_to_one",
            )
            for model_name in models:
                predictions, status, model_failures = walk_forward_model(
                    features,
                    model_name=model_name,
                    target_start=args.target_start,
                    target_end=args.target_end,
                    include_weather=False,
                    target_mode="residual",
                    anchor_column=CALENDAR_ANCHOR,
                )
                failures.extend(
                    [
                        {
                            "segmentation": segmentation.name,
                            "segment": segment,
                            **failure,
                        }
                        for failure in model_failures
                    ]
                )
                if not status.available:
                    failures.append(
                        {
                            "segmentation": segmentation.name,
                            "segment": segment,
                            "model": model_name,
                            "error": status.reason,
                        }
                    )
                    continue
                predictions.insert(0, "segment", segment)
                predictions.insert(0, "segmentation", segmentation.name)
                segment_prediction_outputs.append(predictions)

        for model_name in models:
            frames = [
                frame
                for frame in segment_prediction_outputs
                if not frame.empty
                and "model" in frame
                and frame["segmentation"].iloc[0] == segmentation.name
                and frame["model"].iloc[0] == model_name
            ]
            if not frames:
                continue
            summed = _sum_segment_predictions(
                frames,
                method=f"{segmentation.name}__{model_name}__market_lags",
            )
            all_predictions.append(summed)
        print(f"[segmented] done segmentation={segmentation.name}", flush=True)

    predictions = pd.concat(all_predictions, ignore_index=True)
    daily, summary = _summarize(predictions)
    profile = pd.concat(profiles, ignore_index=True)
    segment_predictions = pd.concat(segment_prediction_outputs, ignore_index=True)

    predictions.to_csv(
        args.output_dir / "predictions_96.csv.gz",
        index=False,
        compression="gzip",
    )
    segment_predictions.to_csv(
        args.output_dir / "segment_predictions_96.csv.gz",
        index=False,
        compression="gzip",
    )
    daily.to_csv(
        args.output_dir / "daily_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        args.output_dir / "summary_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    profile.to_csv(
        args.output_dir / "segment_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )
    market_coverage.to_csv(
        args.output_dir / "market_field_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "version": VERSION,
        "target_start": args.target_start.isoformat(),
        "target_end": args.target_end.isoformat(),
        "segment_freeze_cutoff": freeze_cutoff.isoformat(),
        "segment_freeze_policy": (
            "28-day behavior and semantic labels visible by first target D-3; "
            "no target-period regrouping"
        ),
        "market_visibility": (
            "RT outcomes only; every feature source date is target D-3 or earlier"
        ),
        "weather_policy": "NONE",
        "champion_eligible": False,
        "eligibility_reason": (
            "RT market lags obey simulated D-3 window, but the export does not "
            "preserve historical publication/available_at evidence"
        ),
        "models": list(models),
        "segmentations": [item.name for item in segmentations],
        "prediction_methods": summary["model"].tolist(),
        "load_source_sha256": aggregate.source_sha256,
        "pii_output_check": "aggregate segment labels only",
        "failures": len(failures),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-input", type=Path, required=True)
    parser.add_argument("--market-input", type=Path, required=True)
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
    parser.add_argument("--models", default="ridge")
    parser.add_argument(
        "--segmentations",
        default="behavior,gbt_industry,industry_behavior",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
