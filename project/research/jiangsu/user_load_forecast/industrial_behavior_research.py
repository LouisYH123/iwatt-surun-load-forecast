"""Recent-window industrial semantics, load behavior, and market diagnostics.

The module combines two independent views of a retail user:

* a conservative GB/T 4754—2017 major-class proxy derived from the legal name;
* an operational archetype learned from recent visible 96-point load curves.

Public Jiangsu RT market outcomes are used only through D-3.  Same-day market
relationships are descriptive diagnostics; they are never treated as
target-day forecast features without a historical publication timestamp.
Generated files contain aggregate segments only, not user names, account IDs,
or anonymous user identifiers.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from project.research.jiangsu.user_load_forecast.calendar_features import day_type
from project.research.jiangsu.user_load_forecast.load_data import sha256_file
from project.research.jiangsu.user_load_forecast.model_backtest import (
    prepare_output_dir,
)
from project.research.jiangsu.user_load_forecast.user_segmentation import (
    AMBIGUOUS,
    GBT_INDUSTRY_VERSION,
    SEGMENTATION_VERSION,
    UNKNOWN,
    SegmentedUserPanel,
    read_segmented_user_panel,
)


VERSION = "surun_industrial_behavior_market_v1"
MARKET_FIELDS = (
    "SYSTEM_LOAD_POWER",
    "NEW_ENERGY_PHOTOVOLTAIC_POWER",
    "NEW_ENERGY_WIND_POWER",
    "CALL_WIRE_POWER_huadong",
    "AREA_CLEARING_PRICE",
)
CLUSTER_FEATURES = (
    "load_factor",
    "night_share",
    "daytime_share",
    "evening_share",
    "lunch_ratio",
    "weekend_ratio",
    "day_total_cv",
    "weekly_curve_wmape",
    "negative_point_rate",
    "zero_point_rate",
    "active_interval_rate",
    "recent_7_vs_previous_pct",
)


def _date_range(start: date, end: date) -> list[date]:
    return [
        start + timedelta(days=offset)
        for offset in range((end - start).days + 1)
    ]


def _ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) < 1e-9:
        return np.nan
    return float(numerator / denominator)


def _curve_characteristics(
    dated_curves: dict[date, np.ndarray],
    *,
    window_start: date,
    window_end: date,
) -> dict[str, float]:
    dates = [
        item
        for item in sorted(dated_curves)
        if window_start <= item <= window_end
    ]
    if not dates:
        return {}
    stack = np.vstack([dated_curves[item] for item in dates])
    mean_curve = stack.mean(axis=0)
    positive_curve = np.clip(mean_curve, 0.0, None)
    positive_total = float(positive_curve.sum())
    daily_total = stack.sum(axis=1)
    positive_peak = float(positive_curve.max())

    weekday_totals = [
        float(dated_curves[item].sum())
        for item in dates
        if day_type(item) == "workday"
    ]
    weekend_totals = [
        float(dated_curves[item].sum())
        for item in dates
        if day_type(item) in {"weekend", "holiday"}
    ]
    weekly_errors = []
    date_set = set(dates)
    for item in dates:
        previous = item - timedelta(days=7)
        if previous not in date_set:
            continue
        actual = dated_curves[item]
        reference = dated_curves[previous]
        denominator = float(np.abs(actual).sum())
        if denominator > 1e-9:
            weekly_errors.append(float(np.abs(actual - reference).sum() / denominator))

    last_7_start = window_end - timedelta(days=6)
    previous_dates = [item for item in dates if item < last_7_start]
    last_dates = [item for item in dates if item >= last_7_start]
    last_mean = (
        float(np.mean([dated_curves[item].sum() for item in last_dates]))
        if last_dates
        else np.nan
    )
    previous_mean = (
        float(np.mean([dated_curves[item].sum() for item in previous_dates]))
        if previous_dates
        else np.nan
    )
    lunch = float(positive_curve[46:54].mean())
    shoulder = float(
        np.mean(np.concatenate([positive_curve[36:44], positive_curve[56:64]]))
    )
    mean_positive = float(positive_curve.mean())
    return {
        "observed_days": len(dates),
        "average_daily_energy": float(np.mean(daily_total)),
        "positive_energy": float(np.clip(stack, 0.0, None).sum()),
        "load_factor": _ratio(mean_positive, positive_peak),
        "night_share": _ratio(float(positive_curve[:24].sum()), positive_total),
        "daytime_share": _ratio(float(positive_curve[32:72].sum()), positive_total),
        "evening_share": _ratio(float(positive_curve[72:].sum()), positive_total),
        "lunch_ratio": _ratio(lunch, shoulder),
        "weekend_ratio": _ratio(
            float(np.mean(weekend_totals)) if weekend_totals else np.nan,
            float(np.mean(weekday_totals)) if weekday_totals else np.nan,
        ),
        "day_total_cv": _ratio(float(np.std(daily_total)), abs(float(np.mean(daily_total)))),
        "weekly_curve_wmape": (
            float(np.median(weekly_errors)) if weekly_errors else np.nan
        ),
        "negative_point_rate": float(np.mean(stack < 0)),
        "zero_point_rate": float(np.mean(np.isclose(stack, 0.0))),
        "active_interval_rate": (
            float(np.mean(positive_curve >= 0.5 * positive_peak))
            if positive_peak > 0
            else 0.0
        ),
        "peak_interval": int(np.argmax(mean_curve) + 1),
        "trough_interval": int(np.argmin(mean_curve) + 1),
        "recent_7_vs_previous_pct": (
            (_ratio(last_mean, previous_mean) - 1.0) * 100
            if np.isfinite(last_mean) and np.isfinite(previous_mean)
            else np.nan
        ),
    }


def build_user_behavior_features(
    panel: SegmentedUserPanel,
    *,
    cutoff: date,
    recent_days: int,
    minimum_observed_days: int,
) -> pd.DataFrame:
    start = cutoff - timedelta(days=recent_days - 1)
    rows = []
    for user_id, curves in panel.curves.items():
        features = _curve_characteristics(
            curves,
            window_start=start,
            window_end=cutoff,
        )
        if features.get("observed_days", 0) < minimum_observed_days:
            continue
        if features.get("positive_energy", 0.0) <= 0:
            continue
        rows.append(
            {
                "user_id": user_id,
                "gbt_industry": panel.gbt_industry[user_id].value,
                "gbt_confidence": panel.gbt_industry[user_id].confidence,
                **features,
            }
        )
    return pd.DataFrame(rows)


def _behavior_name(row: pd.Series) -> str:
    if row["negative_point_rate"] >= 0.001:
        return "双向负荷/疑似光伏反送"
    if row["day_total_cv"] >= 0.45 or row["weekly_curve_wmape"] >= 0.40:
        return "间歇生产/高波动"
    if row["night_share"] >= 0.22 and row["load_factor"] >= 0.62:
        return "连续生产"
    if row["daytime_share"] >= 0.52 and row["night_share"] <= 0.12:
        return "白班生产"
    if row["night_share"] >= 0.17 and row["evening_share"] >= 0.22:
        return "多班或夜班生产"
    if row["lunch_ratio"] <= 0.78:
        return "午间停机型"
    return "扩展工时/混合型"


def cluster_user_behavior(
    features: pd.DataFrame,
    *,
    clusters: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(features) < clusters:
        raise ValueError(f"Only {len(features)} users for {clusters} clusters")
    usable = features.copy()
    matrix = usable[list(CLUSTER_FEATURES)].replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median())
    # Prevent a handful of near-zero or negative-load users from consuming
    # entire KMeans clusters.  They remain represented, while the main
    # operational clusters are learned from the central 98% feature range.
    for column in matrix:
        matrix[column] = matrix[column].clip(
            lower=matrix[column].quantile(0.01),
            upper=matrix[column].quantile(0.99),
        )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    weights = np.sqrt(
        np.maximum(
            usable["positive_energy"].to_numpy(dtype=float)
            / max(float(usable["positive_energy"].median()), 1e-9),
            0.05,
        )
    )
    weights = np.clip(weights, 0.2, 8.0)
    model = KMeans(
        n_clusters=clusters,
        n_init=30,
        random_state=42,
    )
    usable["cluster_id"] = model.fit_predict(scaled, sample_weight=weights)

    centroid_rows = []
    for cluster_id, group in usable.groupby("cluster_id"):
        centroid = group[list(CLUSTER_FEATURES)].median()
        behavior = _behavior_name(centroid)
        centroid_rows.append(
            {
                "cluster_id": int(cluster_id),
                "behavior_type": f"B{int(cluster_id)} {behavior}",
                "user_count": len(group),
                "recent_positive_energy": group["positive_energy"].sum(),
                **{column: float(centroid[column]) for column in CLUSTER_FEATURES},
                "average_daily_energy_median": group["average_daily_energy"].median(),
                "peak_interval_median": group["peak_interval"].median(),
            }
        )
    centroids = pd.DataFrame(centroid_rows)
    behavior_map = centroids.set_index("cluster_id")["behavior_type"].to_dict()
    usable["behavior_type"] = usable["cluster_id"].map(behavior_map)
    return usable, centroids


def cluster_selection_diagnostics(
    features: pd.DataFrame,
    *,
    minimum_clusters: int = 3,
    maximum_clusters: int = 8,
) -> pd.DataFrame:
    matrix = features[list(CLUSTER_FEATURES)].replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median())
    for column in matrix:
        matrix[column] = matrix[column].clip(
            lower=matrix[column].quantile(0.01),
            upper=matrix[column].quantile(0.99),
        )
    scaled = StandardScaler().fit_transform(matrix)
    weights = np.sqrt(
        np.maximum(
            features["positive_energy"].to_numpy(dtype=float)
            / max(float(features["positive_energy"].median()), 1e-9),
            0.05,
        )
    )
    weights = np.clip(weights, 0.2, 8.0)
    total_energy = features["positive_energy"].sum()
    rows = []
    for clusters in range(minimum_clusters, maximum_clusters + 1):
        labels = KMeans(
            n_clusters=clusters,
            n_init=30,
            random_state=42,
        ).fit_predict(scaled, sample_weight=weights)
        counts = np.bincount(labels, minlength=clusters)
        shares = [
            features.loc[labels == cluster, "positive_energy"].sum()
            / total_energy
            for cluster in range(clusters)
        ]
        rows.append(
            {
                "clusters": clusters,
                "silhouette_score": float(silhouette_score(scaled, labels)),
                "minimum_cluster_users": int(min(counts)),
                "minimum_cluster_load_share": float(min(shares)),
                "maximum_cluster_load_share": float(max(shares)),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_curves(
    panel: SegmentedUserPanel,
    users: set[str],
) -> dict[date, np.ndarray]:
    curves: dict[date, np.ndarray] = defaultdict(lambda: np.zeros(96))
    for user_id in users:
        for trade_date, curve in panel.curves[user_id].items():
            curves[trade_date] += curve
    return dict(curves)


def build_segment_profiles(
    panel: SegmentedUserPanel,
    assigned: pd.DataFrame,
    *,
    cutoff: date,
    recent_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    recent_start = cutoff - timedelta(days=recent_days - 1)
    prior_end = recent_start - timedelta(days=1)
    prior_start = prior_end - timedelta(days=recent_days - 1)
    total_recent = assigned["positive_energy"].sum()

    industry_rows = []
    for industry, group in assigned.groupby("gbt_industry"):
        users = set(group["user_id"])
        curves = _aggregate_curves(panel, users)
        recent = _curve_characteristics(
            curves,
            window_start=recent_start,
            window_end=cutoff,
        )
        prior = _curve_characteristics(
            curves,
            window_start=prior_start,
            window_end=prior_end,
        )
        industry_rows.append(
            {
                "gbt_industry": industry,
                "user_count": len(users),
                "recent_load_share": (
                    group["positive_energy"].sum() / total_recent
                    if total_recent
                    else 0.0
                ),
                **{f"recent_{key}": value for key, value in recent.items()},
                "recent_vs_prior_energy_pct": (
                    (
                        _ratio(
                            recent.get("average_daily_energy", np.nan),
                            prior.get("average_daily_energy", np.nan),
                        )
                        - 1.0
                    )
                    * 100
                    if prior
                    else np.nan
                ),
            }
        )
    industry_profile = pd.DataFrame(industry_rows).sort_values(
        "recent_load_share",
        ascending=False,
    )

    crosswalk = (
        assigned.groupby(["gbt_industry", "behavior_type"], as_index=False)
        .agg(
            user_count=("user_id", "size"),
            recent_positive_energy=("positive_energy", "sum"),
        )
    )
    crosswalk["recent_load_share"] = (
        crosswalk["recent_positive_energy"] / total_recent
        if total_recent
        else 0.0
    )
    return industry_profile, crosswalk.sort_values(
        "recent_load_share",
        ascending=False,
    )


def build_industry_cohort_change(
    panel: SegmentedUserPanel,
    *,
    cutoff: date,
    recent_days: int,
    minimum_active_days: int,
) -> pd.DataFrame:
    recent_start = cutoff - timedelta(days=recent_days - 1)
    prior_end = recent_start - timedelta(days=1)
    prior_start = prior_end - timedelta(days=recent_days - 1)

    def window_stats(
        curves: dict[date, np.ndarray],
        start: date,
        end: date,
    ) -> tuple[int, float]:
        selected = [
            curve
            for trade_date, curve in curves.items()
            if start <= trade_date <= end
        ]
        return len(selected), float(sum(curve.sum() for curve in selected))

    rows = []
    labels = sorted(set(label.value for label in panel.gbt_industry.values()))
    for industry in labels:
        users = [
            user_id
            for user_id, label in panel.gbt_industry.items()
            if label.value == industry
        ]
        stats = {}
        for user_id in users:
            recent_count, recent_energy = window_stats(
                panel.curves[user_id],
                recent_start,
                cutoff,
            )
            prior_count, prior_energy = window_stats(
                panel.curves[user_id],
                prior_start,
                prior_end,
            )
            stats[user_id] = {
                "recent_count": recent_count,
                "recent_energy": recent_energy,
                "prior_count": prior_count,
                "prior_energy": prior_energy,
            }
        stable = {
            user_id
            for user_id, value in stats.items()
            if value["recent_count"] >= minimum_active_days
            and value["prior_count"] >= minimum_active_days
        }
        recent_only = {
            user_id
            for user_id, value in stats.items()
            if value["recent_count"] >= minimum_active_days
            and value["prior_count"] < minimum_active_days
        }
        prior_only = {
            user_id
            for user_id, value in stats.items()
            if value["prior_count"] >= minimum_active_days
            and value["recent_count"] < minimum_active_days
        }
        recent_energy = sum(value["recent_energy"] for value in stats.values())
        prior_energy = sum(value["prior_energy"] for value in stats.values())
        stable_recent = sum(stats[user_id]["recent_energy"] for user_id in stable)
        stable_prior = sum(stats[user_id]["prior_energy"] for user_id in stable)
        recent_only_energy = sum(
            stats[user_id]["recent_energy"] for user_id in recent_only
        )
        prior_only_energy = sum(
            stats[user_id]["prior_energy"] for user_id in prior_only
        )
        rows.append(
            {
                "gbt_industry": industry,
                "all_user_count": len(users),
                "stable_user_count": len(stable),
                "recent_only_user_count": len(recent_only),
                "prior_only_user_count": len(prior_only),
                "all_cohort_change_pct": (
                    (_ratio(recent_energy, prior_energy) - 1.0) * 100
                    if prior_energy
                    else np.nan
                ),
                "stable_cohort_change_pct": (
                    (_ratio(stable_recent, stable_prior) - 1.0) * 100
                    if stable_prior
                    else np.nan
                ),
                "recent_only_energy_share": _ratio(
                    recent_only_energy,
                    recent_energy,
                ),
                "prior_only_energy_share": _ratio(
                    prior_only_energy,
                    prior_energy,
                ),
                "recent_energy": recent_energy,
                "prior_energy": prior_energy,
                "recent_window_start": recent_start,
                "recent_window_end": cutoff,
                "prior_window_start": prior_start,
                "prior_window_end": prior_end,
            }
        )
    return pd.DataFrame(rows).sort_values("recent_energy", ascending=False)


def build_anomaly_summary(assigned: pd.DataFrame) -> pd.DataFrame:
    total_energy = assigned["positive_energy"].sum()
    definitions = (
        ("any_negative_point", assigned["negative_point_rate"] > 0),
        ("negative_point_rate_ge_0.1pct", assigned["negative_point_rate"] >= 0.001),
        ("zero_point_rate_ge_25pct", assigned["zero_point_rate"] >= 0.25),
        ("day_total_cv_ge_45pct", assigned["day_total_cv"] >= 0.45),
        ("weekly_curve_wmape_ge_40pct", assigned["weekly_curve_wmape"] >= 0.40),
    )
    return pd.DataFrame(
        [
            {
                "anomaly": name,
                "user_count": int(mask.sum()),
                "recent_positive_energy": assigned.loc[
                    mask,
                    "positive_energy",
                ].sum(),
                "recent_load_share": (
                    assigned.loc[mask, "positive_energy"].sum() / total_energy
                    if total_energy
                    else 0.0
                ),
            }
            for name, mask in definitions
        ]
    )


def read_visible_market_outcomes(
    path: Path,
    *,
    start: date,
    cutoff: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [
        "trade_date",
        "trade_interval",
        "data_name",
        "value",
        "data_type",
    ]
    frames = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=200_000):
        chunk = chunk.loc[
            chunk["data_type"].eq("RT")
            & chunk["data_name"].isin(MARKET_FIELDS)
        ].copy()
        if chunk.empty:
            continue
        chunk["trade_date"] = pd.to_datetime(chunk["trade_date"]).dt.date
        chunk = chunk.loc[
            (chunk["trade_date"] >= start)
            & (chunk["trade_date"] <= cutoff)
        ]
        if not chunk.empty:
            frames.append(chunk)
    if not frames:
        raise ValueError("No visible RT market outcomes found")
    long = pd.concat(frames, ignore_index=True)
    duplicate = long.duplicated(["trade_date", "trade_interval", "data_name"])
    if duplicate.any():
        raise ValueError(f"Duplicate market outcomes: {int(duplicate.sum())}")
    matrix = long.pivot(
        index=["trade_date", "trade_interval"],
        columns="data_name",
        values="value",
    ).sort_index()
    coverage_rows = []
    expected_days = len(_date_range(start, cutoff))
    for field in MARKET_FIELDS:
        series = matrix[field] if field in matrix else pd.Series(dtype=float)
        counts = (
            series.groupby(level="trade_date").count()
            if not series.empty
            else pd.Series(dtype=int)
        )
        coverage_rows.append(
            {
                "data_type": "RT",
                "data_name": field,
                "window_start": start,
                "window_end": cutoff,
                "expected_days": expected_days,
                "covered_days": int((counts == 96).sum()),
                "minimum_points_per_day": int(counts.min()) if len(counts) else 0,
                "maximum_points_per_day": int(counts.max()) if len(counts) else 0,
                "visibility": "HISTORICAL_OUTCOME_VISIBLE_THROUGH_D3",
            }
        )
    return matrix, pd.DataFrame(coverage_rows)


def _same_weekday_residual(series: pd.Series) -> pd.Series:
    values = series.sort_index()
    residual = {}
    for trade_date, actual in values.items():
        references = [
            trade_date - timedelta(days=lag)
            for lag in (7, 14, 21, 28)
            if trade_date - timedelta(days=lag) in values.index
        ]
        if len(references) < 3:
            residual[trade_date] = np.nan
            continue
        baseline = float(values.loc[references].mean())
        residual[trade_date] = (
            (float(actual) - baseline) / abs(baseline)
            if abs(baseline) > 1e-9
            else np.nan
        )
    return pd.Series(residual)


def _market_daily_features(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trade_date, group in matrix.groupby(level="trade_date"):
        group = group.droplevel("trade_date")
        system = group["SYSTEM_LOAD_POWER"]
        solar = group["NEW_ENERGY_PHOTOVOLTAIC_POWER"]
        wind = group["NEW_ENERGY_WIND_POWER"]
        rows.append(
            {
                "trade_date": trade_date,
                "system_load_mean": system.mean(),
                "system_load_peak": system.max(),
                "net_load_mean": (system - solar - wind).mean(),
                "solar_mean": solar.mean(),
                "wind_mean": wind.mean(),
                "huadong_import_mean": group["CALL_WIRE_POWER_huadong"].mean(),
                "rt_price_mean": group["AREA_CLEARING_PRICE"].mean(),
                "rt_price_std": group["AREA_CLEARING_PRICE"].std(),
            }
        )
    return pd.DataFrame(rows).set_index("trade_date").sort_index()


def build_market_relationships(
    panel: SegmentedUserPanel,
    assigned: pd.DataFrame,
    market: pd.DataFrame,
    *,
    minimum_load_share: float,
) -> pd.DataFrame:
    market_daily = _market_daily_features(market)
    market_residuals = {
        column: _same_weekday_residual(market_daily[column])
        for column in market_daily.columns
    }
    total_energy = assigned["positive_energy"].sum()
    rows = []
    for industry, group in assigned.groupby("gbt_industry"):
        share = group["positive_energy"].sum() / total_energy
        if share < minimum_load_share:
            continue
        curves = _aggregate_curves(panel, set(group["user_id"]))
        daily_energy = pd.Series(
            {
                trade_date: float(curve.sum())
                for trade_date, curve in curves.items()
                if trade_date in market_daily.index
            }
        ).sort_index()
        segment_residual = _same_weekday_residual(daily_energy)
        common = segment_residual.dropna().index
        record = {
            "gbt_industry": industry,
            "recent_load_share": share,
            "relationship_days": len(common),
        }
        for column, residual in market_residuals.items():
            joined = pd.concat(
                [segment_residual.rename("segment"), residual.rename("market")],
                axis=1,
            ).dropna()
            record[f"corr_{column}_residual"] = (
                float(joined["segment"].corr(joined["market"]))
                if len(joined) >= 14
                else np.nan
            )
            for lag in (3, 7):
                lagged = pd.Series(
                    {
                        target_date: residual.get(
                            target_date - timedelta(days=lag),
                            np.nan,
                        )
                        for target_date in segment_residual.index
                    }
                )
                lagged_joined = pd.concat(
                    [
                        segment_residual.rename("segment"),
                        lagged.rename("market_lag"),
                    ],
                    axis=1,
                ).dropna()
                record[f"corr_{column}_residual_lag{lag}"] = (
                    float(
                        lagged_joined["segment"].corr(
                            lagged_joined["market_lag"]
                        )
                    )
                    if len(lagged_joined) >= 14
                    else np.nan
                )

        shape_correlations = []
        for trade_date in sorted(set(curves) & set(market_daily.index)):
            system_curve = market.xs(trade_date, level="trade_date")[
                "SYSTEM_LOAD_POWER"
            ].sort_index()
            if len(system_curve) != 96:
                continue
            segment_curve = curves[trade_date]
            if np.std(segment_curve) < 1e-9 or system_curve.std() < 1e-9:
                continue
            shape_correlations.append(
                float(np.corrcoef(segment_curve, system_curve.to_numpy())[0, 1])
            )
        record["median_intraday_corr_system_load"] = (
            float(np.median(shape_correlations))
            if shape_correlations
            else np.nan
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values("recent_load_share", ascending=False)


def _markdown_table(frame: pd.DataFrame) -> str:
    def display(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|")

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(display(value) for value in row) + " |")
    return "\n".join(lines)


def write_summary(
    path: Path,
    *,
    analysis_date: date,
    cutoff: date,
    recent_start: date,
    user_count: int,
    industry_profile: pd.DataFrame,
    cohort_change: pd.DataFrame,
    centroids: pd.DataFrame,
    relationships: pd.DataFrame,
) -> None:
    known_share = industry_profile.loc[
        ~industry_profile["gbt_industry"].isin([UNKNOWN, AMBIGUOUS]),
        "recent_load_share",
    ].sum()
    top_industries = industry_profile.merge(
        cohort_change[
            [
                "gbt_industry",
                "stable_cohort_change_pct",
                "recent_only_energy_share",
            ]
        ],
        on="gbt_industry",
        how="left",
    )[
        [
            "gbt_industry",
            "user_count",
            "recent_load_share",
            "recent_vs_prior_energy_pct",
            "stable_cohort_change_pct",
            "recent_only_energy_share",
            "recent_load_factor",
            "recent_night_share",
            "recent_weekend_ratio",
            "recent_weekly_curve_wmape",
        ]
    ].head(15)
    behavior = centroids[
        [
            "behavior_type",
            "user_count",
            "recent_load_share",
            "load_factor",
            "night_share",
            "daytime_share",
            "day_total_cv",
            "weekly_curve_wmape",
        ]
    ].sort_values("recent_load_share", ascending=False)
    relationship_display = relationships[
        [
            "gbt_industry",
            "recent_load_share",
            "corr_system_load_mean_residual",
            "corr_system_load_mean_residual_lag3",
            "corr_net_load_mean_residual",
            "corr_net_load_mean_residual_lag3",
            "corr_rt_price_mean_residual",
            "corr_rt_price_mean_residual_lag3",
            "median_intraday_corr_system_load",
        ]
    ].head(12)
    lines = [
        "# 江苏零售用户工业分类与最近用电行为研究",
        "",
        "## 可读窗口",
        "",
        f"- 分析目标日：{analysis_date.isoformat()}。",
        f"- 历史实际负荷和RT市场结果最晚读取：{cutoff.isoformat()}（D-3）。",
        f"- 最近行为窗口：{recent_start.isoformat()}至{cutoff.isoformat()}。",
        "- DA、目标日市场计划和未版本化天气不进入本研究的可晋级输入。",
        "",
        "## 分类覆盖",
        "",
        f"- 进入行为聚类的用户数：{user_count}。",
        f"- GB/T两位行业名称代理覆盖最近正负荷：{known_share:.2%}。",
        "- 名称行业是研究代理；正式行业代码应来自用户注册或合同主数据。",
        "",
        "## 主要行业最近特征",
        "",
        _markdown_table(top_industries),
        "",
        "## 用电行为聚类",
        "",
        _markdown_table(behavior),
        "",
        "## 与江苏公开RT市场环境的关系",
        "",
        "以下相关性使用同星期历史残差，只用于结构诊断，不代表因果关系。",
        "",
        _markdown_table(relationship_display),
        "",
        "## 建模建议",
        "",
        "1. 预测主分层采用“GB/T行业 × 用电行为类型”，不要只依赖企业名称。",
        "2. 每个目标日重新用D-3截止的28/56天窗口计算行为特征，但聚类中心按版本冻结。",
        "3. 系统负荷、新能源、联络线和实时价格结果只能以D-3及更早滞后进入模型。",
        "4. 目标日DA/计划量只有在保存issue_at、available_at后才能作为日前特征。",
        "5. 连续生产和多班用户优先使用稳定曲线模型；白班和高波动用户加强日历、天气和用户事件修正。",
        "",
        "## 公开依据",
        "",
        "- 国家统计局《国民经济行业分类》（GB/T 4754—2017）："
        "https://www.stats.gov.cn/xxgk/tjbz/gjtjbz/201710/t20171017_1758922.html",
        "- 国家统计局第1号修改单："
        "https://www.stats.gov.cn/xxgk/tjbz/gjtjbz/201905/t20190521_1758938.html",
        "- 江苏能源监管办2025年信息披露通知："
        "https://jsb.nea.gov.cn/dtyw/tz/202505/t20250523_281496.html",
        "- 江苏能源监管办2026年持续信息披露通知："
        "https://jsb.nea.gov.cn/dtyw/tz/202606/t20260603_302444.html",
        "- 《江苏省电力中长期市场实施细则（2026版）》："
        "https://jsb.nea.gov.cn/dtyw/tz/202603/t20260306_297521.html",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-input", type=Path, required=True)
    parser.add_argument("--market-input", type=Path, required=True)
    parser.add_argument(
        "--analysis-date",
        type=date.fromisoformat,
        default=date(2026, 7, 25),
    )
    parser.add_argument("--label-lag-days", type=int, default=3)
    parser.add_argument("--recent-days", type=int, default=28)
    parser.add_argument("--market-lookback-days", type=int, default=112)
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--minimum-observed-days", type=int, default=14)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir)
    cutoff = args.analysis_date - timedelta(days=args.label_lag_days)
    recent_start = cutoff - timedelta(days=args.recent_days - 1)
    market_start = cutoff - timedelta(days=args.market_lookback_days - 1)
    panel = read_segmented_user_panel(args.load_input)
    user_features = build_user_behavior_features(
        panel,
        cutoff=cutoff,
        recent_days=args.recent_days,
        minimum_observed_days=args.minimum_observed_days,
    )
    assigned, centroids = cluster_user_behavior(
        user_features,
        clusters=args.clusters,
    )
    cluster_selection = cluster_selection_diagnostics(user_features)
    total_energy = assigned["positive_energy"].sum()
    centroids["recent_load_share"] = (
        centroids["recent_positive_energy"] / total_energy
    )
    industry_profile, crosswalk = build_segment_profiles(
        panel,
        assigned,
        cutoff=cutoff,
        recent_days=args.recent_days,
    )
    cohort_change = build_industry_cohort_change(
        panel,
        cutoff=cutoff,
        recent_days=args.recent_days,
        minimum_active_days=args.minimum_observed_days,
    )
    anomaly_summary = build_anomaly_summary(assigned)
    market, market_coverage = read_visible_market_outcomes(
        args.market_input,
        start=market_start,
        cutoff=cutoff,
    )
    relationships = build_market_relationships(
        panel,
        assigned,
        market,
        minimum_load_share=0.005,
    )
    visibility_ledger = pd.DataFrame(
        [
            {
                "source": "user_actual_consumption_raw",
                "data_class": "user_actual_outcome",
                "read_window_start": recent_start,
                "read_window_end": cutoff,
                "use_policy": "RECENT_BEHAVIOR_AND_LAGGED_FEATURES",
                "champion_eligible": True,
                "reason": "simulated T+2; target D reads labels through D-3",
            },
            {
                "source": "spot_prices_raw/RT",
                "data_class": "public_market_outcome",
                "read_window_start": market_start,
                "read_window_end": cutoff,
                "use_policy": "HISTORICAL_LAGS_ONLY",
                "champion_eligible": True,
                "reason": "target D reads market outcomes through D-3",
            },
            {
                "source": "spot_prices_raw/DA",
                "data_class": "public_market_forecast_or_clearing",
                "read_window_start": None,
                "read_window_end": None,
                "use_policy": "DIAGNOSTIC_ONLY",
                "champion_eligible": False,
                "reason": "snapshot lacks historical issue_at/available_at",
            },
            {
                "source": "weather_raw",
                "data_class": "weather_proxy",
                "read_window_start": None,
                "read_window_end": None,
                "use_policy": "DIAGNOSTIC_ONLY",
                "champion_eligible": False,
                "reason": "snapshot lacks historical forecast issue time",
            },
        ]
    )

    # Aggregate outputs only; the per-user assignment frame remains in memory.
    centroids.to_csv(
        args.output_dir / "behavior_cluster_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    industry_profile.to_csv(
        args.output_dir / "gbt_industry_recent_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )
    crosswalk.to_csv(
        args.output_dir / "industry_behavior_crosswalk.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cohort_change.to_csv(
        args.output_dir / "industry_cohort_change.csv",
        index=False,
        encoding="utf-8-sig",
    )
    anomaly_summary.to_csv(
        args.output_dir / "recent_anomaly_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    relationships.to_csv(
        args.output_dir / "industry_market_relationship.csv",
        index=False,
        encoding="utf-8-sig",
    )
    market_coverage.to_csv(
        args.output_dir / "market_field_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    visibility_ledger.to_csv(
        args.output_dir / "visibility_ledger.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cluster_selection.to_csv(
        args.output_dir / "cluster_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "version": VERSION,
        "segmentation_version": SEGMENTATION_VERSION,
        "gbt_industry_version": GBT_INDUSTRY_VERSION,
        "province_display": "江苏",
        "province_slug": "jiangsu",
        "raw_schema": "jiangsu",
        "analysis_date": args.analysis_date.isoformat(),
        "latest_readable_date": cutoff.isoformat(),
        "label_lag_days": args.label_lag_days,
        "recent_window_start": recent_start.isoformat(),
        "recent_window_end": cutoff.isoformat(),
        "market_window_start": market_start.isoformat(),
        "market_window_end": cutoff.isoformat(),
        "cluster_count": args.clusters,
        "cluster_features": list(CLUSTER_FEATURES),
        "users_clustered": len(assigned),
        "load_source_sha256": sha256_file(args.load_input),
        "market_source_sha256": sha256_file(args.market_input),
        "pii_output_check": "PASS: aggregate-only outputs",
        "market_visibility": (
            "RT outcomes through D-3 only; DA and target-day fields forbidden"
        ),
        "official_taxonomy_reference": (
            "GB/T 4754-2017 with Amendment No.1 (2019)"
        ),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary(
        args.output_dir / "research_summary.md",
        analysis_date=args.analysis_date,
        cutoff=cutoff,
        recent_start=recent_start,
        user_count=len(assigned),
        industry_profile=industry_profile,
        cohort_change=cohort_change,
        centroids=centroids,
        relationships=relationships,
    )
    print((args.output_dir / "research_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
