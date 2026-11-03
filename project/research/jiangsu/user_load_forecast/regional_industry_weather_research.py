"""Research city-weighted weather and name-derived industry heterogeneity.

This is a diagnostic proxy experiment, not a production forecast:

* city weights use positive load visible through target_start - 3 days;
* target-day weather comes from the current unversioned weather snapshot;
* user city and industry labels are inferred from legal-name keywords;
* generated outputs are aggregate-only and contain no user identifiers.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.features import (
    build_feature_frame,
)
from project.research.jiangsu.user_load_forecast.load_data import (
    read_aggregated_load_snapshot,
)
from project.research.jiangsu.user_load_forecast.model_backtest import (
    prepare_output_dir,
    run_model_pool,
    summarize_predictions,
)
from project.research.jiangsu.user_load_forecast.regional_weather import (
    REGIONAL_WEATHER_VERSION,
    CityWeatherPanel,
    read_city_weather_panel,
    weighted_weather_store,
)
from project.research.jiangsu.user_load_forecast.user_segmentation import (
    AMBIGUOUS,
    JIANGSU_CITIES,
    SEGMENTATION_VERSION,
    UNKNOWN,
    SegmentedUserPanel,
    city_load_weights,
    read_segmented_user_panel,
    segment_profile,
)


RESEARCH_VERSION = "surun_regional_industry_weather_v1"


def _uniform_weights() -> dict[str, float]:
    return {city: 1.0 / len(JIANGSU_CITIES) for city in JIANGSU_CITIES}


def _city_daily_weather(panel: CityWeatherPanel) -> pd.DataFrame:
    frame = panel.frame.copy()
    frame["trade_date"] = frame["timestamp"].dt.date
    return (
        frame.groupby(["city", "trade_date"], as_index=False)
        .agg(
            temp_mean=("temp_2m", "mean"),
            humidity_mean=("humidity_2m", "mean"),
            rain_total=("rain", "sum"),
            radiation_mean=("radiation", "mean"),
            wind_mean=("wind_10m", "mean"),
        )
    )


def _weighted_daily_weather(
    city_daily: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    frame = city_daily.copy()
    frame["_weight"] = frame["city"].map(weights).astype(float)
    for column in (
        "temp_mean",
        "humidity_mean",
        "rain_total",
        "radiation_mean",
        "wind_mean",
    ):
        frame[column] = frame[column] * frame["_weight"]
    daily = (
        frame.groupby("trade_date", as_index=False)[
            [
                "temp_mean",
                "humidity_mean",
                "rain_total",
                "radiation_mean",
                "wind_mean",
            ]
        ]
        .sum()
    )
    daily["cooling_degree"] = (daily["temp_mean"] - 26.0).clip(lower=0.0)
    return daily


def _group_daily_energy(
    panel: SegmentedUserPanel,
    users: set[str],
) -> dict[date, float]:
    values: dict[date, float] = defaultdict(float)
    for user_id in users:
        for trade_date, curve in panel.curves[user_id].items():
            values[trade_date] += float(curve.sum())
    return dict(values)


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    usable = left.notna() & right.notna()
    if usable.sum() < 8:
        return np.nan
    return float(left.loc[usable].corr(right.loc[usable]))


def _single_slope(x: pd.Series, y: pd.Series) -> float:
    usable = x.notna() & y.notna()
    if usable.sum() < 8 or float(x.loc[usable].std()) < 1e-9:
        return np.nan
    matrix = np.column_stack(
        [np.ones(int(usable.sum())), x.loc[usable].to_numpy(dtype=float)]
    )
    coefficient = np.linalg.lstsq(
        matrix,
        y.loc[usable].to_numpy(dtype=float),
        rcond=None,
    )[0]
    return float(coefficient[1])


def build_sensitivity_diagnostics(
    panel: SegmentedUserPanel,
    weather: CityWeatherPanel,
    *,
    history_cutoff: date,
    target_start: date,
    target_end: date,
    minimum_load_share: float = 0.005,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure same-weekday residual sensitivity for city and industry groups."""
    city_daily = _city_daily_weather(weather)
    diagnostic_rows = []
    observation_rows = []

    for dimension in ("city", "industry"):
        profile = segment_profile(panel, dimension=dimension, cutoff=history_cutoff)
        labels = getattr(panel, dimension)
        for profile_row in profile.itertuples(index=False):
            if profile_row.positive_energy_share < minimum_load_share:
                continue
            segment = profile_row.segment
            users = {
                user_id
                for user_id, label in labels.items()
                if label.value == segment
            }
            if not users:
                continue
            weights, weight_meta = city_load_weights(
                panel,
                cutoff=history_cutoff,
                user_subset=users,
            )
            daily_weather = _weighted_daily_weather(city_daily, weights).set_index(
                "trade_date"
            )
            energy = _group_daily_energy(panel, users)
            rows = []
            for trade_date in sorted(energy):
                if not target_start <= trade_date <= target_end:
                    continue
                references = [
                    trade_date - timedelta(days=lag)
                    for lag in (7, 14, 21, 28)
                    if trade_date - timedelta(days=lag) in energy
                    and trade_date - timedelta(days=lag) in daily_weather.index
                ]
                if len(references) < 3 or trade_date not in daily_weather.index:
                    continue
                baseline_energy = float(np.mean([energy[item] for item in references]))
                if abs(baseline_energy) < 1e-9:
                    continue
                weather_row = daily_weather.loc[trade_date]
                reference_weather = daily_weather.loc[references].mean()
                rows.append(
                    {
                        "dimension": dimension,
                        "segment": segment,
                        "trade_date": trade_date,
                        "load_residual_pct": (
                            (energy[trade_date] - baseline_energy)
                            / abs(baseline_energy)
                            * 100
                        ),
                        "delta_temp_mean": (
                            weather_row["temp_mean"]
                            - reference_weather["temp_mean"]
                        ),
                        "delta_cooling_degree": (
                            weather_row["cooling_degree"]
                            - reference_weather["cooling_degree"]
                        ),
                        "delta_humidity_mean": (
                            weather_row["humidity_mean"]
                            - reference_weather["humidity_mean"]
                        ),
                        "delta_rain_total": (
                            weather_row["rain_total"]
                            - reference_weather["rain_total"]
                        ),
                    }
                )
            if not rows:
                continue
            observations = pd.DataFrame(rows)
            observation_rows.append(observations)
            diagnostic_rows.append(
                {
                    "dimension": dimension,
                    "segment": segment,
                    "user_count": profile_row.user_count,
                    "pretest_positive_energy_share": (
                        profile_row.positive_energy_share
                    ),
                    "known_city_energy_share_within_segment": (
                        weight_meta["known_city_energy_share"]
                    ),
                    "observation_days": len(observations),
                    "temp_residual_corr": _safe_corr(
                        observations["delta_temp_mean"],
                        observations["load_residual_pct"],
                    ),
                    "cooling_residual_corr": _safe_corr(
                        observations["delta_cooling_degree"],
                        observations["load_residual_pct"],
                    ),
                    "humidity_residual_corr": _safe_corr(
                        observations["delta_humidity_mean"],
                        observations["load_residual_pct"],
                    ),
                    "rain_residual_corr": _safe_corr(
                        observations["delta_rain_total"],
                        observations["load_residual_pct"],
                    ),
                    "load_residual_pp_per_degree": _single_slope(
                        observations["delta_temp_mean"],
                        observations["load_residual_pct"],
                    ),
                    "load_residual_pp_per_cooling_degree": _single_slope(
                        observations["delta_cooling_degree"],
                        observations["load_residual_pct"],
                    ),
                    "label_status": (
                        "AUTHORITATIVE_CITY_NAME_PROXY"
                        if dimension == "city"
                        and segment not in {UNKNOWN, AMBIGUOUS}
                        else "RESEARCH_NAME_PROXY"
                    ),
                }
            )
    diagnostics = pd.DataFrame(diagnostic_rows)
    observations = (
        pd.concat(observation_rows, ignore_index=True)
        if observation_rows
        else pd.DataFrame()
    )
    return diagnostics, observations


def _write_summary(
    output_dir: Path,
    model_summary: pd.DataFrame,
    city_profile: pd.DataFrame,
    industry_profile: pd.DataFrame,
    sensitivity: pd.DataFrame,
    weight_meta: dict,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        def display(value: object) -> str:
            if pd.isna(value):
                return ""
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.4f}"
            return str(value).replace("|", "\\|")

        headers = [str(column) for column in frame.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in frame.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(display(value) for value in row) + " |")
        return "\n".join(lines)

    overall = model_summary.loc[model_summary["segment_type"].eq("overall")].copy()
    pivot = overall.pivot(
        index="model",
        columns="weather_variant",
        values="wmape_pct",
    )
    if {"equal_13_city", "pretest_load_weighted"}.issubset(pivot.columns):
        pivot["weighted_minus_equal_pp"] = (
            pivot["pretest_load_weighted"] - pivot["equal_13_city"]
        )
    city_known = city_profile.loc[
        ~city_profile["segment"].isin([UNKNOWN, AMBIGUOUS]),
        "positive_energy_share",
    ].sum()
    industry_known = industry_profile.loc[
        ~industry_profile["segment"].isin([UNKNOWN, AMBIGUOUS]),
        "positive_energy_share",
    ].sum()
    lines = [
        "# 江苏用户负荷：分市气象与行业异质性研究",
        "",
        "## 结论口径",
        "",
        "- 用户城市与行业均来自企业名称规则，只能作为研究代理。",
        "- 城市权重只使用首个回测日前 D-3 已回流的正负荷历史。",
        "- 天气仍是未版本化的事后快照，因此所有天气结果均不可晋级 Champion。",
        f"- 城市代理覆盖历史正负荷：{city_known:.2%}。",
        f"- 行业单一关键词代理覆盖历史正负荷：{industry_known:.2%}。",
        f"- 城市明确部分占比最高的城市："
        f"{city_profile.iloc[0]['segment']} "
        f"({city_profile.iloc[0]['positive_energy_share']:.2%})。",
        f"- 未定位负荷按13市等比例摊回；已定位城市负荷占比："
        f"{weight_meta['known_city_energy_share']:.2%}。",
        "",
        "## 天气聚合回测",
        "",
        markdown_table(pivot.reset_index()),
        "",
        "## 行业天气敏感性",
        "",
    ]
    industry_sensitivity = sensitivity.loc[
        sensitivity["dimension"].eq("industry")
    ].sort_values(
        "pretest_positive_energy_share",
        ascending=False,
    )
    display_columns = [
        "segment",
        "user_count",
        "pretest_positive_energy_share",
        "observation_days",
        "temp_residual_corr",
        "load_residual_pp_per_degree",
    ]
    lines.append(markdown_table(industry_sensitivity[display_columns]))
    (output_dir / "research_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-input", type=Path, required=True)
    parser.add_argument("--weather-input", type=Path, required=True)
    parser.add_argument("--weather-source", default="Ts_weather_1")
    parser.add_argument(
        "--models",
        default="gbdt,hist_gbdt,extra_trees",
    )
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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_output_dir(args.output_dir)
    history_cutoff = args.target_start - timedelta(days=3)
    segmented = read_segmented_user_panel(args.load_input)
    aggregate = read_aggregated_load_snapshot(args.load_input)
    city_weather = read_city_weather_panel(
        args.weather_input,
        source=args.weather_source,
    )

    city_profile = segment_profile(
        segmented,
        dimension="city",
        cutoff=history_cutoff,
    )
    industry_profile = segment_profile(
        segmented,
        dimension="industry",
        cutoff=history_cutoff,
    )
    load_weights, weight_meta = city_load_weights(
        segmented,
        cutoff=history_cutoff,
    )
    variants = {
        "equal_13_city": _uniform_weights(),
        "pretest_load_weighted": load_weights,
    }
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())

    summaries = []
    statuses = []
    for variant, weights in variants.items():
        print(f"[regional-weather] start variant={variant}", flush=True)
        variant_dir = args.output_dir / variant
        variant_dir.mkdir()
        weather_store = weighted_weather_store(
            city_weather,
            city_weights=weights,
            variant=variant,
        )
        features = build_feature_frame(aggregate, weather=weather_store)
        predictions, summary, status, failures = run_model_pool(
            features,
            models=models,
            target_start=args.target_start,
            target_end=args.target_end,
            include_weather=True,
            target_mode="residual",
        )
        daily, _ = summarize_predictions(predictions)
        predictions.insert(1, "weather_variant", variant)
        daily.insert(1, "weather_variant", variant)
        summary.insert(1, "weather_variant", variant)
        status.insert(1, "weather_variant", variant)
        features.to_csv(
            variant_dir / "feature_snapshot.csv.gz",
            index=False,
            compression="gzip",
        )
        predictions.to_csv(
            variant_dir / "predictions_96.csv.gz",
            index=False,
            compression="gzip",
        )
        daily.to_csv(
            variant_dir / "daily_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        summary.to_csv(
            variant_dir / "summary_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        status.to_csv(
            variant_dir / "model_status.csv",
            index=False,
            encoding="utf-8-sig",
        )
        (variant_dir / "failures.json").write_text(
            json.dumps(failures, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summaries.append(summary)
        statuses.append(status)
        print(f"[regional-weather] done variant={variant}", flush=True)

    model_summary = pd.concat(summaries, ignore_index=True)
    model_status = pd.concat(statuses, ignore_index=True)
    sensitivity, observations = build_sensitivity_diagnostics(
        segmented,
        city_weather,
        history_cutoff=history_cutoff,
        target_start=args.target_start,
        target_end=args.target_end,
    )
    city_profile.to_csv(
        args.output_dir / "city_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )
    industry_profile.to_csv(
        args.output_dir / "industry_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "city": city,
                "equal_weight": variants["equal_13_city"][city],
                "pretest_load_weight": variants["pretest_load_weighted"][city],
            }
            for city in JIANGSU_CITIES
        ]
    ).to_csv(
        args.output_dir / "city_weather_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )
    model_summary.to_csv(
        args.output_dir / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    model_status.to_csv(
        args.output_dir / "model_status.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensitivity.to_csv(
        args.output_dir / "segment_weather_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    observations.to_csv(
        args.output_dir / "segment_weather_observations.csv.gz",
        index=False,
        compression="gzip",
    )
    metadata = {
        "version": RESEARCH_VERSION,
        "segmentation_version": SEGMENTATION_VERSION,
        "regional_weather_version": REGIONAL_WEATHER_VERSION,
        "province_display": "江苏",
        "province_slug": "jiangsu",
        "raw_schema": "jiangsu",
        "load_source_sha256": aggregate.source_sha256,
        "weather_source_sha256": city_weather.source_sha256,
        "weather_source": args.weather_source,
        "weather_policy": "RESEARCH_PROXY_UNVERIFIED_ISSUE_TIME",
        "champion_eligible": False,
        "target_start": args.target_start.isoformat(),
        "target_end": args.target_end.isoformat(),
        "weight_history_cutoff": history_cutoff.isoformat(),
        "weight_policy": weight_meta,
        "models": list(models),
        "as_of_policy": (
            "load labels through D-3; city weights frozen at target_start D-3; "
            "target-day weather is diagnostic proxy only"
        ),
        "pii_output_check": "aggregate segments only; no names or account identifiers",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary(
        args.output_dir,
        model_summary,
        city_profile,
        industry_profile,
        sensitivity,
        weight_meta,
    )
    print(
        model_summary.loc[
            model_summary["segment_type"].eq("overall"),
            ["weather_variant", "model", "wmape_pct", "daily_wmape_p90_pct"],
        ]
        .sort_values(["model", "wmape_pct"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
