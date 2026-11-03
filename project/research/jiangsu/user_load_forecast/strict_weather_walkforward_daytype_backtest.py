"""Daily D-3 walk-forward weather models with validation-selected day-type routing."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from project.research.jiangsu.user_load_forecast.model_backtest import prepare_output_dir
from project.research.jiangsu.user_load_forecast.pv_sensitive_segment_backtest import (
    STRICT_MARKET_POLICY,
)
from project.research.jiangsu.user_load_forecast.pv_weather_challenger_audit import (
    score_window,
)
from project.research.jiangsu.user_load_forecast.strict_operational_weather_backtest import (
    WEATHER_POLICY,
    daily_metrics,
    sha256_file,
)
from project.research.jiangsu.user_load_forecast.strict_operational_weather_model_sweep import (
    dense_preprocessor,
    model_factories,
)
from project.research.jiangsu.user_load_forecast.strict_weekend_hierarchical_expert_backtest import (
    build_frames,
    grouped_window_metrics,
    selection_metrics,
)


VERSION = "surun_strict_weather_walkforward_daytype_v1"
VARIANTS = ("core_weather", "solar_weather")
WINDOWS = (None, 42, 28)


def candidate_name(variant: str, window_days: int | None) -> str:
    suffix = "expanding" if window_days is None else f"w{window_days}"
    return f"walkforward_{variant}__{suffix}"


def fit_one_day(
    frame: pd.DataFrame,
    *,
    model_name: str,
    features: tuple[str, ...],
    target_date: date,
    window_days: int | None,
) -> pd.DataFrame:
    eligible = frame.dropna(subset=list(features)).copy()
    train_end = target_date - timedelta(days=3)
    train = eligible["target_date"].le(train_end)
    if window_days is not None:
        train_start = train_end - timedelta(days=window_days - 1)
        train &= eligible["target_date"].ge(train_start)
    predict = eligible["target_date"].eq(target_date)
    if not train.any() or not predict.any():
        raise ValueError(f"Empty walk-forward train/predict for {model_name}/{target_date}")
    pipeline = Pipeline(
        [
            ("preprocessor", dense_preprocessor(features)),
            ("model", model_factories()["gbdt_huber"]()),
        ]
    )
    pipeline.fit(
        eligible.loc[train, list(features)],
        eligible.loc[train, "residual_ratio"],
    )
    residual = np.clip(
        pipeline.predict(eligible.loc[predict, list(features)]), -0.35, 0.35
    )
    output = eligible.loc[
        predict,
        [
            "target_date",
            "trade_interval",
            "day_type",
            "actual_load",
            "calendar_prediction",
            "load_label_cutoff",
            "market_rt_cutoff",
            "market_source_max_date",
        ],
    ].copy()
    output.insert(0, "model", model_name)
    output["predicted_load"] = np.maximum(
        output["calendar_prediction"].to_numpy(dtype=float) * (1.0 + residual), 0.0
    )
    output["error"] = output["predicted_load"] - output["actual_load"]
    output["absolute_error"] = output["error"].abs()
    output["fit_train_end"] = train_end
    output["fit_train_start"] = (
        eligible.loc[train, "target_date"].min() if train.any() else pd.NaT
    )
    return output


def run_candidates(
    frame: pd.DataFrame,
    variants: dict[str, tuple[str, ...]],
    target_dates: list[date],
) -> dict[str, pd.DataFrame]:
    parts: dict[str, list[pd.DataFrame]] = {}
    for target_date in target_dates:
        for variant in VARIANTS:
            for window_days in WINDOWS:
                name = candidate_name(variant, window_days)
                prediction = fit_one_day(
                    frame,
                    model_name=name,
                    features=variants[variant],
                    target_date=target_date,
                    window_days=window_days,
                )
                parts.setdefault(name, []).append(prediction)
    return {name: pd.concat(value, ignore_index=True) for name, value in parts.items()}


def routing_group(day_type: pd.Series) -> pd.Series:
    return day_type.replace({"adjusted_workday": "workday"})


def build_router(
    candidates: dict[str, pd.DataFrame],
    route: dict[str, str],
    *,
    model_name: str,
) -> pd.DataFrame:
    ordered = {
        name: frame.sort_values(["target_date", "trade_interval"]).reset_index(drop=True)
        for name, frame in candidates.items()
    }
    base = next(iter(ordered.values())).copy()
    prediction = base["predicted_load"].to_numpy(dtype=float).copy()
    groups = routing_group(base["day_type"]).to_numpy()
    for group, candidate in route.items():
        source = ordered[candidate]
        keys = ["target_date", "trade_interval", "day_type", "actual_load"]
        if not base[keys].equals(source[keys]):
            raise ValueError("Walk-forward candidate rows are not aligned")
        mask = groups == group
        prediction[mask] = source.loc[mask, "predicted_load"].to_numpy(dtype=float)
    base["model"] = model_name
    base["predicted_load"] = prediction
    base["error"] = base["predicted_load"] - base["actual_load"]
    base["absolute_error"] = base["error"].abs()
    return base


def choose_route(candidates: dict[str, pd.DataFrame]) -> dict[str, str]:
    route = {}
    for group in ("workday", "weekend", "holiday"):
        rows = []
        for name, frame in candidates.items():
            scoped = frame.loc[routing_group(frame["day_type"]).eq(group)]
            if scoped.empty:
                continue
            rows.append(
                (
                    100.0 * scoped["absolute_error"].sum() / scoped["actual_load"].sum(),
                    name,
                )
            )
        route[group] = min(rows)[1]
    return route


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    frame, variants = build_frames(args)
    available = set(frame["target_date"].unique())
    selection_dates = [
        value.date()
        for value in pd.date_range(args.selection_start, args.selection_end, freq="D")
        if value.date() in available
    ]
    evaluation_dates = [
        value
        for value in sorted(available)
        if args.holdout_start <= value <= args.new_diagnostic_end
    ]

    selection_candidates = run_candidates(frame, variants, selection_dates)
    route = choose_route(selection_candidates)
    selection_candidates["walkforward_daytype_router"] = build_router(
        selection_candidates,
        route,
        model_name="walkforward_daytype_router",
    )
    selection_rows = [
        {"model": name, **selection_metrics(prediction)}
        for name, prediction in selection_candidates.items()
    ]
    selection = pd.DataFrame(selection_rows).sort_values(
        ["wmape_pct", "weekend_wmape_pct", "daily_wmape_p90_pct", "model"]
    )
    selected = str(selection.iloc[0]["model"])

    final_candidates = run_candidates(frame, variants, evaluation_dates)
    final_candidates["walkforward_daytype_router"] = build_router(
        final_candidates,
        route,
        model_name="walkforward_daytype_router",
    )
    selected_prediction = final_candidates[selected].copy()
    selected_prediction["model"] = "walkforward_selected"
    predictions = pd.concat(
        [*final_candidates.values(), selected_prediction], ignore_index=True
    )

    summary_rows = []
    group_rows = []
    for window, start, end in (
        ("legacy_holdout", args.holdout_start, args.legacy_holdout_end),
        ("new_diagnostic", args.new_diagnostic_start, args.new_diagnostic_end),
    ):
        for model in predictions["model"].unique():
            score = score_window(
                predictions.loc[predictions["model"].eq(model)],
                window=window,
                start=start,
                end=end,
            )
            if score:
                summary_rows.append(score)
        group_rows.extend(
            grouped_window_metrics(predictions, window=window, start=start, end=end)
        )
    summary = pd.DataFrame(summary_rows).sort_values(["window", "wmape_pct", "model"])
    groups = pd.DataFrame(group_rows).sort_values(
        ["window", "group", "wmape_pct", "model"]
    )
    daily = daily_metrics(predictions)

    selection_predictions = pd.concat(selection_candidates.values(), ignore_index=True)
    selection_predictions.to_csv(
        args.output_dir / "selection_predictions_96.csv.gz",
        index=False,
        compression="gzip",
    )
    predictions.to_csv(
        args.output_dir / "predictions_96.csv.gz", index=False, compression="gzip"
    )
    selection.to_csv(
        args.output_dir / "selection_summary.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        args.output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig"
    )
    groups.to_csv(
        args.output_dir / "group_metrics.csv", index=False, encoding="utf-8-sig"
    )
    daily.to_csv(
        args.output_dir / "daily_metrics.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "version": VERSION,
        "built_at": datetime.now().astimezone().isoformat(),
        "status": "OFFLINE_RESEARCH_ONLY",
        "selection_policy": "MAY_WINDOW_ONLY_BEFORE_LEGACY_HOLDOUT",
        "selection_window": [args.selection_start.isoformat(), args.selection_end.isoformat()],
        "candidate_count": len(selection_candidates),
        "route": route,
        "selected": selected,
        "selected_metrics": selection.iloc[0].to_dict(),
        "retraining_policy": "DAILY_TARGET_MINUS_3_LABEL_CUTOFF",
        "market_policy": STRICT_MARKET_POLICY,
        "weather_policy": WEATHER_POLICY,
        "target_day_da_feature_check": "PASS_NO_TARGET_DAY_DA",
        "portfolio_input": str(args.portfolio_input.resolve()),
        "portfolio_sha256": sha256_file(args.portfolio_input),
        "market_inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.market_input
        ],
        "weather_input": str(args.weather_input.resolve()),
        "weather_sha256": sha256_file(args.weather_input),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
    print(summary[["model", "window", "wmape_pct"]].to_string(index=False))
    print(groups.loc[groups["group"].isin(["workday", "weekend", "holiday"])][
        ["model", "window", "group", "wmape_pct", "bias_pct"]
    ].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-input", type=Path, required=True)
    parser.add_argument("--market-input", type=Path, action="append", required=True)
    parser.add_argument("--weather-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selection-start", type=date.fromisoformat, default=date(2026, 5, 1)
    )
    parser.add_argument(
        "--selection-end", type=date.fromisoformat, default=date(2026, 5, 31)
    )
    parser.add_argument(
        "--holdout-start", type=date.fromisoformat, default=date(2026, 6, 15)
    )
    parser.add_argument(
        "--legacy-holdout-end", type=date.fromisoformat, default=date(2026, 7, 25)
    )
    parser.add_argument(
        "--new-diagnostic-start", type=date.fromisoformat, default=date(2026, 7, 26)
    )
    parser.add_argument(
        "--new-diagnostic-end", type=date.fromisoformat, default=date(2026, 8, 5)
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
