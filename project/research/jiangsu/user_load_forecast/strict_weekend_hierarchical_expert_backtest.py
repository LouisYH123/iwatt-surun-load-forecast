"""Backtest a shrinkage-controlled non-workday expert under strict visibility.

Candidate selection uses three expanding, pre-holdout rolling-origin folds.
The legacy holdout and latest diagnostic dates are evaluated only after the
expert family, feature set, interval gate, and blend weight are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from project.research.jiangsu.user_load_forecast.market_level_shape_backtest import (
    read_market_matrix,
)
from project.research.jiangsu.user_load_forecast.model_backtest import (
    prepare_output_dir,
)
from project.research.jiangsu.user_load_forecast.pv_sensitive_segment_backtest import (
    STRICT_MARKET_POLICY,
    feature_sets,
    prepare_feature_frame,
    read_portfolio_curves,
)
from project.research.jiangsu.user_load_forecast.pv_weather_challenger_audit import (
    WEATHER_FEATURES,
    attach_weather,
    combine_market_inputs,
    score_window,
)
from project.research.jiangsu.user_load_forecast.strict_operational_weather_backtest import (
    WEATHER_POLICY,
    daily_metrics,
    sha256_file,
)
from project.research.jiangsu.user_load_forecast.strict_operational_weather_model_sweep import (
    dense_preprocessor,
    fit_candidate,
    model_factories,
)


VERSION = "surun_strict_weekend_hierarchical_expert_v2"
ANCHOR_FAMILY = "gbdt_huber"
ANCHOR_VARIANTS = (
    "core_weather",
    "solar_weather",
    "core_weather_recency",
    "solar_weather_recency",
)
EXPERT_FAMILIES = ("weekend_huber", "weekend_hist", "weekend_ridge")
EXPERT_VARIANTS = ("core_weather_recency", "solar_weather_recency")
ALPHAS = (0.25, 0.5, 0.75, 1.0)
GATES = (
    "weekend_all",
    "weekend_morning_25_52",
    "weekend_morning_33_52",
    "weekend_daylight_25_72",
)
RECENCY_FEATURES = (
    "log_recent_same_weekday_load",
    "recent_same_weekday_vs_calendar",
    "recent_same_weekday_vs_previous",
    "recent_same_weekday_age_days",
)
SELECTION_FOLDS = (
    (date(2026, 4, 28), date(2026, 5, 1), date(2026, 5, 10)),
    (date(2026, 5, 8), date(2026, 5, 11), date(2026, 5, 20)),
    (date(2026, 5, 18), date(2026, 5, 21), date(2026, 5, 31)),
)


def expert_factories() -> dict[str, Callable[[], object]]:
    return {
        "weekend_huber": lambda: GradientBoostingRegressor(
            loss="huber",
            n_estimators=100,
            learning_rate=0.03,
            max_depth=1,
            min_samples_leaf=24,
            random_state=42,
        ),
        "weekend_hist": lambda: HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=160,
            learning_rate=0.03,
            max_leaf_nodes=7,
            min_samples_leaf=32,
            l2_regularization=8.0,
            random_state=42,
        ),
        "weekend_ridge": lambda: Ridge(alpha=25.0),
    }


def fit_weekend_expert(
    frame: pd.DataFrame,
    *,
    model_name: str,
    features: tuple[str, ...],
    estimator_factory: Callable[[], object],
    train_end: date,
    predict_start: date,
) -> pd.DataFrame:
    eligible = frame.dropna(subset=list(features)).copy()
    train = eligible["target_date"].le(train_end) & eligible["day_type"].eq("weekend")
    predict = eligible["target_date"].ge(predict_start) & eligible["day_type"].eq(
        "weekend"
    )
    if not train.any() or not predict.any():
        raise ValueError(f"Empty weekend train/predict window for {model_name}")
    pipeline = Pipeline(
        [
            ("preprocessor", dense_preprocessor(features)),
            ("model", estimator_factory()),
        ]
    )
    pipeline.fit(
        eligible.loc[train, list(features)],
        eligible.loc[train, "residual_ratio"],
    )
    residual = np.clip(
        pipeline.predict(eligible.loc[predict, list(features)]), -0.35, 0.35
    )
    result = eligible.loc[
        predict,
        ["target_date", "trade_interval", "day_type", "calendar_prediction"],
    ].copy()
    result["expert_prediction"] = np.maximum(
        result["calendar_prediction"].to_numpy(dtype=float) * (1.0 + residual),
        0.0,
    )
    return result.drop(columns="calendar_prediction")


def gate_mask(frame: pd.DataFrame, gate: str) -> np.ndarray:
    weekend = frame["day_type"].eq("weekend").to_numpy()
    interval = frame["trade_interval"].to_numpy(dtype=int)
    if gate == "weekend_all":
        return weekend
    if gate == "weekend_morning_25_52":
        return weekend & (interval >= 25) & (interval <= 52)
    if gate == "weekend_morning_33_52":
        return weekend & (interval >= 33) & (interval <= 52)
    if gate == "weekend_daylight_25_72":
        return weekend & (interval >= 25) & (interval <= 72)
    raise ValueError(f"Unknown gate: {gate}")


def blend_prediction(
    anchor: pd.DataFrame,
    expert: pd.DataFrame,
    *,
    model_name: str,
    gate: str,
    alpha: float,
) -> pd.DataFrame:
    output = anchor.merge(
        expert,
        on=["target_date", "trade_interval", "day_type"],
        how="left",
        validate="one_to_one",
    )
    mask = gate_mask(output, gate) & output["expert_prediction"].notna().to_numpy()
    prediction = output["predicted_load"].to_numpy(dtype=float).copy()
    prediction[mask] = (
        (1.0 - alpha) * prediction[mask]
        + alpha * output.loc[mask, "expert_prediction"].to_numpy(dtype=float)
    )
    output["model"] = model_name
    output["predicted_load"] = prediction
    output["error"] = output["predicted_load"] - output["actual_load"]
    output["absolute_error"] = output["error"].abs()
    return output.drop(columns="expert_prediction")


def simple_metrics(frame: pd.DataFrame) -> dict[str, float]:
    actual = frame["actual_load"].to_numpy(dtype=float)
    error = frame["error"].to_numpy(dtype=float)
    return {
        "wmape_pct": float(100.0 * np.abs(error).sum() / actual.sum()),
        "bias_pct": float(100.0 * error.sum() / actual.sum()),
    }


def selection_metrics(frame: pd.DataFrame) -> dict[str, float]:
    overall = simple_metrics(frame)
    nonworkday = simple_metrics(frame.loc[frame["day_type"].ne("workday")])
    weekend = simple_metrics(frame.loc[frame["day_type"].eq("weekend")])
    holiday_frame = frame.loc[frame["day_type"].eq("holiday")]
    holiday = simple_metrics(holiday_frame) if not holiday_frame.empty else {
        "wmape_pct": np.nan,
        "bias_pct": np.nan,
    }
    daily = daily_metrics(frame)
    return {
        "wmape_pct": overall["wmape_pct"],
        "bias_pct": overall["bias_pct"],
        "nonworkday_wmape_pct": nonworkday["wmape_pct"],
        "weekend_wmape_pct": weekend["wmape_pct"],
        "holiday_wmape_pct": holiday["wmape_pct"],
        "daily_wmape_p90_pct": float(daily["wmape_pct"].quantile(0.9)),
    }


def grouped_window_metrics(
    predictions: pd.DataFrame,
    *,
    window: str,
    start: date,
    end: date,
) -> list[dict[str, object]]:
    scoped = predictions.loc[
        predictions["target_date"].between(start, end)
    ].copy()
    rows = []
    for model, model_frame in scoped.groupby("model", sort=True):
        for group in ("all", "workday", "weekend", "holiday", "nonworkday"):
            if group == "all":
                part = model_frame
            elif group == "nonworkday":
                part = model_frame.loc[model_frame["day_type"].ne("workday")]
            else:
                part = model_frame.loc[model_frame["day_type"].eq(group)]
            if part.empty:
                continue
            metrics = simple_metrics(part)
            rows.append(
                {
                    "model": model,
                    "window": window,
                    "group": group,
                    "window_start": start,
                    "window_end": end,
                    "target_days": part["target_date"].nunique(),
                    **metrics,
                }
            )
    return rows


def attach_same_weekday_recency(
    frame: pd.DataFrame,
    curves: dict[date, np.ndarray],
) -> pd.DataFrame:
    result = frame.copy()
    available_dates = sorted(curves)
    blocks = []
    for target_date in sorted(result["target_date"].unique()):
        cutoff = target_date - pd.Timedelta(days=3)
        cutoff_date = cutoff.date() if hasattr(cutoff, "date") else cutoff
        same_weekday = [
            value
            for value in available_dates
            if value <= cutoff_date and value.weekday() == target_date.weekday()
        ]
        if same_weekday:
            recent_date = same_weekday[-1]
            previous_date = same_weekday[-2] if len(same_weekday) >= 2 else recent_date
            recent = curves[recent_date]
            previous = curves[previous_date]
            age = float((target_date - recent_date).days)
        else:
            recent = np.full(96, np.nan)
            previous = np.full(96, np.nan)
            age = np.nan
        blocks.append(
            pd.DataFrame(
                {
                    "target_date": target_date,
                    "trade_interval": np.arange(1, 97),
                    "recent_same_weekday_load": recent,
                    "previous_same_weekday_load": previous,
                    "recent_same_weekday_age_days": age,
                }
            )
        )
    recency = pd.concat(blocks, ignore_index=True)
    result = result.merge(
        recency,
        on=["target_date", "trade_interval"],
        how="left",
        validate="one_to_one",
    )
    result["log_recent_same_weekday_load"] = np.log1p(
        result["recent_same_weekday_load"]
    )
    result["recent_same_weekday_vs_calendar"] = np.divide(
        result["recent_same_weekday_load"],
        result["calendar_prediction"],
    )
    result["recent_same_weekday_vs_previous"] = np.divide(
        result["recent_same_weekday_load"],
        result["previous_same_weekday_load"],
    )
    return result


def build_frames(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    curves = read_portfolio_curves(args.portfolio_input)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        market_path = temp_root / "combined_market.csv.gz"
        combine_market_inputs(args.market_input, market_path)
        strict_frame, groups = prepare_feature_frame(
            curves,
            component="aggregate_strict_weekend_hierarchical_expert",
            market_path=market_path,
            market_matrix=read_market_matrix(market_path),
            weather_input=args.weather_input,
            temp_root=temp_root,
            visibility_policy="strict_submission_asof",
            strict_predictor_weather_input=None,
            strict_market_fallback=True,
            calendar_start=date(2026, 4, 4),
        )
    frame = attach_weather(strict_frame, args.weather_input)
    frame = attach_same_weekday_recency(frame, curves)
    strict_sets = feature_sets(groups, visibility_policy="strict_submission_asof")
    variants = {
        "core_weather": (*strict_sets["core"], *WEATHER_FEATURES),
        "solar_weather": (*strict_sets["solar"], *WEATHER_FEATURES),
        "core_weather_recency": (
            *strict_sets["core"],
            *WEATHER_FEATURES,
            *RECENCY_FEATURES,
        ),
        "solar_weather_recency": (
            *strict_sets["solar"],
            *WEATHER_FEATURES,
            *RECENCY_FEATURES,
        ),
    }
    return frame, variants


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    frame, variants = build_frames(args)
    anchor_factory = model_factories()[ANCHOR_FAMILY]
    expert_factory_map = expert_factories()
    candidate_predictions: dict[str, list[pd.DataFrame]] = {}
    candidate_configs: dict[str, dict[str, object]] = {}

    for fold_index, (train_end, validation_start, validation_end) in enumerate(
        SELECTION_FOLDS, start=1
    ):
        anchors = {}
        for anchor_variant in ANCHOR_VARIANTS:
            anchor_name = f"anchor_{anchor_variant}"
            prediction = fit_candidate(
                frame,
                model_name=anchor_name,
                features=variants[anchor_variant],
                estimator_factory=anchor_factory,
                train_end=train_end,
                predict_start=validation_start,
            )
            prediction = prediction.loc[
                prediction["target_date"].le(validation_end)
            ].copy()
            prediction["selection_fold"] = fold_index
            anchors[anchor_variant] = prediction
            candidate_predictions.setdefault(anchor_name, []).append(prediction)
            candidate_configs[anchor_name] = {
                "kind": "anchor",
                "anchor_variant": anchor_variant,
            }

        experts = {}
        for family in EXPERT_FAMILIES:
            for expert_variant in EXPERT_VARIANTS:
                expert_key = (family, expert_variant)
                expert_name = f"{family}__{expert_variant}"
                prediction = fit_weekend_expert(
                    frame,
                    model_name=expert_name,
                    features=variants[expert_variant],
                    estimator_factory=expert_factory_map[family],
                    train_end=train_end,
                    predict_start=validation_start,
                )
                experts[expert_key] = prediction.loc[
                    prediction["target_date"].le(validation_end)
                ].copy()

        for anchor_variant, anchor in anchors.items():
            for (family, expert_variant), expert in experts.items():
                for gate in GATES:
                    for alpha in ALPHAS:
                        name = (
                            f"hier_{anchor_variant}__{family}__{expert_variant}"
                            f"__{gate}__a{alpha:g}"
                        )
                        prediction = blend_prediction(
                            anchor,
                            expert,
                            model_name=name,
                            gate=gate,
                            alpha=alpha,
                        )
                        prediction["selection_fold"] = fold_index
                        candidate_predictions.setdefault(name, []).append(prediction)
                        candidate_configs[name] = {
                            "kind": "hierarchical_expert",
                            "anchor_variant": anchor_variant,
                            "expert_family": family,
                            "expert_variant": expert_variant,
                            "gate": gate,
                            "alpha": alpha,
                        }

    selection_rows = []
    pooled_predictions = {}
    for model, parts in candidate_predictions.items():
        pooled = pd.concat(parts, ignore_index=True)
        pooled_predictions[model] = pooled
        selection_rows.append({"model": model, **selection_metrics(pooled)})
    selection = pd.DataFrame(selection_rows).sort_values(
        ["wmape_pct", "nonworkday_wmape_pct", "daily_wmape_p90_pct", "model"]
    )
    selected = str(selection.iloc[0]["model"])
    selected_config = candidate_configs[selected]
    selected_oof = pooled_predictions[selected]

    final_anchors = {}
    for anchor_variant in ANCHOR_VARIANTS:
        name = f"anchor_{anchor_variant}"
        final_anchors[anchor_variant] = fit_candidate(
            frame,
            model_name=name,
            features=variants[anchor_variant],
            estimator_factory=anchor_factory,
            train_end=args.final_train_end,
            predict_start=args.holdout_start,
        )
    if selected_config["kind"] == "anchor":
        selected_final = final_anchors[str(selected_config["anchor_variant"])].copy()
        selected_final["model"] = "weekend_hierarchical_selected"
    else:
        expert = fit_weekend_expert(
            frame,
            model_name="selected_nonworkday_expert",
            features=variants[str(selected_config["expert_variant"])],
            estimator_factory=expert_factory_map[str(selected_config["expert_family"])],
            train_end=args.final_train_end,
            predict_start=args.holdout_start,
        )
        selected_final = blend_prediction(
            final_anchors[str(selected_config["anchor_variant"])],
            expert,
            model_name="weekend_hierarchical_selected",
            gate=str(selected_config["gate"]),
            alpha=float(selected_config["alpha"]),
        )
    final_predictions = pd.concat(
        [*final_anchors.values(), selected_final], ignore_index=True
    )

    summary_rows = []
    group_rows = []
    for window, start, end in (
        ("legacy_holdout", args.holdout_start, args.legacy_holdout_end),
        ("new_diagnostic", args.new_diagnostic_start, args.new_diagnostic_end),
    ):
        for model in final_predictions["model"].unique():
            row = score_window(
                final_predictions.loc[final_predictions["model"].eq(model)],
                window=window,
                start=start,
                end=end,
            )
            if row:
                summary_rows.append(row)
        group_rows.extend(
            grouped_window_metrics(
                final_predictions, window=window, start=start, end=end
            )
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["window", "wmape_pct", "model"]
    )
    groups = pd.DataFrame(group_rows).sort_values(
        ["window", "group", "wmape_pct", "model"]
    )
    daily = daily_metrics(final_predictions)

    anchor_name = f"anchor_{selected_config['anchor_variant']}"
    legacy_selected = groups.loc[
        groups["window"].eq("legacy_holdout")
        & groups["group"].eq("all")
        & groups["model"].eq("weekend_hierarchical_selected"),
        "wmape_pct",
    ].iloc[0]
    legacy_anchor = groups.loc[
        groups["window"].eq("legacy_holdout")
        & groups["group"].eq("all")
        & groups["model"].eq(anchor_name),
        "wmape_pct",
    ].iloc[0]
    nonwork_selected = groups.loc[
        groups["window"].eq("legacy_holdout")
        & groups["group"].eq("nonworkday")
        & groups["model"].eq("weekend_hierarchical_selected"),
        "wmape_pct",
    ].iloc[0]
    nonwork_anchor = groups.loc[
        groups["window"].eq("legacy_holdout")
        & groups["group"].eq("nonworkday")
        & groups["model"].eq(anchor_name),
        "wmape_pct",
    ].iloc[0]
    promotion = bool(
        selected_config["kind"] == "hierarchical_expert"
        and legacy_selected < legacy_anchor
        and nonwork_selected < nonwork_anchor
    )

    selected_oof.to_csv(
        args.output_dir / "selected_oof_predictions_96.csv.gz",
        index=False,
        compression="gzip",
    )
    final_predictions.to_csv(
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
        "status": "PROMOTION_CANDIDATE" if promotion else "NO_PROMOTION",
        "selection_policy": "THREE_PRE_HOLDOUT_ROLLING_ORIGIN_FOLDS_ONLY",
        "selection_folds": [
            {
                "train_end": train_end.isoformat(),
                "validation_start": start.isoformat(),
                "validation_end": end.isoformat(),
            }
            for train_end, start, end in SELECTION_FOLDS
        ],
        "candidate_count": len(candidate_configs),
        "selected": selected,
        "selected_config": selected_config,
        "selected_oof_metrics": selection.iloc[0].to_dict(),
        "selected_anchor": anchor_name,
        "legacy_selected_wmape_pct": float(legacy_selected),
        "legacy_anchor_wmape_pct": float(legacy_anchor),
        "legacy_selected_nonworkday_wmape_pct": float(nonwork_selected),
        "legacy_anchor_nonworkday_wmape_pct": float(nonwork_anchor),
        "promotion_recommended": promotion,
        "market_policy": STRICT_MARKET_POLICY,
        "strict_market_fallback": "LATEST_CAUSALLY_VISIBLE_DATE_WITH_AGE_FEATURE",
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
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "run_summary.md").write_text(
        "# 江苏用户负荷：周末/节假日分层专家离线回测\n\n"
        f"- 预留出滚动验证比较 {len(candidate_configs)} 个候选。\n"
        f"- 选择：`{selected}`。\n"
        f"- 41天整体：专家 {legacy_selected:.4f}%，锚点 {legacy_anchor:.4f}%。\n"
        f"- 41天非工作日：专家 {nonwork_selected:.4f}%，锚点 {nonwork_anchor:.4f}%。\n"
        f"- 结论：{'建议进入下一轮门禁' if promotion else '不晋级'}。\n",
        encoding="utf-8",
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
        "--final-train-end", type=date.fromisoformat, default=date(2026, 6, 12)
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
