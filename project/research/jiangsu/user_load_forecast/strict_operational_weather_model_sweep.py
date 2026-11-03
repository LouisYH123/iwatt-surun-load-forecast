"""Offline model-family sweep under strict T-2 market and operational D weather.

Candidate selection uses only the frozen internal-validation window.  The
legacy holdout and new diagnostic windows are evaluated after selection and
cannot be used to choose the promoted model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

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


VERSION = "surun_strict_operational_weather_model_sweep_v1"
INCUMBENT = "gbdt_huber__solar_weather"


def dense_preprocessor(features: tuple[str, ...]) -> ColumnTransformer:
    numeric = [column for column in features if column != "day_type"]
    return ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            (
                "day_type",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["day_type"],
            ),
        ]
    )


def model_factories() -> dict[str, Callable[[], object]]:
    return {
        "gbdt_huber": lambda: GradientBoostingRegressor(
            loss="huber",
            n_estimators=120,
            learning_rate=0.04,
            max_depth=2,
            min_samples_leaf=12,
            random_state=42,
        ),
        "hist_gbdt": lambda: HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=220,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=24,
            l2_regularization=2.0,
            random_state=42,
        ),
        "extra_trees": lambda: ExtraTreesRegressor(
            n_estimators=260,
            min_samples_leaf=8,
            max_features=0.8,
            n_jobs=2,
            random_state=42,
        ),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=220,
            min_samples_leaf=8,
            max_features=0.8,
            n_jobs=2,
            random_state=42,
        ),
        "xgboost": lambda: XGBRegressor(
            objective="reg:squarederror",
            n_estimators=280,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=8.0,
            reg_alpha=0.05,
            n_jobs=2,
            random_state=42,
            verbosity=0,
        ),
        "lightgbm": lambda: LGBMRegressor(
            objective="huber",
            n_estimators=280,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            reg_alpha=0.05,
            deterministic=True,
            force_col_wise=True,
            n_jobs=2,
            random_state=42,
            verbosity=-1,
        ),
        "catboost": lambda: CatBoostRegressor(
            loss_function="RMSE",
            iterations=320,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=10.0,
            random_seed=42,
            thread_count=2,
            verbose=False,
            allow_writing_files=False,
        ),
    }


def fit_candidate(
    frame: pd.DataFrame,
    *,
    model_name: str,
    features: tuple[str, ...],
    estimator_factory: Callable[[], object],
    train_end: date,
    predict_start: date,
) -> pd.DataFrame:
    eligible = frame.dropna(subset=list(features)).copy()
    train = eligible["target_date"] <= train_end
    predict = eligible["target_date"] >= predict_start
    if not train.any() or not predict.any():
        raise ValueError(f"Empty train/predict window for {model_name}")
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
        pipeline.predict(eligible.loc[predict, list(features)]),
        -0.35,
        0.35,
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
        output["calendar_prediction"].to_numpy() * (1.0 + residual),
        0.0,
    )
    output["error"] = output["predicted_load"] - output["actual_load"]
    output["absolute_error"] = output["error"].abs()
    return output


def metric(summary: pd.DataFrame, model: str, window: str) -> float:
    return float(
        summary.loc[
            summary["model"].eq(model) & summary["window"].eq(window),
            "wmape_pct",
        ].iloc[0]
    )


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    curves = read_portfolio_curves(args.portfolio_input)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        market_path = temp_root / "combined_market.csv.gz"
        combine_market_inputs(args.market_input, market_path)
        market_matrix = read_market_matrix(market_path)
        strict_frame, groups = prepare_feature_frame(
            curves,
            component="aggregate_strict_weather_model_sweep",
            market_path=market_path,
            market_matrix=market_matrix,
            weather_input=args.weather_input,
            temp_root=temp_root,
            visibility_policy="strict_submission_asof",
            strict_predictor_weather_input=None,
            strict_market_fallback=args.strict_market_fallback,
        )
    frame = attach_weather(strict_frame, args.weather_input)
    strict_sets = feature_sets(groups, visibility_policy="strict_submission_asof")
    feature_variants = {
        "core_weather": (*strict_sets["core"], *WEATHER_FEATURES),
        "solar_weather": (*strict_sets["solar"], *WEATHER_FEATURES),
    }
    forbidden = {
        variant: [column for column in features if column.startswith("target_da_")]
        for variant, features in feature_variants.items()
    }
    if any(forbidden.values()):
        raise ValueError(f"Target-day DA leaked into sweep: {forbidden}")

    factories = model_factories()
    validation_predictions: dict[str, pd.DataFrame] = {}
    validation_rows = []
    candidate_features: dict[str, tuple[str, ...]] = {}
    for family, factory in factories.items():
        for variant, features in feature_variants.items():
            model_name = f"{family}__{variant}"
            candidate_features[model_name] = features
            prediction = fit_candidate(
                frame,
                model_name=model_name,
                features=features,
                estimator_factory=factory,
                train_end=args.validation_train_end,
                predict_start=args.validation_start,
            )
            prediction = prediction.loc[
                prediction["target_date"] <= args.validation_end
            ].copy()
            validation_predictions[model_name] = prediction
            row = score_window(
                prediction,
                window="internal_validation",
                start=args.validation_start,
                end=args.validation_end,
            )
            if row:
                validation_rows.append(row)
    validation = pd.DataFrame(validation_rows).sort_values(
        ["wmape_pct", "daily_wmape_p90_pct", "model"]
    )
    selected = str(validation.iloc[0]["model"])

    final_predictions: dict[str, pd.DataFrame] = {}
    summary_rows = []
    for family, factory in factories.items():
        for variant, features in feature_variants.items():
            model_name = f"{family}__{variant}"
            prediction = fit_candidate(
                frame,
                model_name=model_name,
                features=features,
                estimator_factory=factory,
                train_end=args.final_train_end,
                predict_start=args.holdout_start,
            )
            final_predictions[model_name] = prediction
            for window, start, end in (
                ("legacy_holdout", args.holdout_start, args.legacy_holdout_end),
                ("new_diagnostic", args.new_diagnostic_start, args.new_diagnostic_end),
            ):
                row = score_window(prediction, window=window, start=start, end=end)
                if row:
                    summary_rows.append(row)
    predictions = pd.concat(final_predictions.values(), ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["window", "wmape_pct", "model"]
    )
    daily = daily_metrics(predictions)

    selected_validation = float(validation.iloc[0]["wmape_pct"])
    incumbent_validation = float(
        validation.loc[validation["model"].eq(INCUMBENT), "wmape_pct"].iloc[0]
    )
    passes_validation = selected != INCUMBENT and selected_validation < incumbent_validation
    passes_holdout = metric(summary, selected, "legacy_holdout") < metric(
        summary, INCUMBENT, "legacy_holdout"
    )
    passes_new = metric(summary, selected, "new_diagnostic") <= metric(
        summary, INCUMBENT, "new_diagnostic"
    )
    promotion = bool(passes_validation and passes_holdout and passes_new)

    predictions.to_csv(
        args.output_dir / "predictions_96.csv.gz", index=False, compression="gzip"
    )
    daily.to_csv(
        args.output_dir / "daily_metrics.csv", index=False, encoding="utf-8-sig"
    )
    validation.to_csv(
        args.output_dir / "validation_summary.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        args.output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "version": VERSION,
        "status": "PROMOTION_CANDIDATE" if promotion else "NO_PROMOTION",
        "market_policy": STRICT_MARKET_POLICY,
        "strict_market_fallback": (
            "LATEST_CAUSALLY_VISIBLE_DATE_WITH_AGE_FEATURE"
            if args.strict_market_fallback
            else "ERROR_ON_MISSING"
        ),
        "weather_policy": WEATHER_POLICY,
        "target_day_da_feature_check": "PASS_NO_TARGET_DAY_DA",
        "selection_policy": "INTERNAL_VALIDATION_ONLY",
        "candidate_count": len(candidate_features),
        "selected_on_internal_validation": selected,
        "incumbent": INCUMBENT,
        "passes_validation_gate": passes_validation,
        "passes_legacy_holdout_gate": passes_holdout,
        "passes_new_diagnostic_gate": passes_new,
        "promotion_recommended": promotion,
        "portfolio_input": str(args.portfolio_input.resolve()),
        "portfolio_sha256": sha256_file(args.portfolio_input),
        "market_inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.market_input
        ],
        "weather_input": str(args.weather_input.resolve()),
        "weather_sha256": sha256_file(args.weather_input),
        "candidate_features": {
            model: list(features) for model, features in candidate_features.items()
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 江苏用户负荷：严格运营气象离线模型扫描",
        "",
        f"- 共比较 {len(candidate_features)} 个模型×特征候选。",
        f"- 验证窗预选：`{selected}`，WMAPE {selected_validation:.4f}%。",
        f"- 当前气象GBDT基准验证WMAPE：{incumbent_validation:.4f}%。",
        (
            f"- 41天留出：预选模型 {metric(summary, selected, 'legacy_holdout'):.4f}%，"
            f"基准 {metric(summary, INCUMBENT, 'legacy_holdout'):.4f}%。"
        ),
        (
            f"- 最近3天：预选模型 {metric(summary, selected, 'new_diagnostic'):.4f}%，"
            f"基准 {metric(summary, INCUMBENT, 'new_diagnostic'):.4f}%。"
        ),
        (
            "- 三重门禁："
            f"验证{'通过' if passes_validation else '未通过'}，"
            f"历史留出{'通过' if passes_holdout else '未通过'}，"
            f"新增日期{'通过' if passes_new else '未通过'}；"
            f"结论：{'建议晋级' if promotion else '不晋级'}。"
        ),
    ]
    (args.output_dir / "run_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(validation[["model", "wmape_pct"]].to_string(index=False))
    print()
    print(summary[["model", "window", "wmape_pct"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-input", type=Path, required=True)
    parser.add_argument("--market-input", type=Path, action="append", required=True)
    parser.add_argument("--weather-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--strict-market-fallback",
        action="store_true",
        help=(
            "For missing T-2/T-3/T-7 market points, use only the latest earlier "
            "complete date and expose its age as a feature."
        ),
    )
    parser.add_argument(
        "--validation-train-end", type=date.fromisoformat, default=date(2026, 5, 29)
    )
    parser.add_argument(
        "--validation-start", type=date.fromisoformat, default=date(2026, 6, 1)
    )
    parser.add_argument(
        "--validation-end", type=date.fromisoformat, default=date(2026, 6, 12)
    )
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
        "--new-diagnostic-end", type=date.fromisoformat, default=date(2026, 7, 28)
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
