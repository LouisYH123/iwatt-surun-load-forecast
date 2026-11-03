"""Fit a validation-only nonnegative ensemble of the offline sweep leaders."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.fusion import nonnegative_weights
from project.research.jiangsu.user_load_forecast.market_level_shape_backtest import (
    read_market_matrix,
)
from project.research.jiangsu.user_load_forecast.model_backtest import prepare_output_dir
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
    INCUMBENT,
    fit_candidate,
    model_factories,
)


VERSION = "surun_strict_operational_weather_validation_ensemble_v1"
MODEL_ID = "validation_nonnegative_top5_ensemble"


def fused_prediction(
    predictions: dict[str, pd.DataFrame],
    models: list[str],
    weights: np.ndarray,
    model_id: str,
) -> pd.DataFrame:
    keys = ["target_date", "trade_interval"]
    base = predictions[models[0]].sort_values(keys).reset_index(drop=True).copy()
    matrix = []
    for model in models:
        candidate = predictions[model].sort_values(keys).reset_index(drop=True)
        if not candidate[keys].equals(base[keys]):
            raise ValueError(f"Prediction keys do not align for {model}")
        matrix.append(candidate["predicted_load"].to_numpy(dtype=float))
    base["model"] = model_id
    base["predicted_load"] = np.column_stack(matrix) @ weights
    base["error"] = base["predicted_load"] - base["actual_load"]
    base["absolute_error"] = base["error"].abs()
    return base


def metric(summary: pd.DataFrame, model: str, window: str) -> float:
    return float(
        summary.loc[
            summary["model"].eq(model) & summary["window"].eq(window),
            "wmape_pct",
        ].iloc[0]
    )


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    validation_summary = pd.read_csv(args.sweep_dir / "validation_summary.csv")
    leaders = validation_summary.head(args.top_n)["model"].astype(str).tolist()
    if INCUMBENT not in leaders:
        leaders.append(INCUMBENT)

    curves = read_portfolio_curves(args.portfolio_input)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        market_path = temp_root / "combined_market.csv.gz"
        combine_market_inputs(args.market_input, market_path)
        strict_frame, groups = prepare_feature_frame(
            curves,
            component="aggregate_strict_weather_validation_ensemble",
            market_path=market_path,
            market_matrix=read_market_matrix(market_path),
            weather_input=args.weather_input,
            temp_root=temp_root,
            visibility_policy="strict_submission_asof",
            strict_predictor_weather_input=None,
            strict_market_fallback=args.strict_market_fallback,
        )
    frame = attach_weather(strict_frame, args.weather_input)
    strict_sets = feature_sets(groups, visibility_policy="strict_submission_asof")
    variants = {
        "core_weather": (*strict_sets["core"], *WEATHER_FEATURES),
        "solar_weather": (*strict_sets["solar"], *WEATHER_FEATURES),
    }
    factories = model_factories()
    validation_predictions = {}
    for model in leaders:
        family, variant = model.split("__", 1)
        prediction = fit_candidate(
            frame,
            model_name=model,
            features=variants[variant],
            estimator_factory=factories[family],
            train_end=args.validation_train_end,
            predict_start=args.validation_start,
        )
        validation_predictions[model] = prediction.loc[
            prediction["target_date"] <= args.validation_end
        ].copy()

    ordered = [
        validation_predictions[model]
        .sort_values(["target_date", "trade_interval"])
        .reset_index(drop=True)
        for model in leaders
    ]
    prediction_matrix = np.column_stack(
        [item["predicted_load"].to_numpy(dtype=float) for item in ordered]
    )
    actual = ordered[0]["actual_load"].to_numpy(dtype=float)
    weights, optimizer_success = nonnegative_weights(prediction_matrix, actual)
    validation_fusion = fused_prediction(
        validation_predictions, leaders, weights, MODEL_ID
    )
    validation_row = score_window(
        validation_fusion,
        window="internal_validation",
        start=args.validation_start,
        end=args.validation_end,
    )

    sweep_predictions = pd.read_csv(args.sweep_dir / "predictions_96.csv.gz")
    sweep_predictions["target_date"] = pd.to_datetime(
        sweep_predictions["target_date"]
    ).dt.date
    final_predictions = {
        model: sweep_predictions.loc[sweep_predictions["model"].eq(model)].copy()
        for model in leaders
    }
    final_fusion = fused_prediction(final_predictions, leaders, weights, MODEL_ID)
    incumbent = sweep_predictions.loc[
        sweep_predictions["model"].eq(INCUMBENT)
    ].copy()
    summary_rows = []
    for prediction in (final_fusion, incumbent):
        for window, start, end in (
            ("legacy_holdout", args.holdout_start, args.legacy_holdout_end),
            ("new_diagnostic", args.new_diagnostic_start, args.new_diagnostic_end),
        ):
            row = score_window(prediction, window=window, start=start, end=end)
            if row:
                summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    passes_holdout = metric(summary, MODEL_ID, "legacy_holdout") < metric(
        summary, INCUMBENT, "legacy_holdout"
    )
    passes_new = metric(summary, MODEL_ID, "new_diagnostic") <= metric(
        summary, INCUMBENT, "new_diagnostic"
    )
    promotion = bool(passes_holdout and passes_new)

    final_fusion.to_csv(
        args.output_dir / "predictions_96.csv.gz", index=False, compression="gzip"
    )
    daily_metrics(final_fusion).to_csv(
        args.output_dir / "daily_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([validation_row]).to_csv(
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
        "weight_training_window": {
            "start": args.validation_start.isoformat(),
            "end": args.validation_end.isoformat(),
        },
        "leaders_selected_from_validation_only": leaders,
        "weights": {model: float(weight) for model, weight in zip(leaders, weights)},
        "optimizer_success": optimizer_success,
        "incumbent": INCUMBENT,
        "passes_legacy_holdout_gate": passes_holdout,
        "passes_new_diagnostic_gate": passes_new,
        "promotion_recommended": promotion,
        "sweep_dir": str(args.sweep_dir.resolve()),
        "portfolio_sha256": sha256_file(args.portfolio_input),
        "weather_sha256": sha256_file(args.weather_input),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 江苏用户负荷：严格运营气象验证窗融合",
        "",
        f"- 验证窗融合WMAPE：{float(validation_row['wmape_pct']):.4f}%。",
        (
            f"- 41天留出：融合 {metric(summary, MODEL_ID, 'legacy_holdout'):.4f}%，"
            f"气象GBDT {metric(summary, INCUMBENT, 'legacy_holdout'):.4f}%。"
        ),
        (
            f"- 最近3天：融合 {metric(summary, MODEL_ID, 'new_diagnostic'):.4f}%，"
            f"气象GBDT {metric(summary, INCUMBENT, 'new_diagnostic'):.4f}%。"
        ),
        f"- 结论：{'建议晋级' if promotion else '不晋级'}。",
    ]
    (args.output_dir / "run_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(summary[["model", "window", "wmape_pct"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-input", type=Path, required=True)
    parser.add_argument("--market-input", type=Path, action="append", required=True)
    parser.add_argument("--weather-input", type=Path, required=True)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--strict-market-fallback",
        action="store_true",
        help=(
            "For missing T-2/T-3/T-7 market points, use only the latest earlier "
            "complete date and expose its age as a feature."
        ),
    )
    parser.add_argument("--top-n", type=int, default=5)
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
