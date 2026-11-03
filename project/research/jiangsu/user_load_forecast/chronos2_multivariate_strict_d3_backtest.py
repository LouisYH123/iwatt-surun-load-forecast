"""Strict D-3 Chronos-2 multivariate backtest for Jiangsu portfolio load.

The 96 intraday intervals are treated as 96 jointly forecast variates on a
regular daily calendar.  For target D, context ends at D-3; the model forecasts
three daily steps and step three is used for D.  Model and blend selection use
only the frozen May validation window.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import chronos
import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline
from huggingface_hub import snapshot_download

from project.research.jiangsu.user_load_forecast.chronos_bolt_strict_d3_backtest import (
    blend,
    complete_matrix,
    read_anchor,
    wmape,
)
from project.research.jiangsu.user_load_forecast.model_backtest import prepare_output_dir
from project.research.jiangsu.user_load_forecast.pv_sensitive_segment_backtest import (
    read_portfolio_curves,
)
from project.research.jiangsu.user_load_forecast.pv_weather_challenger_audit import (
    score_window,
)
from project.research.jiangsu.user_load_forecast.strict_operational_weather_backtest import (
    daily_metrics,
    sha256_file,
)


VERSION = "surun_chronos2_multivariate_strict_d3_v1"
MODEL_ID = "amazon/chronos-2"
CONTEXT_DAYS = (None, 42, 28)
BLEND_ALPHAS = (0.10, 0.25, 0.50, 0.75)


def alias(context_days: int | None) -> str:
    context = "all" if context_days is None else str(context_days)
    return f"chronos2__multivariate96_daily_ctx{context}"


def forecast_target_contexts(
    pipeline: Chronos2Pipeline,
    curves: dict[date, np.ndarray],
    *,
    target_date: date,
) -> dict[str, tuple[np.ndarray, date, date, int]]:
    cutoff = target_date - timedelta(days=3)
    inputs: list[np.ndarray] = []
    metadata: list[tuple[str, date, int]] = []
    for context_days in CONTEXT_DAYS:
        dates, matrix = complete_matrix(
            curves,
            cutoff=cutoff,
            context_days=context_days,
        )
        # Chronos-2 input shape for one task is (n_variates, history_length).
        inputs.append(matrix.T.copy())
        metadata.append((alias(context_days), dates[0], int(np.isnan(matrix).sum())))

    quantiles, _median = pipeline.predict_quantiles(
        inputs,
        prediction_length=3,
        quantile_levels=[0.5],
        batch_size=288,
    )
    outputs: dict[str, tuple[np.ndarray, date, date, int]] = {}
    for prediction, (name, context_start, missing_points) in zip(
        quantiles, metadata, strict=True
    ):
        values = prediction[:, 2, 0].detach().cpu().numpy().astype(float)
        outputs[name] = (
            np.maximum(values, 0.0),
            context_start,
            cutoff,
            missing_points,
        )
    return outputs


def run_model(
    curves: dict[date, np.ndarray], target_dates: list[date]
) -> dict[str, pd.DataFrame]:
    local_snapshot = snapshot_download(MODEL_ID, local_files_only=True)
    pipeline = Chronos2Pipeline.from_pretrained(
        local_snapshot,
        device_map="cpu",
        dtype=torch.float32,
        local_files_only=True,
    )
    rows: dict[str, list[dict[str, object]]] = {
        alias(context): [] for context in CONTEXT_DAYS
    }
    for target_date in target_dates:
        actual = curves[target_date]
        forecasts = forecast_target_contexts(
            pipeline,
            curves,
            target_date=target_date,
        )
        for name, (prediction, context_start, cutoff, missing_points) in forecasts.items():
            for interval in range(1, 97):
                error = float(prediction[interval - 1] - actual[interval - 1])
                rows[name].append(
                    {
                        "model": name,
                        "target_date": target_date,
                        "trade_interval": interval,
                        "actual_load": float(actual[interval - 1]),
                        "predicted_load": float(prediction[interval - 1]),
                        "error": error,
                        "absolute_error": abs(error),
                        "load_label_cutoff": cutoff,
                        "context_start": context_start,
                        "context_end": cutoff,
                        "context_missing_points": missing_points,
                    }
                )
    del pipeline
    return {name: pd.DataFrame(values) for name, values in rows.items()}


def read_bolt_candidate(
    path: Path, model: str, start: date, end: date
) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame["target_date"] = pd.to_datetime(frame["target_date"]).dt.date
    output = frame.loc[
        frame["model"].eq(model) & frame["target_date"].between(start, end)
    ].copy()
    if output.empty:
        raise ValueError(f"Bolt candidate absent: {model} in {path}")
    return output


def average_foundation_models(
    chronos2_prediction: pd.DataFrame,
    bolt_prediction: pd.DataFrame,
    *,
    model_name: str,
) -> pd.DataFrame:
    # Reuse blend alignment and arithmetic with equal weights.
    return blend(
        chronos2_prediction,
        bolt_prediction,
        alpha=0.5,
        model_name=model_name,
    )


def candidate_pool(
    chronos2_predictions: dict[str, pd.DataFrame],
    *,
    bolt_predictions_path: Path,
    anchor: pd.DataFrame,
    start: date,
    end: date,
    anchor_model: str,
) -> dict[str, pd.DataFrame]:
    candidates: dict[str, pd.DataFrame] = {}
    for context_days in CONTEXT_DAYS:
        name = alias(context_days)
        prediction = chronos2_predictions[name]
        prediction = prediction.loc[
            prediction["target_date"].between(start, end)
        ].copy()
        candidates[name] = prediction

        context = "all" if context_days is None else str(context_days)
        bolt_name = f"chronos_bolt_small__daily96_ctx{context}"
        bolt = read_bolt_candidate(
            bolt_predictions_path,
            bolt_name,
            start,
            end,
        )
        ensemble_name = f"foundation_mean__{name}__{bolt_name}"
        candidates[ensemble_name] = average_foundation_models(
            prediction,
            bolt,
            model_name=ensemble_name,
        )

    bases = list(candidates.items())
    for base_name, base_prediction in bases:
        for alpha in BLEND_ALPHAS:
            blend_name = f"blend_a{alpha:g}__{base_name}__{anchor_model}"
            candidates[blend_name] = blend(
                base_prediction,
                anchor,
                alpha=alpha,
                model_name=blend_name,
            )

    anchor_output = anchor.copy()
    anchor_output["model"] = f"anchor__{anchor_model}"
    candidates[f"anchor__{anchor_model}"] = anchor_output
    return candidates


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    curves = read_portfolio_curves(args.portfolio_input)
    available = set(curves)
    validation_dates = [
        value.date()
        for value in pd.date_range(args.validation_start, args.validation_end, freq="D")
        if value.date() in available
    ]
    evaluation_dates = [
        value
        for value in sorted(available)
        if args.holdout_start <= value <= args.new_diagnostic_end
    ]
    target_dates = sorted(set(validation_dates + evaluation_dates))
    chronos2_predictions = run_model(curves, target_dates)

    validation_anchor = read_anchor(
        args.anchor_validation_input,
        args.anchor_model,
        args.validation_start,
        args.validation_end,
    )
    evaluation_anchor = read_anchor(
        args.anchor_evaluation_input,
        args.anchor_model,
        args.holdout_start,
        args.new_diagnostic_end,
    )
    validation_candidates = candidate_pool(
        chronos2_predictions,
        bolt_predictions_path=args.bolt_validation_predictions,
        anchor=validation_anchor,
        start=args.validation_start,
        end=args.validation_end,
        anchor_model=args.anchor_model,
    )
    validation_summary = pd.DataFrame(
        [
            {"model": model, "wmape_pct": wmape(frame)}
            for model, frame in validation_candidates.items()
        ]
    ).sort_values(["wmape_pct", "model"])
    selected = str(validation_summary.iloc[0]["model"])

    evaluation_candidates = candidate_pool(
        chronos2_predictions,
        bolt_predictions_path=args.bolt_evaluation_predictions,
        anchor=evaluation_anchor,
        start=args.holdout_start,
        end=args.new_diagnostic_end,
        anchor_model=args.anchor_model,
    )
    predictions = pd.concat(evaluation_candidates.values(), ignore_index=True)
    summary_rows = []
    for model, model_frame in predictions.groupby("model", sort=True):
        for window, start, end in (
            ("legacy_holdout", args.holdout_start, args.legacy_holdout_end),
            ("new_diagnostic", args.new_diagnostic_start, args.new_diagnostic_end),
        ):
            row = score_window(model_frame, window=window, start=start, end=end)
            if row:
                summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["window", "wmape_pct", "model"]
    )

    predictions.to_csv(
        args.output_dir / "predictions_96.csv.gz", index=False, compression="gzip"
    )
    pd.concat(validation_candidates.values(), ignore_index=True).to_csv(
        args.output_dir / "validation_predictions_96.csv.gz",
        index=False,
        compression="gzip",
    )
    validation_summary.to_csv(
        args.output_dir / "validation_summary.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        args.output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig"
    )
    daily_metrics(predictions).to_csv(
        args.output_dir / "daily_metrics.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "version": VERSION,
        "built_at": datetime.now().astimezone().isoformat(),
        "status": "OFFLINE_RESEARCH_ONLY",
        "network_mode": "offline",
        "forecasting_policy": "MULTIVARIATE_96_DAILY_FORECAST_3_STEPS_TAKE_STEP_3",
        "load_label_policy": "STRICT_TARGET_MINUS_3",
        "selection_policy": "MAY_VALIDATION_ONLY",
        "selected": selected,
        "selected_validation_wmape_pct": float(validation_summary.iloc[0]["wmape_pct"]),
        "model_id": MODEL_ID,
        "context_days": ["all" if value is None else value for value in CONTEXT_DAYS],
        "blend_alphas": list(BLEND_ALPHAS),
        "chronos_version": getattr(chronos, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "device": "cpu",
        "portfolio_input": str(args.portfolio_input.resolve()),
        "portfolio_sha256": sha256_file(args.portfolio_input),
        "anchor_validation_input": str(args.anchor_validation_input.resolve()),
        "anchor_validation_sha256": sha256_file(args.anchor_validation_input),
        "anchor_evaluation_input": str(args.anchor_evaluation_input.resolve()),
        "anchor_evaluation_sha256": sha256_file(args.anchor_evaluation_input),
        "bolt_validation_predictions": str(args.bolt_validation_predictions.resolve()),
        "bolt_validation_sha256": sha256_file(args.bolt_validation_predictions),
        "bolt_evaluation_predictions": str(args.bolt_evaluation_predictions.resolve()),
        "bolt_evaluation_sha256": sha256_file(args.bolt_evaluation_predictions),
        "anchor_model": args.anchor_model,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(validation_summary.head(20).to_string(index=False))
    selected_summary = summary.loc[
        summary["model"].isin([selected, f"anchor__{args.anchor_model}"])
    ]
    print(selected_summary[["model", "window", "wmape_pct"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-input", type=Path, required=True)
    parser.add_argument("--anchor-validation-input", type=Path, required=True)
    parser.add_argument("--anchor-evaluation-input", type=Path, required=True)
    parser.add_argument("--bolt-validation-predictions", type=Path, required=True)
    parser.add_argument("--bolt-evaluation-predictions", type=Path, required=True)
    parser.add_argument("--anchor-model", default="anchor_core_weather")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-start", type=date.fromisoformat, default=date(2026, 5, 1))
    parser.add_argument("--validation-end", type=date.fromisoformat, default=date(2026, 5, 31))
    parser.add_argument("--holdout-start", type=date.fromisoformat, default=date(2026, 6, 15))
    parser.add_argument("--legacy-holdout-end", type=date.fromisoformat, default=date(2026, 7, 25))
    parser.add_argument("--new-diagnostic-start", type=date.fromisoformat, default=date(2026, 7, 26))
    parser.add_argument("--new-diagnostic-end", type=date.fromisoformat, default=date(2026, 8, 5))
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
