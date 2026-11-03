"""Strict D-3 zero-shot Chronos-Bolt backtest for Jiangsu portfolio load.

Each of the 96 intraday intervals is represented as a regular daily series.
For target D, context ends at D-3, Chronos forecasts three daily steps, and the
third step becomes the target interval prediction. Candidate and blend
selection use only the frozen May validation window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import chronos
import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline
from huggingface_hub import snapshot_download

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


VERSION = "surun_chronos_bolt_strict_d3_v1"
MODEL_IDS = (
    "amazon/chronos-bolt-tiny",
    "amazon/chronos-bolt-mini",
    "amazon/chronos-bolt-small",
)
CONTEXT_DAYS = (None, 42, 28)
BLEND_ALPHAS = (0.25, 0.50, 0.75)


def alias(model_id: str, context_days: int | None) -> str:
    size = model_id.rsplit("-", 1)[-1]
    context = "all" if context_days is None else str(context_days)
    return f"chronos_bolt_{size}__daily96_ctx{context}"


def complete_matrix(
    curves: dict[date, np.ndarray],
    *,
    cutoff: date,
    context_days: int | None,
) -> tuple[list[date], np.ndarray]:
    start = min(curves)
    if context_days is not None:
        start = max(start, cutoff - timedelta(days=context_days - 1))
    dates = [value.date() for value in pd.date_range(start, cutoff, freq="D")]
    matrix = np.full((len(dates), 96), np.nan, dtype=np.float32)
    for row, trade_date in enumerate(dates):
        if trade_date in curves:
            matrix[row] = curves[trade_date].astype(np.float32)
    return dates, matrix


def forecast_target(
    pipeline: object,
    curves: dict[date, np.ndarray],
    *,
    target_date: date,
    context_days: int | None,
) -> tuple[np.ndarray, date, date, int]:
    cutoff = target_date - timedelta(days=3)
    dates, matrix = complete_matrix(
        curves, cutoff=cutoff, context_days=context_days
    )
    contexts = [
        torch.from_numpy(matrix[:, interval].copy()) for interval in range(96)
    ]
    quantiles, _mean = pipeline.predict_quantiles(
        contexts,
        prediction_length=3,
        quantile_levels=[0.5],
    )
    prediction = quantiles[:, 2, 0].detach().cpu().numpy().astype(float)
    prediction = np.maximum(prediction, 0.0)
    return prediction, dates[0], cutoff, int(np.isnan(matrix).sum())


def run_model(
    model_id: str,
    curves: dict[date, np.ndarray],
    target_dates: list[date],
) -> dict[str, pd.DataFrame]:
    local_snapshot = snapshot_download(model_id, local_files_only=True)
    pipeline = BaseChronosPipeline.from_pretrained(
        local_snapshot,
        device_map="cpu",
        dtype=torch.float32,
        local_files_only=True,
    )
    outputs: dict[str, list[dict[str, object]]] = {
        alias(model_id, context): [] for context in CONTEXT_DAYS
    }
    for target_date in target_dates:
        actual = curves[target_date]
        for context_days in CONTEXT_DAYS:
            model = alias(model_id, context_days)
            prediction, context_start, cutoff, missing_points = forecast_target(
                pipeline,
                curves,
                target_date=target_date,
                context_days=context_days,
            )
            for interval in range(1, 97):
                error = float(prediction[interval - 1] - actual[interval - 1])
                outputs[model].append(
                    {
                        "model": model,
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
    return {name: pd.DataFrame(rows) for name, rows in outputs.items()}


def read_anchor(path: Path, model: str, start: date, end: date) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["target_date"] = pd.to_datetime(frame["target_date"]).dt.date
    selected = frame.loc[
        frame["model"].eq(model) & frame["target_date"].between(start, end)
    ].copy()
    if selected.empty:
        raise ValueError(f"Anchor model absent: {model} in {path}")
    return selected


def blend(
    chronos_prediction: pd.DataFrame,
    anchor: pd.DataFrame,
    *,
    alpha: float,
    model_name: str,
) -> pd.DataFrame:
    keys = ["target_date", "trade_interval"]
    left = chronos_prediction.sort_values(keys).reset_index(drop=True)
    right = anchor.sort_values(keys).reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise ValueError(f"Chronos/anchor keys do not align for {model_name}")
    output = left.copy()
    output["model"] = model_name
    output["predicted_load"] = (
        alpha * left["predicted_load"].to_numpy(dtype=float)
        + (1.0 - alpha) * right["predicted_load"].to_numpy(dtype=float)
    )
    output["error"] = output["predicted_load"] - output["actual_load"]
    output["absolute_error"] = output["error"].abs()
    return output


def wmape(frame: pd.DataFrame) -> float:
    return float(
        100.0 * frame["absolute_error"].sum() / frame["actual_load"].abs().sum()
    )


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

    chronos_predictions: dict[str, pd.DataFrame] = {}
    for model_id in args.model_id:
        chronos_predictions.update(run_model(model_id, curves, target_dates))

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
    validation_rows = []
    validation_candidates: dict[str, pd.DataFrame] = {}
    for name, prediction in chronos_predictions.items():
        validation = prediction.loc[
            prediction["target_date"].isin(validation_dates)
        ].copy()
        validation_candidates[name] = validation
        validation_rows.append(
            {"model": name, "kind": "chronos", "wmape_pct": wmape(validation)}
        )
        for alpha in BLEND_ALPHAS:
            blend_name = f"blend_a{alpha:g}__{name}__{args.anchor_model}"
            candidate = blend(
                validation,
                validation_anchor,
                alpha=alpha,
                model_name=blend_name,
            )
            validation_candidates[blend_name] = candidate
            validation_rows.append(
                {"model": blend_name, "kind": "blend", "wmape_pct": wmape(candidate)}
            )
    anchor_validation = validation_anchor.copy()
    anchor_validation["model"] = f"anchor__{args.anchor_model}"
    validation_candidates[f"anchor__{args.anchor_model}"] = anchor_validation
    validation_rows.append(
        {
            "model": f"anchor__{args.anchor_model}",
            "kind": "anchor",
            "wmape_pct": wmape(anchor_validation),
        }
    )
    validation_summary = pd.DataFrame(validation_rows).sort_values(
        ["wmape_pct", "model"]
    )
    selected = str(validation_summary.iloc[0]["model"])

    final_outputs: list[pd.DataFrame] = []
    for name, prediction in chronos_predictions.items():
        evaluation = prediction.loc[
            prediction["target_date"].isin(evaluation_dates)
        ].copy()
        final_outputs.append(evaluation)
        for alpha in BLEND_ALPHAS:
            blend_name = f"blend_a{alpha:g}__{name}__{args.anchor_model}"
            final_outputs.append(
                blend(
                    evaluation,
                    evaluation_anchor,
                    alpha=alpha,
                    model_name=blend_name,
                )
            )
    anchor_evaluation = evaluation_anchor.copy()
    anchor_evaluation["model"] = f"anchor__{args.anchor_model}"
    final_outputs.append(anchor_evaluation)
    predictions = pd.concat(final_outputs, ignore_index=True)

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
    daily = daily_metrics(predictions)

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
    daily.to_csv(
        args.output_dir / "daily_metrics.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "version": VERSION,
        "built_at": datetime.now().astimezone().isoformat(),
        "status": "OFFLINE_RESEARCH_ONLY",
        "forecasting_policy": "96_DAILY_INTERVAL_SERIES_FORECAST_3_STEPS_TAKE_STEP_3",
        "load_label_policy": "STRICT_TARGET_MINUS_3",
        "selection_policy": "MAY_VALIDATION_ONLY",
        "selected": selected,
        "selected_validation_wmape_pct": float(validation_summary.iloc[0]["wmape_pct"]),
        "model_ids": args.model_id,
        "context_days": ["all" if value is None else value for value in CONTEXT_DAYS],
        "chronos_version": getattr(chronos, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "device": "cpu",
        "portfolio_input": str(args.portfolio_input.resolve()),
        "portfolio_sha256": sha256_file(args.portfolio_input),
        "anchor_validation_input": str(args.anchor_validation_input.resolve()),
        "anchor_validation_sha256": sha256_file(args.anchor_validation_input),
        "anchor_evaluation_input": str(args.anchor_evaluation_input.resolve()),
        "anchor_evaluation_sha256": sha256_file(args.anchor_evaluation_input),
        "anchor_model": args.anchor_model,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(validation_summary.head(20).to_string(index=False))
    print(summary[["model", "window", "wmape_pct"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-input", type=Path, required=True)
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--anchor-validation-input", type=Path, required=True)
    parser.add_argument("--anchor-evaluation-input", type=Path, required=True)
    parser.add_argument("--anchor-model", default="anchor_core_weather")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-start", type=date.fromisoformat, default=date(2026, 5, 1))
    parser.add_argument("--validation-end", type=date.fromisoformat, default=date(2026, 5, 31))
    parser.add_argument("--holdout-start", type=date.fromisoformat, default=date(2026, 6, 15))
    parser.add_argument("--legacy-holdout-end", type=date.fromisoformat, default=date(2026, 7, 25))
    parser.add_argument("--new-diagnostic-start", type=date.fromisoformat, default=date(2026, 7, 26))
    parser.add_argument("--new-diagnostic-end", type=date.fromisoformat, default=date(2026, 8, 5))
    args = parser.parse_args()
    if not args.model_id:
        args.model_id = list(MODEL_IDS)
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
