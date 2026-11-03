"""Delay stress test for fixed Jiangsu Chronos foundation-model candidates.

This experiment does no model selection.  It freezes the previously selected
Chronos-Bolt and Chronos-2 contexts, then simulates latest mature load labels at
D-3, D-4, D-5, and D-7.  The model forecasts `label_age_days` daily steps and
uses the final step for target D.  A causal four-same-weekday mean is included
as a deterministic fallback reference.
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
from chronos import BaseChronosPipeline, Chronos2Pipeline
from huggingface_hub import snapshot_download

from project.research.jiangsu.user_load_forecast.chronos_bolt_strict_d3_backtest import (
    complete_matrix,
)
from project.research.jiangsu.user_load_forecast.model_backtest import prepare_output_dir
from project.research.jiangsu.user_load_forecast.pv_sensitive_segment_backtest import (
    read_portfolio_curves,
)
from project.research.jiangsu.user_load_forecast.strict_operational_weather_backtest import (
    sha256_file,
)


VERSION = "surun_chronos_foundation_delay_stress_v1"
LABEL_AGES = (3, 4, 5, 7)
BOLT_MODEL_ID = "amazon/chronos-bolt-small"
CHRONOS2_MODEL_ID = "amazon/chronos-2"
BOLT_CONTEXT_DAYS = None
CHRONOS2_CONTEXT_DAYS = 42


def local_model_snapshot(model_id: str) -> str:
    """Resolve an already-cached model without any network request."""
    return snapshot_download(model_id, local_files_only=True)


def same_weekday_mean(
    curves: dict[date, np.ndarray], *, target_date: date, cutoff: date
) -> np.ndarray:
    eligible = [
        value
        for value in sorted(curves)
        if value <= cutoff and value.weekday() == target_date.weekday()
    ]
    if not eligible:
        raise ValueError(f"No causal same-weekday history for {target_date}")
    return np.mean([curves[value] for value in eligible[-4:]], axis=0)


def build_contexts(
    curves: dict[date, np.ndarray],
    target_dates: list[date],
    *,
    label_age_days: int,
    context_days: int | None,
) -> tuple[list[np.ndarray], list[tuple[date, date, date, int]]]:
    inputs = []
    metadata = []
    for target_date in target_dates:
        cutoff = target_date - timedelta(days=label_age_days)
        dates, matrix = complete_matrix(
            curves,
            cutoff=cutoff,
            context_days=context_days,
        )
        inputs.append(matrix)
        metadata.append((target_date, dates[0], cutoff, int(np.isnan(matrix).sum())))
    return inputs, metadata


def prediction_rows(
    *,
    model: str,
    label_age_days: int,
    target_date: date,
    actual: np.ndarray,
    prediction: np.ndarray,
    context_start: date,
    cutoff: date,
    missing_points: int,
    day_type: str,
) -> list[dict[str, object]]:
    rows = []
    for interval in range(1, 97):
        error = float(prediction[interval - 1] - actual[interval - 1])
        rows.append(
            {
                "model": model,
                "label_age_days": label_age_days,
                "target_date": target_date,
                "trade_interval": interval,
                "day_type": day_type,
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
    return rows


def score_groups(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    frame["period"] = pd.cut(
        frame["target_date"],
        bins=pd.to_datetime(
            [
                "2026-06-14",
                "2026-06-30",
                "2026-07-15",
                "2026-07-25",
                "2026-08-05",
            ]
        ),
        labels=["jun15_jun30", "jul01_jul15", "jul16_jul25", "jul26_aug05"],
    )
    rows = []
    group_specs = [("all", frame)]
    group_specs.extend((str(value), part) for value, part in frame.groupby("period", observed=True))
    group_specs.extend((f"daytype_{value}", part) for value, part in frame.groupby("day_type"))
    for group, scoped in group_specs:
        for (model, age), part in scoped.groupby(["model", "label_age_days"], sort=True):
            daily = part.groupby("target_date").agg(
                absolute_error=("absolute_error", "sum"),
                actual_load=("actual_load", lambda values: values.abs().sum()),
            )
            daily_wmape = 100.0 * daily["absolute_error"] / daily["actual_load"]
            rows.append(
                {
                    "group": group,
                    "model": model,
                    "label_age_days": age,
                    "target_days": part["target_date"].nunique(),
                    "wmape_pct": 100.0 * part["absolute_error"].sum() / part["actual_load"].abs().sum(),
                    "daily_wmape_p90_pct": daily_wmape.quantile(0.9),
                    "daily_wmape_max_pct": daily_wmape.max(),
                    "bias_pct": 100.0 * part["error"].sum() / part["actual_load"].abs().sum(),
                }
            )
    return pd.DataFrame(rows).sort_values(["group", "label_age_days", "wmape_pct", "model"])


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    curves = read_portfolio_curves(args.portfolio_input)
    target_dates = [
        value for value in sorted(curves) if args.target_start <= value <= args.target_end
    ]
    anchor = pd.read_csv(args.anchor_predictions, low_memory=False)
    anchor["target_date"] = pd.to_datetime(anchor["target_date"]).dt.date
    day_types = (
        anchor.loc[anchor["model"].eq(args.anchor_model)]
        .dropna(subset=["day_type"])
        .groupby("target_date")["day_type"]
        .first()
        .to_dict()
    )
    if set(target_dates) - set(day_types):
        raise ValueError("Missing day type metadata for stress targets")

    chronos2_snapshot = local_model_snapshot(CHRONOS2_MODEL_ID)
    bolt_snapshot = local_model_snapshot(BOLT_MODEL_ID)
    chronos2 = Chronos2Pipeline.from_pretrained(
        chronos2_snapshot,
        device_map="cpu",
        dtype=torch.float32,
        local_files_only=True,
    )
    bolt = BaseChronosPipeline.from_pretrained(
        bolt_snapshot,
        device_map="cpu",
        dtype=torch.float32,
        local_files_only=True,
    )
    rows: list[dict[str, object]] = []
    for label_age_days in LABEL_AGES:
        c2_matrices, c2_metadata = build_contexts(
            curves,
            target_dates,
            label_age_days=label_age_days,
            context_days=CHRONOS2_CONTEXT_DAYS,
        )
        c2_quantiles, _ = chronos2.predict_quantiles(
            [matrix.T.copy() for matrix in c2_matrices],
            prediction_length=label_age_days,
            quantile_levels=[0.5],
            batch_size=384,
        )

        bolt_matrices, bolt_metadata = build_contexts(
            curves,
            target_dates,
            label_age_days=label_age_days,
            context_days=BOLT_CONTEXT_DAYS,
        )
        bolt_inputs = [
            torch.from_numpy(matrix[:, interval].copy())
            for matrix in bolt_matrices
            for interval in range(96)
        ]
        bolt_quantiles, _ = bolt.predict_quantiles(
            bolt_inputs,
            prediction_length=label_age_days,
            quantile_levels=[0.5],
        )
        bolt_values = bolt_quantiles[:, label_age_days - 1, 0].detach().cpu().numpy()

        for index, (target_date, context_start, cutoff, missing_points) in enumerate(c2_metadata):
            actual = curves[target_date]
            c2_prediction = np.maximum(
                c2_quantiles[index][:, label_age_days - 1, 0].detach().cpu().numpy(),
                0.0,
            )
            rows.extend(
                prediction_rows(
                    model="chronos2_multivariate_ctx42",
                    label_age_days=label_age_days,
                    target_date=target_date,
                    actual=actual,
                    prediction=c2_prediction,
                    context_start=context_start,
                    cutoff=cutoff,
                    missing_points=missing_points,
                    day_type=day_types[target_date],
                )
            )

            btarget, bstart, bcutoff, bmissing = bolt_metadata[index]
            assert btarget == target_date and bcutoff == cutoff
            start = index * 96
            bolt_prediction = np.maximum(bolt_values[start : start + 96], 0.0)
            rows.extend(
                prediction_rows(
                    model="chronos_bolt_small_ctxall",
                    label_age_days=label_age_days,
                    target_date=target_date,
                    actual=actual,
                    prediction=bolt_prediction,
                    context_start=bstart,
                    cutoff=bcutoff,
                    missing_points=bmissing,
                    day_type=day_types[target_date],
                )
            )

            baseline = same_weekday_mean(
                curves,
                target_date=target_date,
                cutoff=cutoff,
            )
            rows.extend(
                prediction_rows(
                    model="same_weekday_mean4",
                    label_age_days=label_age_days,
                    target_date=target_date,
                    actual=actual,
                    prediction=baseline,
                    context_start=min(curves),
                    cutoff=cutoff,
                    missing_points=0,
                    day_type=day_types[target_date],
                )
            )

    predictions = pd.DataFrame(rows)
    summary = score_groups(predictions)
    predictions.to_csv(
        args.output_dir / "predictions_96.csv.gz", index=False, compression="gzip"
    )
    summary.to_csv(
        args.output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "version": VERSION,
        "built_at": datetime.now().astimezone().isoformat(),
        "status": "OFFLINE_RESEARCH_ONLY",
        "network_mode": "offline",
        "selection_policy": "NO_SELECTION_FIXED_PRIOR_CANDIDATES",
        "label_age_days": list(LABEL_AGES),
        "target_start": args.target_start.isoformat(),
        "target_end": args.target_end.isoformat(),
        "target_days": len(target_dates),
        "chronos2_model_id": CHRONOS2_MODEL_ID,
        "chronos2_local_snapshot": chronos2_snapshot,
        "chronos2_context_days": CHRONOS2_CONTEXT_DAYS,
        "bolt_model_id": BOLT_MODEL_ID,
        "bolt_local_snapshot": bolt_snapshot,
        "bolt_context_days": "all",
        "chronos_version": getattr(chronos, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "portfolio_input": str(args.portfolio_input.resolve()),
        "portfolio_sha256": sha256_file(args.portfolio_input),
        "anchor_predictions": str(args.anchor_predictions.resolve()),
        "anchor_predictions_sha256": sha256_file(args.anchor_predictions),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(summary.loc[summary["group"].eq("all")].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-input", type=Path, required=True)
    parser.add_argument("--anchor-predictions", type=Path, required=True)
    parser.add_argument("--anchor-model", default="anchor_core_weather")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-start", type=date.fromisoformat, default=date(2026, 6, 15))
    parser.add_argument("--target-end", type=date.fromisoformat, default=date(2026, 8, 5))
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
