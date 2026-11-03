"""Backtest target-day public DA system load as a portfolio-load correction.

The target-day DA system curve is treated as visible on D-1.  Historical RT
system load and DA/RT bias statistics are restricted to D-3 or earlier.  The
current export does not preserve historical publication timestamps, so all
results remain a research proxy until issue_at/available_at evidence exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.model_backtest import (
    prepare_output_dir,
)


VERSION = "surun_market_da_signal_backtest_v1"
MARKET_POLICY = "RESEARCH_PROXY_UNVERIFIED_PUBLICATION_TIME"
VARIANTS = (
    "target_da_vs_source_da",
    "target_da_vs_source_rt",
    "bias_corrected_da_vs_source_rt",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_system_market(path: Path) -> pd.DataFrame:
    parts = []
    usecols = [
        "trade_date",
        "trade_interval",
        "data_name",
        "data_type",
        "value",
        "created_at",
        "updated_at",
    ]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=300_000):
        selected = chunk.loc[
            chunk["data_name"].eq("SYSTEM_LOAD_POWER")
            & chunk["data_type"].isin(["DA", "RT"])
        ].copy()
        if not selected.empty:
            parts.append(selected)
    if not parts:
        raise ValueError("No SYSTEM_LOAD_POWER DA/RT rows found")
    market = pd.concat(parts, ignore_index=True)
    market["trade_date"] = pd.to_datetime(market["trade_date"]).dt.date
    duplicate = market.duplicated(
        ["trade_date", "trade_interval", "data_type"],
        keep=False,
    )
    if duplicate.any():
        raise ValueError("Duplicate SYSTEM_LOAD_POWER DA/RT keys")
    wide = (
        market.pivot(
            index=["trade_date", "trade_interval"],
            columns="data_type",
            values="value",
        )
        .reset_index()
        .sort_values(["trade_interval", "trade_date"])
    )
    wide["rt_da_ratio"] = wide["RT"] / wide["DA"]
    wide["visible_bias_ratio_28d"] = wide.groupby(
        "trade_interval",
        sort=False,
    )["rt_da_ratio"].transform(
        lambda value: value.rolling(28, min_periods=7).median()
    )
    return wide


def parse_visible_sources(value: object, cutoff: date) -> list[date]:
    sources = []
    for item in re.split(r"[|,]", str(value)):
        item = item.strip()
        if not item:
            continue
        source_date = pd.Timestamp(item).date()
        if source_date <= cutoff:
            sources.append(source_date)
    return sources


def build_signal_frame(
    calendar_predictions: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "target_date",
        "trade_interval",
        "actual_load",
        "predicted_load",
        "source_dates",
    }
    missing = required - set(calendar_predictions.columns)
    if missing:
        raise ValueError(f"Calendar predictions missing {sorted(missing)}")
    calendar = calendar_predictions.copy()
    calendar["target_date"] = pd.to_datetime(calendar["target_date"]).dt.date
    if "model" in calendar.columns:
        calendar = calendar.drop(columns="model")
    calendar = calendar.rename(
        columns={"predicted_load": "calendar_prediction"}
    )
    lookup = market.set_index(["trade_date", "trade_interval"])
    rows = []
    for row in calendar.itertuples(index=False):
        target_date = row.target_date
        trade_interval = int(row.trade_interval)
        cutoff = target_date - timedelta(days=3)
        target_key = (target_date, trade_interval)
        cutoff_key = (cutoff, trade_interval)
        if target_key not in lookup.index:
            raise ValueError(f"Missing target market key: {target_key}")
        target_da = lookup.loc[target_key, "DA"]
        if pd.isna(target_da):
            raise ValueError(f"Missing target DA system load: {target_key}")
        bias_ratio = (
            lookup.loc[cutoff_key, "visible_bias_ratio_28d"]
            if cutoff_key in lookup.index
            else np.nan
        )
        if pd.isna(bias_ratio):
            bias_ratio = 1.0
        sources = parse_visible_sources(row.source_dates, cutoff)
        if not sources:
            raise ValueError(
                f"No visible calendar sources for target {target_date}"
            )
        source_da = []
        source_rt = []
        for source_date in sources:
            key = (source_date, trade_interval)
            if key not in lookup.index:
                continue
            source_row = lookup.loc[key]
            if pd.notna(source_row["DA"]):
                source_da.append(float(source_row["DA"]))
            if pd.notna(source_row["RT"]):
                source_rt.append(float(source_row["RT"]))
        if not source_da or not source_rt:
            raise ValueError(
                f"Missing source DA/RT system load for {target_date} "
                f"interval {trade_interval}"
            )
        source_da_mean = float(np.mean(source_da))
        source_rt_mean = float(np.mean(source_rt))
        rows.append(
            {
                "target_date": target_date,
                "trade_interval": trade_interval,
                "market_label_cutoff": cutoff,
                "market_source_max_date": max(sources),
                "target_system_load_da": float(target_da),
                "source_system_load_da_mean": source_da_mean,
                "source_system_load_rt_mean": source_rt_mean,
                "visible_da_rt_bias_ratio_28d": float(bias_ratio),
                "target_da_vs_source_da": float(target_da) / source_da_mean,
                "target_da_vs_source_rt": float(target_da) / source_rt_mean,
                "bias_corrected_da_vs_source_rt": (
                    float(target_da) * float(bias_ratio) / source_rt_mean
                ),
            }
        )
    signals = pd.DataFrame(rows)
    frame = calendar.merge(
        signals,
        on=["target_date", "trade_interval"],
        validate="one_to_one",
    )
    if not (
        frame["market_source_max_date"] <= frame["market_label_cutoff"]
    ).all():
        raise ValueError("RT source-date leakage detected")
    counts = frame.groupby("target_date")["trade_interval"].agg(
        lambda value: set(value.astype(int))
    )
    if not counts.map(lambda value: value == set(range(1, 97))).all():
        raise ValueError("Every target date must contain intervals 1..96")
    return frame


def metrics(frame: pd.DataFrame) -> dict:
    daily = (
        frame.groupby("target_date", as_index=False)
        .agg(
            absolute_error_sum=("absolute_error", "sum"),
            actual_abs_sum=("actual_load", lambda value: value.abs().sum()),
        )
    )
    daily["wmape_pct"] = (
        daily["absolute_error_sum"] / daily["actual_abs_sum"] * 100
    )
    return {
        "target_days": len(daily),
        "wmape_pct": (
            frame["absolute_error"].sum()
            / frame["actual_load"].abs().sum()
            * 100
        ),
        "daily_wmape_p90_pct": daily["wmape_pct"].quantile(0.9),
        "daily_wmape_max_pct": daily["wmape_pct"].max(),
    }


def calibrate_variant(
    frame: pd.DataFrame,
    *,
    base_name: str,
    base_column: str,
    variant: str,
    calibration_end: date,
    holdout_start: date,
    beta_values: np.ndarray,
    p90_tolerance_pp: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    calibration_mask = frame["target_date"] <= calibration_end
    holdout_mask = frame["target_date"] >= holdout_start
    legal_mask = calibration_mask | holdout_mask
    grid_rows = []
    for beta in beta_values:
        candidate = frame[
            ["target_date", "trade_interval", "actual_load"]
        ].copy()
        candidate["predicted_load"] = frame[base_column] * (
            1.0 + beta * (frame[variant] - 1.0)
        )
        candidate["absolute_error"] = (
            candidate["predicted_load"] - candidate["actual_load"]
        ).abs()
        for window, mask in (
            ("calibration", calibration_mask),
            ("holdout", holdout_mask),
            ("overall", legal_mask),
        ):
            grid_rows.append(
                {
                    "base": base_name,
                    "variant": variant,
                    "beta": float(beta),
                    "window": window,
                    **metrics(candidate.loc[mask]),
                }
            )
    grid = pd.DataFrame(grid_rows)
    calibration = grid.loc[grid["window"].eq("calibration")]
    baseline = calibration.loc[np.isclose(calibration["beta"], 0.0)].iloc[0]
    eligible = calibration.loc[
        calibration["daily_wmape_p90_pct"]
        <= baseline["daily_wmape_p90_pct"] + p90_tolerance_pp
    ]
    if eligible.empty:
        raise ValueError(f"No beta passes P90 guard for {base_name}/{variant}")
    selected_beta = float(
        eligible.loc[eligible["wmape_pct"].idxmin(), "beta"]
    )
    predictions = frame.copy()
    predictions.insert(
        0,
        "model",
        f"market_da_elasticity__{base_name}__{variant}",
    )
    predictions["base"] = base_name
    predictions["variant"] = variant
    predictions["beta"] = selected_beta
    predictions["window"] = np.select(
        [calibration_mask, holdout_mask],
        ["calibration", "holdout"],
        default="label_maturity_embargo",
    )
    predictions["predicted_load"] = predictions[base_column] * (
        1.0 + selected_beta * (predictions[variant] - 1.0)
    )
    predictions["error"] = (
        predictions["predicted_load"] - predictions["actual_load"]
    )
    predictions["absolute_error"] = predictions["error"].abs()
    return predictions.loc[legal_mask].copy(), grid, selected_beta


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    calendar = pd.read_csv(args.calendar_prediction_file)
    market = read_system_market(args.market_input)
    frame = build_signal_frame(calendar, market)
    holdout_start = args.holdout_start or (
        args.calibration_end + timedelta(days=3)
    )
    if holdout_start < args.calibration_end + timedelta(days=3):
        raise ValueError(
            "holdout_start must be at least calibration_end + 3 days "
            "under the D-3 label-maturity policy"
        )
    bases = {"calendar": "calendar_prediction"}
    if args.strict_prediction_file:
        strict = pd.read_csv(args.strict_prediction_file)
        strict["target_date"] = pd.to_datetime(strict["target_date"]).dt.date
        strict = strict[
            ["target_date", "trade_interval", "actual_load", "predicted_load"]
        ].rename(
            columns={
                "actual_load": "strict_actual_load",
                "predicted_load": "strict_prediction",
            }
        )
        frame = frame.merge(
            strict,
            on=["target_date", "trade_interval"],
            validate="one_to_one",
        )
        if not np.allclose(
            frame["actual_load"],
            frame["strict_actual_load"],
        ):
            raise ValueError("Calendar and strict predictions disagree on actual")
        frame = frame.drop(columns="strict_actual_load")
        bases["strict_fusion"] = "strict_prediction"

    beta_values = np.round(
        np.arange(
            args.beta_min,
            args.beta_max + args.beta_step / 2,
            args.beta_step,
        ),
        10,
    )
    prediction_frames = []
    grid_frames = []
    selected_rows = []
    for base_name, base_column in bases.items():
        for variant in VARIANTS:
            predictions, grid, selected_beta = calibrate_variant(
                frame,
                base_name=base_name,
                base_column=base_column,
                variant=variant,
                calibration_end=args.calibration_end,
                holdout_start=holdout_start,
                beta_values=beta_values,
                p90_tolerance_pp=args.calibration_p90_tolerance_pp,
            )
            prediction_frames.append(predictions)
            grid_frames.append(grid)
            selected_rows.append(
                grid.loc[np.isclose(grid["beta"], selected_beta)]
            )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    grid = pd.concat(grid_frames, ignore_index=True)
    selected_summary = pd.concat(selected_rows, ignore_index=True)
    daily = (
        predictions.groupby(
            ["model", "base", "variant", "window", "target_date"],
            as_index=False,
        )
        .agg(
            absolute_error_sum=("absolute_error", "sum"),
            actual_abs_sum=("actual_load", lambda value: value.abs().sum()),
        )
    )
    daily["wmape_pct"] = (
        daily["absolute_error_sum"] / daily["actual_abs_sum"] * 100
    )
    keep = [
        "model",
        "base",
        "variant",
        "beta",
        "window",
        "target_date",
        "trade_interval",
        "market_label_cutoff",
        "market_source_max_date",
        "target_system_load_da",
        "source_system_load_da_mean",
        "source_system_load_rt_mean",
        "visible_da_rt_bias_ratio_28d",
        "actual_load",
        "predicted_load",
        "error",
        "absolute_error",
    ]
    predictions[keep].to_csv(
        args.output_dir / "predictions_96.csv.gz",
        index=False,
        compression="gzip",
    )
    daily.to_csv(
        args.output_dir / "daily_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    grid.to_csv(
        args.output_dir / "beta_grid_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected_summary.to_csv(
        args.output_dir / "selected_beta_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "version": VERSION,
        "target_start": min(frame["target_date"]).isoformat(),
        "target_end": max(frame["target_date"]).isoformat(),
        "calibration_end": args.calibration_end.isoformat(),
        "holdout_start": holdout_start.isoformat(),
        "label_maturity_embargo_days": (
            holdout_start - args.calibration_end
        ).days - 1,
        "market_policy": MARKET_POLICY,
        "target_da_visibility_assumption": (
            "target-day SYSTEM_LOAD_POWER DA is visible by D-1 submission"
        ),
        "historical_market_visibility": (
            "RT and DA/RT bias source dates are target D-3 or earlier"
        ),
        "champion_eligible": False,
        "eligibility_reason": (
            "source export created_at is a 2026-07-28 backfill timestamp and "
            "does not prove historical target-DA publication time"
        ),
        "selection_rule": (
            "minimum calibration WMAPE subject to calibration P90 no more "
            f"than beta=0 + {args.calibration_p90_tolerance_pp:.3f}pp"
        ),
        "market_input": str(args.market_input.resolve()),
        "market_input_sha256": sha256_file(args.market_input),
        "calendar_prediction_sha256": sha256_file(
            args.calendar_prediction_file
        ),
        "strict_prediction_sha256": (
            sha256_file(args.strict_prediction_file)
            if args.strict_prediction_file
            else None
        ),
        "market_target_da_coverage": float(
            frame["target_system_load_da"].notna().mean()
        ),
        "prediction_rows": len(predictions),
        "pii_output_check": "aggregate-only outputs",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(selected_summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar-prediction-file", type=Path, required=True)
    parser.add_argument("--strict-prediction-file", type=Path)
    parser.add_argument("--market-input", type=Path, required=True)
    parser.add_argument(
        "--calibration-end",
        type=date.fromisoformat,
        default=date(2026, 6, 14),
    )
    parser.add_argument("--holdout-start", type=date.fromisoformat)
    parser.add_argument("--beta-min", type=float, default=-0.5)
    parser.add_argument("--beta-max", type=float, default=1.5)
    parser.add_argument("--beta-step", type=float, default=0.025)
    parser.add_argument("--calibration-p90-tolerance-pp", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
