"""Strict D-3 account-level weekend-state segmentation and forecast backtest.

Customer identifiers and rankings are written only to the private output.
The public output contains aggregate segment statistics and predictions.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.calendar_features import day_type
from project.research.jiangsu.user_load_forecast.detect_user_pv_candidates import (
    parse_pg_array,
    sha256_file,
)
from project.research.jiangsu.user_load_forecast.model_backtest import prepare_output_dir
from project.research.jiangsu.user_load_forecast.pv_weather_challenger_audit import (
    score_window,
)
from project.research.jiangsu.user_load_forecast.strict_operational_weather_backtest import (
    daily_metrics,
)


VERSION = "surun_user_weekend_state_segment_v1"
SEGMENTS = ("weekend_off_stable", "weekend_active_stable", "weekend_variable", "insufficient")
METHODS = (
    "mean4", "median4", "mean2", "last1",
    "last25_mean75", "last50_mean50", "last75_mean25",
    "active1_median4", "active2_median4", "active3_median4", "active5_median4",
    "active2_last50_mean50", "active3_last50_mean50", "active5_last50_mean50",
)
ZERO = np.zeros(96, dtype=float)


def normalize_account(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def read_account_curves(
    history_input: Path,
    update_inputs: list[Path],
) -> tuple[dict[tuple[str, str, date], np.ndarray], dict[str, str]]:
    curves: dict[tuple[str, str, date], np.ndarray] = defaultdict(lambda: np.zeros(96))
    names: dict[str, str] = {}
    opener = gzip.open if history_input.name.endswith(".gz") else open
    with opener(history_input, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            account = normalize_account(row["account_id"])
            name = row["retail_user_name"].strip()
            trade_date = date.fromisoformat(row["trade_date"])
            curves[(account, name, trade_date)] += parse_pg_array(row["actual_values"])
            names[account] = name
    for path in update_inputs:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        time_columns = [
            column for column in frame.columns
            if isinstance(column, str) and len(column) == 5 and column[2] == ":"
        ]
        if not {"零售用户名", "户号", "日期"}.issubset(frame.columns) or len(time_columns) != 96:
            raise ValueError(f"Invalid account update: {path}")
        frame[time_columns] = frame[time_columns].apply(pd.to_numeric, errors="raise")
        for (account_raw, name_raw, date_raw), group in frame.groupby(
            ["户号", "零售用户名", "日期"], dropna=False
        ):
            account = normalize_account(account_raw)
            name = str(name_raw).strip()
            trade_date = pd.Timestamp(date_raw).date()
            curves[(account, name, trade_date)] += group[time_columns].sum().to_numpy(float)
            names[account] = name
    return dict(curves), names


def index_curves(
    raw: dict[tuple[str, str, date], np.ndarray],
) -> tuple[dict[str, dict[date, np.ndarray]], dict[date, dict[str, np.ndarray]]]:
    by_account: dict[str, dict[date, np.ndarray]] = defaultdict(dict)
    by_date: dict[date, dict[str, np.ndarray]] = defaultdict(dict)
    for (account, _name, trade_date), curve in raw.items():
        by_account[account][trade_date] = curve
        by_date[trade_date][account] = curve
    return dict(by_account), dict(by_date)


def classify_account(history: dict[date, np.ndarray], cutoff: date) -> tuple[str, dict[str, float]]:
    visible = [(d, c) for d, c in history.items() if d <= cutoff]
    weekday = [c.sum() for d, c in visible if day_type(d) in {"workday", "adjusted_workday"}]
    weekend = [c.sum() for d, c in visible if day_type(d) == "weekend"]
    metrics = {
        "visible_days": float(len(visible)),
        "weekday_days": float(len(weekday)),
        "weekend_days": float(len(weekend)),
        "weekend_weekday_energy_ratio": np.nan,
        "weekend_energy_cv": np.nan,
        "weekend_morning_energy_share": np.nan,
    }
    if len(weekday) < 10 or len(weekend) < 3:
        return "insufficient", metrics
    weekday_median = float(np.median(weekday))
    weekend_median = float(np.median(weekend))
    ratio = weekend_median / weekday_median if abs(weekday_median) > 1e-9 else 0.0
    cv = float(np.std(weekend) / (abs(np.mean(weekend)) + 1e-9))
    weekend_curves = [c for d, c in visible if day_type(d) == "weekend"]
    morning_share = float(
        np.median([c[31:49].sum() / (c.sum() + 1e-9) for c in weekend_curves])
    )
    metrics.update(
        {
            "weekend_weekday_energy_ratio": ratio,
            "weekend_energy_cv": cv,
            "weekend_morning_energy_share": morning_share,
        }
    )
    if ratio < 0.30 and cv <= 0.40:
        return "weekend_off_stable", metrics
    if cv <= 0.25:
        return "weekend_active_stable", metrics
    return "weekend_variable", metrics


def source_curves(history: dict[date, np.ndarray], target: date) -> list[np.ndarray]:
    cutoff = target - timedelta(days=3)
    visible_dates = sorted(d for d in history if d <= cutoff)
    same_weekday = [d for d in visible_dates if d.weekday() == target.weekday()]
    selected = same_weekday[-4:]
    if not selected:
        same_type = [d for d in visible_dates if day_type(d) == day_type(target)]
        selected = same_type[-4:]
    if not selected:
        selected = visible_dates[-4:]
    return [history[d] for d in selected]


def activity_lookback(method: str) -> int | None:
    if not method.startswith("active"):
        return None
    return int(method.split("_", 1)[0].removeprefix("active"))


def base_method(method: str) -> str:
    return method.split("_", 1)[1] if method.startswith("active") else method


def predict_method(
    sources: list[np.ndarray],
    method: str,
    *,
    active: bool = True,
) -> np.ndarray:
    if not active:
        return ZERO.copy()
    method = base_method(method)
    if not sources:
        return ZERO.copy()
    matrix = np.vstack(sources)
    mean4 = matrix.mean(axis=0)
    if method == "mean4":
        return mean4
    if method == "median4":
        return np.median(matrix, axis=0)
    if method == "mean2":
        return matrix[-2:].mean(axis=0)
    if method == "last1":
        return matrix[-1]
    alpha = {
        "last25_mean75": 0.25,
        "last50_mean50": 0.50,
        "last75_mean25": 0.75,
    }[method]
    return alpha * matrix[-1] + (1.0 - alpha) * mean4


def build_segment_method_predictions(
    by_account: dict[str, dict[date, np.ndarray]],
    by_date: dict[date, dict[str, np.ndarray]],
    target_dates: list[date],
    *,
    methods: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    all_accounts = sorted(by_account)
    available_dates = sorted(by_date)
    for target in target_dates:
        cutoff = target - timedelta(days=3)
        visible_system_dates = [d for d in available_dates if d <= cutoff]
        actual_accounts = by_date.get(target, {})
        aggregate = {
            (segment, method): np.zeros(96, dtype=float)
            for segment in SEGMENTS for method in methods
        }
        actual_segment = {segment: np.zeros(96, dtype=float) for segment in SEGMENTS}
        for account in all_accounts:
            history = by_account[account]
            known = any(d <= cutoff for d in history)
            segment, _ = classify_account(history, cutoff) if known else ("insufficient", {})
            actual = actual_accounts.get(account, ZERO)
            actual_segment[segment] += actual
            sources = source_curves(history, target) if known else []
            for method in methods:
                lookback = activity_lookback(method)
                active = True
                if lookback is not None:
                    recent_dates = visible_system_dates[-lookback:]
                    active = any(d in history for d in recent_dates)
                aggregate[(segment, method)] += predict_method(
                    sources, method, active=active
                )
        for segment in SEGMENTS:
            actual = actual_segment[segment]
            for method in methods:
                prediction = aggregate[(segment, method)]
                for interval in range(1, 97):
                    error = float(prediction[interval - 1] - actual[interval - 1])
                    rows.append(
                        {
                            "segment": segment,
                            "method": method,
                            "target_date": target,
                            "trade_interval": interval,
                            "actual_load": float(actual[interval - 1]),
                            "predicted_load": float(prediction[interval - 1]),
                            "error": error,
                            "absolute_error": abs(error),
                            "load_label_cutoff": cutoff,
                        }
                    )
    return pd.DataFrame(rows)


def choose_methods(validation: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:
    rows = []
    for (segment, method), group in validation.groupby(["segment", "method"]):
        actual = group["actual_load"].abs().sum()
        rows.append(
            {
                "segment": segment,
                "method": method,
                "wmape_pct": float(100.0 * group["absolute_error"].sum() / (actual + 1e-9)),
                "bias_pct": float(100.0 * group["error"].sum() / (group["actual_load"].sum() + 1e-9)),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["segment", "wmape_pct", "method"])
    selected = {
        segment: str(group.iloc[0]["method"])
        for segment, group in summary.groupby("segment", sort=True)
    }
    return selected, summary


def aggregate_models(frame: pd.DataFrame, selected: dict[str, str]) -> pd.DataFrame:
    outputs = []
    for model in ("user_mean4_baseline", "user_weekend_state_router"):
        if model == "user_mean4_baseline":
            scoped = frame.loc[frame["method"].eq("mean4")]
        else:
            mask = pd.Series(False, index=frame.index)
            for segment, method in selected.items():
                mask |= frame["segment"].eq(segment) & frame["method"].eq(method)
            scoped = frame.loc[mask]
        output = scoped.groupby(["target_date", "trade_interval"], as_index=False).agg(
            actual_load=("actual_load", "sum"),
            predicted_load=("predicted_load", "sum"),
            load_label_cutoff=("load_label_cutoff", "max"),
        )
        output.insert(0, "model", model)
        output["error"] = output["predicted_load"] - output["actual_load"]
        output["absolute_error"] = output["error"].abs()
        outputs.append(output)
    return pd.concat(outputs, ignore_index=True)


def private_account_audit(
    by_account: dict[str, dict[date, np.ndarray]],
    by_date: dict[date, dict[str, np.ndarray]],
    names: dict[str, str],
    selected: dict[str, str],
    target_dates: list[date],
    latest_cutoff: date,
    available_dates: list[date],
) -> pd.DataFrame:
    rows = []
    for account, history in by_account.items():
        segment, features = classify_account(history, latest_cutoff)
        method = selected.get(segment, "mean4")
        morning_ae = 0.0
        total_ae = 0.0
        actual_energy = 0.0
        signed_morning_error = 0.0
        scored_days = 0
        for target in target_dates:
            cutoff = target - timedelta(days=3)
            dynamic_segment, _ = classify_account(history, cutoff)
            dynamic_method = selected.get(dynamic_segment, "mean4")
            lookback = activity_lookback(dynamic_method)
            active = True
            if lookback is not None:
                recent_dates = [d for d in available_dates if d <= cutoff][-lookback:]
                active = any(d in history for d in recent_dates)
            prediction = predict_method(
                source_curves(history, target), dynamic_method, active=active
            )
            actual = by_date.get(target, {}).get(account, ZERO)
            error = prediction - actual
            morning_ae += float(np.abs(error[31:49]).sum())
            signed_morning_error += float(error[31:49].sum())
            total_ae += float(np.abs(error).sum())
            actual_energy += float(np.abs(actual).sum())
            scored_days += 1
        rows.append(
            {
                "account_id": account,
                "retail_user_name": names.get(account, ""),
                "latest_segment": segment,
                "selected_method": method,
                **features,
                "scored_days": scored_days,
                "actual_energy": actual_energy,
                "morning_32_49_absolute_error": morning_ae,
                "morning_32_49_signed_error": signed_morning_error,
                "total_absolute_error": total_ae,
                "account_wmape_pct": 100.0 * total_ae / (actual_energy + 1e-9),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["morning_32_49_absolute_error", "total_absolute_error"], ascending=False
    )


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    prepare_output_dir(args.private_output_dir)
    raw, names = read_account_curves(args.user_history_input, args.user_update_input)
    by_account, by_date = index_curves(raw)
    available_dates = sorted(by_date)
    validation_dates = [
        d for d in available_dates if args.validation_start <= d <= args.validation_end
    ]
    evaluation_dates = [
        d for d in available_dates if args.holdout_start <= d <= args.new_diagnostic_end
    ]
    validation = build_segment_method_predictions(
        by_account, by_date, validation_dates, methods=METHODS
    )
    selected, selection_summary = choose_methods(validation)
    evaluation = build_segment_method_predictions(
        by_account, by_date, evaluation_dates, methods=METHODS
    )
    predictions = aggregate_models(evaluation, selected)
    summary_rows = []
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
    summary = pd.DataFrame(summary_rows).sort_values(["window", "wmape_pct", "model"])
    daily = daily_metrics(predictions)
    segment_metrics = []
    selected_rows = pd.concat(
        [
            evaluation.loc[
                evaluation["segment"].eq(segment) & evaluation["method"].eq(method)
            ]
            for segment, method in selected.items()
        ],
        ignore_index=True,
    )
    for (segment, window), group in selected_rows.assign(
        window=lambda x: np.where(
            x["target_date"].between(args.holdout_start, args.legacy_holdout_end),
            "legacy_holdout",
            "new_diagnostic",
        )
    ).groupby(["segment", "window"]):
        segment_metrics.append(
            {
                "segment": segment,
                "window": window,
                "target_days": group["target_date"].nunique(),
                "wmape_pct": 100.0 * group["absolute_error"].sum() / (group["actual_load"].abs().sum() + 1e-9),
                "actual_energy_share_pct": 100.0 * group["actual_load"].abs().sum() / (
                    selected_rows.loc[selected_rows["target_date"].isin(group["target_date"].unique()), "actual_load"].abs().sum() + 1e-9
                ),
            }
        )
    segment_metrics_frame = pd.DataFrame(segment_metrics)
    latest_cutoff = args.new_diagnostic_end - timedelta(days=3)
    audit = private_account_audit(
        by_account, by_date, names, selected, evaluation_dates, latest_cutoff,
        available_dates,
    )
    segment_counts = audit.groupby("latest_segment", as_index=False).agg(
        accounts=("account_id", "nunique"),
        actual_energy=("actual_energy", "sum"),
        morning_32_49_absolute_error=("morning_32_49_absolute_error", "sum"),
    )

    predictions.to_csv(args.output_dir / "predictions_96.csv.gz", index=False, compression="gzip")
    daily.to_csv(args.output_dir / "daily_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    selection_summary.to_csv(args.output_dir / "selection_summary.csv", index=False, encoding="utf-8-sig")
    segment_metrics_frame.to_csv(args.output_dir / "segment_metrics.csv", index=False, encoding="utf-8-sig")
    segment_counts.to_csv(args.output_dir / "segment_counts.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(
        args.private_output_dir / "account_weekend_state_and_error_ranking.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )
    audit.head(100).to_csv(
        args.private_output_dir / "top100_weekend_morning_error_accounts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "version": VERSION,
        "built_at": datetime.now().astimezone().isoformat(),
        "status": "OFFLINE_RESEARCH_ONLY",
        "privacy": "PUBLIC_SEGMENT_AGGREGATES_PRIVATE_ACCOUNT_IDENTIFIERS",
        "selection_policy": "MAY_VALIDATION_ONLY",
        "selected_methods": selected,
        "load_label_policy": "STRICT_TARGET_MINUS_3",
        "account_count": len(by_account),
        "available_date_min": min(available_dates).isoformat(),
        "available_date_max": max(available_dates).isoformat(),
        "history_input": str(args.user_history_input.resolve()),
        "history_sha256": sha256_file(args.user_history_input),
        "update_inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.user_update_input
        ],
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(summary[["model", "window", "wmape_pct"]].to_string(index=False))
    print(segment_counts.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-history-input", type=Path, required=True)
    parser.add_argument("--user-update-input", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
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
