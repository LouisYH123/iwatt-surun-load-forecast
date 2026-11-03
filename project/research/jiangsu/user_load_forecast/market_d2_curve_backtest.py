"""Evaluate D-2 public RT market features and full 96-point curve fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from project.research.jiangsu.user_load_forecast.fusion import (
    nonnegative_weights,
)
from project.research.jiangsu.user_load_forecast.market_da_signal_backtest import (
    read_system_market,
)
from project.research.jiangsu.user_load_forecast.model_backtest import (
    prepare_output_dir,
)


VERSION = "surun_market_d2_curve_backtest_v1"
BASE_FEATURES = (
    "target_da_vs_source_rt",
    "log_calendar_prediction",
    "interval_sin",
    "interval_cos",
    "trade_interval",
    "day_type",
)
D2_FEATURES = BASE_FEATURES + (
    "d2_rt_da_bias",
    "d2_rt_vs_source_rt",
    "target_da_vs_d2_rt",
    "d2_rt_vs_d3_rt",
    "d2_rt_vs_d7_rt",
    "d2_da_vs_d3_da",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def visible_source_dates(value: object, cutoff: date) -> list[date]:
    sources = []
    for item in re.split(r"[|,]", str(value)):
        item = item.strip()
        if not item:
            continue
        source_date = pd.Timestamp(item).date()
        if source_date <= cutoff:
            sources.append(source_date)
    return sources


def build_features(
    calendar_prediction_file: Path,
    market_input: Path,
) -> pd.DataFrame:
    calendar = pd.read_csv(calendar_prediction_file)
    calendar = calendar.loc[
        calendar["model"].eq("calendar_similar_day_mean4")
    ].copy()
    calendar["target_date"] = pd.to_datetime(
        calendar["target_date"]
    ).dt.date
    market = read_system_market(market_input)
    lookup = market.set_index(["trade_date", "trade_interval"])
    rows = []
    for row in calendar.itertuples(index=False):
        target_date = row.target_date
        trade_interval = int(row.trade_interval)
        load_cutoff = target_date - timedelta(days=3)
        d2 = target_date - timedelta(days=2)
        d3 = target_date - timedelta(days=3)
        d7 = target_date - timedelta(days=7)
        sources = visible_source_dates(row.source_dates, load_cutoff)
        if not sources:
            raise ValueError(f"No D-3-visible sources for {target_date}")
        source_rt = [
            float(lookup.loc[(source, trade_interval), "RT"])
            for source in sources
            if (source, trade_interval) in lookup.index
            and pd.notna(lookup.loc[(source, trade_interval), "RT"])
        ]
        if not source_rt:
            raise ValueError(
                f"No source RT for {target_date}/{trade_interval}"
            )
        required_keys = [
            (target_date, trade_interval),
            (d2, trade_interval),
            (d3, trade_interval),
            (d7, trade_interval),
        ]
        missing = [key for key in required_keys if key not in lookup.index]
        if missing:
            raise ValueError(f"Missing market feature keys: {missing[:3]}")
        target = lookup.loc[(target_date, trade_interval)]
        market_d2 = lookup.loc[(d2, trade_interval)]
        market_d3 = lookup.loc[(d3, trade_interval)]
        market_d7 = lookup.loc[(d7, trade_interval)]
        values = [
            target["DA"],
            market_d2["DA"],
            market_d2["RT"],
            market_d3["DA"],
            market_d3["RT"],
            market_d7["RT"],
        ]
        if pd.isna(values).any():
            raise ValueError(
                f"Missing DA/RT value for {target_date}/{trade_interval}"
            )
        source_rt_mean = float(np.mean(source_rt))
        interval_angle = 2.0 * np.pi * (trade_interval - 1) / 96.0
        rows.append(
            {
                "target_date": target_date,
                "trade_interval": trade_interval,
                "day_type": row.day_type,
                "actual_load": float(row.actual_load),
                "calendar_prediction": float(row.predicted_load),
                "load_label_cutoff": load_cutoff,
                "market_rt_cutoff": d2,
                "market_source_max_date": max(sources),
                "target_da_vs_source_rt": float(target["DA"])
                / source_rt_mean,
                "log_calendar_prediction": np.log1p(row.predicted_load),
                "interval_sin": np.sin(interval_angle),
                "interval_cos": np.cos(interval_angle),
                "d2_rt_da_bias": float(market_d2["RT"])
                / float(market_d2["DA"]),
                "d2_rt_vs_source_rt": float(market_d2["RT"])
                / source_rt_mean,
                "target_da_vs_d2_rt": float(target["DA"])
                / float(market_d2["RT"]),
                "d2_rt_vs_d3_rt": float(market_d2["RT"])
                / float(market_d3["RT"]),
                "d2_rt_vs_d7_rt": float(market_d2["RT"])
                / float(market_d7["RT"]),
                "d2_da_vs_d3_da": float(market_d2["DA"])
                / float(market_d3["DA"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["residual_ratio"] = (
        frame["actual_load"] - frame["calendar_prediction"]
    ) / frame["calendar_prediction"]
    if not (
        frame["market_source_max_date"] <= frame["load_label_cutoff"]
    ).all():
        raise ValueError("Portfolio/source RT leakage detected")
    if not (
        frame["market_rt_cutoff"]
        == frame["target_date"].map(lambda value: value - timedelta(days=2))
    ).all():
        raise ValueError("D-2 market cutoff check failed")
    counts = frame.groupby("target_date")["trade_interval"].agg(
        lambda value: set(value.astype(int))
    )
    if not counts.map(lambda value: value == set(range(1, 97))).all():
        raise ValueError("Every target date must contain intervals 1..96")
    return frame


def model_pipeline(feature_columns: tuple[str, ...]) -> Pipeline:
    numeric = [
        column for column in feature_columns if column != "day_type"
    ]
    preprocessor = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            (
                "day_type",
                OneHotEncoder(handle_unknown="ignore"),
                ["day_type"],
            ),
        ]
    )
    model = GradientBoostingRegressor(
        loss="huber",
        n_estimators=120,
        learning_rate=0.04,
        max_depth=2,
        min_samples_leaf=12,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def fit_candidate(
    frame: pd.DataFrame,
    *,
    model_name: str,
    feature_columns: tuple[str, ...],
    train_end: date,
    holdout_start: date,
) -> pd.DataFrame:
    train = frame["target_date"] <= train_end
    holdout = frame["target_date"] >= holdout_start
    pipeline = model_pipeline(feature_columns)
    pipeline.fit(
        frame.loc[train, list(feature_columns)],
        frame.loc[train, "residual_ratio"],
    )
    residual = np.clip(
        pipeline.predict(frame.loc[holdout, list(feature_columns)]),
        -0.35,
        0.35,
    )
    predictions = frame.loc[
        holdout,
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
    predictions.insert(0, "model", model_name)
    predictions["predicted_load"] = (
        predictions["calendar_prediction"].to_numpy() * (1.0 + residual)
    )
    predictions["error"] = (
        predictions["predicted_load"] - predictions["actual_load"]
    )
    predictions["absolute_error"] = predictions["error"].abs()
    return predictions


def learn_daytype_guard(
    frame: pd.DataFrame,
    *,
    train_end: date,
    validation_start: date,
    validation_end: date,
) -> dict[str, float]:
    if validation_start < train_end + timedelta(days=3):
        raise ValueError(
            "guard validation must start at least train_end + 3 days"
        )
    validation = fit_candidate(
        frame,
        model_name="guard_internal_validation",
        feature_columns=D2_FEATURES,
        train_end=train_end,
        holdout_start=validation_start,
    )
    validation = validation.loc[
        validation["target_date"] <= validation_end
    ].copy()
    if validation.empty:
        raise ValueError("No internal guard-validation dates")
    weights = {}
    for group_name, group_mask in (
        ("workday", validation["day_type"].eq("workday")),
        ("nonwork", ~validation["day_type"].eq("workday")),
    ):
        group = validation.loc[group_mask]
        if group.empty:
            raise ValueError(f"No guard-validation rows for {group_name}")
        rows = []
        for alpha in np.round(np.arange(0.0, 1.0001, 0.025), 10):
            predicted = (
                alpha * group["predicted_load"]
                + (1.0 - alpha) * group["calendar_prediction"]
            )
            score = (
                (predicted - group["actual_load"]).abs().sum()
                / group["actual_load"].abs().sum()
                * 100
            )
            rows.append((score, float(alpha)))
        weights[group_name] = min(rows)[1]
    return weights


def apply_daytype_guard(
    predictions: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    guarded = predictions.copy()
    guarded["model"] = "gbdt_d2_daytype_guard"
    guarded["guard_group"] = np.where(
        guarded["day_type"].eq("workday"),
        "workday",
        "nonwork",
    )
    guarded["guard_d2_weight"] = guarded["guard_group"].map(weights)
    guarded["predicted_load"] = (
        guarded["guard_d2_weight"] * guarded["predicted_load"]
        + (1.0 - guarded["guard_d2_weight"])
        * guarded["calendar_prediction"]
    )
    guarded["error"] = (
        guarded["predicted_load"] - guarded["actual_load"]
    )
    guarded["absolute_error"] = guarded["error"].abs()
    return guarded


def read_external_candidate(
    path: Path,
    *,
    model: str,
    candidate: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["model"].eq(model)].copy()
    if frame.empty:
        raise ValueError(f"{path}: missing model {model}")
    frame["target_date"] = pd.to_datetime(frame["target_date"]).dt.date
    frame = frame[
        ["target_date", "trade_interval", "actual_load", "predicted_load"]
    ].rename(
        columns={
            "actual_load": f"actual__{candidate}",
            "predicted_load": candidate,
        }
    )
    if frame.duplicated(["target_date", "trade_interval"]).any():
        raise ValueError(f"{path}: duplicate keys for {model}")
    return frame


def add_external_candidates(
    predictions: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    merged = predictions.rename(
        columns={
            "predicted_load": "d2",
            "calendar_prediction": "calendar",
        }
    ).copy()
    specifications = (
        (
            args.xgboost_prediction_file,
            "xgboost",
            "xgboost",
        ),
        (
            args.strict_prediction_file,
            "dynamic_nonnegative_fusion",
            "strict",
        ),
        (
            args.industry_prediction_file,
            "industry_behavior__recent_trend",
            "industry",
        ),
    )
    for path, model, candidate in specifications:
        external = read_external_candidate(
            path,
            model=model,
            candidate=candidate,
        )
        merged = merged.merge(
            external,
            on=["target_date", "trade_interval"],
            validate="one_to_one",
        )
        actual_column = f"actual__{candidate}"
        if not np.allclose(
            merged["actual_load"],
            merged[actual_column],
        ):
            raise ValueError(f"Actual values disagree for {candidate}")
        merged = merged.drop(columns=actual_column)
    return merged


def learn_guarded_fusion(
    frame: pd.DataFrame,
    *,
    final_d2_predictions: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, float]]:
    validation = fit_candidate(
        frame,
        model_name="fusion_internal_validation",
        feature_columns=D2_FEATURES,
        train_end=args.guard_train_end,
        holdout_start=args.guard_validation_start,
    )
    validation = validation.loc[
        validation["target_date"] <= args.train_end
    ].copy()
    validation = add_external_candidates(validation, args)
    candidates = ["d2", "xgboost", "calendar", "strict", "industry"]
    weights, optimizer_success = nonnegative_weights(
        validation[candidates].to_numpy(dtype=float),
        validation["actual_load"].to_numpy(dtype=float),
    )
    if not optimizer_success:
        raise ValueError("Guarded-fusion optimizer failed")
    weight_map = {
        candidate: float(weight)
        for candidate, weight in zip(candidates, weights, strict=True)
    }
    holdout = add_external_candidates(final_d2_predictions, args)
    fused = final_d2_predictions.copy()
    fused["model"] = "gbdt_d2_guarded_fusion"
    fused["predicted_load"] = (
        holdout[candidates].to_numpy(dtype=float) @ weights
    )
    fused["error"] = fused["predicted_load"] - fused["actual_load"]
    fused["absolute_error"] = fused["error"].abs()
    return fused, weight_map


def curve_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    daily_rows = []
    for target_date, group in predictions.groupby("target_date", sort=True):
        group = group.sort_values("trade_interval")
        actual = group["actual_load"].to_numpy(dtype=float)
        predicted = group["predicted_load"].to_numpy(dtype=float)
        absolute_error = np.abs(predicted - actual)
        actual_total = float(actual.sum())
        predicted_total = float(predicted.sum())
        actual_peak_index = int(np.argmax(actual))
        predicted_peak_index = int(np.argmax(predicted))
        daily_rows.append(
            {
                "target_date": target_date,
                "wmape_pct": absolute_error.sum()
                / np.abs(actual).sum()
                * 100,
                "nrmse_pct": np.sqrt(np.mean((predicted - actual) ** 2))
                / np.mean(np.abs(actual))
                * 100,
                "curve_correlation": np.corrcoef(actual, predicted)[0, 1],
                "shape_wmape_pct": np.abs(
                    predicted / predicted_total - actual / actual_total
                ).sum()
                * 100,
                "daily_energy_bias_pct": (
                    predicted_total - actual_total
                )
                / actual_total
                * 100,
                "peak_load_error_pct": (
                    predicted[predicted_peak_index] - actual[actual_peak_index]
                )
                / actual[actual_peak_index]
                * 100,
                "peak_time_error_minutes": abs(
                    predicted_peak_index - actual_peak_index
                )
                * 15,
            }
        )
    daily = pd.DataFrame(daily_rows)
    point_actual = predictions["actual_load"].to_numpy(dtype=float)
    point_prediction = predictions["predicted_load"].to_numpy(dtype=float)
    summary = {
        "target_days": len(daily),
        "wmape_pct": predictions["absolute_error"].sum()
        / predictions["actual_load"].abs().sum()
        * 100,
        "daily_wmape_p90_pct": daily["wmape_pct"].quantile(0.9),
        "daily_wmape_max_pct": daily["wmape_pct"].max(),
        "point_correlation": np.corrcoef(
            point_actual,
            point_prediction,
        )[0, 1],
        "median_daily_curve_correlation": daily[
            "curve_correlation"
        ].median(),
        "p10_daily_curve_correlation": daily[
            "curve_correlation"
        ].quantile(0.1),
        "mean_nrmse_pct": daily["nrmse_pct"].mean(),
        "mean_shape_wmape_pct": daily["shape_wmape_pct"].mean(),
        "mean_abs_daily_energy_bias_pct": daily[
            "daily_energy_bias_pct"
        ].abs().mean(),
        "mean_peak_time_error_minutes": daily[
            "peak_time_error_minutes"
        ].mean(),
        "mean_abs_peak_load_error_pct": daily[
            "peak_load_error_pct"
        ].abs().mean(),
    }
    return daily, summary


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    if args.holdout_start < args.train_end + timedelta(days=3):
        raise ValueError(
            "holdout_start must be at least train_end + 3 days "
            "under the user-load D-3 label policy"
        )
    frame = build_features(
        args.calendar_prediction_file,
        args.market_input,
    )
    specifications = {
        "gbdt_target_da_only": BASE_FEATURES,
        "gbdt_target_da_plus_d2_rt": D2_FEATURES,
    }
    prediction_frames = []
    daily_frames = []
    summary_rows = []
    for model_name, feature_columns in specifications.items():
        predictions = fit_candidate(
            frame,
            model_name=model_name,
            feature_columns=feature_columns,
            train_end=args.train_end,
            holdout_start=args.holdout_start,
        )
        daily, summary = curve_metrics(predictions)
        daily.insert(0, "model", model_name)
        summary_rows.append(
            {
                "model": model_name,
                "feature_columns": "|".join(feature_columns),
                **summary,
            }
        )
        prediction_frames.append(predictions)
        daily_frames.append(daily)
    guard_weights = learn_daytype_guard(
        frame,
        train_end=args.guard_train_end,
        validation_start=args.guard_validation_start,
        validation_end=args.train_end,
    )
    d2_predictions = prediction_frames[
        list(specifications).index("gbdt_target_da_plus_d2_rt")
    ]
    guarded_predictions = apply_daytype_guard(
        d2_predictions,
        guard_weights,
    )
    guarded_daily, guarded_summary = curve_metrics(guarded_predictions)
    guarded_daily.insert(0, "model", "gbdt_d2_daytype_guard")
    summary_rows.append(
        {
            "model": "gbdt_d2_daytype_guard",
            "feature_columns": (
                "|".join(D2_FEATURES) + "|daytype_guard"
            ),
            **guarded_summary,
        }
    )
    prediction_frames.append(guarded_predictions)
    daily_frames.append(guarded_daily)
    fusion_weights = None
    external_paths = (
        args.xgboost_prediction_file,
        args.strict_prediction_file,
        args.industry_prediction_file,
    )
    if any(external_paths) and not all(external_paths):
        raise ValueError(
            "Provide all three external prediction files or none"
        )
    if all(external_paths):
        fused_predictions, fusion_weights = learn_guarded_fusion(
            frame,
            final_d2_predictions=d2_predictions,
            args=args,
        )
        fused_daily, fused_summary = curve_metrics(fused_predictions)
        fused_daily.insert(0, "model", "gbdt_d2_guarded_fusion")
        summary_rows.append(
            {
                "model": "gbdt_d2_guarded_fusion",
                "feature_columns": (
                    "|".join(D2_FEATURES)
                    + "|xgboost|calendar|strict|industry"
                ),
                **fused_summary,
            }
        )
        prediction_frames.append(fused_predictions)
        daily_frames.append(fused_daily)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    daily_metrics = pd.concat(daily_frames, ignore_index=True)
    summary_metrics = pd.DataFrame(summary_rows)
    mean_curves = (
        predictions.groupby(["model", "trade_interval"], as_index=False)
        .agg(
            actual_load_mean=("actual_load", "mean"),
            predicted_load_mean=("predicted_load", "mean"),
        )
    )
    predictions.to_csv(
        args.output_dir / "predictions_96.csv.gz",
        index=False,
        compression="gzip",
    )
    daily_metrics.to_csv(
        args.output_dir / "daily_curve_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_metrics.to_csv(
        args.output_dir / "summary_curve_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    mean_curves.to_csv(
        args.output_dir / "mean_curve_96.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "version": VERSION,
        "train_end": args.train_end.isoformat(),
        "holdout_start": args.holdout_start.isoformat(),
        "holdout_end": max(frame["target_date"]).isoformat(),
        "guard_train_end": args.guard_train_end.isoformat(),
        "guard_validation_start": args.guard_validation_start.isoformat(),
        "guard_validation_end": args.train_end.isoformat(),
        "guard_weights": guard_weights,
        "guarded_fusion_weights": fusion_weights,
        "user_load_label_policy": "training labels through holdout D-3",
        "market_rt_visibility": "D-2 RT is visible at D-1 submission",
        "target_market_visibility": "target-day DA is visible at D-1 submission",
        "market_policy": "RESEARCH_PROXY_UNVERIFIED_PUBLICATION_TIME",
        "champion_eligible": False,
        "eligibility_reason": (
            "market export lacks historical issue_at/available_at evidence"
        ),
        "calendar_prediction_sha256": sha256_file(
            args.calendar_prediction_file
        ),
        "market_input_sha256": sha256_file(args.market_input),
        "models": list(specifications),
        "external_prediction_sha256": (
            {
                "xgboost": sha256_file(args.xgboost_prediction_file),
                "strict": sha256_file(args.strict_prediction_file),
                "industry": sha256_file(args.industry_prediction_file),
            }
            if all(external_paths)
            else None
        ),
        "summary": {
            row["model"]: {
                key: (
                    int(row[key])
                    if key == "target_days"
                    else float(row[key])
                )
                for key in (
                    "target_days",
                    "wmape_pct",
                    "daily_wmape_p90_pct",
                    "daily_wmape_max_pct",
                    "point_correlation",
                    "median_daily_curve_correlation",
                    "p10_daily_curve_correlation",
                    "mean_nrmse_pct",
                    "mean_shape_wmape_pct",
                    "mean_abs_daily_energy_bias_pct",
                    "mean_peak_time_error_minutes",
                    "mean_abs_peak_load_error_pct",
                )
            }
            for _, row in summary_metrics.iterrows()
        },
        "pii_output_check": "aggregate-only outputs",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary_metrics.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar-prediction-file", type=Path, required=True)
    parser.add_argument("--market-input", type=Path, required=True)
    parser.add_argument(
        "--train-end",
        type=date.fromisoformat,
        default=date(2026, 6, 12),
    )
    parser.add_argument(
        "--holdout-start",
        type=date.fromisoformat,
        default=date(2026, 6, 15),
    )
    parser.add_argument(
        "--guard-train-end",
        type=date.fromisoformat,
        default=date(2026, 5, 29),
    )
    parser.add_argument(
        "--guard-validation-start",
        type=date.fromisoformat,
        default=date(2026, 6, 1),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xgboost-prediction-file", type=Path)
    parser.add_argument("--strict-prediction-file", type=Path)
    parser.add_argument("--industry-prediction-file", type=Path)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
