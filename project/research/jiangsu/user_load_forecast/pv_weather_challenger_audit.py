"""Audit PV/weather challenger features against the frozen Jiangsu core model.

This is a research-only workflow. Historical market and weather exports do not
preserve trustworthy issue_at/available_at timestamps, so none of the resulting
models is eligible for Champion promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.calendar_baseline import (
    select_similar_dates,
)
from project.research.jiangsu.user_load_forecast.calendar_features import day_type
from project.research.jiangsu.user_load_forecast.market_d2_curve_backtest import (
    build_features,
    curve_metrics,
    fit_candidate,
)
from project.research.jiangsu.user_load_forecast.market_level_shape_backtest import (
    MARKET_POLICY,
    add_expanded_market_features,
    read_market_matrix,
)
from project.research.jiangsu.user_load_forecast.model_backtest import (
    prepare_output_dir,
)
from project.research.jiangsu.user_load_forecast.weather_features import (
    WEATHER_POLICY_PROXY,
    read_weather_snapshot,
)


VERSION = "surun_pv_weather_challenger_audit_v1"
WEATHER_FEATURES = (
    "weather_temp_2m",
    "weather_humidity_2m",
    "weather_rain",
    "weather_radiation",
    "weather_wind_10m",
    "weather_temp_mean",
    "weather_temp_max",
    "weather_temp_min",
    "weather_temp_range",
    "weather_humidity_mean",
    "weather_humidity_max",
    "weather_rain_daily",
    "weather_radiation_mean",
    "weather_radiation_max",
    "weather_wind_mean",
    "weather_wind_max",
    "weather_high_temp",
    "weather_muggy",
    "weather_heavy_rain",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_calendar_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"trade_date", "trade_interval", "actual_load"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Portfolio input missing columns: {sorted(required - set(frame.columns))}"
        )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame["trade_interval"] = pd.to_numeric(
        frame["trade_interval"], errors="raise"
    ).astype(int)
    frame["actual_load"] = pd.to_numeric(frame["actual_load"], errors="raise")
    if frame.duplicated(["trade_date", "trade_interval"]).any():
        raise ValueError("Portfolio input contains duplicate date/interval keys")
    counts = frame.groupby("trade_date")["trade_interval"].nunique()
    if not counts.eq(96).all():
        raise ValueError("Portfolio input contains incomplete days")
    curves = {
        target_date: group.sort_values("trade_interval")["actual_load"].to_numpy(
            dtype=float
        )
        for target_date, group in frame.groupby("trade_date", sort=True)
    }
    dates = sorted(curves)
    rows = []
    for target_date in dates:
        if target_date < date(2026, 5, 1):
            continue
        sources = select_similar_dates(target_date, dates)
        prediction = np.mean([curves[source] for source in sources], axis=0)
        for offset, (actual, predicted) in enumerate(
            zip(curves[target_date], prediction, strict=True),
            start=1,
        ):
            rows.append(
                {
                    "model": "calendar_similar_day_mean4",
                    "target_date": target_date,
                    "trade_interval": offset,
                    "source_dates": "|".join(
                        source.isoformat() for source in sources
                    ),
                    "day_type": day_type(target_date),
                    "actual_load": float(actual),
                    "predicted_load": float(predicted),
                }
            )
    return pd.DataFrame(rows)


def combine_market_inputs(paths: list[Path], output: Path) -> None:
    frames = [pd.read_csv(path) for path in paths]
    market = pd.concat(frames, ignore_index=True)
    keys = ["trade_date", "trade_interval", "data_name", "data_type"]
    market = market.drop_duplicates(keys, keep="last")
    market.to_csv(output, index=False, compression="gzip")


def attach_weather(frame: pd.DataFrame, weather_input: Path) -> pd.DataFrame:
    weather = read_weather_snapshot(
        weather_input,
        policy=WEATHER_POLICY_PROXY,
    )
    weather_frames = []
    for target_date in sorted(frame["target_date"].unique()):
        features = weather.features_for_day(target_date)
        features.insert(0, "target_date", target_date)
        weather_frames.append(features)
    weather_frame = pd.concat(weather_frames, ignore_index=True).rename(
        columns={"trade_interval": "trade_interval"}
    )
    return frame.merge(
        weather_frame,
        on=["target_date", "trade_interval"],
        how="left",
        validate="one_to_one",
    )


def calendar_candidate(frame: pd.DataFrame, start: date) -> pd.DataFrame:
    result = frame.loc[
        frame["target_date"] >= start,
        [
            "target_date",
            "trade_interval",
            "day_type",
            "actual_load",
            "calendar_prediction",
        ],
    ].copy()
    result.insert(0, "model", "calendar_similar_day_mean4")
    result["predicted_load"] = result["calendar_prediction"]
    result["error"] = result["predicted_load"] - result["actual_load"]
    result["absolute_error"] = result["error"].abs()
    return result


def score_window(
    predictions: pd.DataFrame,
    *,
    window: str,
    start: date,
    end: date,
) -> dict | None:
    selected = predictions.loc[
        predictions["target_date"].between(start, end)
    ].copy()
    if selected.empty:
        return None
    _, metrics = curve_metrics(selected)
    return {
        "model": str(selected["model"].iloc[0]),
        "window": window,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        **metrics,
    }


def daily_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, target_date), group in predictions.groupby(
        ["model", "target_date"], sort=True
    ):
        denominator = group["actual_load"].abs().sum()
        rows.append(
            {
                "model": model,
                "target_date": target_date,
                "wmape_pct": group["absolute_error"].sum() / denominator * 100,
                "bias_pct": group["error"].sum() / denominator * 100,
            }
        )
    return pd.DataFrame(rows)


def correlation_diagnostics(frame: pd.DataFrame) -> dict:
    diagnostics = frame.copy()
    diagnostics["calendar_residual"] = (
        diagnostics["actual_load"] - diagnostics["calendar_prediction"]
    )
    daylight = diagnostics["trade_interval"].between(25, 72)
    midday = diagnostics["trade_interval"].between(41, 60)
    history = diagnostics["target_date"].between(
        date(2026, 5, 1), date(2026, 7, 25)
    )

    def correlation(mask: pd.Series, feature: str) -> float | None:
        selected = diagnostics.loc[mask, ["calendar_residual", feature]].dropna()
        if selected.empty:
            return None
        return float(selected["calendar_residual"].corr(selected[feature]))

    return {
        "historical_daylight_residual_vs_target_da_pv_share": correlation(
            history & daylight,
            "target_da_photovoltaic_share",
        ),
        "historical_midday_residual_vs_target_da_pv_share": correlation(
            history & midday,
            "target_da_photovoltaic_share",
        ),
        "historical_midday_residual_vs_weather_radiation": correlation(
            history & midday,
            "weather_radiation",
        ),
    }


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    calendar = build_calendar_predictions(args.portfolio_input)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        calendar_path = temp_root / "calendar_predictions.csv"
        market_path = temp_root / "combined_market.csv.gz"
        calendar.to_csv(calendar_path, index=False)
        combine_market_inputs(args.market_input, market_path)
        base = build_features(calendar_path, market_path)
        frame, feature_groups = add_expanded_market_features(
            base,
            read_market_matrix(market_path),
        )
    frame = attach_weather(frame, args.weather_input)

    solar_features = tuple(
        dict.fromkeys(
            [
                *feature_groups["curve_context"],
                "target_da_photovoltaic_share",
                "d2_rt_photovoltaic_share",
                "d2_photovoltaic_rt_da_gap_share",
                "target_da_net_load_share",
                "d2_rt_net_load_share",
            ]
        )
    )
    solar_wind_features = tuple(
        dict.fromkeys(
            [
                *solar_features,
                "target_da_wind_share",
                "d2_rt_wind_share",
                "d2_wind_rt_da_gap_share",
            ]
        )
    )
    feature_sets = {
        "frozen_gbdt_core": feature_groups["core"],
        "frozen_gbdt_solar": solar_features,
        "frozen_gbdt_solar_wind": solar_wind_features,
        "frozen_gbdt_physical": feature_groups["physical"],
        "frozen_gbdt_core_weather": (
            *feature_groups["core"],
            *WEATHER_FEATURES,
        ),
        "frozen_gbdt_solar_weather": (
            *solar_features,
            *WEATHER_FEATURES,
        ),
    }

    validation_predictions = {}
    for model, features in feature_sets.items():
        eligible = frame.dropna(subset=list(features)).copy()
        prediction = fit_candidate(
            eligible,
            model_name=model,
            feature_columns=features,
            train_end=args.validation_train_end,
            holdout_start=args.validation_start,
        )
        validation_predictions[model] = prediction.loc[
            prediction["target_date"] <= args.validation_end
        ].copy()
    validation_summary = []
    for prediction in validation_predictions.values():
        row = score_window(
            prediction,
            window="internal_validation",
            start=args.validation_start,
            end=args.validation_end,
        )
        if row:
            validation_summary.append(row)
    validation_summary_frame = pd.DataFrame(validation_summary).sort_values(
        ["wmape_pct", "daily_wmape_p90_pct", "model"]
    )
    selected_model = str(validation_summary_frame.iloc[0]["model"])

    final_predictions = {}
    for model, features in feature_sets.items():
        eligible = frame.dropna(subset=list(features)).copy()
        final_predictions[model] = fit_candidate(
            eligible,
            model_name=model,
            feature_columns=features,
            train_end=args.final_train_end,
            holdout_start=args.holdout_start,
        )
    final_predictions["calendar_similar_day_mean4"] = calendar_candidate(
        frame,
        args.holdout_start,
    )
    predictions = pd.concat(final_predictions.values(), ignore_index=True)

    summary_rows = []
    for prediction in final_predictions.values():
        for window, start, end in (
            (
                "legacy_holdout",
                args.holdout_start,
                args.legacy_holdout_end,
            ),
            (
                "new_diagnostic",
                args.new_diagnostic_start,
                args.new_diagnostic_end,
            ),
        ):
            row = score_window(
                prediction,
                window=window,
                start=start,
                end=end,
            )
            if row:
                summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["window", "wmape_pct", "model"]
    )
    daily = daily_metrics(predictions)
    diagnostics = correlation_diagnostics(frame)

    predictions.to_csv(
        args.output_dir / "predictions_96.csv.gz",
        index=False,
        compression="gzip",
    )
    daily.to_csv(
        args.output_dir / "daily_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_summary_frame.to_csv(
        args.output_dir / "validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        args.output_dir / "summary_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([diagnostics]).to_csv(
        args.output_dir / "diagnostic_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "version": VERSION,
        "province_display": "江苏",
        "province_slug": "jiangsu",
        "portfolio_input": str(args.portfolio_input.resolve()),
        "portfolio_sha256": sha256_file(args.portfolio_input),
        "market_inputs": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for path in args.market_input
        ],
        "weather_input": str(args.weather_input.resolve()),
        "weather_sha256": sha256_file(args.weather_input),
        "market_policy": MARKET_POLICY,
        "weather_policy": WEATHER_POLICY_PROXY,
        "load_label_policy": "SIMULATED_T_PLUS_2_D_MINUS_3",
        "validation_train_end": args.validation_train_end.isoformat(),
        "validation_start": args.validation_start.isoformat(),
        "validation_end": args.validation_end.isoformat(),
        "final_train_end": args.final_train_end.isoformat(),
        "holdout_start": args.holdout_start.isoformat(),
        "legacy_holdout_end": args.legacy_holdout_end.isoformat(),
        "new_diagnostic_start": args.new_diagnostic_start.isoformat(),
        "new_diagnostic_end": args.new_diagnostic_end.isoformat(),
        "selected_on_internal_validation": selected_model,
        "champion_eligible": False,
        "eligibility_reason": (
            "Historical market/weather exports lack trustworthy "
            "issue_at/available_at; new dates also motivated this audit."
        ),
        "pii_output_check": "aggregate-only outputs",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    new_rows = summary.loc[summary["window"].eq("new_diagnostic")].copy()
    selected_new = new_rows.loc[new_rows["model"].eq(selected_model)]
    best_new = new_rows.sort_values("wmape_pct").iloc[0]
    lines = [
        "# 江苏苏润负荷：光伏、气象和新能源候选审计",
        "",
        "## 结论",
        "",
        (
            f"- 内部验证预选模型为 `{selected_model}`；"
            "新增光伏、风电或天气字段未通过预选门禁。"
        ),
        (
            f"- 新增日期诊断窗口最低为 `{best_new['model']}`，"
            f"WMAPE {best_new['wmape_pct']:.4f}%。"
        ),
    ]
    if not selected_new.empty:
        lines.append(
            f"- 预选模型在新增日期窗口WMAPE "
            f"{selected_new.iloc[0]['wmape_pct']:.4f}%。"
        )
    lines.extend(
        [
            (
                "- 历史午间日历残差与目标日DA光伏占比相关系数："
                f"{diagnostics['historical_midday_residual_vs_target_da_pv_share']:.4f}。"
            ),
            (
                "- 历史午间日历残差与辐照度相关系数："
                f"{diagnostics['historical_midday_residual_vs_weather_radiation']:.4f}。"
            ),
            "",
            "## 状态",
            "",
            "- 光伏/辐照度信号真实存在，但当前短样本直接扩列会增加方差。",
            "- 建议保留核心模型，并将光伏建模改为用户分组或白天形状保护层。",
            "- 本次为研究代理，不能据此直接替换生产Champion。",
        ]
    )
    (args.output_dir / "run_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(validation_summary_frame[["model", "wmape_pct"]].to_string(index=False))
    print()
    print(new_rows[["model", "target_days", "wmape_pct"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-input", type=Path, required=True)
    parser.add_argument(
        "--market-input",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--weather-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--validation-train-end",
        type=date.fromisoformat,
        default=date(2026, 5, 29),
    )
    parser.add_argument(
        "--validation-start",
        type=date.fromisoformat,
        default=date(2026, 6, 1),
    )
    parser.add_argument(
        "--validation-end",
        type=date.fromisoformat,
        default=date(2026, 6, 12),
    )
    parser.add_argument(
        "--final-train-end",
        type=date.fromisoformat,
        default=date(2026, 6, 12),
    )
    parser.add_argument(
        "--holdout-start",
        type=date.fromisoformat,
        default=date(2026, 6, 15),
    )
    parser.add_argument(
        "--legacy-holdout-end",
        type=date.fromisoformat,
        default=date(2026, 7, 25),
    )
    parser.add_argument(
        "--new-diagnostic-start",
        type=date.fromisoformat,
        default=date(2026, 7, 26),
    )
    parser.add_argument(
        "--new-diagnostic-end",
        type=date.fromisoformat,
        default=date(2026, 7, 28),
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
