"""Create a shadow D+1 forecast with strict T-2 market and live D weather.

The target-day weather is captured from the configured API and archived with
the run.  The run deliberately ignores target-day DA market fields.  It is a
shadow research artifact and does not publish to the production dashboard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from project.ingest.fetch_jiangsu_weather_raw import (
    CITY_COORDS,
    fetch_variable_with_retry,
)
from project.ingest.fetch_jiangsu_raw import fetch_stage
from project.ingest.fetch_weather_grid_raw import VARIABLE_MAP
from project.research.jiangsu.user_load_forecast.calendar_baseline import (
    select_similar_dates,
)
from project.research.jiangsu.user_load_forecast.market_d2_curve_backtest import (
    model_pipeline,
)
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
)
from project.research.jiangsu.user_load_forecast.strict_operational_weather_backtest import (
    WEATHER_POLICY,
)


VERSION = "surun_strict_operational_weather_shadow_forecast_v1"
MODEL_ID = "strict_gbdt_solar_weather_operational_shadow"
CHINA = ZoneInfo("Asia/Shanghai")
WEATHER_SOURCE = "Ts_weather_1"
WEATHER_VARIABLES = (
    "温度2米",
    "风速10米",
    "相对湿度",
    "辐照度",
    "降雨量",
)
STRICT_RT_COMPONENTS = (
    "SYSTEM_LOAD_POWER",
    "NEW_ENERGY_PHOTOVOLTAIC_POWER",
    "NEW_ENERGY_WIND_POWER",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_target_weather(
    *,
    base_url: str,
    target_date: date,
    output_path: Path,
    retries: int,
) -> dict:
    """Fetch, validate and archive the five weather variables used by the model."""
    requested_at = datetime.now(CHINA)
    merged: pd.DataFrame | None = None
    for variable in WEATHER_VARIABLES:
        part = fetch_variable_with_retry(
            base_url,
            WEATHER_SOURCE,
            variable,
            target_date.isoformat(),
            target_date.isoformat(),
            retries,
            2.0,
        )
        if part.empty:
            raise ValueError(f"Empty target weather response: {variable}")
        value_column = VARIABLE_MAP[variable]
        selected = part[["city", "timestamp", value_column]].copy()
        merged = (
            selected
            if merged is None
            else merged.merge(
                selected,
                on=["city", "timestamp"],
                how="outer",
                validate="one_to_one",
            )
        )
    if merged is None:
        raise ValueError("No target weather returned")
    merged.insert(1, "source", WEATHER_SOURCE)
    expected_columns = [
        "temp_2m",
        "humidity_2m",
        "rain",
        "radiation",
        "wind_10m",
    ]
    for column in expected_columns:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    target_start = pd.Timestamp(target_date)
    target_end = target_start + pd.Timedelta(days=1)
    target_rows = merged.loc[
        (merged["timestamp"] > target_start)
        & (merged["timestamp"] <= target_end)
    ].copy()
    expected_cities = {item.city for item in CITY_COORDS}
    if set(target_rows["city"]) != expected_cities:
        raise ValueError("Target weather does not contain all 13 Jiangsu cities")
    if target_rows["timestamp"].nunique() != 24 or len(target_rows) != 13 * 24:
        raise ValueError(
            "Target weather must contain 13 cities x 24 hour-ending timestamps"
        )
    if target_rows[expected_columns].isna().any().any():
        missing = target_rows[expected_columns].isna().sum().to_dict()
        raise ValueError(f"Target weather contains missing model fields: {missing}")
    target_rows.sort_values(["timestamp", "city"]).to_csv(
        output_path,
        index=False,
        compression="gzip",
    )
    received_at = datetime.now(CHINA)
    return {
        "endpoint": f"{base_url.rstrip('/')}/weather_grid_info",
        "source": WEATHER_SOURCE,
        "variables": list(WEATHER_VARIABLES),
        "requested_at": requested_at.isoformat(),
        "received_at": received_at.isoformat(),
        "rows": int(len(target_rows)),
        "cities": int(target_rows["city"].nunique()),
        "hours": int(target_rows["timestamp"].nunique()),
        "sha256": sha256_file(output_path),
    }


def combine_weather_inputs(
    history_path: Path,
    target_path: Path,
    output_path: Path,
) -> None:
    history = pd.read_csv(history_path)
    target = pd.read_csv(target_path)
    combined = pd.concat([history, target], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(
        ["city", "source", "timestamp"], keep="last"
    )
    combined.to_csv(output_path, index=False, compression="gzip")


def capture_d2_market_rt(
    *,
    base_url: str,
    target_date: date,
    output_path: Path,
) -> dict:
    """Capture the target-minus-two RT fields required by the strict model."""
    market_date = target_date - timedelta(days=2)
    requested_at = datetime.now(CHINA)
    warnings: list[str] = []
    request_args = argparse.Namespace(
        base_url=base_url,
        province="江苏",
        shift_days=1,
        timeout=180,
    )
    frame, stats = fetch_stage(
        request_args,
        0,
        market_date.isoformat(),
        market_date.isoformat(),
        warnings,
    )
    if frame.empty:
        raise ValueError(f"Empty T-2 market RT response for {market_date}")
    required = frame.loc[frame["data_name"].isin(STRICT_RT_COMPONENTS)].copy()
    coverage = required.groupby("data_name")["value"].agg(
        rows="size", non_null="count"
    )
    missing = {
        component: (
            int(coverage.loc[component, "non_null"])
            if component in coverage.index
            else 0
        )
        for component in STRICT_RT_COMPONENTS
        if component not in coverage.index
        or int(coverage.loc[component, "non_null"]) != 96
    }
    if missing:
        raise ValueError(f"Incomplete T-2 strict RT components: {missing}")
    frame.to_csv(output_path, index=False, compression="gzip")
    received_at = datetime.now(CHINA)
    return {
        "endpoint": f"{base_url.rstrip('/')}/data",
        "market_date": market_date.isoformat(),
        "stage": "RT",
        "requested_at": requested_at.isoformat(),
        "received_at": received_at.isoformat(),
        "rows": int(len(frame)),
        "strict_component_non_null": {
            component: int(coverage.loc[component, "non_null"])
            for component in STRICT_RT_COMPONENTS
        },
        "normalization_stats": stats,
        "warnings": warnings,
        "sha256": sha256_file(output_path),
    }


def dashboard_curve(path: Path | None, target_date: date) -> tuple[str, np.ndarray] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    preferred = (
        "operational_deadline_fusion",
        "operational_calendar_d3",
    )
    day = next(
        (item for item in payload.get("days", []) if item.get("date") == target_date.isoformat()),
        None,
    )
    if not day:
        return None
    predictions = day.get("predictions", {})
    for model_id in preferred:
        values = predictions.get(model_id)
        if isinstance(values, list) and len(values) == 96:
            return model_id, np.asarray(values, dtype=float)
    return None


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    target_weather_path = args.output_dir / "target_weather_snapshot.csv.gz"
    weather_capture = capture_target_weather(
        base_url=args.weather_api_url,
        target_date=args.target_date,
        output_path=target_weather_path,
        retries=args.weather_retries,
    )
    d2_market_path = args.output_dir / "d2_market_rt_snapshot.csv.gz"
    market_capture = capture_d2_market_rt(
        base_url=args.market_api_url,
        target_date=args.target_date,
        output_path=d2_market_path,
    )
    curves = read_portfolio_curves(args.portfolio_input)
    if args.target_date in curves:
        raise ValueError("Target date already has actual load; use a backtest instead")
    latest_actual = max(curves)
    label_cutoff = min(latest_actual, args.target_date - timedelta(days=3))
    source_dates = select_similar_dates(args.target_date, sorted(curves))
    calendar_prediction = np.mean(
        [curves[source] for source in source_dates], axis=0
    )
    augmented_curves = {**curves, args.target_date: calendar_prediction.copy()}

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        market_path = temp_root / "combined_market.csv.gz"
        weather_path = temp_root / "combined_weather.csv.gz"
        combine_market_inputs([*args.market_input, d2_market_path], market_path)
        combine_weather_inputs(args.weather_history_input, target_weather_path, weather_path)
        market_matrix = read_market_matrix(market_path)
        strict_frame, strict_groups = prepare_feature_frame(
            augmented_curves,
            component="aggregate_strict_operational_weather_shadow",
            market_path=market_path,
            market_matrix=market_matrix,
            weather_input=weather_path,
            temp_root=temp_root,
            visibility_policy="strict_submission_asof",
            strict_predictor_weather_input=None,
        )
        frame = attach_weather(strict_frame, weather_path)

    strict_sets = feature_sets(
        strict_groups, visibility_policy="strict_submission_asof"
    )
    features = (*strict_sets["solar"], *WEATHER_FEATURES)
    forbidden = [column for column in features if column.startswith("target_da_")]
    if forbidden:
        raise ValueError(f"Target-day DA leaked into shadow features: {forbidden}")
    eligible = frame.dropna(subset=list(features)).copy()
    train = eligible["target_date"] <= label_cutoff
    target = eligible["target_date"].eq(args.target_date)
    if train.sum() == 0 or target.sum() != 96:
        raise ValueError(
            f"Invalid train/target rows: train={int(train.sum())}, target={int(target.sum())}"
        )
    pipeline = model_pipeline(features)
    pipeline.fit(
        eligible.loc[train, list(features)],
        eligible.loc[train, "residual_ratio"],
    )
    residual = np.clip(
        pipeline.predict(eligible.loc[target, list(features)]),
        -0.35,
        0.35,
    )
    target_frame = eligible.loc[target].sort_values("trade_interval").copy()
    prediction = target_frame["calendar_prediction"].to_numpy() * (1.0 + residual)
    output = pd.DataFrame(
        {
            "model": MODEL_ID,
            "target_date": args.target_date.isoformat(),
            "trade_interval": target_frame["trade_interval"].astype(int),
            "predicted_load": np.maximum(prediction, 0.0),
            "calendar_prediction": target_frame["calendar_prediction"].to_numpy(),
            "residual_ratio": residual,
            "day_type": target_frame["day_type"].to_numpy(),
        }
    )
    output.to_csv(
        args.output_dir / "predictions_96.csv.gz",
        index=False,
        compression="gzip",
    )

    comparison = dashboard_curve(args.comparison_dashboard, args.target_date)
    comparison_summary = None
    if comparison:
        comparison_model, comparison_values = comparison
        candidate_values = output["predicted_load"].to_numpy(dtype=float)
        denominator = np.abs(comparison_values).sum()
        comparison_summary = {
            "model": comparison_model,
            "input": str(args.comparison_dashboard.resolve()),
            "mean_absolute_point_difference_pct": float(
                np.abs(candidate_values - comparison_values).sum() / denominator * 100
            ),
            "daily_energy_difference_pct": float(
                (candidate_values.sum() - comparison_values.sum()) / denominator * 100
            ),
            "candidate_peak_interval": int(np.argmax(candidate_values) + 1),
            "comparison_peak_interval": int(np.argmax(comparison_values) + 1),
        }

    metadata = {
        "version": VERSION,
        "status": "SHADOW_ONLY_NOT_PUBLISHED",
        "model": MODEL_ID,
        "target_date": args.target_date.isoformat(),
        "generated_at": datetime.now(CHINA).isoformat(),
        "latest_actual_date": latest_actual.isoformat(),
        "label_cutoff": label_cutoff.isoformat(),
        "normal_label_cutoff": (args.target_date - timedelta(days=3)).isoformat(),
        "label_age_days": (args.target_date - label_cutoff).days,
        "calendar_source_dates": [item.isoformat() for item in source_dates],
        "market_policy": STRICT_MARKET_POLICY,
        "weather_policy": WEATHER_POLICY,
        "target_day_da_feature_check": "PASS_NO_TARGET_DAY_DA",
        "weather_capture": weather_capture,
        "d2_market_capture": market_capture,
        "portfolio_input": str(args.portfolio_input.resolve()),
        "portfolio_sha256": sha256_file(args.portfolio_input),
        "market_inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.market_input
        ],
        "weather_history_input": str(args.weather_history_input.resolve()),
        "weather_history_sha256": sha256_file(args.weather_history_input),
        "feature_columns": list(features),
        "training_rows": int(train.sum()),
        "comparison": comparison_summary,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 江苏用户负荷运营气象影子预测",
        "",
        f"- 目标日：{args.target_date.isoformat()}。",
        f"- 模型：`{MODEL_ID}`。",
        f"- 最新实际负荷：{latest_actual.isoformat()}；训练标签截止：{label_cutoff.isoformat()}。",
        "- 市场特征最大日期为目标日T-2；目标日DA完全禁用。",
        f"- D日天气抓取完成：{weather_capture['cities']}市 × {weather_capture['hours']}小时。",
        f"- T-2市场RT抓取完成：{market_capture['market_date']}严格字段均为96点。",
        "- 状态：仅影子观察，未发布到生产看板。",
    ]
    if comparison_summary:
        lines.extend(
            [
                "",
                "## 与当前兜底曲线对比",
                "",
                f"- 对照模型：`{comparison_summary['model']}`。",
                (
                    "- 逐点绝对差异："
                    f"{comparison_summary['mean_absolute_point_difference_pct']:.2f}%。"
                ),
                (
                    "- 日电量差异："
                    f"{comparison_summary['daily_energy_difference_pct']:+.2f}%。"
                ),
                "- 目标日实际负荷尚未发布，因此当前不能计算预测误差。",
            ]
        )
    (args.output_dir / "run_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-input", type=Path, required=True)
    parser.add_argument("--market-input", type=Path, action="append", required=True)
    parser.add_argument("--weather-history-input", type=Path, required=True)
    parser.add_argument("--weather-api-url", default="http://localhost:8000")
    parser.add_argument("--market-api-url", default="http://localhost:8000")
    parser.add_argument("--weather-retries", type=int, default=3)
    parser.add_argument("--target-date", type=date.fromisoformat, required=True)
    parser.add_argument("--comparison-dashboard", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
