"""Research expanded Jiangsu market features and level/shape load models.

The user-load label policy remains D-3.  Target-day market inputs are DA only;
D-2 market inputs may use DA and RT.  Because the historical market export
does not preserve publication timestamps, every result from this entrypoint is
a research proxy and is not eligible for production champion promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from project.research.jiangsu.user_load_forecast.fusion import (
    nonnegative_weights,
)
from project.research.jiangsu.user_load_forecast.market_d2_curve_backtest import (
    D2_FEATURES,
    add_external_candidates,
    build_features,
    curve_metrics,
    fit_candidate,
    read_external_candidate,
)
from project.research.jiangsu.user_load_forecast.model_backtest import (
    prepare_output_dir,
)


VERSION = "surun_market_level_shape_backtest_v1"
MARKET_POLICY = "RESEARCH_PROXY_UNVERIFIED_PUBLICATION_TIME"

TOTAL_COMPONENTS = {
    "wind": "NEW_ENERGY_WIND_POWER",
    "photovoltaic": "NEW_ENERGY_PHOTOVOLTAIC_POWER",
    "call_wire": "CALL_WIRE_POWER_huadong",
    "non_market": "NON_MARKET_UNIT_POWER",
    "coal": "COAL_POWER",
    "storage": "STORAGE_POWER",
}
REGIONAL_COMPONENTS = {
    "wind": (
        "NEW_ENERGY_AREA_WIND_POWER_north",
        "NEW_ENERGY_AREA_WIND_POWER_south",
    ),
    "photovoltaic": (
        "NEW_ENERGY_AREA_PHOTOVOLTAIC_POWER_north",
        "NEW_ENERGY_AREA_PHOTOVOLTAIC_POWER_south",
    ),
    "non_market": (
        "NON_MARKET_AREA_UNIT_POWER_north",
        "NON_MARKET_AREA_UNIT_POWER_south",
    ),
    "coal": ("COAL_POWER_north", "COAL_POWER_south"),
    "storage": ("STORAGE_POWER_north", "STORAGE_POWER_south"),
}
DA_ONLY_COMPONENTS = {
    "positive_reserve": "SPARE_DEMAND_POSITIVE_POWER",
    "negative_reserve": "SPARE_DEMAND_NEGATIVE_POWER",
}
PRICE_COMPONENT = "AREA_CLEARING_PRICE"
SYSTEM_COMPONENT = "SYSTEM_LOAD_POWER"

MARKET_NAMES = (
    SYSTEM_COMPONENT,
    PRICE_COMPONENT,
    *TOTAL_COMPONENTS.values(),
    *DA_ONLY_COMPONENTS.values(),
    *(
        item
        for pair in REGIONAL_COMPONENTS.values()
        for item in pair
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_market_matrix(path: Path) -> pd.DataFrame:
    """Read only the selected, fully covered Jiangsu market fields."""
    parts = []
    usecols = ["trade_date", "trade_interval", "data_name", "data_type", "value"]
    selected_names = set(MARKET_NAMES)
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=300_000):
        selected = chunk.loc[
            chunk["data_name"].isin(selected_names)
            & chunk["data_type"].isin(["DA", "RT"])
        ].copy()
        if not selected.empty:
            parts.append(selected)
    if not parts:
        raise ValueError("No selected Jiangsu market rows found")
    market = pd.concat(parts, ignore_index=True)
    market["trade_date"] = pd.to_datetime(market["trade_date"]).dt.date
    duplicate = market.duplicated(
        ["trade_date", "trade_interval", "data_name", "data_type"],
        keep=False,
    )
    if duplicate.any():
        keys = market.loc[
            duplicate,
            ["trade_date", "trade_interval", "data_name", "data_type"],
        ].head(3)
        raise ValueError(f"Duplicate market keys: {keys.to_dict('records')}")
    wide = market.pivot(
        index=["trade_date", "trade_interval"],
        columns=["data_name", "data_type"],
        values="value",
    ).sort_index()
    wide.columns = [
        f"{data_name}__{data_type}"
        for data_name, data_type in wide.columns
    ]
    return wide


def _market_values(
    frame: pd.DataFrame,
    market: pd.DataFrame,
    *,
    day_offset: int,
    data_name: str,
    data_type: str,
) -> np.ndarray:
    dates = frame["target_date"].map(
        lambda value: value - timedelta(days=day_offset)
    )
    keys = pd.MultiIndex.from_arrays(
        [dates, frame["trade_interval"].astype(int)],
        names=["trade_date", "trade_interval"],
    )
    column = f"{data_name}__{data_type}"
    if column not in market.columns:
        raise ValueError(f"Market column is absent: {column}")
    values = market[column].reindex(keys).to_numpy(dtype=float)
    missing = ~np.isfinite(values)
    if missing.any():
        sample = [
            (str(keys[index][0]), int(keys[index][1]), column)
            for index in np.flatnonzero(missing)[:3]
        ]
        raise ValueError(f"Missing required market values: {sample}")
    return values


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=np.abs(denominator) > 1e-9,
    )
    return result


def add_expanded_market_features(
    frame: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    """Add physical, regional and price feature groups without future RT."""
    result = frame.copy()
    target_system_da = _market_values(
        result,
        market,
        day_offset=0,
        data_name=SYSTEM_COMPONENT,
        data_type="DA",
    )
    d2_system_da = _market_values(
        result,
        market,
        day_offset=2,
        data_name=SYSTEM_COMPONENT,
        data_type="DA",
    )
    d2_system_rt = _market_values(
        result,
        market,
        day_offset=2,
        data_name=SYSTEM_COMPONENT,
        data_type="RT",
    )

    physical_columns = []
    for slug, data_name in TOTAL_COMPONENTS.items():
        target_da = _market_values(
            result,
            market,
            day_offset=0,
            data_name=data_name,
            data_type="DA",
        )
        d2_da = _market_values(
            result,
            market,
            day_offset=2,
            data_name=data_name,
            data_type="DA",
        )
        d2_rt = _market_values(
            result,
            market,
            day_offset=2,
            data_name=data_name,
            data_type="RT",
        )
        target_share = f"target_da_{slug}_share"
        d2_share = f"d2_rt_{slug}_share"
        d2_gap = f"d2_{slug}_rt_da_gap_share"
        result[target_share] = _safe_ratio(target_da, target_system_da)
        result[d2_share] = _safe_ratio(d2_rt, d2_system_rt)
        result[d2_gap] = _safe_ratio(d2_rt - d2_da, d2_system_rt)
        physical_columns.extend((target_share, d2_share, d2_gap))

    for slug, data_name in DA_ONLY_COMPONENTS.items():
        target_da = _market_values(
            result,
            market,
            day_offset=0,
            data_name=data_name,
            data_type="DA",
        )
        column = f"target_da_{slug}_share"
        result[column] = _safe_ratio(target_da, target_system_da)
        physical_columns.append(column)

    target_wind = _market_values(
        result,
        market,
        day_offset=0,
        data_name=TOTAL_COMPONENTS["wind"],
        data_type="DA",
    )
    target_pv = _market_values(
        result,
        market,
        day_offset=0,
        data_name=TOTAL_COMPONENTS["photovoltaic"],
        data_type="DA",
    )
    d2_wind = _market_values(
        result,
        market,
        day_offset=2,
        data_name=TOTAL_COMPONENTS["wind"],
        data_type="RT",
    )
    d2_pv = _market_values(
        result,
        market,
        day_offset=2,
        data_name=TOTAL_COMPONENTS["photovoltaic"],
        data_type="RT",
    )
    result["target_da_net_load_share"] = _safe_ratio(
        target_system_da - target_wind - target_pv,
        target_system_da,
    )
    result["d2_rt_net_load_share"] = _safe_ratio(
        d2_system_rt - d2_wind - d2_pv,
        d2_system_rt,
    )
    physical_columns.extend(
        ("target_da_net_load_share", "d2_rt_net_load_share")
    )

    regional_columns = []
    for slug, (north_name, south_name) in REGIONAL_COMPONENTS.items():
        target_north = _market_values(
            result,
            market,
            day_offset=0,
            data_name=north_name,
            data_type="DA",
        )
        target_south = _market_values(
            result,
            market,
            day_offset=0,
            data_name=south_name,
            data_type="DA",
        )
        d2_north_da = _market_values(
            result,
            market,
            day_offset=2,
            data_name=north_name,
            data_type="DA",
        )
        d2_south_da = _market_values(
            result,
            market,
            day_offset=2,
            data_name=south_name,
            data_type="DA",
        )
        d2_north_rt = _market_values(
            result,
            market,
            day_offset=2,
            data_name=north_name,
            data_type="RT",
        )
        d2_south_rt = _market_values(
            result,
            market,
            day_offset=2,
            data_name=south_name,
            data_type="RT",
        )
        target_column = f"target_da_{slug}_south_minus_north_share"
        d2_column = f"d2_rt_{slug}_south_minus_north_share"
        gap_column = f"d2_{slug}_regional_rt_da_gap_share"
        result[target_column] = _safe_ratio(
            target_south - target_north,
            target_system_da,
        )
        result[d2_column] = _safe_ratio(
            d2_south_rt - d2_north_rt,
            d2_system_rt,
        )
        result[gap_column] = _safe_ratio(
            (d2_south_rt - d2_north_rt)
            - (d2_south_da - d2_north_da),
            d2_system_rt,
        )
        regional_columns.extend((target_column, d2_column, gap_column))

    target_price_da = _market_values(
        result,
        market,
        day_offset=0,
        data_name=PRICE_COMPONENT,
        data_type="DA",
    )
    d2_price_da = _market_values(
        result,
        market,
        day_offset=2,
        data_name=PRICE_COMPONENT,
        data_type="DA",
    )
    d2_price_rt = _market_values(
        result,
        market,
        day_offset=2,
        data_name=PRICE_COMPONENT,
        data_type="RT",
    )
    result["target_da_price"] = target_price_da
    result["d2_rt_price"] = d2_price_rt
    result["d2_price_rt_da_spread"] = d2_price_rt - d2_price_da
    price_columns = (
        "target_da_price",
        "d2_rt_price",
        "d2_price_rt_da_spread",
    )

    result["calendar_daily_mean"] = result.groupby(
        "target_date"
    )["calendar_prediction"].transform("mean")
    result["calendar_shape_ratio"] = (
        result["calendar_prediction"] / result["calendar_daily_mean"]
    )
    day_of_year = result["target_date"].map(
        lambda value: value.timetuple().tm_yday
    )
    angle = 2.0 * np.pi * day_of_year / 365.25
    result["day_of_year_sin"] = np.sin(angle)
    result["day_of_year_cos"] = np.cos(angle)
    context_columns = (
        "calendar_shape_ratio",
        "day_of_year_sin",
        "day_of_year_cos",
    )

    feature_groups = {
        "core": tuple(D2_FEATURES),
        "curve_context": tuple(D2_FEATURES) + context_columns,
        "physical": (
            tuple(D2_FEATURES)
            + context_columns
            + tuple(physical_columns)
        ),
        "regional": (
            tuple(D2_FEATURES)
            + context_columns
            + tuple(physical_columns)
            + tuple(regional_columns)
        ),
        "price": (
            tuple(D2_FEATURES)
            + context_columns
            + tuple(physical_columns)
            + tuple(regional_columns)
            + price_columns
        ),
    }
    all_added = sorted(
        set(feature_groups["price"]) - set(D2_FEATURES)
    )
    bad = ~np.isfinite(result[all_added].to_numpy(dtype=float))
    if bad.any():
        row_index, column_index = np.argwhere(bad)[0]
        raise ValueError(
            "Expanded feature is not finite: "
            f"{result.iloc[row_index]['target_date']}/"
            f"{result.iloc[row_index]['trade_interval']}/"
            f"{all_added[column_index]}"
        )
    return result, feature_groups


def ridge_pipeline(feature_columns: tuple[str, ...]) -> Pipeline:
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
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=20.0)),
        ]
    )


def shape_pipeline(feature_columns: tuple[str, ...]) -> Pipeline:
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
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingRegressor(
                    loss="huber",
                    n_estimators=120,
                    learning_rate=0.04,
                    max_depth=2,
                    min_samples_leaf=12,
                    random_state=42,
                ),
            ),
        ]
    )


def build_daily_frame(
    frame: pd.DataFrame,
    point_features: tuple[str, ...],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Collapse target-known point features into one row per date."""
    daily_rows = []
    excluded = {
        "day_type",
        "trade_interval",
        "interval_sin",
        "interval_cos",
        "log_calendar_prediction",
        "day_of_year_sin",
        "day_of_year_cos",
    }
    aggregatable = [
        column for column in point_features if column not in excluded
    ]
    for target_date, group in frame.groupby("target_date", sort=True):
        group = group.sort_values("trade_interval")
        actual_total = float(group["actual_load"].sum())
        calendar_total = float(group["calendar_prediction"].sum())
        calendar = group["calendar_prediction"].to_numpy(dtype=float)
        row = {
            "target_date": target_date,
            "day_type": group["day_type"].iloc[0],
            "log_calendar_energy": np.log1p(calendar_total),
            "calendar_shape_std": float(calendar.std() / calendar.mean()),
            "calendar_peak_to_mean": float(calendar.max() / calendar.mean()),
            "calendar_valley_to_mean": float(calendar.min() / calendar.mean()),
            "day_of_year_sin": float(group["day_of_year_sin"].iloc[0]),
            "day_of_year_cos": float(group["day_of_year_cos"].iloc[0]),
            "actual_energy": actual_total,
            "calendar_energy": calendar_total,
            "energy_residual_ratio": (
                actual_total - calendar_total
            )
            / calendar_total,
        }
        for column in aggregatable:
            values = group[column].to_numpy(dtype=float)
            row[f"{column}__mean"] = float(values.mean())
            row[f"{column}__std"] = float(values.std())
        for column in (
            "target_da_vs_source_rt",
            "d2_rt_da_bias",
            "target_da_vs_d2_rt",
        ):
            if column in group.columns:
                values = group[column].to_numpy(dtype=float)
                row[f"{column}__min"] = float(values.min())
                row[f"{column}__max"] = float(values.max())
        daily_rows.append(row)
    daily = pd.DataFrame(daily_rows)
    feature_columns = tuple(
        column
        for column in daily.columns
        if column
        not in {
            "target_date",
            "actual_energy",
            "calendar_energy",
            "energy_residual_ratio",
        }
    )
    return daily, feature_columns


def fit_two_stage_candidate(
    frame: pd.DataFrame,
    *,
    model_name: str,
    point_features: tuple[str, ...],
    train_end: date,
    holdout_start: date,
) -> pd.DataFrame:
    train = frame["target_date"] <= train_end
    holdout = frame["target_date"] >= holdout_start
    daily, daily_features = build_daily_frame(frame, point_features)
    daily_train = daily["target_date"] <= train_end
    daily_holdout = daily["target_date"] >= holdout_start

    level_model = ridge_pipeline(daily_features)
    level_model.fit(
        daily.loc[daily_train, list(daily_features)],
        daily.loc[daily_train, "energy_residual_ratio"],
    )
    level_residual = np.clip(
        level_model.predict(
            daily.loc[daily_holdout, list(daily_features)]
        ),
        -0.35,
        0.35,
    )
    predicted_energy = (
        daily.loc[daily_holdout, "calendar_energy"].to_numpy(dtype=float)
        * (1.0 + level_residual)
    )
    energy_by_date = dict(
        zip(
            daily.loc[daily_holdout, "target_date"],
            predicted_energy,
            strict=True,
        )
    )

    actual_total = frame.groupby("target_date")["actual_load"].transform("sum")
    calendar_total = frame.groupby(
        "target_date"
    )["calendar_prediction"].transform("sum")
    actual_share = frame["actual_load"] / actual_total
    calendar_share = frame["calendar_prediction"] / calendar_total
    log_shape_ratio = np.log(
        np.clip(actual_share, 1e-12, None)
        / np.clip(calendar_share, 1e-12, None)
    )
    shape_model = shape_pipeline(point_features)
    shape_model.fit(
        frame.loc[train, list(point_features)],
        log_shape_ratio.loc[train],
    )
    predicted_log_shape = np.clip(
        shape_model.predict(frame.loc[holdout, list(point_features)]),
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
    predictions["raw_shape"] = (
        calendar_share.loc[holdout].to_numpy(dtype=float)
        * np.exp(predicted_log_shape)
    )
    predictions["shape_sum"] = predictions.groupby(
        "target_date"
    )["raw_shape"].transform("sum")
    predictions["normalized_shape"] = (
        predictions["raw_shape"] / predictions["shape_sum"]
    )
    predictions["predicted_daily_energy"] = predictions["target_date"].map(
        energy_by_date
    )
    predictions["predicted_load"] = (
        predictions["normalized_shape"]
        * predictions["predicted_daily_energy"]
    )
    predictions["error"] = (
        predictions["predicted_load"] - predictions["actual_load"]
    )
    predictions["absolute_error"] = predictions["error"].abs()
    return predictions


def merge_named_prediction(
    frame: pd.DataFrame,
    prediction: pd.DataFrame,
    *,
    column: str,
) -> pd.DataFrame:
    candidate = prediction[
        ["target_date", "trade_interval", "actual_load", "predicted_load"]
    ].rename(
        columns={
            "actual_load": f"actual__{column}",
            "predicted_load": column,
        }
    )
    merged = frame.merge(
        candidate,
        on=["target_date", "trade_interval"],
        validate="one_to_one",
    )
    if not np.allclose(merged["actual_load"], merged[f"actual__{column}"]):
        raise ValueError(f"Actual values disagree for {column}")
    return merged.drop(columns=f"actual__{column}")


def learn_augmented_fusion(
    *,
    core: pd.DataFrame,
    direct_physical: pd.DataFrame,
    two_stage_physical: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, float]]:
    merged = add_external_candidates(core, args)
    merged = merge_named_prediction(
        merged,
        direct_physical,
        column="direct_physical",
    )
    merged = merge_named_prediction(
        merged,
        two_stage_physical,
        column="two_stage_physical",
    )
    candidates = [
        "d2",
        "direct_physical",
        "two_stage_physical",
        "xgboost",
        "calendar",
        "strict",
        "industry",
    ]
    weights, success = nonnegative_weights(
        merged[candidates].to_numpy(dtype=float),
        merged["actual_load"].to_numpy(dtype=float),
    )
    if not success:
        raise ValueError("Augmented fusion optimizer failed")
    weight_map = {
        candidate: float(weight)
        for candidate, weight in zip(candidates, weights, strict=True)
    }
    return merged, weight_map


def apply_augmented_fusion(
    *,
    core: pd.DataFrame,
    direct_physical: pd.DataFrame,
    two_stage_physical: pd.DataFrame,
    weights: dict[str, float],
    args: argparse.Namespace,
) -> pd.DataFrame:
    merged = add_external_candidates(core, args)
    merged = merge_named_prediction(
        merged,
        direct_physical,
        column="direct_physical",
    )
    merged = merge_named_prediction(
        merged,
        two_stage_physical,
        column="two_stage_physical",
    )
    candidates = list(weights)
    weight_array = np.array([weights[name] for name in candidates])
    fused = core.copy()
    fused["model"] = "augmented_robust_fusion"
    fused["predicted_load"] = (
        merged[candidates].to_numpy(dtype=float) @ weight_array
    )
    fused["error"] = fused["predicted_load"] - fused["actual_load"]
    fused["absolute_error"] = fused["error"].abs()
    return fused


def read_benchmark(path: Path) -> pd.DataFrame:
    benchmark = read_external_candidate(
        path,
        model="gbdt_d2_guarded_fusion",
        candidate="predicted_load",
    ).rename(columns={"actual__predicted_load": "actual_load"})
    benchmark.insert(0, "model", "benchmark_d2_robust_fusion")
    benchmark["error"] = (
        benchmark["predicted_load"] - benchmark["actual_load"]
    )
    benchmark["absolute_error"] = benchmark["error"].abs()
    return benchmark


def summarize_candidates(
    predictions: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    daily_frames = []
    for model_name, frame in predictions.items():
        daily, summary = curve_metrics(frame)
        daily.insert(0, "model", model_name)
        daily_frames.append(daily)
        summary_rows.append({"model": model_name, **summary})
    return pd.DataFrame(summary_rows), pd.concat(daily_frames, ignore_index=True)


def _summary_markdown(
    *,
    selected_model: str,
    validation_summary: pd.DataFrame,
    holdout_summary: pd.DataFrame,
    fusion_weights: dict[str, float],
) -> str:
    selected_row = holdout_summary.loc[
        holdout_summary["model"].eq(selected_model)
    ].iloc[0]
    benchmark_row = holdout_summary.loc[
        holdout_summary["model"].eq("benchmark_d2_robust_fusion")
    ].iloc[0]
    best_row = holdout_summary.sort_values("wmape_pct").iloc[0]
    lines = [
        "# 江苏苏润负荷：扩展市场特征与水平/形状模型",
        "",
        "## 口径",
        "",
        "- 用户负荷标签截止D-3；市场RT最多使用到D-2。",
        "- 最终模型训练截止2026-06-12，留出从2026-06-15开始。",
        "- 历史市场导出缺少issue_at/available_at，全部结果仅为研究代理。",
        "",
        "## 选择与结果",
        "",
        f"- 内部验证预先选择：`{selected_model}`。",
        (
            f"- 该模型留出WMAPE {selected_row['wmape_pct']:.4f}%，"
            f"P90 {selected_row['daily_wmape_p90_pct']:.4f}%，"
            f"形状WMAPE {selected_row['mean_shape_wmape_pct']:.4f}%。"
        ),
        (
            "- 原D-2稳健融合基准："
            f"{benchmark_row['wmape_pct']:.4f}%。"
        ),
        (
            f"- 留出最低结果：`{best_row['model']}`，"
            f"WMAPE {best_row['wmape_pct']:.4f}%。"
        ),
        "",
        "## 增强融合权重",
        "",
    ]
    for name, weight in fusion_weights.items():
        lines.append(f"- `{name}`：{weight:.6f}")
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "模型选择只使用2026-06-01至06-12内部验证；最终41天窗口用于一次性",
            "样本外核验。报告仍列出全部消融结果，但不允许根据留出排名反向调参。",
            "",
            "完整数值见`validation_summary.csv`和`holdout_summary.csv`。",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    if args.guard_validation_start < args.guard_train_end + timedelta(days=3):
        raise ValueError("Internal validation violates the D-3 label embargo")
    if args.holdout_start < args.train_end + timedelta(days=3):
        raise ValueError("Final holdout violates the D-3 label embargo")

    base = build_features(args.calendar_prediction_file, args.market_input)
    market = read_market_matrix(args.market_input)
    frame, feature_groups = add_expanded_market_features(base, market)

    validation_predictions: dict[str, pd.DataFrame] = {}
    for group_name, features in feature_groups.items():
        direct_name = f"direct_gbdt__{group_name}"
        direct = fit_candidate(
            frame,
            model_name=direct_name,
            feature_columns=features,
            train_end=args.guard_train_end,
            holdout_start=args.guard_validation_start,
        )
        validation_predictions[direct_name] = direct.loc[
            direct["target_date"] <= args.train_end
        ].copy()

        two_stage_name = f"two_stage_ridge_gbdt__{group_name}"
        two_stage = fit_two_stage_candidate(
            frame,
            model_name=two_stage_name,
            point_features=features,
            train_end=args.guard_train_end,
            holdout_start=args.guard_validation_start,
        )
        validation_predictions[two_stage_name] = two_stage.loc[
            two_stage["target_date"] <= args.train_end
        ].copy()

    validation_summary, validation_daily = summarize_candidates(
        validation_predictions
    )
    selected_model = str(
        validation_summary.sort_values(
            ["wmape_pct", "daily_wmape_p90_pct", "model"]
        ).iloc[0]["model"]
    )

    _, fusion_weights = learn_augmented_fusion(
        core=validation_predictions["direct_gbdt__core"],
        direct_physical=validation_predictions["direct_gbdt__physical"],
        two_stage_physical=validation_predictions[
            "two_stage_ridge_gbdt__physical"
        ],
        args=args,
    )

    holdout_predictions: dict[str, pd.DataFrame] = {}
    for group_name, features in feature_groups.items():
        direct_name = f"direct_gbdt__{group_name}"
        holdout_predictions[direct_name] = fit_candidate(
            frame,
            model_name=direct_name,
            feature_columns=features,
            train_end=args.train_end,
            holdout_start=args.holdout_start,
        )
        two_stage_name = f"two_stage_ridge_gbdt__{group_name}"
        holdout_predictions[two_stage_name] = fit_two_stage_candidate(
            frame,
            model_name=two_stage_name,
            point_features=features,
            train_end=args.train_end,
            holdout_start=args.holdout_start,
        )

    holdout_predictions["augmented_robust_fusion"] = (
        apply_augmented_fusion(
            core=holdout_predictions["direct_gbdt__core"],
            direct_physical=holdout_predictions["direct_gbdt__physical"],
            two_stage_physical=holdout_predictions[
                "two_stage_ridge_gbdt__physical"
            ],
            weights=fusion_weights,
            args=args,
        )
    )
    benchmark = read_benchmark(args.benchmark_prediction_file)
    benchmark = benchmark.loc[
        benchmark["target_date"] >= args.holdout_start
    ].copy()
    holdout_predictions["benchmark_d2_robust_fusion"] = benchmark

    holdout_summary, holdout_daily = summarize_candidates(
        holdout_predictions
    )
    holdout_summary["selected_on_internal_validation"] = (
        holdout_summary["model"].eq(selected_model)
    )
    holdout_summary["research_proxy_only"] = True

    pd.concat(
        holdout_predictions.values(),
        ignore_index=True,
        sort=False,
    ).to_csv(
        args.output_dir / "predictions_96.csv.gz",
        index=False,
        compression="gzip",
    )
    validation_summary.to_csv(
        args.output_dir / "validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_daily.to_csv(
        args.output_dir / "validation_daily_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    holdout_summary.to_csv(
        args.output_dir / "holdout_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    holdout_daily.to_csv(
        args.output_dir / "holdout_daily_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    feature_rows = [
        {
            "feature_group": group_name,
            "feature_count": len(features),
            "features": "|".join(features),
        }
        for group_name, features in feature_groups.items()
    ]
    pd.DataFrame(feature_rows).to_csv(
        args.output_dir / "feature_groups.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "version": VERSION,
        "province_display": "江苏",
        "province_slug": "jiangsu",
        "raw_schema": "file_snapshot_only",
        "train_end": args.train_end.isoformat(),
        "holdout_start": args.holdout_start.isoformat(),
        "holdout_end": max(frame["target_date"]).isoformat(),
        "guard_train_end": args.guard_train_end.isoformat(),
        "guard_validation_start": args.guard_validation_start.isoformat(),
        "guard_validation_end": args.train_end.isoformat(),
        "user_load_label_policy": "labels and source load through D-3",
        "market_rt_visibility": "D-2 RT assumed visible at D-1 submission",
        "target_market_visibility": "target-day DA assumed visible at D-1",
        "market_policy": MARKET_POLICY,
        "champion_eligible": False,
        "eligibility_reason": (
            "market export lacks historical issue_at/available_at evidence"
        ),
        "selected_model_on_internal_validation": selected_model,
        "selection_metric": "validation WMAPE; P90 and name break ties",
        "augmented_fusion_weights": fusion_weights,
        "feature_groups": {
            name: list(columns)
            for name, columns in feature_groups.items()
        },
        "calendar_prediction_sha256": sha256_file(
            args.calendar_prediction_file
        ),
        "market_input_sha256": sha256_file(args.market_input),
        "benchmark_prediction_sha256": sha256_file(
            args.benchmark_prediction_file
        ),
        "external_prediction_sha256": {
            "xgboost": sha256_file(args.xgboost_prediction_file),
            "strict": sha256_file(args.strict_prediction_file),
            "industry": sha256_file(args.industry_prediction_file),
        },
        "pii_output_check": "aggregate-only outputs",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "run_summary.md").write_text(
        _summary_markdown(
            selected_model=selected_model,
            validation_summary=validation_summary,
            holdout_summary=holdout_summary,
            fusion_weights=fusion_weights,
        ),
        encoding="utf-8",
    )
    print(
        holdout_summary.sort_values("wmape_pct").to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar-prediction-file", type=Path, required=True)
    parser.add_argument("--market-input", type=Path, required=True)
    parser.add_argument("--benchmark-prediction-file", type=Path, required=True)
    parser.add_argument("--xgboost-prediction-file", type=Path, required=True)
    parser.add_argument("--strict-prediction-file", type=Path, required=True)
    parser.add_argument("--industry-prediction-file", type=Path, required=True)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
