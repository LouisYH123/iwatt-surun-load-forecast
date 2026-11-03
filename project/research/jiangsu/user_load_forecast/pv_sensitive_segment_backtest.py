"""Backtest a leakage-controlled PV-sensitive user segment correction.

The aggregate core GBDT remains the anchor.  Candidate users are detected only
with load/PV/radiation observations available by each training cutoff, then a
separate candidate-user model and a non-candidate model are summed.  A coarse
validation grid selects the user tier, feature set, daylight guard, and blend
weight before the final holdout is scored.

Customer-level outputs are private.  Province report outputs are aggregate only.
The strict mode never accesses target-day DA and uses weather predictors only
from snapshots that pass the strict issue_at/available_at contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.calendar_baseline import (
    select_similar_dates,
)
from project.research.jiangsu.user_load_forecast.calendar_features import day_type
from project.research.jiangsu.user_load_forecast.detect_user_pv_candidates import (
    analyze,
    read_absolute_pv_curves,
    read_midday_radiation,
    read_user_curves,
)
from project.research.jiangsu.user_load_forecast.market_d2_curve_backtest import (
    build_features,
    curve_metrics,
    fit_candidate,
    visible_source_dates,
)
from project.research.jiangsu.user_load_forecast.market_da_signal_backtest import (
    read_system_market,
)
from project.research.jiangsu.user_load_forecast.market_level_shape_backtest import (
    MARKET_POLICY,
    SYSTEM_COMPONENT,
    TOTAL_COMPONENTS,
    add_expanded_market_features,
    read_market_matrix,
)
from project.research.jiangsu.user_load_forecast.model_backtest import (
    prepare_output_dir,
)
from project.research.jiangsu.user_load_forecast.pv_weather_challenger_audit import (
    WEATHER_FEATURES,
    attach_weather,
    combine_market_inputs,
    score_window,
)
from project.research.jiangsu.user_load_forecast.weather_features import (
    WEATHER_POLICY_PROXY,
    WEATHER_POLICY_STRICT,
    read_weather_snapshot,
)


VERSION = "surun_pv_sensitive_segment_backtest_v1.2"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
TIER_SCOPES = {
    "direct_high": {"DIRECT_EXPORT_SIGNATURE", "HIGH_STATISTICAL"},
    "direct_high_medium": {
        "DIRECT_EXPORT_SIGNATURE",
        "HIGH_STATISTICAL",
        "MEDIUM_STATISTICAL",
    },
}
GUARDS = ("daylight", "all_intervals")
PROXY_PV_FEATURE_SETS = ("core", "solar", "solar_weather")
STRICT_PV_FEATURE_SETS = ("core", "solar", "solar_weather")
STRICT_MARKET_POLICY = "STRICT_SUBMISSION_ASOF_T_MINUS_2_MARKET_DATE"
STRICT_WEATHER_POLICY = "HISTORICAL_ONLY_FOR_SEGMENT_DETECTION_NOT_PREDICTOR"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_portfolio_curves(path: Path) -> dict[date, np.ndarray]:
    frame = pd.read_csv(path)
    required = {"trade_date", "trade_interval", "actual_load"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Portfolio input missing {sorted(required - set(frame))}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame["trade_interval"] = pd.to_numeric(
        frame["trade_interval"], errors="raise"
    ).astype(int)
    frame["actual_load"] = pd.to_numeric(frame["actual_load"], errors="raise")
    if frame.duplicated(["trade_date", "trade_interval"]).any():
        raise ValueError("Portfolio input has duplicate date/interval rows")
    counts = frame.groupby("trade_date")["trade_interval"].nunique()
    if not counts.eq(96).all():
        raise ValueError("Portfolio input has incomplete days")
    return {
        target_date: group.sort_values("trade_interval")["actual_load"].to_numpy(
            dtype=float
        )
        for target_date, group in frame.groupby("trade_date", sort=True)
    }


def calendar_frame(
    curves: dict[date, np.ndarray],
    *,
    model_name: str,
    target_start: date = date(2026, 5, 1),
) -> pd.DataFrame:
    dates = sorted(curves)
    rows = []
    for target_date in dates:
        if target_date < target_start:
            continue
        sources = select_similar_dates(target_date, dates)
        prediction = np.mean([curves[source] for source in sources], axis=0)
        for interval, (actual, predicted) in enumerate(
            zip(curves[target_date], prediction, strict=True), start=1
        ):
            rows.append(
                {
                    "model": "calendar_similar_day_mean4",
                    "component": model_name,
                    "target_date": target_date,
                    "trade_interval": interval,
                    "source_dates": "|".join(item.isoformat() for item in sources),
                    "day_type": day_type(target_date),
                    "actual_load": float(actual),
                    "predicted_load": float(predicted),
                }
            )
    return pd.DataFrame(rows)


def component_curves(
    portfolio_curves: dict[date, np.ndarray],
    user_curves: dict[tuple[str, date], np.ndarray],
    members: set[str],
) -> tuple[dict[date, np.ndarray], dict[date, np.ndarray]]:
    candidate = {
        target_date: np.zeros(96, dtype=float) for target_date in portfolio_curves
    }
    for (user, target_date), curve in user_curves.items():
        if user in members and target_date in candidate:
            candidate[target_date] += curve
    remainder = {
        target_date: portfolio_curves[target_date] - candidate[target_date]
        for target_date in portfolio_curves
    }
    return candidate, remainder


def build_strict_features(
    calendar_prediction_file: Path,
    market_input: Path,
    *,
    allow_causal_market_fallback: bool = False,
) -> pd.DataFrame:
    """Build features whose newest market trade date is target minus two."""
    calendar = pd.read_csv(calendar_prediction_file)
    calendar = calendar.loc[
        calendar["model"].eq("calendar_similar_day_mean4")
    ].copy()
    calendar["target_date"] = pd.to_datetime(calendar["target_date"]).dt.date
    market = read_system_market(market_input).set_index(
        ["trade_date", "trade_interval"]
    )
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
            float(market.loc[(source, trade_interval), "RT"])
            for source in sources
            if (source, trade_interval) in market.index
            and pd.notna(market.loc[(source, trade_interval), "RT"])
        ]
        if not source_rt:
            raise ValueError(
                f"No D-3-visible source RT for {target_date}/{trade_interval}"
            )
        resolved = {}
        for label, nominal_date, required_columns in (
            ("d2", d2, ("DA", "RT")),
            ("d3", d3, ("DA", "RT")),
            ("d7", d7, ("RT",)),
        ):
            candidate_dates = [nominal_date]
            if allow_causal_market_fallback:
                candidate_dates = sorted(
                    {
                        key[0]
                        for key in market.index
                        if key[1] == trade_interval and key[0] <= nominal_date
                    },
                    reverse=True,
                )
            match = None
            for candidate_date in candidate_dates:
                key = (candidate_date, trade_interval)
                if key not in market.index:
                    continue
                candidate = market.loc[key]
                if all(pd.notna(candidate[column]) for column in required_columns):
                    match = (candidate_date, candidate)
                    break
            if match is None:
                if allow_causal_market_fallback:
                    raise ValueError(
                        "No causal strict market fallback for "
                        f"{target_date}/{trade_interval}/{label}"
                    )
                raise ValueError(
                    f"Missing strict DA/RT value for {target_date}/{trade_interval}"
                )
            resolved[label] = match
        d2_effective, market_d2 = resolved["d2"]
        d3_effective, market_d3 = resolved["d3"]
        d7_effective, market_d7 = resolved["d7"]
        source_rt_mean = float(np.mean(source_rt))
        angle = 2.0 * np.pi * (trade_interval - 1) / 96.0
        rows.append(
            {
                "target_date": target_date,
                "trade_interval": trade_interval,
                "day_type": row.day_type,
                "actual_load": float(row.actual_load),
                "calendar_prediction": float(row.predicted_load),
                "load_label_cutoff": load_cutoff,
                "market_rt_cutoff": d2,
                "market_source_max_date": max([d2_effective, *sources]),
                "market_d2_effective_date": d2_effective,
                "market_d3_effective_date": d3_effective,
                "market_d7_effective_date": d7_effective,
                "d2_market_age_days": float((d2 - d2_effective).days),
                "log_calendar_prediction": np.log1p(row.predicted_load),
                "interval_sin": np.sin(angle),
                "interval_cos": np.cos(angle),
                "d2_rt_da_bias": float(market_d2["RT"])
                / float(market_d2["DA"]),
                "d2_rt_vs_source_rt": float(market_d2["RT"])
                / source_rt_mean,
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
    expected_cutoff = frame["target_date"].map(
        lambda value: value - timedelta(days=2)
    )
    if not frame["market_source_max_date"].le(expected_cutoff).all():
        raise ValueError("Strict market visibility check failed")
    return frame


def strict_market_values(
    frame: pd.DataFrame,
    market: pd.DataFrame,
    *,
    day_offset: int,
    data_name: str,
    data_type: str,
) -> np.ndarray:
    effective_column = f"market_d{day_offset}_effective_date"
    dates = (
        frame[effective_column]
        if effective_column in frame.columns
        else frame["target_date"].map(
            lambda value: value - timedelta(days=day_offset)
        )
    )
    keys = pd.MultiIndex.from_arrays(
        [dates, frame["trade_interval"].astype(int)],
        names=["trade_date", "trade_interval"],
    )
    column = f"{data_name}__{data_type}"
    if column not in market.columns:
        raise ValueError(f"Strict market column is absent: {column}")
    values = market[column].reindex(keys).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Missing strict T-2 values for {column}")
    return values


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=np.abs(denominator) > 1e-9,
    )


def add_strict_market_features(
    frame: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    """Add only T-2 market features; target-day DA is never accessed."""
    result = frame.copy()
    system_d2_rt = strict_market_values(
        result,
        market,
        day_offset=2,
        data_name=SYSTEM_COMPONENT,
        data_type="RT",
    )
    pv_name = TOTAL_COMPONENTS["photovoltaic"]
    wind_name = TOTAL_COMPONENTS["wind"]
    pv_d2_da = strict_market_values(
        result, market, day_offset=2, data_name=pv_name, data_type="DA"
    )
    pv_d2_rt = strict_market_values(
        result, market, day_offset=2, data_name=pv_name, data_type="RT"
    )
    pv_d3_rt = strict_market_values(
        result, market, day_offset=3, data_name=pv_name, data_type="RT"
    )
    pv_d7_rt = strict_market_values(
        result, market, day_offset=7, data_name=pv_name, data_type="RT"
    )
    wind_d2_rt = strict_market_values(
        result, market, day_offset=2, data_name=wind_name, data_type="RT"
    )
    result["d2_rt_photovoltaic_share"] = safe_ratio(pv_d2_rt, system_d2_rt)
    result["d2_photovoltaic_rt_da_gap_share"] = safe_ratio(
        pv_d2_rt - pv_d2_da, system_d2_rt
    )
    result["d2_pv_vs_d3_gap_share"] = safe_ratio(
        pv_d2_rt - pv_d3_rt, system_d2_rt
    )
    result["d2_pv_vs_d7_gap_share"] = safe_ratio(
        pv_d2_rt - pv_d7_rt, system_d2_rt
    )
    result["d2_rt_net_load_share"] = safe_ratio(
        system_d2_rt - pv_d2_rt - wind_d2_rt, system_d2_rt
    )
    daily_pv_mean = pd.Series(pv_d2_rt, index=result.index).groupby(
        result["target_date"]
    ).transform("mean")
    result["d2_pv_shape_ratio"] = safe_ratio(
        pv_d2_rt, daily_pv_mean.to_numpy(dtype=float)
    )
    result["calendar_daily_mean"] = result.groupby(
        "target_date"
    )["calendar_prediction"].transform("mean")
    result["calendar_shape_ratio"] = safe_ratio(
        result["calendar_prediction"].to_numpy(dtype=float),
        result["calendar_daily_mean"].to_numpy(dtype=float),
    )
    day_of_year = result["target_date"].map(lambda value: value.timetuple().tm_yday)
    seasonal_angle = 2.0 * np.pi * day_of_year / 365.25
    result["day_of_year_sin"] = np.sin(seasonal_angle)
    result["day_of_year_cos"] = np.cos(seasonal_angle)
    core = (
        "log_calendar_prediction",
        "interval_sin",
        "interval_cos",
        "trade_interval",
        "day_type",
        "d2_market_age_days",
        "d2_rt_da_bias",
        "d2_rt_vs_source_rt",
        "d2_rt_vs_d3_rt",
        "d2_rt_vs_d7_rt",
        "d2_da_vs_d3_da",
    )
    solar = (
        *core,
        "calendar_shape_ratio",
        "day_of_year_sin",
        "day_of_year_cos",
        "d2_rt_photovoltaic_share",
        "d2_photovoltaic_rt_da_gap_share",
        "d2_pv_vs_d3_gap_share",
        "d2_pv_vs_d7_gap_share",
        "d2_rt_net_load_share",
        "d2_pv_shape_ratio",
    )
    forbidden = [column for column in (*core, *solar) if column.startswith("target_da_")]
    if forbidden:
        raise ValueError(f"Forbidden target-day DA features: {forbidden}")
    return result, {"core": core, "solar": solar}


def attach_strict_weather(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    weather = read_weather_snapshot(path, policy=WEATHER_POLICY_STRICT)
    weather_frames = []
    for target_date in sorted(frame["target_date"].unique()):
        features = weather.features_for_day(target_date)
        features.insert(0, "target_date", target_date)
        weather_frames.append(features)
    weather_frame = pd.concat(weather_frames, ignore_index=True)
    return frame.merge(
        weather_frame,
        on=["target_date", "trade_interval"],
        how="left",
        validate="one_to_one",
    )


def prepare_feature_frame(
    curves: dict[date, np.ndarray],
    *,
    component: str,
    market_path: Path,
    market_matrix: pd.DataFrame,
    weather_input: Path,
    temp_root: Path,
    visibility_policy: str,
    strict_predictor_weather_input: Path | None,
    strict_market_fallback: bool = False,
    calendar_start: date = date(2026, 5, 1),
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    calendar = calendar_frame(
        curves,
        model_name=component,
        target_start=calendar_start,
    )
    calendar_path = temp_root / f"{component}_calendar.csv"
    calendar.to_csv(calendar_path, index=False)
    if visibility_policy == "strict_submission_asof":
        base = build_strict_features(
            calendar_path,
            market_path,
            allow_causal_market_fallback=strict_market_fallback,
        )
        expanded, feature_groups = add_strict_market_features(base, market_matrix)
        if strict_predictor_weather_input is not None:
            expanded = attach_strict_weather(
                expanded, strict_predictor_weather_input
            )
            feature_groups["weather"] = WEATHER_FEATURES
    else:
        base = build_features(calendar_path, market_path)
        expanded, feature_groups = add_expanded_market_features(base, market_matrix)
        expanded = attach_weather(expanded, weather_input)
    return expanded, feature_groups


def feature_sets(
    feature_groups: dict[str, tuple[str, ...]],
    *,
    visibility_policy: str,
) -> dict[str, tuple[str, ...]]:
    if visibility_policy == "strict_submission_asof":
        result = {
            "core": feature_groups["core"],
            "solar": feature_groups["solar"],
        }
        if all(column in feature_groups.get("weather", ()) for column in WEATHER_FEATURES):
            result["solar_weather"] = (*feature_groups["solar"], *WEATHER_FEATURES)
        return result
    solar = tuple(
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
    return {
        "core": feature_groups["core"],
        "solar": solar,
        "solar_weather": (*solar, *WEATHER_FEATURES),
    }


def fit_with_weather_fallback(
    frame: pd.DataFrame,
    *,
    model_name: str,
    feature_name: str,
    features: dict[str, tuple[str, ...]],
    train_end: date,
    holdout_start: date,
) -> pd.DataFrame:
    if feature_name != "solar_weather":
        return fit_candidate(
            frame.dropna(subset=list(features[feature_name])).copy(),
            model_name=model_name,
            feature_columns=features[feature_name],
            train_end=train_end,
            holdout_start=holdout_start,
        )
    fallback = fit_candidate(
        frame.dropna(subset=list(features["solar"])).copy(),
        model_name=f"{model_name}_fallback_solar",
        feature_columns=features["solar"],
        train_end=train_end,
        holdout_start=holdout_start,
    )
    eligible = frame.dropna(subset=list(features["solar_weather"])).copy()
    weather = fit_candidate(
        eligible,
        model_name=model_name,
        feature_columns=features["solar_weather"],
        train_end=train_end,
        holdout_start=holdout_start,
    )
    keys = ["target_date", "trade_interval"]
    weather_lookup = weather.set_index(keys)["predicted_load"]
    result = fallback.copy()
    index = pd.MultiIndex.from_frame(result[keys])
    replacement = weather_lookup.reindex(index).to_numpy(dtype=float)
    use_weather = np.isfinite(replacement)
    result.loc[use_weather, "predicted_load"] = replacement[use_weather]
    result["model"] = model_name
    result["weather_fallback"] = ~use_weather
    result["error"] = result["predicted_load"] - result["actual_load"]
    result["absolute_error"] = result["error"].abs()
    return result


def combine_segment_predictions(
    core: pd.DataFrame,
    candidate: pd.DataFrame,
    remainder: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["target_date", "trade_interval"]
    result = core[
        keys + ["day_type", "actual_load", "predicted_load"]
    ].rename(columns={"predicted_load": "core_prediction"})
    result = result.merge(
        candidate[keys + ["actual_load", "predicted_load"]].rename(
            columns={
                "actual_load": "candidate_actual",
                "predicted_load": "candidate_prediction",
            }
        ),
        on=keys,
        validate="one_to_one",
    )
    result = result.merge(
        remainder[keys + ["actual_load", "predicted_load"]].rename(
            columns={
                "actual_load": "remainder_actual",
                "predicted_load": "remainder_prediction",
            }
        ),
        on=keys,
        validate="one_to_one",
    )
    component_actual = result["candidate_actual"] + result["remainder_actual"]
    if not np.allclose(component_actual, result["actual_load"], atol=1e-6):
        raise ValueError("Candidate and remainder actuals do not sum to portfolio")
    result["segmented_prediction"] = (
        result["candidate_prediction"] + result["remainder_prediction"]
    )
    return result


def apply_guard(
    combined: pd.DataFrame,
    *,
    model_name: str,
    alpha: float,
    guard: str,
) -> pd.DataFrame:
    result = combined.copy()
    if guard == "daylight":
        active = result["trade_interval"].between(25, 72).to_numpy(dtype=float)
    elif guard == "all_intervals":
        active = np.ones(len(result), dtype=float)
    else:
        raise ValueError(f"Unknown guard: {guard}")
    result.insert(0, "model", model_name)
    result["blend_alpha"] = alpha
    result["guard"] = guard
    result["predicted_load"] = result["core_prediction"] + (
        alpha
        * active
        * (result["segmented_prediction"] - result["core_prediction"])
    )
    result["error"] = result["predicted_load"] - result["actual_load"]
    result["absolute_error"] = result["error"].abs()
    return result


def detect_at_cutoff(
    user_curves: dict[tuple[str, date], np.ndarray],
    pv_curves: dict[date, np.ndarray],
    radiation: dict[date, float],
    *,
    cutoff: date,
    permutations: int,
    random_seed: int,
    min_d7_pairs: int,
) -> pd.DataFrame:
    return analyze(
        {key: value for key, value in user_curves.items() if key[1] <= cutoff},
        {key: value for key, value in pv_curves.items() if key <= cutoff},
        {key: value for key, value in radiation.items() if key <= cutoff},
        permutations=permutations,
        random_seed=random_seed,
        min_d7_pairs=min_d7_pairs,
    )


def members_for_scope(candidates: pd.DataFrame, scope: str) -> set[str]:
    tiers = TIER_SCOPES[scope]
    return set(
        candidates.loc[
            candidates["evidence_tier"].isin(tiers), "retail_user_name"
        ]
    )


def score_prediction(
    prediction: pd.DataFrame,
    *,
    start: date,
    end: date,
) -> dict:
    selected = prediction.loc[prediction["target_date"].between(start, end)].copy()
    _, metrics = curve_metrics(selected)
    return metrics


def candidate_summary(
    candidates: pd.DataFrame,
    *,
    cutoff: date,
    stage: str,
) -> list[dict]:
    rows = []
    for scope in TIER_SCOPES:
        selected = candidates["evidence_tier"].isin(TIER_SCOPES[scope])
        rows.append(
            {
                "stage": stage,
                "candidate_cutoff": cutoff.isoformat(),
                "tier_scope": scope,
                "candidate_users": int(selected.sum()),
                "training_energy_share_pct": float(
                    candidates.loc[selected, "portfolio_energy_share_pct"].sum()
                ),
            }
        )
    return rows


def run(args: argparse.Namespace) -> None:
    if (
        args.strict_predictor_weather_input is not None
        and args.visibility_policy != "strict_submission_asof"
    ):
        raise ValueError("Strict predictor weather requires strict visibility policy")
    prepare_output_dir(args.output_dir)
    if "reports/provinces" in str(args.private_output_dir.resolve()):
        raise ValueError("Private customer output cannot enter public reports")
    if args.private_output_dir.exists() and any(args.private_output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.private_output_dir}")
    args.private_output_dir.mkdir(parents=True, exist_ok=True)

    portfolio_curves = read_portfolio_curves(args.portfolio_input)
    user_curves = read_user_curves(args.user_history_input, args.user_update_input)
    pv_curves = read_absolute_pv_curves(args.market_input)
    radiation = read_midday_radiation(args.weather_input)
    strict_mode = args.visibility_policy == "strict_submission_asof"
    core_model_name = "strict_gbdt_core" if strict_mode else "frozen_gbdt_core"

    validation_candidates = detect_at_cutoff(
        user_curves,
        pv_curves,
        radiation,
        cutoff=args.validation_train_end,
        permutations=args.permutations,
        random_seed=args.random_seed,
        min_d7_pairs=args.min_d7_pairs,
    )
    final_candidates = detect_at_cutoff(
        user_curves,
        pv_curves,
        radiation,
        cutoff=args.final_train_end,
        permutations=args.permutations,
        random_seed=args.random_seed,
        min_d7_pairs=args.min_d7_pairs,
    )
    validation_candidates.to_csv(
        args.private_output_dir / "validation_cutoff_candidates.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )
    final_candidates.to_csv(
        args.private_output_dir / "final_cutoff_candidates.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        market_path = temp_root / "combined_market.csv.gz"
        combine_market_inputs(args.market_input, market_path)
        market_matrix = read_market_matrix(market_path)

        aggregate_frame, aggregate_groups = prepare_feature_frame(
            portfolio_curves,
            component="aggregate",
            market_path=market_path,
            market_matrix=market_matrix,
            weather_input=args.weather_input,
            temp_root=temp_root,
            visibility_policy=args.visibility_policy,
            strict_predictor_weather_input=args.strict_predictor_weather_input,
            strict_market_fallback=args.strict_market_fallback,
        )
        aggregate_features = feature_sets(
            aggregate_groups, visibility_policy=args.visibility_policy
        )
        pv_feature_sets = tuple(
            feature
            for feature in (
                STRICT_PV_FEATURE_SETS if strict_mode else PROXY_PV_FEATURE_SETS
            )
            if feature in aggregate_features
        )
        validation_core = fit_candidate(
            aggregate_frame,
            model_name=core_model_name,
            feature_columns=aggregate_features["core"],
            train_end=args.validation_train_end,
            holdout_start=args.validation_start,
        )

        validation_grid = []
        validation_combined: dict[tuple[str, str], pd.DataFrame] = {}
        for scope in TIER_SCOPES:
            members = members_for_scope(validation_candidates, scope)
            if not members:
                continue
            candidate_curves, remainder_curves = component_curves(
                portfolio_curves, user_curves, members
            )
            candidate_frame, candidate_groups = prepare_feature_frame(
                candidate_curves,
                component=f"validation_{scope}_candidate",
                market_path=market_path,
                market_matrix=market_matrix,
                weather_input=args.weather_input,
                temp_root=temp_root,
                visibility_policy=args.visibility_policy,
                strict_predictor_weather_input=args.strict_predictor_weather_input,
                strict_market_fallback=args.strict_market_fallback,
            )
            remainder_frame, remainder_groups = prepare_feature_frame(
                remainder_curves,
                component=f"validation_{scope}_remainder",
                market_path=market_path,
                market_matrix=market_matrix,
                weather_input=args.weather_input,
                temp_root=temp_root,
                visibility_policy=args.visibility_policy,
                strict_predictor_weather_input=args.strict_predictor_weather_input,
                strict_market_fallback=args.strict_market_fallback,
            )
            candidate_features = feature_sets(
                candidate_groups, visibility_policy=args.visibility_policy
            )
            remainder_features = feature_sets(
                remainder_groups, visibility_policy=args.visibility_policy
            )
            remainder_prediction = fit_candidate(
                remainder_frame,
                model_name=f"{scope}_remainder_core",
                feature_columns=remainder_features["core"],
                train_end=args.validation_train_end,
                holdout_start=args.validation_start,
            )
            for pv_features in pv_feature_sets:
                candidate_prediction = fit_with_weather_fallback(
                    candidate_frame,
                    model_name=f"{scope}_candidate_{pv_features}",
                    feature_name=pv_features,
                    features=candidate_features,
                    train_end=args.validation_train_end,
                    holdout_start=args.validation_start,
                )
                combined = combine_segment_predictions(
                    validation_core, candidate_prediction, remainder_prediction
                )
                validation_combined[(scope, pv_features)] = combined
                for guard in GUARDS:
                    for alpha in ALPHAS:
                        model_name = f"pv_segment_{scope}_{pv_features}_{guard}_a{alpha:g}"
                        prediction = apply_guard(
                            combined,
                            model_name=model_name,
                            alpha=alpha,
                            guard=guard,
                        )
                        metrics = score_prediction(
                            prediction,
                            start=args.validation_start,
                            end=args.validation_end,
                        )
                        validation_grid.append(
                            {
                                "model": model_name,
                                "tier_scope": scope,
                                "pv_feature_set": pv_features,
                                "guard": guard,
                                "alpha": alpha,
                                **metrics,
                            }
                        )
        validation_grid_frame = pd.DataFrame(validation_grid).sort_values(
            ["wmape_pct", "daily_wmape_p90_pct", "model"]
        )
        selected = validation_grid_frame.iloc[0]

        final_core = fit_candidate(
            aggregate_frame,
            model_name=core_model_name,
            feature_columns=aggregate_features["core"],
            train_end=args.final_train_end,
            holdout_start=args.holdout_start,
        )
        selected_scope = str(selected["tier_scope"])
        selected_features = str(selected["pv_feature_set"])
        final_members = members_for_scope(final_candidates, selected_scope)
        candidate_curves, remainder_curves = component_curves(
            portfolio_curves, user_curves, final_members
        )
        candidate_frame, candidate_groups = prepare_feature_frame(
            candidate_curves,
            component="final_candidate",
            market_path=market_path,
            market_matrix=market_matrix,
            weather_input=args.weather_input,
            temp_root=temp_root,
            visibility_policy=args.visibility_policy,
            strict_predictor_weather_input=args.strict_predictor_weather_input,
            strict_market_fallback=args.strict_market_fallback,
        )
        remainder_frame, remainder_groups = prepare_feature_frame(
            remainder_curves,
            component="final_remainder",
            market_path=market_path,
            market_matrix=market_matrix,
            weather_input=args.weather_input,
            temp_root=temp_root,
            visibility_policy=args.visibility_policy,
            strict_predictor_weather_input=args.strict_predictor_weather_input,
            strict_market_fallback=args.strict_market_fallback,
        )
        candidate_features = feature_sets(
            candidate_groups, visibility_policy=args.visibility_policy
        )
        remainder_features = feature_sets(
            remainder_groups, visibility_policy=args.visibility_policy
        )
        final_candidate = fit_with_weather_fallback(
            candidate_frame,
            model_name=f"selected_candidate_{selected_features}",
            feature_name=selected_features,
            features=candidate_features,
            train_end=args.final_train_end,
            holdout_start=args.holdout_start,
        )
        final_remainder = fit_candidate(
            remainder_frame,
            model_name="selected_remainder_core",
            feature_columns=remainder_features["core"],
            train_end=args.final_train_end,
            holdout_start=args.holdout_start,
        )
        final_combined = combine_segment_predictions(
            final_core, final_candidate, final_remainder
        )

    selected_prediction = apply_guard(
        final_combined,
        model_name="pv_sensitive_segment_selected",
        alpha=float(selected["alpha"]),
        guard=str(selected["guard"]),
    )
    full_segment_prediction = apply_guard(
        final_combined,
        model_name="pv_sensitive_segment_full",
        alpha=1.0,
        guard=str(selected["guard"]),
    )
    core_prediction = final_core.copy()
    core_prediction["model"] = core_model_name
    predictions = pd.concat(
        [core_prediction, selected_prediction, full_segment_prediction],
        ignore_index=True,
        sort=False,
    )

    summary_rows = []
    for model, prediction in (
        (core_model_name, core_prediction),
        ("pv_sensitive_segment_selected", selected_prediction),
        ("pv_sensitive_segment_full", full_segment_prediction),
    ):
        for window, start, end in (
            ("legacy_holdout", args.holdout_start, args.legacy_holdout_end),
            ("new_diagnostic", args.new_diagnostic_start, args.new_diagnostic_end),
        ):
            row = score_window(prediction, window=window, start=start, end=end)
            if row:
                row["model"] = model
                summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["window", "wmape_pct", "model"]
    )
    daily_rows = []
    for (model, target_date), group in predictions.groupby(
        ["model", "target_date"], sort=True
    ):
        denominator = group["actual_load"].abs().sum()
        daily_rows.append(
            {
                "model": model,
                "target_date": target_date,
                "wmape_pct": group["absolute_error"].sum() / denominator * 100,
                "bias_pct": group["error"].sum() / denominator * 100,
            }
        )
    daily = pd.DataFrame(daily_rows)
    group_summary = pd.DataFrame(
        candidate_summary(
            validation_candidates,
            cutoff=args.validation_train_end,
            stage="validation",
        )
        + candidate_summary(
            final_candidates,
            cutoff=args.final_train_end,
            stage="final",
        )
    )

    validation_grid_frame.to_csv(
        args.output_dir / "validation_grid.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        args.output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig"
    )
    daily.to_csv(
        args.output_dir / "daily_metrics.csv", index=False, encoding="utf-8-sig"
    )
    group_summary.to_csv(
        args.output_dir / "candidate_group_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        args.output_dir / "predictions_96.csv.gz",
        index=False,
        compression="gzip",
    )

    core_validation = validation_grid_frame.loc[
        validation_grid_frame["alpha"].eq(0.0), "wmape_pct"
    ].iloc[0]
    selected_validation = float(selected["wmape_pct"])
    passes_validation = bool(
        float(selected["alpha"]) > 0.0 and selected_validation < core_validation
    )
    core_holdout_row = summary.loc[
        summary["model"].eq(core_model_name)
        & summary["window"].eq("legacy_holdout")
    ].iloc[0]
    selected_holdout_row = summary.loc[
        summary["model"].eq("pv_sensitive_segment_selected")
        & summary["window"].eq("legacy_holdout")
    ].iloc[0]
    core_new_row = summary.loc[
        summary["model"].eq(core_model_name)
        & summary["window"].eq("new_diagnostic")
    ].iloc[0]
    selected_new_row = summary.loc[
        summary["model"].eq("pv_sensitive_segment_selected")
        & summary["window"].eq("new_diagnostic")
    ].iloc[0]
    passes_holdout = bool(
        selected_holdout_row["wmape_pct"] < core_holdout_row["wmape_pct"]
    )
    passes_new_diagnostic = bool(
        selected_new_row["wmape_pct"] <= core_new_row["wmape_pct"]
    )
    promotion_recommended = bool(
        strict_mode
        and passes_validation
        and passes_holdout
        and passes_new_diagnostic
    )
    metadata = {
        "version": VERSION,
        "status": (
            "STRICT_SUBMISSION_ASOF_PROMOTION_CANDIDATE"
            if promotion_recommended
            else "STRICT_SUBMISSION_ASOF_REJECTED_BY_PERFORMANCE_GATE"
            if strict_mode
            else "RESEARCH_PROXY_NOT_CHAMPION_ELIGIBLE"
        ),
        "portfolio_input": str(args.portfolio_input.resolve()),
        "portfolio_sha256": sha256_file(args.portfolio_input),
        "user_history_input": str(args.user_history_input.resolve()),
        "user_history_sha256": sha256_file(args.user_history_input),
        "user_update_inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.user_update_input
        ],
        "market_inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.market_input
        ],
        "weather_input": str(args.weather_input.resolve()),
        "weather_sha256": sha256_file(args.weather_input),
        "visibility_policy": args.visibility_policy,
        "market_policy": STRICT_MARKET_POLICY if strict_mode else MARKET_POLICY,
        "strict_market_fallback": (
            "LATEST_CAUSALLY_VISIBLE_DATE_WITH_AGE_FEATURE"
            if strict_mode and args.strict_market_fallback
            else "ERROR_ON_MISSING"
        ),
        "weather_policy": (
            WEATHER_POLICY_STRICT
            if args.strict_predictor_weather_input is not None
            else STRICT_WEATHER_POLICY
            if strict_mode
            else WEATHER_POLICY_PROXY
        ),
        "strict_predictor_weather_input": (
            str(args.strict_predictor_weather_input.resolve())
            if args.strict_predictor_weather_input is not None
            else None
        ),
        "strict_predictor_weather_sha256": (
            sha256_file(args.strict_predictor_weather_input)
            if args.strict_predictor_weather_input is not None
            else None
        ),
        "load_label_policy": "SIMULATED_T_PLUS_2_D_MINUS_3",
        "candidate_detection_policy": "TRAIN_CUTOFF_ONLY_NO_HOLDOUT_LABELS",
        "validation_candidate_cutoff": args.validation_train_end.isoformat(),
        "final_candidate_cutoff": args.final_train_end.isoformat(),
        "min_d7_pairs": args.min_d7_pairs,
        "permutations": args.permutations,
        "random_seed": args.random_seed,
        "selected": {
            "tier_scope": selected_scope,
            "pv_feature_set": selected_features,
            "guard": str(selected["guard"]),
            "alpha": float(selected["alpha"]),
            "validation_wmape_pct": selected_validation,
            "core_validation_wmape_pct": float(core_validation),
        },
        "passes_validation_gate": passes_validation,
        "passes_legacy_holdout_gate": passes_holdout,
        "passes_new_diagnostic_gate": passes_new_diagnostic,
        "promotion_recommended": promotion_recommended,
        "data_boundary_eligible": bool(strict_mode),
        "champion_eligible": promotion_recommended,
        "strict_core_feature_columns": list(aggregate_features["core"]),
        "selected_candidate_feature_columns": list(
            candidate_features[selected_features]
        ),
        "target_day_da_feature_check": (
            "PASS_NO_TARGET_DAY_DA"
            if strict_mode
            and not any(
                column.startswith("target_da_")
                for column in (
                    *aggregate_features["core"],
                    *candidate_features[selected_features],
                )
            )
            else "NOT_APPLICABLE_RESEARCH_PROXY"
        ),
        "eligibility_reason": (
            "Strict T-2 market-date cutoff; no target-day DA or proxy weather."
            if strict_mode and args.strict_predictor_weather_input is None
            else "Strict T-2 market cutoff plus auditable strict weather snapshot."
            if strict_mode
            else "Historical market/weather exports lack trustworthy issue_at/available_at."
        ),
        "privacy": "PUBLIC_AGGREGATE_OUTPUT_AND_SEPARATE_PRIVATE_CUSTOMER_OUTPUT",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.private_output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "version": VERSION,
                "privacy": "PRIVATE_CUSTOMER_LEVEL_OUTPUT",
                "public_output_dir": str(args.output_dir.resolve()),
                "validation_candidate_cutoff": args.validation_train_end.isoformat(),
                "final_candidate_cutoff": args.final_train_end.isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    selected_holdout = summary.loc[
        summary["model"].eq("pv_sensitive_segment_selected")
        & summary["window"].eq("legacy_holdout")
    ].iloc[0]
    core_holdout = summary.loc[
        summary["model"].eq(core_model_name)
        & summary["window"].eq("legacy_holdout")
    ].iloc[0]
    selected_new = summary.loc[
        summary["model"].eq("pv_sensitive_segment_selected")
        & summary["window"].eq("new_diagnostic")
    ].iloc[0]
    core_new = summary.loc[
        summary["model"].eq(core_model_name)
        & summary["window"].eq("new_diagnostic")
    ].iloc[0]
    lines = [
        "# 江苏苏润负荷：光伏敏感用户分组修正回测",
        "",
        "## 验证选择",
        "",
        (
            f"- 仅用截至 {args.validation_train_end.isoformat()} 的证据识别验证用户，"
            f"选择 `{selected_scope}` / `{selected_features}` / "
            f"`{selected['guard']}` / alpha={float(selected['alpha']):.2f}。"
        ),
        (
            f"- 验证WMAPE：核心模型 {float(core_validation):.4f}%，"
            f"选中修正 {selected_validation:.4f}%；"
            f"门禁：{'通过' if passes_validation else '未通过'}。"
        ),
        "",
        "## 冻结回测",
        "",
        (
            f"- 2026-06-15至2026-07-25：核心 {core_holdout['wmape_pct']:.4f}%，"
            f"修正 {selected_holdout['wmape_pct']:.4f}%。"
        ),
        (
            f"- 2026-07-26至2026-07-28：核心 {core_new['wmape_pct']:.4f}%，"
            f"修正 {selected_new['wmape_pct']:.4f}%。"
        ),
        (
            f"- 严格晋级门禁：验证{'通过' if passes_validation else '未通过'}，"
            f"历史留出{'通过' if passes_holdout else '未通过'}，"
            f"新增日期{'通过' if passes_new_diagnostic else '未通过'}；"
            f"结论：{'建议晋级' if promotion_recommended else '不晋级'}。"
        ),
        "",
        "## 状态",
        "",
        "- 用户识别和模型训练均按各自截止日冻结，未使用测试期标签。",
        "- 用户明细仅保存在私有目录，省级报告只含聚合统计。",
        (
            "- 严格轨道禁用目标日DA；市场最大交易日为T-2。"
            if strict_mode
            else "- 市场与天气发布时间不可审计，本结果仍是研究代理，不可直接替换Champion。"
        ),
    ]
    (args.output_dir / "run_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(validation_grid_frame.head(12).to_string(index=False))
    print()
    print(group_summary.to_string(index=False))
    print()
    print(summary[["model", "window", "wmape_pct"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-input", type=Path, required=True)
    parser.add_argument("--user-history-input", type=Path, required=True)
    parser.add_argument("--user-update-input", type=Path, action="append", default=[])
    parser.add_argument("--market-input", type=Path, action="append", required=True)
    parser.add_argument("--weather-input", type=Path, required=True)
    parser.add_argument(
        "--strict-market-fallback",
        action="store_true",
        help=(
            "In strict mode, replace a missing market point only with the latest "
            "earlier complete point and expose the fallback age as a feature."
        ),
    )
    parser.add_argument(
        "--strict-predictor-weather-input",
        type=Path,
        help=(
            "Optional target-day forecast snapshot with auditable issue_at/available_at; "
            "used only in strict mode."
        ),
    )
    parser.add_argument(
        "--visibility-policy",
        choices=("research_proxy", "strict_submission_asof"),
        default="research_proxy",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
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
        "--final-train-end", type=date.fromisoformat, default=date(2026, 6, 12)
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
    parser.add_argument("--min-d7-pairs", type=int, default=40)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
