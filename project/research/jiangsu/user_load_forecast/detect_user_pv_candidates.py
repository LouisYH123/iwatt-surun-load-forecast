"""Detect retail users with load-shape evidence of behind-the-meter PV.

The output contains customer names and must be written to a private analysis
directory, never to the province-level public report tree. Results are evidence
tiers for manual verification, not confirmed equipment ownership.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "surun_user_pv_candidate_detection_v1.1"
DAYLIGHT = np.arange(24, 72)
MIDDAY = np.arange(40, 60)
SHOULDER = np.r_[np.arange(28, 36), np.arange(64, 72)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pg_array(raw: str) -> np.ndarray:
    values = np.fromstring(raw.strip().strip("{}"), sep=",", dtype=float)
    if len(values) != 96 or not np.isfinite(values).all():
        raise ValueError("Invalid 96-point PostgreSQL actual_values array")
    return values


def read_user_curves(
    history_input: Path,
    update_inputs: list[Path],
) -> dict[tuple[str, date], np.ndarray]:
    curves: dict[tuple[str, date], np.ndarray] = defaultdict(
        lambda: np.zeros(96, dtype=float)
    )
    opener = gzip.open if history_input.name.endswith(".gz") else open
    with opener(history_input, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["retail_user_name"].strip(),
                date.fromisoformat(row["trade_date"]),
            )
            curves[key] += parse_pg_array(row["actual_values"])

    for path in update_inputs:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        required = {"零售用户名", "日期"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path}: missing Chinese user/date columns")
        time_columns = [
            column
            for column in frame.columns
            if isinstance(column, str)
            and len(column) == 5
            and column[2] == ":"
        ]
        if len(time_columns) != 96:
            raise ValueError(f"{path}: expected 96 time columns")
        frame[time_columns] = frame[time_columns].apply(
            pd.to_numeric,
            errors="raise",
        )
        for (user, raw_date), group in frame.groupby(["零售用户名", "日期"]):
            key = (str(user).strip(), pd.Timestamp(raw_date).date())
            curves[key] += group[time_columns].sum().to_numpy(dtype=float)
    return dict(curves)


def read_absolute_pv_curves(paths: list[Path]) -> dict[date, np.ndarray]:
    parts = []
    usecols = ["trade_date", "trade_interval", "data_name", "data_type", "value"]
    for path in paths:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=300_000):
            selected = chunk.loc[
                chunk["data_type"].eq("RT")
                & chunk["data_name"].eq("NEW_ENERGY_PHOTOVOLTAIC_POWER")
            ].copy()
            if not selected.empty:
                parts.append(selected)
    if not parts:
        raise ValueError("No RT photovoltaic output rows found")
    frame = pd.concat(parts, ignore_index=True).drop_duplicates(
        ["trade_date", "trade_interval"],
        keep="last",
    )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    curves = {}
    for trade_date, group in frame.groupby("trade_date", sort=True):
        group = group.sort_values("trade_interval")
        if group["trade_interval"].nunique() == 96:
            curves[trade_date] = group["value"].to_numpy(dtype=float)
    return curves


def read_midday_radiation(path: Path) -> dict[date, float]:
    frame = pd.read_csv(
        path,
        usecols=["timestamp", "source", "radiation"],
    )
    frame = frame.loc[frame["source"].eq("Ts_weather_1")].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["trade_date"] = frame["timestamp"].dt.date
    frame = frame.loc[frame["timestamp"].dt.hour.between(10, 15)]
    daily = frame.groupby("trade_date").agg(
        radiation=("radiation", "mean"),
        hours=("timestamp", "nunique"),
    )
    return daily.loc[daily["hours"].ge(6), "radiation"].to_dict()


def permutation_pvalue(
    left: np.ndarray,
    right: np.ndarray,
    *,
    rng: np.random.Generator,
    repeats: int,
) -> tuple[float, float, float]:
    observed = float(np.corrcoef(left.ravel(), right.ravel())[0, 1])
    null = []
    for _ in range(repeats):
        permutation = rng.permutation(len(right))
        null.append(
            float(
                np.corrcoef(
                    left.ravel(),
                    right[permutation].ravel(),
                )[0, 1]
            )
        )
    pvalue = (1 + sum(value <= observed for value in null)) / (repeats + 1)
    zscore = (observed - float(np.mean(null))) / (float(np.std(null)) or 1.0)
    return observed, float(pvalue), float(zscore)


def evidence_tier(row: dict) -> str:
    if (
        row["negative_daylight_days"] >= 5
        and row["negative_point_daylight_purity"] >= 0.90
    ):
        return "DIRECT_EXPORT_SIGNATURE"
    if (
        row["d7_shape_corr_abs_pv"] <= -0.40
        and row["d7_mid_ratio_vs_radiation_corr"] <= -0.70
        and row["shape_permutation_p"] <= 0.01
        and row["daily_permutation_p"] <= 0.01
    ):
        return "HIGH_STATISTICAL"
    if (
        row["d7_shape_corr_abs_pv"] <= -0.30
        and row["d7_mid_ratio_vs_radiation_corr"] <= -0.55
        and row["shape_permutation_p"] <= 0.05
        and row["daily_permutation_p"] <= 0.05
    ):
        return "MEDIUM_STATISTICAL"
    return "REVIEW_ONLY"


def analyze(
    user_curves: dict[tuple[str, date], np.ndarray],
    pv_curves: dict[date, np.ndarray],
    radiation: dict[date, float],
    *,
    permutations: int,
    random_seed: int,
    min_d7_pairs: int = 60,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    all_energy = sum(float(np.abs(curve).sum()) for curve in user_curves.values())
    users = sorted({user for user, _ in user_curves})
    rows = []
    for user in users:
        all_user_dates = sorted(
            trade_date
            for candidate, trade_date in user_curves
            if candidate == user
        )
        dates = sorted(
            trade_date for trade_date in all_user_dates
            if trade_date in pv_curves and trade_date in radiation
        )
        normalized = {}
        midday_ratio = {}
        negative_daylight_days = 0
        negative_daylight_points = 0
        negative_night_points = 0
        user_energy = 0.0
        for trade_date in all_user_dates:
            curve = user_curves[(user, trade_date)]
            user_energy += float(np.abs(curve).sum())
            negative_daylight_days += int((curve[DAYLIGHT] < 0).any())
            negative_daylight_points += int((curve[DAYLIGHT] < 0).sum())
            negative_night_points += int(
                (np.r_[curve[:24], curve[72:]] < 0).sum()
            )
        for trade_date in dates:
            curve = user_curves[(user, trade_date)]
            shoulder_scale = float(np.mean(np.abs(curve[SHOULDER])))
            if shoulder_scale <= max(5.0, 0.05 * float(np.mean(np.abs(curve)))):
                continue
            normalized[trade_date] = np.clip(
                curve[DAYLIGHT] / shoulder_scale,
                -3.0,
                6.0,
            )
            midday_ratio[trade_date] = float(
                curve[MIDDAY].mean() / curve[SHOULDER].mean()
            )

        pair_dates = [
            trade_date
            for trade_date in normalized
            if trade_date - timedelta(days=7) in normalized
        ]
        if len(pair_dates) >= min_d7_pairs:
            load_difference = np.vstack(
                [
                    normalized[trade_date]
                    - normalized[trade_date - timedelta(days=7)]
                    for trade_date in pair_dates
                ]
            )
            pv_difference = np.vstack(
                [
                    pv_curves[trade_date][DAYLIGHT]
                    - pv_curves[trade_date - timedelta(days=7)][DAYLIGHT]
                    for trade_date in pair_dates
                ]
            )
            load_difference -= load_difference.mean(axis=1, keepdims=True)
            pv_difference -= pv_difference.mean(axis=1, keepdims=True)
            shape_corr, shape_p, shape_z = permutation_pvalue(
                load_difference,
                pv_difference,
                rng=rng,
                repeats=permutations,
            )

            ratio_difference = np.asarray(
                [
                    midday_ratio[trade_date]
                    - midday_ratio[trade_date - timedelta(days=7)]
                    for trade_date in pair_dates
                ],
                dtype=float,
            )
            radiation_difference = np.asarray(
                [
                    radiation[trade_date]
                    - radiation[trade_date - timedelta(days=7)]
                    for trade_date in pair_dates
                ],
                dtype=float,
            )
            daily_corr, daily_p, daily_z = permutation_pvalue(
                ratio_difference[:, None],
                radiation_difference[:, None],
                rng=rng,
                repeats=permutations,
            )
        else:
            shape_corr = shape_p = shape_z = np.nan
            daily_corr = daily_p = daily_z = np.nan
        row = {
            "retail_user_name": user,
            "observed_days": len(normalized),
            "d7_pairs": len(pair_dates),
            "d7_shape_corr_abs_pv": shape_corr,
            "shape_permutation_p": shape_p,
            "shape_z_vs_placebo": shape_z,
            "d7_mid_ratio_vs_radiation_corr": daily_corr,
            "daily_permutation_p": daily_p,
            "daily_z_vs_placebo": daily_z,
            "median_mid_to_shoulder_ratio": float(
                np.median(list(midday_ratio.values()))
                if midday_ratio
                else np.nan
            ),
            "negative_daylight_days": negative_daylight_days,
            "negative_daylight_points": negative_daylight_points,
            "negative_night_points": negative_night_points,
            "negative_point_daylight_purity": (
                negative_daylight_points
                / max(1, negative_daylight_points + negative_night_points)
            ),
            "portfolio_energy_share_pct": user_energy / all_energy * 100,
        }
        row["evidence_tier"] = evidence_tier(row)
        rows.append(row)
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    if "reports/provinces" in str(args.output_dir.resolve()):
        raise ValueError("User-level candidate output must not enter public reports")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    user_curves = read_user_curves(args.history_input, args.update_input)
    pv_curves = read_absolute_pv_curves(args.market_input)
    radiation = read_midday_radiation(args.weather_input)
    result = analyze(
        user_curves,
        pv_curves,
        radiation,
        permutations=args.permutations,
        random_seed=args.random_seed,
    )
    tier_order = {
        "DIRECT_EXPORT_SIGNATURE": 0,
        "HIGH_STATISTICAL": 1,
        "MEDIUM_STATISTICAL": 2,
        "REVIEW_ONLY": 3,
    }
    result["tier_order"] = result["evidence_tier"].map(tier_order)
    result = result.sort_values(
        [
            "tier_order",
            "d7_shape_corr_abs_pv",
            "d7_mid_ratio_vs_radiation_corr",
            "portfolio_energy_share_pct",
        ],
        ascending=[True, True, True, False],
    ).drop(columns="tier_order")
    result.to_csv(
        args.output_dir / "pv_candidate_users.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )

    summary = {
        "version": VERSION,
        "status": "CANDIDATE_EVIDENCE_NOT_CONFIRMED_ASSET_OWNERSHIP",
        "scored_users": len(result),
        "tier_counts": {
            tier: int(result["evidence_tier"].eq(tier).sum())
            for tier in tier_order
        },
        "candidate_energy_share_pct": float(
            result.loc[
                ~result["evidence_tier"].eq("REVIEW_ONLY"),
                "portfolio_energy_share_pct",
            ].sum()
        ),
        "history_input": {
            "path": str(args.history_input.resolve()),
            "sha256": sha256_file(args.history_input),
        },
        "update_inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.update_input
        ],
        "market_inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.market_input
        ],
        "weather_input": {
            "path": str(args.weather_input.resolve()),
            "sha256": sha256_file(args.weather_input),
        },
        "random_seed": args.random_seed,
        "permutations": args.permutations,
        "privacy": "PRIVATE_USER_LEVEL_OUTPUT",
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        "\n".join(
            [
                "# 苏润用户自用光伏候选筛查",
                "",
                "- 这是负荷曲线证据，不是设备资产确权。",
                "- `DIRECT_EXPORT_SIGNATURE`：长期日间负值/反向电量，证据最强。",
                "- `HIGH_STATISTICAL`：D-7差分后的日内形状与绝对光伏出力、辐照度均显著反向同步。",
                "- `MEDIUM_STATISTICAL`：方向一致但效应较弱。",
                "- `REVIEW_ONLY`：不应据此判断存在自用光伏。",
                "- 最终仍需合同、并网、计量或客户资料确认。",
                "",
                "本目录含客户名称，只能用于私有分析。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print(
        result.loc[
            ~result["evidence_tier"].eq("REVIEW_ONLY"),
            [
                "retail_user_name",
                "evidence_tier",
                "d7_shape_corr_abs_pv",
                "d7_mid_ratio_vs_radiation_corr",
                "negative_daylight_days",
                "portfolio_energy_share_pct",
            ],
        ]
        .head(30)
        .to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-input", type=Path, required=True)
    parser.add_argument("--update-input", type=Path, action="append", default=[])
    parser.add_argument("--market-input", type=Path, action="append", required=True)
    parser.add_argument("--weather-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
