"""Backtest user-confirmed target-day weather under a strict T-2 market cutoff.

The weather series is accepted under an explicit operational visibility
contract supplied by the user.  It is not represented as independently
auditable historical forecast versions: no issue_at values are invented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

from project.research.jiangsu.user_load_forecast.market_d2_curve_backtest import (
    curve_metrics,
    fit_candidate,
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
    score_window,
)


VERSION = "surun_strict_operational_weather_backtest_v1"
WEATHER_POLICY = "USER_CONFIRMED_TARGET_DAY_D_WEATHER_STABLY_VISIBLE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    portfolio_curves = read_portfolio_curves(args.portfolio_input)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        market_path = temp_root / "combined_market.csv.gz"
        combine_market_inputs(args.market_input, market_path)
        market_matrix = read_market_matrix(market_path)
        strict_frame, strict_groups = prepare_feature_frame(
            portfolio_curves,
            component="aggregate_strict_operational_weather",
            market_path=market_path,
            market_matrix=market_matrix,
            weather_input=args.weather_input,
            temp_root=temp_root,
            visibility_policy="strict_submission_asof",
            strict_predictor_weather_input=None,
        )
    frame = attach_weather(strict_frame, args.weather_input)
    strict_sets = feature_sets(
        strict_groups, visibility_policy="strict_submission_asof"
    )
    feature_sets_by_model = {
        "strict_gbdt_core": strict_sets["core"],
        "strict_gbdt_solar": strict_sets["solar"],
        "strict_gbdt_core_weather_operational": (
            *strict_sets["core"],
            *WEATHER_FEATURES,
        ),
        "strict_gbdt_solar_weather_operational": (
            *strict_sets["solar"],
            *WEATHER_FEATURES,
        ),
    }
    forbidden = {
        model: [column for column in features if column.startswith("target_da_")]
        for model, features in feature_sets_by_model.items()
    }
    if any(forbidden.values()):
        raise ValueError(f"Target-day DA leaked into strict features: {forbidden}")

    validation_predictions = {}
    validation_rows = []
    for model, features in feature_sets_by_model.items():
        eligible = frame.dropna(subset=list(features)).copy()
        prediction = fit_candidate(
            eligible,
            model_name=model,
            feature_columns=features,
            train_end=args.validation_train_end,
            holdout_start=args.validation_start,
        )
        prediction = prediction.loc[
            prediction["target_date"] <= args.validation_end
        ].copy()
        validation_predictions[model] = prediction
        row = score_window(
            prediction,
            window="internal_validation",
            start=args.validation_start,
            end=args.validation_end,
        )
        if row:
            validation_rows.append(row)
    validation = pd.DataFrame(validation_rows).sort_values(
        ["wmape_pct", "daily_wmape_p90_pct", "model"]
    )
    selected_model = str(validation.iloc[0]["model"])

    final_predictions = {}
    for model, features in feature_sets_by_model.items():
        eligible = frame.dropna(subset=list(features)).copy()
        final_predictions[model] = fit_candidate(
            eligible,
            model_name=model,
            feature_columns=features,
            train_end=args.final_train_end,
            holdout_start=args.holdout_start,
        )
    predictions = pd.concat(final_predictions.values(), ignore_index=True)
    summary_rows = []
    for prediction in final_predictions.values():
        for window, start, end in (
            ("legacy_holdout", args.holdout_start, args.legacy_holdout_end),
            ("new_diagnostic", args.new_diagnostic_start, args.new_diagnostic_end),
        ):
            row = score_window(prediction, window=window, start=start, end=end)
            if row:
                summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["window", "wmape_pct", "model"]
    )
    daily = daily_metrics(predictions)

    def metric(model: str, window: str) -> float:
        return float(
            summary.loc[
                summary["model"].eq(model) & summary["window"].eq(window),
                "wmape_pct",
            ].iloc[0]
        )

    baseline = "strict_gbdt_core"
    selected_uses_weather = "weather" in selected_model
    passes_validation = bool(
        selected_uses_weather
        and float(validation.iloc[0]["wmape_pct"])
        < float(validation.loc[validation["model"].eq(baseline), "wmape_pct"].iloc[0])
    )
    passes_holdout = metric(selected_model, "legacy_holdout") < metric(
        baseline, "legacy_holdout"
    )
    passes_new = metric(selected_model, "new_diagnostic") <= metric(
        baseline, "new_diagnostic"
    )
    promotion_recommended = bool(
        passes_validation and passes_holdout and passes_new
    )

    predictions.to_csv(
        args.output_dir / "predictions_96.csv.gz",
        index=False,
        compression="gzip",
    )
    daily.to_csv(
        args.output_dir / "daily_metrics.csv", index=False, encoding="utf-8-sig"
    )
    validation.to_csv(
        args.output_dir / "validation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        args.output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "version": VERSION,
        "status": (
            "STRICT_OPERATIONAL_WEATHER_PROMOTION_CANDIDATE"
            if promotion_recommended
            else "STRICT_OPERATIONAL_WEATHER_REJECTED_BY_PERFORMANCE_GATE"
        ),
        "portfolio_input": str(args.portfolio_input.resolve()),
        "portfolio_sha256": sha256_file(args.portfolio_input),
        "market_inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.market_input
        ],
        "weather_input": str(args.weather_input.resolve()),
        "weather_sha256": sha256_file(args.weather_input),
        "market_policy": STRICT_MARKET_POLICY,
        "weather_policy": WEATHER_POLICY,
        "independent_historical_weather_issue_at_audit": False,
        "weather_contract_basis": "Explicit user confirmation in current task",
        "target_day_da_feature_check": "PASS_NO_TARGET_DAY_DA",
        "selected_on_internal_validation": selected_model,
        "feature_columns": {
            model: list(features) for model, features in feature_sets_by_model.items()
        },
        "passes_validation_gate": passes_validation,
        "passes_legacy_holdout_gate": passes_holdout,
        "passes_new_diagnostic_gate": passes_new,
        "promotion_recommended": promotion_recommended,
        "champion_eligible": promotion_recommended,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 江苏苏润负荷：严格市场边界下的运营气象回测",
        "",
        "## 边界",
        "",
        "- 市场最大交易日为T-2，禁用全部目标日DA。",
        "- 目标日气象按用户确认的D日稳定可见口径使用。",
        "- 不伪造历史issue_at；该可见性依赖运营合同确认。",
        "",
        "## 结果",
        "",
        (
            f"- 验证选择 `{selected_model}`："
            f"{float(validation.iloc[0]['wmape_pct']):.4f}%。"
        ),
        (
            f"- 2026-06-15至2026-07-25：严格核心 "
            f"{metric(baseline, 'legacy_holdout'):.4f}%，选中模型 "
            f"{metric(selected_model, 'legacy_holdout'):.4f}%。"
        ),
        (
            f"- 2026-07-26至2026-07-28：严格核心 "
            f"{metric(baseline, 'new_diagnostic'):.4f}%，选中模型 "
            f"{metric(selected_model, 'new_diagnostic'):.4f}%。"
        ),
        (
            f"- 三重门禁：验证{'通过' if passes_validation else '未通过'}，"
            f"历史留出{'通过' if passes_holdout else '未通过'}，"
            f"新增日期{'通过' if passes_new else '未通过'}；"
            f"结论：{'建议晋级' if promotion_recommended else '不晋级'}。"
        ),
    ]
    (args.output_dir / "run_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(validation[["model", "wmape_pct"]].to_string(index=False))
    print()
    print(summary[["model", "window", "wmape_pct"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-input", type=Path, required=True)
    parser.add_argument("--market-input", type=Path, action="append", required=True)
    parser.add_argument("--weather-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
