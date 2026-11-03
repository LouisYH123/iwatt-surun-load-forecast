"""Backtest D-3-visible portfolio feedback for volatile weekend intervals."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.model_backtest import prepare_output_dir
from project.research.jiangsu.user_load_forecast.pv_sensitive_segment_backtest import (
    STRICT_MARKET_POLICY,
    read_portfolio_curves,
)
from project.research.jiangsu.user_load_forecast.pv_weather_challenger_audit import (
    score_window,
)
from project.research.jiangsu.user_load_forecast.strict_operational_weather_backtest import (
    WEATHER_POLICY,
    daily_metrics,
    sha256_file,
)
from project.research.jiangsu.user_load_forecast.strict_operational_weather_model_sweep import (
    fit_candidate,
    model_factories,
)
from project.research.jiangsu.user_load_forecast.strict_weekend_hierarchical_expert_backtest import (
    SELECTION_FOLDS,
    build_frames,
    gate_mask,
    grouped_window_metrics,
    selection_metrics,
)


VERSION = "surun_strict_weekend_d3_feedback_v1"
ANCHOR_VARIANTS = ("core_weather", "solar_weather")
FEEDBACKS = (
    "feedback_point_ratio",
    "feedback_daily_ratio",
    "feedback_daylight_ratio",
    "feedback_point_daily_blend",
)
GATES = (
    "weekend_all",
    "weekend_morning_25_52",
    "weekend_morning_33_52",
    "weekend_daylight_25_72",
)
BETAS = (0.25, 0.5, 0.75, 1.0)


def attach_d3_feedback(
    frame: pd.DataFrame,
    curves: dict[date, np.ndarray],
) -> pd.DataFrame:
    result = frame.copy()
    calendar_curves = {
        target_date: group.sort_values("trade_interval")[
            "calendar_prediction"
        ].to_numpy(dtype=float)
        for target_date, group in result.groupby("target_date", sort=True)
    }
    available = sorted(set(curves) & set(calendar_curves))
    blocks = []
    for target_date in sorted(result["target_date"].unique()):
        cutoff = target_date - timedelta(days=3)
        sources = [value for value in available if value <= cutoff][-3:]
        if sources:
            point_ratios = np.vstack(
                [
                    np.divide(
                        curves[source],
                        calendar_curves[source],
                        out=np.ones(96, dtype=float),
                        where=np.abs(calendar_curves[source]) > 1e-9,
                    )
                    for source in sources
                ]
            )
            point = np.median(point_ratios, axis=0)
            daily = float(
                np.median(
                    [
                        curves[source].sum() / calendar_curves[source].sum()
                        for source in sources
                    ]
                )
            )
            daylight = float(
                np.median(
                    [
                        curves[source][24:72].sum()
                        / calendar_curves[source][24:72].sum()
                        for source in sources
                    ]
                )
            )
            age = float((target_date - sources[-1]).days)
        else:
            point = np.ones(96, dtype=float)
            daily = 1.0
            daylight = 1.0
            age = np.nan
        blocks.append(
            pd.DataFrame(
                {
                    "target_date": target_date,
                    "trade_interval": np.arange(1, 97),
                    "feedback_point_ratio": point,
                    "feedback_daily_ratio": daily,
                    "feedback_daylight_ratio": daylight,
                    "feedback_source_age_days": age,
                }
            )
        )
    feedback = pd.concat(blocks, ignore_index=True)
    feedback["feedback_point_daily_blend"] = 0.5 * (
        feedback["feedback_point_ratio"] + feedback["feedback_daily_ratio"]
    )
    return result.merge(
        feedback,
        on=["target_date", "trade_interval"],
        how="left",
        validate="one_to_one",
    )


def apply_feedback(
    anchor: pd.DataFrame,
    feedback_frame: pd.DataFrame,
    *,
    model_name: str,
    feedback: str,
    gate: str,
    beta: float,
) -> pd.DataFrame:
    values = feedback_frame[
        ["target_date", "trade_interval", feedback]
    ].drop_duplicates(["target_date", "trade_interval"])
    output = anchor.merge(
        values,
        on=["target_date", "trade_interval"],
        how="left",
        validate="many_to_one",
    )
    mask = gate_mask(output, gate)
    ratio = np.clip(output[feedback].to_numpy(dtype=float), 0.85, 1.15)
    factor = 1.0 + beta * (ratio - 1.0)
    prediction = output["predicted_load"].to_numpy(dtype=float).copy()
    prediction[mask] *= factor[mask]
    output["model"] = model_name
    output["predicted_load"] = prediction
    output["error"] = output["predicted_load"] - output["actual_load"]
    output["absolute_error"] = output["error"].abs()
    return output.drop(columns=feedback)


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    frame, variants = build_frames(args)
    curves = read_portfolio_curves(args.portfolio_input)
    frame = attach_d3_feedback(frame, curves)
    factory = model_factories()["gbdt_huber"]
    candidate_parts: dict[str, list[pd.DataFrame]] = {}
    configs: dict[str, dict[str, object]] = {}

    for fold_index, (train_end, start, end) in enumerate(SELECTION_FOLDS, start=1):
        for anchor_variant in ANCHOR_VARIANTS:
            anchor_name = f"anchor_{anchor_variant}"
            anchor = fit_candidate(
                frame,
                model_name=anchor_name,
                features=variants[anchor_variant],
                estimator_factory=factory,
                train_end=train_end,
                predict_start=start,
            )
            anchor = anchor.loc[anchor["target_date"].le(end)].copy()
            anchor["selection_fold"] = fold_index
            candidate_parts.setdefault(anchor_name, []).append(anchor)
            configs[anchor_name] = {
                "kind": "anchor",
                "anchor_variant": anchor_variant,
            }
            for feedback in FEEDBACKS:
                for gate in GATES:
                    for beta in BETAS:
                        name = (
                            f"feedback_{anchor_variant}__{feedback}__{gate}"
                            f"__b{beta:g}"
                        )
                        prediction = apply_feedback(
                            anchor,
                            frame,
                            model_name=name,
                            feedback=feedback,
                            gate=gate,
                            beta=beta,
                        )
                        prediction["selection_fold"] = fold_index
                        candidate_parts.setdefault(name, []).append(prediction)
                        configs[name] = {
                            "kind": "feedback",
                            "anchor_variant": anchor_variant,
                            "feedback": feedback,
                            "gate": gate,
                            "beta": beta,
                        }

    rows = []
    pooled = {}
    for model, parts in candidate_parts.items():
        prediction = pd.concat(parts, ignore_index=True)
        pooled[model] = prediction
        rows.append({"model": model, **selection_metrics(prediction)})
    selection = pd.DataFrame(rows).sort_values(
        ["wmape_pct", "weekend_wmape_pct", "daily_wmape_p90_pct", "model"]
    )
    selected = str(selection.iloc[0]["model"])
    config = configs[selected]

    final_anchors = {
        variant: fit_candidate(
            frame,
            model_name=f"anchor_{variant}",
            features=variants[variant],
            estimator_factory=factory,
            train_end=args.final_train_end,
            predict_start=args.holdout_start,
        )
        for variant in ANCHOR_VARIANTS
    }
    if config["kind"] == "anchor":
        selected_final = final_anchors[str(config["anchor_variant"])].copy()
        selected_final["model"] = "weekend_feedback_selected"
    else:
        selected_final = apply_feedback(
            final_anchors[str(config["anchor_variant"])],
            frame,
            model_name="weekend_feedback_selected",
            feedback=str(config["feedback"]),
            gate=str(config["gate"]),
            beta=float(config["beta"]),
        )
    predictions = pd.concat([*final_anchors.values(), selected_final], ignore_index=True)

    summary_rows = []
    group_rows = []
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
        group_rows.extend(
            grouped_window_metrics(predictions, window=window, start=start, end=end)
        )
    summary = pd.DataFrame(summary_rows).sort_values(["window", "wmape_pct", "model"])
    groups = pd.DataFrame(group_rows).sort_values(
        ["window", "group", "wmape_pct", "model"]
    )
    daily = daily_metrics(predictions)

    anchor_name = f"anchor_{config['anchor_variant']}"
    def group_value(model: str, group: str) -> float:
        return float(
            groups.loc[
                groups["window"].eq("legacy_holdout")
                & groups["group"].eq(group)
                & groups["model"].eq(model),
                "wmape_pct",
            ].iloc[0]
        )

    selected_overall = group_value("weekend_feedback_selected", "all")
    anchor_overall = group_value(anchor_name, "all")
    selected_weekend = group_value("weekend_feedback_selected", "weekend")
    anchor_weekend = group_value(anchor_name, "weekend")
    promotion = bool(
        config["kind"] == "feedback"
        and selected_overall < anchor_overall
        and selected_weekend < anchor_weekend
    )

    pooled[selected].to_csv(
        args.output_dir / "selected_oof_predictions_96.csv.gz",
        index=False,
        compression="gzip",
    )
    predictions.to_csv(
        args.output_dir / "predictions_96.csv.gz", index=False, compression="gzip"
    )
    selection.to_csv(
        args.output_dir / "selection_summary.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        args.output_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig"
    )
    groups.to_csv(
        args.output_dir / "group_metrics.csv", index=False, encoding="utf-8-sig"
    )
    daily.to_csv(
        args.output_dir / "daily_metrics.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "version": VERSION,
        "built_at": datetime.now().astimezone().isoformat(),
        "status": "PROMOTION_CANDIDATE" if promotion else "NO_PROMOTION",
        "selection_policy": "THREE_PRE_HOLDOUT_ROLLING_ORIGIN_FOLDS_ONLY",
        "candidate_count": len(configs),
        "selected": selected,
        "selected_config": config,
        "selected_oof_metrics": selection.iloc[0].to_dict(),
        "legacy_selected_wmape_pct": selected_overall,
        "legacy_anchor_wmape_pct": anchor_overall,
        "legacy_selected_weekend_wmape_pct": selected_weekend,
        "legacy_anchor_weekend_wmape_pct": anchor_weekend,
        "promotion_recommended": promotion,
        "load_feedback_policy": "LATEST_THREE_ACTUAL_DATES_AT_OR_BEFORE_TARGET_MINUS_3",
        "market_policy": STRICT_MARKET_POLICY,
        "weather_policy": WEATHER_POLICY,
        "target_day_da_feature_check": "PASS_NO_TARGET_DAY_DA",
        "portfolio_input": str(args.portfolio_input.resolve()),
        "portfolio_sha256": sha256_file(args.portfolio_input),
        "market_inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.market_input
        ],
        "weather_input": str(args.weather_input.resolve()),
        "weather_sha256": sha256_file(args.weather_input),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (args.output_dir / "run_summary.md").write_text(
        "# 江苏用户负荷：周末 D-3 组合残差反馈\n\n"
        f"- 滚动验证比较 {len(configs)} 个候选。\n"
        f"- 选择：`{selected}`。\n"
        f"- 41天整体：反馈 {selected_overall:.4f}%，锚点 {anchor_overall:.4f}%。\n"
        f"- 41天周末：反馈 {selected_weekend:.4f}%，锚点 {anchor_weekend:.4f}%。\n"
        f"- 结论：{'建议进入下一轮门禁' if promotion else '不晋级'}。\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
    print(summary[["model", "window", "wmape_pct"]].to_string(index=False))
    print(groups.loc[groups["group"].isin(["workday", "weekend", "holiday"])][
        ["model", "window", "group", "wmape_pct", "bias_pct"]
    ].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-input", type=Path, required=True)
    parser.add_argument("--market-input", type=Path, action="append", required=True)
    parser.add_argument("--weather-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
        "--new-diagnostic-end", type=date.fromisoformat, default=date(2026, 8, 5)
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
