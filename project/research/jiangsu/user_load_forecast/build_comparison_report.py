"""Build one comparison table from completed Surun backtest artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_rows(backtest_root: Path) -> list[dict]:
    rows = []
    baseline = pd.read_csv(
        backtest_root / "baseline_v1_20260501_20260725/summary_metrics.csv",
        encoding="utf-8-sig",
    )
    for item in baseline.loc[baseline["segment_type"].eq("overall")].to_dict(
        orient="records"
    ):
        rows.append(
            {
                "track": "strict",
                "method": item["model"],
                "weather_policy": "NONE",
                "champion_eligible": True,
                "wmape_pct": item["wmape_pct"],
                "daily_wmape_p90_pct": None,
                "worst_date": None,
                "notes": "deterministic baseline",
            }
        )

    calendar = _json(
        backtest_root
        / "calendar_baseline_v1_20260501_20260725/run_metadata.json"
    )
    rows.append(
        {
            "track": "strict",
            "method": "calendar_similar_day_mean4",
            "weather_policy": "NONE",
            "champion_eligible": True,
            "wmape_pct": calendar["overall_wmape_pct"],
            "daily_wmape_p90_pct": calendar["daily_wmape_p90_pct"],
            "worst_date": calendar["worst_date"],
            "notes": "holiday/workday/weekend/adjusted-workday aware",
        }
    )

    for directory, track, policy, eligible in (
        (
            "model_pool_strict_v1_20260501_20260725",
            "strict",
            "NONE",
            True,
        ),
        (
            "model_pool_weather_proxy_v1_20260501_20260725",
            "weather_proxy",
            "RESEARCH_PROXY_UNVERIFIED_ISSUE_TIME",
            False,
        ),
    ):
        summary_path = backtest_root / directory / "summary_metrics.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path, encoding="utf-8-sig")
        for item in summary.loc[
            summary["segment_type"].eq("overall")
        ].to_dict(orient="records"):
            rows.append(
                {
                    "track": track,
                    "method": item["model"],
                    "weather_policy": policy,
                    "champion_eligible": eligible,
                    "wmape_pct": item["wmape_pct"],
                    "daily_wmape_p90_pct": item["daily_wmape_p90_pct"],
                    "worst_date": item["worst_date"],
                    "notes": f"{item['feature_variant']} / {item['target_mode']}",
                }
            )

    user = _json(
        backtest_root
        / "user_structure_v1_20260501_20260725/run_metadata.json"
    )
    rows.append(
        {
            "track": "strict",
            "method": "topn75_longtail_trend",
            "weather_policy": "NONE",
            "champion_eligible": True,
            "wmape_pct": user["overall_wmape_pct"],
            "daily_wmape_p90_pct": None,
            "worst_date": None,
            "notes": user["user_status_policy"],
        }
    )

    for directory, track, policy in (
        (
            "fusion_strict_calendar_v1_20260501_20260725",
            "strict",
            "NONE",
        ),
        (
            "fusion_weather_proxy_calendar_v1_20260501_20260725",
            "weather_proxy",
            "RESEARCH_PROXY_UNVERIFIED_ISSUE_TIME",
        ),
    ):
        metadata_path = backtest_root / directory / "run_metadata.json"
        if not metadata_path.exists():
            continue
        metadata = _json(metadata_path)
        rows.append(
            {
                "track": track,
                "method": "dynamic_nonnegative_fusion",
                "weather_policy": policy,
                "champion_eligible": metadata["champion_eligible"],
                "wmape_pct": metadata["wmape_pct"],
                "daily_wmape_p90_pct": metadata["daily_wmape_p90_pct"],
                "worst_date": metadata["worst_date"],
                "notes": f"{metadata['candidate_count']} candidates",
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(build_rows(args.backtest_root)).sort_values(
        ["track", "wmape_pct", "method"]
    )
    frame.to_csv(
        args.output_dir / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    strict = frame.loc[frame["champion_eligible"].eq(True)].sort_values("wmape_pct")
    proxy = frame.loc[frame["track"].eq("weather_proxy")].sort_values("wmape_pct")
    lines = [
        "# 苏润组合负荷研究比较 v1",
        "",
        "## 严格可晋级轨",
        "",
        "| 方法 | WMAPE | P90日WMAPE |",
        "|---|---:|---:|",
    ]
    for item in strict.itertuples(index=False):
        p90 = (
            f"{item.daily_wmape_p90_pct:.3f}%"
            if pd.notna(item.daily_wmape_p90_pct)
            else "—"
        )
        lines.append(f"| {item.method} | {item.wmape_pct:.3f}% | {p90} |")
    lines.extend(
        [
            "",
            "## 天气研究代理轨（不可晋级）",
            "",
            "| 方法 | WMAPE | P90日WMAPE |",
            "|---|---:|---:|",
        ]
    )
    for item in proxy.itertuples(index=False):
        p90 = (
            f"{item.daily_wmape_p90_pct:.3f}%"
            if pd.notna(item.daily_wmape_p90_pct)
            else "—"
        )
        lines.append(f"| {item.method} | {item.wmape_pct:.3f}% | {p90} |")
    lines.extend(
        [
            "",
            "天气代理轨使用的历史表缺少 `forecast_issue_at/available_at`，"
            "只能证明潜在增益，不能作为生产精度结论。",
            "",
            "XGBoost在当前环境中因依赖包缺失且网络不可用未执行；"
            "状态已保存在模型池 `model_status.csv`，未使用其他模型冒充。",
            "",
        ]
    )
    (args.output_dir / "research_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
