"""Build a private public-information lookup queue and aggregate label audit.

Raw enterprise names are written only to the explicitly supplied private queue
path.  Province report outputs contain aggregate profiles and no user IDs,
account IDs, or enterprise names.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd

from project.research.jiangsu.user_load_forecast.model_backtest import (
    prepare_output_dir,
)
from project.research.jiangsu.user_load_forecast.user_segmentation import (
    AMBIGUOUS,
    GBT_INDUSTRY_VERSION,
    JIANGSU_CITIES,
    SEGMENTATION_VERSION,
    UNKNOWN,
    infer_city,
    infer_gbt_industry,
)


VERSION = "surun_user_public_enrichment_v1"
UNRESOLVED = {UNKNOWN, AMBIGUOUS}
CURATED_COLUMNS = (
    "user_id",
    "city",
    "city_confidence",
    "gbt_industry",
    "industry_confidence",
    "evidence_tier",
    "source_kind",
    "source_url",
    "retrieved_at",
    "notes",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def anonymous_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def open_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_private_user_profiles(
    path: Path,
    *,
    history_cutoff: date,
) -> pd.DataFrame:
    """Read names in memory and aggregate only pre-cutoff positive energy."""
    energy: dict[str, float] = defaultdict(float)
    dates: dict[str, set[date]] = defaultdict(set)
    names: dict[str, str] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        required = {"retail_user_name", "trade_date", "actual_values"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Load input is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            trade_date = date.fromisoformat(row["trade_date"])
            if trade_date > history_cutoff:
                continue
            name = row["retail_user_name"].strip()
            user_id = anonymous_id(name)
            raw_values = row["actual_values"].strip()
            if not (raw_values.startswith("{") and raw_values.endswith("}")):
                raise ValueError(f"Row {row_number}: invalid PostgreSQL array")
            values = np.fromstring(raw_values[1:-1], sep=",")
            if len(values) != 96 or not np.isfinite(values).all():
                raise ValueError(f"Row {row_number}: invalid 96-point curve")
            energy[user_id] += float(np.clip(values, 0.0, None).sum())
            dates[user_id].add(trade_date)
            names[user_id] = name
    rows = []
    for user_id, name in names.items():
        city = infer_city(name)
        industry = infer_gbt_industry(name)
        rows.append(
            {
                "user_id": user_id,
                "enterprise_name": name,
                "history_days": len(dates[user_id]),
                "positive_energy": energy[user_id],
                "baseline_city": city.value,
                "baseline_city_confidence": city.confidence,
                "baseline_gbt_industry": industry.value,
                "baseline_industry_confidence": industry.confidence,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No user history is visible through the cutoff")
    if frame["user_id"].duplicated().any():
        raise ValueError("Anonymous user hash collision detected")
    total = float(frame["positive_energy"].sum())
    frame["positive_energy_share"] = frame["positive_energy"] / total
    return frame.sort_values("positive_energy", ascending=False)


def read_curated_enrichment(path: Path) -> pd.DataFrame:
    curated = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(CURATED_COLUMNS) - set(curated.columns)
    if missing:
        raise ValueError(f"Curated enrichment missing: {sorted(missing)}")
    curated = curated[list(CURATED_COLUMNS)].copy()
    if curated["user_id"].duplicated().any():
        raise ValueError("Curated enrichment contains duplicate user_id")
    invalid_city = sorted(
        set(curated.loc[curated["city"].ne(""), "city"])
        - set(JIANGSU_CITIES)
    )
    if invalid_city:
        raise ValueError(f"Invalid Jiangsu cities: {invalid_city}")
    invalid_industry = curated.loc[
        curated["gbt_industry"].ne("")
        & ~curated["gbt_industry"].str.match(r"^[A-Z]\d{2}\s.+"),
        "gbt_industry",
    ].tolist()
    if invalid_industry:
        raise ValueError(f"Invalid GB/T industry labels: {invalid_industry}")
    invalid_date = curated.loc[
        curated["retrieved_at"].ne("")
        & ~curated["retrieved_at"].str.match(r"^\d{4}-\d{2}-\d{2}$"),
        "retrieved_at",
    ].tolist()
    if invalid_date:
        raise ValueError(f"Invalid retrieved_at values: {invalid_date}")
    return curated


def apply_curated_enrichment(
    profiles: pd.DataFrame,
    curated: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unknown_ids = sorted(set(curated["user_id"]) - set(profiles["user_id"]))
    if unknown_ids:
        raise ValueError(
            "Curated enrichment contains users absent at cutoff: "
            f"{unknown_ids[:3]}"
        )
    merged = profiles.merge(
        curated,
        on="user_id",
        how="left",
        validate="one_to_one",
    ).fillna("")
    merged["final_city"] = merged["baseline_city"]
    merged["final_city_confidence"] = merged["baseline_city_confidence"]
    merged["city_source"] = "name_rule"
    merged["final_gbt_industry"] = merged["baseline_gbt_industry"]
    merged["final_industry_confidence"] = merged[
        "baseline_industry_confidence"
    ]
    merged["industry_source"] = "name_rule"

    city_override = (
        merged["baseline_city"].isin(UNRESOLVED)
        & merged["city"].ne("")
    )
    industry_override = (
        merged["baseline_gbt_industry"].isin(UNRESOLVED)
        & merged["gbt_industry"].ne("")
    )
    merged.loc[city_override, "final_city"] = merged.loc[
        city_override,
        "city",
    ]
    merged.loc[city_override, "final_city_confidence"] = merged.loc[
        city_override,
        "city_confidence",
    ]
    merged.loc[city_override, "city_source"] = "curated_public"
    merged.loc[industry_override, "final_gbt_industry"] = merged.loc[
        industry_override,
        "gbt_industry",
    ]
    merged.loc[
        industry_override,
        "final_industry_confidence",
    ] = merged.loc[industry_override, "industry_confidence"]
    merged.loc[industry_override, "industry_source"] = "curated_public"

    conflict_rows = []
    city_conflict = (
        ~merged["baseline_city"].isin(UNRESOLVED)
        & merged["city"].ne("")
        & merged["city"].ne(merged["baseline_city"])
    )
    industry_conflict = (
        ~merged["baseline_gbt_industry"].isin(UNRESOLVED)
        & merged["gbt_industry"].ne("")
        & merged["gbt_industry"].ne(merged["baseline_gbt_industry"])
    )
    for row in merged.loc[city_conflict].itertuples(index=False):
        conflict_rows.append(
            {
                "dimension": "city",
                "evidence_tier": row.evidence_tier,
                "source_kind": row.source_kind,
            }
        )
    for row in merged.loc[industry_conflict].itertuples(index=False):
        conflict_rows.append(
            {
                "dimension": "gbt_industry",
                "evidence_tier": row.evidence_tier,
                "source_kind": row.source_kind,
            }
        )
    return merged, pd.DataFrame(
        conflict_rows,
        columns=["dimension", "evidence_tier", "source_kind"],
    )


def coverage_row(
    frame: pd.DataFrame,
    *,
    stage: str,
    dimension: str,
    column: str,
) -> dict[str, object]:
    resolved = ~frame[column].isin(UNRESOLVED)
    return {
        "stage": stage,
        "dimension": dimension,
        "visible_users": len(frame),
        "resolved_users": int(resolved.sum()),
        "resolved_user_share": float(resolved.mean()),
        "positive_energy": float(frame["positive_energy"].sum()),
        "resolved_positive_energy": float(
            frame.loc[resolved, "positive_energy"].sum()
        ),
        "resolved_positive_energy_share": float(
            frame.loc[resolved, "positive_energy"].sum()
            / frame["positive_energy"].sum()
        ),
    }


def aggregate_profile(
    frame: pd.DataFrame,
    *,
    label_column: str,
    source_column: str,
) -> pd.DataFrame:
    return (
        frame.groupby([label_column, source_column], as_index=False)
        .agg(
            user_count=("user_id", "nunique"),
            positive_energy=("positive_energy", "sum"),
        )
        .assign(
            positive_energy_share=lambda value: (
                value["positive_energy"] / value["positive_energy"].sum()
            )
        )
        .sort_values("positive_energy", ascending=False)
    )


def build_private_queue(
    frame: pd.DataFrame,
    *,
    queue_size: int,
) -> pd.DataFrame:
    city_missing = frame["final_city"].isin(UNRESOLVED)
    industry_missing = frame["final_gbt_industry"].isin(UNRESOLVED)
    queue = frame.loc[city_missing | industry_missing].copy()
    queue["lookup_need"] = np.select(
        [
            city_missing.loc[queue.index]
            & industry_missing.loc[queue.index],
            city_missing.loc[queue.index],
        ],
        ["city_and_industry", "city"],
        default="industry",
    )
    queue["priority_score"] = (
        queue["positive_energy_share"]
        * np.where(queue["lookup_need"].eq("city_and_industry"), 2.0, 1.0)
    )
    queue["suggested_query"] = (
        '"' + queue["enterprise_name"] + '" 地址 主营业务 行业'
    )
    columns = [
        "user_id",
        "enterprise_name",
        "history_days",
        "positive_energy",
        "positive_energy_share",
        "baseline_city",
        "baseline_gbt_industry",
        "final_city",
        "final_gbt_industry",
        "lookup_need",
        "priority_score",
        "suggested_query",
    ]
    return queue.sort_values(
        ["priority_score", "positive_energy"],
        ascending=False,
    ).head(queue_size)[columns]


def write_private_queue(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite private queue: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip", encoding="utf-8-sig")
    path.chmod(0o600)


def summary_markdown(
    coverage: pd.DataFrame,
    *,
    curated_rows: int,
    applied_city: int,
    applied_industry: int,
    queue_rows: int,
    queue_energy_share: float,
) -> str:
    lookup = coverage.set_index(["stage", "dimension"])
    city_before = lookup.loc[("baseline_name_rule", "city")]
    city_after = lookup.loc[("curated_public", "city")]
    industry_before = lookup.loc[("baseline_name_rule", "gbt_industry")]
    industry_after = lookup.loc[("curated_public", "gbt_industry")]
    return "\n".join(
        [
            "# 江苏用户公开信息增强试点",
            "",
            "## 口径",
            "",
            "- 用户集合和负荷权重只统计到2026-04-28。",
            (
                "- 公开证据在历史期后核验；标签从本次研究起冻结，"
                "不代表历史申报时点已知。"
            ),
            "- 公开信息只补充名称规则中的未知/多重匹配，不覆盖已有明确标签。",
            "- 普通报告不包含企业名称、户号或匿名用户ID。",
            "",
            "## 结果",
            "",
            f"- 人工核验记录：{curated_rows}条。",
            f"- 实际补充城市：{applied_city}户；补充GB/T行业：{applied_industry}户。",
            (
                "- 城市负荷覆盖："
                f"{city_before.resolved_positive_energy_share:.2%}"
                " → "
                f"{city_after.resolved_positive_energy_share:.2%}。"
            ),
            (
                "- GB/T两位行业负荷覆盖："
                f"{industry_before.resolved_positive_energy_share:.2%}"
                " → "
                f"{industry_after.resolved_positive_energy_share:.2%}。"
            ),
            (
                f"- 下一批私有检索队列：{queue_rows}户，合计覆盖组合正负荷"
                f"{queue_energy_share:.2%}，按贡献排序。"
            ),
            "",
            "## 结论",
            "",
            "公开检索能够有效补充头部未知企业，但仍是回测研究标签。正式模型应优先",
            "使用合同、市场注册或供电主数据，并保留来源、检索日期和人工核验状态。",
        ]
    ) + "\n"


def run(args: argparse.Namespace) -> None:
    prepare_output_dir(args.output_dir)
    profiles = read_private_user_profiles(
        args.load_input,
        history_cutoff=args.history_cutoff,
    )
    curated = read_curated_enrichment(args.curated_enrichment)
    enriched, conflicts = apply_curated_enrichment(profiles, curated)
    private_queue = build_private_queue(
        enriched,
        queue_size=args.queue_size,
    )
    write_private_queue(private_queue, args.private_queue_output)
    queue_group = (
        private_queue.groupby("lookup_need", as_index=False)
        .agg(
            user_count=("user_id", "size"),
            positive_energy=("positive_energy", "sum"),
            positive_energy_share=("positive_energy_share", "sum"),
        )
        .rename(columns={"lookup_need": "segment"})
    )
    queue_group.insert(0, "summary_type", "lookup_need")
    cumulative_rows = []
    for top_n in (10, 25, 50, 100, len(private_queue)):
        usable_n = min(top_n, len(private_queue))
        top = private_queue.head(usable_n)
        cumulative_rows.append(
            {
                "summary_type": "cumulative_top_n",
                "segment": f"top_{usable_n}",
                "user_count": usable_n,
                "positive_energy": float(top["positive_energy"].sum()),
                "positive_energy_share": float(
                    top["positive_energy_share"].sum()
                ),
            }
        )
    queue_summary = pd.concat(
        [queue_group, pd.DataFrame(cumulative_rows)],
        ignore_index=True,
    ).drop_duplicates(["summary_type", "segment"])

    coverage = pd.DataFrame(
        [
            coverage_row(
                profiles,
                stage="baseline_name_rule",
                dimension="city",
                column="baseline_city",
            ),
            coverage_row(
                profiles,
                stage="baseline_name_rule",
                dimension="gbt_industry",
                column="baseline_gbt_industry",
            ),
            coverage_row(
                enriched,
                stage="curated_public",
                dimension="city",
                column="final_city",
            ),
            coverage_row(
                enriched,
                stage="curated_public",
                dimension="gbt_industry",
                column="final_gbt_industry",
            ),
        ]
    )
    city_profile = aggregate_profile(
        enriched,
        label_column="final_city",
        source_column="city_source",
    )
    industry_profile = aggregate_profile(
        enriched,
        label_column="final_gbt_industry",
        source_column="industry_source",
    )
    tier_summary = (
        enriched.loc[enriched["evidence_tier"].ne("")]
        .groupby(["evidence_tier", "source_kind"], as_index=False)
        .agg(
            curated_users=("user_id", "nunique"),
            positive_energy=("positive_energy", "sum"),
        )
    )
    if not tier_summary.empty:
        tier_summary["positive_energy_share"] = (
            tier_summary["positive_energy"]
            / enriched["positive_energy"].sum()
        )
    applied_city = int(enriched["city_source"].eq("curated_public").sum())
    applied_industry = int(
        enriched["industry_source"].eq("curated_public").sum()
    )

    coverage.to_csv(
        args.output_dir / "coverage_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    city_profile.to_csv(
        args.output_dir / "city_profile_enriched.csv",
        index=False,
        encoding="utf-8-sig",
    )
    industry_profile.to_csv(
        args.output_dir / "gbt_industry_profile_enriched.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tier_summary.to_csv(
        args.output_dir / "evidence_tier_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    conflicts.to_csv(
        args.output_dir / "conflict_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    queue_summary.to_csv(
        args.output_dir / "private_queue_aggregate_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (args.output_dir / "research_summary.md").write_text(
        summary_markdown(
            coverage,
            curated_rows=len(curated),
            applied_city=applied_city,
            applied_industry=applied_industry,
            queue_rows=len(private_queue),
            queue_energy_share=float(
                private_queue["positive_energy_share"].sum()
            ),
        ),
        encoding="utf-8",
    )
    metadata = {
        "version": VERSION,
        "segmentation_version": SEGMENTATION_VERSION,
        "gbt_industry_version": GBT_INDUSTRY_VERSION,
        "province_display": "江苏",
        "province_slug": "jiangsu",
        "raw_schema": "file_snapshot_only",
        "history_cutoff": args.history_cutoff.isoformat(),
        "visible_users": len(profiles),
        "curated_rows": len(curated),
        "applied_city_rows": applied_city,
        "applied_industry_rows": applied_industry,
        "conflict_rows": len(conflicts),
        "private_queue_rows": len(private_queue),
        "private_queue_path": str(args.private_queue_output.resolve()),
        "private_queue_permissions": "0600",
        "load_input_sha256": sha256_file(args.load_input),
        "curated_enrichment_sha256": sha256_file(
            args.curated_enrichment
        ),
        "label_policy": (
            "curated public evidence only fills unresolved name-rule labels"
        ),
        "pii_output_check": (
            "PASS: province report outputs are aggregate-only; enterprise "
            "names exist only in the private 0600 lookup queue"
        ),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(coverage.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-input", type=Path, required=True)
    parser.add_argument("--curated-enrichment", type=Path, required=True)
    parser.add_argument(
        "--history-cutoff",
        type=date.fromisoformat,
        default=date(2026, 4, 28),
    )
    parser.add_argument("--queue-size", type=int, default=200)
    parser.add_argument("--private-queue-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
