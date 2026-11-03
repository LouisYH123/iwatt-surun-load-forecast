"""Anonymous geographic and industry proxies for Jiangsu retail users.

The raw load snapshot contains a retail-user name but no authoritative city or
industry code.  This module therefore treats name-derived labels as research
proxies only.  It never exports the source names or account identifiers.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd


SEGMENTATION_VERSION = "jiangsu_name_proxy_v1"
GBT_INDUSTRY_VERSION = "GBT4754_2017_AMENDMENT1_NAME_PROXY_V1"
UNKNOWN = "未知"
AMBIGUOUS = "多重匹配"

JIANGSU_CITIES = (
    "南京",
    "无锡",
    "徐州",
    "常州",
    "苏州",
    "南通",
    "连云港",
    "淮安",
    "盐城",
    "扬州",
    "镇江",
    "泰州",
    "宿迁",
)

# County-level names improve coverage when the prefecture name is absent.
# Very ambiguous district names are intentionally omitted.
CITY_COUNTY_TOKENS = {
    "南京": ("玄武", "秦淮", "建邺", "浦口", "栖霞", "雨花台", "江宁", "六合", "溧水", "高淳"),
    "无锡": ("江阴", "宜兴", "梁溪", "锡山", "惠山", "新吴"),
    "徐州": ("丰县", "沛县", "睢宁", "新沂", "邳州", "铜山", "贾汪", "泉山", "云龙"),
    "常州": ("溧阳", "金坛", "武进", "天宁"),
    "苏州": ("张家港", "昆山", "太仓", "常熟", "吴江", "吴中", "相城", "姑苏", "虎丘"),
    "南通": ("如皋", "启东", "海安", "如东", "崇川", "海门"),
    "连云港": ("灌云", "灌南", "赣榆"),
    "淮安": ("涟水", "盱眙", "金湖", "洪泽", "淮阴", "清江浦"),
    "盐城": ("东台", "大丰", "响水", "滨海", "阜宁", "射阳", "建湖", "亭湖", "盐都"),
    "扬州": ("宝应", "高邮", "仪征", "江都", "邗江", "广陵"),
    "镇江": ("丹阳", "扬中", "句容", "丹徒", "京口", "润州"),
    "泰州": ("兴化", "靖江", "泰兴", "姜堰", "海陵", "高港"),
    "宿迁": ("沭阳", "泗阳", "泗洪", "宿豫", "宿城"),
}

# Avoid generic terms such as “科技” and “实业”; they create attractive but
# unreliable industry labels.  Multiple matched sectors remain ambiguous.
INDUSTRY_TOKENS = {
    "金属冶炼及制品": (
        "钢铁",
        "不锈钢",
        "铝业",
        "铜业",
        "金属",
        "合金",
        "铸造",
        "锻造",
        "紧固件",
        "五金",
        "电镀",
    ),
    "化工及新材料": (
        "化工",
        "化学",
        "新材料",
        "树脂",
        "涂料",
        "颜料",
        "染料",
        "农药",
        "化纤",
        "工业气体",
    ),
    "纺织服装": (
        "纺织",
        "织造",
        "印染",
        "服饰",
        "服装",
        "针织",
        "家纺",
        "纱线",
        "制衣",
    ),
    "机械装备": (
        "机械",
        "机电",
        "装备",
        "电机",
        "泵业",
        "阀门",
        "轴承",
        "工具",
        "模具",
        "汽配",
        "汽车零部件",
    ),
    "电子电气": (
        "电子",
        "电器",
        "电气",
        "电缆",
        "线缆",
        "半导体",
        "光电",
        "通信",
        "集成电路",
    ),
    "建材非金属": (
        "建材",
        "水泥",
        "混凝土",
        "玻璃",
        "陶瓷",
        "石材",
        "砖瓦",
        "耐火",
        "砂浆",
    ),
    "食品农副": (
        "食品",
        "饮料",
        "粮油",
        "饲料",
        "乳业",
        "酒业",
        "农产品",
        "屠宰",
        "水产",
    ),
    "医药健康": (
        "医药",
        "药业",
        "医疗",
        "制药",
        "医院",
        "生物医药",
        "生物制药",
    ),
    "造纸包装印刷": ("造纸", "纸业", "包装", "印刷", "纸品"),
    "橡塑制品": ("塑料", "塑胶", "橡胶", "轮胎"),
    "木材家具": ("木业", "家具", "家居", "木材", "地板"),
    "环保水务": ("环保", "环境工程", "污水", "水务", "再生资源", "固废"),
    "交通物流": ("物流", "运输", "港务", "港口", "仓储", "船舶", "航运"),
    "商业公共服务": (
        "商贸",
        "商业",
        "酒店",
        "宾馆",
        "学校",
        "大学",
        "学院",
        "物业",
        "文旅",
    ),
    "能源": ("能源", "电力", "热电", "光伏", "风电", "燃气", "煤炭"),
}

# Two-digit GB/T 4754—2017 major-class proxies.  The mapping uses only specific
# legal-name tokens and is deliberately incomplete.  It is not a substitute
# for the authoritative industry code in a market-registration/user master.
GBT_INDUSTRY_RULES = (
    ("C13", "农副食品加工业", ("粮油", "饲料", "屠宰", "水产", "米业", "面粉", "油脂")),
    ("C14", "食品制造业", ("食品", "乳业", "调味品", "烘焙", "糖果")),
    ("C15", "酒、饮料和精制茶制造业", ("饮料", "酒业", "酿酒", "茶业")),
    ("C17", "纺织业", ("纺织", "织造", "印染", "针织", "家纺", "纱线")),
    ("C18", "纺织服装、服饰业", ("服装", "服饰", "制衣")),
    ("C19", "皮革、毛皮、羽毛及其制品和制鞋业", ("皮革", "鞋业", "羽绒")),
    ("C20", "木材加工和木竹藤棕草制品业", ("木业", "木材", "地板")),
    ("C21", "家具制造业", ("家具", "家居")),
    ("C22", "造纸和纸制品业", ("造纸", "纸业", "纸品")),
    ("C23", "印刷和记录媒介复制业", ("印刷",)),
    ("C24", "文教工美体育和娱乐用品制造业", ("文具", "体育用品", "工艺品")),
    ("C25", "石油煤炭及其他燃料加工业", ("石油", "石化", "焦化")),
    (
        "C26",
        "化学原料和化学制品制造业",
        ("化工", "化学", "树脂", "涂料", "颜料", "染料", "农药", "肥料", "工业气体"),
    ),
    ("C27", "医药制造业", ("医药", "药业", "制药", "生物医药", "生物制药")),
    ("C28", "化学纤维制造业", ("化纤", "纤维制造")),
    ("C29", "橡胶和塑料制品业", ("塑料", "塑胶", "橡胶", "轮胎")),
    (
        "C30",
        "非金属矿物制品业",
        ("水泥", "混凝土", "玻璃", "陶瓷", "石材", "砖瓦", "耐火", "砂浆"),
    ),
    ("C31", "黑色金属冶炼和压延加工业", ("钢铁", "不锈钢", "轧钢")),
    ("C32", "有色金属冶炼和压延加工业", ("铝业", "铜业", "有色金属")),
    (
        "C33",
        "金属制品业",
        ("金属制品", "五金", "紧固件", "电镀", "铸造", "锻造"),
    ),
    (
        "C34",
        "通用设备制造业",
        ("通用设备", "机械", "泵业", "阀门", "轴承", "工具", "模具"),
    ),
    (
        "C35",
        "专用设备制造业",
        ("专用设备", "环保设备", "纺织机械", "矿山机械", "包装机械"),
    ),
    ("C36", "汽车制造业", ("汽车零部件", "汽配", "汽车制造")),
    ("C37", "铁路船舶航空航天和其他运输设备制造业", ("船舶", "轨道交通", "航空")),
    (
        "C38",
        "电气机械和器材制造业",
        ("电机", "电器", "电气", "电缆", "线缆", "电池"),
    ),
    (
        "C39",
        "计算机通信和其他电子设备制造业",
        ("电子", "半导体", "光电", "通信", "集成电路", "计算机"),
    ),
    ("C40", "仪器仪表制造业", ("仪器", "仪表", "计量")),
    ("C42", "废弃资源综合利用业", ("再生资源", "资源回收", "废弃物利用")),
    ("D44", "电力热力生产和供应业", ("电力", "热电", "供热")),
    ("D45", "燃气生产和供应业", ("燃气", "天然气")),
    ("D46", "水的生产和供应业", ("水务", "供水", "污水处理")),
    ("E47", "房屋建筑业", ("建筑", "建设工程")),
    ("G54", "道路运输业", ("道路运输", "汽车运输")),
    ("G55", "水上运输业", ("航运", "水运")),
    ("G59", "装卸搬运和仓储业", ("物流", "仓储", "港务", "港口")),
    ("K70", "房地产业", ("房地产", "置业")),
    ("L72", "商务服务业", ("商业管理", "酒店", "宾馆", "物业")),
    ("M74", "专业技术服务业", ("检测", "检验", "认证", "勘察设计")),
    ("N77", "生态保护和环境治理业", ("环保", "环境治理", "固废")),
    ("P83", "教育", ("学校", "大学", "学院")),
    ("Q84", "卫生", ("医院", "医疗")),
)


@dataclass(frozen=True)
class ProxyLabel:
    value: str
    confidence: str


@dataclass
class SegmentedUserPanel:
    """Anonymous user curves with name-derived research labels."""

    curves: dict[str, dict[date, np.ndarray]]
    city: dict[str, ProxyLabel]
    industry: dict[str, ProxyLabel]
    gbt_industry: dict[str, ProxyLabel]
    negative_users: set[str]

    @property
    def user_count(self) -> int:
        return len(self.curves)


def _open_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _anonymous_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def infer_city(name: str) -> ProxyLabel:
    """Infer a Jiangsu prefecture without guessing across multiple matches."""
    direct = [city for city in JIANGSU_CITIES if city in name]
    if len(direct) == 1:
        return ProxyLabel(direct[0], "HIGH_NAME_EXPLICIT")
    if len(direct) > 1:
        return ProxyLabel(AMBIGUOUS, "LOW_AMBIGUOUS")

    county = [
        city
        for city, tokens in CITY_COUNTY_TOKENS.items()
        if any(token in name for token in tokens)
    ]
    county = sorted(set(county))
    if len(county) == 1:
        return ProxyLabel(county[0], "MEDIUM_COUNTY_NAME")
    if len(county) > 1:
        return ProxyLabel(AMBIGUOUS, "LOW_AMBIGUOUS")
    return ProxyLabel(UNKNOWN, "NONE")


def infer_industry(name: str) -> ProxyLabel:
    """Infer a coarse sector from specific legal-name keywords."""
    matches = [
        industry
        for industry, tokens in INDUSTRY_TOKENS.items()
        if any(token in name for token in tokens)
    ]
    if len(matches) == 1:
        return ProxyLabel(matches[0], "MEDIUM_NAME_KEYWORD")
    if len(matches) > 1:
        return ProxyLabel(AMBIGUOUS, "LOW_AMBIGUOUS")
    return ProxyLabel(UNKNOWN, "NONE")


def infer_gbt_industry(name: str) -> ProxyLabel:
    """Infer a two-digit GB/T major class using the most specific name token."""
    matches = []
    for code, industry, tokens in GBT_INDUSTRY_RULES:
        matched = [token for token in tokens if token in name]
        if matched:
            matches.append((code, industry, max(len(token) for token in matched)))
    if not matches:
        return ProxyLabel(UNKNOWN, "NONE")
    longest = max(item[2] for item in matches)
    best = [(code, industry) for code, industry, length in matches if length == longest]
    best = sorted(set(best))
    if len(best) != 1:
        return ProxyLabel(AMBIGUOUS, "LOW_AMBIGUOUS")
    code, industry = best[0]
    return ProxyLabel(f"{code} {industry}", "MEDIUM_NAME_KEYWORD")


def read_segmented_user_panel(path: Path) -> SegmentedUserPanel:
    """Read the raw snapshot without retaining or exporting source identities."""
    path = path.resolve()
    curves: dict[str, dict[date, np.ndarray]] = defaultdict(
        lambda: defaultdict(lambda: np.zeros(96, dtype=float))
    )
    city: dict[str, ProxyLabel] = {}
    industry: dict[str, ProxyLabel] = {}
    gbt_industry: dict[str, ProxyLabel] = {}
    negative_users: set[str] = set()

    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            name = row["retail_user_name"].strip()
            user_id = _anonymous_id(name)
            trade_date = date.fromisoformat(row["trade_date"])
            raw = row["actual_values"].strip()
            if not (raw.startswith("{") and raw.endswith("}")):
                raise ValueError(f"Row {row_number}: invalid PostgreSQL array")
            values = np.asarray([float(item) for item in raw[1:-1].split(",")])
            if len(values) != 96 or not np.isfinite(values).all():
                raise ValueError(f"Row {row_number}: invalid 96-point curve")
            curves[user_id][trade_date] += values
            if user_id not in city:
                city[user_id] = infer_city(name)
                industry[user_id] = infer_industry(name)
                gbt_industry[user_id] = infer_gbt_industry(name)
            if np.any(values < 0):
                negative_users.add(user_id)

    return SegmentedUserPanel(
        curves={user_id: dict(value) for user_id, value in curves.items()},
        city=city,
        industry=industry,
        gbt_industry=gbt_industry,
        negative_users=negative_users,
    )


def _visible_positive_energy(
    panel: SegmentedUserPanel,
    cutoff: date,
) -> dict[str, float]:
    values = {}
    for user_id, curves in panel.curves.items():
        total = sum(
            max(float(curve.sum()), 0.0)
            for trade_date, curve in curves.items()
            if trade_date <= cutoff
        )
        if total > 0:
            values[user_id] = total
    return values


def segment_profile(
    panel: SegmentedUserPanel,
    *,
    dimension: str,
    cutoff: date,
) -> pd.DataFrame:
    """Return aggregate-only segment coverage through a visibility cutoff."""
    if dimension not in {"city", "industry", "gbt_industry"}:
        raise ValueError("dimension must be city, industry, or gbt_industry")
    labels: dict[str, ProxyLabel] = getattr(panel, dimension)
    energy = _visible_positive_energy(panel, cutoff)
    total = sum(energy.values())
    rows = []
    for value in sorted(set(label.value for label in labels.values())):
        users = [user_id for user_id, label in labels.items() if label.value == value]
        visible_users = [user_id for user_id in users if user_id in energy]
        segment_energy = sum(energy[user_id] for user_id in visible_users)
        confidences = defaultdict(int)
        for user_id in users:
            confidences[labels[user_id].confidence] += 1
        rows.append(
            {
                "dimension": dimension,
                "segment": value,
                "user_count": len(users),
                "visible_user_count": len(visible_users),
                "positive_energy": segment_energy,
                "positive_energy_share": segment_energy / total if total else 0.0,
                "confidence_counts": ";".join(
                    f"{key}:{confidences[key]}" for key in sorted(confidences)
                ),
                "history_cutoff": cutoff,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["positive_energy_share", "segment"],
        ascending=[False, True],
    )


def city_load_weights(
    panel: SegmentedUserPanel,
    *,
    cutoff: date,
    user_subset: set[str] | None = None,
) -> tuple[dict[str, float], dict]:
    """Build leakage-safe city weights from positive load through ``cutoff``.

    Unresolved users are spread equally over all 13 cities.  This preserves
    their contribution without pretending to know their location.
    """
    energy = _visible_positive_energy(panel, cutoff)
    if user_subset is not None:
        energy = {
            user_id: value
            for user_id, value in energy.items()
            if user_id in user_subset
        }
    total = sum(energy.values())
    if total <= 0:
        raise ValueError(f"No positive visible energy through {cutoff}")

    known = {city: 0.0 for city in JIANGSU_CITIES}
    unresolved = 0.0
    for user_id, value in energy.items():
        city = panel.city[user_id].value
        if city in known:
            known[city] += value
        else:
            unresolved += value
    unresolved_per_city = unresolved / len(JIANGSU_CITIES)
    weights = {
        city: (known[city] + unresolved_per_city) / total
        for city in JIANGSU_CITIES
    }
    metadata = {
        "history_cutoff": cutoff.isoformat(),
        "visible_users": len(energy),
        "positive_energy": total,
        "known_city_energy_share": sum(known.values()) / total,
        "unresolved_energy_share": unresolved / total,
        "unknown_allocation": "equal_across_13_cities",
        "weight_sum": sum(weights.values()),
    }
    return weights, metadata
