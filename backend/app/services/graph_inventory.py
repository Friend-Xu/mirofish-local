"""图谱实体清点：把"图谱里到底有多少个可用角色"变成可核对、可失败的信号。

为什么必须有（`掩盖台账.md` N-34）：同一张图谱、同一批节点（53 个），
`filter_defined_entities` 看到的"带类型实体"从 **32 掉到 2**——节点还在，是节点的
**类型标签**读不到了。于是 prepare 生成 2 份人设、跑完一整场模拟、出一份报告，
全程没有任何一层说出"这次只有 2 个角色"。用户拿到的是"三个角色的推演"，
却以为是场景全貌。

规则（这是边界校验，不是兜底）：

* 每个 `graph_id` 保存一份**高水位**清点（带类型实体数的历史最大值）；
* 本次带类型实体数 < 高水位的一半（且高水位 ≥ 4）→ 抛 `GraphInventoryRegression`，
  错误里写明：图谱节点数、本次带类型数、历史高水位、类型集合差异；
* 否则记录并抬高高水位。

为什么"骤降"可以判死：图谱里的实体只会增不会减（合并去重最多让个位数变化，
图谱删除会换 `graph_id`）。所以数量掉一半以上只可能是标签损坏或读取出错——
这两种情况都不允许拿剩下的实体去跑模拟。
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger("mirofish.graph_inventory")

# 低于这个基数不做比例判断：小图（刚建好、只有两三个实体）本来就少，
# 按比例判会把"正常的小图"判成损坏。
MIN_BASELINE_FOR_RATIO_CHECK = 4


class GraphInventoryRegression(RuntimeError):
    """图谱的带类型实体数相对历史高水位骤降（节点还在，标签没了）。"""


def inventory_dir() -> str:
    return os.path.join(Config.UPLOAD_FOLDER, "graphs", "inventory")


def inventory_path(graph_id: str) -> str:
    if not graph_id:
        raise ValueError("graph_id 不能为空：清点必须挂在具体图谱上")
    safe = graph_id.replace(os.sep, "_").replace("/", "_")
    return os.path.join(inventory_dir(), f"{safe}.json")


def load_inventory(graph_id: str) -> Optional[Dict[str, Any]]:
    """读取该图谱的清点记录；没有记录返回 None（第一次跑）。"""

    path = inventory_path(graph_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_inventory(graph_id: str, record: Dict[str, Any]) -> None:
    """原子写：清点文件是"上一次到底有多少角色"的唯一证据，不能被半截写坏。"""

    path = inventory_path(graph_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle, temp_path = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=".tmp-", suffix=".json"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_path, path)
    except BaseException:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def guard_graph_inventory(
    graph_id: str,
    *,
    total_nodes: int,
    typed_entities: int,
    entity_types: List[str],
) -> Dict[str, Any]:
    """核对并更新该图谱的实体清点；骤降即抛 `GraphInventoryRegression`。

    Returns:
        更新后的清点记录（`high_water_typed` 为历史最大值）。
    """

    previous = load_inventory(graph_id) or {}
    high_water = int(previous.get("high_water_typed") or 0)
    previous_types = list(previous.get("entity_types") or [])

    if (
        high_water >= MIN_BASELINE_FOR_RATIO_CHECK
        and typed_entities < high_water / 2
    ):
        missing = sorted(set(previous_types) - set(entity_types))
        raise GraphInventoryRegression(
            f"图谱 {graph_id} 的带类型实体数骤降：本次 {typed_entities} 个"
            f"（类型 {sorted(entity_types)}），历史高水位 {high_water} 个"
            f"（类型 {sorted(previous_types)}），节点总数 {total_nodes}。"
            + (
                f" 消失的类型：{missing}。"
                if missing
                else ""
            )
            + "节点还在而类型没了，说明实体节点的类型标签读不到了"
            "（不是实体被删除）。用这样的图谱跑模拟会静默缩小角色范围，"
            "因此本次准备直接失败；修好图谱标签后重试。"
        )

    record = {
        "graph_id": graph_id,
        "high_water_typed": max(high_water, typed_entities),
        "last_typed": typed_entities,
        "last_total_nodes": total_nodes,
        "entity_types": sorted(entity_types),
        "last_checked_at": datetime.now().isoformat(),
        "checks": int(previous.get("checks") or 0) + 1,
        "regressions": list(previous.get("regressions") or []),
    }
    _write_inventory(graph_id, record)
    logger.info(
        "图谱实体清点: graph_id=%s 带类型实体=%d（高水位 %d）节点总数=%d 类型=%s",
        graph_id,
        typed_entities,
        record["high_water_typed"],
        total_nodes,
        sorted(entity_types),
    )
    return record
