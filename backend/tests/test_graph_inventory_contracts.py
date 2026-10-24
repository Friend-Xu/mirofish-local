"""图谱实体清点契约：带类型实体数骤降必须让 prepare 失败（掩盖台账 N-34）。

实测现场：同一张图谱、同一批节点（53 个），`filter_defined_entities` 看到的带类型实体
从 **32 掉到 2**——节点还在，标签读不到了。于是 prepare 生成 2 份人设、跑完整场模拟、
出一份报告，全程没有任何一层说出"这次只有 2 个角色"。

证据等级：本文件是**单测**（直接调用清点函数 / 造清点文件）。
"真实图谱的标签会掉"那一刻由 10-10 22:34→22:36 的跑批日志证明，单测只锁规则。
"""

import json
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services import graph_inventory  # noqa: E402
from app.services.graph_inventory import (  # noqa: E402
    GraphInventoryRegression,
    guard_graph_inventory,
)


@pytest.fixture
def inventory_env(tmp_path, monkeypatch):
    monkeypatch.setattr(
        graph_inventory, "inventory_dir", lambda: str(tmp_path / "inventory")
    )
    return tmp_path


def test_first_check_records_high_water(inventory_env):
    record = guard_graph_inventory(
        "graph_a", total_nodes=53, typed_entities=32,
        entity_types=["Person", "Engineer"],
    )
    assert record["high_water_typed"] == 32
    saved = json.loads(
        (inventory_env / "inventory" / "graph_a.json").read_text(encoding="utf-8")
    )
    assert saved["high_water_typed"] == 32
    assert saved["last_total_nodes"] == 53
    assert saved["checks"] == 1


def test_growth_raises_the_high_water(inventory_env):
    guard_graph_inventory("graph_a", total_nodes=53, typed_entities=32, entity_types=["Person"])
    record = guard_graph_inventory("graph_a", total_nodes=73, typed_entities=35,
                                   entity_types=["Person", "Engineer"])
    assert record["high_water_typed"] == 35
    assert record["checks"] == 2


def test_collapse_is_rejected_with_the_numbers(inventory_env):
    """实测现场的数字：节点 53 不变，带类型实体 32 → 2。"""

    guard_graph_inventory(
        "graph_a", total_nodes=53, typed_entities=32,
        entity_types=["Person", "Engineer", "Hrprofessional"],
    )
    with pytest.raises(GraphInventoryRegression) as error:
        guard_graph_inventory(
            "graph_a", total_nodes=53, typed_entities=2, entity_types=["Person"]
        )
    message = str(error.value)
    assert "32" in message and "本次 2 个" in message
    assert "graph_a" in message
    assert "Engineer" in message and "Hrprofessional" in message, "必须列出消失的类型"
    assert "标签读不到" in message, "必须说清是标签损坏而不是实体被删"


def test_rejected_check_does_not_lower_the_high_water(inventory_env):
    guard_graph_inventory("graph_a", total_nodes=53, typed_entities=32, entity_types=["Person"])
    with pytest.raises(GraphInventoryRegression):
        guard_graph_inventory("graph_a", total_nodes=53, typed_entities=2, entity_types=["Person"])
    saved = json.loads(
        (inventory_env / "inventory" / "graph_a.json").read_text(encoding="utf-8")
    )
    assert saved["high_water_typed"] == 32, "被拒绝的那次不允许进入清点"


def test_small_graphs_are_not_judged_by_ratio(inventory_env):
    """只有 2–3 个实体的图谱本来就少，按比例判会把正常的小图判成损坏。"""

    guard_graph_inventory("graph_small", total_nodes=8, typed_entities=3,
                          entity_types=["Person"])
    record = guard_graph_inventory("graph_small", total_nodes=8, typed_entities=2,
                                   entity_types=["Person"])
    assert record["high_water_typed"] == 3


def test_small_dedup_shrink_is_tolerated(inventory_env):
    """实体合并去重会让数量小幅下降，掉一半以内不判损坏。"""

    guard_graph_inventory("graph_a", total_nodes=73, typed_entities=35, entity_types=["Person"])
    record = guard_graph_inventory("graph_a", total_nodes=73, typed_entities=30,
                                   entity_types=["Person"])
    assert record["last_typed"] == 30
    assert record["high_water_typed"] == 35, "小幅下降不抬高水位，但也不清空历史"


def test_graphs_are_tracked_separately(inventory_env):
    guard_graph_inventory("graph_a", total_nodes=53, typed_entities=32, entity_types=["Person"])
    record = guard_graph_inventory("graph_b", total_nodes=10, typed_entities=4,
                                   entity_types=["Person"])
    assert record["high_water_typed"] == 4


def test_inventory_write_is_atomic(inventory_env):
    guard_graph_inventory("graph_a", total_nodes=53, typed_entities=32, entity_types=["Person"])
    leftovers = [
        p.name for p in (inventory_env / "inventory").iterdir()
        if p.name.startswith(".tmp-")
    ]
    assert leftovers == []


def test_empty_graph_id_is_rejected(inventory_env):
    with pytest.raises(ValueError):
        guard_graph_inventory("", total_nodes=1, typed_entities=1, entity_types=[])


def test_guard_is_wired_into_prepare():
    """静态断言：清点必须挂在 prepare 的实体读取之后（唯一的模拟准备路径）。"""

    manager = (BACKEND_DIR / "app" / "services" / "simulation_manager.py").read_text(
        encoding="utf-8"
    )
    assert "guard_graph_inventory(" in manager
    assert "filtered.filtered_count" in manager
    # 必须在 prepare_simulation 里，而不是被搬到某个没人调的角落
    prepare_body = manager.split("def prepare_simulation", 1)[1]
    assert "guard_graph_inventory(" in prepare_body.split("def ", 1)[0]
