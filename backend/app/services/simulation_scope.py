"""模拟范围声明：报告必须写明"这次模拟到底跑了什么"。

为什么必须有（验收标准 R3；`掩盖台账.md` N-34 修法②）：

同一个场景，图谱里有 32 个可用实体，prepare 只生成了 2 份人设（N-34），
整场模拟只有 2 个角色跑完；twitter 半场只有广播、没有任何传播路径
（`information_flow.has_information_flow=false`，N-35）。这两件事在报告正文、
`meta.json`、`run-status` 里都不出现——读者拿到的是一份看起来完整的报告，
却无从知道"这次只带了一小部分角色、有一个平台根本没有传播"。
这就是"歪曲"的定义，所以范围声明必须是产物的一部分，而不是可选的附注。

规则（这是边界校验，不是兜底）：

* 所有数字只来自模拟产物（`state.json` / `run_state.json` /
  `<platform>/actions.jsonl` 的 `simulation_start`、`simulation_end`），
  **不经过模型复述**——让模型转述数字正是把真实范围写成另一个数的入口；
* 缺任何一项都抛错，不给默认值：缺少产物意味着"这次模拟的范围无法确定"，
  拿不确定的范围出报告比不出报告更糟；
* 参与者数在两个产物里必须一致（prepare 的人设数 vs `simulation_start`
  记录的实际参与数）：实测 `sim_4267208ff665` 是 21 份人设、实际只跑了
  17 个 agent，这种静默缩小不允许被报告吞掉；
* 未启用的平台写明"未启用"，启用却没有传播的平台写明"不构成传播"——
  沉默必须被显式化，读者才不会以为每个平台都参与了。
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger("mirofish.simulation_scope")

# 平台顺序固定，产物里的顺序才稳定（与 state.json 的字段顺序一致）
PLATFORMS = ("twitter", "reddit")

# `simulation_end.information_flow` 里被声明所引用的字段：缺一个就声明不出来
REQUIRED_FLOW_FIELDS = ("has_timeline", "has_information_flow", "social_edges", "counts")
REQUIRED_FLOW_COUNTS = ("post", "comment", "like", "follow", "rec")


class SimulationScopeUnavailable(RuntimeError):
    """本次模拟的范围无法确定（产物缺失或互相矛盾）。

    报告不允许在没有范围声明的情况下完成：带着不确定的范围交付，
    读者会把"2 个角色的推演"当成场景全貌。
    """


def simulation_dir(simulation_id: str) -> str:
    """模拟产物目录。

    与其他模拟模块同一个根（`Config.UPLOAD_FOLDER/simulations`），
    在调用时读取 `Config`，测试才能把它指到临时目录。
    """

    if not simulation_id:
        raise SimulationScopeUnavailable("simulation_id 为空：范围必须挂在具体模拟上")
    return os.path.join(Config.UPLOAD_FOLDER, "simulations", simulation_id)


def _load_json_object(path: str, what: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise SimulationScopeUnavailable(f"缺少{what}：{path} 不存在")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as error:
        raise SimulationScopeUnavailable(f"{what} 读不动（{path}）：{error}") from error
    if not isinstance(payload, dict):
        raise SimulationScopeUnavailable(f"{what} 不是对象（{path}）")
    return payload


def _require_keys(payload: Dict[str, Any], keys: List[str], what: str) -> None:
    missing = [key for key in keys if payload.get(key) is None]
    if missing:
        raise SimulationScopeUnavailable(f"{what} 缺字段 {missing}")


def _count_action_types(actions: List[Dict[str, Any]]) -> Dict[str, int]:
    """动作构成：按 `action_type` 计数，与上面的"动作数"同源同口径。"""

    counts: Dict[str, int] = {}
    for row in actions:
        name = str(row["action_type"])
        counts[name] = counts.get(name, 0) + 1
    return counts


def _read_actions(actions_path: str, platform: str) -> Dict[str, Any]:
    """读一个平台的 `actions.jsonl`，返回该平台的范围事实。

    动作数 = 带 `action_type` 的行数（`模拟质量验收标准.md` 的同一口径）。
    """

    if not os.path.exists(actions_path):
        raise SimulationScopeUnavailable(
            f"平台 {platform} 已启用，但没有动作日志：{actions_path} 不存在"
        )

    rows: List[Dict[str, Any]] = []
    with open(actions_path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as error:
                raise SimulationScopeUnavailable(
                    f"{actions_path} 第 {line_number} 行不是 JSON：{error}"
                ) from error

    actions = [row for row in rows if "action_type" in row]
    starts = [row for row in rows if row.get("event_type") == "simulation_start"]
    ends = [row for row in rows if row.get("event_type") == "simulation_end"]

    if not starts:
        raise SimulationScopeUnavailable(
            f"平台 {platform} 的动作日志里没有 simulation_start：{actions_path}"
        )
    if not ends:
        raise SimulationScopeUnavailable(
            f"平台 {platform} 的动作日志里没有 simulation_end（模拟没跑完）："
            f"{actions_path}"
        )

    start = starts[-1]
    _require_keys(start, ["agents_count"], f"平台 {platform} 的 simulation_start")
    flow = ends[-1].get("information_flow")
    if not isinstance(flow, dict):
        raise SimulationScopeUnavailable(
            f"平台 {platform} 的 simulation_end 没有 information_flow（该模拟产物"
            f"早于 F12，无法声明这个平台有没有传播）：{actions_path}"
        )
    if not flow.get("exists"):
        raise SimulationScopeUnavailable(
            f"平台 {platform} 的平台库不存在，无法判断有没有传播："
            f"{flow.get('note') or flow.get('db_path')}"
        )
    _require_keys(flow, list(REQUIRED_FLOW_FIELDS), f"平台 {platform} 的 information_flow")
    _require_keys(
        flow.get("counts") or {},
        list(REQUIRED_FLOW_COUNTS),
        f"平台 {platform} 的 information_flow.counts",
    )

    return {
        "actions": len(actions),
        "agents_count": int(start["agents_count"]),
        "rounds_in_start_event": start.get("total_rounds"),
        "action_types": _count_action_types(actions),
        "trace_by_action": {
            str(key): int(value)
            for key, value in (flow.get("trace_by_action") or {}).items()
        },
        "information_flow": {
            "has_timeline": bool(flow["has_timeline"]),
            "has_information_flow": bool(flow["has_information_flow"]),
            "social_edges": int(flow["social_edges"]),
            "counts": {key: flow["counts"][key] for key in REQUIRED_FLOW_COUNTS},
        },
    }


def collect_simulation_scope(simulation_id: str) -> Dict[str, Any]:
    """收集"本次模拟的范围与各平台传播"，任何一项缺失即抛错。

    Returns:
        dict：写进 `meta.json` 的 `simulation_scope`，也是范围声明的唯一来源。
    """

    sim_dir = simulation_dir(simulation_id)

    state = _load_json_object(
        os.path.join(sim_dir, "state.json"), f"模拟状态（{simulation_id}）"
    )
    _require_keys(
        state,
        ["graph_id", "entities_count", "profiles_count", "entity_types",
         "enable_twitter", "enable_reddit"],
        "state.json",
    )

    run_state = _load_json_object(
        os.path.join(sim_dir, "run_state.json"), f"运行状态（{simulation_id}）"
    )
    _require_keys(run_state, ["total_rounds", "current_round"], "run_state.json")

    platforms: Dict[str, Any] = {}
    total_actions = 0
    for platform in PLATFORMS:
        enabled = bool(state[f"enable_{platform}"])
        if not enabled:
            platforms[platform] = {"enabled": False}
            continue
        facts = _read_actions(
            os.path.join(sim_dir, platform, "actions.jsonl"), platform
        )
        if facts["agents_count"] != int(state["profiles_count"]):
            raise SimulationScopeUnavailable(
                f"参与者数在两份产物里矛盾：prepare 生成人设 "
                f"{state['profiles_count']} 份，平台 {platform} 的 "
                f"simulation_start 记录实际参与 {facts['agents_count']} 个 agent。"
                "这种静默缩小不允许被报告当成正常范围声明。"
            )
        total_actions += facts["actions"]
        platforms[platform] = {"enabled": True, **facts}

    scope = {
        "simulation_id": simulation_id,
        "graph_id": str(state["graph_id"]),
        "agents": {
            "entities_count": int(state["entities_count"]),
            "profiles_count": int(state["profiles_count"]),
            "entity_types": list(state["entity_types"]),
        },
        "rounds": {
            "configured": int(run_state["total_rounds"]),
            "executed": int(run_state["current_round"]),
        },
        "total_actions": total_actions,
        "platforms": platforms,
        "gathered_at": datetime.now().isoformat(),
    }

    logger.info(
        "模拟范围: simulation_id=%s agents=%d(类型 %d 类) 动作=%d 平台=%s",
        simulation_id,
        scope["agents"]["profiles_count"],
        len(scope["agents"]["entity_types"]),
        total_actions,
        {
            name: (
                "未启用"
                if not item["enabled"]
                else f"{item['actions']} 条/传播={'有' if item['information_flow']['has_information_flow'] else '无'}"
            )
            for name, item in platforms.items()
        },
    )
    return scope


def _platform_line(platform: str, facts: Dict[str, Any]) -> str:
    """一个平台一句话，说清"跑了多少、有没有传播"。"""

    if not facts["enabled"]:
        return f"| {platform} | — | **未启用** | 该平台本次没有参与，任何结论都不适用于它 |"

    flow = facts["information_flow"]
    counts = flow["counts"]
    composition = "、".join(
        f"{name} {count} 条"
        for name, count in sorted(
            facts["action_types"].items(), key=lambda pair: (-pair[1], pair[0])
        )[:5]
    )
    if flow["has_information_flow"]:
        verdict = "**构成传播**"
        evidence = (
            f"推荐时间线 {counts['rec']} 行、互动边 {flow['social_edges']} 条"
        )
    else:
        verdict = "**不构成传播**"
        evidence = (
            f"推荐时间线 {counts['rec']} 行（agent 看不到别人的帖子）、"
            f"关注/点赞/评论共 {flow['social_edges']} 条"
        )
    return (
        f"| {platform} | {facts['actions']} 条 | {verdict} | "
        f"{evidence}；动作构成：{composition or '无'} |"
    )


def render_scope_markdown(scope: Dict[str, Any]) -> str:
    """把范围声明渲染成报告正文的确定性文本（不经过模型）。"""

    agents = scope["agents"]
    rounds = scope["rounds"]
    platforms = scope["platforms"]
    types = "、".join(agents["entity_types"]) or "无"

    lines = [
        "## 模拟范围与传播声明",
        "",
        "> 本节由报告生成器按模拟产物直接生成，不经过模型复述；数字可在 "
        f"`uploads/simulations/{scope['simulation_id']}/` 中逐项核对。",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 参与者 | {agents['profiles_count']} 个 agent"
        f"（图谱中带类型实体 {agents['entities_count']} 个） |",
        f"| 实体类型 | {len(agents['entity_types'])} 类：{types} |",
        f"| 轮次 | 配置 {rounds['configured']} 轮、执行 {rounds['executed']} 轮 |",
        f"| 动作总数 | {scope['total_actions']} 条 |",
        "",
        "| 平台 | 动作数 | 是否构成传播 | 依据 |",
        "|---|---|---|---|",
    ]
    for platform in PLATFORMS:
        if platform in platforms:
            lines.append(_platform_line(platform, platforms[platform]))

    no_flow = [
        name
        for name, facts in platforms.items()
        if facts["enabled"] and not facts["information_flow"]["has_information_flow"]
    ]
    disabled = [name for name, facts in platforms.items() if not facts["enabled"]]
    lines.append("")
    if no_flow:
        plural = len(no_flow) > 1
        lines.append(
            "**涉及传播的结论不适用于 "
            + "、".join(no_flow)
            + "**："
            + ("这些平台" if plural else "该平台")
            + "本次只有广播、没有传播路径（推荐时间线为空、没有形成互动边），用"
            + ("它们" if plural else "它")
            + "的动作数据下「舆情扩散/传播」类结论前必须另行取证。"
        )
    if disabled:
        lines.append(
            "**"
            + "、".join(disabled)
            + " 本次未启用**：该平台不在本次模拟范围内，任何结论都不适用于它。"
        )
    lines.append("")
    return "\n".join(lines)
