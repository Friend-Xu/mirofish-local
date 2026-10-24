"""动作记账契约测试：`actions.jsonl` 是 runner 账目 / 前端动作列表 / 图谱回写的唯一输入。

对应规约：`MiroFish/AGENTS.md` §1（一个能力一条路径）、§1.1（空产物即失败）；
掩盖台账：N-21（单平台不写动作日志）、N-22（事件字段与读取方不一致）、N-23（种子帖静默丢失）。

证据等级说明：本文件覆盖的是**记账器本身**与**接线是否还在**。
"单平台运行真的会产出 actions.jsonl"由 L-a 实跑（真脚本 + 真 OASIS + 真回写）证明，
文件里的静态断言不能替代它。
"""

import json
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from action_recorder import (  # noqa: E402 - 依赖上面的 sys.path 设置
    ActionRecorder,
    ActionRecordingError,
    agent_names_from_config,
)

AGENT_CONFIG = {
    "agent_configs": [
        {"agent_id": 0, "entity_name": "同期同事"},
        {"agent_id": 1, "entity_name": "新入职同事"},
    ],
    "time_config": {"total_simulation_hours": 12, "minutes_per_round": 60},
}


def _make_simulation_dir(tmp_path: Path, rows):
    """造一个仿真目录：`<platform>_simulation.db` 表结构与 OASIS 一致（最小集）。"""

    sim_dir = tmp_path / "sim_x"
    sim_dir.mkdir()
    db_path = sim_dir / "twitter_simulation.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE trace (user_id INTEGER, action TEXT, info TEXT, created_at TEXT);
        CREATE TABLE post (post_id INTEGER, user_id INTEGER, content TEXT,
                           original_post_id INTEGER, quote_content TEXT);
        CREATE TABLE user (user_id INTEGER, agent_id INTEGER, name TEXT, user_name TEXT);
        CREATE TABLE comment (comment_id INTEGER, user_id INTEGER, content TEXT);
        CREATE TABLE follow (follow_id INTEGER, followee_id INTEGER);
        """
    )
    conn.executemany("INSERT INTO user VALUES (?,?,?,?)", [(0, 0, "甲", None), (1, 1, "乙", None)])
    conn.execute(
        "INSERT INTO post VALUES (1, 0, '种子帖内容', NULL, NULL)"
    )
    conn.executemany(
        "INSERT INTO trace (user_id, action, info, created_at) VALUES (?,?,?,?)", rows
    )
    conn.commit()
    conn.close()
    return sim_dir


def _read_lines(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------- 记账路径


def test_recorder_writes_actions_and_round_events(tmp_path):
    """一轮记账写出 round_start / 动作 / round_end，且动作数由库里的行决定。"""

    sim_dir = _make_simulation_dir(
        tmp_path,
        [
            (0, "sign_up", '{"name": "甲"}', "0"),
            (1, "sign_up", '{"name": "乙"}', "0"),
            (0, "create_post", '{"content": "第一条"}', "0"),
            (1, "like_post", '{"post_id": 1}', "0"),
        ],
    )
    recorder = ActionRecorder("twitter", str(sim_dir), AGENT_CONFIG)
    recorder.log_simulation_start()
    produced = recorder.record_round(0, 12)

    assert produced == 2, "sign_up 是环境写的行，不是 Agent 的活动"
    assert recorder.filtered_rows == 2
    assert recorder.total_actions == 2

    entries = _read_lines(sim_dir / "twitter" / "actions.jsonl")
    events = [e.get("event_type") for e in entries]
    assert events == ["simulation_start", "round_start", None, None, "round_end"]

    actions = [e for e in entries if e.get("event_type") is None]
    assert [a["action_type"] for a in actions] == ["CREATE_POST", "LIKE_POST"]
    assert [a["agent_name"] for a in actions] == ["同期同事", "新入职同事"]
    # 动作上下文补充：点赞要带上被赞帖子的内容与作者
    assert actions[1]["action_args"]["post_content"] == "种子帖内容"
    assert actions[1]["action_args"]["post_author_name"] == "同期同事"


def test_each_action_satisfies_the_writeback_contract(tmp_path):
    """动作行必须满足图谱回写要求的字段（缺一个就会被回写拒绝）。"""

    from app.services.zep_graph_memory_updater import ZepGraphMemoryUpdater

    sim_dir = _make_simulation_dir(
        tmp_path, [(0, "create_post", '{"content": "第一条"}', "0")]
    )
    recorder = ActionRecorder("twitter", str(sim_dir), AGENT_CONFIG)
    recorder.record_round(1, 13)

    actions = [
        e
        for e in _read_lines(sim_dir / "twitter" / "actions.jsonl")
        if e.get("event_type") is None
    ]
    assert actions, "这一轮没有写出任何动作"
    for action in actions:
        for field in ZepGraphMemoryUpdater.REQUIRED_ACTION_FIELDS:
            assert action.get(field) is not None, f"{field} 缺失: {action}"


def test_second_round_only_sees_new_rows(tmp_path):
    """rowid 游标必须推进：否则同一批动作会被记两遍（回写也会写两遍）。"""

    sim_dir = _make_simulation_dir(
        tmp_path, [(0, "create_post", '{"content": "第一条"}', "0")]
    )
    recorder = ActionRecorder("twitter", str(sim_dir), AGENT_CONFIG)
    assert recorder.record_round(1, 12) == 1

    conn = sqlite3.connect(sim_dir / "twitter_simulation.db")
    conn.execute(
        "INSERT INTO trace (user_id, action, info, created_at) VALUES (0, 'create_post', ?, '0')",
        ('{"content": "第二条"}',),
    )
    conn.commit()
    conn.close()

    assert recorder.record_round(2, 13) == 1
    assert recorder.total_actions == 2


def test_seed_actions_do_not_count_as_agent_actions(tmp_path):
    """round 0 的种子帖是配置给的输入：只有它落地说明模拟没跑出东西。"""

    sim_dir = _make_simulation_dir(
        tmp_path, [(0, "create_post", '{"content": "种子"}', "0")]
    )
    recorder = ActionRecorder("twitter", str(sim_dir), AGENT_CONFIG)
    assert recorder.record_round(0, 12) == 1

    assert recorder.total_actions == 1
    assert recorder.seed_actions == 1
    assert recorder.agent_actions == 0


def test_environment_only_rows_are_not_activity(tmp_path):
    """整场只有 sign_up / refresh → 动作数 0（不允许被当成"跑过了"）。"""

    sim_dir = _make_simulation_dir(
        tmp_path,
        [(0, "sign_up", "{}", "0"), (1, "refresh", "{}", "0")],
    )
    recorder = ActionRecorder("twitter", str(sim_dir), AGENT_CONFIG)
    assert recorder.record_round(1, 12) == 0
    assert recorder.total_actions == 0
    assert recorder.filtered_rows == 2


# --------------------------------------------------------------- 失败必须可见


def test_missing_database_raises(tmp_path):
    """库不存在时必须抛错：返回空列表就等于"这轮没有动作"，两者长得一样。"""

    sim_dir = tmp_path / "sim_missing"
    sim_dir.mkdir()
    recorder = ActionRecorder("twitter", str(sim_dir), AGENT_CONFIG)
    with pytest.raises(ActionRecordingError) as error:
        recorder.record_round(1, 12)
    assert "数据库不存在" in str(error.value)


def test_corrupt_action_info_raises(tmp_path):
    """info 不是合法 JSON：不能拿空字典顶上（那会记成一条"无参数的动作"）。"""

    sim_dir = _make_simulation_dir(tmp_path, [(0, "create_post", "{不是 JSON", "0")])
    recorder = ActionRecorder("twitter", str(sim_dir), AGENT_CONFIG)
    with pytest.raises(ActionRecordingError) as error:
        recorder.record_round(1, 12)
    assert "不是合法 JSON" in str(error.value)


def test_unknown_platform_rejected(tmp_path):
    with pytest.raises(ValueError):
        ActionRecorder("weibo", str(tmp_path), AGENT_CONFIG)


# --------------------------------------------------------------- 事件字段（N-22）


def test_round_end_carries_simulated_hours(tmp_path):
    """runner 从 round_end 读 `simulated_hours`：只写 `actions_count` 会让模拟时间恒为 0。"""

    sim_dir = _make_simulation_dir(tmp_path, [(0, "create_post", '{"content": "x"}', "0")])
    recorder = ActionRecorder("twitter", str(sim_dir), AGENT_CONFIG)
    recorder.record_round(3, 21)

    round_end = [
        e
        for e in _read_lines(sim_dir / "twitter" / "actions.jsonl")
        if e.get("event_type") == "round_end"
    ]
    assert round_end and round_end[0]["simulated_hours"] == 21
    assert round_end[0]["actions_count"] == 1


def test_simulation_start_rounds_follow_minutes_per_round(tmp_path):
    """总轮数由时长与每轮分钟数算出（以前硬编码 ×2，只对 30 分钟/轮成立）。"""

    sim_dir = _make_simulation_dir(tmp_path, [])
    config = dict(AGENT_CONFIG, time_config={"total_simulation_hours": 12, "minutes_per_round": 60})
    recorder = ActionRecorder("twitter", str(sim_dir), config)
    recorder.log_simulation_start()

    start = _read_lines(sim_dir / "twitter" / "actions.jsonl")[0]
    assert start["total_rounds"] == 12


def test_simulation_start_rejects_unknown_round_length(tmp_path):
    sim_dir = _make_simulation_dir(tmp_path, [])
    config = dict(AGENT_CONFIG, time_config={"total_simulation_hours": 12, "minutes_per_round": 0})
    recorder = ActionRecorder("twitter", str(sim_dir), config)
    with pytest.raises(ValueError):
        recorder.log_simulation_start()


# --------------------------------------------------------------- 接线（静态）


def test_all_three_scripts_use_the_single_recorder():
    """三个 run_*_simulation.py 都必须用同一个 ActionRecorder，且不得再内联一份实现。

    这是**静态检查**：它锁的是"记账路径只有一条"这个结构，不证明记账真的发生
    （那由 L-a 实跑证明）。因此这里同时禁止脚本里重新出现 `fetch_new_actions_from_db` 的定义。
    """

    scripts = [
        SCRIPTS_DIR / "run_twitter_simulation.py",
        SCRIPTS_DIR / "run_reddit_simulation.py",
        SCRIPTS_DIR / "run_parallel_simulation.py",
    ]
    for script in scripts:
        source = script.read_text(encoding="utf-8")
        assert "from action_recorder import ActionRecorder" in source, script.name
        assert "ActionRecorder(" in source, f"{script.name} 没有构造记账器"
        assert "def fetch_new_actions_from_db" not in source, f"{script.name} 又内联了一份实现"
        assert "def get_agent_names_from_config" not in source, f"{script.name} 又内联了一份实现"


def test_agent_names_from_config_maps_entity_names():
    assert agent_names_from_config(AGENT_CONFIG) == {0: "同期同事", 1: "新入职同事"}


# --------------------------------------------------- 平台信息流（N-35）


def _flow_db(path: Path, *, rec_rows: int, follow_rows: int):
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE post (post_id INTEGER, user_id INTEGER);
        CREATE TABLE comment (comment_id INTEGER, user_id INTEGER);
        CREATE TABLE like (like_id INTEGER, user_id INTEGER);
        CREATE TABLE follow (follow_id INTEGER, follower_id INTEGER, followee_id INTEGER);
        CREATE TABLE rec (user_id INTEGER, post_id INTEGER);
        CREATE TABLE trace (user_id INTEGER, action TEXT);
        """
    )
    conn.execute("INSERT INTO post VALUES (1, 0)")
    conn.execute("INSERT INTO trace VALUES (0, 'create_post')")
    conn.executemany("INSERT INTO rec VALUES (?, ?)", [(0, 1)] * rec_rows)
    conn.executemany("INSERT INTO follow VALUES (?,?,?)", [(i, 0, 1) for i in range(follow_rows)])
    conn.commit()
    conn.close()


def test_information_flow_reports_an_empty_world(tmp_path):
    """实测现场：twitter 35 agent × 11 轮，rec 与 follow 都是 0 行。"""

    from action_recorder import platform_information_flow

    db_path = tmp_path / "twitter_simulation.db"
    _flow_db(db_path, rec_rows=0, follow_rows=0)

    flow = platform_information_flow(str(db_path))
    assert flow["exists"] is True
    assert flow["has_timeline"] is False
    assert flow["has_information_flow"] is False
    assert flow["counts"]["rec"] == 0 and flow["counts"]["follow"] == 0
    assert "推荐时间线(rec)为空" in flow["note"]
    assert "不得当作舆情结论" in flow["note"]
    assert flow["trace_by_action"] == {"create_post": 1}


def test_information_flow_ignores_a_few_edges_without_a_timeline(tmp_path):
    """实测现场（补 REFRESH 之后）：有 2 条关注、1 条点赞，但 `rec` 仍是 0——
    agent 依然看不到别人的帖子，不能因为"有几条边"就说这个平台有信息流。"""

    from action_recorder import platform_information_flow

    db_path = tmp_path / "twitter_simulation.db"
    _flow_db(db_path, rec_rows=0, follow_rows=2)

    flow = platform_information_flow(str(db_path))
    assert flow["has_timeline"] is False
    assert flow["has_information_flow"] is False
    assert flow["social_edges"] >= 2
    assert "只有" in flow["note"]


def test_information_flow_is_true_when_the_world_has_edges(tmp_path):
    from action_recorder import platform_information_flow

    db_path = tmp_path / "reddit_simulation.db"
    _flow_db(db_path, rec_rows=5, follow_rows=3)

    flow = platform_information_flow(str(db_path))
    assert flow["has_information_flow"] is True
    assert "note" not in flow


def test_information_flow_flags_a_missing_database(tmp_path):
    from action_recorder import platform_information_flow

    flow = platform_information_flow(str(tmp_path / "nope.db"))
    assert flow["exists"] is False
    assert "不存在" in flow["note"]


def test_simulation_end_carries_the_information_flow(tmp_path):
    """产物自己必须说明"这个平台上有没有传播"（否则零互动不会被任何一层报出来）。"""

    sim_dir = _make_simulation_dir(tmp_path, [])
    _flow_db(sim_dir / "twitter_simulation.db", rec_rows=0, follow_rows=0)
    recorder = ActionRecorder("twitter", str(sim_dir), AGENT_CONFIG)
    recorder.log_simulation_end(total_rounds=3)

    end = [
        line for line in _read_lines(sim_dir / "twitter" / "actions.jsonl")
        if line["event_type"] == "simulation_end"
    ]
    assert end and end[0]["information_flow"]["has_information_flow"] is False


def test_twitter_action_menu_can_refresh_the_timeline():
    """twitter 的动作菜单必须有 REFRESH（N-35）：OASIS 用它把 rec 表拉进 observation。

    这是**静态断言**："补上 REFRESH 之后 twitter 真的会有互动"必须由复跑证明，
    此处只锁"菜单里不能再缺这个动作"。
    """

    parallel = (SCRIPTS_DIR / "run_parallel_simulation.py").read_text(encoding="utf-8")
    twitter_only = (SCRIPTS_DIR / "run_twitter_simulation.py").read_text(encoding="utf-8")

    def _list_block(source: str, name: str) -> str:
        start = source.index(f"{name} = [")
        return source[start:source.index("]", start)]

    assert "ActionType.REFRESH" in _list_block(parallel, "TWITTER_ACTIONS")
    assert "ActionType.REFRESH" in _list_block(twitter_only, "AVAILABLE_ACTIONS")
    # reddit 一直是有的；两个平台在"能不能看到别人的帖子"这件事上必须一致
    assert "ActionType.REFRESH" in _list_block(parallel, "REDDIT_ACTIONS")
