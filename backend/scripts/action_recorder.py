"""动作记账：把 OASIS 数据库里真实执行的动作落到 ``<platform>/actions.jsonl``。

这个文件是 **runner 账目、前端动作列表、图谱回写** 三者的唯一输入。
历史上这套逻辑内联在 ``run_parallel_simulation.py`` 里，两个单平台脚本没有接线：
走 API 的单平台运行于是"库里有动作、日志里 0 条、图谱 0 条回写、报告没有模拟内容"
（掩盖台账 N-21）。记账路径只允许有一条——三个 ``run_*_simulation.py`` 都用这里的
``ActionRecorder``。

记账口径：动作数**从数据库里数**，不从"激活了几个 Agent"推。Agent 被激活后可能
什么都不做（do_nothing），用请求数/激活数冒充动作数是审计里的 A5；我在 N-18 的
第一版修复里又犯过一次。``sign_up`` / ``refresh`` 是环境自己写的行，不是 Agent 的活动，
写入前过滤掉，但它们会计入 ``filtered_rows``，不允许被当成"这条路径跑过了"。
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from action_logger import PlatformActionLogger

logger = logging.getLogger(__name__)


class ActionRecordingError(RuntimeError):
    """动作记账失败：数据库读不到、结构不对、写入不可用。

    以前这里 ``except Exception`` 之后返回空列表——"没有动作"和"读不出动作"于是
    长得一模一样（掩盖台账 N-21 就是这样瞒过账目的）。
    """


# 需要过滤掉的非核心动作类型（环境自己写的行，不是 Agent 的活动）
FILTERED_ACTIONS = {'refresh', 'sign_up'}

# 动作类型映射表（数据库中的名称 -> 标准名称）
ACTION_TYPE_MAP = {
    'create_post': 'CREATE_POST',
    'like_post': 'LIKE_POST',
    'dislike_post': 'DISLIKE_POST',
    'repost': 'REPOST',
    'quote_post': 'QUOTE_POST',
    'follow': 'FOLLOW',
    'mute': 'MUTE',
    'create_comment': 'CREATE_COMMENT',
    'like_comment': 'LIKE_COMMENT',
    'dislike_comment': 'DISLIKE_COMMENT',
    'search_posts': 'SEARCH_POSTS',
    'search_user': 'SEARCH_USER',
    'trend': 'TREND',
    'do_nothing': 'DO_NOTHING',
    'interview': 'INTERVIEW',
}


def platform_information_flow(db_path: str) -> Dict[str, Any]:
    """读平台库，报告这一轮运行里"世界有没有信息流"。

    为什么要有（掩盖台账 N-35）：twitter 半场实测 35 agent × 11 轮只发帖、
    零点赞零转发零关注，平台库里 `rec`（推荐时间线）与 `follow` 都是空表——
    而这件事在 `actions.jsonl`、`run-status`、报告里都不出现。把它写进
    `simulation_end` 事件，产物自己就会说明"这个平台上的动作不构成传播"。
    """

    if not os.path.exists(db_path):
        return {
            "db_path": db_path,
            "exists": False,
            "note": "平台库不存在：无法判断本轮有没有信息流",
        }

    tables = ("post", "comment", "like", "follow", "rec", "trace")
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        present = {
            row[0]
            for row in connection.execute(
                "select name from sqlite_master where type='table'"
            )
        }
        counts: Dict[str, Optional[int]] = {}
        for table in tables:
            counts[table] = (
                connection.execute(f"select count(*) from {table}").fetchone()[0]
                if table in present
                else None
            )
        trace_by_action: Dict[str, int] = {}
        if "trace" in present:
            trace_by_action = {
                str(action): count
                for action, count in connection.execute(
                    "select action, count(*) from trace group by action"
                )
            }
    finally:
        connection.close()

    # 判据是**推荐时间线**（`rec` 表）有没有内容：它决定 agent 能不能看到别人的帖子。
    # 只用 follow/like 判会误判——实测 twitter 补上 REFRESH 后出现了 2 条关注、1 条点赞，
    # 但 `rec` 仍然是 0（OASIS 的 twhin 推荐表没产出任何候选），世界依旧是广播。
    has_timeline = bool(counts.get("rec"))
    social_edges = sum(counts.get(table) or 0 for table in ("follow", "like", "comment"))
    result: Dict[str, Any] = {
        "db_path": os.path.basename(db_path),
        "exists": True,
        "counts": counts,
        "trace_by_action": trace_by_action,
        "has_timeline": has_timeline,
        "social_edges": social_edges,
        "has_information_flow": has_timeline,
    }
    if not has_timeline:
        result["note"] = (
            "推荐时间线(rec)为空：agent 看不到别人的帖子（本轮 refresh "
            f"{trace_by_action.get('refresh', 0)} 次拿到的都是空结果），"
            f"平台上的互动只有 {social_edges} 条（follow/like/comment）。"
            "引用该平台数据时不得当作舆情结论"
        )
    return result


def agent_names_from_config(config: Dict[str, Any]) -> Dict[int, str]:
    """
    从 simulation_config 中获取 agent_id -> entity_name 的映射

    这样可以在 actions.jsonl 中显示真实的实体名称，而不是 "Agent_0" 这样的代号

    Args:
        config: simulation_config.json 的内容

    Returns:
        agent_id -> entity_name 的映射字典
    """
    agent_names = {}
    agent_configs = config.get("agent_configs", [])

    for agent_config in agent_configs:
        agent_id = agent_config.get("agent_id")
        entity_name = agent_config.get("entity_name", f"Agent_{agent_id}")
        if agent_id is not None:
            agent_names[agent_id] = entity_name

    return agent_names


def fetch_new_actions_from_db(
    db_path: str,
    last_rowid: int,
    agent_names: Dict[int, str]
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    从数据库中获取新的动作记录，并补充完整的上下文信息

    Args:
        db_path: 数据库文件路径
        last_rowid: 上次读取的最大 rowid 值（使用 rowid 而不是 created_at，因为不同平台的 created_at 格式不同）
        agent_names: agent_id -> agent_name 映射

    Returns:
        (actions_list, new_last_rowid, filtered_rows)
        - actions_list: 动作列表，每个元素包含 agent_id, agent_name, action_type, action_args（含上下文信息）
        - new_last_rowid: 新的最大 rowid 值
        - filtered_rows: 被 FILTERED_ACTIONS 过滤掉的 trace 行数（含被丢弃的行，用于对账）

    Raises:
        ActionRecordingError: 数据库不存在或读取失败（不允许用空列表冒充"没有动作"）
    """
    actions: List[Dict[str, Any]] = []
    new_last_rowid = last_rowid
    filtered_rows = 0

    if not os.path.exists(db_path):
        raise ActionRecordingError(
            f"动作数据库不存在: {db_path}——记账没有数据来源，"
            f"运行不允许继续（空日志会伪装成'这轮没有动作'）"
        )

    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as error:
        raise ActionRecordingError(f"无法打开动作数据库 {db_path}: {error}") from error

    try:
        cursor = conn.cursor()

        # 使用 rowid 来追踪已处理的记录（rowid 是 SQLite 的内置自增字段）
        # 这样可以避免 created_at 格式差异问题（Twitter 用整数，Reddit 用日期时间字符串）
        cursor.execute("""
            SELECT rowid, user_id, action, info
            FROM trace
            WHERE rowid > ?
            ORDER BY rowid ASC
        """, (last_rowid,))

        for rowid, user_id, action, info_json in cursor.fetchall():
            # 更新最大 rowid
            new_last_rowid = rowid

            # 过滤非核心动作
            if action in FILTERED_ACTIONS:
                filtered_rows += 1
                continue

            # 解析动作参数：解析不出来说明上游写入已损坏，不能拿空字典顶上
            try:
                action_args = json.loads(info_json) if info_json else {}
            except json.JSONDecodeError as error:
                raise ActionRecordingError(
                    f"trace.rowid={rowid} 的 info 不是合法 JSON（action={action}）: {error}"
                ) from error

            # 精简 action_args，只保留关键字段（保留完整内容，不截断）
            simplified_args = {}
            if 'content' in action_args:
                simplified_args['content'] = action_args['content']
            if 'post_id' in action_args:
                simplified_args['post_id'] = action_args['post_id']
            if 'comment_id' in action_args:
                simplified_args['comment_id'] = action_args['comment_id']
            if 'quoted_id' in action_args:
                simplified_args['quoted_id'] = action_args['quoted_id']
            if 'new_post_id' in action_args:
                simplified_args['new_post_id'] = action_args['new_post_id']
            if 'follow_id' in action_args:
                simplified_args['follow_id'] = action_args['follow_id']
            if 'query' in action_args:
                simplified_args['query'] = action_args['query']
            if 'like_id' in action_args:
                simplified_args['like_id'] = action_args['like_id']
            if 'dislike_id' in action_args:
                simplified_args['dislike_id'] = action_args['dislike_id']

            # 转换动作类型名称
            action_type = ACTION_TYPE_MAP.get(action, action.upper())

            # 补充上下文信息（帖子内容、用户名等）
            _enrich_action_context(cursor, action_type, simplified_args, agent_names)

            actions.append({
                'agent_id': user_id,
                'agent_name': agent_names.get(user_id, f'Agent_{user_id}'),
                'action_type': action_type,
                'action_args': simplified_args,
            })

        return actions, new_last_rowid, filtered_rows
    except sqlite3.Error as error:
        raise ActionRecordingError(f"读取动作数据库失败（{db_path}）: {error}") from error
    finally:
        conn.close()


def _enrich_action_context(
    cursor,
    action_type: str,
    action_args: Dict[str, Any],
    agent_names: Dict[int, str]
) -> None:
    """
    为动作补充上下文信息（帖子内容、用户名等）

    Args:
        cursor: 数据库游标
        action_type: 动作类型
        action_args: 动作参数（会被修改）
        agent_names: agent_id -> agent_name 映射
    """
    try:
        # 点赞/踩帖子：补充帖子内容和作者
        if action_type in ('LIKE_POST', 'DISLIKE_POST'):
            post_id = action_args.get('post_id')
            if post_id:
                post_info = _get_post_info(cursor, post_id, agent_names)
                if post_info:
                    action_args['post_content'] = post_info.get('content', '')
                    action_args['post_author_name'] = post_info.get('author_name', '')

        # 转发帖子：补充原帖内容和作者
        elif action_type == 'REPOST':
            new_post_id = action_args.get('new_post_id')
            if new_post_id:
                # 转发帖子的 original_post_id 指向原帖
                cursor.execute("""
                    SELECT original_post_id FROM post WHERE post_id = ?
                """, (new_post_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    original_post_id = row[0]
                    original_info = _get_post_info(cursor, original_post_id, agent_names)
                    if original_info:
                        action_args['original_content'] = original_info.get('content', '')
                        action_args['original_author_name'] = original_info.get('author_name', '')

        # 引用帖子：补充原帖内容、作者和引用评论
        elif action_type == 'QUOTE_POST':
            quoted_id = action_args.get('quoted_id')
            new_post_id = action_args.get('new_post_id')

            if quoted_id:
                original_info = _get_post_info(cursor, quoted_id, agent_names)
                if original_info:
                    action_args['original_content'] = original_info.get('content', '')
                    action_args['original_author_name'] = original_info.get('author_name', '')

            # 获取引用帖子的评论内容（quote_content）
            if new_post_id:
                cursor.execute("""
                    SELECT quote_content FROM post WHERE post_id = ?
                """, (new_post_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    action_args['quote_content'] = row[0]

        # 关注用户：补充被关注用户的名称
        elif action_type == 'FOLLOW':
            follow_id = action_args.get('follow_id')
            if follow_id:
                # 从 follow 表获取 followee_id
                cursor.execute("""
                    SELECT followee_id FROM follow WHERE follow_id = ?
                """, (follow_id,))
                row = cursor.fetchone()
                if row:
                    followee_id = row[0]
                    target_name = _get_user_name(cursor, followee_id, agent_names)
                    if target_name:
                        action_args['target_user_name'] = target_name

        # 屏蔽用户：补充被屏蔽用户的名称
        elif action_type == 'MUTE':
            # 从 action_args 中获取 user_id 或 target_id
            target_id = action_args.get('user_id') or action_args.get('target_id')
            if target_id:
                target_name = _get_user_name(cursor, target_id, agent_names)
                if target_name:
                    action_args['target_user_name'] = target_name

        # 点赞/踩评论：补充评论内容和作者
        elif action_type in ('LIKE_COMMENT', 'DISLIKE_COMMENT'):
            comment_id = action_args.get('comment_id')
            if comment_id:
                comment_info = _get_comment_info(cursor, comment_id, agent_names)
                if comment_info:
                    action_args['comment_content'] = comment_info.get('content', '')
                    action_args['comment_author_name'] = comment_info.get('author_name', '')

        # 发表评论：补充所评论的帖子信息
        elif action_type == 'CREATE_COMMENT':
            post_id = action_args.get('post_id')
            if post_id:
                post_info = _get_post_info(cursor, post_id, agent_names)
                if post_info:
                    action_args['post_content'] = post_info.get('content', '')
                    action_args['post_author_name'] = post_info.get('author_name', '')

    except sqlite3.Error as error:
        # 上下文补不上不影响"这条动作确实发生过"：动作本身照样记账，
        # 缺的是上下文（不是伪造的上下文），所以这里只报告不中断。
        print(f"补充动作上下文失败（动作仍然记账，缺上下文）: {action_type}: {error}")


def _get_post_info(
    cursor,
    post_id: int,
    agent_names: Dict[int, str]
) -> Optional[Dict[str, str]]:
    """
    获取帖子信息

    Args:
        cursor: 数据库游标
        post_id: 帖子ID
        agent_names: agent_id -> agent_name 映射

    Returns:
        包含 content 和 author_name 的字典，或 None
    """
    try:
        cursor.execute("""
            SELECT p.content, p.user_id, u.agent_id
            FROM post p
            LEFT JOIN user u ON p.user_id = u.user_id
            WHERE p.post_id = ?
        """, (post_id,))
        row = cursor.fetchone()
        if row:
            content = row[0] or ''
            user_id = row[1]
            agent_id = row[2]

            # 优先使用 agent_names 中的名称
            author_name = ''
            if agent_id is not None and agent_id in agent_names:
                author_name = agent_names[agent_id]
            elif user_id:
                # 从 user 表获取名称
                cursor.execute("SELECT name, user_name FROM user WHERE user_id = ?", (user_id,))
                user_row = cursor.fetchone()
                if user_row:
                    author_name = user_row[0] or user_row[1] or ''

            return {'content': content, 'author_name': author_name}
    except sqlite3.Error:
        # 查不到就是查不到：返回 None，由调用方决定少写哪个上下文字段（不补默认值）
        pass
    return None


def _get_user_name(
    cursor,
    user_id: int,
    agent_names: Dict[int, str]
) -> Optional[str]:
    """
    获取用户名称

    Args:
        cursor: 数据库游标
        user_id: 用户ID
        agent_names: agent_id -> agent_name 映射

    Returns:
        用户名称，或 None
    """
    try:
        cursor.execute("""
            SELECT agent_id, name, user_name FROM user WHERE user_id = ?
        """, (user_id,))
        row = cursor.fetchone()
        if row:
            agent_id = row[0]
            name = row[1]
            user_name = row[2]

            # 优先使用 agent_names 中的名称
            if agent_id is not None and agent_id in agent_names:
                return agent_names[agent_id]
            return name or user_name or ''
    except sqlite3.Error:
        pass
    return None


def _get_comment_info(
    cursor,
    comment_id: int,
    agent_names: Dict[int, str]
) -> Optional[Dict[str, str]]:
    """
    获取评论信息

    Args:
        cursor: 数据库游标
        comment_id: 评论ID
        agent_names: agent_id -> agent_name 映射

    Returns:
        包含 content 和 author_name 的字典，或 None
    """
    try:
        cursor.execute("""
            SELECT c.content, c.user_id, u.agent_id
            FROM comment c
            LEFT JOIN user u ON c.user_id = u.user_id
            WHERE c.comment_id = ?
        """, (comment_id,))
        row = cursor.fetchone()
        if row:
            content = row[0] or ''
            user_id = row[1]
            agent_id = row[2]

            # 优先使用 agent_names 中的名称
            author_name = ''
            if agent_id is not None and agent_id in agent_names:
                author_name = agent_names[agent_id]
            elif user_id:
                # 从 user 表获取名称
                cursor.execute("SELECT name, user_name FROM user WHERE user_id = ?", (user_id,))
                user_row = cursor.fetchone()
                if user_row:
                    author_name = user_row[0] or user_row[1] or ''

            return {'content': content, 'author_name': author_name}
    except sqlite3.Error:
        pass
    return None


class TimelineNotBuilt(RuntimeError):
    """平台上有帖子，却没有推荐时间线：这一轮的世界里 agent 看不到别人的帖子。

    这是**失败**，不是可以继续跑的状态。实测（台账 N-35/N-46）：twitter 的
    `rec` 表一直是空的，35 个 agent × 多轮只产出单向发帖、0 赞 0 转发 0 关注，
    而"不构成传播"当时只写在 `simulation_end` 的附注里，运行的判定仍是成功。
    """

    def __init__(self, platform: str, round_num: int, db_path: str, counts: Dict[str, Any]):
        self.platform = platform
        self.round_num = round_num
        self.db_path = db_path
        self.counts = counts
        super().__init__(
            f"{platform} 第 {round_num} 轮之后仍然没有推荐时间线："
            f"post={counts.get('post')} 行、rec={counts.get('rec')} 行"
            f"（{db_path}）。推荐表为空 = agent 看不到别人的帖子，动作空间只剩发帖，"
            "这一轮的产出不构成传播。检查 twitter 的推荐模型"
            "（backend/scripts/twitter_recsys_model.py）与 OASIS 推荐表更新日志"
            "（social.rec / social.twitter 的 ERROR）。"
        )


class ActionRecorder:
    """单平台动作记账器：库里真实执行的动作 → ``<platform>/actions.jsonl``。

    一次 ``record_round()`` 写出三种事件：``round_start``、每个动作、``round_end``。
    轮次边界和动作条目由同一个调用写出，调用方不可能只写一半；动作数也由它返回，
    所以"记了账"和"数出了动作"永远来自同一条路径。
    """

    def __init__(self, platform: str, simulation_dir: str, config: Dict[str, Any]):
        if platform not in ("twitter", "reddit"):
            raise ValueError(
                f"未知平台: {platform!r}——actions.jsonl 的目录名由平台决定"
            )
        self.platform = platform
        self.simulation_dir = simulation_dir
        self.config = config
        self.db_path = os.path.join(simulation_dir, f"{platform}_simulation.db")
        self.action_logger = PlatformActionLogger(platform, simulation_dir)
        self._agent_names = agent_names_from_config(config)
        self._last_rowid = 0
        self.total_actions = 0     # 已写入 actions.jsonl 的动作数
        self.seed_actions = 0      # 其中属于 round 0 的种子帖
        self.filtered_rows = 0     # 被过滤掉的 trace 行（sign_up / refresh）

    @property
    def agent_actions(self) -> int:
        """Agent 真的做出来的动作数（不含 round 0 的种子帖）。

        种子帖是配置指定的输入，不是模拟跑出来的活动：一场只有种子帖的运行
        仍然是"没跑出东西"，不能用它判成功（AGENTS.md §1.1 空产物即失败）。
        """
        return self.total_actions - self.seed_actions

    def bind_agent_graph(self, agent_graph) -> None:
        """用 OASIS Agent 的默认名补全 config 里没写的 agent_id。"""
        for agent_id, agent in agent_graph.get_agents():
            self._agent_names.setdefault(
                agent_id, getattr(agent, 'name', f'Agent_{agent_id}')
            )

    def log_simulation_start(self) -> None:
        self.action_logger.log_simulation_start(self.config)

    def record_round(self, round_num: int, simulated_hour: int) -> int:
        """记一轮：round_start → 库里新增的真实动作 → round_end，返回本轮动作数。"""
        self.action_logger.log_round_start(round_num, simulated_hour)

        actions, self._last_rowid, filtered_rows = fetch_new_actions_from_db(
            self.db_path, self._last_rowid, self._agent_names
        )
        self.filtered_rows += filtered_rows

        for action_data in actions:
            self.action_logger.log_action(
                round_num=round_num,
                agent_id=action_data['agent_id'],
                agent_name=action_data['agent_name'],
                action_type=action_data['action_type'],
                action_args=action_data['action_args'],
            )
            self.total_actions += 1
            if round_num == 0:
                self.seed_actions += 1

        self.action_logger.log_round_end(round_num, len(actions), simulated_hour)
        return len(actions)

    def require_timeline(self, round_num: int) -> None:
        """刚跑过一轮：平台上有帖子就必须有推荐时间线，否则抛 `TimelineNotBuilt`。

        调用点在三个 `run_*` 脚本的循环里，条件是"这一轮真的 step 过"（有活跃 agent）：
        只有 step 才会触发 OASIS 的 `update_rec_table()`，也才谈得上"时间线该有了"。

        判据直接读平台库（与 `simulation_end` 的 `information_flow` 同源）：
        `post` 有行而 `rec` 为空。库不存在同样是失败——没法判断就没有通过可言。
        """

        flow = platform_information_flow(self.db_path)
        if not flow.get("exists"):
            raise TimelineNotBuilt(
                self.platform, round_num, self.db_path, {"post": None, "rec": None}
            )
        counts = flow["counts"]
        if (counts.get("post") or 0) > 0 and not counts.get("rec"):
            raise TimelineNotBuilt(self.platform, round_num, self.db_path, counts)

    def log_simulation_end(self, total_rounds: int) -> None:
        flow = platform_information_flow(self.db_path)
        if flow.get("exists") and not flow.get("has_information_flow"):
            logger.error(
                "%s 本轮没有任何信息流（推荐时间线/关注关系为空）：%s",
                self.platform, flow.get("note"),
            )
        self.action_logger.log_simulation_end(
            total_rounds, self.total_actions, information_flow=flow
        )
