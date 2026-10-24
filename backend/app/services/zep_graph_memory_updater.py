"""
Zep图谱记忆更新服务
将模拟中的Agent活动动态更新到Zep图谱中
"""

import json
import os
import time
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from queue import Queue, Empty

from ..config import Config
from ..utils.logger import get_logger
from ..utils.locale import get_locale, set_locale
from ..utils.zep import ZEP_INGESTION_WAIT_TIMEOUT_SECONDS
from .zep_factory import get_graph_store

logger = get_logger('mirofish.zep_graph_memory_updater')

# 未能写入图谱的活动落盘目录（绝不随进程消失）
PENDING_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'uploads', 'graphs', 'pending'
)
PENDING_DIR = os.path.abspath(PENDING_DIR)


def pending_backlog(graph_id: str) -> Dict[str, Any]:
    """读取某个图谱"落盘待处理"的积压（只读，供 API/报告声明完整性）。

    返回条数、最后一次原因、最早/最晚时间。文件不存在表示没有积压。
    """

    path = os.path.join(PENDING_DIR, f"{graph_id}.jsonl")
    if not os.path.exists(path):
        return {"graph_id": graph_id, "pending_items": 0, "path": None, "reasons": {}}

    counts: Dict[str, int] = {}
    first_at = last_at = None
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counts["(无法解析的行)"] = counts.get("(无法解析的行)", 0) + 1
                continue
            reason = str(row.get("unsent_reason") or "未记录原因")
            counts[reason] = counts.get(reason, 0) + 1
            stamp = row.get("timestamp")
            if stamp:
                first_at = first_at or stamp
                last_at = stamp
    return {
        "graph_id": graph_id,
        "pending_items": sum(counts.values()),
        "path": path,
        "reasons": counts,
        "first_at": first_at,
        "last_at": last_at,
    }


@dataclass
class AgentActivity:
    """Agent活动记录"""
    platform: str           # twitter / reddit
    agent_id: int
    agent_name: str
    action_type: str        # CREATE_POST, LIKE_POST, etc.
    action_args: Dict[str, Any]
    round_num: int
    timestamp: str
    
    def to_episode_text(self) -> str:
        """
        将活动转换为可以发送给Zep的文本描述
        
        采用自然语言描述格式，让Zep能够从中提取实体和关系
        不添加模拟相关的前缀，避免误导图谱更新
        """
        # 根据不同的动作类型生成不同的描述
        action_descriptions = {
            "CREATE_POST": self._describe_create_post,
            "LIKE_POST": self._describe_like_post,
            "DISLIKE_POST": self._describe_dislike_post,
            "REPOST": self._describe_repost,
            "QUOTE_POST": self._describe_quote_post,
            "FOLLOW": self._describe_follow,
            "CREATE_COMMENT": self._describe_create_comment,
            "LIKE_COMMENT": self._describe_like_comment,
            "DISLIKE_COMMENT": self._describe_dislike_comment,
            "SEARCH_POSTS": self._describe_search,
            "SEARCH_USER": self._describe_search_user,
            "MUTE": self._describe_mute,
        }
        
        describe_func = action_descriptions.get(self.action_type, self._describe_generic)
        description = describe_func()
        
        # Keep the event time in the source text as well as episode metadata so
        # temporal extraction does not collapse a multi-action batch.
        return (
            f"[{self.timestamp}] [{self.platform} round {self.round_num}] "
            f"{self.agent_name}: {description}"
        )
    
    def _describe_create_post(self) -> str:
        content = self.action_args.get("content", "")
        if content:
            return f"发布了一条帖子：「{content}」"
        return "发布了一条帖子"
    
    def _describe_like_post(self) -> str:
        """点赞帖子 - 包含帖子原文和作者信息"""
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        
        if post_content and post_author:
            return f"点赞了{post_author}的帖子：「{post_content}」"
        elif post_content:
            return f"点赞了一条帖子：「{post_content}」"
        elif post_author:
            return f"点赞了{post_author}的一条帖子"
        return "点赞了一条帖子"
    
    def _describe_dislike_post(self) -> str:
        """踩帖子 - 包含帖子原文和作者信息"""
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        
        if post_content and post_author:
            return f"踩了{post_author}的帖子：「{post_content}」"
        elif post_content:
            return f"踩了一条帖子：「{post_content}」"
        elif post_author:
            return f"踩了{post_author}的一条帖子"
        return "踩了一条帖子"
    
    def _describe_repost(self) -> str:
        """转发帖子 - 包含原帖内容和作者信息"""
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")
        
        if original_content and original_author:
            return f"转发了{original_author}的帖子：「{original_content}」"
        elif original_content:
            return f"转发了一条帖子：「{original_content}」"
        elif original_author:
            return f"转发了{original_author}的一条帖子"
        return "转发了一条帖子"
    
    def _describe_quote_post(self) -> str:
        """引用帖子 - 包含原帖内容、作者信息和引用评论"""
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")
        quote_content = self.action_args.get("quote_content", "") or self.action_args.get("content", "")
        
        base = ""
        if original_content and original_author:
            base = f"引用了{original_author}的帖子「{original_content}」"
        elif original_content:
            base = f"引用了一条帖子「{original_content}」"
        elif original_author:
            base = f"引用了{original_author}的一条帖子"
        else:
            base = "引用了一条帖子"
        
        if quote_content:
            base += f"，并评论道：「{quote_content}」"
        return base
    
    def _describe_follow(self) -> str:
        """关注用户 - 包含被关注用户的名称"""
        target_user_name = self.action_args.get("target_user_name", "")
        
        if target_user_name:
            return f"关注了用户「{target_user_name}」"
        return "关注了一个用户"
    
    def _describe_create_comment(self) -> str:
        """发表评论 - 包含评论内容和所评论的帖子信息"""
        content = self.action_args.get("content", "")
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        
        if content:
            if post_content and post_author:
                return f"在{post_author}的帖子「{post_content}」下评论道：「{content}」"
            elif post_content:
                return f"在帖子「{post_content}」下评论道：「{content}」"
            elif post_author:
                return f"在{post_author}的帖子下评论道：「{content}」"
            return f"评论道：「{content}」"
        return "发表了评论"
    
    def _describe_like_comment(self) -> str:
        """点赞评论 - 包含评论内容和作者信息"""
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")
        
        if comment_content and comment_author:
            return f"点赞了{comment_author}的评论：「{comment_content}」"
        elif comment_content:
            return f"点赞了一条评论：「{comment_content}」"
        elif comment_author:
            return f"点赞了{comment_author}的一条评论"
        return "点赞了一条评论"
    
    def _describe_dislike_comment(self) -> str:
        """踩评论 - 包含评论内容和作者信息"""
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")
        
        if comment_content and comment_author:
            return f"踩了{comment_author}的评论：「{comment_content}」"
        elif comment_content:
            return f"踩了一条评论：「{comment_content}」"
        elif comment_author:
            return f"踩了{comment_author}的一条评论"
        return "踩了一条评论"
    
    def _describe_search(self) -> str:
        """搜索帖子 - 包含搜索关键词"""
        query = self.action_args.get("query", "") or self.action_args.get("keyword", "")
        return f"搜索了「{query}」" if query else "进行了搜索"
    
    def _describe_search_user(self) -> str:
        """搜索用户 - 包含搜索关键词"""
        query = self.action_args.get("query", "") or self.action_args.get("username", "")
        return f"搜索了用户「{query}」" if query else "搜索了用户"
    
    def _describe_mute(self) -> str:
        """屏蔽用户 - 包含被屏蔽用户的名称"""
        target_user_name = self.action_args.get("target_user_name", "")
        
        if target_user_name:
            return f"屏蔽了用户「{target_user_name}」"
        return "屏蔽了一个用户"
    
    def _describe_generic(self) -> str:
        # 对于未知的动作类型，生成通用描述
        return f"执行了{self.action_type}操作"


class _DrainDeadlineExceeded(TimeoutError):
    def __init__(self, processed_count: int):
        super().__init__("Zep updater drain deadline elapsed")
        self.processed_count = processed_count


class ZepGraphMemoryUpdater:
    """
    Zep图谱记忆更新器
    
    监控模拟的actions日志文件，将新的agent活动实时更新到Zep图谱中。
    按平台分组，每累积BATCH_SIZE条活动后批量发送到Zep。
    
    所有有意义的行为都会被更新到Zep，action_args中会包含完整的上下文信息：
    - 点赞/踩的帖子原文
    - 转发/引用的帖子原文
    - 关注/屏蔽的用户名
    - 点赞/踩的评论原文
    """
    
    # 批量发送大小（每个平台累积多少条后发送）。
    # 类属性是默认值；实例会按 Config.GRAPH_MEMORY_BATCH_SIZE 覆盖：
    #   cloud → 5（保持原有行为）；graphiti → 20（本地抽取按"每批一次 LLM 调用"
    #   计费与计时，5 条一批会让整场回写耗时过长）。可用 GRAPH_MEMORY_BATCH_SIZE 覆盖。
    BATCH_SIZE = 5
    
    # 平台名称映射（用于控制台显示）
    PLATFORM_DISPLAY_NAMES = {
        'twitter': '世界1',
        'reddit': '世界2',
    }
    
    # 发送间隔（秒），避免请求过快
    SEND_INTERVAL = 0.5

    # 空闲刷新（秒）：队列静默超过该时长且缓冲区仍有活动时，立即发送"不足一批"的内容。
    # 没有这个刷新，当某平台活动数 < BATCH_SIZE 时缓冲区永远不会触发，
    # 而 SimulationRunner 会一直等待"图谱写入完成"，表现为模拟状态永久 running。
    IDLE_FLUSH_SECONDS = 10.0
    # 缓冲里长时间一条都没送出去时的告警阈值（只告警，不另开发送路径）
    STALL_WARN_SECONDS = 60.0
    
    # Zep recommends keeping an episode below 10,000 characters. Leave room
    # for future source formatting changes.
    MAX_EPISODE_CHARS = 9_500
    
    def __init__(
        self,
        graph_id: str,
        api_key: Optional[str] = None,
        simulation_id: Optional[str] = None,
    ):
        """
        初始化更新器
        
        Args:
            graph_id: Zep图谱ID
            api_key: Zep API Key（可选，默认从配置读取）
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id or "unknown"
        self.api_key = api_key or Config.ZEP_API_KEY

        # 后端由 ZEP_BACKEND 决定；api_key 仅在 cloud 模式下需要。
        self.client = get_graph_store(self.api_key)
        
        # 活动队列
        self._activity_queue: Queue = Queue()
        
        # 按平台分组的活动缓冲区（每个平台各自累积到BATCH_SIZE后批量发送）
        self._platform_buffers: Dict[str, List[AgentActivity]] = {
            'twitter': [],
            'reddit': [],
        }
        self._buffer_lock = threading.Lock()
        self._acceptance_lock = threading.Lock()
        
        # 控制标志
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        
        # 统计
        self._total_activities = 0  # 实际添加到队列的活动数
        self._total_sent = 0        # 成功发送到Zep的批次数
        self._total_items_sent = 0  # 成功发送到Zep的活动条数
        self._failed_count = 0      # 发送失败的批次数
        self._skipped_count = 0     # 被过滤跳过的活动数（DO_NOTHING）
        self._failed_batches: List[Dict[str, Any]] = []
        self._pending_episode_uuids: List[str] = []
        # 工作线程的故障（一旦发生就不允许被当成"正常完成"）
        self._worker_error: Optional[str] = None
        # 因超过单 episode 上限而被截断的活动数（原文已落盘）
        self._truncated_count = 0
        # 因超时/失败而落盘待处理的活动数（交付率的分母里必须扣掉，否则
        # "模拟产出的活动"会被静默当成"已进图谱"）
        self._persisted_count = 0
        # 停滞告警节流：同一次停滞只报一次
        self._stall_reported_at = 0.0

        # 批大小由配置决定（默认与云端一致：5）。禁止为了"看起来快"而在代码里调大。
        configured_batch = getattr(Config, 'GRAPH_MEMORY_BATCH_SIZE', None)
        if isinstance(configured_batch, int) and 1 <= configured_batch <= 50:
            self.BATCH_SIZE = configured_batch

        # 最后一次收到活动的时刻，用于空闲刷新判定
        self._last_activity_at = 0.0
        # 最后一次成功发送/冲刷的时刻（停滞告警用）
        self._last_flush_at = 0.0

        logger.info(f"ZepGraphMemoryUpdater 初始化完成: graph_id={graph_id}, batch_size={self.BATCH_SIZE}")
    
    def _get_platform_display_name(self, platform: str) -> str:
        """获取平台的显示名称"""
        return self.PLATFORM_DISPLAY_NAMES.get(platform.lower(), platform)
    
    def start(self):
        """启动后台工作线程"""
        if self._running:
            return

        # Capture locale before spawning background thread
        current_locale = get_locale()

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(current_locale,),
            daemon=True,
            name=f"ZepMemoryUpdater-{self.graph_id[:8]}"
        )
        self._worker_thread.start()
        logger.info(f"ZepGraphMemoryUpdater 已启动: graph_id={self.graph_id}")
    
    def stop(self):
        """Drain the worker, flush tail events, and wait for Cloud ingestion."""
        deadline = time.time() + ZEP_INGESTION_WAIT_TIMEOUT_SECONDS
        # Serialize the accepting->closed transition with add_activity's
        # check+enqueue operation. This closes the small race where a producer
        # could enqueue after both the worker and final flush had exited.
        with self._acceptance_lock:
            self._running = False

        if self._worker_thread and self._worker_thread.is_alive():
            join_timeout = max(0.0, deadline - time.time())
            self._worker_thread.join(timeout=join_timeout)
            if self._worker_thread.is_alive():
                raise TimeoutError(
                    f"Zep updater worker did not stop within {join_timeout:.0f}s"
                )

        # The worker has drained the queue. Only now is it safe to flush
        # buffers; doing this before join loses an item already dequeued by the
        # worker but not yet buffered.
        self._flush_remaining(deadline=deadline)

        # 工作线程出过故障就不能报告"已完成"：批次已落盘，运行必须判为未完成。
        if self._worker_error:
            raise RuntimeError(
                f"Zep updater 工作线程异常，图谱写入未完成: {self._worker_error}"
            )

        if self._failed_batches:
            raise RuntimeError(
                f"{len(self._failed_batches)} Zep activity batch(es) failed; "
                "simulation graph ingestion is incomplete"
            )

        self._wait_for_pending_episodes(deadline=deadline)
        
        logger.info(f"ZepGraphMemoryUpdater 已停止: graph_id={self.graph_id}, "
                   f"total_activities={self._total_activities}, "
                   f"batches_sent={self._total_sent}, "
                   f"items_sent={self._total_items_sent}, "
                   f"failed={self._failed_count}, "
                   f"skipped={self._skipped_count}")
    
    def add_activity(self, activity: AgentActivity):
        """
        添加一个agent活动到队列
        
        所有有意义的行为都会被添加到队列，包括：
        - CREATE_POST（发帖）
        - CREATE_COMMENT（评论）
        - QUOTE_POST（引用帖子）
        - SEARCH_POSTS（搜索帖子）
        - SEARCH_USER（搜索用户）
        - LIKE_POST/DISLIKE_POST（点赞/踩帖子）
        - REPOST（转发）
        - FOLLOW（关注）
        - MUTE（屏蔽）
        - LIKE_COMMENT/DISLIKE_COMMENT（点赞/踩评论）
        
        action_args中会包含完整的上下文信息（如帖子原文、用户名等）。
        
        Args:
            activity: Agent活动记录
        """
        # 跳过DO_NOTHING类型的活动
        if activity.action_type == "DO_NOTHING":
            self._skipped_count += 1
            return

        with self._acceptance_lock:
            if not self._running:
                raise RuntimeError("Zep graph updater is not running")
            self._activity_queue.put(activity)
            self._total_activities += 1
            self._last_activity_at = time.time()
        logger.debug(f"添加活动到Zep队列: {activity.agent_name} - {activity.action_type}")
    
    # actions.jsonl 里每条动作必须携带的字段：缺任何一个都说明上游写入有问题，
    # 补默认值只会把它伪装成一条正常活动（历史事故：缺失时间戳用 now() 顶上）。
    REQUIRED_ACTION_FIELDS = ("agent_id", "agent_name", "action_type", "round", "timestamp")

    def add_activity_from_dict(self, data: Dict[str, Any], platform: str):
        """
        从字典数据添加活动
        
        Args:
            data: 从actions.jsonl解析的字典数据
            platform: 平台名称 (twitter/reddit)
        """
        # 跳过事件类型的条目
        if "event_type" in data:
            return
        if data.get("success") is False:
            self._skipped_count += 1
            return

        missing = [name for name in self.REQUIRED_ACTION_FIELDS if data.get(name) is None]
        if missing:
            raise ValueError(
                f"{platform} actions.jsonl 记录缺少字段 {missing}，"
                f"无法写入图谱（不允许用默认值补齐）：{data!r}"
            )

        activity = AgentActivity(
            platform=platform,
            agent_id=data["agent_id"],
            agent_name=data["agent_name"],
            action_type=data["action_type"],
            action_args=data.get("action_args") or {},
            round_num=data["round"],
            timestamp=data["timestamp"],
        )
        
        self.add_activity(activity)
    
    def _worker_loop(self, locale: str = 'zh'):
        """后台工作循环 - 按平台批量发送活动到Zep"""
        set_locale(locale)
        while self._running or not self._activity_queue.empty():
            try:
                # 尝试从队列获取活动（超时1秒）
                try:
                    activity = self._activity_queue.get(timeout=1)
                    
                    # 将活动添加到对应平台的缓冲区
                    platform = activity.platform.lower()
                    batch = None
                    with self._buffer_lock:
                        if platform not in self._platform_buffers:
                            self._platform_buffers[platform] = []
                        self._platform_buffers[platform].append(activity)
                        
                        # 检查该平台是否达到批量大小
                        if len(self._platform_buffers[platform]) >= self.BATCH_SIZE:
                            batch = self._platform_buffers[platform][:self.BATCH_SIZE]
                            self._platform_buffers[platform] = self._platform_buffers[platform][self.BATCH_SIZE:]

                    # Never hold the buffer lock across network I/O or sleep.
                    if batch:
                        try:
                            self._send_batch_activities(batch, platform)
                        except BaseException as error:  # noqa: BLE001
                            # batch 已从缓冲区切走：这里若只 log 一句，这批活动就没了。
                            # 先落盘，再记录故障并停止循环（失败必须可见，不能继续装作正常）。
                            self._persist_unsent(
                                platform, batch, reason=f"worker loop error: {error}"
                            )
                            self._worker_error = f"回写工作线程异常: {error}"
                            logger.error(
                                "回写工作线程异常，批次已落盘并停止工作循环: %s", error
                            )
                            return
                        with self._buffer_lock:
                            self._last_flush_at = time.time()
                        time.sleep(self.SEND_INTERVAL)
                    
                except Empty:
                    self._flush_idle_buffers()

                # 队列一直有活动时不会走到上面的 Empty 分支，于是"缓冲有内容却
                # 从没送出"是静默的：这里把停滞变成告警（不另开发送路径）。
                self._report_stall_if_idle(time.time())

            except Exception as e:
                logger.error(f"工作循环异常: {e}")
                self._worker_error = f"回写工作循环异常: {e}"
                time.sleep(1)

    def _flush_idle_buffers(self) -> None:
        """空闲刷新：把不足一批的缓冲活动也发出去。

        触发条件：队列已空 + 距最后一条活动超过 IDLE_FLUSH_SECONDS + 该平台缓冲区非空。
        这样"活动总数 < BATCH_SIZE"的运行也能正常收敛（否则 runner 会永久等待）。
        """

        now = time.time()
        with self._buffer_lock:
            if not self._last_activity_at or now - self._last_activity_at < self.IDLE_FLUSH_SECONDS:
                return
            pending = [
                (platform, list(buffer))
                for platform, buffer in self._platform_buffers.items()
                if buffer
            ]
            # 不能"先清空再发送"：发送失败时这些活动会同时从缓冲区和内存里消失。
            # 发送成功后再删；失败由 _send_batch_activities 负责落盘，这里保留缓冲。
        for platform, activities in pending:
            logger.info(
                "空闲刷新：发送 %s 平台剩余的 %s 条活动（不足一批）",
                platform,
                len(activities),
            )
            try:
                self._send_batch_activities(activities, platform)
            except _DrainDeadlineExceeded as error:
                with self._buffer_lock:
                    del self._platform_buffers[platform][:error.processed_count]
                raise
            else:
                with self._buffer_lock:
                    del self._platform_buffers[platform][:len(activities)]
                    self._last_flush_at = time.time()
            time.sleep(self.SEND_INTERVAL)
    
    def _build_episode_payloads(
        self,
        activities: List[AgentActivity],
    ) -> List[tuple[List[AgentActivity], str]]:
        payloads: List[tuple[List[AgentActivity], str]] = []
        current_activities: List[AgentActivity] = []
        current_lines: List[str] = []
        current_length = 0

        for activity in activities:
            text = activity.to_episode_text()
            if len(text) > self.MAX_EPISODE_CHARS:
                # 不允许静默截断：截掉的内容既不在图里也不在任何失败清单里。
                # 这里把原文落盘留证，并在统计里计数，绝不"假装写进去了"。
                self._persist_unsent(
                    activity.platform.lower(),
                    [activity],
                    reason=(
                        f"single activity text {len(text)} chars exceeds "
                        f"MAX_EPISODE_CHARS={self.MAX_EPISODE_CHARS}; truncated"
                    ),
                )
                self._truncated_count += 1
                logger.error(
                    "活动文本 %d 字超过单 episode 上限 %d，已截断并落盘原文（agent=%s, action=%s）",
                    len(text), self.MAX_EPISODE_CHARS, activity.agent_name, activity.action_type,
                )
                marker = "... [truncated by MiroFish]"
                text = text[: self.MAX_EPISODE_CHARS - len(marker)] + marker
            projected_length = current_length + (1 if current_lines else 0) + len(text)
            if current_lines and projected_length > self.MAX_EPISODE_CHARS:
                payloads.append((current_activities, "\n".join(current_lines)))
                current_activities = []
                current_lines = []
                current_length = 0
            current_activities.append(activity)
            current_lines.append(text)
            current_length += (1 if len(current_lines) > 1 else 0) + len(text)

        if current_lines:
            payloads.append((current_activities, "\n".join(current_lines)))
        return payloads

    def _send_batch_activities(
        self,
        activities: List[AgentActivity],
        platform: str,
        *,
        deadline: float | None = None,
    ) -> int:
        """
        批量发送活动到Zep图谱（合并为一条文本）
        
        Args:
            activities: Agent活动列表
            platform: 平台名称
        """
        if not activities:
            return 0

        processed_count = 0
        for payload_activities, combined_text in self._build_episode_payloads(activities):
            if deadline is not None and time.time() >= deadline:
                raise _DrainDeadlineExceeded(processed_count)
            try:
                episode_uuid = self.client.add_episode(
                    self.graph_id,
                    combined_text,
                    "text",
                    created_at=self._to_rfc3339(payload_activities[-1].timestamp),
                    source_description="MiroFish simulation activity batch",
                    metadata={
                        "source": "mirofish_simulation",
                        "simulation_id": self.simulation_id,
                        "platform": platform,
                        "activity_count": len(payload_activities),
                        "first_round": min(a.round_num for a in payload_activities),
                        "last_round": max(a.round_num for a in payload_activities),
                        "agent_ids": ",".join(
                            str(value)
                            for value in sorted({a.agent_id for a in payload_activities})
                        ),
                        "action_types": ",".join(
                            value
                            for value in sorted({a.action_type for a in payload_activities})
                            if value
                        ) or "unknown",
                    },
                )

                if not episode_uuid:
                    raise RuntimeError("图存储 add_episode 未返回 episode UUID")
                self._pending_episode_uuids.append(str(episode_uuid))
                self._total_sent += 1
                self._total_items_sent += len(payload_activities)
                display_name = self._get_platform_display_name(platform)
                logger.info(f"成功批量发送 {len(payload_activities)} 条{display_name}活动到图谱 {self.graph_id}")
                logger.debug(f"批量内容预览: {combined_text[:200]}...")

            except Exception as e:
                # graph.add has no idempotency key. Replaying an ambiguous
                # response can duplicate extracted facts, so fail closed and
                # surface the incomplete batch to SimulationRunner.
                logger.error(f"批量发送到Zep失败，未自动重放非幂等写入: {e}")
                self._failed_count += 1
                self._failed_batches.append({
                    "platform": platform,
                    "activities": payload_activities,
                    "error": str(e),
                })
                # 内存里的失败批次会在进程退出时消失：落盘留痕，绝不静默丢数据
                self._persist_unsent(
                    platform, payload_activities, reason=f"send failed: {e}"
                )
            finally:
                # Successes have a confirmed episode UUID; failures are kept
                # durably in _failed_batches and must never be replayed. Either
                # way this payload is accounted for before moving on.
                processed_count += len(payload_activities)
        return processed_count

    @staticmethod
    def _to_rfc3339(value: str) -> str:
        """把活动时间戳转成 RFC3339；非法值直接抛错，不伪造当前时间。"""

        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"活动时间戳无法解析: {value!r}") from error
        if parsed.tzinfo is None:
            parsed = parsed.astimezone()
        return parsed.isoformat()

    def _flush_remaining(self, *, deadline: float | None = None):
        """发送队列和缓冲区中剩余的活动"""
        # 首先处理队列中剩余的活动，添加到缓冲区
        while not self._activity_queue.empty():
            try:
                activity = self._activity_queue.get_nowait()
                platform = activity.platform.lower()
                with self._buffer_lock:
                    if platform not in self._platform_buffers:
                        self._platform_buffers[platform] = []
                    self._platform_buffers[platform].append(activity)
            except Empty:
                break
        
        for platform in list(self._platform_buffers):
            with self._buffer_lock:
                buffer = list(self._platform_buffers.get(platform, []))
            if not buffer:
                continue
            display_name = self._get_platform_display_name(platform)
            logger.info(f"发送{display_name}平台剩余的 {len(buffer)} 条活动")
            if deadline is not None and time.time() >= deadline:
                self._persist_unsent(platform, buffer, reason="drain deadline elapsed")
                raise TimeoutError(
                    "Zep updater drain deadline elapsed before flushing all activities"
                )
            try:
                processed_count = self._send_batch_activities(
                    buffer,
                    platform,
                    deadline=deadline,
                )
            except _DrainDeadlineExceeded as error:
                unprocessed = buffer[error.processed_count:]
                with self._buffer_lock:
                    del self._platform_buffers[platform][:error.processed_count]
                self._persist_unsent(platform, unprocessed, reason="drain deadline elapsed")
                raise TimeoutError(str(error)) from error
            else:
                with self._buffer_lock:
                    del self._platform_buffers[platform][:processed_count]

    def _persist_unsent(
        self,
        platform: str,
        activities: List[AgentActivity],
        *,
        reason: str,
    ) -> Optional[str]:
        """把没能写入图谱的活动落盘，绝不让它们随进程消失。

        超时本身仍然抛错（失败必须可见），但"可见"不等于"可以丢数据"：
        这些活动会被追加到 uploads/graphs/pending/<graph_id>.jsonl，
        并以 ERROR 记录条数与路径，供人工或后续重放处理。
        """

        if not activities:
            return None
        try:
            os.makedirs(PENDING_DIR, exist_ok=True)
            path = os.path.join(PENDING_DIR, f"{self.graph_id}.jsonl")
            with open(path, "a", encoding="utf-8") as handle:
                for activity in activities:
                    handle.write(
                        json.dumps(
                            {
                                "simulation_id": self.simulation_id,
                                "graph_id": self.graph_id,
                                "platform": activity.platform,
                                "agent_id": activity.agent_id,
                                "agent_name": activity.agent_name,
                                "action_type": activity.action_type,
                                "action_args": activity.action_args,
                                "round": activity.round_num,
                                "timestamp": activity.timestamp,
                                "unsent_reason": reason,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        except OSError as error:
            logger.error(
                "未写入图谱的 %s 条 %s 活动落盘失败（数据即将丢失）: %s",
                len(activities), platform, error,
            )
            raise
        logger.error(
            "%s 条 %s 活动未能写入图谱（%s），已落盘待处理: %s",
            len(activities), platform, reason, path,
        )
        with self._buffer_lock:
            self._persisted_count += len(activities)
        return path

    def _report_stall_if_idle(self, now: float) -> None:
        """缓冲里有活动但长时间一条都没送出去 → 告警（只告警，不另开发送路径）。

        历史上这里的问题不是"缺少第二条发送路径"，而是"没人知道它卡住了"：
        runner 状态停在 running。告警把"卡住"变成可见事件。
        """

        with self._buffer_lock:
            buffered = sum(len(b) for b in self._platform_buffers.values())
            last = self._last_flush_at or self._last_activity_at
        if not buffered or not last:
            return
        idle_for = now - last
        if idle_for < self.STALL_WARN_SECONDS:
            return
        if now - self._stall_reported_at < self.STALL_WARN_SECONDS:
            return
        self._stall_reported_at = now
        logger.warning(
            "回写停滞告警：缓冲区还有 %d 条活动，已 %.0f 秒没有成功发送"
            "（batch_size=%d，queue=%d）——运行状态会显示未完成，"
            "停止时会落盘到 %s",
            buffered, idle_for, self.BATCH_SIZE, self._activity_queue.qsize(), PENDING_DIR,
        )

    def _wait_for_pending_episodes(self, *, deadline: float | None = None) -> None:
        pending = set(self._pending_episode_uuids)
        if not pending:
            return

        if deadline is None:
            deadline = time.time() + ZEP_INGESTION_WAIT_TIMEOUT_SECONDS
        while pending:
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Zep simulation ingestion timed out with {len(pending)} "
                    "episode(s) pending"
                )
            for episode_uuid in list(pending):
                if self.client.get_episode_status(episode_uuid).processed:
                    pending.remove(episode_uuid)
            if pending:
                time.sleep(3)
        self._pending_episode_uuids = []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。

        交付率 = 真正进图谱的条数 / 应当进图谱的条数。没有这个数字，
        "回写慢"只能靠感觉说（见掩盖台账 M-02）。

        口径：`_total_activities` 只在活动**被接收进队列**时累加，DO_NOTHING 在累加之前
        就 return 了；所以"应入库条数"就是 `_total_activities`，不能再减一次
        `_skipped_count`。历史实现减了，于是 5 条活动（1 条 DO_NOTHING）算出
        expected=4、delivery_ratio=1.25、undelivered_items=-1——一个会**高估交付**的
        代理指标：真的只送到 4 条时它也报 1.0（"全部送达"）。见掩盖台账 N-27。
        """

        with self._buffer_lock:
            buffer_sizes = {p: len(b) for p, b in self._platform_buffers.items()}
            buffered = sum(buffer_sizes.values())

        expected = max(0, self._total_activities)
        delivered = self._total_items_sent
        undelivered = expected - delivered
        delivery_ratio = (delivered / expected) if expected else None

        return {
            "graph_id": self.graph_id,
            "batch_size": self.BATCH_SIZE,
            "total_activities": self._total_activities,  # 添加到队列的活动总数
            "expected_items": expected,                  # 扣除 DO_NOTHING 后应入库条数
            "batches_sent": self._total_sent,            # 成功发送的批次数
            "items_sent": delivered,                     # 成功发送的活动条数
            "delivery_ratio": delivery_ratio,            # 交付率（None 表示还没有应入库数据）
            "undelivered_items": undelivered,            # 尚未入库的条数（含在途缓冲）
            "failed_count": self._failed_count,          # 发送失败的批次数
            "persisted_count": self._persisted_count,    # 已落盘待处理的活动条数
            "pending_episode_count": len(self._pending_episode_uuids),
            "skipped_count": self._skipped_count,        # 被过滤跳过的活动数（DO_NOTHING）
            "queue_size": self._activity_queue.qsize(),
            "buffer_sizes": buffer_sizes,                # 各平台缓冲区大小
            "buffered_items": buffered,
            "running": self._running,
            "worker_error": self._worker_error,          # 非 None 表示图谱写入未完成
            "truncated_count": self._truncated_count,    # 超长被截断（原文已落盘）
        }


class ZepGraphMemoryManager:
    """
    管理多个模拟的Zep图谱记忆更新器
    
    每个模拟可以有自己的更新器实例
    """
    
    _updaters: Dict[str, ZepGraphMemoryUpdater] = {}
    _lock = threading.Lock()
    
    @classmethod
    def create_updater(cls, simulation_id: str, graph_id: str) -> ZepGraphMemoryUpdater:
        """
        为模拟创建图谱记忆更新器
        
        Args:
            simulation_id: 模拟ID
            graph_id: Zep图谱ID
            
        Returns:
            ZepGraphMemoryUpdater实例
        """
        with cls._lock:
            # 如果已存在，先停止旧的
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()
            
            updater = ZepGraphMemoryUpdater(
                graph_id,
                simulation_id=simulation_id,
            )
            updater.start()
            cls._updaters[simulation_id] = updater
            cls._stop_all_done = False
            
            logger.info(f"创建图谱记忆更新器: simulation_id={simulation_id}, graph_id={graph_id}")
            return updater
    
    @classmethod
    def get_updater(cls, simulation_id: str) -> Optional[ZepGraphMemoryUpdater]:
        """获取模拟的更新器"""
        with cls._lock:
            return cls._updaters.get(simulation_id)

    @classmethod
    def get_simulation_ids_for_graph(cls, graph_id: str) -> List[str]:
        """Return simulations whose updater still owns or drains this graph."""

        with cls._lock:
            return sorted(
                simulation_id
                for simulation_id, updater in cls._updaters.items()
                if updater.graph_id == graph_id
            )

    @classmethod
    def get_simulation_ids(cls) -> List[str]:
        """Return every simulation with a retained updater."""

        with cls._lock:
            return sorted(cls._updaters)

    @classmethod
    def discard_inactive_updater(cls, simulation_id: str) -> bool:
        """Discard a failed, fully stopped updater during graph destruction."""

        with cls._lock:
            updater = cls._updaters.get(simulation_id)
            if updater is None:
                return False
            worker_alive = bool(
                updater._worker_thread and updater._worker_thread.is_alive()
            )
            if updater._running or worker_alive:
                raise RuntimeError(
                    f"Zep updater for {simulation_id} is still active"
                )
            cls._updaters.pop(simulation_id, None)
        logger.warning(
            "Discarded incomplete Zep updater during explicit graph deletion: "
            "simulation_id=%s, graph_id=%s",
            simulation_id,
            updater.graph_id,
        )
        return True
    
    @classmethod
    def stop_updater(cls, simulation_id: str):
        """停止并移除模拟的更新器"""
        with cls._lock:
            updater = cls._updaters.get(simulation_id)
        if updater is None:
            return

        # Do not hold the manager lock through up to several minutes of Cloud
        # polling. Crucially, only remove the updater after a successful drain;
        # on failure it remains visible to report/deletion barriers and can be
        # stopped again.
        updater.stop()
        with cls._lock:
            if cls._updaters.get(simulation_id) is updater:
                cls._updaters.pop(simulation_id, None)
        logger.info(f"已停止图谱记忆更新器: simulation_id={simulation_id}")
    
    # 防止 stop_all 重复调用的标志
    _stop_all_done = False
    
    @classmethod
    def stop_all(cls):
        """停止所有更新器"""
        # 防止重复调用
        if cls._stop_all_done:
            return

        with cls._lock:
            simulation_ids = list(cls._updaters)

        errors = []
        for simulation_id in simulation_ids:
            try:
                cls.stop_updater(simulation_id)
            except Exception as error:
                # Keep a failed updater registered so the caller can retry and
                # lifecycle/report guards still see the incomplete ingestion.
                logger.error(
                    "停止更新器失败: simulation_id=%s, error=%s",
                    simulation_id,
                    error,
                )
                errors.append((simulation_id, error))

        with cls._lock:
            cls._stop_all_done = not cls._updaters

        if errors:
            details = "; ".join(
                f"{simulation_id}: {error}"
                for simulation_id, error in errors
            )
            raise RuntimeError(f"部分图谱更新器未完整停止: {details}")
        logger.info("已停止所有图谱记忆更新器")
    
    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """获取所有更新器的统计信息"""
        return {
            sim_id: updater.get_stats() 
            for sim_id, updater in cls._updaters.items()
        }
