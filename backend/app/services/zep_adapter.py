"""图存储适配器：统一 Zep Cloud 与本地 Graphiti 两种后端。

设计约束
--------
1. 本模块**不得** import `zep_cloud` 或 `graphiti_core`；两个实现各自惰性 import 自己的 SDK。
2. 调用方（graph_builder / zep_entity_reader / zep_tools / zep_graph_memory_updater /
   oasis_profile_generator / zep_paging）只依赖本模块定义的类型与方法。
3. 后端由 `ZEP_BACKEND` 选择，默认 `cloud`（零行为变更）。

来源说明：接口与数据结构移植自 tt-a1i/MiroFish-local（AGPL-3.0）
`backend/app/services/zep_adapter.py`，并按本项目 main（@7657031）的调用面扩展。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# ============================================================
# 异常契约
# ============================================================


class GraphStoreError(RuntimeError):
    """图存储层错误基类。"""


class GraphNotFoundError(GraphStoreError):
    """图谱 / 节点 / 边不存在（等价于 zep_cloud.NotFoundError）。"""


class GraphStoreUnavailableError(GraphStoreError):
    """后端不可用：依赖缺失、连接失败或配置不完整。"""


# ============================================================
# 数据结构（字段与 Zep Cloud SDK 对齐，供既有消费方直接读取）
# ============================================================


@dataclass
class GraphNode:
    """图谱节点。"""

    uuid: str
    name: str
    labels: List[str] = field(default_factory=list)
    summary: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None


@dataclass
class GraphEdge:
    """图谱边（含双时态字段）。"""

    uuid: str
    name: str = ""
    fact: str = ""
    source_node_uuid: str = ""
    target_node_uuid: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    episodes: List[str] = field(default_factory=list)
    fact_type: Optional[str] = None


@dataclass
class SearchResult:
    """检索结果。"""

    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)


@dataclass
class EpisodeStatus:
    """Episode 处理状态。"""

    uuid: str
    processed: bool


@dataclass
class EpisodeBatch:
    """批量写入结果。

    字段与 `graph_builder.BatchSubmission` 保持一致，便于无差别替换。
    """

    batch_id: str
    operation_id: str
    episode_uuids: List[str] = field(default_factory=list)
    item_count: int = 0


# 进度回调：(message, progress 0..1)
ProgressCallback = Callable[[str, float], None]
# 批次身份回调：(batch_id | None, operation_id)
BatchCreatedCallback = Callable[[Optional[str], str], None]


# ============================================================
# 适配器接口
# ============================================================


class ZepClientAdapter(ABC):
    """统一的图存储客户端接口。"""

    # 后端名称，用于日志与自检
    backend_name: str = "unknown"

    # ---------------- Graph ----------------

    @abstractmethod
    def create_graph(self, graph_id: str, name: str, description: str) -> None:
        """创建图谱。已存在时按后端语义处理（Cloud 幂等失败，Graphiti 记录元数据）。"""

    @abstractmethod
    def graph_exists(self, graph_id: str) -> bool:
        """图谱是否存在。"""

    @abstractmethod
    def delete_graph(self, graph_id: str) -> None:
        """删除图谱及其全部数据。不存在时抛 GraphNotFoundError。"""

    @abstractmethod
    def build_ontology_models(self, ontology: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """把本体 JSON 转成后端原生模型。

        Returns:
            (entities, edges)：Cloud 为 zep_cloud 的 EntityModel/EdgeModel 派生类；
            Graphiti 为 pydantic BaseModel 派生类。
        """

    @abstractmethod
    def set_ontology(
        self,
        graph_ids: List[str],
        entities: Optional[Dict[str, Any]] = None,
        edges: Optional[Dict[str, Any]] = None,
    ) -> None:
        """注册本体。"""

    # ---------------- Episode ----------------

    @abstractmethod
    def add_episode(
        self,
        graph_id: str,
        data: str,
        episode_type: str = "text",
        *,
        created_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_description: Optional[str] = None,
    ) -> str:
        """写入单条 episode，返回 episode UUID。"""

    @abstractmethod
    def add_episodes(
        self,
        graph_id: str,
        chunks: List[str],
        *,
        batch_size: int = 350,
        progress_cb: Optional[ProgressCallback] = None,
        batch_created_cb: Optional[BatchCreatedCallback] = None,
    ) -> EpisodeBatch:
        """批量写入文档 chunk。Cloud 走 Batch API，Graphiti 走 add_episode_bulk。"""

    @abstractmethod
    def get_episode_status(self, episode_uuid: str) -> EpisodeStatus:
        """查询 episode 处理状态。"""

    def wait_for_batch(
        self,
        batch: EpisodeBatch,
        *,
        progress_cb: Optional[ProgressCallback] = None,
        timeout: Optional[int] = None,
    ) -> List[str]:
        """等待批量写入进入终态并返回 episode UUID 列表。

        Cloud 需要轮询 Batch API；Graphiti 是同步写入，默认实现直接返回。
        """

        return list(batch.episode_uuids)

    def get_batch_summary(self, batch_id: str) -> Any:
        """读取已持久化的批次身份（重启对账用）。

        Cloud 返回 Batch API 的对象；本地后端没有批次概念，返回 None。
        """

        return None

    def wait_for_episode(self, episode_uuid: str, timeout: int = 300) -> bool:
        """等待 episode 处理完成（默认轮询；Graphiti 同步处理，覆盖为直接返回 True）。"""

        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.get_episode_status(episode_uuid).processed:
                    return True
            except GraphNotFoundError:
                return False
            time.sleep(2)
        return False

    # ---------------- Node / Edge ----------------

    @abstractmethod
    def get_all_nodes(
        self,
        graph_id: str,
        *,
        page_size: int = 100,
        max_items: Optional[int] = None,
    ) -> List[GraphNode]:
        """列出图谱全部节点（内部分页）。"""

    @abstractmethod
    def get_node(self, node_uuid: str) -> Optional[GraphNode]:
        """取单个节点；不存在返回 None。"""

    @abstractmethod
    def get_node_edges(self, node_uuid: str) -> List[GraphEdge]:
        """取与节点相关的全部边。"""

    @abstractmethod
    def get_all_edges(
        self,
        graph_id: str,
        *,
        page_size: int = 100,
        max_items: Optional[int] = None,
    ) -> List[GraphEdge]:
        """列出图谱全部边（内部分页）。"""

    # ---------------- Search ----------------

    @abstractmethod
    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
        reranker: Optional[str] = None,
    ) -> SearchResult:
        """图检索。scope: edges | nodes | both。

        `reranker=None` 表示"由后端配置决定"：graphiti 后端看 `GRAPHITI_RERANKER`
        （只支持它真能用的取值），cloud 后端用 Zep Cloud 的默认重排。
        接口层不再替任何后端规定默认值——那样会让"配置是唯一权威"变成"接口里
        悄悄有一个默认值"（历史事故见 `掩盖台账.md` M-07 / N-07）。
        """

    # ---------------- 生命周期 ----------------

    def close(self) -> None:
        """释放资源。默认空实现；Graphiti 实现需关闭驱动与私有事件循环。"""

    def backend_info(self) -> Dict[str, Any]:
        """自检信息（只读，用于 /api/health 或冒烟脚本）。"""

        return {"backend": self.backend_name}
