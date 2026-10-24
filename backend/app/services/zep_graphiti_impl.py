"""本地 Graphiti 后端实现（ZEP_BACKEND=graphiti）。

设计要点
--------
* **图数据库**：嵌入式 Kuzu（`graphiti-core[kuzu]`）。不需要 Docker / JVM /
  常驻服务，数据落在项目内单个文件，靠 `group_id` 做多图谱隔离。
* **异步桥**：Flask 是同步的、Graphiti 是 async 的，因此用一个私有事件循环
  线程承载所有 Graphiti 调用（参考实现的同款做法）。
* **本体透传**：`set_ontology` 缓存 pydantic 模型，`add_episode*` 时透传
  `entity_types/edge_types/edge_type_map`。**这是让 `filter_defined_entities`
  能按业务类型过滤的前提**（参考实现在这里是 no-op，会导致造人设阶段实体数为 0）。
* **属性清洗**：写库前用 `graphiti_patch` 把嵌套属性序列化，规避图库只接受
  原始类型及其数组的限制。

来源说明：结构移植自 tt-a1i/MiroFish-local（AGPL-3.0）
`backend/app/services/zep_graphiti_impl.py`，驱动、本体、检索与读路径按本项目
main（@7657031）的调用面重写。
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid as uuid_module
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ..config import Config
from ..utils.local_embedder import graphiti_embedder_client
from ..utils.logger import get_logger
from ..utils.ontology import build_edge_models, build_entity_models
from .zep_adapter import (
    EpisodeBatch,
    EpisodeStatus,
    GraphEdge,
    GraphNode,
    GraphNotFoundError,
    GraphStoreError,
    GraphStoreUnavailableError,
    SearchResult,
    ZepClientAdapter,
)

logger = get_logger("mirofish.zep_graphiti")

# Graphiti 的 EntityNode 字段不允许出现在自定义实体类型里
GRAPHITI_RESERVED_FIELDS = frozenset(
    {"uuid", "name", "group_id", "labels", "attributes", "summary",
     "created_at", "name_embedding"}
)

_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None
_loop_lock = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """返回承载 Graphiti 协程的私有事件循环（进程内单例）。"""

    global _loop, _loop_thread
    with _loop_lock:
        if _loop is None or _loop.is_closed() or not _loop.is_running():
            _loop = asyncio.new_event_loop()
            _loop_thread = threading.Thread(
                target=_loop.run_forever, name="graphiti-async-loop", daemon=True
            )
            _loop_thread.start()
            logger.debug("已启动 Graphiti 私有事件循环")
    return _loop


def _run(coro, timeout: Optional[float] = None):
    """在私有循环上同步等待一个协程。

    注意：不能在私有循环内部调用（本模块所有入口都是同步方法，天然满足）。
    """

    loop = _ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


def shutdown_loop() -> None:
    """停止私有事件循环（进程退出或测试清理用）。"""

    global _loop, _loop_thread
    with _loop_lock:
        if _loop is not None and _loop.is_running():
            _loop.call_soon_threadsafe(_loop.stop)
        _loop = None
        _loop_thread = None


# 本后端检索只有一条路径：bm25 + cosine 的 RRF 融合。
# Graphiti 构造时 `cross_encoder=None`，因此"交叉编码器重排"在这条路径上并不存在；
# 把它列为可选值只会造出"配置写着 cross_encoder、实际走 rrf"的静默改写。
SUPPORTED_RERANKERS = ("rrf",)


def _optional_str(value: Any) -> Optional[str]:
    """把可空字段转成字符串或 None。

    不能直接 `str(value) or None`：`str(None)` 是字符串 `"None"`，它是真值，
    于是"没有时间"会被读成"有时间"，`is_invalid` / `is_expired` 全部成立
    （实测：本地图所有边被判为历史事实，见 N-08 / N-14）。
    """

    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nat"}:
        return None
    return text


def _reranker_enums(name: str) -> tuple[Any, Any]:
    """把配置里的 reranker 名映射成 graphiti 的枚举。

    映射本身就是"配置是否真的生效"的保证：解析结果必须一路传到
    `EdgeSearchConfig` / `NodeSearchConfig`，否则校验只是在自欺。
    """

    from graphiti_core.search.search_config import EdgeReranker, NodeReranker

    mapping = {
        "rrf": (EdgeReranker.rrf, NodeReranker.rrf),
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise GraphStoreError(
            f"reranker={name!r} 没有对应的 graphiti 枚举；本后端可选 "
            f"{' / '.join(mapping)}（配置值必须真的生效，不接受映射不到的取值）"
        ) from error


def _enforce_kuzu_single_threaded(driver: Any) -> int:
    """把 Kuzu 的每条连接限制成单线程执行，返回被设置的连接数。

    为什么需要它：graphiti 的 `KuzuDriver` 只暴露 `max_concurrent_queries`，
    `max_threads_per_query` 在 `kuzu.AsyncConnection` 构造时就固定为 0（查询内部不限制并行）。
    本机实测（faulthandler dump，`掩盖台账.md` N-24）：写入回写期间
    `kuzu.Connection.execute` 以 access violation 杀死整个后端进程——没有 Python 异常、
    没有 atexit、`state.json` 永远停在 `running`。

    拿不到 `driver.client.connections` 时抛错而不是跳过：这意味着"单线程"这个前提没有被建立，
    而它正是避免上面那次进程级崩溃的条件（AGENTS.md §1.1 缺前提即失败）。
    """

    client = getattr(driver, "client", None)
    connections = list(getattr(client, "connections", None) or [])
    if not connections:
        raise GraphStoreUnavailableError(
            "无法为 Kuzu 连接设置单线程执行（driver.client.connections 不可用）；"
            "Kuzu 并行查询在本机会以 access violation 杀死整个后端进程（N-24），"
            "不允许在无法建立该前提的情况下继续"
        )
    for connection in connections:
        connection.set_max_threads_for_exec(Config.GRAPHITI_MAX_THREADS_PER_QUERY)
    return len(connections)


class GraphitiClient(ZepClientAdapter):
    """本地 Graphiti + Kuzu 适配器。"""

    backend_name = "graphiti"

    def __init__(self) -> None:
        self.db_path = Config.GRAPHITI_DB_PATH
        self._graphiti: Any = None
        self._driver: Any = None
        self._initialized = False
        self._init_lock = threading.Lock()
        self._ontology_cache: Dict[str, Dict[str, Any]] = {}
        self._graph_metadata: Dict[str, Dict[str, Any]] = {}
        self._last_ontology: Dict[str, Any] = {}

    # ---------------- 初始化 ----------------

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            try:
                self._do_initialize()
            except ImportError as error:
                raise GraphStoreUnavailableError(
                    "graphiti-core 未安装，无法使用 ZEP_BACKEND=graphiti。"
                    "请运行: uv sync --extra graphiti"
                ) from error
            except Exception as error:  # noqa: BLE001
                raise GraphStoreUnavailableError(
                    f"本地 Graphiti 初始化失败: {error}"
                ) from error
            self._initialized = True

    def _do_initialize(self) -> None:
        from graphiti_core import Graphiti
        from graphiti_core.driver.kuzu_driver import KuzuDriver

        from .graphiti_patch import apply_driver_compat, apply_llm_prompt_patch, apply_patch

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # 这两处补丁是本地化分支能跑的前提：补丁没打上就不是"少一点优化"，
        # 而是写入会整批失败、结构化输出会把 schema 抄回来。因此失败即初始化失败。
        if not apply_patch():
            raise GraphStoreUnavailableError(
                "Graphiti 属性清洗补丁未应用（图库只接受原始类型属性："
                "未打补丁时嵌套属性会导致整批 episode 写入失败）"
            )
        if not apply_llm_prompt_patch():
            raise GraphStoreUnavailableError(
                "Graphiti 结构化输出补丁未应用（兼容端点会把 schema 定义当应答返回："
                "未打补丁时 edge/实体抽取会大面积校验失败）"
            )

        # Graphiti 的索引维度与 LLM 并发由环境变量控制，这里显式对齐本项目配置
        os.environ.setdefault("EMBEDDING_DIM", str(Config.GRAPHITI_EMBEDDING_DIM))
        os.environ.setdefault("SEMAPHORE_LIMIT", "10")
        os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
        # Graphiti 在未显式传入 cross_encoder 时会构造默认的 OpenAI reranker，
        # 它读取 OPENAI_* 环境变量。我们的检索走 RRF，不需要该 reranker，
        # 但仍需让构造通过；与参考实现一致地从 LLM_* 映射。
        if Config.LLM_API_KEY:
            os.environ.setdefault("OPENAI_API_KEY", Config.LLM_API_KEY)
        if Config.LLM_BASE_URL:
            os.environ.setdefault("OPENAI_BASE_URL", Config.LLM_BASE_URL)

        self._driver = KuzuDriver(
            db=self.db_path,
            max_concurrent_queries=Config.GRAPHITI_MAX_CONCURRENT_QUERIES,
        )
        # 单连接 + 单线程：并发执行 Kuzu 查询会以原生 access violation 杀死后端进程（N-24）
        connection_count = _enforce_kuzu_single_threaded(self._driver)
        logger.info(
            "Kuzu 执行并发已锁定: connections=%d, max_threads_per_query=%d（N-24：并行执行会崩进程）",
            connection_count,
            Config.GRAPHITI_MAX_THREADS_PER_QUERY,
        )
        apply_driver_compat(self._driver)
        from .graphiti_patch import rebuild_kuzu_fulltext_indexes_async

        # 索引建不起来就是初始化失败：检索没有第二条路径（见 AGENTS.md 禁止兜底）。
        # 每次都重建：库里那份旧索引会让"更新已存在节点"以 access violation 崩掉整个进程（N-24）。
        _run(rebuild_kuzu_fulltext_indexes_async(self._driver), timeout=180)
        self._graphiti = Graphiti(
            graph_driver=self._driver,
            llm_client=self._build_llm_client(),
            embedder=graphiti_embedder_client(Config.GRAPHITI_EMBEDDING_MODEL_PATH),
            cross_encoder=None,
        )
        _run(self._graphiti.build_indices_and_constraints(), timeout=300)
        logger.info("本地 Graphiti 就绪: db=%s", self.db_path)

    def _build_llm_client(self) -> Any:
        """OpenAI 兼容 LLM 客户端（DeepSeek 等非 OpenAI 官方端点走 json_object）。"""

        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        model = Config.GRAPHITI_LLM_MODEL or Config.LLM_MODEL_NAME
        small_model = Config.GRAPHITI_LLM_SMALL_MODEL or model
        config = LLMConfig(
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL,
            model=model,
            small_model=small_model,
            temperature=0,
            max_tokens=Config.GRAPHITI_LLM_MAX_TOKENS,
        )
        client = OpenAIGenericClient(config=config)
        # 非 OpenAI 官方端点通常不支持严格 json_schema，退回 json_object
        mode = Config.GRAPHITI_LLM_STRUCTURED_MODE
        if mode and hasattr(client, "structured_output_mode"):
            try:
                client.structured_output_mode = mode
            except Exception:  # noqa: BLE001 - 只影响抽取稳健性，不阻断初始化
                logger.warning("无法设置 structured_output_mode=%s", mode)
        return client

    # ---------------- 本体 ----------------

    def build_ontology_models(
        self, ontology: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        from pydantic import BaseModel, Field

        # 记住原始本体 JSON，便于 set_ontology 时落盘（重启后仍能恢复）
        self._last_ontology = dict(ontology or {})
        entity_types = build_entity_models(
            ontology,
            base_model=BaseModel,
            attribute_type=Optional[str],
            field_factory=Field,
            reserved_names=GRAPHITI_RESERVED_FIELDS,
        )
        # Graphiti 的边类型模型必须暴露 fact 字段（LLM 抽取 schema 的必需项）
        raw_edges = build_edge_models(
            ontology,
            base_model=BaseModel,
            attribute_type=Optional[str],
            field_factory=Field,
            reserved_names=GRAPHITI_RESERVED_FIELDS,
            extra_fields={"fact": Field(description="fact representing the edge")},
        )
        return entity_types, raw_edges

    def set_ontology(
        self,
        graph_ids: List[str],
        entities: Optional[Dict[str, Any]] = None,
        edges: Optional[Dict[str, Any]] = None,
    ) -> None:
        entity_types = entities or {}
        edge_types: Dict[str, Any] = {}
        edge_type_map: Dict[tuple, List[str]] = {}
        for name, payload in (edges or {}).items():
            edge_class, source_targets = payload
            edge_types[name] = edge_class
            for pair in source_targets or []:
                key = (pair.get("source", "Entity"), pair.get("target", "Entity"))
                edge_type_map.setdefault(key, []).append(name)

        for graph_id in graph_ids:
            self._ontology_cache[graph_id] = {
                "entity_types": entity_types or None,
                "edge_types": edge_types or None,
                "edge_type_map": edge_type_map or None,
            }
            logger.info(
                "本体已注册: graph_id=%s entity_types=%d edge_types=%d",
                graph_id,
                len(entity_types),
                len(edge_types),
            )
        self._persist_ontology(graph_ids)

    # ---------------- 本体持久化（跨进程重启） ----------------

    @property
    def _ontology_dir(self) -> str:
        return os.path.join(os.path.dirname(self.db_path), "ontology")

    def _persist_ontology(self, graph_ids: List[str]) -> None:
        """把原始本体 JSON 落盘。

        进程内的本体缓存重启即失效（回写阶段会退化成无本体抽取），因此按
        graph_id 持久化一份，缓存未命中时自动重建。
        """

        ontology = getattr(self, "_last_ontology", None)
        if not ontology:
            return
        try:
            os.makedirs(self._ontology_dir, exist_ok=True)
            for graph_id in graph_ids:
                path = os.path.join(self._ontology_dir, f"{graph_id}.json")
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump({"ontology": ontology}, handle, ensure_ascii=False)
        except Exception as error:  # noqa: BLE001 - 持久化失败不应阻断建图
            logger.warning("本体落盘失败: %s", error)

    def _load_persisted_ontology(self, graph_id: str) -> bool:
        path = os.path.join(self._ontology_dir, f"{graph_id}.json")
        if not os.path.exists(path):
            return False
        try:
            with open(path, encoding="utf-8") as handle:
                ontology = json.load(handle).get("ontology") or {}
            entities, edges = self.build_ontology_models(ontology)
            self.set_ontology([graph_id], entities=entities, edges=edges)
            logger.info("已从磁盘恢复本体: graph_id=%s", graph_id)
            return True
        except Exception as error:  # noqa: BLE001
            logger.warning("恢复本体失败(%s): %s", graph_id, error)
            return False

    def _ontology_kwargs(self, graph_id: str) -> Dict[str, Any]:
        if graph_id not in self._ontology_cache:
            # 进程重启后缓存为空：尝试从磁盘恢复，避免静默退化成无本体抽取
            self._load_persisted_ontology(graph_id)
        cached = self._ontology_cache.get(graph_id)
        instructions = Config.GRAPHITI_EXTRACTION_INSTRUCTIONS or None
        if not cached:
            logger.warning(
                "graph_id=%s 没有注册本体，抽取将使用 Graphiti 默认类型；"
                "这会导致按业务实体类型过滤时命中为 0",
                graph_id,
            )
            return {
                "entity_types": None,
                "edge_types": None,
                "edge_type_map": None,
                "custom_extraction_instructions": instructions,
            }
        return {**cached, "custom_extraction_instructions": instructions}

    # ---------------- Graph ----------------

    def create_graph(self, graph_id: str, name: str, description: str) -> None:
        self._ensure_initialized()
        self._graph_metadata[graph_id] = {
            "name": name,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def _execute(self, cypher: str, **params: Any) -> List[Dict[str, Any]]:
        """执行一条查询并返回记录列表（Graphiti 驱动返回三元组）。"""

        result = _run(self._driver.execute_query(cypher, **params))
        if isinstance(result, tuple):
            records = result[0]
        else:
            records = result
        return list(records or [])

    def graph_exists(self, graph_id: str) -> bool:
        self._ensure_initialized()
        if graph_id in self._graph_metadata:
            return True
        records = self._execute(
            "MATCH (n {group_id: $group_id}) RETURN count(n) AS c", group_id=graph_id
        )
        if not records:
            return False
        first = records[0]
        count = first.get("c") if isinstance(first, dict) else first
        return bool(count)

    def delete_graph(self, graph_id: str) -> None:
        self._ensure_initialized()
        try:
            self._execute(
                "MATCH (n {group_id: $group_id}) DETACH DELETE n", group_id=graph_id
            )
        except Exception as error:  # noqa: BLE001
            raise GraphStoreError(f"删除本地图谱失败: {error}") from error
        self._graph_metadata.pop(graph_id, None)
        self._ontology_cache.pop(graph_id, None)
        logger.info("本地图谱已删除: graph_id=%s", graph_id)

    # ---------------- Episode ----------------

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
        self._ensure_initialized()
        from graphiti_core.nodes import EpisodeType

        reference_time = self._parse_time(created_at)
        kwargs = self._ontology_kwargs(graph_id)
        result = self._add_episode_with_retry(
            name=f"episode_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            data=data,
            reference_time=reference_time,
            episode_type=episode_type,
            graph_id=graph_id,
            kwargs=kwargs,
            source_description=source_description or "mirofish",
        )
        episode = getattr(result, "episode", None)
        episode_uuid = getattr(episode, "uuid", None) or getattr(episode, "uuid_", None)
        if not episode_uuid:
            raise GraphStoreError("Graphiti add_episode 未返回 episode UUID")
        return str(episode_uuid)

    # 结构化输出在兼容端点上偶发失败（同一提示词多数成功、少数回显 schema）。
    # 这类失败发生在写库之前，重放不会产生重复数据（且图内去重会收敛同义事实），
    # 因此按"有限次数 + 退避"重试，而不是让整批回写失败。
    ADD_EPISODE_ATTEMPTS = 3
    RETRYABLE_ERROR_NAMES = (
        "ValidationError",
        "EmptyResponseError",
        "RateLimitError",
        "JSONDecodeError",
    )

    def _add_episode_with_retry(
        self,
        *,
        name: str,
        data: str,
        reference_time: datetime,
        episode_type: str,
        graph_id: str,
        kwargs: Dict[str, Any],
        source_description: str,
    ) -> Any:
        from graphiti_core.nodes import EpisodeType

        last_error: Optional[BaseException] = None
        for attempt in range(1, self.ADD_EPISODE_ATTEMPTS + 1):
            try:
                return _run(
                    self._graphiti.add_episode(
                        name=name,
                        episode_body=data,
                        source_description=source_description,
                        reference_time=reference_time,
                        source=self._episode_type(episode_type, EpisodeType),
                        group_id=graph_id,
                        entity_types=kwargs["entity_types"],
                        edge_types=kwargs["edge_types"],
                        edge_type_map=kwargs["edge_type_map"],
                        custom_extraction_instructions=kwargs[
                            "custom_extraction_instructions"
                        ],
                    )
                )
            except Exception as error:  # noqa: BLE001 - 需要按类型判定是否重试
                last_error = error
                retryable = type(error).__name__ in self.RETRYABLE_ERROR_NAMES or any(
                    marker in str(error)
                    for marker in ("validation error", "Field required", "Unterminated string")
                )
                if not retryable or attempt == self.ADD_EPISODE_ATTEMPTS:
                    break
                logger.warning(
                    "add_episode 第 %s/%s 次失败（%s），退避后重试: %s",
                    attempt,
                    self.ADD_EPISODE_ATTEMPTS,
                    type(error).__name__,
                    str(error).replace("\n", " ")[:160],
                )
                time.sleep(1.5 * attempt)

        raise GraphStoreError(
            f"add_episode 连续 {self.ADD_EPISODE_ATTEMPTS} 次失败: "
            f"{type(last_error).__name__}: {str(last_error).replace(chr(10), ' ')[:300]}"
        ) from last_error

    def add_episodes(
        self,
        graph_id: str,
        chunks: List[str],
        *,
        batch_size: int = 350,
        progress_cb: Optional[Callable[[str, float], None]] = None,
        batch_created_cb: Optional[Callable[[Optional[str], str]]] = None,
    ) -> EpisodeBatch:
        self._ensure_initialized()
        from graphiti_core.nodes import EpisodeType
        from graphiti_core.utils.bulk_utils import RawEpisode

        total = len(chunks)
        slice_size = max(1, Config.GRAPHITI_BULK_SLICE)
        operation_id = uuid_module.uuid4().hex
        batch_id = f"graphiti-{uuid_module.uuid4().hex[:12]}"
        if batch_created_cb:
            batch_created_cb(batch_id, operation_id)

        kwargs = self._ontology_kwargs(graph_id)
        episode_uuids: List[str] = []
        reference_time = datetime.now(timezone.utc)

        for start in range(0, total, slice_size):
            window = chunks[start:start + slice_size]
            raw_episodes = [
                RawEpisode(
                    name=f"episode_{graph_id}_{start + offset}",
                    content=chunk,
                    source_description="mirofish_document_chunk",
                    source=EpisodeType.text,
                    reference_time=reference_time,
                )
                for offset, chunk in enumerate(window)
            ]
            if progress_cb:
                progress_cb(
                    f"本地图谱正在抽取第 {start // slice_size + 1} 批（{len(window)} 段）",
                    min((start + len(window)) / max(total, 1), 1.0),
                )
            result = _run(
                self._graphiti.add_episode_bulk(
                    bulk_episodes=raw_episodes,
                    group_id=graph_id,
                    entity_types=kwargs["entity_types"],
                    edge_types=kwargs["edge_types"],
                    edge_type_map=kwargs["edge_type_map"],
                    custom_extraction_instructions=kwargs["custom_extraction_instructions"],
                ),
                timeout=None,
            )
            for episode in getattr(result, "episodes", None) or []:
                episode_uuid = getattr(episode, "uuid", None)
                if episode_uuid:
                    episode_uuids.append(str(episode_uuid))

        if progress_cb:
            progress_cb(f"本地图谱抽取完成（{len(episode_uuids)} 个 episode）", 1.0)

        return EpisodeBatch(
            batch_id=batch_id,
            operation_id=operation_id,
            episode_uuids=episode_uuids,
            item_count=total,
        )

    def get_episode_status(self, episode_uuid: str) -> EpisodeStatus:
        # Graphiti 同步写入：add_episode/add_episode_bulk 返回即已处理。
        return EpisodeStatus(uuid=episode_uuid, processed=True)

    def wait_for_episode(self, episode_uuid: str, timeout: int = 300) -> bool:
        return True

    # ---------------- Node / Edge ----------------

    def get_all_nodes(
        self,
        graph_id: str,
        *,
        page_size: int = 100,
        max_items: Optional[int] = None,
    ) -> List[GraphNode]:
        self._ensure_initialized()
        from graphiti_core.nodes import EntityNode

        limit = min(max(page_size, 1), Config.GRAPHITI_PAGE_SIZE)
        nodes: List[GraphNode] = []
        cursor: Optional[str] = None
        try:
            from graphiti_core.errors import GroupsNodesNotFoundError
        except ImportError:  # pragma: no cover - 版本差异兜底
            GroupsNodesNotFoundError = ()  # type: ignore[assignment]
        while True:
            try:
                batch = _run(
                    EntityNode.get_by_group_ids(
                        self._driver, [graph_id], limit=limit, uuid_cursor=cursor
                    )
                )
            except GroupsNodesNotFoundError:  # 空图不是错误
                break
            if not batch:
                break
            for node in batch:
                nodes.append(self._to_graph_node(node))
                if max_items is not None and len(nodes) >= max_items:
                    return nodes
            if len(batch) < limit:
                break
            cursor = getattr(batch[-1], "uuid", None)
            if not cursor:
                break
        return nodes

    def get_all_edges(
        self,
        graph_id: str,
        *,
        page_size: int = 100,
        max_items: Optional[int] = None,
    ) -> List[GraphEdge]:
        self._ensure_initialized()
        from graphiti_core.edges import EntityEdge

        limit = min(max(page_size, 1), Config.GRAPHITI_PAGE_SIZE)
        edges: List[GraphEdge] = []
        cursor: Optional[str] = None
        try:
            from graphiti_core.errors import GroupsEdgesNotFoundError
        except ImportError:  # pragma: no cover - 版本差异兜底
            GroupsEdgesNotFoundError = ()  # type: ignore[assignment]
        while True:
            try:
                batch = _run(
                    EntityEdge.get_by_group_ids(
                        self._driver, [graph_id], limit=limit, uuid_cursor=cursor
                    )
                )
            except GroupsEdgesNotFoundError:  # 空图不是错误
                break
            if not batch:
                break
            for edge in batch:
                edges.append(self._to_graph_edge(edge))
                if max_items is not None and len(edges) >= max_items:
                    return edges
            if len(batch) < limit:
                break
            cursor = getattr(batch[-1], "uuid", None)
            if not cursor:
                break
        return edges

    def get_node(self, node_uuid: str) -> Optional[GraphNode]:
        self._ensure_initialized()
        from graphiti_core.nodes import EntityNode

        try:
            node = _run(EntityNode.get_by_uuid(self._driver, node_uuid))
        except GraphNotFoundError:
            return None
        except Exception as error:  # noqa: BLE001 - 本地驱动没有统一 404 类型
            if "not found" in str(error).lower():
                return None
            raise GraphStoreError(f"读取节点失败: {error}") from error
        if node is None:
            return None
        return self._to_graph_node(node)

    def get_node_edges(self, node_uuid: str) -> List[GraphEdge]:
        """取与节点相关的边（双向）。

        Graphiti 的 ``EntityEdge.get_by_node_uuid`` 只覆盖部分情形，且 Kuzu 把边
        建模为 `RelatesToNode_` 节点（经 RELATES_TO 关系与 Entity 相连），因此这里
        直接用无向 Cypher 取两侧，保证 incoming/outgoing 都不漏。
        """

        self._ensure_initialized()
        from graphiti_core.edges import EntityEdge

        records = self._execute(
            "MATCH (n:Entity {uuid: $uuid})-[r:RELATES_TO]-(e:RelatesToNode_) "
            "RETURN e.uuid AS uuid, e.name AS name, e.fact AS fact, "
            "e.group_id AS group_id, e.valid_at AS valid_at, e.invalid_at AS invalid_at, "
            "e.expired_at AS expired_at, e.reference_time AS created_at, "
            "e.episodes AS episodes",
            uuid=node_uuid,
        )
        if records:
            edges: List[GraphEdge] = []
            for record in records:
                source_uuid, target_uuid = self._relates_to_endpoints(record.get("uuid"))
                edges.append(
                    GraphEdge(
                        uuid=str(record.get("uuid") or ""),
                        name=record.get("name") or "",
                        fact=record.get("fact") or "",
                        source_node_uuid=source_uuid,
                        target_node_uuid=target_uuid,
                        attributes={},
                        created_at=str(record.get("created_at") or "") or None,
                        valid_at=str(record.get("valid_at") or "") or None,
                        invalid_at=str(record.get("invalid_at") or "") or None,
                        expired_at=str(record.get("expired_at") or "") or None,
                        episodes=[str(item) for item in (record.get("episodes") or [])],
                    )
                )
            return edges

        # 兜底：走 Graphiti 原生 API
        edges = _run(EntityEdge.get_by_node_uuid(self._driver, node_uuid))
        return [self._to_graph_edge(edge) for edge in (edges or [])]

    def _relates_to_endpoints(self, edge_uuid: Any) -> tuple[str, str]:
        """解析 RelatesToNode_ 两侧的 Entity uuid（source, target）。"""

        if not edge_uuid:
            return "", ""
        endpoints = self._execute(
            "MATCH (s:Entity)-[:RELATES_TO]->(e:RelatesToNode_ {uuid: $uuid}) "
            "RETURN s.uuid AS s LIMIT 1",
            uuid=str(edge_uuid),
        )
        sources = self._execute(
            "MATCH (e:RelatesToNode_ {uuid: $uuid})-[:RELATES_TO]->(t:Entity) "
            "RETURN t.uuid AS t LIMIT 1",
            uuid=str(edge_uuid),
        )
        source = (endpoints[0].get("s") if endpoints else "") or ""
        target = (sources[0].get("t") if sources else "") or ""
        return str(source), str(target)

    # ---------------- Search ----------------

    def _resolve_reranker(self, requested: Optional[str]) -> str:
        """确定本次检索使用的 reranker，并且保证这个值真的会被用上。

        配置是唯一权威：`requested` 只在未配置时作为默认值；与配置不一致即报错。
        这里有过两层历史事故：
        1. `cross_encoder` 被静默改成 `rrf`（靠 URL 猜端点能力）；
        2. 加完校验之后，`_execute_search()` 仍硬编码 `EdgeReranker.rrf`，
           于是"配置成 cross_encoder"变成**校验通过但实际不生效**——
           和第一层一模一样的静默改写，只是方向相反。

        现在只允许本后端真实可用的取值：检索是 bm25 + cosine 的 RRF 融合，
        Graphiti 构造时 `cross_encoder=None`，所以 `cross_encoder` / `none`
        一律在配置阶段就拒绝，而不是留一个看起来存在、实际换路的开关。
        """

        configured = (Config.GRAPHITI_RERANKER or "").strip().lower()
        requested_name = (requested or "").strip().lower()
        if configured and requested_name and requested_name != configured:
            raise GraphStoreError(
                f"检索请求的 reranker={requested_name} 与配置 GRAPHITI_RERANKER="
                f"{configured} 不一致；配置为唯一权威，请显式修改配置"
            )
        effective = configured or requested_name or SUPPORTED_RERANKERS[0]
        if effective not in SUPPORTED_RERANKERS:
            raise GraphStoreError(
                f"GRAPHITI_RERANKER={effective!r} 在本后端不可用：检索只有 "
                "bm25 + cosine 的 RRF 一条路径（Graphiti 以 cross_encoder=None 构造）。"
                f"可选：{' / '.join(SUPPORTED_RERANKERS)}；"
                "如需交叉编码器重排，必须先实现并实测接入，不能先出现在配置里"
            )
        return effective

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
        reranker: Optional[str] = None,
    ) -> SearchResult:
        self._ensure_initialized()
        from ..utils.zep import normalize_zep_search_limit, normalize_zep_search_query

        normalized_query = normalize_zep_search_query(query)
        normalized_limit = normalize_zep_search_limit(limit)
        effective = self._resolve_reranker(reranker)

        # 只有一条检索路径：`_execute_search`（bm25 + cosine + RRF）。
        # 这里不再有 "reranker=none → 换用 graphiti.search()" 或
        # `hasattr(search_)` 能力探测换路的写法——那会让同一次检索因为一个配置值
        # 或一个上游版本差异而换算法，且换完以后检索语义与阈值都变了。
        nodes, edges, top_score = self._execute_search(
            normalized_query, graph_id, scope, normalized_limit, reranker=effective,
        )
        self._log_search_outcome(
            normalized_query, normalized_limit,
            Config.GRAPHITI_MIN_SCORE, top_score, len(edges) + len(nodes),
        )
        return SearchResult(nodes=nodes, edges=edges)

    @staticmethod
    def _log_search_outcome(
        query: str,
        limit: int,
        threshold: Optional[float],
        top_score: Optional[float],
        returned: int,
    ) -> None:
        """每次检索都记录：阈值、条数、上限、最高分。

        没有这行日志，"返回了 N 条"和"返回了 N 条相关事实"就无法区分——
        这正是上一轮把检索结论做废的原因。
        """

        threshold_text = "上游默认(0.6)" if threshold is None else f"{threshold}"
        score_text = "-" if top_score is None else f"{top_score:.4f}"
        logger.info(
            "图谱检索: 阈值=%s 返回=%d/%d 最高分=%s query=%s",
            threshold_text, returned, limit, score_text, query[:60],
        )
        if returned >= limit and limit > 0:
            logger.warning(
                "检索饱和: 返回条数(%d)达到请求上限(%d)，阈值(%s)没有过滤掉任何结果；"
                "相关性未经筛选，引用前必须人工核对",
                returned, limit, threshold_text,
            )

    def _execute_search(
        self,
        query: str,
        graph_id: str,
        scope: str,
        limit: int,
        reranker: str = SUPPORTED_RERANKERS[0],
    ) -> tuple[List[GraphNode], List[GraphEdge], Optional[float]]:
        """执行一次混合检索（bm25 + cosine），只有这一条路径。

        `reranker` 是 `_resolve_reranker()` 的返回值，必须真的传进 graphiti 的
        搜索配置——否则配置项等于装饰。
        """

        from graphiti_core.search.search_config import (
            EdgeSearchConfig,
            EdgeSearchMethod,
            NodeSearchConfig,
            NodeSearchMethod,
            SearchConfig,
        )

        edge_reranker, node_reranker = _reranker_enums(reranker)
        threshold = Config.GRAPHITI_MIN_SCORE
        min_score_kwargs = {} if threshold is None else {"sim_min_score": threshold}

        def edge_config() -> EdgeSearchConfig:
            return EdgeSearchConfig(
                search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
                reranker=edge_reranker,
                **min_score_kwargs,
            )

        def node_config() -> NodeSearchConfig:
            return NodeSearchConfig(
                search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
                reranker=node_reranker,
                **min_score_kwargs,
            )

        if scope == "nodes":
            config = SearchConfig(node_config=node_config(), limit=limit)
        elif scope == "edges":
            config = SearchConfig(edge_config=edge_config(), limit=limit)
        else:
            config = SearchConfig(
                edge_config=edge_config(), node_config=node_config(), limit=limit
            )

        result = _run(
            self._graphiti.search_(
                query=query, config=config, group_ids=[graph_id]
            ),
            timeout=Config.GRAPHITI_SEARCH_TIMEOUT,
        )
        nodes = [self._to_graph_node(n) for n in (getattr(result, "nodes", None) or [])]
        edges = [self._to_graph_edge(e) for e in (getattr(result, "edges", None) or [])]
        scores = list(getattr(result, "edge_reranker_scores", None) or []) + list(
            getattr(result, "node_reranker_scores", None) or []
        )
        top_score = max(scores) if scores else None
        return nodes, edges, top_score

    # ---------------- 转换与工具 ----------------

    @staticmethod
    def _episode_type(value: str, episode_type_enum):
        mapping = {
            "text": episode_type_enum.text,
            "message": episode_type_enum.message,
            "json": episode_type_enum.json,
        }
        key = (value or "").lower()
        if key not in mapping:
            raise GraphStoreError(
                f"不支持的 episode 类型 {value!r}（可选 text / message / json）"
            )
        return mapping[key]

    @staticmethod
    def _parse_time(value: Optional[str]) -> datetime:
        """解析时间戳；缺失或非法都抛错，不再回落到 now()。

        回落会把"这条数据的真实时间未知"伪装成一个看起来正常的时间，
        下游（时效性判断、时间线）会照此计算。
        """
        if not value:
            raise GraphStoreError("缺少时间戳，无法确定该数据的 reference_time")
        normalized = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise GraphStoreError(f"时间戳无法解析: {value!r}") from error
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    @staticmethod
    def _to_graph_node(node: Any) -> GraphNode:
        uuid_value = getattr(node, "uuid", None) or getattr(node, "uuid_", "")
        created_at = getattr(node, "created_at", None)
        return GraphNode(
            uuid=str(uuid_value or ""),
            name=getattr(node, "name", "") or "",
            labels=list(getattr(node, "labels", None) or []),
            summary=getattr(node, "summary", "") or "",
            attributes=dict(getattr(node, "attributes", None) or {}),
            created_at=str(created_at) if created_at else None,
        )

    @staticmethod
    def _to_graph_edge(edge: Any) -> GraphEdge:
        """把 graphiti 的边对象转成本项目的 GraphEdge。

        时间字段必须用 `_optional_str`：`str(None)` 会得到字符串 `"None"`，
        而它是真值——于是 `EdgeInfo.is_invalid` / `is_expired`（`invalid_at is not None`）
        对**每一条边**都成立。实测后果：本地图 `mirofish_2a5aa6e3eb04467c` 的
        `panorama_search` 报"当前有效事实 0 条 / 历史 68 条"，而云端同一工具是
        353 / 4（见 `掩盖台账.md` N-08 / N-14）。
        """

        uuid_value = getattr(edge, "uuid", None) or getattr(edge, "uuid_", "")
        episodes = getattr(edge, "episodes", None) or []
        return GraphEdge(
            uuid=str(uuid_value or ""),
            name=getattr(edge, "name", "") or "",
            fact=getattr(edge, "fact", "") or "",
            source_node_uuid=str(getattr(edge, "source_node_uuid", "") or ""),
            target_node_uuid=str(getattr(edge, "target_node_uuid", "") or ""),
            attributes=dict(getattr(edge, "attributes", None) or {}),
            created_at=_optional_str(getattr(edge, "created_at", None)),
            valid_at=_optional_str(getattr(edge, "valid_at", None)),
            invalid_at=_optional_str(getattr(edge, "invalid_at", None)),
            expired_at=_optional_str(getattr(edge, "expired_at", None)),
            episodes=[str(item) for item in episodes],
            fact_type=getattr(edge, "fact_type", None) or getattr(edge, "name", None),
        )

    # ---------------- 生命周期 ----------------

    def close(self) -> None:
        driver = self._driver
        self._driver = None
        self._graphiti = None
        self._initialized = False
        if driver is not None:
            close = getattr(driver, "close", None)
            if close:
                try:
                    result = close()
                    if asyncio.iscoroutine(result):
                        _run(result, timeout=30)
                except Exception as error:  # noqa: BLE001
                    logger.warning("关闭 Kuzu 驱动失败: %s", error)

    def backend_info(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_name,
            "db_path": self.db_path,
            "embedding_model": Config.GRAPHITI_EMBEDDING_MODEL,
            "reranker": Config.GRAPHITI_RERANKER,
        }
