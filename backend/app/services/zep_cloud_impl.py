"""Zep Cloud 后端实现：包装 zep-cloud SDK，实现 ZepClientAdapter。

来源说明：接口实现思路移植自 tt-a1i/MiroFish-local（AGPL-3.0）
`backend/app/services/zep_cloud_impl.py`；批量写入与分页逻辑直接取自本项目
main（@7657031）`services/graph_builder.py` 的既有实现，以保证行为完全一致。
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Callable, Dict, List, Optional

from zep_cloud import BatchAddItem, EntityEdgeSourceTarget, NotFoundError

from ..config import Config
from ..utils.locale import t
from ..utils.logger import get_logger
from ..utils.ontology import (
    build_edge_models,
    build_entity_models,
)
from ..utils.zep import (
    ZEP_INGESTION_WAIT_TIMEOUT_SECONDS,
    call_zep_read_with_retry,
    get_zep_client,
    is_retryable_zep_error,
)
from ..utils.zep_paging import fetch_all_edges, fetch_all_nodes
from .zep_adapter import (
    EpisodeBatch,
    EpisodeStatus,
    GraphEdge,
    GraphNode,
    GraphNotFoundError,
    GraphStoreUnavailableError,
    SearchResult,
    ZepClientAdapter,
)

logger = get_logger("mirofish.zep_cloud")

TERMINAL_BATCH_STATES = {"succeeded", "partial", "failed", "invalid", "canceled"}


def _node_uuid(obj: Any) -> str:
    return getattr(obj, "uuid_", None) or getattr(obj, "uuid", "") or ""


class ZepCloudClient(ZepClientAdapter):
    """Zep Cloud 适配器。"""

    backend_name = "cloud"

    def __init__(self, api_key: Optional[str] = None):
        key = (api_key or Config.ZEP_API_KEY or "").strip()
        if not key:
            raise GraphStoreUnavailableError(
                "ZEP_API_KEY 未配置（ZEP_BACKEND=cloud 时需要）"
            )
        self._api_key = key
        self.client = get_zep_client(key)

    # ---------------- 转换 ----------------

    @staticmethod
    def _to_node(node: Any) -> GraphNode:
        return GraphNode(
            uuid=_node_uuid(node),
            name=getattr(node, "name", "") or "",
            labels=list(getattr(node, "labels", None) or []),
            summary=getattr(node, "summary", "") or "",
            attributes=dict(getattr(node, "attributes", None) or {}),
            created_at=getattr(node, "created_at", None),
        )

    @staticmethod
    def _to_edge(edge: Any) -> GraphEdge:
        return GraphEdge(
            uuid=_node_uuid(edge),
            name=getattr(edge, "name", "") or "",
            fact=getattr(edge, "fact", "") or "",
            source_node_uuid=getattr(edge, "source_node_uuid", "") or "",
            target_node_uuid=getattr(edge, "target_node_uuid", "") or "",
            attributes=dict(getattr(edge, "attributes", None) or {}),
            created_at=getattr(edge, "created_at", None),
            valid_at=getattr(edge, "valid_at", None),
            invalid_at=getattr(edge, "invalid_at", None),
            expired_at=getattr(edge, "expired_at", None),
            episodes=list(getattr(edge, "episodes", None) or []),
            fact_type=getattr(edge, "fact_type", None),
        )

    # ---------------- Graph ----------------

    def create_graph(self, graph_id: str, name: str, description: str) -> None:
        """创建图谱；响应丢失时用只读 GET 对账，避免非幂等重放。"""

        try:
            self.client.graph.create(
                graph_id=graph_id,
                name=name,
                description=description,
            )
            return
        except Exception as error:  # noqa: BLE001 - 需要按类型分流
            if not is_retryable_zep_error(error):
                raise

            reconciliation_error: Optional[BaseException] = None
            for attempt in range(3):
                try:
                    call_zep_read_with_retry(
                        lambda: self.client.graph.get(graph_id),
                        operation_name=f"reconcile graph create {graph_id}",
                    )
                    reconciliation_error = None
                    break
                except NotFoundError as not_found:
                    reconciliation_error = not_found
                    if attempt < 2:
                        time.sleep(attempt + 1)
                except Exception as read_error:  # noqa: BLE001
                    reconciliation_error = read_error
                    break
            if reconciliation_error is not None:
                raise error from reconciliation_error

    def graph_exists(self, graph_id: str) -> bool:
        try:
            call_zep_read_with_retry(
                lambda: self.client.graph.get(graph_id),
                operation_name=f"graph exists {graph_id}",
            )
            return True
        except NotFoundError:
            return False

    def delete_graph(self, graph_id: str) -> None:
        try:
            self.client.graph.delete(graph_id=graph_id)
        except NotFoundError as not_found:
            raise GraphNotFoundError(graph_id) from not_found

    def build_ontology_models(self, ontology: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """构建 zep_cloud 的 EntityModel / EdgeModel 派生类。"""

        import warnings
        from typing import Optional as _Optional

        from pydantic import Field
        from zep_cloud.external_clients.ontology import (
            EdgeModel,
            EntityModel,
            EntityText,
        )

        # 抑制 Pydantic v2 关于 Field(default=None) 的警告：这是 Zep SDK 要求的用法。
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

        entity_types = build_entity_models(
            ontology,
            base_model=EntityModel,
            attribute_type=_Optional[EntityText],
            field_factory=Field,
        )
        raw_edges = build_edge_models(
            ontology,
            base_model=EdgeModel,
            attribute_type=_Optional[str],
            field_factory=Field,
        )

        edge_definitions: Dict[str, Any] = {}
        for name, (edge_class, source_targets) in raw_edges.items():
            zep_targets = [
                EntityEdgeSourceTarget(
                    source=pair.get("source", "Entity"),
                    target=pair.get("target", "Entity"),
                )
                for pair in source_targets
            ]
            if zep_targets:
                edge_definitions[name] = (edge_class, zep_targets)

        return entity_types, edge_definitions

    def set_ontology(
        self,
        graph_ids: List[str],
        entities: Optional[Dict[str, Any]] = None,
        edges: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not entities and not edges:
            return
        self.client.graph.set_ontology(
            graph_ids=graph_ids,
            # Zep iterates entities.items(), so edge-only ontologies must
            # pass an empty dictionary rather than None.
            entities=entities if entities else {},
            edges=edges if edges else None,
        )

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
        episode = self.client.graph.add(
            graph_id=graph_id,
            type=episode_type,
            data=data,
            created_at=created_at,
            source_description=source_description or "mirofish_simulation",
            metadata=metadata or {},
        )
        episode_uuid = getattr(episode, "uuid_", None) or getattr(episode, "uuid", None)
        if not episode_uuid:
            raise RuntimeError("Zep graph.add returned no episode UUID")
        return str(episode_uuid)

    def add_episodes(
        self,
        graph_id: str,
        chunks: List[str],
        *,
        batch_size: int = 350,
        progress_cb: Optional[Callable[[str, float], None]] = None,
        batch_created_cb: Optional[Callable[[Optional[str], str], None]] = None,
    ) -> EpisodeBatch:
        """Submit document chunks through Zep's Batch API.

        Mutating calls are deliberately not retried: create/add are not
        documented as idempotent, and an ambiguous replay can duplicate graph
        episodes. The returned batch identity allows callers to persist and
        reconcile the operation instead.
        """

        if not graph_id:
            raise ValueError("graph_id is required")

        total_chunks = len(chunks)
        payload_hash = hashlib.sha256("\0".join(chunks).encode("utf-8")).hexdigest()
        operation_id = hashlib.sha256(
            f"{graph_id}:{payload_hash}".encode("utf-8")
        ).hexdigest()
        if batch_created_cb:
            batch_created_cb(None, operation_id)

        try:
            batch = self.client.batch.create(
                metadata={
                    "mirofish_operation_id": operation_id,
                    "graph_id": graph_id,
                    "chunk_count": total_chunks,
                }
            )
        except Exception as error:  # noqa: BLE001
            if not is_retryable_zep_error(error):
                raise
            batch = self._find_batch_by_operation_id(graph_id, operation_id)
            if batch is None:
                raise RuntimeError(
                    "Zep batch creation is unconfirmed and no matching operation was found"
                ) from error

        batch_id = getattr(batch, "batch_id", None)
        if not batch_id:
            raise RuntimeError("Zep Batch API returned no batch_id")
        if batch_created_cb:
            batch_created_cb(batch_id, operation_id)

        episode_uuids: List[str] = []
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size

            if progress_cb:
                progress = (i + len(batch_chunks)) / total_chunks
                progress_cb(
                    t(
                        'progress.sendingBatch',
                        current=batch_num,
                        total=total_batches,
                        chunks=len(batch_chunks),
                    ),
                    progress,
                )

            items = [
                BatchAddItem(
                    type="graph_episode",
                    graph_id=graph_id,
                    data=chunk,
                    data_type="text",
                    source_description="MiroFish source document chunk",
                    metadata={
                        "mirofish_operation_id": operation_id,
                        "chunk_index": i + offset,
                        "chunk_sha256": hashlib.sha256(
                            chunk.encode("utf-8")
                        ).hexdigest(),
                    },
                )
                for offset, chunk in enumerate(batch_chunks)
            ]

            expected_item_count = i + len(items)
            try:
                item_details = self.client.batch.add(
                    batch_id=batch_id,
                    items=items,
                )
            except Exception as error:  # noqa: BLE001
                if progress_cb:
                    progress_cb(
                        t('progress.batchFailed', batch=batch_num, error=str(error)), 0
                    )
                if is_retryable_zep_error(error):
                    recovered_items = self._reconcile_batch_item_count(
                        batch_id,
                        expected_item_count,
                    )
                    if self._covers_expected(recovered_items, expected_item_count):
                        item_details = recovered_items[i:expected_item_count]
                    else:
                        raise RuntimeError(
                            f"Zep batch {batch_id} item submission is unconfirmed; "
                            "the draft was not processed or replayed"
                        ) from error
                else:
                    raise RuntimeError(
                        f"Zep batch {batch_id} item submission failed"
                    ) from error

            if len(item_details or []) != len(items):
                recovered_items = self._reconcile_batch_item_count(
                    batch_id,
                    expected_item_count,
                )
                if self._covers_expected(recovered_items, expected_item_count):
                    item_details = recovered_items[i:expected_item_count]
                else:
                    raise RuntimeError(
                        f"Zep batch {batch_id} acknowledged {len(item_details or [])} "
                        f"of {len(items)} items"
                    )
            for item in item_details:
                episode_uuid = getattr(item, "episode_uuid", None)
                if episode_uuid:
                    episode_uuids.append(episode_uuid)

        try:
            self.client.batch.process(batch_id=batch_id)
        except Exception as error:  # noqa: BLE001
            # A process response can be lost after the server accepted it.
            # Reconcile with a safe GET instead of issuing a second POST.
            summary = call_zep_read_with_retry(
                lambda: self.client.batch.get(batch_id=batch_id),
                operation_name=f"reconcile batch {batch_id}",
            )
            if getattr(summary, "status", None) in {None, "draft"}:
                raise RuntimeError(
                    f"Zep batch {batch_id} processing is unconfirmed"
                ) from error

        return EpisodeBatch(
            batch_id=batch_id,
            operation_id=operation_id,
            episode_uuids=episode_uuids,
            item_count=total_chunks,
        )

    def wait_for_batch(
        self,
        batch: EpisodeBatch,
        *,
        progress_cb: Optional[Callable[[str, float], None]] = None,
        timeout: Optional[int] = None,
    ) -> List[str]:
        """Wait for a Batch API terminal state and validate every item."""

        timeout = timeout or ZEP_INGESTION_WAIT_TIMEOUT_SECONDS
        start_time = time.time()
        status = None

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Zep batch {batch.batch_id} did not finish within {timeout}s"
                )

            summary = call_zep_read_with_retry(
                lambda: self.client.batch.get(batch_id=batch.batch_id),
                operation_name=f"poll batch {batch.batch_id}",
            )
            status = getattr(summary, "status", None)
            progress = getattr(summary, "progress", None)
            percent = float(getattr(progress, "percent_complete", 0) or 0) / 100
            if progress_cb:
                completed = int(getattr(progress, "succeeded_items", 0) or 0)
                progress_cb(
                    t(
                        'progress.zepProcessing',
                        completed=completed,
                        total=batch.item_count,
                        pending=max(batch.item_count - completed, 0),
                        elapsed=int(time.time() - start_time),
                    ),
                    min(max(percent, 0.0), 1.0),
                )

            if status in TERMINAL_BATCH_STATES:
                break
            time.sleep(3)

        items = self._list_batch_items(batch.batch_id)
        if status != "succeeded":
            failed_items = [
                item
                for item in items
                if getattr(item, "status", None) not in {"succeeded", "skipped"}
            ]
            first_error = getattr(failed_items[0], "error", None) if failed_items else None
            raise RuntimeError(
                f"Zep batch {batch.batch_id} ended as {status}; "
                f"failed_items={len(failed_items)}; first_error={first_error}"
            )
        if len(items) != batch.item_count:
            raise RuntimeError(
                f"Zep batch {batch.batch_id} contains {len(items)} items, "
                f"expected {batch.item_count}"
            )

        ordered_items = sorted(
            items,
            key=lambda item: getattr(item, "sequence_index", 0) or 0,
        )
        episode_uuids: List[str] = []
        for item in ordered_items:
            item_status = getattr(item, "status", None)
            episode_uuid = getattr(item, "episode_uuid", None)
            source_uuid = getattr(item, "source_uuid", None)
            if item_status != "succeeded" or not episode_uuid:
                raise RuntimeError(
                    f"Zep batch {batch.batch_id} returned an incomplete item"
                )
            if source_uuid and source_uuid != episode_uuid:
                raise RuntimeError(
                    f"Zep batch {batch.batch_id} returned mismatched episode UUIDs"
                )
            episode_uuids.append(episode_uuid)

        if progress_cb:
            progress_cb(
                t(
                    'progress.processingComplete',
                    completed=len(episode_uuids),
                    total=batch.item_count,
                ),
                1.0,
            )
        return episode_uuids

    def get_episode_status(self, episode_uuid: str) -> EpisodeStatus:
        try:
            episode = call_zep_read_with_retry(
                lambda: self.client.graph.episode.get(uuid_=episode_uuid),
                operation_name=f"poll episode {episode_uuid}",
            )
        except NotFoundError as not_found:
            raise GraphNotFoundError(episode_uuid) from not_found
        return EpisodeStatus(uuid=episode_uuid, processed=bool(getattr(episode, "processed", False)))

    def get_batch_summary(self, batch_id: str) -> Any:
        """读取批次摘要（重启对账用）。"""

        return call_zep_read_with_retry(
            lambda: self.client.batch.get(batch_id=batch_id),
            operation_name=f"get batch {batch_id}",
        )

    # ---------------- 内部：batch 对账 ----------------

    def _list_batch_items(self, batch_id: str) -> List[Any]:
        items: List[Any] = []
        cursor: int | None = None
        seen_cursors: set[int] = set()
        while True:
            page = call_zep_read_with_retry(
                lambda: self.client.batch.list_items(
                    batch_id=batch_id,
                    limit=100,
                    cursor=cursor,
                ),
                operation_name=f"list batch items {batch_id}",
            )
            items.extend(getattr(page, "items", None) or [])
            next_cursor = getattr(page, "next_cursor", None)
            if next_cursor is None:
                break
            if next_cursor == cursor or next_cursor in seen_cursors:
                raise RuntimeError(f"Zep batch {batch_id} item cursor did not advance")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        return items

    def _reconcile_batch_item_count(
        self,
        batch_id: str,
        expected_item_count: int,
        *,
        max_attempts: int = 3,
    ) -> List[Any]:
        """Allow a short propagation window after an ambiguous add reply."""

        items: List[Any] = []
        for attempt in range(1, max_attempts + 1):
            items = self._list_batch_items(batch_id)
            if len(items) >= expected_item_count:
                return items
            if attempt < max_attempts:
                time.sleep(attempt)
        return items

    @staticmethod
    def _covers_expected(items: List[Any], expected_item_count: int) -> bool:
        indexes = {getattr(item, "sequence_index", None) for item in items}
        return len(items) == expected_item_count and indexes == set(range(expected_item_count))

    def _find_batch_by_operation_id(
        self,
        graph_id: str,
        operation_id: str,
        *,
        max_attempts: int = 3,
    ) -> Any | None:
        """Find one server-created batch after an ambiguous create reply."""

        for attempt in range(1, max_attempts + 1):
            matches: List[Any] = []
            cursor: int | None = None
            seen_cursors: set[int] = set()
            while True:
                page = call_zep_read_with_retry(
                    lambda: self.client.batch.list(limit=100, cursor=cursor),
                    operation_name=f"reconcile batch create {operation_id}",
                )
                for batch in getattr(page, "batches", None) or []:
                    metadata = getattr(batch, "metadata", None) or {}
                    if (
                        metadata.get("mirofish_operation_id") == operation_id
                        and metadata.get("graph_id") == graph_id
                    ):
                        matches.append(batch)
                next_cursor = getattr(page, "next_cursor", None)
                if next_cursor is None:
                    break
                if next_cursor == cursor or next_cursor in seen_cursors:
                    raise RuntimeError("Zep batch list cursor did not advance")
                seen_cursors.add(next_cursor)
                cursor = next_cursor

            if len(matches) > 1:
                raise RuntimeError(
                    f"Multiple Zep batches match operation {operation_id}; refusing ambiguity"
                )
            if matches:
                return matches[0]
            if attempt < max_attempts:
                time.sleep(attempt)
        return None

    # ---------------- Node / Edge ----------------

    def get_all_nodes(
        self,
        graph_id: str,
        *,
        page_size: int = 100,
        max_items: Optional[int] = None,
    ) -> List[GraphNode]:
        nodes = fetch_all_nodes(
            self.client, graph_id, page_size=page_size, max_items=max_items
        )
        return [self._to_node(node) for node in nodes]

    def get_all_edges(
        self,
        graph_id: str,
        *,
        page_size: int = 100,
        max_items: Optional[int] = None,
    ) -> List[GraphEdge]:
        edges = fetch_all_edges(
            self.client, graph_id, page_size=page_size, max_items=max_items
        )
        return [self._to_edge(edge) for edge in edges]

    def get_node(self, node_uuid: str) -> Optional[GraphNode]:
        try:
            node = call_zep_read_with_retry(
                lambda: self.client.graph.node.get(uuid_=node_uuid),
                operation_name=f"get node {node_uuid}",
            )
        except NotFoundError:
            return None
        if not node:
            return None
        return self._to_node(node)

    def get_node_edges(self, node_uuid: str) -> List[GraphEdge]:
        """取节点相关边。

        注意：Zep Cloud 3.25 的 ``graph.node.get_edges`` 实测只返回该节点作为
        source 的边。需要完整上下文（incoming + outgoing）时，调用方应传入
        graph_id 并对全图边做筛选（见 ``ZepEntityReader.get_node_edges``）。
        """

        edges = call_zep_read_with_retry(
            lambda: self.client.graph.node.get_edges(node_uuid=node_uuid),
            operation_name=f"get node edges {node_uuid}",
        )
        return [self._to_edge(edge) for edge in (edges or [])]

    # ---------------- Search ----------------

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
        reranker: str = "cross_encoder",
    ) -> SearchResult:
        from ..utils.zep import normalize_zep_search_limit, normalize_zep_search_query

        zep_query = normalize_zep_search_query(query)
        zep_limit = normalize_zep_search_limit(limit)

        raw = call_zep_read_with_retry(
            lambda: self.client.graph.search(
                graph_id=graph_id,
                query=zep_query,
                limit=zep_limit,
                scope=scope,
                reranker=reranker,
            ),
            operation_name=f"graph search ({graph_id})",
        )

        nodes = [self._to_node(node) for node in (getattr(raw, "nodes", None) or [])]
        edges = [self._to_edge(edge) for edge in (getattr(raw, "edges", None) or [])]
        return SearchResult(nodes=nodes, edges=edges)

    # ---------------- 生命周期 ----------------

    def close(self) -> None:
        """Zep Cloud 客户端是进程级共享的，不需要在此关闭。"""

    def backend_info(self) -> Dict[str, Any]:
        return {"backend": self.backend_name, "api_key_suffix": self._api_key[-4:]}
