"""图存储后端工厂：按 ZEP_BACKEND 选择 Zep Cloud 或本地 Graphiti。

设计说明
--------
* 具体实现只在函数体内 import，未安装 graphiti-core 时 cloud 模式不受影响。
* 进程级单例通过 lru_cache 复用（Neo4j/FalkorDB/Kuzu 驱动与 HTTP 连接池都需要复用）。
* 这里不重复实现 Zep SDK 客户端；那是 `utils/zep.py` 的职责。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from ..config import Config
from ..utils.logger import get_logger
from .zep_adapter import GraphStoreUnavailableError, ZepClientAdapter

logger = get_logger("mirofish.zep_factory")


def resolve_backend(backend: Optional[str] = None) -> str:
    """返回规范化后的后端名，并对非法取值报错。"""

    value = (backend or Config.ZEP_BACKEND or "cloud").strip().lower()
    if value not in {"cloud", "graphiti"}:
        raise GraphStoreUnavailableError(
            f"ZEP_BACKEND 取值非法：{value}（应为 cloud 或 graphiti）"
        )
    return value


def create_graph_store(
    backend: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ZepClientAdapter:
    """构建新的图存储客户端（测试与需要独立实例时使用）。"""

    resolved = resolve_backend(backend)
    if resolved == "graphiti":
        return _create_graphiti_client()
    return _create_cloud_client(api_key)


def _create_cloud_client(api_key: Optional[str] = None) -> ZepClientAdapter:
    from .zep_cloud_impl import ZepCloudClient

    logger.info("创建 Zep Cloud 图存储客户端")
    return ZepCloudClient(api_key=api_key)


def _create_graphiti_client() -> ZepClientAdapter:
    from .zep_graphiti_impl import GraphitiClient

    logger.info("创建本地 Graphiti 图存储客户端: %s", Config.GRAPHITI_DB_PATH)
    return GraphitiClient()


@lru_cache(maxsize=4)
def _cached_graph_store(backend: str) -> ZepClientAdapter:
    return create_graph_store(backend=backend)


def get_graph_store(api_key: Optional[str] = None) -> ZepClientAdapter:
    """返回进程级共享的图存储客户端。"""

    backend = resolve_backend()
    if backend == "cloud" and api_key:
        # 显式传入的 key 不进入缓存键，避免把密钥留在缓存里；直接新建实例。
        return _create_cloud_client(api_key)
    return _cached_graph_store(backend)


def reset_graph_store() -> None:
    """清空缓存（测试与受控重配置用）。"""

    _cached_graph_store.cache_clear()
