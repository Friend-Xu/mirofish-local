"""本地 embedding 实现：包装 sentence-transformers，供 Graphiti 使用。

为什么需要它：DeepSeek / 大多数国产 LLM 网关不提供 embedding 接口，而 Graphiti
的语义检索必须有 embedder。本项目已随 camel-oasis 安装了
``sentence-transformers``，因此直接加载本地模型（默认 BAAI/bge-small-zh-v1.5，
512 维）即可，无需任何额外服务与联网。

离线优先：模型目录由 ``GRAPHITI_EMBEDDING_MODEL_PATH`` 指定。加载前先确认目录存在
（缺失即失败并打印 provisioning 命令），并且**强制 `local_files_only=True`**——
否则句柄会被当成 HuggingFace 模型 id 去联网拉取，把"模型没准备好"变成"网络很慢"。
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any, Iterable, List, Optional

from .logger import get_logger

logger = get_logger("mirofish.local_embedder")

# 进程内共享的模型实例：sentence-transformers 3.0.0 + torch 2.9.1 下**并发构造**
# 同一个模型是不安全的——`from_pretrained` 的惰性(meta)初始化会竞态，部分实例
# 拿到 meta 张量后 `.to(device)` 直接抛 NotImplementedError。
# 实测（隔离测试）：单线程 2/2 成功；4 线程并发构造 2/4 失败。
# 人设生成按实体并行跑，每个线程都会碰 embedder，所以这里必须"只构造一次"。
_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_MODEL_LOCK = threading.Lock()


def _shared_model(model_path: str, device: str) -> Any:
    """按 (路径, 设备) 复用同一个模型实例；构造过程用锁串行化。"""

    key = (model_path, device)
    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("加载本地 embedding 模型: %s (device=%s)", model_path, device)
            # local_files_only=True：只从项目内目录加载，绝不联网
            model = SentenceTransformer(
                model_path, device=device, local_files_only=True
            )
            _MODEL_CACHE[key] = model
            logger.info(
                "embedding 模型就绪: path=%s dim=%s",
                model_path,
                model.get_sentence_embedding_dimension(),
            )
        return model


class EmbeddingModelMissingError(RuntimeError):
    """本地 embedding 模型目录不存在。

    历史问题：`SentenceTransformer(path)` 在 path 不是目录时会把字符串当模型 id
    去联网下载，于是"没准备好模型"表现为超时/网络错误（见掩盖台账 M-09 / N-12）。
    """


def _require_model_dir(model_path: str) -> str:
    """确认模型目录存在（含权重文件），否则抛出带 provisioning 命令的错误。"""

    resolved = os.path.abspath(model_path)
    if not os.path.isdir(resolved):
        raise EmbeddingModelMissingError(
            f"本地 embedding 模型目录不存在: {resolved}（配置 GRAPHITI_EMBEDDING_MODEL_PATH）\n"
            "准备步骤（在项目目录内执行，模型留在项目里）：\n"
            "  backend\\.venv\\Scripts\\python.exe scripts\\provision-models.py "
            "--fetch bge-small-zh-v1.5\n"
            "  backend\\.venv\\Scripts\\python.exe scripts\\provision-models.py --verify\n"
            "（默认走 hf-mirror：HF 直连在本机不通，这是已知前提）\n"
            "详见 docs/DEPLOYMENT.md §4「模型与推理引擎」。"
        )
    if not os.listdir(resolved):
        raise EmbeddingModelMissingError(f"本地 embedding 模型目录为空: {resolved}")
    return resolved


class LocalSentenceTransformerEmbedder:
    """同步的 sentence-transformers 封装成 Graphiti 需要的异步接口。

    实现的是 ``graphiti_core.embedder.client.EmbedderClient`` 协议
    （``async create`` / ``async create_batch``）；不直接继承是为了避免
    未安装 graphiti-core 时导入本模块即失败。
    """

    def __init__(self, model_path: str, *, device: str = "cpu", batch_size: int = 32):
        self.model_path = _require_model_dir(model_path)
        self.device = device
        self.batch_size = batch_size
        self._model: Any = None
        self._dim: Optional[int] = None

    # ---------------- 内部 ----------------

    def _load(self) -> Any:
        if self._model is None:
            self._model = _shared_model(self.model_path, self.device)
            self._dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def _encode_sync(self, texts: List[str]) -> List[List[float]]:
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [list(map(float, vector)) for vector in vectors]

    # ---------------- Graphiti 协议 ----------------

    async def create(self, input_data: str | Iterable[str] | Any) -> List[float]:
        if isinstance(input_data, str):
            texts = [input_data]
        else:
            texts = [str(item) for item in input_data]
        vectors = await asyncio.to_thread(self._encode_sync, texts)
        return vectors[0]

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        if not input_data_list:
            return []
        return await asyncio.to_thread(self._encode_sync, list(input_data_list))

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            self._load()
        return int(self._dim or 0)


def build_embedder(model_path: str, *, dim: Optional[int] = None) -> Any:
    """返回一个满足 Graphiti EmbedderClient 协议的 embedder。"""

    embedder = LocalSentenceTransformerEmbedder(model_path)
    actual = embedder.embedding_dim
    if dim is not None and actual and actual != dim:
        logger.warning(
            "GRAPHITI_EMBEDDING_DIM=%s 与模型实际维度 %s 不一致，以模型为准", dim, actual
        )
    return embedder


def graphiti_embedder_client(model_path: str) -> Any:
    """返回真正继承自 graphiti EmbedderClient 的实例（供类型校验严格的路径使用）。"""

    from graphiti_core.embedder.client import EmbedderClient

    base = LocalSentenceTransformerEmbedder(model_path)

    class _Embedder(EmbedderClient):  # type: ignore[misc]
        async def create(self, input_data):  # noqa: ANN001
            return await base.create(input_data)

        async def create_batch(self, input_data_list):  # noqa: ANN001
            return await base.create_batch(input_data_list)

    return _Embedder()
