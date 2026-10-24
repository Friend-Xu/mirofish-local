"""embedding 加载契约测试：模型只构造一次，并发不竞态。

背景（L-a 现场实测）：`sentence-transformers 3.0.0` + `torch 2.9.1+cpu` 下
4 个线程并发 `SentenceTransformer(path)`，2/4 抛
`NotImplementedError: Cannot copy out of meta tensor`。
人设生成按实体并行，每个线程都会碰 embedder——旧代码把这个失败吞成"备用人设"，
所以问题一直不可见（见 `掩盖台账.md` N-16）。
"""

import threading
import time

import pytest

from app.utils import local_embedder
from app.utils.local_embedder import (
    EmbeddingModelMissingError,
    LocalSentenceTransformerEmbedder,
    _shared_model,
)


def test_missing_model_dir_raises_with_provisioning_hint(tmp_path):
    with pytest.raises(EmbeddingModelMissingError) as error:
        LocalSentenceTransformerEmbedder(str(tmp_path / "not-here"))
    message = str(error.value)
    assert "不存在" in message
    # F18：模型是项目自有目录，供应入口是 provision-models.py（不再教人 snapshot_download）
    assert "provision-models.py" in message, "错误里要给出准备命令"
    assert "bge-small-zh-v1.5" in message


def test_empty_model_dir_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(EmbeddingModelMissingError, match="为空"):
        LocalSentenceTransformerEmbedder(str(tmp_path / "empty"))


class _SlowModel:
    """替身：记录构造次数，构造期间故意慢，放大并发竞态窗口。"""

    constructions = 0

    def __init__(self, path, device=None, local_files_only=False):  # noqa: ANN001
        _SlowModel.constructions += 1
        time.sleep(0.15)
        self.path = path

    def get_sentence_embedding_dimension(self):  # noqa: ANN201
        return 512


def test_model_is_constructed_exactly_once_across_threads(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(local_embedder, "_MODEL_CACHE", {})
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _SlowModel)
    _SlowModel.constructions = 0

    results = []

    def worker():
        results.append(_shared_model(str(model_dir), "cpu"))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert _SlowModel.constructions == 1, "模型必须只构造一次（并发构造会拿到 meta 张量）"
    assert len({id(model) for model in results}) == 1, "所有线程必须复用同一个实例"
