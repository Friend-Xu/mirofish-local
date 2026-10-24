"""测试辅助：把仿 Zep SDK 客户端包成真实的适配器实例。

为什么需要：服务层现在只依赖 `ZepClientAdapter`（cloud / graphiti 双后端），
而这些测试要保护的是 **Zep Cloud SDK 层面的调用契约**（batch 三段式、
graph.node.get_edges、search 参数上限等）。把仿 SDK 包进真实的
`ZepCloudClient` 后，断言对象不变，接口层依然被覆盖。
"""

from types import SimpleNamespace


def cloud_store(sdk_client):
    """用真实 ZepCloudClient 包装一个仿 Zep SDK 客户端。"""

    from app.services.zep_cloud_impl import ZepCloudClient

    store = object.__new__(ZepCloudClient)
    store._api_key = "test-key"
    store.client = sdk_client
    return store


def fake_sdk_client(**kwargs):
    """`SimpleNamespace(**kwargs)` 的可直替版本：返回包装后的适配器。"""

    return cloud_store(SimpleNamespace(**kwargs))
