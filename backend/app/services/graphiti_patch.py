"""graphiti-core 批量写入的属性类型清洗补丁。

问题
----
图数据库（Neo4j / Kuzu / FalkorDB）的节点与边属性**只接受原始类型**
（str / int / float / bool / null）及其同构数组。启用自定义本体后，LLM 抽取
可能返回嵌套 dict 或含嵌套结构的 list，直接写库会整批 episode 失败。

做法
----
包装 `graphiti_core.utils.bulk_utils.add_nodes_and_edges_bulk_tx`：写库前遍历
entity nodes / edges，把 `attributes` 逐项 sanitize（dict → JSON 字符串；
含嵌套的 list → JSON 字符串；纯标量数组原样保留），再调用原函数。

幂等：`_patch_applied` 保证重复调用只生效一次；导入失败只告警不抛错。

来源说明：移植自 tt-a1i/MiroFish-local（AGPL-3.0）`backend/app/services/graphiti_patch.py`
（对应 graphiti-core Issue #683）。待上游修复后可整体删除本模块。
"""

from __future__ import annotations

import functools
import json
import os
from datetime import datetime
from typing import Any, Dict

from ..utils.logger import get_logger

logger = get_logger("mirofish.graphiti_patch")

_patch_applied = False


def sanitize_for_graph(value: Any, path: str = "") -> Any:
    """把任意值转换成图库可接受的属性类型。"""

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as error:
            logger.warning("无法序列化 dict 属性 %s: %s", path, error)
            return str(value)

    if isinstance(value, (list, tuple)):
        is_flat = all(
            isinstance(item, (str, int, float, bool, type(None))) for item in value
        )
        if is_flat:
            return list(value)
        try:
            return json.dumps(list(value), ensure_ascii=False, default=str)
        except (TypeError, ValueError) as error:
            logger.warning("无法序列化 list 属性 %s: %s", path, error)
            return str(value)

    return str(value)


def sanitize_attributes(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """清洗一整个属性字典。"""

    if not attrs:
        return {}
    return {
        key: sanitize_for_graph(value, path=str(key)) for key, value in attrs.items()
    }


def apply_driver_compat(driver: Any) -> None:
    """修正 KuzuDriver 缺少基类 `_database` 属性导致的崩溃。

    graphiti-core 0.30.2 的 `Graphiti._resolve_request_scope` 会读取
    `self.driver._database`，而 `KuzuDriver.__init__` 从未设置该属性（基类
    `GraphDriver` 声明了它）→ `add_episode` 直接 AttributeError。
    补一个与 `get_default_group_id(GraphProvider.KUZU)` 一致的默认库名即可：
    随后 `group_id == _database` 不成立，走 `driver.clone(database=group_id)`
    分支，而基类 `clone()` 返回 self —— 于是行为正是我们需要的"单库 +
    group_id 属性隔离"。
    """

    try:
        from graphiti_core.driver.kuzu_driver import KuzuDriver

        if isinstance(driver, KuzuDriver) and not hasattr(driver, "_database"):
            driver._database = ""
            logger.info("已为 KuzuDriver 补上 _database（graphiti-core 兼容）")
    except ImportError:
        pass


KUZU_FTS_INDEXES = (
    ("Entity", "node_name_and_summary", ("name", "summary")),
    ("RelatesToNode_", "edge_name_and_fact", ("name", "fact")),
)


class KuzuFulltextUnavailableError(RuntimeError):
    """Kuzu 全文索引不可用。

    检索只有一条路径（bm25 + cosine 混合），没有"降级为纯向量"的第二条路：
    缺索引时继续跑，只会让检索质量静默变化，用户无法察觉。
    """


def _result_rows(result: Any) -> list:
    """把 Kuzu 的 QueryResult 读成行列表（没有 next 的实现按空处理）。"""

    rows = []
    if hasattr(result, "has_next") and hasattr(result, "get_next"):
        while result.has_next():
            rows.append(list(result.get_next()))
    return rows


async def _existing_fts_indexes(client: Any) -> set:
    """库里已存在的 FTS 索引集合，形如 {("Entity", "node_name_and_summary"), ...}。

    用 `SHOW_INDEXES()` 查，而不是靠 "index already exists" 之类错误文本猜：
    错误文本是上游实现细节，猜错就会把"真的没建成"当成"已经有了"。
    """

    result = await client.execute(
        "CALL SHOW_INDEXES() RETURN table_name, index_name, index_type"
    )
    found = set()
    for row in _result_rows(result):
        table_name, index_name, index_type = row[0], row[1], row[2]
        if str(index_type).upper() == "FTS":
            found.add((str(table_name), str(index_name)))
    return found


async def rebuild_kuzu_fulltext_indexes_async(driver: Any) -> bool:
    """**重建** Kuzu 全文索引（不是"缺了才建"）；任何一步失败都抛 `KuzuFulltextUnavailableError`。

    为什么必须重建：Kuzu 0.11.3 的 bm25 索引在跨进程使用后会进入"维护即崩"状态——
    对**已存在**节点做 SET/MERGE 时，索引维护在 C++ 层触发 access violation，
    直接杀死整个后端进程（没有 Python 异常、没有 atexit，见 `掩盖台账.md` N-24）。

    离线二分证据（`.tools/_kuzu_repro.py`，对图谱副本操作）：
    * `MATCH (n:Entity {uuid: $uuid}) SET n.summary = $summary` → 崩（两个不同 uuid 都崩）；
    * 同一句，先 `DROP_FTS_INDEX` 掉两个索引 → 成功；
    * 同样的数据 `CREATE_FTS_INDEX` 重建后再执行 → 成功；
    * 全新库上"建索引 + 插入 + 更新" → 成功。
    结论：坏的是**库里那份索引**，重建即可恢复；上游 Kuzu 已停止维护、graphiti 已标记 deprecated。

    代价是启动时按图谱规模重建一次（本项目图谱是几十~几百节点量级）；
    收益是"回写不会打死后端"。graphiti-core 0.30.2 的
    `KuzuDriver.build_indices_and_constraints` 是空实现，`SCHEMA_QUERIES` 只建表不建索引，
    所以这两个索引本来就由这里负责。
    """

    client = getattr(driver, "client", None)
    if client is None:
        raise KuzuFulltextUnavailableError(
            "KuzuDriver 未暴露 client，无法重建全文索引"
        )

    failures: list[str] = []
    for statement in ("INSTALL fts", "LOAD EXTENSION fts"):
        try:
            await client.execute(statement)
        except Exception as error:  # noqa: BLE001 - 收集后统一抛，避免掩盖后续步骤的真实原因
            failures.append(f"{statement}: {error}")

    existing: set = set()
    try:
        existing = await _existing_fts_indexes(client)
    except Exception as error:  # noqa: BLE001
        failures.append(f"SHOW_INDEXES: {error}")

    for label, index_name, properties in KUZU_FTS_INDEXES:
        if (label, index_name) in existing:
            try:
                await client.execute(f"CALL DROP_FTS_INDEX('{label}', '{index_name}')")
            except Exception as error:  # noqa: BLE001
                failures.append(f"DROP {index_name}: {error}")
        property_list = ", ".join(f"'{name}'" for name in properties)
        cypher = f"CALL CREATE_FTS_INDEX('{label}', '{index_name}', [{property_list}])"
        try:
            await client.execute(cypher)
        except Exception as error:  # noqa: BLE001
            failures.append(f"{index_name}: {error}")

    if failures:
        raise KuzuFulltextUnavailableError(
            "Kuzu 全文索引重建失败（检索无第二条路径）：" + "; ".join(failures)
        )

    logger.info(
        "Kuzu 全文索引已重建（bm25 可用；旧索引会让写回以 access violation 崩进程，见 N-24）"
    )
    return True


def _schema_example(schema: Any, defs: Dict[str, Any] | None = None, depth: int = 0) -> Any:
    """从 JSON Schema 生成一个填充好的示例实例（用于提示词，避免模型回显 schema）。"""

    if depth > 6 or not isinstance(schema, dict):
        return None
    defs = defs or {}
    if "$ref" in schema:
        ref = str(schema["$ref"]).split("/")[-1]
        return _schema_example(defs.get(ref, {}), defs, depth + 1)
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema and isinstance(schema[key], list) and schema[key]:
            for candidate in schema[key]:
                if isinstance(candidate, dict) and candidate.get("type") != "null":
                    return _schema_example(candidate, defs, depth + 1)
            return _schema_example(schema[key][0], defs, depth + 1)

    schema_type = schema.get("type")
    if schema_type == "object" or "properties" in schema:
        return {
            name: _schema_example(value, defs, depth + 1)
            for name, value in (schema.get("properties") or {}).items()
        }
    if schema_type == "array":
        item_schema = schema.get("items") or {}
        item_example = _schema_example(item_schema, defs, depth + 1)
        # 数组给"一个元素"的示例，避免模型理解成"返回空数组"
        if isinstance(item_example, (dict, list)) and item_example:
            return [item_example]
        return []
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0.0
    if schema_type == "boolean":
        return False
    if schema_type == "string":
        return ""
    return None


def _looks_like_schema_echo(payload: Any, response_model: Any) -> bool:
    """判断返回体是不是"把 schema 定义抄回来了"而不是实例。"""

    if not isinstance(payload, dict) or response_model is None:
        return False
    expected = set(getattr(response_model, "model_fields", {}).keys())
    if expected and expected & set(payload.keys()):
        return False
    if "properties" in payload and isinstance(payload["properties"], dict):
        return True
    return bool(expected) and not (expected & set(payload.keys()))


# ── 节点类型标签的写边界守卫（掩盖台账 N-34）────────────────────────────

GENERIC_NODE_LABELS = {"Entity", "Node"}


class NodeLabelLedgerError(RuntimeError):
    """类型标签变更台账写不下去——守卫的证据链断了，不能假装只是日志问题。"""


def label_ledger_path() -> str:
    """类型标签变更台账（每次"要写空/要改写"都留一行）。

    调用时读 `Config.UPLOAD_FOLDER`，测试才能把它指到临时目录
    （N-42 的教训：运行目录一旦是导入期常量，测试就会写真实目录）。
    """

    from ..config import Config

    return os.path.join(Config.UPLOAD_FOLDER, "_diagnostics", "node-label-mutations.jsonl")


def _specific_labels(labels: Any) -> list:
    """只保留具体类型（丢掉 Entity/Node 这类通用标签）。"""

    return [
        str(label)
        for label in (labels or [])
        if label and str(label) not in GENERIC_NODE_LABELS
    ]


async def _labels_in_db(driver: Any, uuid: str) -> list:
    """库里该节点的 `labels` 属性（本项目唯一的类型来源）。"""

    records, _, _ = await driver.execute_query(
        "MATCH (n:Entity {uuid: $uuid}) RETURN n.labels AS labels", uuid=uuid
    )
    if not records:
        return []
    return list(records[0].get("labels") or [])


def _record_label_mutation(payload: Dict[str, Any]) -> None:
    """把一次"标签将要被改写"记进台账（append + flush，不吞写失败）。"""

    path = label_ledger_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
    except OSError as error:
        raise NodeLabelLedgerError(
            f"类型标签变更台账写不下去（{path}）：{error}"
        ) from error


async def retain_node_labels(driver: Any, nodes: Any) -> int:
    """写库前把"已存在节点的类型标签"并集回来，返回被修回的节点数。

    为什么必须有（`掩盖台账.md` N-34：同一张图谱的带类型实体从 **32 掉到 2**，
    节点总数一直是 53）：graphiti 写库时整列覆盖 `n.labels`——
    `bulk_utils.py:178` 算的是 `list(set(node.labels + ['Entity']))`，
    `models/nodes/node_db_queries.py` 用 `SET n.labels = $labels` 写回去。
    于是"抽到一个同名实体、但这次没给出具体类型"就会把图里原有的
    Engineer / Headhunter / CorporateManager 抹成只剩 Entity：节点还在、类型没了。
    graphiti 只在 dedupe 命中的分支合并标签（`dedup_helpers._promote_resolved_node`），
    真正落到写库这一层把类型抹掉时，**没有任何一层会说话**。

    规则（这是边界校验，不是兜底）：
    * 类型标签只会累积：库里有具体类型时，本次写入的最终标签 = 库中 ∪ 本次传入；
    * 本次传入**没有**具体类型（抽取没给类型）或传入了**不同**的具体类型，
      都记进台账 + ERROR 日志（节点 uuid/名字/图谱/两次的标签集合）——
      这正是"谁把 labels 写空"要定位的东西；
    * 库里没有这个节点（新节点）则什么都不做。

    为什么不是"直接失败"：`labels` 的语义本来就是单调累积的，库才是"这个节点是什么"
    的权威；本次抽取没给类型不代表要丢掉图里已知的类型。真正需要被看见的是
    "抽取质量退化了"，所以它进台账 + ERROR，而不是让整条 episode 写不进去。
    """

    provider = getattr(driver, "provider", None)
    if str(getattr(provider, "name", provider) or "").upper() != "KUZU":
        # 只有 Kuzu 把类型放在 `labels` 属性里（本项目唯一的本地引擎）
        return 0

    repaired = 0
    for node in nodes or []:
        uuid = getattr(node, "uuid", None)
        if not uuid:
            continue
        existing = _specific_labels(await _labels_in_db(driver, uuid))
        if not existing:
            continue

        incoming_labels = list(getattr(node, "labels", None) or [])
        incoming = _specific_labels(incoming_labels)
        if not incoming:
            reason = "incoming_has_no_specific_type"
        elif set(incoming) - set(existing):
            reason = "incoming_has_a_different_type"
        else:
            continue

        union = list(dict.fromkeys([*incoming_labels, *existing]))
        node.labels = union
        repaired += 1
        payload = {
            "at": datetime.now().isoformat(),
            "graph_id": getattr(node, "group_id", None),
            "node_uuid": uuid,
            "node_name": getattr(node, "name", None),
            "labels_in_db": existing,
            "labels_incoming": incoming,
            "labels_written": _specific_labels(union),
            "reason": reason,
        }
        logger.error(
            "类型标签只增不减：节点 %s（%s）库中类型 %s，本次传入 %s（%s）→ 写入 %s",
            payload["node_name"],
            uuid,
            existing,
            incoming or "（没有具体类型）",
            reason,
            payload["labels_written"],
        )
        _record_label_mutation(payload)

    return repaired


def apply_llm_prompt_patch() -> bool:
    """替换 json_object 模式下的结构化输出提示（DeepSeek 等兼容端点）。
    背景：graphiti 在 json_object 模式下把 `model_json_schema()` **原文**贴进提示词。
    对 DeepSeek 这类端点，提示词末尾就是一份 schema 定义，模型会直接把它抄回来
    （`{"properties": {...}}`），于是 Pydantic 校验报 "Field required"（实测：
    `EdgeDuplicate` 与每个自定义实体类型 `entity_type` 全部命中）。

    实测结论：只在报错后追加一句"请返回实例"**无效**——因为原实现每次都会再把
    schema 贴一遍，提示词末尾仍然是 schema。因此这里改为**整体接管** json_object
    分支：注入"填充好的示例骨架"而非 schema 原文，并保留一次回显兜底重试。

    `json_schema` 模式与 `response_model is None` 的情况原样委托给原实现。
    """

    try:
        from graphiti_core.llm_client.config import ModelSize
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        if getattr(OpenAIGenericClient, "_mirofish_prompt_patched", False):
            return True

        original = OpenAIGenericClient.generate_response

        def _example_hint(response_model: Any) -> str:
            try:
                schema = response_model.model_json_schema()
                example = _schema_example(schema, schema.get("$defs") or {})
                return json.dumps(example, ensure_ascii=False)
            except Exception:  # noqa: BLE001
                return "{}"

        @functools.wraps(original)
        async def patched_generate_response(
            self,
            messages,
            response_model=None,
            max_tokens=None,
            model_size=ModelSize.medium,
            group_id=None,
            prompt_name=None,
            *,
            attribute_extraction=False,
        ):
            if (
                response_model is None
                or getattr(self, "structured_output_mode", None) != "json_object"
            ):
                return await original(
                    self,
                    messages,
                    response_model,
                    max_tokens=max_tokens,
                    model_size=model_size,
                    group_id=group_id,
                    prompt_name=prompt_name,
                    attribute_extraction=attribute_extraction,
                )

            # ---- json_object 分支：用示例骨架取代 schema 原文 ----
            preamble = getattr(self, "_apply_attribute_extraction_preamble", None)
            if callable(preamble):
                try:
                    preamble(messages, attribute_extraction)
                except Exception:  # noqa: BLE001 - 前言注入失败不影响主流程
                    pass
            if max_tokens is None:
                max_tokens = getattr(self, "max_tokens", 8192)

            example = _example_hint(response_model)
            instruction = (
                "\n\nRespond with a single JSON object that has exactly this shape, "
                "with real content filled in (the values below are placeholders):\n\n"
                f"{example}\n\n"
                'Do NOT return the schema itself and do NOT wrap the object in "properties".'
            )
            try:
                messages[-1].content += instruction
            except Exception:  # noqa: BLE001 - 消息可能不可变
                pass

            payload = await self._generate_response(
                messages, response_model, max_tokens, model_size
            )
            if _looks_like_schema_echo(payload, response_model):
                logger.warning(
                    "结构化输出回显 schema（%s），追加显式提醒后重试一次",
                    getattr(response_model, "__name__", "?"),
                )
                try:
                    messages[-1].content += (
                        "\n\nYour previous answer was the schema definition, not an instance. "
                        f"Output ONLY a JSON object like this: {example}"
                    )
                except Exception:  # noqa: BLE001
                    pass
                payload = await self._generate_response(
                    messages, response_model, max_tokens, model_size
                )
                if _looks_like_schema_echo(payload, response_model):
                    logger.error(
                        "结构化输出仍然回显 schema（%s）",
                        getattr(response_model, "__name__", "?"),
                    )
            return payload

        OpenAIGenericClient.generate_response = patched_generate_response
        OpenAIGenericClient._mirofish_prompt_patched = True
        logger.info("Graphiti LLM 结构化输出补丁已应用（示例骨架注入）")
        return True
    except ImportError as error:
        logger.warning("无法应用 Graphiti LLM 补丁: %s", error)
        return False
    except Exception as error:  # noqa: BLE001
        logger.error("应用 Graphiti LLM 补丁失败: %s", error)
        return False


def apply_patch() -> bool:
    """把补丁应用到 graphiti-core。返回是否可用。"""

    global _patch_applied

    if _patch_applied:
        logger.debug("Graphiti 属性补丁已应用，跳过")
        return True

    try:
        from graphiti_core.nodes import EntityNode
        from graphiti_core.utils import bulk_utils

        original = bulk_utils.add_nodes_and_edges_bulk_tx

        @functools.wraps(original)
        async def patched_add_nodes_and_edges_bulk_tx(
            tx,
            episodic_nodes,
            episodic_edges,
            entity_nodes,
            entity_edges,
            embedder,
            driver,
        ):
            # 与 graphiti-core 0.30.x 的函数签名保持一致
            # 类型标签的写边界守卫（掩盖台账 N-34）：先并集、再写库
            await retain_node_labels(driver, entity_nodes)
            for node in entity_nodes or []:
                if getattr(node, "attributes", None):
                    node.attributes = sanitize_attributes(node.attributes)
            for edge in entity_edges or []:
                if getattr(edge, "attributes", None):
                    edge.attributes = sanitize_attributes(edge.attributes)
            return await original(
                tx,
                episodic_nodes,
                episodic_edges,
                entity_nodes,
                entity_edges,
                embedder,
                driver,
            )

        bulk_utils.add_nodes_and_edges_bulk_tx = patched_add_nodes_and_edges_bulk_tx

        # 单节点保存同样会整列覆盖 `n.labels`，用同一个守卫（一处能力一条路径）
        original_save = EntityNode.save

        @functools.wraps(original_save)
        async def patched_entity_node_save(self, driver):
            await retain_node_labels(driver, [self])
            return await original_save(self, driver)

        EntityNode.save = patched_entity_node_save
        _patch_applied = True
        logger.info("Graphiti bulk_utils 属性清洗补丁已应用（含类型标签守卫）")
        return True

    except ImportError as error:
        logger.warning("无法导入 graphiti_core.utils.bulk_utils: %s", error)
        return False
    except Exception as error:  # noqa: BLE001
        logger.error("应用 Graphiti 补丁失败: %s", error)
        return False
