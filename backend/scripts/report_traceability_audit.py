"""报告可追溯抽检（R1/R2）：把"结论能不能追到图谱事实"变成可核对的工作表。

为什么要有（`模拟质量验收标准.md` §2）：
R1（结论可追溯率）与 R2（无依据内容）此前从未测过——基线只覆盖模拟段，
"报告结论能不能追到模拟事实"没有任何数字。这两项必须**人工判定**，
所以本工具的职责是把候选证据摆到眼前，而不是替人下结论：

* 每条抽样陈述给出：命中的图谱实体（含类型）、图谱里能作为支撑的事实
  （边 uuid / 事实原文 / 源实体→目标实体）、以及可疑的专名与数字；
* 判定列留空由人填写（`supported` / `partial` / `unsupported`），
  以及"是否与范围声明矛盾"（例如拿没有传播的 twitter 讲传播）；
* `--summarize` 只在判定填齐后算数字；未判定的条目计为"未测"，不算通过。

**前置：后端必须停止**（`掩盖台账.md` M-11）。图谱是嵌入式 Kuzu 库，
两个进程同时打开同一个库是它已知的风险（N-24/N-34），因此本工具启动前
先探测 :5001，后端还在就直接退出。

跑法（项目自己的解释器；**必须显式指定本地图谱后端**）：
    cd backend && ZEP_BACKEND=graphiti .venv/Scripts/python.exe \
        scripts/report_traceability_audit.py --report report_xxxx --per-section 5 --seed 20261011

为什么必须显式指定 `ZEP_BACKEND`（`掩盖台账.md` V-10）：`.env` 里没有这一项，
`Config` 默认 `cloud`。不指定就会拿着本地 graph_id 去查 Zep Cloud（实测 404
`not found`），而工具以为自己在读本地图谱——同一类"第二个引擎"的失败。
本工具启动即校验后端，不是本地图谱就直接退出。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import socket
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.abspath(BACKEND_DIR))

from app.services.report_grounding import (  # noqa: E402
    FACT_LABEL,
    INFERENCE_LABEL,
    INFERENCE_PREFIX,
)

REPORTS_DIR = os.path.join(os.path.abspath(BACKEND_DIR), "uploads", "reports")

# 机械可疑项：都只是**代理指标**（要证明"报告里有图谱里不存在的公司/数字"必须人看）
PATTERNS = {
    "latin": re.compile(r"[A-Za-z][A-Za-z0-9\-\.]{2,}"),
    "number_with_unit": re.compile(
        r"\d+(?:\.\d+)?\s*(?:%|％|mm|μm|um|nm|ms|s|秒|分钟|小时|天|周|个月|年|"
        r"条|人|倍|万|亿|ppm|MTTR|MCBF|MTBF|RORZE)"
    ),
    "org_like": re.compile(r"[\u4e00-\u9fff]{2,12}(?:公司|厂|集团|机构|平台|大学|研究院|实验室)"),
    "quoted": re.compile(r"[\"“”「」『』]([^\"“”「」『』]{4,})[\"“”「」『』]"),
    "date_like": re.compile(r"\d{4}\s*[年./-]\s*\d{1,2}"),
}

# 抽样切句时丢弃的碎片（标题、纯标点、过短的行）
MIN_CLAIM_CHARS = 15


def _backend_listening(host: str = "127.0.0.1", port: int = 5001) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^\s*(?:[-*+]\s+|\d+[.、)]\s*)", "", line)  # 列表项符号
    line = re.sub(r"^\s*>\s*", "", line)                      # 引用
    line = line.replace("**", "").replace("__", "")
    line = re.sub(r"`([^`]*)`", r"\1", line)                   # 行内代码
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)       # 链接
    return line.strip()


def split_claims(section_text: str):
    """把一章切成"结论性陈述"候选（确定性、可复现）。

    返回 `[{"text": ..., "claim_kind": "fact"|"inference"|"unlabeled"}]`。
    N-44 之后的新报告用 `**模拟事实（可溯源）**` / `**作者推断（非模拟事实）**`
    两个块标签；老报告没有标签，全部记 `unlabeled`（R1_all 仍可算，R1_fact 不适用）。

    标签必须在 `_clean_line` 去掉 `**` 之前识别，否则块信息会丢。
    """

    records = []
    claim_kind = "unlabeled"
    in_code = False
    for raw_line in section_text.splitlines():
        stripped = raw_line.strip()
        if stripped == FACT_LABEL:
            claim_kind = "fact"
            continue
        if stripped == INFERENCE_LABEL:
            claim_kind = "inference"
            continue
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        # 粗体小标题不是“结论性陈述”（如 `**一、猎头的实际落点动作**`）；
        # 但它不能吃掉 `**[推断] ...**` 这种带标签的推断行。
        if (
            stripped.startswith("**")
            and stripped.endswith("**")
            and not stripped.lstrip("* ").startswith(INFERENCE_PREFIX)
        ):
            continue
        line = _clean_line(raw_line)
        if not line or line.startswith("#"):
            continue
        if set(line) <= set("|-: "):  # 表格分隔行
            continue
        if line.rstrip().endswith(("：", ":")):
            # 以冒号结尾的是引导句（“三条如下：”），不是结论性陈述。
            continue
        for piece in re.split(r"(?<=[。！？；])", line):
            piece = piece.strip()
            if len(piece) < MIN_CLAIM_CHARS:
                continue
            if not re.search(r"[\u4e00-\u9fff]{6,}", piece):
                continue
            kind = claim_kind
            if piece.lstrip("* ").startswith(INFERENCE_PREFIX):
                kind = "inference"
            records.append({"text": piece, "claim_kind": kind})
    return records


def _mentions(claim: str, entities):
    """命中的图谱实体：最长匹配优先，避免"某工程师"吃掉"资深某工程师"。"""

    hits = []
    for entity in entities:
        name = entity["name"]
        if len(name) < 2 or name not in claim:
            continue
        hits.append((name, entity))
    hits.sort(key=lambda item: -len(item[0]))
    kept = []
    for name, entity in hits:
        if any(name in kept_name for kept_name, _ in kept):
            continue
        kept.append((name, entity))
    return [
        {
            "name": name,
            "type": next((l for l in entity["labels"] if l not in ("Entity", "Node")), ""),
            "uuid": entity["uuid"],
        }
        for name, _ in kept
    ]


def _suspicious(claim: str):
    found = {}
    for kind, pattern in PATTERNS.items():
        values = []
        for match in pattern.finditer(claim):
            value = match.group(1) if kind == "quoted" else match.group(0)
            value = value.strip()
            if value and value not in values:
                values.append(value)
        if values:
            found[kind] = values
    return found


def _quote_key(text: str) -> str:
    """把引语/摘要压成可比较的键：去空白、引号与常见标点。

    只用于“节点摘要里是否出现同一句引语”的确定性匹配；不改变原文事实。
    """

    return re.sub(r"[\s\"'“”‘’「」『』，。；：、！？,.;:!?()（）。—…\-]+", "", text or "")


def _quoted_spans(claim: str):
    """claim 里的引号内容（>=8 个有效字符），用于找包含同一引语的节点摘要。"""

    spans = []
    for value in _suspicious(claim).get("quoted", []):
        key = _quote_key(value)
        if len(key) >= 8:
            spans.append(key)
    return spans


def _support(
    service,
    graph_id: str,
    claim: str,
    name_by_uuid,
    entities,
    mentioned_entities=(),
    limit: int = 5,
):
    """候选支撑：边 + 节点摘要。

    为什么必须两种都取（台账 N-50）：`search_graph(scope="edges")` 只返回边，
    R1 的判定口径却允许"边事实或节点摘要"。只搜边时，只存在于节点摘要里的引语
    （第二轮 `S5-5`）在工作表上显示 0 条候选，R1 被低估。

    节点候选有两个来源：
    1. `scope="nodes"` 的语义检索；
    2. claim 命中的实体（`entity_mentions`）的节点摘要——语义检索不一定召回它，
       但实体名命中是确定的，且 `S5-5` 正是这种情况。
    """

    support = []
    seen = set()

    def _add(kind, uuid, fact, source, target, name):
        key = (kind, uuid or fact)
        if not fact or key in seen:
            return
        seen.add(key)
        support.append(
            {
                "kind": kind,
                "uuid": uuid,
                "fact": fact,
                "source": source,
                "target": target,
                "name": name,
            }
        )

    edge_result = service.search_graph(
        graph_id=graph_id, query=claim[:120], limit=limit, scope="edges"
    )
    for edge in edge_result.edges:
        _add(
            "edge",
            edge.get("uuid"),
            edge.get("fact"),
            name_by_uuid.get(edge.get("source_node_uuid"), ""),
            name_by_uuid.get(edge.get("target_node_uuid"), ""),
            edge.get("name"),
        )

    node_result = service.search_graph(
        graph_id=graph_id, query=claim[:120], limit=limit, scope="nodes"
    )
    for node in node_result.nodes:
        _add(
            "node",
            node.get("uuid"),
            node.get("summary"),
            node.get("name"),
            "",
            node.get("labels"),
        )

    entity_by_uuid = {entity["uuid"]: entity for entity in entities}
    for mention in mentioned_entities:
        entity = entity_by_uuid.get(mention.get("uuid"))
        if not entity:
            continue
        _add(
            "node",
            entity["uuid"],
            entity.get("summary"),
            entity.get("name"),
            "",
            entity.get("labels"),
        )

    # 引语命中：claim 里的引号句子可能只出现在某个节点摘要里，而该实体名没有
    # 出现在 claim 中（S5-5：HR 说「…支援型工程师…翻盘成本翻倍」只在
    # `HR / 招聘负责人` 摘要里，claim 只写了 "HR"）。语义节点检索不保证召回它，
    # 这里按“同一句引语”做确定性补充；不命中就不补。
    quote_keys = _quoted_spans(claim)
    if quote_keys:
        for entity in entities:
            summary_key = _quote_key(entity.get("summary"))
            if not summary_key:
                continue
            if any(key in summary_key for key in quote_keys):
                _add(
                    "node",
                    entity["uuid"],
                    entity.get("summary"),
                    entity.get("name"),
                    "",
                    entity.get("labels"),
                )
    return support


def _markdown(worksheet) -> str:
    lines = [
        f"# 报告可追溯抽检工作表（{worksheet['report_id']}）",
        "",
        f"- 图谱：`{worksheet['graph_id']}`  模拟：`{worksheet['simulation_id']}`",
        f"- 抽样：每章 {worksheet['per_section']} 条，种子 {worksheet['seed']}",
        f"- 生成时间：{worksheet['generated_at']}",
        f"- 范围声明（R3）：{'有' if worksheet.get('simulation_scope') else '**没有**（该报告不满足 R3）'}",
        "",
        "> 判定列由人填写：`supported` / `partial` / `unsupported`；",
        "> `conflict_with_scope` 填 `true/false`（该条是否与范围声明矛盾，例如拿无传播平台讲传播）。",
        "> `support` 是检索到的候选证据，不是判定结论。",
        "",
        "---",
        "",
    ]
    for item in worksheet["claims"]:
        lines.append(f"## 第 {item['section_index']} 章 · 抽样 {item['claim_index']}")
        lines.append("")
        lines.append(
            f"**陈述（{item.get('claim_kind', 'unlabeled')}）**：{item['text']}"
        )
        lines.append("")
        lines.append(
            "- 命中实体："
            + (
                "、".join(f"{m['name']}（{m['type']}）" for m in item["entity_mentions"])
                or "无"
            )
        )
        if item["suspicious"]:
            lines.append(
                "- 可疑项（代理）："
                + "；".join(
                    f"{kind}: {'、'.join(values)}"
                    for kind, values in item["suspicious"].items()
                )
            )
        lines.append(f"- 候选支撑（{len(item['support'])} 条）：")
        for support in item["support"][:5]:
            route = (
                f"{support['source']} → {support['target']}"
                if support.get("target")
                else f"{support['source']}（节点摘要）"
            )
            lines.append(
                f"    - [{support['kind']}] {route}："
                f"{(support['fact'] or '')[:160]}"
            )
        lines.append("")
        lines.append(
            f"- verdict: `{item['verdict']}`   conflict_with_scope: `{item['conflict_with_scope']}`"
        )
        lines.append("")
    return "\n".join(lines)


def apply_verdicts(worksheet, verdicts_path: str) -> int:
    """把人工判定合并进工作表。

    判定文件的格式（键 = `S<章>-<抽样序号>`）：
        {"S1-1": {"verdict": "supported", "note": "…", "conflict_with_scope": false}}
    只接受 `supported` / `partial` / `unsupported`；缺字段即报错，不给默认判定。
    """

    with open(verdicts_path, "r", encoding="utf-8") as handle:
        verdicts = json.load(handle)

    allowed = {"supported", "partial", "unsupported"}
    applied = 0
    known = set()
    for claim in worksheet["claims"]:
        key = f"S{claim['section_index']}-{claim['claim_index']}"
        known.add(key)
        entry = verdicts.get(key)
        if entry is None:
            # 空判定是显式的：不许用默认值把"没看"伪装成"已判定"
            continue
        verdict = entry.get("verdict")
        if verdict not in allowed:
            raise SystemExit(f"{key} 的判定非法：{verdict!r}（只接受 {sorted(allowed)}）")
        claim["verdict"] = verdict
        claim["verdict_note"] = entry.get("note", "")
        claim["conflict_with_scope"] = entry.get("conflict_with_scope", False)
        claim["confirmed_unsupported"] = entry.get("confirmed_unsupported", [])
        applied += 1

    unknown = sorted(set(verdicts) - known)
    if unknown:
        raise SystemExit(f"判定文件里有工作表里不存在的条目：{unknown}")
    return applied


def _quote_violations(worksheet) -> int:
    grounding = worksheet.get("grounding") or {}
    count = 0
    for section in grounding.get("sections") or []:
        for item in section.get("final_violations") or []:
            if item.get("code") in {"quote_not_verbatim", "paraphrase_in_quotes"}:
                count += 1
    return count


def summarize(worksheet) -> int:
    claims = worksheet["claims"]
    filled = [c for c in claims if c["verdict"]]
    counts = {"supported": 0, "partial": 0, "unsupported": 0}
    for claim in filled:
        if claim["verdict"] in counts:
            counts[claim["verdict"]] += 1
    unresolved = len(claims) - len(filled)
    total_judged = len(filled)
    rate = counts["supported"] / total_judged if total_judged else None

    fact = [c for c in filled if c.get("claim_kind") == "fact"]
    fact_supported = sum(1 for c in fact if c["verdict"] == "supported")
    fact_rate = fact_supported / len(fact) if fact else None
    unlabeled = sum(1 for c in claims if c.get("claim_kind") == "unlabeled")

    print(f"抽样条数            : {len(claims)}（已判定 {total_judged}，未判定 {unresolved}）")
    print(f"supported/partial/unsupported: {counts['supported']}/{counts['partial']}/{counts['unsupported']}")
    print(f"R1_all（全部条目）  : " + (f"{rate:.1%}" if rate is not None else "未测（判定未填）"))
    if fact:
        print(
            f"R1_fact（事实块）   : {fact_rate:.1%}"
            f"（{fact_supported}/{len(fact)}）"
        )
    else:
        print("R1_fact（事实块）   : 未测（该报告没有事实块标签）")
    print(f"unlabeled 条数      : {unlabeled}（目标 0）")
    print(f"quote_violations    : {_quote_violations(worksheet)} 条（目标 0；老报告未测）")
    if unresolved:
        print("  ⚠️ 还有未判定条目：R1 只能算「已判定部分」，不能当整份报告的结论")
    conflicts = [c for c in claims if c["conflict_with_scope"] is True]
    print(f"与范围声明矛盾      : {len(conflicts)} 条")
    return 0 if unresolved == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="report_id")
    parser.add_argument("--per-section", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--graph", default=None, help="默认取报告的 graph_id")
    parser.add_argument("--out", default=None, help="工作表 JSON 路径")
    parser.add_argument(
        "--apply-verdicts", default=None,
        help="把人工判定 JSON 合并进工作表（键=S<章>-<序号>）",
    )
    parser.add_argument("--summarize", action="store_true", help="按已填判定算 R1/R2")
    args = parser.parse_args()

    report_dir = os.path.join(REPORTS_DIR, args.report)
    out_json = args.out or os.path.join(report_dir, "traceability_audit.json")

    if args.apply_verdicts:
        if not os.path.exists(out_json):
            print(f"找不到工作表：{out_json}")
            return 1
        worksheet = _load_json(out_json)
        applied = apply_verdicts(worksheet, args.apply_verdicts)
        with open(out_json, "w", encoding="utf-8") as handle:
            json.dump(worksheet, handle, ensure_ascii=False, indent=2)
        with open(os.path.splitext(out_json)[0] + ".md", "w", encoding="utf-8") as handle:
            handle.write(_markdown(worksheet))
        print(f"已合并 {applied} 条判定：{out_json}")
        return summarize(worksheet)

    if args.summarize:
        if not os.path.exists(out_json):
            print(f"找不到工作表：{out_json}")
            return 1
        return summarize(_load_json(out_json))

    if _backend_listening():
        print(
            "后端还在 :5001 上跑着：本工具要独占嵌入式 Kuzu 库"
            "（两个进程同时打开同一个库是 N-24/N-34 的已知风险）。\n"
            "先停后端再跑（台账 M-11）。"
        )
        return 1

    if not os.path.isdir(report_dir):
        print(f"报告目录不存在：{report_dir}")
        return 1

    meta_path = os.path.join(report_dir, "meta.json")
    if not os.path.exists(meta_path):
        print(f"报告元信息不存在：{meta_path}")
        return 1
    meta = _load_json(meta_path)
    graph_id = args.graph or meta.get("graph_id")
    if not graph_id:
        print("报告里没有 graph_id，且未用 --graph 指定")
        return 1

    from app.config import Config  # noqa: E402
    from app.services.zep_tools import ZepToolsService  # noqa: E402

    if Config.ZEP_BACKEND != "graphiti":
        print(
            f"当前图谱后端是 {Config.ZEP_BACKEND!r}，不是本地图谱（graphiti）："
            "本工具只对本地图谱有意义，去查云端会用同一个 graph_id 得到 404，"
            "而你可能以为读的是本地图谱。\n"
            "请用 ZEP_BACKEND=graphiti 重新运行（与启动器的设置一致）。"
        )
        return 1

    service = ZepToolsService()
    nodes = service.get_all_nodes(graph_id)
    entities = [
        {
            "uuid": node.uuid,
            "name": node.name,
            "labels": list(node.labels or []),
            "summary": node.summary,
        }
        for node in nodes
        if node.name
    ]
    name_by_uuid = {entity["uuid"]: entity["name"] for entity in entities}
    print(f"图谱实体：{len(entities)} 个（{graph_id}）")

    section_paths = sorted(glob.glob(os.path.join(report_dir, "section_*.md")))
    if not section_paths:
        print(f"报告目录里没有 section_*.md：{report_dir}")
        return 1

    claims = []
    for path in section_paths:
        index = int(os.path.basename(path).replace("section_", "").replace(".md", ""))
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        pool = split_claims(text)
        rng = random.Random(f"{args.seed}:{index}")
        sample = rng.sample(pool, min(args.per_section, len(pool)))
        print(f"  第 {index} 章：候选 {len(pool)} 条，抽 {len(sample)} 条")
        for claim_index, claim in enumerate(sample, 1):
            text = claim["text"]
            mentions = _mentions(text, entities)
            claims.append(
                {
                    "section_index": index,
                    "claim_index": claim_index,
                    "text": text,
                    "claim_kind": claim["claim_kind"],
                    "entity_mentions": mentions,
                    "suspicious": _suspicious(text),
                    "support": _support(
                        service,
                        graph_id,
                        text,
                        name_by_uuid,
                        entities,
                        mentioned_entities=mentions,
                    ),
                    "verdict": None,
                    "verdict_note": "",
                    "conflict_with_scope": None,
                    "confirmed_unsupported": [],
                }
            )

    worksheet = {
        "report_id": args.report,
        "simulation_id": meta.get("simulation_id"),
        "graph_id": graph_id,
        "report_status": meta.get("status"),
        "simulation_scope": meta.get("simulation_scope"),
        "grounding": meta.get("grounding"),
        "per_section": args.per_section,
        "seed": args.seed,
        "graph_entities": len(entities),
        "generated_at": datetime.now().isoformat(),
        "claims": claims,
    }
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(worksheet, handle, ensure_ascii=False, indent=2)
    md_path = os.path.splitext(out_json)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(_markdown(worksheet))

    with_support = sum(1 for claim in claims if claim["support"])
    print()
    print(f"工作表：{out_json}")
    print(f"         {md_path}")
    print(f"抽样总数：{len(claims)} 条")
    print(f"  命中图谱实体：{sum(1 for c in claims if c['entity_mentions'])} 条")
    print(f"  有候选支撑  ：{with_support} 条   ← **代理指标**，不是 R1")
    print(f"  带可疑项    ：{sum(1 for c in claims if c['suspicious'])} 条   ← **代理指标**，R2 要人确认")
    print("下一步：人工填写每个条目的 verdict / conflict_with_scope，再跑 --summarize")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
