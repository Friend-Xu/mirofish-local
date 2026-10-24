"""报告章节的“事实/推断”与“引用保真”校验（N-44 / N-45）。

为什么单独成模块：这是报告生成侧唯一的判据来源，抽检工具（审计）也要读同一组
块标签与词表；两处各写一套词会漂移。校验器只做**结构与可核性**判断，不判断
“这条事实是否真的成立”——后者由 `report-traceability-audit.py` 抽检。

规则（与 `SECTION_SYSTEM_PROMPT_TEMPLATE` 的规则 6/7 一一对应）：

* 章节正文只允许两个块：`**模拟事实（可溯源）**` 与 `**作者推断（非模拟事实）**`；
* 事实块只写工具结果里的事实/逐字引用；程度词、量化词、推断词不得出现在
  引号外；要写判断就放进推断块；
* 推断块每条非空行以 `[推断]` 开头；
* 引号内必须逐字；翻译加 `[译文]`；概括用 `[转述]` 且不加引号。

违反任何一条都返回 `GroundingViolation`，由报告生成侧决定“一次纠正后失败”，
**不做清洗/截断/静默改写**。
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

FACT_LABEL = "**模拟事实（可溯源）**"
INFERENCE_LABEL = "**作者推断（非模拟事实）**"
INFERENCE_PREFIX = "[推断]"
TRANSLATION_MARK = "[译文]"
PARAPHRASE_MARK = "[转述]"

# 程度/量化/强弱判断：出现在事实块引号外，就不是“模拟事实”，必须进推断块。
# 只收有明确评价/量化含义的词；`最后/最终/最近/最新/最初/最早` 等时间词排除，
# 避免把正常事实陈述误判成强度判断。
STRENGTH_PATTERN = re.compile(
    r"最(?![后终近初新早])|唯一|一半|多数|少数|全部|所有|只有|至少|显著|大幅|"
    r"几乎|必然|完全|普遍|一致|罕见|空前|顶级|非常|十分|相当|明显|远远|领先|落后|饱和"
)

# 推断/因果/评价连接词：出现在事实块引号外，就必须显式标成推断。
INFERENCE_PATTERN = re.compile(
    r"意味着|说明(?!书)|表明|反映出|揭示了|可见|因此|所以|本质上|实际上|其实|"
    r"显然|可以说|堪称|预示|指向|归结为|更像是|证明了|体现了|暴露出"
)
NOT_BUT_PATTERN = re.compile(r"不是[^。；\n]{0,30}而是")

# 引号：逐字符扫描成对引号（不跨行、不跨引号类型）。
# 不能用 `["“「『](...)[\"”」』]` 这种混合字符类：它会把 `『』` 当成外层 `"`
# 的闭合，导致引语被截断，引号外的正文被误判（实测第一版错误 21 条）。
_OPEN_TO_CLOSE = {'"': '"', "“": "”", "「": "」", "『": "』"}
ELLIPSIS_SPLIT = re.compile(r"…|\.{3,}|——")
_NORMALIZE_PATTERN = re.compile(
    r"[\s\"'“”‘’「」『』，。；：、！？,.;:!?()（）。—…\-\\]+"
)


@dataclass(frozen=True)
class GroundingViolation:
    code: str
    detail: str
    text: str = ""


def normalize_for_match(text: str) -> str:
    """压成可比较的键：去空白、引号、常见标点。

    只用于“引语是否出现在工具结果里”的逐字核对；不修改原文。
    """

    return _NORMALIZE_PATTERN.sub("", text or "")


def iter_quotes(text: str):
    """产出 `(start, end, content, quote_char)`；只认同一行内、同类型的成对引号。

    `“...”` 内部的 `『...』` 算外层引语的内容，不会把外层截断；`"` 同理。
    """

    i = 0
    length = len(text or "")
    while i < length:
        opener = text[i]
        closer = _OPEN_TO_CLOSE.get(opener)
        if closer is None:
            i += 1
            continue
        end = text.find(closer, i + 1)
        newline = text.find("\n", i + 1)
        if end == -1 or (newline != -1 and newline < end):
            i += 1
            continue
        content = text[i + 1:end]
        if len(content.strip()) < 2:
            i += 1
            continue
        yield i, end + 1, content, opener
        i = end + 1


def closest_tool_segment(quote: str, tool_results: Iterable[str]):
    """给一条未逐字的引语找最接近的工具原文片段（只用于纠正提示，不改正文）。"""

    candidates = []
    for result in tool_results or []:
        for line in re.split(r"[\n。；]", result or ""):
            line = line.strip()
            if len(normalize_for_match(line)) >= 12:
                candidates.append(line)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda line: difflib.SequenceMatcher(None, quote, line).ratio(),
    )


def _strip_quotes(text: str) -> str:
    pieces = []
    cursor = 0
    for start, end, _, _ in iter_quotes(text or ""):
        pieces.append((text or "")[cursor:start])
        pieces.append(" ")
        cursor = end
    pieces.append((text or "")[cursor:])
    return "".join(pieces)


def _line_starts_with_inference_prefix(line: str) -> bool:
    # 允许模型用列表符号/引用符号包一层；标签本身仍必须是 [推断]。
    return line.lstrip("*-+> ").startswith(INFERENCE_PREFIX)


def _iter_block_lines(content: str):
    """按块标签产出 `(kind, line)`；标签本身不产出。

    kind 为 `"fact"` / `"inference"` / `None`（块外）。
    """

    kind = None
    for raw in (content or "").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped == FACT_LABEL:
            kind = "fact"
            continue
        if stripped == INFERENCE_LABEL:
            kind = "inference"
            continue
        yield kind, stripped


def _quote_checks(content: str, tool_results: Sequence[str]) -> List[GroundingViolation]:
    violations: List[GroundingViolation] = []
    tool_keys = [normalize_for_match(result) for result in tool_results if result]
    for start, _end, quote, _opener in iter_quotes(content or ""):
        quote = quote.strip()
        before = (content or "")[max(0, start - 24):start]
        if PARAPHRASE_MARK in before:
            violations.append(
                GroundingViolation(
                    "paraphrase_in_quotes",
                    f"[转述] 不得使用引号，应直接写转述内容：{quote[:60]}",
                    quote,
                )
            )
            continue
        if TRANSLATION_MARK in before:
            # 译文明确标注；不要求逐字（官方提示词也允许翻译）。
            continue
        segments = [
            segment for segment in ELLIPSIS_SPLIT.split(quote)
            if len(normalize_for_match(segment)) >= 8
        ]
        if not segments:
            continue
        for segment in segments:
            key = normalize_for_match(segment)
            if not any(key in tool_key for tool_key in tool_keys):
                violations.append(
                    GroundingViolation(
                        "quote_not_verbatim",
                        f"引号内内容未在工具结果中逐字找到：{segment[:80]}",
                        segment,
                    )
                )
                break
    return violations


def validate_section_grounding(
    content: str,
    tool_results: Iterable[str],
) -> List[GroundingViolation]:
    """返回章节的违规项；空列表 = 通过。

    这是确定性校验：不调用 LLM，不做修复。修复由报告生成侧的一次纠正完成，
    纠正后仍失败必须让报告 `failed`（`GroundingValidationError`）。
    """

    tool_results = list(tool_results or [])
    violations: List[GroundingViolation] = []
    blocks = list(_iter_block_lines(content))
    if not blocks:
        return [
            GroundingViolation(
                "missing_fact_block",
                f"章节没有正文块；必须以 {FACT_LABEL} 开头",
            )
        ]

    has_fact = any(kind == "fact" for kind, _ in blocks)
    if not has_fact:
        violations.append(
            GroundingViolation(
                "missing_fact_block",
                f"章节缺少 {FACT_LABEL} 块",
            )
        )

    for kind, line in blocks:
        if kind is None:
            violations.append(
                GroundingViolation(
                    "unlabeled_prose",
                    f"正文必须在事实块或推断块内：{line[:80]}",
                    line,
                )
            )
            continue
        if line.startswith("#"):
            violations.append(
                GroundingViolation(
                    "heading_not_allowed",
                    f"章节正文不允许 Markdown 标题：{line[:80]}",
                    line,
                )
            )
            continue
        if kind == "fact":
            if _line_starts_with_inference_prefix(line):
                violations.append(
                    GroundingViolation(
                        "inference_in_fact_block",
                        f"事实块里出现了 [推断]：{line[:80]}",
                        line,
                    )
                )
                continue
            if line.rstrip().endswith(("：", ":")):
                violations.append(
                    GroundingViolation(
                        "lead_in_in_fact_block",
                        f"事实块不得写以冒号结尾的引导句；直接放引用或事实：{line[:80]}",
                        line,
                    )
                )
                continue
            scan = _strip_quotes(line)
            strength = STRENGTH_PATTERN.search(scan)
            if strength:
                violations.append(
                    GroundingViolation(
                        "unlabeled_strength",
                        f"事实块引号外出现程度/量化词「{strength.group(0)}」：{line[:80]}",
                        line,
                    )
                )
                continue
            inference = INFERENCE_PATTERN.search(scan) or NOT_BUT_PATTERN.search(scan)
            if inference:
                violations.append(
                    GroundingViolation(
                        "unlabeled_inference",
                        f"事实块引号外出现推断/因果词「{inference.group(0)}」：{line[:80]}",
                        line,
                    )
                )
        else:  # inference
            if not _line_starts_with_inference_prefix(line):
                violations.append(
                    GroundingViolation(
                        "inference_missing_prefix",
                        f"推断块每条必须以 {INFERENCE_PREFIX} 开头：{line[:80]}",
                        line,
                    )
                )

    violations.extend(_quote_checks(content, tool_results))

    # 同一段正文可能触发多条同类问题；保留前 20 条即可读，不隐藏计数。
    if len(violations) > 20:
        violations = violations[:20] + [
            GroundingViolation(
                "more_violations",
                f"另有 {len(violations) - 20} 条同类违规未展开",
            )
        ]
    return violations


def build_correction_prompt(
    violations: Sequence[GroundingViolation],
    tool_results: Sequence[str] = (),
) -> str:
    """把违规项写回给模型；只描述规则，不替模型改正文。"""

    lines = ["【证据校验未通过，请只修正违规处并输出完整章节】", ""]
    if violations:
        lines.append("违规项：")
        for violation in violations:
            lines.append(f"- [{violation.code}] {violation.detail}")
            if violation.code == "quote_not_verbatim":
                source = closest_tool_segment(violation.text, tool_results)
                if source:
                    lines.append(f"  最接近的事实原文（请原样复制，不要改字）：{source[:300]}")
        lines.append("")
    lines.extend(
        [
            "修正要求：",
            f"1. 正文只允许两个块：{FACT_LABEL} 与 {INFERENCE_LABEL}；块外不要写正文。",
            "2. 事实块只写工具结果里的事实或逐字引用；程度词/量化词/推断词必须放进推断块。"
            "不要写以冒号结尾的引导句（如“三条如下：”），直接放引用或事实。"
            " `[转述]` 行仍属于事实块，同样不得含程度/量化/推断词；"
            "含这些词的转述整行移到推断块并改成 [推断]。",
            f"3. 推断块每条非空行必须以 {INFERENCE_PREFIX} 开头；无推断可省略该块。",
            f"4. 引号内必须逐字；翻译用 {TRANSLATION_MARK}；概括用 {PARAPHRASE_MARK} 且不加引号。",
            "5. **只改违规处**：未违规的段落保持原样，不要重写全文，也不要新增未经核对的引号。",
            "6. unlabeled_strength / unlabeled_inference：把整行移到推断块并改成 [推断]，"
            "  或删掉；不要只换词。`[转述]` 行也按这条处理。",
            "7. lead_in_in_fact_block：删除该引导句，或移到推断块。",
            "8. 对每一条 quote_not_verbatim：要么从下面的事实里原样复制一段连续原文，"
            f"要么改成 {PARAPHRASE_MARK} 并去掉全部引号；禁止保留引号改字/拼接/省略。",
            "9. 不要使用 Markdown 标题，不要调用工具，只输出 Final Answer: 后的章节正文。",
        ]
    )
    return "\n".join(lines)
