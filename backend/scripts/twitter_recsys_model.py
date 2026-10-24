"""Twitter 推荐时间线的模型：从项目自有目录加载，离线可用，失败即不启动。

为什么必须有这个模块（实测，见 `掩盖台账.md` N-46 / F17 / F18）：

OASIS 的 twitter 平台固定用 `recsys_type="twhin-bert"`（`oasis/environment/env.py:81`），
它的推荐表 `rec` 就是 agent 能看到别人帖子的唯一入口。模型是
`Twitter/twhin-bert-base`（约 1.08 GB），由 `oasis/social_platform/recsys.py` 用
**仓库名**加载：

    AutoTokenizer.from_pretrained("Twitter/twhin-bert-base", model_max_length=512)

问题出在 transformers 的这个分支（`tokenization_utils_base.py:2390`）：只要
`vocab_size > 100000` **且入参不是本地目录**，它就会调 `_patch_mistral_regex()`
→ `is_base_mistral()` → `huggingface_hub.model_info()`，也就是**一次 Hub API 请求**，
哪怕模型文件已经全在本地。实测本模型 `vocab_size = 250002`，必然命中这个分支。
而启动脚本为了不让图谱 embedding 联网下载，设了 `HF_HUB_OFFLINE=1` /
`TRANSFORMERS_OFFLINE=1`（`启动MiroFish-本地图谱.cmd`），于是这次请求抛
`OfflineModeIsEnabled`：

    huggingface_hub.errors.OfflineModeIsEnabled: Cannot reach
    https://huggingface.co/api/models/Twitter/twhin-bert-base: offline mode is enabled.
      ← recsys.load_model → Exception: Failed to load the model: Twitter/twhin-bert-base

OASIS 把这个异常吞在自己的 `Platform.update_rec_table()` 里
（`platform.py:372-375`：`except Exception as e: twitter_log.error(e); return`），
`rec` 表因此一直是空的——agent 看不到任何别人的帖子，动作空间里只剩"发帖"。
实测：18 个模拟产物里 twitter 的 `rec` **全部为 0**，reddit（不需要模型）全部非空。

修法分两层，都不是"再兜一层"：

* **加载走项目自有目录**：`<项目>/models/twhin-bert-base/`（1.08 GB，5 个文件），
  入参是目录 → transformers 走 `_is_local` 分支 → 不请求 Hub。
  位置与内容由 `model.json` 清单固定（repo、revision、每个文件的字节数与 sha256），
  由 `local_models.verify_model()` 校验：**缺文件、字节数不符、sha256 不符、出现
  清单外的第二份权重，都抛 `RecsysModelUnavailable`**，不下载也不回退。
  （F18 之前它躺在 HuggingFace 的 hub 缓存布局里：加载要读 `refs/main`，
  同一个 1.12 GB 权重存两份，总额 2.16 GB，还混着 `.no_exist/`、`tmp*` 残留。）
* **失败要看得见**：`social.rec` / `social.twitter` 的 ERROR 不再被压掉；
  `ActionRecorder.require_timeline()` 让"真的 step 过一轮、库里有帖子、`rec` 却为空"
  变成运行失败（`TimelineNotBuilt`），不再只是产物里的一句附注。

加载时机与常驻：本模块在**每次模拟开始时**加载一次（`run_*_simulation.py` 建环境前），
之后注入 OASIS 的模块全局，整个进程内常驻、不重载、不卸载；
每个新模拟是新的子进程，所以那 8–9 秒的加载每轮运行付一次，不是每轮付一次。

句向量（N-47）：本模块也拥有**读出的那一步**。OASIS 的
`oasis/social_platform/process_recsys_posts.py:33` 返回 `outputs.pooler_output`，
而 twhin 权重里**没有** `pooler.*`（215 个张量里一个都没有）→ `BertModel` 每次加载
都随机初始化那层矩阵，同一条帖子两次加载的向量余弦实测只有 0.0379。而相似度是**真的**
参与排序的：`recsys.py:583` 的 `cosine_similarities * scores[...]` 在
`if enable_like_score:` 分支（`:571`）**外面**，同一个 8 空格层级。所以读出必须是确定的：
注入时把 `process_batch` 换成确定性均值池化（`_install_embedding_patch`），
**装不上就拒绝启动**（不许继续用随机投影，也不许加环境变量开关）。
"""

from __future__ import annotations

import os
import pathlib
import time
from typing import Any, Tuple

from local_models import (
    ModelFilesUnavailable,
    model_dir,
    verify_model,
)

#: 模型目录名（`<项目>/models/twhin-bert-base/`）
MODEL_NAME = "twhin-bert-base"

#: OASIS 加载 tokenizer 时用的长度上限（与 `oasis/social_platform/recsys.py:68` 一致）
MODEL_MAX_LENGTH = 512


class RecsysModelUnavailable(ModelFilesUnavailable):
    """twitter 推荐模型不可用：位置、缺失/不符的文件、原始异常都写在信息里。

    这是一个**边界**错误，不是可以继续跑的状态：没有这个模型，twitter 的推荐时间线
    建不起来，agent 看不到别人的帖子，整场模拟只剩广播。
    """


def resolve_device() -> Tuple[str, str]:
    """这次推理在哪个设备上跑：返回 `(设备, 可读描述)`。

    为什么必须显式做这件事：OASIS **自己**的加载器会 `.to(device)`
    （`recsys.py:78-79`），但它只在 `twhin_model is None` 时才自己加载；我们已经在
    `load_and_inject()` 里把模型注入进去了，于是**那条 `.to(device)` 不会执行**。
    实测（掩盖台账 N-47）：即使装了 CUDA 版 torch，注入的模型仍留在 CPU——
    每轮 4000 条语料的编码就是那次"机器卡死"的直接来源。

    判据与 OASIS 一致（`recsys.py:46`：`"cuda" if torch.cuda.is_available() else "cpu"`），
    **没有环境变量开关**：有卡就用卡；要 CPU 就显式设 `CUDA_VISIBLE_DEVICES=`。
    """

    import torch

    if torch.cuda.is_available():
        return "cuda", f"cuda:0 / {torch.cuda.get_device_name(0)}"
    return "cpu", "cpu"


def pin_offline_runtime() -> None:
    """把运行期钉成"不联网、不下载"。必须在 `import oasis`（→ transformers）之前调用。

    只设两个离线开关，**不设 `HF_HOME`**：模型已经从项目自有目录按路径加载，
    环境里残留的 `HF_HOME` 影响不到任何东西——加载位置不能取决于谁启动的后端。
    """

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ═══════════════════════════════════════════════════════════════════════════
# 批大小守卫（掩盖台账 N-48）：OASIS 把 `batch_size=1000` 写死，而它在 8 GB 卡上装不下
# ═══════════════════════════════════════════════════════════════════════════

#: OASIS 请求的批大小（`recsys.py:533-536` 的 `generate_post_vector(..., batch_size=1000)`）。
#: **我们不改 vendored 包**：改的是注入后的函数，这样重装 camel-oasis 也不会丢。
OASIS_REQUESTED_BATCH = 1000

#: 实测量化（8 GB RTX 4060 Laptop、twhin、批内最长 267 token、走生产分块入口）：
#:   batch 30 → 1648 MB；32 → 1684；64 → 2277；100 → 2945
#:   256 / 512 → **把卡打成 `GPU is lost`**（只能重启恢复）
#:   1000（OASIS 生产值）→ **OOM：`13.69 GiB allocated`**（8 GB 卡的 1.7 倍）
#:
#: 线性部分：`峰值 ≈ 1081 MB（模型权重）+ 18.53 MB × 批内条数`（seq=267）。
#: 单张注意力矩阵 = `heads × seq² × 4 字节` = 12 × 267² × 4 / 2^20 = 3.26 MB，
#: 而每批单位实测 18.53 MB → 其它中间量（QKV / softmax / FFN）是它的 5.7 倍。
PEAK_OVER_ATTENTION_MATRIX = 5.7

#: 兜底批大小（A）：**算不出安全值时**用它。实测 seq=267 峰值 2277 MB，8 GB 卡安全。
FALLBACK_SAFE_BATCH = 64

#: 算安全批时先从可用显存里扣掉的部分：CUDA 上下文、缓存池碎片、输出向量与桌面的余量。
VRAM_RESERVE_MB = 768

#: 最近一次守卫决策（跨轮只在变化时打印，避免 168 轮刷屏）。
_batch_policy: dict = {}
_last_batch_notice: dict = {}

#: 句向量策略（N-47）：确定性均值池化，替代 OASIS 的随机 `pooler_output`。
_embedding_policy: dict = {
    "pooling": "mean_last_hidden_state",
    "deterministic": True,
    "source": "MiroFish twitter_recsys_model._mean_pooled_process_batch",
    "replaces": "oasis.social_platform.process_recsys_posts.process_batch (pooler_output)",
}


def _attention_matrix_mb(longest_tokens: int, heads: int) -> float:
    """一张注意力分数矩阵的显存：`batch=1` 时的 `heads × seq² × 4 字节`。"""

    return heads * longest_tokens * longest_tokens * 4 / (1024 * 1024)


def _batch_unit_mb(longest_tokens: int, heads: int) -> float:
    """**每个批内条数**要多少显存（实测系数 5.7 倍于单张注意力矩阵）。"""

    return _attention_matrix_mb(longest_tokens, heads) * PEAK_OVER_ATTENTION_MATRIX


def _free_vram_mb() -> float | None:
    """当前可用显存（MB）；CPU 或查询失败返回 None。"""

    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / 1024 / 1024
    except Exception:  # noqa: BLE001 - 查不到就是"算不出"，由调用方走兜底
        return None


def _longest_token_bound(tokenizer, texts: list) -> int:
    """批内最长序列的**上界**：用字符数当 token 数上限，不额外跑一遍 tokenizer。

    BERT 的 token 数不会超过字符数（中文 1 字≈1 token，拉丁词会被切成 <字数的
    subword），所以 `min(model_max_length, max(len(text)))` 是一个安全上界。
    """

    if not texts:
        return 1
    return max(1, min(MODEL_MAX_LENGTH, max(len(text) for text in texts)))


def _model_shape(model) -> tuple[int, int]:
    """从模型配置读 `(heads, 最坏序列长度)`；读不到就返回 (0, 0) 由调用方兜底。"""

    try:
        config = model.config
        heads = int(getattr(config, "num_attention_heads", 0) or 0)
        max_positions = int(getattr(config, "max_position_embeddings", 0) or 0)
        worst_seq = min(MODEL_MAX_LENGTH, max_positions) if max_positions else MODEL_MAX_LENGTH
        return heads, worst_seq
    except Exception:  # noqa: BLE001 - 读不到配置 = 算不出，走兜底
        return 0, 0


def resolve_batch_policy(model, device: str) -> dict:
    """**跑之前**定这场的批大小策略：B（按可用显存算）打底，A（固定安全档）兜底。

    `B` 的判据是"当前可用显存能不能放下**最坏情况**（`max_position_embeddings` 那么长的
    序列）下的至少 1 条"——算出来 < 1 就是**真的装不下**，抛错、整轮不启动。
    算得出就记下来：每轮再按当时的真实序列长度与当时的可用显存重新压一次。

    Returns:
        `{"mode": ..., "preflight_batch": ..., "reason": ..., ...}`；
        `mode` 是 `cuda-adaptive`（B）或 `cpu-fallback` / `estimate-unavailable`（A）。
    """

    policy: dict = {"device": device, "requested": OASIS_REQUESTED_BATCH}
    if device != "cuda":
        # 没有显存可算，但 CPU 上同样**不许**用 1000：实测一次 4035 条 × 267 token 的
        # 中间张量到 GB 级，把机器打死过一次。这里直接落到实测安全档（A 兜底）。
        policy.update(mode="cpu-fallback", preflight_batch=FALLBACK_SAFE_BATCH,
                      reason="CPU 无显存可算，使用实测安全档")
        return policy

    heads, worst_seq = _model_shape(model)
    free_mb = _free_vram_mb()
    if not heads or not worst_seq or free_mb is None:
        policy.update(mode="estimate-unavailable", preflight_batch=FALLBACK_SAFE_BATCH,
                      heads=heads, worst_seq=worst_seq, free_mb=free_mb,
                      reason="读不到模型结构或可用显存，使用实测安全档")
        return policy

    unit_mb = _batch_unit_mb(worst_seq, heads)
    usable_mb = free_mb - VRAM_RESERVE_MB
    safe = int(usable_mb / unit_mb) if unit_mb > 0 else 0
    policy.update(
        mode="cuda-adaptive", heads=heads, worst_seq=worst_seq, free_mb=round(free_mb),
        usable_mb=round(usable_mb), unit_mb=round(unit_mb, 2),
        preflight_batch=safe,
    )
    if safe < 1:
        raise RecsysModelUnavailable(
            f"显存不够跑推荐时间线：可用 {free_mb:.0f} MB，减去预留 {VRAM_RESERVE_MB} MB "
            f"只剩 {usable_mb:.0f} MB，而最坏情况（{worst_seq} token × {heads} 头）"
            f"每批 1 条就要约 {unit_mb:.0f} MB。\n"
            f"这不是可以继续跑的状态：算不出安全批大小就不启动（台账 N-48）。\n"
            f"  修法：关掉占显存的程序，或换显存更大的设备。"
        )
    return policy


def batch_policy() -> dict:
    """最近一次 `load_and_inject()` 定下的批大小策略（给运行日志/诊断用）。"""

    return dict(_batch_policy)


def _safe_batch_now(tokenizer, model, texts: list, requested: int) -> tuple[int, str]:
    """**每一轮**的实际批大小：按当时的真实序列与当时的可用显存再压一次。"""

    policy = _batch_policy
    if policy.get("mode") != "cuda-adaptive":
        chosen = max(1, min(requested, policy.get("preflight_batch", FALLBACK_SAFE_BATCH)))
        return chosen, policy.get("reason", "非自适应模式使用实测安全档")

    heads = policy["heads"]
    longest = _longest_token_bound(tokenizer, texts)
    unit_mb = _batch_unit_mb(longest, heads)
    free_mb = _free_vram_mb()
    if free_mb is None:
        return min(requested, FALLBACK_SAFE_BATCH), "查询可用显存失败，使用实测安全档"
    safe = int((free_mb - VRAM_RESERVE_MB) / unit_mb) if unit_mb > 0 else 0
    safe = max(1, safe)
    # 只许**变小**：OASIS 请求 1000 时压到安全值；它请求 50 时不动它
    chosen = max(1, min(requested, safe, len(texts)))
    reason = (
        f"可用显存 {free_mb:.0f} MB、批内最长约 {longest} token（{heads} 头）、"
        f"每批单位约 {unit_mb:.1f} MB → 安全批 {safe}"
    )
    return chosen, reason


def _mean_pooled_process_batch(model, tokenizer, batch_texts):
    """**确定性**句向量：`last_hidden_state` 按 attention mask 做均值池化。

    为什么要换掉 OASIS 自带的 `process_batch`（台账 N-47）：

    * OASIS 返回的是 `outputs.pooler_output`（`process_recsys_posts.py:33`），
      而 twhin 的 `model.safetensors` 里**没有任何 `pooler.*` 张量**——
      `BertModel` 会建一层 `BertPooler` 并**随机初始化**（transformers 加载时自己会警告
      `pooler.dense.bias / pooler.dense.weight are newly initialized`）。
      于是"相似度"其实是**随机投影**：同一条文本两次加载余弦只有 **0.0379**（实测），
      推荐表只有 18% 用户一致（实测）。
    * 用户在 `recsys.py:583` 的 `cosine_similarities * scores[...]` 里**真的用到**这个余弦
      （它在 `if enable_like_score:` 外面），所以随机向量会真的把推荐表搅乱。

    均值池化是 BERT 类 MLM checkpoint 的标准句表示做法：**不含可训练参数**，
    同样的输入永远给同样的向量；padding 用 mask 排除，所以一条文本单独跑和混在批里跑
    结果一致。返回 CPU 上的 `(n, hidden)` 张量，与 OASIS 的调用约定一致。
    """

    import torch

    device = next(model.parameters()).device
    with torch.no_grad():
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                           truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        hidden = model(**inputs).last_hidden_state          # (B, T, H)
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return (summed / counts).cpu()


def _install_embedding_patch() -> dict:
    """把 OASIS 的 `process_batch` 换成确定性均值池化（N-47）。

    改的是 `oasis.social_platform.process_recsys_posts.process_batch`：
    `generate_post_vector()` 在该模块内部按名字查找它，所以换掉之后
    OASIS 的推荐表拿到的就是确定性向量。同样**不改 vendored 源码**
    （重装 camel-oasis 不会丢；有静态测试锁住"源码没被改"）。
    """

    from oasis.social_platform import process_recsys_posts as oasis_posts

    original = getattr(oasis_posts, "process_batch", None)
    if original is None:
        raise RecsysModelUnavailable(
            "OASIS 的 process_recsys_posts 里找不到 process_batch：确定性池化装不上，"
            "推荐表会继续用随机 pooler 的向量（同一条文本两次加载余弦 0.0379）——拒绝启动"
        )
    if getattr(original, "_mirofish_mean_pooled", False):
        return _embedding_policy

    _mean_pooled_process_batch._mirofish_mean_pooled = True
    _mean_pooled_process_batch._mirofish_original = original
    oasis_posts.process_batch = _mean_pooled_process_batch
    return _embedding_policy


def embedding_policy() -> dict:
    """当前句向量策略（给运行日志/诊断用）。"""

    return dict(_embedding_policy)


def _install_batch_guard(tokenizer, model) -> None:
    """把 OASIS 的 `generate_post_vector` 包一层：**只允许把批变小，不允许它用 1000**。

    为什么改这里而不是改 vendored 包：`camel-oasis` 是第三方依赖，重装/升级会覆盖源码；
    而 `recsys.py` 调用的是本模块作用域里的 `generate_post_vector` 名字，
    我们在注入时把它换掉即可，且 OASIS 自己的代码一行不动。
    """

    global _batch_policy

    from oasis.social_platform import recsys as oasis_recsys

    original = getattr(oasis_recsys, "generate_post_vector", None)
    if original is None:
        raise RecsysModelUnavailable(
            "OASIS 的推荐模块里找不到 generate_post_vector：批大小守卫装不上，"
            "生产会继续用写死的 1000（8 GB 卡实测 OOM）——拒绝启动"
        )
    if getattr(original, "_mirofish_batch_guarded", False):
        return

    def guarded(model_arg, tokenizer_arg, texts, batch_size=None):
        requested = int(batch_size or OASIS_REQUESTED_BATCH)
        chosen, reason = _safe_batch_now(tokenizer_arg, model_arg, texts, requested)
        notice = {"requested": requested, "chosen": chosen, "reason": reason,
                  "items": len(texts)}
        if notice != _last_batch_notice:
            _last_batch_notice.clear()
            _last_batch_notice.update(notice)
            print(
                f"推荐表批大小: OASIS 请求 {requested} → 实际 {chosen}"
                f"（{len(texts)} 条语料；{reason}）",
                flush=True,
            )
        return original(model_arg, tokenizer_arg, texts, batch_size=chosen)

    guarded._mirofish_batch_guarded = True
    guarded._mirofish_original = original
    oasis_recsys.generate_post_vector = guarded


def load_and_inject() -> Tuple[pathlib.Path, float, str]:
    """校验项目内的 twhin 模型 → 从该目录加载 → 搬到 GPU → 注入 OASIS 的 recsys 模块。

    注入是"让 OASIS 用它已经有的东西"：`rec_sys_personalized_twh` 只在
    `twhin_model is None or twhin_tokenizer is None` 时才自己去加载
    （`recsys.py:432`），所以这里先把同一份模型按**本地目录**加载好放进去，
    它就不会再去请求 Hub，也不会加载第二份。**代价是那条 `.to(device)` 也不会执行**
    （OASIS 自带的加载器有，见 `recsys.py:78-79`），所以设备由这里负责。

    Returns:
        (模型目录, 加载耗时秒数, 设备描述)——设备必须进日志：产物里要能看出
        "这一轮到底在 GPU 还是 CPU 上算的"。

    Raises:
        RecsysModelUnavailable: 清单/文件/哈希任一不符、加载失败、显存不足
            （保留原始异常为 `__cause__`）。
    """

    try:
        directory = model_dir(MODEL_NAME)
        # 加载前先按清单校验一次（1.08 GB 实测 1.3 秒）：被截断、被替换、
        # 目录里多出第二份权重，都要在"开始跑模拟"之前失败，而不是跑到一半崩。
        verify_model(MODEL_NAME, check_hashes=True)
    except ModelFilesUnavailable as error:
        # 统一成这个模块的边界错误类型：调用方只需要认一个名字
        raise RecsysModelUnavailable(str(error)) from error

    device, description = resolve_device()

    started = time.time()
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        # 传目录而不是仓库名：transformers 只有在入参是目录时才走 _is_local 分支，
        # 才不会去调 model_info()（那条路在离线模式下必然失败）。
        tokenizer = AutoTokenizer.from_pretrained(
            str(directory), model_max_length=MODEL_MAX_LENGTH
        )
        model = AutoModel.from_pretrained(str(directory))
        # 权重先落在 CPU，再整块搬进显存；`.eval()` 关掉 dropout（池化前的两个
        # dropout 是训练期的，推理期必须关）。
        model = model.to(device).eval()
    except Exception as error:  # noqa: BLE001 - 统一成这个模块的边界错误
        # 显存不足也走这条：带着原始异常抛出，整轮不启动（不允许悄悄退回 CPU 跑）
        raise RecsysModelUnavailable(
            f"从项目自有目录加载 twhin 模型失败：{directory}（设备 {description}）"
            f"（{type(error).__name__}: {error}）"
        ) from error

    from oasis.social_platform import recsys as oasis_recsys

    # 批大小守卫（N-48）：策略在模型上卡之后定（那时才能量到"还剩多少显存"），
    # 守卫装在注入之前——否则第一次 `update_rec_table()` 就会用 OASIS 写死的 1000。
    global _batch_policy, _last_batch_notice
    _last_batch_notice = {}
    _batch_policy = resolve_batch_policy(model, device)
    _install_batch_guard(tokenizer, model)
    # 句向量守卫（N-47）：把随机 pooler 换成确定性均值池化。
    _install_embedding_patch()

    # 同一进程内重复调用时先把上一份挪走：直接覆盖会让旧模型留在显存里等 GC
    oasis_recsys.twhin_tokenizer = tokenizer
    oasis_recsys.twhin_model = model
    return directory, time.time() - started, description


def describe() -> dict:
    """清单摘要（给探针/诊断用）：不加载模型，只读 `model.json`。"""

    from local_models import read_manifest

    manifest = read_manifest(MODEL_NAME)
    return {
        "name": manifest["name"],
        "repo": manifest["repo"],
        "revision": manifest["revision"],
        "files": {
            name: meta["bytes"] for name, meta in manifest["files"].items()
        },
    }


def _main() -> int:
    """手动核对：校验 + 加载一次，打印目录与耗时。"""

    pin_offline_runtime()
    try:
        info: Any = describe()
        print(f"清单: {info['name']} repo={info['repo']} revision={info['revision'][:12]}…")
        started = time.time()
        directory, seconds, device = load_and_inject()
    except RecsysModelUnavailable as error:
        print(f"FAIL: {error}")
        return 1
    print(
        f"OK: {directory}  校验+加载 {time.time() - started:.1f} s"
        f"（其中 transformers 加载 {seconds:.1f} s）  设备: {device}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
