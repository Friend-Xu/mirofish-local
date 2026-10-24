r"""每轮耗时 vs 帖子数曲线（推荐表性能基线）。

OASIS 的 twitter 推荐表**每一轮都重算全量语料**的向量：

    rec_sys_personalized_twh(..., post_table, latest_post_count, ...)
      t_items 累积所有帖子 → corpus = user_profiles + coarse_filtering(t_items, 4000)
      generate_post_vector(model, tokenizer, corpus, batch_size=1000)

（`oasis/social_platform/recsys.py:460-536`）三点决定了成本怎么长：

* `t_items` 只增不减，所以第 n 轮的语料 = **全部历史帖子**，不是"这一轮的新帖子"；
* `coarse_filtering(..., 4000)` 把帖子截到 4000 条（超过就 `random.sample` 随机抽）；
* `user_profiles` 每轮都被追加 `# Recent post:<最近一条帖子>`（`:509-520`），所以画像
  那部分也随轮数变长——本工具按真实 bio 测，**画像部分的增长没有建模**（见 §未测）。

于是"168 轮 × 每轮 9 秒"这种线性外推是错的：每轮语料不同，累计是 O(R^2)。
本工具把这条曲线测成数字。

测什么、不测什么（代理指标必须标注）：

* **测**：生产同款 `process_batch()`（`oasis/social_platform/process_recsys_posts.py`
  的 tokenizer 调用与 pooling）在**真实语料**上的耗时——文本取自真实平台库
  （`user.bio` + `post.content`），设备 = CPU、`batch_size` = 生产值 1000。
* **不测**：`update_rec_table()` 的外壳（读库、算相似度、topk）。那部分与语料大小无关，
  而整轮的成本可以用组件级重放工具直接量（同一份库、同一条路径）。
  两边的数字必须对得上，对不上就说明这份拆解不代表生产。
* 语料上限 4000 条帖子（`coarse_filtering` 的 scale）：超过它就**随机抽 4000 条**，
  剩下的帖子连候选都不是。所以曲线在 4000 处封顶，且 4000 之后"推荐表本身不可复现"
  （`random.sample` 没有种子）。

用法（项目自己的解释器）：

    backend\.venv\Scripts\python.exe scripts\measure-recsys-round-cost.py \
        --sim sim_ad40c7be867e --items 40,92,200,...,4035 --allow-resample \
        --posts-per-round 8 --max-rounds 168 \
        --out uploads/_diagnostics/recsys-round-cost-curve.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sqlite3
import statistics
import sys
import time

PROJECT = pathlib.Path(__file__).resolve().parent.parent
BACKEND = PROJECT / "backend"
SIM_ROOT = BACKEND / "uploads" / "simulations"

#: `rec_sys_personalized_twh` 里 `coarse_filtering(..., 4000)` 的 scale：
#: 语料再多也只有 4000 条帖子进入编码（超出的被随机丢弃，不是全部）。
POST_SCALE = 4000

#: 生产的 `generate_post_vector(..., batch_size=1000)`
PRODUCTION_BATCH_SIZE = 1000

#: 本工具的默认批大小。**故意小于生产值**：`process_batch` 用 `padding=True`，整批会 pad 到
#: 批内最长序列，中间张量按 `batch × seq` 长——实测把机器打死过一次
#: （CPU：`not enough memory: 1,380,725,352 bytes`；GPU：直接把卡打成 `GPU is lost`，只能重启）。
#: 要测生产批大小就显式 `--batch-size 1000`，**先过 `--vram-budget-mb` 那道守卫**。
SAFE_BATCH_SIZE = 100

#: 池化探针的默认批大小（它一次要连开两个模型，更保守）
SAFE_POOLER_BATCH_SIZE = 64

#: 单个测量点的 padding 后 token 上限（批大小 × 批内最长序列，逐批累加）。
#: 2,000,000 ≈ 200 批 × 最长 267 token 的量级；超了说明这个点不该在 CPU 上跑。
MAX_PADDED_TOKENS = 2_000_000

#: 估算要乘的**实测系数**：`estimate_forward_memory_mb()` 只算"最大的那一块"中间张量，
#: 而一次前向里 query/key/value、softmax、FFN 中间量是**同时**存在的。
#: 实测（8 GB 卡，twhin，批内最长 267 token，走生产分块入口）：
#:   batch=30  → 峰值 **1648 MB**（估算 280 MB）
#:   batch=32  → 峰值 **1684 MB**
#:   batch=64  → 峰值 **2277 MB**
#:   batch=100 → 峰值 **2945 MB**
#: 即 ≈ **29.4 MB / 批单位**（seq=267 时），且**与语料总条数无关**（92 条与 400 条都是 1647）。
#: 取 **35（含 19% 余量）**做跑之前的粗筛；换模型、换 seq、换设备都要重测这个数字。
PEAK_MB_PER_BATCH_UNIT = 35.0

#: 显存预算（MB）：`batch × PEAK_MB_PER_BATCH_UNIT` 的上限（跑之前那道筛）。
#: 8 GB 的卡上桌面占约 0.5 GB、模型 FP32 占 1.08 GB、CUDA 上下文与碎片再占一部分。
VRAM_BUDGET_MB = 5000

#: **实测硬上限**（MB）：单个测量点的 `torch.cuda.max_memory_allocated()` 不许超过它。
#: 这是真正的判据——粗筛估不准时靠它兜底，并把已测到的点落盘。
VRAM_CEILING_MB = 7000

#: 隐藏层宽度（`models/twhin-bert-base/config.json`：`hidden_size = 768`）
HIDDEN_SIZE = 768

#: BERT-base 的结构常量（同上：12 头）
ATTENTION_HEADS = 12


def estimate_forward_memory_mb(batch_size: int, longest_tokens: int,
                               heads: int = ATTENTION_HEADS,
                               hidden: int = HIDDEN_SIZE) -> dict:
    """估一次前向里**最大的那块中间张量**，返回两项并取较大者。

    twhin 的位置编码是 `relative_key`，所以 transformers 用不了 flash/SDPA 的省内存内核
    （加载时它自己会警告 `Falling back to the manual attention implementation`），
    **整张注意力矩阵会真的物化**。但还有一块更容易被忽略的：

        hidden_states : batch × seq × 768 × 4 字节
        attention     : batch × heads × seq × seq × 4 字节

    谁大取决于 `768` 与 `heads × seq` 的比较。**实测教训**：8 GB 的卡被
    `batch=1000 × seq=267` 打成 `GPU is lost`（只能重启恢复）——那一次瓶颈是 hidden
    （1000×267×768×4 ≈ 0.78 GB/份，而多层与 FFN 中间量并存时会翻几倍），
    不是注意力矩阵。只盯着注意力会严重低估，所以这里两个都算、取 max。
    """

    hidden_mb = batch_size * longest_tokens * hidden * 4 / 1024 / 1024
    attention_mb = batch_size * heads * longest_tokens * longest_tokens * 4 / 1024 / 1024
    return {
        "batch_size": batch_size,
        "longest_tokens": longest_tokens,
        "hidden_mb": round(hidden_mb, 1),
        "attention_mb": round(attention_mb, 1),
        "peak_mb": round(max(hidden_mb, attention_mb), 1),
        "bottleneck": "hidden_states" if hidden_mb >= attention_mb else "attention_matrix",
    }


def assert_vram_budget(label: str, batch_size: int, longest_tokens: int,
                       budget_mb: int, free_mb: float | None,
                       actual_batch: int | None = None) -> dict:
    """跑之前先算、超预算就拒绝并打印数字——不允许靠 OOM（或掉卡）来发现。

    `actual_batch` 是**这一批实际会有多少条**（`min(请求的 batch, 语料条数)`）：
    传 92 条语料配 `batch_size=1000` 时，真正的批就是 92 条——按 1000 算会把
    估算放大十倍，把本来安全的点误拒（实测：92 条 × batch 1000 的峰值是 2796 MB）。

    **没有"跳过这道闸"的开关**：实测教训是跳过它就会掉卡（8 GB 卡 `GPU is lost`
    三次，每次都只能重启恢复）。要跑更大的批，就只能改 `--batch-size`。
    """

    effective = batch_size if actual_batch is None else min(batch_size, actual_batch)
    estimate = estimate_forward_memory_mb(effective, longest_tokens)
    single = estimate["peak_mb"]
    # 跑之前的粗筛用**实测系数**（MB/批单位），它把整次前向的所有中间量都算进去了；
    # 单块估算只作为参考一起打印，不用它做判据。
    needed = round(effective * PEAK_MB_PER_BATCH_UNIT, 1)
    estimate.update({"label": label, "budget_mb": budget_mb,
                     "effective_batch": effective,
                     "estimate_mb": single, "measured_peak_estimate_mb": needed,
                     "peak_mb_per_batch_unit": PEAK_MB_PER_BATCH_UNIT,
                     "free_vram_mb": None if free_mb is None else round(free_mb)})
    if needed > budget_mb:
        raise SystemExit(
            f"{label}: 这个规模按实测系数 {PEAK_MB_PER_BATCH_UNIT} MB/批单位 × "
            f"{effective} 条/批 估峰值约 {needed:.0f} MB，超过预算 {budget_mb} MB"
            + (f"（当前可用显存 {free_mb:.0f} MB）" if free_mb else "")
            + "。这个规模有把显存打满的风险（实测把 8 GB 卡打成 `GPU is lost` 两次，\n"
            f"  只能重启恢复；单块最大中间张量约 {single:.0f} MB，{estimate['bottleneck']}）：\n"
            f"  · 调小 --batch-size，或\n"
            f"  · 显式调小 --max-padded-tokens，或\n"
            f"  · 确认预算后再用 --vram-budget-mb 覆盖"
        )
    if free_mb is not None and needed > free_mb * 0.85:
        raise SystemExit(
            f"{label}: 估峰值约 {needed:.0f} MB，占当前可用显存 {free_mb:.0f} MB 的"
            f"{needed / max(1.0, free_mb) * 100:.0f}%（留 15% 给碎片与上下文）。"
            f"请调小 --batch-size 或先释放显存"
        )
    return estimate


#: 跨运行参考值（**不能用它做减法**）：2026-10-11 12:46 在空闲机器上跑整轮
#: `update_rec_table()`（40 profile + 52 帖子 = 92 条语料）= 6.46 秒（裸调用 6.23 秒）。
#: 本工具自己的历史里同一份数据先测到 8.85 s、再测到 10.50 s——笔记本 CPU 持续负载下
#: 的热漂移比外壳成本还大，所以真正的校准必须在**同一次运行里**量（`--anchor-items`）。
ANCHOR = {"corpus_items": 92, "seconds": 6.46,
          "source": "组件级重放工具 --arm project-module"
                    "（2026-10-11 12:46，sim_ad40c7be867e；跨运行参考，不做减法）"}


def abort_and_save(out_path, *, reason: str, rows: list, **extra) -> None:
    """中止测量时把**已经测到的点**落盘：部分曲线比一条错误信息有用。

    不写盘就等于让已经付出的那几分钟白费，而且下次还得重测一遍已知的点。
    """

    if not out_path:
        return
    payload = {"aborted": True, "reason": reason, "curve": rows}
    payload.update(extra)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")


def read_corpus(db_path: pathlib.Path) -> dict:
    """从真实平台库取语料：`user.bio` + `post.content`（与 OASIS 的 corpus 同源）。"""

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        bios = [row[0] or "" for row in connection.execute("select bio from user order by user_id")]
        posts = [row[0] for row in connection.execute("select content from post order by post_id")]
    finally:
        connection.close()

    if not bios or not posts:
        raise SystemExit(
            f"语料为空（profile={len(bios)} post={len(posts)}）：{db_path}——"
            "没有语料就测不出曲线，不接受空曲线"
        )
    return {
        "db": str(db_path),
        "profiles": bios,
        "posts": posts,
        "profile_chars": [len(t) for t in bios],
        "post_chars": [len(t) for t in posts],
    }


def build_corpus(corpus: dict, items: int, allow_resample: bool = False) -> list:
    """构造恰好 `items` 条语料的文本列表。

    OASIS 的顺序是 `user_profiles + posts`。语料多于真实文本量时，只有显式给出
    `--allow-resample` 才**循环取用**真实文本（长度分布与词汇仍来自真实产物，
    但同一条文本会重复出现，这会改变 batch 内的最长序列 → 时间只能当上界看）。
    默认拒绝：拿循环文本冒充"真实语料"会让这条曲线看起来比实际更可信。
    """

    profiles, posts = corpus["profiles"], corpus["posts"]
    if items > len(profiles) + len(posts) and not allow_resample:
        raise SystemExit(
            f"目标语料 {items} 条超过真实库的文本量 {len(profiles) + len(posts)}；"
            f"要按真实文本循环取用来测大语料，显式加 --allow-resample"
        )

    take_profiles = min(len(profiles), items)
    texts = profiles[:take_profiles]
    remaining = items - take_profiles
    index = 0
    while remaining > 0:
        texts.append(posts[index % len(posts)])
        index += 1
        remaining -= 1
    return texts


def seed_pooler(model, seed: int) -> None:
    """把 pooler 重新随机初始化——**复刻一次新进程加载模型时发生的事**。

    `BertPooler` 自己没有 `_init_weights`（那是 `PreTrainedModel` 的方法），不能直接调。
    这里照 `transformers/modeling_utils.py::PreTrainedModel._init_weights` 对 `nn.Linear`
    的分支逐字重写：`weight.normal_(0, config.initializer_range)` + `bias.zero_()`。
    真实加载走的是同一个分布（`AutoModel.from_pretrained` 对缺失的 pooler 权重调它）。
    """

    import torch

    torch.manual_seed(seed)
    std = model.config.initializer_range
    dense = model.pooler.dense
    dense.weight.data.normal_(mean=0.0, std=std)
    if dense.bias is not None:
        dense.bias.data.zero_()


def summarize_pooler(vectors_per_seed: list, profiles: int, items: int, seeds: list,
                     readout: str) -> dict:
    """把两个种子下的向量变成"推荐表差多少"的数字（详见 `main` 里的 --pooler-seeds）。

    为什么这值得测：`process_batch`（`process_recsys_posts.py:33`）返回的是
    `outputs.pooler_output`，而 twhin 权重里**没有** `pooler.*`（215 个张量里一个都没有）
    ——`AutoModel` 会把 `BertModel.pooler` 建出来并**随机初始化**（加载时 transformers
    自己会警告 `pooler.dense.* are newly initialized`）。于是每次进程加载模型，
    所有向量都要过一遍**另一组随机矩阵**。

    本函数只回答一个问题：**同一份语料、两次加载，排序依据稳不稳**。不涉及质量判断。
    注意 `identical_top_lists` 是**余弦 top-2**（纯相似度），不是 SQL 里那张 `rec` 表：
    真实排序是 `余弦 × 新旧分数`（`recsys.py:583`），新旧分数在同库同一时刻是常量，
    所以这里量的是它的相似度那一半。真实 `rec` 表的两次运行对照见
    组件级重放工具的 `--dump-rec/--compare-rec`。
    """

    import numpy as np
    import torch
    from sklearn.metrics.pairwise import cosine_similarity

    max_rec_post_len = 2  # OASIS twitter 平台的取值（max_rec_post_len=2）
    tops, user_vectors, post_vectors, similarities_per_seed = [], [], [], []
    for vectors in vectors_per_seed:
        users, posts = vectors[:profiles], vectors[profiles:]
        user_vectors.append(users)
        post_vectors.append(posts)
        similarities = cosine_similarity(users, posts)  # 与 `recsys.py:567` 同一个函数
        similarities_per_seed.append(similarities)
        rank = torch.tensor(similarities).topk(max_rec_post_len, dim=1).indices.numpy()
        tops.append([sorted(int(i) for i in row) for row in rank])

    def _row_cosine(left, right):
        return (left * right).sum(axis=1) / (
            np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1))

    post_cosine = _row_cosine(post_vectors[0], post_vectors[1])
    user_cosine = _row_cosine(user_vectors[0], user_vectors[1])
    overlap = [len(set(a) & set(b)) for a, b in zip(tops[0], tops[1])]
    # 相似度本身的分布：它是"兴趣信号"能与新旧分数竞争的量度。余弦全挤在 0.9x 时，
    # 0.6–1.0 的新旧分数乘子仍然主导排序——这是要如实报告的事实，不是要藏起来的结果。
    flat = similarities_per_seed[0].ravel()

    report = {
        "items": items,
        "profiles": profiles,
        "posts": items - profiles,
        "seeds": [int(s) for s in seeds],
        "readout": readout,
        "max_rec_post_len": max_rec_post_len,
        "identical_top_lists": sum(1 for o in overlap if o == max_rec_post_len),
        "identical_top_lists_ratio": round(
            sum(1 for o in overlap if o == max_rec_post_len) / max(1, profiles), 4),
        "mean_top_overlap": round(sum(overlap) / max(1, profiles), 3),
        "no_overlap_users": sum(1 for o in overlap if o == 0),
        "same_post_vector_cosine_mean": round(float(np.mean(post_cosine)), 4),
        "same_post_vector_cosine_min": round(float(np.min(post_cosine)), 4),
        "same_user_vector_cosine_mean": round(float(np.mean(user_cosine)), 4),
        "user_post_cosine_min": round(float(flat.min()), 4),
        "user_post_cosine_median": round(float(np.median(flat)), 4),
        "user_post_cosine_max": round(float(flat.max()), 4),
    }
    print(f"pooler 探针（同一语料、两个种子，读出 {readout}，"
          f"{profiles} 个画像 × {items - profiles} 条帖子）:", flush=True)
    print(f"  同一条帖子两次加载的向量余弦均值 {report['same_post_vector_cosine_mean']}"
          f"（最低 {report['same_post_vector_cosine_min']}）", flush=True)
    print(f"  余弦 top-2 完全相同的用户 {report['identical_top_lists']}/{profiles}"
          f"（{report['identical_top_lists_ratio'] * 100:.0f}%），"
          f"平均命中 {report['mean_top_overlap']}/{max_rec_post_len}，"
          f"两条全不同 {report['no_overlap_users']} 个用户", flush=True)
    print(f"  用户画像×帖子的余弦分布: 最低 {report['user_post_cosine_min']}、"
          f"中位 {report['user_post_cosine_median']}、"
          f"最高 {report['user_post_cosine_max']}（新旧分数乘子范围约 0.6–1.0）",
          flush=True)
    return report


def measure_shell(corpus: dict, work_dir: pathlib.Path, items: int) -> dict:
    """在**同一次运行里**量一遍真实 `Platform.update_rec_table()`（整轮）。

    为什么必须放进同一次运行：本工具自己的历史里，92 条语料先测到 8.85 s、再测到
    10.50 s——同一台机器、同一份代码、同一个库（笔记本 CPU 持续负载下的漂移）。
    跨运行拿"整轮锚点"跟"编码耗时"相减，减出来的是两个不同热状态的差，不是外壳成本。

    做法：把平台库复制到工作目录，按 `items` 条真实文本**重写 post 表**（只留该有的行数），
    再用产品同款参数建 `Platform`（`recsys_type="twhin-bert"`、`max_rec_post_len=2`）
    调 `update_rec_table()`。第一次调用额外做一次"预热"（排除首次进入 torch 的开销），
    报告取第二次。
    """

    import asyncio
    import shutil
    import sqlite3

    from oasis.social_platform.channel import Channel
    from oasis.social_platform.platform import Platform

    work_dir.mkdir(parents=True, exist_ok=True)
    db_copy = work_dir / "anchor-twitter_simulation.db"
    shutil.copy2(corpus["db"], db_copy)

    texts = build_corpus(corpus, items, allow_resample=True)
    profiles = corpus["profiles"]
    post_texts = texts[len(profiles):]

    connection = sqlite3.connect(db_copy)
    try:
        cursor = connection.cursor()
        columns = [row[1] for row in cursor.execute("pragma table_info(post)")]
        cursor.execute("delete from post")
        for index, content in enumerate(post_texts, start=1):
            # created_at 全用同一个值 → `latest_post_count` 抓得到它们（真实代码按
            # "最新两个时间戳"数行数，时间戳全不同就会只数到 1 条）
            cursor.execute(
                "insert into post (post_id, user_id, content, created_at) "
                "values (?, ?, ?, ?)",
                (index, (index % len(profiles)) + 1, content, 1000),
            )
        if not post_texts:
            raise SystemExit("anchor 语料里没有帖子：items 必须大于 profile 数")
        connection.commit()
    finally:
        connection.close()

    # OASIS 导入期就往 ./log 写，日志级别是 DEBUG——测时间时把它压掉，别让 IO 干扰
    import logging
    logging.getLogger("social.rec").setLevel(logging.CRITICAL)
    logging.getLogger("social.twitter").setLevel(logging.CRITICAL)

    platform = Platform(
        db_path=str(db_copy),
        channel=Channel(),
        recsys_type="twhin-bert",
        refresh_rec_post_count=2,
        max_rec_post_len=2,
        following_post_count=3,
    )

    times = []
    for _ in range(2):
        started = time.perf_counter()
        asyncio.run(platform.update_rec_table())
        times.append(time.perf_counter() - started)
    # 第二次为报告值：第一次含懒加载/分配开销
    return {
        "corpus_items": items,
        "post_rows": len(post_texts),
        "profile_rows": len(profiles),
        "seconds_first": round(times[0], 3),
        "seconds_second": round(times[1], 3),
        "seconds": round(times[1], 3),
        "db": str(db_copy),
    }


def _resolve_pooler_readout(kind: str):
    """池化探针的读出函数。**必须在模型加载之后解析**：确定性池化是
    `load_and_inject()` 运行时装上的，先解析会拿到 OASIS 原版（随机 pooler）。"""

    from oasis.social_platform.process_recsys_posts import process_batch

    if kind == "legacy":
        original = getattr(process_batch, "_mirofish_original", None)
        if original is None:
            raise SystemExit(
                "--pooler-readout legacy 需要生产路径已经替换过 process_batch"
                "（原函数留在 `_mirofish_original` 上）；当前进程没装上补丁，"
                "复现不出修前的随机读出"
            )
        print("池化探针读出: legacy（OASIS 原版 outputs.pooler_output，仅测量用）",
              flush=True)
        return original
    if not getattr(process_batch, "_mirofish_mean_pooled", False):
        raise SystemExit(
            "池化探针读出: production 期望生产路径已装上确定性池化补丁，"
            "但 process_batch 上没有 `_mirofish_mean_pooled` 标记 —— 拒绝用随机读出"
            "冒充生产路径（同一条文本两次加载余弦 0.0379 的那种）"
        )
    print("池化探针读出: production（生产路径实际编码函数）", flush=True)
    return process_batch


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", default="sim_ad40c7be867e",
                        help="模拟目录名或 twitter_simulation.db 的路径")
    parser.add_argument("--items", default="65,70,140,260,500,1000,2000,4000",
                        help="语料条数（逗号分隔）")
    parser.add_argument("--repeats", type=int, default=2, help="每个点数重复几次（取中位数）")
    parser.add_argument("--batch-size", type=int, default=SAFE_BATCH_SIZE)
    parser.add_argument("--max-padded-tokens", type=int, default=MAX_PADDED_TOKENS,
                        help="单个测量点的 padding 后 token 上限：超了直接拒绝并打印数字，"
                             "不靠 OOM 发现（CPU 上一次吃下 4035 条 × 267 token 就是这样把"
                             "机器打死的）")
    parser.add_argument("--vram-budget-mb", type=int, default=VRAM_BUDGET_MB,
                        help="跑之前的粗筛：`batch × 35 MB/批单位` 的上限（默认 5000 MB）。"
                             "35 是实测值（batch 30/32/64/100 → 1648/1684/2277/2945 MB）")
    parser.add_argument("--vram-ceiling-mb", type=int, default=VRAM_CEILING_MB,
                        help="实测硬上限：单点 torch.cuda.max_memory_allocated() 超过它即中止"
                             "（默认 7000 MB；8 GB 卡实测把卡打成过 GPU is lost 两次，"
                             "只能重启恢复）")
    parser.add_argument("--skip-vram-precheck", action="store_true",
                        help="已废弃：跳过显存粗筛的开关被移除（实测跳过它就会掉卡）。"
                             "传这个参数会直接报错")
    parser.add_argument("--cpu-threads", type=int, default=None,
                        help="torch 线程数（默认 min(8, 逻辑核数)）；取满 20 个逻辑核会把"
                             "桌面饿死，实测卡死一次")
    parser.add_argument("--pooler-seeds", default=None,
                        help="逗号分隔的两个随机种子：探针量一遍「每次加载的随机 pooler」"
                             "对推荐结果的影响（默认不跑）")
    parser.add_argument("--pooler-only", action="store_true",
                        help="只跑池化探针、不加载曲线模型。探针每轮都要新开一个模型，"
                             "而曲线模型会一直占着显存（实测因此同时有两份 1.08 GB 权重，"
                             "把卡打成 GPU is lost）——所以探针要单独起一个进程跑")
    parser.add_argument("--pooler-max-items", type=int, default=1000,
                        help="池化探针的语料硬上限（默认 1000）。实测教训：4035 条时"
                             "探测过程中把卡打成 GPU is lost（两次），那两条数据不做也能"
                             "得出同样结论")
    parser.add_argument("--pooler-items", type=int, default=400,
                        help="pooler 探针用的语料条数（默认 400：探针要连开两个模型，"
                             "语料越大越容易把 8 GB 卡顶爆。实测在 4035 条时把卡打成 "
                             "GPU is lost；这个数字只影响 pooler 那一个统计量，2000 条与 "
                             "400 条的结论一致——两次加载的同一条文本向量余弦都在 0.04 量级）")
    parser.add_argument("--pooler-batch-size", type=int, default=SAFE_POOLER_BATCH_SIZE,
                        help="pooler 探针的 batch：它一次要连开两个模型，默认比曲线更保守"
                             "（64 vs 100）")
    parser.add_argument("--pooler-readout", choices=("production", "legacy"),
                        default="production",
                        help="池化探针用哪个读出：production=生产路径（已换成确定性均值池化，"
                             "推荐表实际拿到的向量）；legacy=OASIS 原版的 "
                             "outputs.pooler_output（复现修前的随机投影，**只存在于测量工具里**，"
                             "产品代码没有这条路径）")
    parser.add_argument("--allow-resample", action="store_true",
                        help="目标语料超过真实文本量时，按真实文本循环取用（时间当上界看）")
    parser.add_argument("--anchor-items", type=int, default=None,
                        help="同一次运行里再量一遍真实 Platform.update_rec_table()（整轮），"
                             "语料取这么多条——跨运行相减会被 CPU 热漂移污染")
    parser.add_argument("--anchor-work-dir", default=None,
                        help="anchor 用的工作目录（默认 backend/uploads/_diagnostics/）")
    parser.add_argument("--posts-per-round", type=float, default=None,
                        help="按这个帖子增长率把曲线换算成累计预算（例如 8）")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="换算累计预算到第几轮（例如 168）")
    parser.add_argument("--out", default=None, help="把原始测量写成 JSON")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None,
                        help="只用于**如实报告**：本工具不搬模型（设备由产品路径决定）。"
                             "要测 CPU 就整个进程设 CUDA_VISIBLE_DEVICES=")
    args = parser.parse_args(argv)

    if args.skip_vram_precheck:
        raise SystemExit(
            "--skip-vram-precheck 已被移除：跳过显存粗筛实测会把 8 GB 卡打成 "
            "`GPU is lost`（三次，每次都只能重启恢复）。要跑更大的批就改 "
            "--batch-size，或者用 --vram-budget-mb 明确提高预算（那是一个有意识的选择，"
            "不是把闸拆掉）"
        )

    db_path = pathlib.Path(args.sim)
    if not db_path.exists():
        db_path = SIM_ROOT / args.sim / "twitter_simulation.db"
    if not db_path.exists():
        raise SystemExit(f"平台库不存在: {args.sim}")

    out_path = None
    if args.out:
        out_path = pathlib.Path(args.out)
        if not out_path.is_absolute():
            out_path = BACKEND / out_path

    sys.path.insert(0, str(BACKEND / "scripts"))
    from oasis.social_platform import recsys as oasis_recsys
    from twitter_recsys_model import load_and_inject, pin_offline_runtime

    pin_offline_runtime()

    def load_model():
        """每次都走产品路径 `load_and_inject()`（真实加载次数 = 真实随机 pooler 组数）。"""
        started = time.perf_counter()
        directory, seconds, device = load_and_inject()
        print(f"模型加载: {directory}（设备 {device}，清单校验+transformers "
              f"{time.perf_counter() - started:.1f} s，其中 transformers {seconds:.1f} s）",
              flush=True)
        return oasis_recsys.twhin_model, oasis_recsys.twhin_tokenizer, seconds

    # 用生产同款的编码函数，绝不在本工具里重写一份 tokenizer 调用。
    # `generate_post_vector` 是**按 batch 分块**的那一层（生产也是它），
    # `process_batch` 只是块内的原始前向。
    from oasis.social_platform.process_recsys_posts import (generate_post_vector,
                                                            process_batch)
    import torch

    def load_tokenizer_only():
        """只加载 tokenizer（模型目录里的 tokenizer.json）：`--pooler-only` 用它算
        "批内最长序列"，**不占显存**。"""

        from transformers import AutoTokenizer
        from local_models import model_dir

        directory = model_dir("twhin-bert-base")
        return AutoTokenizer.from_pretrained(str(directory), model_max_length=512)

    if args.pooler_only:
        # **单独一个进程只跑池化探针**：探针每轮都要新开一个模型来拿到"另一次加载"的
        # 随机 pooler，而曲线模型的局部变量（`model`）一旦加载就会一直活着——
        # 那样显存里同时有两份 1.08 GB 权重（实测就是这么把卡打成 GPU is lost 的）。
        # 曲线这时还不需要，所以只加载 tokenizer（算"批内最长序列"要用），不加载模型。
        model = None
        tokenizer = load_tokenizer_only()
        load_seconds = 0.0
    else:
        model, tokenizer, load_seconds = load_model()
        if model is None or tokenizer is None:
            raise SystemExit("注入失败：OASIS recsys 的 twhin_model/tokenizer 仍为 None")

    # 线程数**不许取满逻辑核**：这是之前把机器（和桌面）饿死的原因之一
    threads = args.cpu_threads if args.cpu_threads else min(8, os.cpu_count() or 1)
    if threads >= (os.cpu_count() or 1) and os.cpu_count() and os.cpu_count() > 2:
        print(f"警告：--cpu-threads {threads} 等于全部逻辑核 {os.cpu_count()}，"
              f"可能把桌面饿死", flush=True)
    torch.set_num_threads(max(1, threads))

    # 设备如实报告：模型由 `load_and_inject()` 搬到 GPU，本工具不改设备
    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.pooler_only else next(model.parameters()).device
    print(f"模型参数设备: {model_device}", flush=True)
    if args.device == "cpu" and model_device.type != "cpu":
        raise SystemExit(
            "要测 CPU 就是这个进程不该看到 GPU：用 `CUDA_VISIBLE_DEVICES=` 启动本工具，"
            "不要去改模型的设备（那测出来的不是生产路径）"
        )

    corpus = read_corpus(db_path)
    posts_total = len(corpus["posts"])
    profiles_total = len(corpus["profiles"])
    print(f"真实语料: {db_path.name} profiles={profiles_total} posts={posts_total}"
          f"（bio 中位 {statistics.median(corpus['profile_chars']):.0f} 字，"
          f"post 中位 {statistics.median(corpus['post_chars']):.0f} 字）", flush=True)
    print(f"设备: {model_device} | torch {torch.__version__} | 线程 {torch.get_num_threads()} | "
          f"batch_size {args.batch_size}", flush=True)

    # 显存守卫：先把"要多少"算出来（按最长序列的注意力矩阵），再决定跑不跑。
    # 只在 GPU 上算——CPU 那侧由 `--max-padded-tokens` 管。
    def free_vram_mb():
        if model_device.type != "cuda":
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / 1024 / 1024

    pooler_report = None
    if args.pooler_seeds:
        seeds = [int(s) for s in args.pooler_seeds.split(",")]
        if len(seeds) != 2:
            raise SystemExit("--pooler-seeds 需要两个种子，例如 1,2")
        if args.pooler_items > args.pooler_max_items:
            raise SystemExit(
                f"--pooler-items {args.pooler_items} 超过硬上限 {args.pooler_max_items}："
                f"探针要连开两个模型，语料越大越容易把卡顶爆（实测 4035 条两次把卡打成 "
                f"GPU is lost，只能重启）。要突破上限请显式调 --pooler-max-items"
            )
        probe_texts = build_corpus(corpus, args.pooler_items,
                                   allow_resample=args.allow_resample)
        # 池化探针要连开两个模型：每个 forward 之前各自过一次预算
        probe_longest = max(
            len(probed_ids)
            for probed_ids in (
                tokenizer(probe_texts[start:start + args.pooler_batch_size],
                          padding=True, truncation=True)["input_ids"]
                for start in range(0, len(probe_texts), args.pooler_batch_size)
            )
        )
        free_now = free_vram_mb()
        probe_batch = min(args.pooler_batch_size, len(probe_texts))
        assert_vram_budget(
            f"pooler 探针（batch={probe_batch}）",
            probe_batch, probe_longest, args.vram_budget_mb, free_now,
            actual_batch=len(probe_texts),
        )
        if free_now is not None:
            estimate = estimate_forward_memory_mb(probe_batch, probe_longest)
            print(f"pooler 探针显存预算: 按实测系数 "
                  f"{PEAK_MB_PER_BATCH_UNIT} MB/批单位估峰值约 "
                  f"{probe_batch * PEAK_MB_PER_BATCH_UNIT:.0f} MB"
                  f"（单块 {estimate['peak_mb']:.0f} MB；预算 {args.vram_budget_mb} MB，"
                  f"可用 {free_now:.0f} MB）", flush=True)
        vectors_per_seed = []
        # 读出函数**在第一次加载之后**才解析：确定性池化补丁是 `load_and_inject()`
        # 运行时装上的，提前解析会拿到 OASIS 原版（随机 pooler）——实测就是这样把
        # "修后"测成了 0.0096。
        readout = None
        for seed in seeds:
            # 上一轮的模型必须先真正放掉再加载下一个：8 GB 的卡装不下两份 1.08 GB 权重 + 中间张量
            oasis_recsys.twhin_model = None
            oasis_recsys.twhin_tokenizer = None
            import gc
            gc.collect()
            if model_device.type == "cuda":
                torch.cuda.empty_cache()
            probed_model, probed_tokenizer, _probe_seconds = load_model()
            probed_model.eval()
            if readout is None:
                readout = _resolve_pooler_readout(args.pooler_readout)
            seed_pooler(probed_model, seed)
            chunks = []
            for start in range(0, len(probe_texts), args.pooler_batch_size):
                chunks.append(readout(
                    probed_model, probed_tokenizer,
                    probe_texts[start:start + args.pooler_batch_size]))
            vectors_per_seed.append(torch.cat(chunks, dim=0).cpu().numpy())
            oasis_recsys.twhin_model = None
            del probed_model, chunks
            gc.collect()
            if model_device.type == "cuda":
                torch.cuda.empty_cache()
        pooler_report = summarize_pooler(vectors_per_seed, profiles_total,
                                         args.pooler_items, seeds,
                                         args.pooler_readout)
        del vectors_per_seed
        gc.collect()
        if model_device.type == "cuda":
            torch.cuda.empty_cache()
        if args.pooler_only:
            # 这个进程的活干完了：曲线要的模型从来没加载过
            if out_path:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(
                    {"sim": str(db_path), "mode": "pooler-only",
                     "device": str(model_device), "pooler": pooler_report},
                    ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"探针结果已写入 {out_path}", flush=True)
            return 0
        # 探针把注入的模型换掉了：曲线部分仍在原来那个模型上跑，注入复原
        oasis_recsys.twhin_model = model
        oasis_recsys.twhin_tokenizer = tokenizer

    items_list = [int(x) for x in args.items.split(",")]
    if not args.allow_resample:
        too_big = [i for i in items_list if i > profiles_total + posts_total]
        if too_big:
            raise SystemExit(
                f"目标语料 {too_big} 超过真实库的文本量 {profiles_total + posts_total}；"
                f"要按真实文本循环取用来测大语料，显式加 --allow-resample"
                f"（跳过这些点会让曲线看起来覆盖了更大的语料）"
            )
    rows = []
    for items in items_list:
        texts = build_corpus(corpus, items, allow_resample=args.allow_resample)
        if model_device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        # token 数按真实 padding 规则算：分 batch 后每个 batch pad 到该 batch 的最长序列。
        padded_tokens = 0
        longest = 0
        for start in range(0, len(texts), args.batch_size):
            encoded = tokenizer(texts[start:start + args.batch_size], return_tensors="pt",
                                padding=True, truncation=True)
            shape = encoded["input_ids"].shape
            padded_tokens += int(shape[0] * shape[1])
            longest = max(longest, int(shape[1]))
        after_tokenize = (round(torch.cuda.max_memory_allocated() / 1024 / 1024)
                          if model_device.type == "cuda" else None)

        # 先算代价再跑：超上限就拒绝并打印数字，不靠 OOM 发现（实测教训，见 MAX_PADDED_TOKENS）
        if padded_tokens > args.max_padded_tokens:
            raise SystemExit(
                f"语料 {items} 条在 batch_size={args.batch_size} 下需要 "
                f"{padded_tokens:,} 个 padding 后 token，超过上限 {args.max_padded_tokens:,}"
                f"（批内最长 {longest} token）。这个规模大概率会吃光内存：\n"
                f"  · 换更小的 --batch-size（当前 {args.batch_size}），或\n"
                f"  · 显式调大 --max-padded-tokens（你自己确认机器扛得住），或\n"
                f"  · 只测更小的语料点"
            )
        # GPU 上还要过显存预算：注意力矩阵会真的物化（relative_key 用不了省内存内核）
        assert_vram_budget(
            f"语料 {items} 条（batch={args.batch_size}，批内最长 {longest} token）",
            args.batch_size, longest, args.vram_budget_mb, free_vram_mb(),
            actual_batch=items,
        )

        times = []
        if model_device.type == "cuda":
            print(f"  显存：分词阶段峰值 {after_tokenize} MB"
                  f"（已分配 {round(torch.cuda.memory_allocated() / 1024 / 1024)} MB，"
                  f"缓存池 {round(torch.cuda.memory_reserved() / 1024 / 1024)} MB）", flush=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        for repeat in range(args.repeats):
            started = time.perf_counter()
            with torch.no_grad():
                # 必须走生产的 `generate_post_vector(..., batch_size=...)`：
                # 它才是"按 batch 分块"的那一层。直接调 `process_batch(整份语料)` 会
                # 把 400 条一次 pad 到 267 token——**--batch-size 完全不起作用**，
                # 实测峰值 8509 MB（超过 8 GB 卡，往宿主内存溢出，秒数从 1.93 掉到 23.55）。
                vectors = generate_post_vector(model, tokenizer, texts,
                                               batch_size=args.batch_size)
            elapsed = time.perf_counter() - started
            times.append(elapsed)
            shape = tuple(vectors.shape)
            print(f"  语料 {items:5d} 条 → {elapsed:6.2f} s（第 {repeat + 1}/{args.repeats} 次，"
                  f"批内最长 {longest} token，{args.batch_size} 条/批，向量 {shape}）",
                  flush=True)
        peak_vram = None
        if model_device.type == "cuda":
            peak_vram = round(torch.cuda.max_memory_allocated() / 1024 / 1024)
            # 实测硬上限：超了就**中止整个测量**，并把已经测到的点留在 JSON 里。
            # 为什么必须中止而不是"记一句警告继续"：超过卡容量后 CUDA 会往宿主内存溢出，
            # 测出来的秒数已经不是 GPU 的速度了（实测 2.19 s → 25.25 s），
            # 继续测下去只会得到一条假的曲线，还可能把卡再打死一次。
            if peak_vram > args.vram_ceiling_mb:
                abort_and_save(
                    out_path, reason=f"语料 {items} 条：显存峰值 {peak_vram} MB > "
                                     f"上限 {args.vram_ceiling_mb} MB",
                    rows=rows, batch_size=args.batch_size, device=str(model_device),
                )
                raise SystemExit(
                    f"语料 {items} 条：显存峰值 {peak_vram} MB 超过实测上限 "
                    f"{args.vram_ceiling_mb} MB——再往下测就是在宿主内存上跑，秒数没有意义。\n"
                    f"  已测到的点写在 {out_path}\n"
                    f"  修法：调小 --batch-size（当前 {args.batch_size}）"
                )
        median = statistics.median(times)
        rows.append({
            "corpus_items": items,
            "unique_texts": len(set(texts)),
            "profile_items": min(profiles_total, items),
            "post_items": max(0, items - profiles_total),
            "padded_tokens": padded_tokens,
            "longest_batch_tokens": longest,
            "batch_size": args.batch_size,
            "device": str(model_device),
            "peak_vram_mb": peak_vram,
            "seconds": [round(t, 3) for t in times],
            "median_seconds": round(median, 3),
            "ms_per_item": round(median / items * 1000, 1),
        })
        print(f"→ {items} 条：中位 {median:.2f} s（{median / items * 1000:.1f} ms/条，"
              f"共 {padded_tokens:,} 个 padding 后 token"
              + (f"，显存峰值 {peak_vram} MB" if peak_vram else "") + "）", flush=True)

    # 校准（同一次运行内）：真实整轮 `update_rec_table()` 的墙钟 vs 同一语料上的编码耗时。
    # 整轮 = 编码 + 读库 + 相似度 + topk，所以外壳成本必须为正且明显小于编码。
    work_dir = pathlib.Path(args.anchor_work_dir) if args.anchor_work_dir else \
        BACKEND / "uploads" / "_diagnostics"
    anchor = dict(ANCHOR)
    anchor["stale_cross_run"] = dict(ANCHOR)  # 跨运行那次，只留档
    if args.anchor_items:
        shell_report = measure_shell(corpus, work_dir / "recsys-anchor", args.anchor_items)
        match = next((row for row in rows if row["corpus_items"] == args.anchor_items), None)
        anchor.update(shell_report)
        if match:
            measured = match["median_seconds"]
            shell = shell_report["seconds"] - measured
            anchor["encoding_seconds"] = measured
            anchor["shell_seconds"] = round(shell, 2)
            anchor["encoding_share"] = round(measured / shell_report["seconds"], 3)
            anchor["verdict"] = (
                "拆解代表生产（外壳为正且小于编码的一半）"
                if 0 <= shell <= shell_report["seconds"] * 0.5
                else "拆解与生产不符，曲线不许直接用"
            )
            print(f"校准（同一次运行）: {args.anchor_items} 条 —— 整轮 "
                  f"{shell_report['seconds']:.2f} s（二次调用）| 其中编码 {measured:.2f} s "
                  f"（{anchor['encoding_share'] * 100:.0f}%）| 外壳 {shell:.2f} s "
                  f"→ {anchor['verdict']}", flush=True)

    # 累计预算：语料 = profiles + 帖子数，帖子按增长率线性增长 → 每轮成本线性上升，
    # 累计是 O(R^2)。用实测点做线性插值，**不外推**到测量范围之外。
    budget = None
    if args.posts_per_round and args.max_rounds:
        def per_round(items: int) -> float:
            points = sorted((row["corpus_items"], row["median_seconds"]) for row in rows)
            if not points or items > points[-1][0]:
                raise SystemExit(
                    f"第 {items} 条语料超出测量范围（最大 {points[-1][0] if points else 0}），"
                    f"不接受外推——要更大的语料就加 --items"
                )
            previous = points[0]
            for point in points:
                if items <= point[0]:
                    if point[0] == previous[0]:
                        return point[1]
                    ratio = (items - previous[0]) / (point[0] - previous[0])
                    return previous[1] + ratio * (point[1] - previous[1])
                previous = point
            return points[-1][1]

        samples = []
        total = 0.0
        for round_index in range(1, args.max_rounds + 1):
            posts = round_index * args.posts_per_round
            items = min(int(profiles_total + posts), POST_SCALE + profiles_total)
            seconds = per_round(items)
            total += seconds
            if round_index in (1, args.max_rounds) or round_index % max(1, args.max_rounds // 6) == 0:
                samples.append({"round": round_index, "corpus_items": items,
                                "seconds": round(seconds, 2)})
        budget = {
            "posts_per_round": args.posts_per_round,
            "max_rounds": args.max_rounds,
            "corpus_items_last_round": items,
            "per_round_samples": samples,
            "total_seconds": round(total, 1),
            "total_minutes": round(total / 60, 1),
            "note": "只含推荐表编码；不含每轮 agent 的 LLM 推理与模型加载"
                    f"（加载 {load_seconds:.1f} s）",
        }
        print(f"累计预算（{args.posts_per_round} 帖/轮 × {args.max_rounds} 轮，"
              f"末轮语料 {items} 条）: {total:.0f} s = {total / 60:.1f} 分钟", flush=True)

    result = {
        "sim": str(db_path),
        "profiles": profiles_total,
        "posts": posts_total,
        "device": str(model_device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "threads": torch.get_num_threads(),
        "batch_size": args.batch_size,
        "vram_budget_mb": args.vram_budget_mb,
        "vram_ceiling_mb": args.vram_ceiling_mb,
        "load_seconds": round(load_seconds, 3),
        "curve": rows,
        "anchor": anchor,
        "budget": budget,
        "post_scale": POST_SCALE,
        "pooler": pooler_report,
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"原始测量已写入 {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
