# MiroFish 本地化分支 · 开发规约

本文件是本目录（`MiroFish/`）的工程约束，优先级高于一般习惯做法。它记录了两条
在本地化移植中付出过代价的规则：**禁止兜底**、**先小样后大样**。

---

## 1. 禁止兜底（最高优先级）

> 兜底 = 首选路径失败后改走第二条路，或者为没取到的值补一个默认值。
> 两者都会在"预期路径并不成立"的情况下报告成功，于是造成它的失败不再可见——
> 既没被发现，也没被上报，更没被修复。

### 1.1 本项目明令禁止的形态

| 禁止 | 典型写法 | 正确做法 |
|---|---|---|
| 失败后换引擎/换算法 | 全文检索失败 → 纯向量；图谱检索失败 → 关键词匹配 | 在本层抛错，带上缺失的东西（索引名、端点能力、原始异常） |
| 关掉护栏来消除空结果 | 把相关性阈值默认成 0 | 阈值只由显式配置设定；未设置时用上游默认值，并在每次检索记录"阈值/条数/上限/最高分" |
| 超时后丢弃未写出的数据 | drain 到期 → 缓冲区随进程消失 | 未写出的内容落盘到 `uploads/graphs/pending/`，ERROR 记录条数与路径，运行状态标为未完成 |
| 空产物算成功 | 报告某章为空仍保存为 completed | 空产物即失败，带上章节名与原因 |
| 重试掩盖系统性缺陷 | 结构化输出反复截断 → 重试到成功为止 | 重试只允许用于瞬时错误（网络/限流/RateLimit），次数有界，每次都记录原始原因；同一错误反复出现即视为缺陷，修因不修果 |
| 边界上发明默认值 | 时间戳缺失 → 用 `now()` 顶上 | 解析失败抛出并带上原始值；缺字段在写入边界就报错 |
| 空状态与失败不可区分 | `except Exception: return []` | 空状态是显式、有类型、有注释的返回值；失败必须抛出 |

### 1.2 已修复的历史违规（回归测试见 `backend/tests/test_no_fallback_contracts.py`）

1. **`GRAPHITI_MIN_SCORE` 默认 0** — 顶掉了 graphiti 上游 0.6 的相关性过滤，
   任何查询都返回满额结果，"检索是否有效"从此不可证伪。现在：未配置即用上游默认值，
   显式配置必须落在 (0, 1]，且每次检索都记录阈值/条数/上限/最高分，条数顶到上限时告警。
   实测（`.tools/retrieval-quality.py`，6 条报告式查询）：恢复 0.6 后返回 5–10 条/查询、
   关键词命中率 52/53 = 98%——原先"0.6 会把结果全部过滤掉"的说法从未被测量过，是错的。
2. **全文索引缺失 → 静默转纯向量检索** — 检索语义被悄悄换掉。现在：
   `rebuild_kuzu_fulltext_indexes_async()` 失败即抛 `KuzuFulltextUnavailableError`，初始化失败。
   （该函数 2026-10-11 起从"缺了才建"改成"**每次初始化都先 DROP 再 CREATE**"：库里那份旧索引
   会让"更新已存在节点"以 access violation 崩掉整个进程，见 §1.2-8 与台账 N-24。）
3. **`cross_encoder` 被静默改写成 `rrf`**（靠 URL 猜端点能力）。现在：配置是唯一权威；
   调用方要求与配置不一致、或端点不提供 logprobs 时，直接抛错要求显式配置
   （调用方原先硬传 `reranker="cross_encoder"/"rrf"` 的地方已一并去掉）。
   —— **本条只算部分修**：`_resolve_reranker()` 的解析结果只被用于校验，随后被丢弃，
   `_execute_search()` 仍硬编码 `EdgeReranker.rrf`/`NodeReranker.rrf`；配置成 `cross_encoder`
   会"校验通过但实际走 rrf"，是同一类静默改写。见 `掩盖台账.md` M-07 / N-07，修复批次 F2。
4. **检索失败后的关键词兜底路径**（`zep_tools._local_search`）— 未使用但留着就会被重新接上。
   已删除；`search_graph` 只在失败时抛出。
5. **drain 超时丢弃未写出的活动** — 模拟"完成"了，图谱里却没有那批内容。现在：
   未写出的活动落盘到 `uploads/graphs/pending/<graph_id>.jsonl`；
   发送失败的批次同样落盘；工作线程异常时先落盘再停止循环，并把故障写进
   `worker_error`，`stop()` 据此报错（不再"完成"）。
6. **人设阶段的随机数冒充事实**（`oasis_profile_generator`）— Agent 的
   `karma/friend_count/follower_count/statuses_count` 原本由 `random.randint` 现编，
   而提示词从未要求模型给出这些字段：同一输入重跑会得到不同的传播权重。
   现在：提示词明确要求这 12 个字段；缺失时只在**一个具名函数**
   `_resolve_oasis_required_fields` 里按实体名**确定性**合成，并逐条 WARNING；
   用户名后缀也改为确定性（`zlib.crc32`），整场推演可复现。
   —— **本条只算部分修**：兜底没有消失，只是从"随机"变成了"可复现"。真因（A2 的规则引擎分支、
   以及"缺字段不失败"）未消除，不允许写成"已修复"。见 `掩盖台账.md` M-06，修复批次 F3。
7. **写入边界补默认值** — Reddit 档案缺 `age/mbti/country/karma` 时补 30/ISTJ/中国/1000
   的情况已改为报错；`actions.jsonl` 必填字段缺失报错；时间戳解析失败抛错，
   不再伪造 `now()`（`_to_rfc3339` / `_parse_time`）。
8. **图库里的旧全文索引 → 写回时整个进程原生崩溃**（2026-10-11，台账 N-24）—
   回写写图谱时后端进程直接消失：无 Python 异常、无 atexit、`run_state` 停在 `running`。
   `faulthandler` 落盘栈顶是 `kuzu.Connection.execute` 的 **access violation**；离线二分证明
   崩因是**库里那份 bm25 索引**（drop 掉即不崩、同样的数据重建后也不崩），与并发、WAL 无关。
   现在：初始化时 `rebuild_kuzu_fulltext_indexes_async()` **先 `SHOW_INDEXES()` 查存在的 FTS 索引
   → DROP → CREATE**，任一步失败即初始化失败；Kuzu 执行并发锁成单连接 + 单线程
   （`GRAPHITI_MAX_CONCURRENT_QUERIES=1` / `GRAPHITI_MAX_THREADS_PER_QUERY=1`，0 非法）。
   **残余风险**：这是"索引坏了就重建"，不是"引擎不再崩"（Kuzu 上游已停止维护）——
   崩溃期间的已接收活动仍会丢，见台账 §5 开放项。
9. **记账只有一条路径**（台账 N-21）— 动作计数、`<platform>/actions.jsonl`、图谱回写输入
   三者必须来自同一个实现：`backend/scripts/action_recorder.py`。三个
   `run_*_simulation.py` 共用它，脚本里**不允许再内联第二份**（`tests/test_action_recording_contracts.py`
   13 条锁死）。崩溃后的可见性同理：`run.py` 启动时 `reconcile_stale_runs()` 把
   "上次没跑完就死了"判成 `failed`，并向孤儿模拟进程发 `close_env`（N-25）。
10. **被 token 上限截断的半截文本不算内容**（台账 N-30）— 报告某一章撞上 `max_tokens` 后
    返回 122 字的半截文本，旧的 `_require_section_content()` 只查"非空"，于是它被当正文落盘，
    报告仍 `completed`、`empty_sections=[]`。现在 `LLMClient.chat()` 读 `finish_reason`：
    `length` 或其它非 `stop` 终止都抛 `LLMResponseError`（`chat_json()` 一直是这条规则）。
    后果是截断会变成**该章节失败、报告 `failed`**——要修就显式调大上限，
    不允许改回接受半截文本。
    上限本身也是显式配置（F11）：`Config.REPORT_SECTION_MAX_TOKENS`（默认 32768）取代写死的
    4096，两个调用点（ReACT 循环、强制收尾）都读它；截断错误里带 `completion_tokens` /
    `reasoning_tokens`，所以"该调多大"是数字不是猜。**为什么 4096 一定会撞**：本端点
    （`deepseek-flash`）返回 `reasoning_content`，推理 token 计入 `completion_tokens`——
    `.tools/llm-output-limit-probe.py` 实测一句"只回答两个字"用掉 54 个推理 token；
    4096 是"推理 + 正文"的总预算，正文因此只剩很少。
11. **报告任务也必须被启动和解**（台账 N-32、N-33）— 后端死在报告生成期间时，
    `reports/<id>/meta.json` 停在 `planning`、`progress.json` 停在 `generating`，之后任何查询
    都看到"还在生成"（实测 `report_7594e1e6716d` 48%、`report_5b1a806e9dbc` 34%）。
    现在 `ReportManager.reconcile_stale_reports()` 挂在入口 `run.py`（与 N-25 同处，**不挂**
    `create_app`）：非终态 → 判 `failed` 并写明退出时刻的两份状态，终态与已生成章节不动；
    `meta/progress/outline.json` 改成原子写（临时文件 + `os.replace`），读不动的 meta 只记
    ERROR 并跳过，不猜状态也不改写。同批修的还有查询侧的隐性缺陷 N-33：
    `get_report_by_simulation()` 以前返回 `os.listdir` 里第一个匹配的报告，实测把停在
    `planning` 的失败报告排在 `completed` 前面（于是 interview 永久不解锁、缓存复用判不出来）；
    现在的规则是**已完成优先、其次最新**，一份完成的都没有就照实返回最新的失败报告。
12. **质量基线现场的四条**（台账 N-34/N-35/N-38/N-39，批次 F12，`模拟质量基线.md`）—
    ① **图谱实体清点**（`app/services/graph_inventory.py`）：同一张图谱、同一批节点（53 个），
    "带类型实体"实测从 32 掉到 2（节点还在、标签读不到），prepare 于是只生成 2 份人设跑完整场，
    没有任何一层报出来。现在按 `graph_id` 记**高水位**，骤降（< 一半且高水位 ≥ 4）即失败并写明
    "节点还在而类型没了"。② **twitter 时间线**：动作菜单缺 `REFRESH`（reddit 一直在用），
    OASIS 里"刷新时间线"正是它，缺了就只剩广播——补上后实测 `refresh=2 / follow=2 / like=1 /
    quote=1`（此前 0 赞 0 转发 0 关注）；`simulation_end` 事件新增 `information_flow` 块
    （各表行数 + trace 动作分布 + `has_timeline`），把"这个平台有没有时间线"写进产物。
    **残余**：OASIS 默认 twitter 平台用 `recsys_type="twhin-bert"`，其更新时间线失败被 OASIS
    自己 `except: return` 吞掉，`rec` 恒为 0——当时记为"换不换 recsys 待定"；后续 F17/F18 已
    定为**保留 twhin**（见 §1.2-16/§1.2-17），不再有换 recsys 的选项。③ **跑完但没人 stop 的运行**
    （N-38）：模拟 10 轮全跑完、产物完好，重启后端却被启动和解判 `failed`，报告从此被拒。
    现在和解增加**磁盘判据**（`<platform>/actions.jsonl` 里有没有 `simulation_end`）：
    已跑完且回写无待处理 → `STOPPED`，未跑完才 `FAILED`。④ **人设必需字段缺失**（N-39）：
    35 个实体里 1 个缺 `age/mbti/country` 就让整场 prepare 报废。现在必需字段集合
    （`bio/persona/age/gender/mbti/country`，**不含**确定性合成的 `karma/...`）在重试循环内校验，
    缺字段 → 把缺的字段名写回提示词后**重试同一条路径**（有界 3 次），仍缺才失败。

13. **报告正文里不得残留工具调用标记**（台账 N-40）— 实测
    `report_69379f275a5f/section_03.md`：模型写完两句正文后又接了一段自己拼装的
    工具调用标记，报告仍 `completed`。真因是 `_parse_tool_calls()` 只认提示词教的
    `<tool_call>{…}</tool_call>`，而端点原生吐的是全角竖线包裹的 `DSML` 方言
    （还把提示词的 `name`/`parameters` 命名混了进去）→ 解析为空 →
    "工具调用 + Final Answer 冲突"的检测分支**从来没被执行过** → `split("Final Answer:")[-1]`
    把残骸切下来当正文，唯一校验点只查"非空"就放行。
    现在：① `_parse_dsml_tool_calls()` 认这套方言的两种形态，`parameters`
    包装层展开，结果仍过 `_is_valid_tool_call`；② `<tool_call>` 块内改用
    `raw_decode`，容忍 JSON 后的杂散闭合标签；③ `_require_section_content()` 与
    `save_section()` 共用判据 `_protocol_markers_in()`：**命中标记即抛
    `SectionGenerationError`**，带上章节名、命中标记与原始响应长度；④ 冲突降级
    的截断改走 `_first_tool_call_end()`，不再写死 `</tool_call>`。
    **不做清洗**——清洗是兜底，它只把残骸换成干净正文，"解析器漏了一次调用"
    仍然不可见。要修就修解析器，不允许改回接受残骸。
    实测（`.tools/verify-dsml-parsing.py`，24 条真实 `llm_response`）：修前解析出
    **0** 条，修后 **23/23**；`section_03.md` 的真实内容被守卫拒绝。
    补充实测（F14 期间）：两次完整报告生成（`report_f8c582256603`、`report_d9ab9406b10c`）
    的成稿与 5 个章节都是 0 处协议标记。

14. **报告必须声明"本次模拟跑了什么"，取不到声明就不出报告**（台账 N-41；R3）—
    报告链取证只走图谱，从不读模拟产物，于是"只有 2 个角色""twitter 根本没有传播"
    这类事实在正文与 `meta.json` 里都不出现，读者会把小范围推演当成场景全貌。
    现在 `app/services/simulation_scope.py` 是**唯一采集点**：只读
    `state.json` / `run_state.json` / `<platform>/actions.jsonl` 的 `simulation_start`
    与 `simulation_end.information_flow`，缺任何一项抛
    `SimulationScopeUnavailable`、**不给默认值**；"人设数"与"实际参与数"不一致也抛错
    （实测 `sim_4267208ff665`：21 份人设只跑了 17 个 agent）。
    `generate_report()` 在组装前取声明，取不到即报告 `failed`；
    `assemble_full_report(..., scope_markdown=)` 是**必填参数**，声明块用占位行穿过
    `_post_process_report`（否则它的 `##` 会被降级成粗体）；`Report.simulation_scope`
    进 `meta.json`，`/generate` 完成结果也带。无传播的平台必须写一句
    "涉及传播的结论不适用于 X"，未启用的平台写"未启用"——**沉默必须被显式化**。
    测试 18 条（`tests/test_report_scope_contracts.py`），实跑 `report_f8c582256603`
    230.7 秒完成、声明块逐项等于产物。

15. **运行目录不许被测试写，类型标签不许被写没**（台账 N-42/N-34）—
    ① 运行目录是**导入期常量**且分散在五个所有者手里（`Config.UPLOAD_FOLDER` /
    `OASIS_SIMULATION_DATA_DIR`、`ReportManager.REPORTS_DIR`、
    `SimulationManager.SIMULATION_DATA_DIR`、`SimulationRunner.RUN_STATE_DIR`、
    `zep_graph_memory_updater.PENDING_DIR`），测试各自打补丁、漏一个就写真实目录
    （实测 `uploads/graphs/pending/graph-1.jsonl` 一次全量 pytest 涨 41,784 字节）。
    现在由 `backend/tests/conftest.py` **会话级统一**钉到项目内临时根，
    `tests/test_run_dir_isolation_contracts.py` 8 条锁住。
    ② `graphiti_patch.retain_node_labels()`：写库（`add_nodes_and_edges_bulk_tx`
    与 `EntityNode.save`）之前对**已存在节点**做类型并集——`labels` 只增不减；
    每次"库里有类型而本次传入没有/不同"都写 ERROR 日志与台账
    `uploads/_diagnostics/node-label-mutations.jsonl`。为什么不是直接失败：
    `labels` 的语义本来就是单调累积，库才是"这个节点是什么"的权威；
    需要被看见的是"抽取质量退化了"，不是丢掉整条 episode。
    ③ N-34 的三个候选机理已**实测排除**：跨进程共享库被 Kuzu 文件锁拒绝、
    FTS 索引每次重建不影响标签、`labels(n)` 不是第二份存储。

16. **twitter 的推荐时间线必须真的建起来，建不起来就是失败**（台账 N-46，批次 F17）—
    twitter 平台用 OASIS 的 `recsys_type="twhin-bert"`，它的 `rec` 表就是"agent 能不能看到
    别人的帖子"。实测**18 个模拟产物里 twitter 的 `rec` 全是 0 行**（reddit 全部非空），
    于是 35 个 agent × 5 轮只有 33 条 `CREATE_POST` + 1 条 `DO_NOTHING`、0 赞 0 关注——
    单向广播。真因有四层，逐层实测：
    ① transformers 的 tokenizer 加载只要 `vocab_size > 100000` **且入参不是本地目录**，
    就会调 `_patch_mistral_regex → is_base_mistral → huggingface_hub.model_info()`
    （`tokenization_utils_base.py:2390-2438`），而 OASIS 传的是**仓库名**
    `"Twitter/twhin-bert-base"`；
    ② 启动脚本为了图谱 embedding 不联网，设了 `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`
    （`启动MiroFish-本地图谱.cmd:71-72`）→ 那次 Hub 请求必然抛 `OfflineModeIsEnabled`
    （实测 0.81 秒失败；同一份库去掉两个开关 → 21.8 秒成功、`rec` 0 → 70 行）；
    ③ OASIS 把它吞在 `Platform.update_rec_table()` 的 `except: twitter_log.error(e); return`
    里（`platform.py:372-375`）；
    ④ 并行脚本的 `disable_oasis_logging()` 又把 `social.rec`/`social.twitter` 压成 CRITICAL、
    摘掉 handler、关掉传播——那条 ERROR 一个字都没落盘（同目录 4 个 OASIS 日志全是 0 字节）。
    现在：`backend/scripts/local_models.py` 拥有"项目自有模型目录 + 清单 + 校验"
    （`<项目>/models/<name>/` + `model.json`：repo、revision、每个文件的字节数与 sha256；
    缺目录 / 缺清单 / 缺键 / 缺文件 / 字节数不符 / sha256 不符 / 清单外多出权重
    → 一律抛错，**不下载、不回退**）；`backend/scripts/twitter_recsys_model.py` 在其上加载：
    `pin_offline_runtime()` 只设 `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`
    （**不再设 `HF_HOME`**——加载位置不由环境决定），`load_and_inject()` 先按清单全量校验
    （1.08 GB 实测 1.3 秒）再按**目录**加载（`os.path.isdir()` 为真 → 走 transformers 的
    `_is_local` 分支 → 不请求 Hub）并注入 `oasis.social_platform.recsys`；
    两个建 twitter 环境的脚本在 `import oasis` **之前** `pin_offline_runtime()`、
    建环境**之前** `load_and_inject()`（加载失败 = 整轮不启动）；
    并行脚本不再压 `social.rec`/`social.twitter` 的 ERROR（**压掉逐条日志可以，压掉失败证据不行**）；
    `ActionRecorder.require_timeline()` 给运行加判据：真的 step 过一轮、库里有帖子而 `rec`
    仍为空 → 抛 `TimelineNotBuilt`，运行失败——"这轮很安静"与"平台根本没在跑"从此分得开。
    **F18 顺带把模型搬出 HuggingFace 的 hub 缓存布局**（那是 hub 的内部约定：加载要读
    `refs/main`，同一个 1.12 GB 权重存两份、实测 2.16 GB，还混着 `.no_exist/`、`tmp*` 残留）：
    供应改由 `scripts/provision-models.py`（`--fetch` / `--write-manifest` / `--verify`，
    表即版本锁；hf-mirror 是当前网络前提，写在 `DEPLOYMENT.md` / `PROJECT_SETUP.md` 里）。
    测试 13 + 10 条（`tests/test_twitter_recsys_contracts.py`、
    `tests/test_local_models_contracts.py`）。

17. **推荐模型必须在 GPU 上跑，而且要有"别再打死机器"的闸**（台账 N-47，批次 F19）—
    twitter 推荐表每一轮都重算**全量语料**（画像 + 当时所有帖子，最多 4000 条）。
    CPU 上实测 92 条 9.09 s、400 条 41.40 s（≈100 ms/条），168 轮累计**小时级**，
    并且会把桌面饿死（实测 `DefaultCPUAllocator: not enough memory: 1,380,725,352 bytes`）。
    三件事一起做：
    ① **引擎**：项目 venv 里的 torch 是从镜像解析来的 **CPU 版**（`torch.version.cuda = None`），
    所以 `torch.cuda.is_available()` 恒为 False——**OASIS 自己不碰 GPU，它只读
    `torch.device("cuda" if torch.cuda.is_available() else "cpu")`（`recsys.py:46/85`），
    用不用卡完全由装的是哪个构建决定**。现在 `pyproject.toml` 显式
    `torch==2.9.1` + `[[tool.uv.index]] pytorch-cu128`（`explicit = true`），
    `scripts/provision-torch.py` 按 SHA256 校验后**离线**装入（`--install` / `--verify`，
    用项目自带的 uv——uv 建的 venv 里没有 pip）。
    ② **设备**：OASIS 自带的加载器有 `.to(device)`（`recsys.py:78-79`），但它只在
    `twhin_model is None` 时自己加载；我们的 `load_and_inject()` 已经先注入了，
    **那条 `.to(device)` 永远不会执行**——所以必须在注入前自己把模型搬到设备上，
    并把设备描述返回给调用方（运行日志里要能看出这一轮在哪算）。加载/显存失败
    一律包成 `RecsysModelUnavailable`（保留 `__cause__`），**整轮不启动**。
    ③ **闸**：8 GB 卡的硬边界实测踩到**三次**（`GPU is lost`，只能重启恢复），三次都在测量工具里——
    一次是池化探针语料 4035 条（连开两个 1.08 GB 模型），一次是同一探针换批大小，
    一次是我明知估算超预算仍**跳过粗筛**去试。
    现在 `measure-recsys-round-cost.py` 走生产的**分块入口**
    `generate_post_vector(batch_size=...)`（绕过它等于让 `--batch-size` 失效），并有三道闸：
    `--max-padded-tokens`、`--vram-budget-mb`（跑之前按**实测系数 35 MB/批单位**
    × 这一批实际条数）、`--vram-ceiling-mb`（实测 `max_memory_allocated()` 超限即中止
    并把已测点落盘），外加 `--pooler-max-items`（探针语料硬上限 1000）。
    **不允许跳过粗筛**：`--skip-vram-precheck` 试过一次就掉卡，已移除、传了直接报错。
    **实测**：整轮 `update_rec_table()` **6.46 s → 0.94 s**；每轮编码 40/92/200/400 条
    = **0.20 / 0.57 / 1.34 / 3.33 s**（CPU 3.47 / 9.09 / — / 13.72），显存峰值 1648 MB
    且与语料条数无关；启动器缺 GPU 即拒绝启动（`MIROFISH_ALLOW_CPU_RECSYS=1` 显式放行）；
    测试 9 条（`tests/test_recsys_device_contracts.py`），**全量 429 passed**。
    **生产批大小装不下 → 已修（N-48，批次 F20）**：OASIS 把 `batch_size=1000` 写死在
    `recsys.py:533-536`，在**4035 条语料**上实测 `torch.OutOfMemoryError：13.69 GiB allocated`
    （8 GB 卡要的是它 1.7 倍）。修法是**运行期包一层**（不改 vendored 包，重装 camel-oasis
    不会丢）：`load_and_inject()` 注入时把 `oasis.social_platform.recsys.generate_post_vector`
    换成守卫，策略是 **B 打底 + A 兜底**——
    B：按 `mem_get_info()` 可用显存 × 模型结构算每批单位显存，算安全批；
    **算出来 < 1 就抛错、整轮不启动**；每轮再按真实语料长度重算，且**只许变小**；
    A：读不到结构/显存或跑在 CPU 上时用实测安全档 64，绝不放行 1000。
    实测（真 GPU、真模型、真库副本）：92 条 → 请求 1000 实际 **47**（0.82 s）；
    **4040 条（画像+4000 帖子上限，真正最坏情况）→ 实际 76，36.84 s，无 OOM**；
    同规模无守卫时是 13.69 GiB OOM。批大小与依据都打进运行日志。
    测试 15 条（`tests/test_recsys_batch_guard_contracts.py`，含"vendored 源码不许被改"）。
    **排序信号这一层 → 已修（N-47，批次 F21）**：原判断"相似度被
    `enable_like_score=False` 整段跳过、排序只看新旧"**是读错缩进**——`recsys.py:583` 的
    `cosine × scores[...]` 与 `:571` 的 `if enable_like_score:` **同为 8 空格层级**（在 `if`
    **外面**），`enable_like_score` 只控制"喜欢过的帖子"的加权；而且 `date_score` 用
    `current_time - created_at` 算、帖子是当轮创建的（实测本库 7 条候选**全部 1.0109**），
    所以新旧分数≈常量，**余弦是唯一的排序信号**。真缺陷是读出来自随机 pooler
    （checkpoint 的 215 个张量里没有一个 `pooler.*`）：同一条文本两次加载余弦 **0.0379**
    （CPU）/ **0.0096**（GPU），纯余弦 top-2 只有 **16/40** 一致。
    现在 `_install_embedding_patch()` 在注入时把 `process_recsys_posts.process_batch`
    换成**确定性均值池化**（无参数、padding 按 mask 排除），**装不上就拒绝启动**；
    实测同一文本两次加载余弦 **1.0000**、top-2 **40/40**，真实 `rec` 表两个进程
    **40/40 逐用户相同**（指纹一致），兴趣跨度 0.0495 vs 随机读出的 0.0204；
    `update_rec_table` 0.77–0.84 s、**4040 条最坏规模 39.0 s 无 OOM**。
    测试 10 条（`tests/test_recsys_embedding_contracts.py`）。
    **残余（未修，是另一层）**：均值池化的各向异性 + 每轮只有 2 个槽位，让 **35/40** 用户拿到同一对帖子（7 条候选口径）——
    "内容信号真的参与"之后暴露的个性化质量问题，受控探针（2026-10-24，`.tools/probe-recsys-personalization.py`）显示撞车率随候选数下降
    （7/13/26/42 条候选最大组合 14/8/7/5 人）、用户间排序相关中位 0.41–0.52、置换 bio 后
    31/40 用户 top-2 改变——即"头部集中 + top-2 边际太小"。**模型去留已定（2026-10-24）：
    保留 twhin**；`random` 不是降级选项（曝光分布是模拟语义的一部分，代码里也没有第二条
    路径）。改法（白化/去主成分/去重/加槽位）待测。

18. **终态的回写账目必须是"排空之后"的数字**（台账 N-49，批次 F23）—
    F22 实测：运行终态 `stopped`、`error=null`、`failed_count=0`，图库副本 Episodic +19
    证明 93 条活动全部写入，但 `/run-status/detail` 仍报 `ingestion_drained=false`、
    `undelivered_items=83`、`delivery_ratio=0.1075`——把"已交付"读成"丢了 83 条"。
    真因：finalize 在 `stop_updater()`（排空）**之前**拍快照，之后不刷新；同步 stop 路径
    干脆没拍。现在唯一屏障 `SimulationRunner._drain_writeback()` 被两处共用，顺序固定：
    STOPPING 快照（排空前，N-26 可见性）→ 排空 → **同一个 updater 刷新 summary** → 写终态；
    排空失败同样刷新（显示未交付）并保留原始异常为 `cause`；刷新失败即运行 FAILED
    （不许沿用旧快照）。契约测试 5 条（`tests/test_writeback_final_snapshot_contracts.py`），
    真实复跑 `sim_20808935737b`：`ingestion_drained=true`、`items_sent=99/99`、
    `undelivered_items=0`、`delivery_ratio=1.0`、`batches_sent=20`，图库副本 Episodic +20。

### 1.3 允许的"显式降级"

一处降级只有在**同时**满足下列条件时才允许存在：

1. 来自书面要求（配置项 / 用户明确指定的行为），不是实现者的临场决定；
2. 只出现在一个具名位置，函数名或类名里写明它的边界；
3. 保留原始失败（`cause` 或 ERROR 日志含原始异常）；
4. 在 API 返回或运行状态里可见，而不是只写进日志。

不满足这四条的一律视为缺陷。

---

## 2. 先小样，后大样

规模化验证只有在链路**已被小样证明**之后才有意义。用一小时的长跑代替小样验证，
既得不出结论，又会把"慢"和"错"混在一起。

| 级别 | 规模 | 时长上限 | 通过标准 |
|---|---|---|---|
| L-a 冒烟 | ≤3 agent × ≤3 轮 × 单平台，且**必须覆盖至少一个活跃时段** | 2 min | 每一步有明确的、可证伪的数量/内容断言；不接受"没报错"，也不接受"0 动作跑完" |
| L-b 组件 | ≤10 agent × ≤10 轮 | 10 min | 本组件的输入→输出账目对得上（写入条数 = 读出条数等） |
| L-c 全链路 | 默认轮数 + 回写关闭 | 30 min | 五个阶段都有产物，且报告结论能追到具体事实 |
| L-d 全量 | 全轮数 + 回写开启 | 无上限但要说明预算 | 只在 L-a…L-c 全绿后跑；必须记录动作数 vs 入库数的交付率 |

规则：

- **任何一次"跑通验证"先跑 L-a**；改动越大越要先跑。
- 阈值、检索、聚类这类"结果质量"参数的改动，必须附**相关性抽检**（命中率或人工判定），
  不接受"有返回结果"作为通过标准。
- 长跑前先估算动作数与回写交付率，明确说出预期时长；预期 > 15 min 时先问一句要不要跑。
- 每次验证都要写清：跑了什么规模、判定标准是什么、结果数字是多少。

---

## 3. 失败的处理方式

每个失败点都要能回答三个问题：

1. **在哪一层暴露？**（抛出点 = 最靠近失败的那一层）
2. **调用方如何知道？**（异常类型 / 返回状态 / 运行状态字段，不能只在日志里 warn）
3. **数据在哪？**（要么已写入，要么已落盘，要么明确不存在）

## 4. 验证纪律

- 代码改动必须带测试；"禁止兜底"相关行为在 `backend/tests/test_no_fallback_contracts.py`
  里锁死（阈值默认值、单一检索路径、饱和告警、落盘不丢数据、工作线程故障可见、
  边界不得补默认值、FTS 索引每次重建），改动这两类行为必须同时改测试。
  本轮另外新增 21 个契约测试文件（记账 13 / Kuzu 并发 6 / 崩溃和解 8 / detail 账目 3 /
  运行完成度 12 / LLM 截断 5 / 采访前置 4 / 空跑判定 10 / 观察性回归 / 报告和解 15 /
  图谱清点 10 / 人设重试 5 / 协议残骸 27 / 报告范围声明 18 / 标签写边界 8 /
  运行目录隔离 8 / twitter 推荐时间线 13 / 项目模型清单 9 / 推荐模型设备契约 9 /
  推荐表批大小守卫 15 / 推荐表句向量 10 / 终态回写账目 5 / 报告 grounding 19 /
  抽检工具 11 / 大纲输出预算 1），**全量 490 passed**（F26 后）。
  `test_writeback_final_snapshot_contracts.py` 锁的是 N-49 那条线：
  排空后账目必须刷新为 delivered（`undelivered=0`、`ratio=1.0`）、排空失败仍显示未交付并保留
  原始异常为 `cause`、STOPPING 快照先于排空、同步 stop 也持久化排空后 summary、
  没开回写不产生任何数字。
  `test_recsys_embedding_contracts.py` 锁的是 N-47 那条线：
  均值池化算法（padding 不参与平均、无梯度）、两次调用逐位相同、单独跑与混在批里一致、
  补丁真的装到 OASIS 上且重复注入不套娃、OASIS 缺 `process_batch` 就拒绝启动、
  策略字段显式并进两个运行脚本日志、`load_and_inject()` 必须装上它，
  以及"vendored 的 `process_recsys_posts.py` 仍是 `return outputs.pooler_output`"。
  `test_recsys_batch_guard_contracts.py` 锁的是 N-48 那条线：
  按可用显存算出的批必须来自公式、连 1 条都放不下就抛错不启动、
  算不出时用实测安全档 64、守卫只许把批变小、装不上就拒绝启动、
  以及"vendored 的 `recsys.py` 一个字节都没改"。
   `test_recsys_device_contracts.py` 锁的是"模型到底在哪个设备上跑"：
   设备解析无环境变量开关、`.to(device)` 必须发生在注入之前、
   设备描述要进两个调用方脚本的日志、加载失败不许注入半个模型、
   测量脚本不许再吃满 CPU/显存（含"必须走生产分块入口"这一条——实测绕过它曾把 8 GB 卡打死）。
  其中 `test_simulation_scripts_reference_only_defined_module_names` 用 symtable 递归查
  "当全局用、却既没导入也没定义"的名字——F18 现场就是少一个 `import time` 让整场模拟
  以 `NameError` 失败，而顺序/字符串级的源码断言全绿。
- 测试跑法与临时目录：`backend/tests/conftest.py` 会话级把全部运行目录钉到项目内临时根，
  `[tool.pytest.ini_options]` 的 basetemp 固定在 `backend/.pytest-tmp`
  （pytest 每次运行开始会清空它）。**跑测试不许写真实 `uploads/`**（N-42），
  `tests/test_run_dir_isolation_contracts.py` 锁这一条。
- 质量基线（`模拟质量基线.md` + `.tools/quality-metrics.py` + `.tools/quality-baseline-run.py`）：
  改模拟行为前后必须用同参数各跑一次基线，并把数字写进那份文档——"报告读起来像样"
  不是质量证据（基线第一版的种子响应指标就把 15/15 算成了 0/15，见 V-09）。
- 提交前用项目自己的解释器跑：`uv run pytest`（拒绝用系统 python）。
- 全量审计结果见 `兜底审计.md`，其中未修项必须保留在文档里，不允许"忘了就算修了"。
- **端到端验证按清单走**：`端到端验证清单.md`（F22 起）。L-a 关回写 → 同参数开回写 →
  图库副本前后对照；每项都要有判据/实测/证据路径。**没跑过的模式写"未跑"**，
  不许用单测或组件级实测代替整场结论。
- **"确定性合成 / 占位文案 / 待办"不算修复**：判定标准是"预期路径不存在时是否失败"，
  不是"这次跑出来的值稳不稳定"。
- **代理指标必须标注**：命中率、条数、测试数量这类指标要写明它代理的是什么，
  不许当作被代理的那件事的结论。
- **没跑过的模式不许写"零回归"**：只有单测覆盖就写"仅单测覆盖，未实跑"。
- 自己与自己相关的掩盖记录在 `掩盖台账.md`（M/N/V 三族）；它的"状态"列与本文档必须一致。

## 5. 未修项（截至 2026-10-24 20:55）

`兜底审计.md` 快照：P0 = 9 条，P1 = 32 条，P2 = 23 条。已修的见 §1.2 与审计 §D。
**本轮新增、尚未登记的缺陷统一记录在 `掩盖台账.md`（N 族）**，那里的状态列是唯一权威。

A 族（`兜底审计.md` 的 P0）与本轮报告链新缺陷，外加现场发现的 **N-16…N-50**（N-34…N-39 见 §1.2-12 与 `模拟质量基线.md`；N-40 见 §1.2-13；N-41 见 §1.2-14；N-42 见 §1.2-15；**N-46/N-47/N-48 见 §1.2-16/§1.2-17**；N-49/N-50 见台账，均已在 F23/F25 修掉）：

| # | 位置 | 问题 | 状态 |
|---|---|---|---|
| N-16 | `local_embedder.py` | 并发构造 embedding 模型竞态（4 线程 2/4 失败），旧代码吞成"备用人设" | **已修**：单例 + 锁，6/6 通过 |
| N-17 | `api/simulation.py::_check_simulation_prepared` | 固定要求 `reddit_profiles.json` → 单平台模拟永远 `not_started` | **已修**：按启用平台判断 |
| N-18 | `run_*_simulation.py` + runner | 短轮数全落在 off_peak → 0 动作却永远 `running` | **已修**：0 动作即失败（非零退出码） |
| N-19 | `simulation_runner.py` | 子进程 stdout 块缓冲 → `simulation.log` 0 字节，现场不可见 | **已修**：`PYTHONUNBUFFERED=1` |
| N-20 | `simulation_time_utils.py` | 凌晨激活数被乘法归零；模拟起点固定 00:00 | **已修**：下限 1 + 起点取 `peak_hours[0]` |
| N-21 | `run_twitter/reddit_simulation.py` | 单平台运行不写 `actions.jsonl` → 动作数 0、**无回写**、报告无内容 | **已修（F9）**：记账抽成唯一路径 `scripts/action_recorder.py`，三个脚本共用；实跑 7 条 / 6 条全部记账 |
| N-22 | `action_logger.py` | `round_end` 只写 `actions_count`，runner 读的是 `simulated_hours` → 模拟时间恒为 0；`simulation_start.total_rounds` 硬编码 ×2 | **已修（F9）**：字段对齐，实测 `twitter_simulated_hours=14` |
| N-23 | `run_*_simulation.py`（种子帖） | 取不到 poster Agent 时 `except: pass` / 只打警告 → 种子事件静默少一条 | **已修（F9）**：三处改为抛错 |
| N-24 | `graphiti_patch.py` + Kuzu 0.11.3 | 写回写图谱时 `kuzu.Connection.execute` **原生 access violation**：进程直接消失、`run_state` 停 `running`（M-05 的真死因） | **已修（F9）**：每次初始化重建 FTS 索引（`SHOW_INDEXES` → DROP → CREATE）；残余风险见台账 §5 |
| N-25 | `simulation_runner.py` + `run.py` | 崩溃后 `run_state` 永远 `running`、模拟子进程成孤儿，无任何和解 | **已修（F9）**：`reconcile_stale_runs()` 挂入口，两次真实崩溃现场都被判 `failed` |
| N-26 | `simulation_runner.py` / `run-status` | 模拟跑完后不进终态，"跑完等采访命令"与"卡住"不可区分（实测卡 611 秒）；stop 之后报告又没法采访（互斥）；终态后账目变 `null` | **已修（F10）**：`simulation_finished`/`graph_memory_enabled`/`ingestion_drained` 三个显式信号；终态快照持久化；报告门槛收敛到 `_report_readiness()`（**跑完+回写排空即可出报告，不必先 stop**） |
| N-30 | `llm_client.LLMClient.chat` + `report_agent` | 章节撞 `max_tokens` 后返回 **122 字半截文本**，只查"非空"就落盘 → 报告 `completed`、`empty_sections=[]`（比"空章节"更隐蔽） | **已修（F10）**：`finish_reason=length` 或非 `stop` 终止即抛 `LLMResponseError`；截断变成报告 `failed` 并指出章节 |
| N-31 | `zep_tools.interview_agents` + `simulation_ipc.check_env_alive` | 给已停止的模拟出报告时，每章采访都白等满 600 秒预算（实测一章 10 分钟）；而 `env_status.json` 在强杀后仍写 `alive`，光看它会漏判 | **已修（F10）**：新增唯一判据 `env_accepts_interviews()`（运行状态非终态 + 进程存在 + `env_status.json` alive）；实测同一报告从 20 分钟卡第 3 章 → **3.5 分钟完成** |
| N-32 | `report` 任务状态 + `run.py` | 后端重启后报告任务永远停在 `generating`（`meta.status=planning`、`progress 48%`），跟 N-25 是同一类陈旧状态 | **已修（F11）**：`ReportManager.reconcile_stale_reports()` 挂入口；`meta/progress/outline.json` 改原子写；重启后两份真实半成品报告被判 `failed`（实测） |
| N-33 | `report_agent.ReportManager.get_report_by_simulation` | 同一个模拟有多份报告目录时返回 `os.listdir` 里第一个匹配的——实测返回停在 `planning` 的失败报告，而两份 `completed` 就在旁边 → interview 永久不解锁、缓存复用判不出来、每次点生成白跑一遍 | **已修（F11）**：规则显式化为"已完成优先、其次最新"，一份完成的都没有就照实返回最新的失败报告；多份时记 INFO 列出候选与选中项 |
| N-27 | `zep_graph_memory_updater.get_stats` | `expected_items` 把 `skipped_count` 再扣一次 → `delivery_ratio=1.25`、`undelivered=-1`（**高估交付**，是我 F7 加的指标） | **已修（F9）**：`expected = total_activities` + 回归测试 |
| N-28 | `config.py` | `GRAPHITI_USE_NATIVE_QUERIES` 声明了却从未被读（改它等于没改） | **已修（F9）**：删除该配置项 + 单测 |
| N-29 | `api/simulation.py`（detail） | 回写更新器注销后读 `run_state.graph_id` → 必然 500，恰好挡住终态后的最终账目 | **已修（F9）**：改从模拟元数据取，取不到就写明；3 条契约测试 |
| N-01 | `report_agent.py:1513-1544`、`1557-1559`、`1713`、`2195-2196` | 空响应被当章节正文采纳并落盘，报告仍 `completed`/`error=null`（第 5 章空的真因，见台账 M-03 证据） | **已修（F1）** |
| N-02 | `api/report.py:92-102` | 报告把"回写完成"当前置条件，而回写比模拟慢约 30×→ 结构上几乎必然 409 | **已修（F7）**：409 带 `pending_backlog` 与 hint |
| N-03 | `api/report.py:103-117` | `STOPPED` 也算成功终态 → 可用不完整图谱出报告且不声明 | **已修（F7）**：`graph_completeness` 进启动响应与任务结果 |
| N-07 | `zep_graphiti_impl.py:745`、`822-834` | reranker 解析结果被丢弃（§1.2-3 的残留） | **已修（F2）** |
| N-06 | `zep_graphiti_impl.py:155,157` | 补丁返回值不检查，补丁没打上也继续跑 | **已修（F2）** |
| N-08 | `zep_graphiti_impl._to_graph_edge`（真因 = N-14） | 本地图谱 `panorama_search` 报"0 条有效 / 68 条历史"，实为 `str(None)` == `"None"` 让每条边都判为历史 | **已修（F4）** |
| N-41 | `report_agent` + 报告产物 | 报告与 `meta.json` 都不声明"本次模拟有多少 agent、多少条动作、哪个平台没有传播"；twitter 半场 `has_information_flow=false` 在报告里读不出来 | **已修（F14）**：`simulation_scope` 唯一采集点（缺项即抛错、参与人数矛盾也抛错）+ 正文声明块（占位行穿过后处理）+ `meta.simulation_scope`；18 条测试，实跑 `report_f8c582256603` |
| N-42 | 五个运行目录的导入期常量 + `GRAPHITI_DB_PATH` 默认值 | 一次 pytest 会写真实运行目录（实测 `uploads/graphs/pending/graph-1.jsonl` 涨 41,784 字节），并与后端共用同一份真实图谱库 | **已修（F16）**：`backend/tests/conftest.py` 会话级统一钉到项目内临时根 + 8 条隔离契约测试；隔离后连续 3 次全量测试该文件不变 |
| N-43 | 回写新增实体没有业务类型 | 145 个实体节点里 **105 个只有 `['Entity']`**（10-10 回写当天新增 85 个）：既不是角色也不可用，却计入节点总数 | **未修**：机理是候选（本体缓存冷/恢复失败 → `entity_types=None`），用 `.tools/replay-writeback-labels.py` 两臂复现后再修 |
| N-44 | 报告不区分"模拟事实"与"作者归纳" | 强度词/量化判断（"最整齐的一次""有一半""唯一"）在模拟产物里没有出处，读者分不清哪句是模拟事实 | **已修（F26，语义层）**：事实/推断块 + 硬校验 + 最多两次纠正；实跑 `report_e3a7f1a1c8bc`（4 章/20 条）**R1_fact = 8/8 = 100%**、unlabeled = 0、quote = 0；**R1_all 已转 legacy：40% 继续记录**（12 条标注推断仍被旧口径计 partial；是否作 N-44 门槛待研究，当前不作门槛）；4 章/20 条不可与 5 章/25 条严格直比 |
| N-45 | 报告把概括写进引号 | 引号里的句子不是模拟原话而是概括（"窗口期可能比现在看起来更长" vs 图谱里的"窗口约 24–36 个月"） | **已修（F26）**：`report_grounding` 逐字校验；翻译 `[译文]`、概括 `[转述]` 且去引号；最终报告 `quote_violations=0`。残余：首轮合规率低，依赖 1–2 次纠正 |
| N-50 | `backend/scripts/report_traceability_audit.py::_support` | 候选支撑只搜 `scope="edges"`，节点摘要里的支持不可见；第二轮 `S5-5` 的引语只在 `HR / 招聘负责人` 节点摘要里，工作表显示 0 条候选 | **已修（F25）**：edges+nodes+命中实体摘要+引语命中节点摘要；测试 11 条；同 seed 复算第一轮 52.0%、第二轮 48.0%（判定未变，S5-5 自动可核） |
| N-46 | OASIS twitter 推荐表（`recsys_type="twhin-bert"`）+ 两个建 twitter 环境的脚本 + `disable_oasis_logging()` | 18 个模拟产物里 twitter 的 `rec` **全部为 0** → agent 看不到别人的帖子，只剩单向广播（33 条发帖 / 0 赞 0 关注）。真因：transformers 在 `vocab_size > 250002` 且入参不是本地目录时会调 `model_info()` 请求 Hub，而离线开关让它必然失败，OASIS 又把它 `except: return` 吞掉，我们这层还把 `social.rec`/`social.twitter` 的 ERROR 一并静音 | **已修（F17 + F18）**：模型放**项目自有目录** `<项目>/models/twhin-bert-base/`（清单 `model.json` 固定 repo/revision/字节数/sha256，加载时全量校验，1.08 GB 实测 1.3 秒；F18 之前躺在 HF hub 布局里、双份占 2.16 GB），`load_and_inject()` 按目录加载并注入 OASIS（离线可用、缺文件或内容不符即抛且**整轮不启动**）、不再静音那两条 ERROR、`ActionRecorder.require_timeline()` 让"有帖子却没有推荐表"变成运行失败。实测 `sim_3a3e1b64c774`：`rec`=80 行、互动边 37（此前 0）；测试 13 + 10 条，全量 419 passed |
| N-49 | `simulation_runner.py`（`_drain_writeback`） | 终态回写账目曾是**排空前的快照**：排空成功后仍显示 `ingestion_drained=false`、`undelivered_items=83/93`、`delivery_ratio=0.1075`（F22 `sim_273040f9ce8f`），而图库副本 Episodic +19 证明 93 条全部写入 | **已修（F23）**：唯一屏障 `_drain_writeback()`（两处共用）= STOPPING 快照 → 排空 → 刷新 summary；契约测试 5 条；复跑 `sim_20808935737b`：`ingestion_drained=true`、`items_sent=99/99`、`undelivered=0`、`ratio=1.0`、图库 +20 |
| A4 | `zep_tools.py`（采访回复缺失分支） | 未拿到回复的 Agent 写入"（该平台未获得回复）"，仍计入 `interviewed_count`，占位文本还会被喂给 LLM 生成摘要 | **已修（F4）** |
| A5 | `run_parallel_simulation.py:471-475/506-512`（`run_twitter/reddit_simulation.py:331-333` 同源） | 用"请求数"冒充"成功数"（含 `response=None`） | **已修（F3）** |
| A6 | `zep_tools.py:1498-1503` | 选择受访 Agent 失败 → 取列表前 N 个并伪造 `reasoning` | **已修（F4）** |
| A7 | `report_agent.py:1247-1258` | 大纲规划失败 → 返回硬编码"未来预测报告"三章模板，API 仍成功 | **已修（F1）** |
| A8 | `simulation_config_generator.py:710-719` | 事件配置失败 → `initial_posts=[]`，没有种子事件的模拟照样跑完出报告 | **已修（F3）** |
| A9 | `frontend/src/components/Step5Interaction.vue:752` | 取不到目标 Agent 的回答时，拿字典里第一个（别人的）回答顶上并记为"agentReplied" | 未修（F7） |

另有 P1 里两处"会真丢数据"的项已在本轮修掉（审计 B15 工作线程吞异常、B17 先清空再发送）；
B16（超长活动截断）改为"落盘原文 + 计数 + ERROR"，**真正的按动作边界拆 episode 仍未做**。
`file_parser.py` 的 `errors='replace'` 与四层编码兜底链已修（F5）：改为显式编码优先级 +
严格解码，全部失败即抛 `TextDecodingError`（附尝试过的编码与开头字节）。

仍未开始 / 仍开放的项目集中在 `掩盖台账.md §5` 的"尚未开始 / 仍开放"表：
**N-34 的触发者**（F16 已把三个候选机理实测排除——跨进程共享库被 Kuzu 文件锁拒绝、
FTS 索引重建不影响标签、`labels(n)` 不是第二份存储；写边界已加类型并集守卫与台账，
但"10-10 22:36 那次 32 → 2"仍未复现，**不许当已修**）、
**N-43 回写新增实体没有业务类型**（145 个实体里 105 个只有 `['Entity']`；机理是候选，未复现）、
**N-46 的性能取舍 + N-47 的个性化残余（性能与排序信号都已收口；模型去留已定，个性化改法待定）**——
真因已修：twitter 已能建出推荐时间线；F21 之后推荐表的排序信号也真的来自内容、且确定性
可复现（同一文本两次加载余弦 1.0000、真实 `rec` 表两个进程 40/40 相同）。**模型去留已定
（2026-10-24）：保留 twhin**——`random` 不是降级选项，曝光分布是模拟语义的一部分；代码里
也没有第二条路径。**仍待定的是个性化质量**：受控探针（2026-10-24，
`.tools/probe-recsys-personalization.py`）显示最大组合随候选数 7/13/26/42 为 14/8/7/5 人、
用户间排序相关中位 0.41–0.52、置换 bio 后 31/40 用户 top-2 改变——即"头部集中 + top-2
边际太小"，不是"所有人都一样"；改法（白化/去主成分/去重/加槽位）待测后定。**F22 已补**：L-a 关回写与同参数开回写都跑通
（`端到端验证清单.md`：A 112 动作/rec 80/图库 0 增量；B 93 动作/rec 80/图库副本 Episodic +19、
排空 368.7 s；真实图库 mtime/大小/sha 未变）；**报告段已第三次实测（F26，2026-10-24：
`report_e3a7f1a1c8bc`，4 章/20 条，R1_fact = 100%（8/8）、R1_all（legacy）= 40%、
unlabeled = 0、quote_violations = 0；R2 = 0；R3/R4/R5 达标）**；**仍未测**：
默认轮数 L-c（10 轮）、L-d 全轮数、cloud。
**N-49 已修（F23）**：终态账目改为排空后的数字，复跑 `sim_20808935737b` 得
`ingestion_drained=true`、`items_sent=99/99`、`undelivered=0`、`ratio=1.0`、图库副本 +20。
**N-44 / N-45 报告侧的两条**（报告不区分"模拟事实"与"作者归纳"、概括被写进引号；
F26 语义层已修：事实/推断块 + 逐字 gate，`R1_fact = 100%`、`unlabeled = 0`、
`quote_violations = 0`；**R1_all 已转 legacy：40% 继续记录**，因为旧口径把 12 条标注推断也计入分母，
是否作为 N-44 门槛待研究；是否限制推断数量/改口径也待研究）、
**N-36/N-37**（平台参数是死配置；`max_rounds` 与实际轮数差 1）、
**N-24 残余①②③**（崩溃期间活动丢失、引擎级风险、崩溃期图谱无事务边界）、
**回写吞吐**（实测每批 5 条约 2–4 分钟，29 条排空约 17 分钟——L-c/L-d 前必须先算预算）、
**报告可追溯率**（本地指标，非上游；F26 已测第三轮：**R1_fact = 100%（8/8）**、R1_all（legacy）= 40%（8/20）、
R2 = 0、unlabeled = 0、quote_violations = 0；第一轮 52%/第二轮 48% 是 5 章/25 条口径；
N-44/N-45 语义层已修；R1_all 已转 legacy，是否作门槛待研究；
抽检工具 N-50 已修）、
**复现性**（两次同参数基线动作数 109+459 vs 52+296，差异来自 LLM 生成配置）、
N-13（合成指标未进 API）、B16（拆 episode）、
M-10（环境变量参数未进 Config）、M-12（启动器双起未查）、旧探针的停机前提、
V-01（检索相关性仍是代理指标）、V-04（容量结论样本量）、V-07（报告耗时无分解）、
前端改动未在浏览器实跑、cloud 未实跑。

修复批次与顺序见 `掩盖台账.md §5`：F1 报告可信链 → F2 reranker/补丁 →
F3 模拟有效性（A2/A8/A5）→ F4 采访与工具契约 → F5 输入边界 →
F6 验证工具与结论纪律 → F7 可观测与结构（含 B16）→ F8 来源可审计 →
F9 记账单路径 + 崩溃可见性 + 图库索引重建（N-21…N-29）→
F10 完成度信号 + 终态账目 + 报告门槛 + 截断即失败 + 采访前置检查（N-26、N-30、N-31）→
F11 报告任务和解 + 多份报告选择 + 章节上限显式化（N-32、N-33、N-30 后果）→
**F12 质量基线现场第一轮修复**（N-35 部分、N-38、N-39；守卫 = N-34，2026-10-11）→
**F13 报告正文不得残留工具调用标记**（N-40，先 A：残骸进正文即该章失败，不清洗；
2026-10-11）→
**F14 报告必须声明模拟范围与各平台传播**（N-41；R3 落地，2026-10-11）→
**F15 报告可追溯抽检**（R1/R2 第一次真测 + R3/R4/R5 同轮实测；工具
`.tools/report-traceability-audit.py`，人工判定记入 `报告可追溯抽检.md`，2026-10-11）→
**F16 运行目录隔离 + 类型标签写边界守卫 + N-34 机理排除**（N-42、N-34 部分、
N-43 登记；2026-10-11）→
**F17 twitter 推荐时间线**（N-46：本地快照目录加载 twhin、加载失败即不启动、
时间线缺失即运行失败；2026-10-11）→
**F18 模型目录迁移与清单校验**（N-46 实现改进 + M-09/N-12 的来源闭环：模型搬出 HF hub
缓存布局到 `<项目>/models/<name>/` + `model.json`（repo/revision/字节数/sha256），
`scripts/provision-models.py` 是唯一供应入口，启动器不再设 `HF_HOME`，`.hf-cache` 删除；
2026-10-11）→
**F19 推荐模型 GPU 加速 + 别再打死机器的闸**（N-47 性能层：CUDA 版 torch 离线装入并锁进
`uv.lock`、注入路径补 `.to(device)`、测量工具三道闸 + 禁止跳过粗筛；2026-10-24）→
**F20 推荐表批大小守卫**（N-48：OASIS 写死 `batch_size=1000` 在生产规模 OOM；
运行期包 `generate_post_vector`，B 按可用显存算 + A 实测安全档兜底，
算不出就不启动；2026-10-24）→
**F21 推荐表句向量确定性（N-47 收口）**（运行期把 `process_recsys_posts.process_batch`
换成确定性均值池化、装不上即不启动、两个运行脚本记录句向量策略、测量工具加
production/legacy 两臂与 `--dump-rec/--compare-rec`；同时改正台账/基线/本文件里
"排序只看新旧、向量不参与排序"的误判——`recsys.py:583` 在 `enable_like_score` 分支外，
且 `date_score` 实测是常量；2026-10-24）→
**F24 报告链第二次同口径复测（当前代码）**（`report_cdffe841ad98` / `sim_20808935737b`：
40 agent × 3 轮 × twitter，99 条动作，回写 drained；R1 = 48.0%（12/25，Agent 初判；
边候选口径 44.0%）、R2 = 0、R3/R4/R5 达标；抽检工具只搜边候选 → N-50；
N-44/N-45 仍未修；2026-10-24）→
**F25 抽检工具节点摘要**（N-50：edges+nodes+命中实体摘要+引语命中；测试 11 条；
同 seed 复算 52%/48% 不变；2026-10-24）→
**F26 报告事实/推断分离 + 引用保真**（N-44/N-45：`report_grounding` 硬校验 + 最多两次纠正；
`report_e3a7f1a1c8bc` 4 章/20 条，R1_fact = 100%、R1_all（legacy）= 40%、unlabeled = 0、
quote = 0、352.2 秒；4 章/20 条不可与 5 章/25 条严格直比；2026-10-24）。

## 6. 未测项（下一轮必须补的数字）

1. **twitter 推荐表每轮耗时 vs 帖子数**：**已测（F19）**——见 `模拟质量基线.md` §10.3。
   GPU（batch 30、走生产分块入口）：40 条 0.20 s / 92 条 0.57 s / 200 条 1.34 s /
   400 条 3.33 s；CPU 同规模 3.47 / 9.09 / —(外推) / 13.72 s。
   **仍未测**：曲线在 400 条以上的点（三次掉卡中断）。
2. **生产的批大小 → 已修（N-48，F20）**：OASIS 硬编码 `generate_post_vector(batch_size=1000)`，
   在**生产规模**（4035 条语料）实测 `torch.OutOfMemoryError：13.69 GiB allocated`。
   现在 `load_and_inject()` 在注入时把 `oasis.social_platform.recsys.generate_post_vector`
   运行期包一层（**不改 vendored 包**）：**B** 按可用显存与模型结构算安全批
   （算不出 → 抛错不启动；每轮按真实语料重算、只许变小），**A** 算不出时用实测安全档 64。
   实测：92 条 → 47（0.82 s）；**4040 条（最坏情况）→ 76（36.84 s）无 OOM**。
   批大小与依据进运行日志。细节见 §1.2-17 与 `模拟质量基线.md` §10.7。
3. **模型加载与常驻**：已实测"每次运行加载一次、进程内常驻、每轮只做推理"。
   GPU 上（页缓存冷/热）：组件臂冷启 14.6 s（含 CUDA 上下文），热启 2.1–2.5 s。
   跨运行复用进程**不做**（每次运行一个 platform/数据库/agent_graph；
   OASIS recsys 有模块级可变全局，跨运行会串味）。
4. **校验开销**：加载时全量 sha256 实测 1.3 秒（1.08 GB），已计入每次运行的固定成本。
5. **显存边界**：8 GB 卡的硬边界踩到三次（`GPU is lost`，只能重启恢复），三次都在测量工具里。
   工具现有三道闸，且**不允许跳过粗筛**——`--skip-vram-precheck` 试过一次就掉卡，
   已移除、传了直接报错。生产路径的闸见第 2 条（已补）。
6. **推荐表确定性（F21 已测）**：同一文本两次加载的向量余弦 **1.0000**（修前 0.0379 CPU /
   0.0096 GPU）；纯余弦 top-2 **40/40**（修前 16/40）；真实 `rec` 表两个进程
   **40/40 逐用户相同、指纹一致**（修前 legacy 臂两次运行指纹相同，但同进程强制换 pooler
   种子后只有 38/40——2/40 用户会被随机读出翻掉）；4040 条最坏规模 **39.0 s、无 OOM**。
   **仍未测**：曲线 400 条以上的点。
7. **端到端（F22/F23/F24/F26 已跑；报告段第三次实测）**：L-a 关回写 `sim_718677a7824b`
   （prepare 395 s、模拟 59 s、112 动作、post 45、rec 80、互动 67、图库副本 0 增量）；
   同参数开回写 `sim_273040f9ce8f`（prepare 378 s、模拟 50 s、93 动作、post 36、rec 80、
   互动 57、stop+排空 368.7 s、图库副本 Episodic 93→112、Entity +8、边 +277）；
   两次都 `information_flow=true`、日志三行齐全（设备 cuda:0 / 批大小 / 句向量）。
   **报告段（F24）**：`report_cdffe841ad98`（40 agent × 3 轮 × twitter，99 条动作）5/5 章节、
   246.6 秒、R1 = 48.0%（12/25，Agent 初判；边候选 44.0%）、R2 = 0、R3/R4/R5 达标、
   0 条与范围声明矛盾；**报告段（F26）**：`report_e3a7f1a1c8bc` 4 章/20 条、352.2 秒、
   **R1_fact = 100%（8/8）**、R1_all（legacy）= 40%、unlabeled = 0、quote = 0、R2 = 0、
   R3/R4/R5 达标；
   **仍未跑**：默认轮数 L-c（10 轮）、L-d、cloud。判据与证据见 `端到端验证清单.md`；
   N-49（终态账目）已在 F23 修掉并复跑验证（`sim_20808935737b`：`items_sent=99/99`、
   `undelivered=0`、`ratio=1.0`、`ingestion_drained=true`、图库副本 +20）。
