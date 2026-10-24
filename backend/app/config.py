"""\n配置管理\n统一从项目根目录的 .env 文件加载配置\n"""

import os
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
# 路径: MiroFish/.env (相对于 backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # 如果根目录没有 .env，尝试加载环境变量（用于生产环境）
    load_dotenv(override=True)


class Config:
    """Flask配置类"""
    
    # Flask配置
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # JSON配置 - 禁用ASCII转义，让中文直接显示
    JSON_AS_ASCII = False
    
    # LLM配置（统一使用OpenAI格式）
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gpt-4o-mini')
    
    # Zep配置
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY')

    # 图存储后端：cloud（Zep Cloud，默认）| graphiti（本地 Graphiti + 嵌入式图谱）
    ZEP_BACKEND = os.environ.get('ZEP_BACKEND', 'cloud').strip().lower()

    # ---- 本地图谱（ZEP_BACKEND=graphiti 时生效）----
    GRAPHITI_DB_PATH = os.environ.get(
        'GRAPHITI_DB_PATH',
        os.path.join(os.path.dirname(__file__), '../uploads/graphs/graphiti.kuzu'),
    )
    GRAPHITI_DB_PATH = os.path.abspath(GRAPHITI_DB_PATH)
    GRAPHITI_LLM_MODEL = os.environ.get('GRAPHITI_LLM_MODEL')
    GRAPHITI_LLM_SMALL_MODEL = os.environ.get('GRAPHITI_LLM_SMALL_MODEL')
    GRAPHITI_LLM_STRUCTURED_MODE = os.environ.get(
        'GRAPHITI_LLM_STRUCTURED_MODE', 'json_object'
    )
    GRAPHITI_LLM_MAX_TOKENS = int(os.environ.get('GRAPHITI_LLM_MAX_TOKENS', '8192'))
    GRAPHITI_EMBEDDING_MODEL = os.environ.get(
        'GRAPHITI_EMBEDDING_MODEL', 'BAAI/bge-small-zh-v1.5'
    )
    GRAPHITI_EMBEDDING_MODEL_PATH = os.environ.get(
        'GRAPHITI_EMBEDDING_MODEL_PATH',
        os.path.join(
            os.path.dirname(__file__), '../../models/bge-small-zh-v1.5'
        ),
    )
    GRAPHITI_EMBEDDING_MODEL_PATH = os.path.abspath(GRAPHITI_EMBEDDING_MODEL_PATH)
    GRAPHITI_EMBEDDING_DIM = int(os.environ.get('GRAPHITI_EMBEDDING_DIM', '512'))
    GRAPHITI_RERANKER = os.environ.get('GRAPHITI_RERANKER', 'rrf').strip().lower()
    # 检索相关性阈值（graphiti 的 sim_min_score）。
    # 未显式配置时为 None：沿用 graphiti 上游默认值（0.6），由实现层记录生效值。
    # 禁止用 0 关闭这道护栏——那会让任何查询都返回满额结果，"检索是否有效"
    # 从此不可证伪（历史事故，见 兜底审计.md）。
    _raw_graphiti_min_score = os.environ.get('GRAPHITI_MIN_SCORE', '').strip()
    GRAPHITI_MIN_SCORE = (
        float(_raw_graphiti_min_score) if _raw_graphiti_min_score else None
    )
    GRAPHITI_BULK_SLICE = int(os.environ.get('GRAPHITI_BULK_SLICE', '5'))
    # 本地图谱抽取补充指令。默认规则针对中文语料常见的"匿名主体"问题：
    # 种子常以「该机械工程师」「主角」「一名…」指代关键人物，若不显式要求，
    # Graphiti 的抽取会漏掉这些人物实体，导致人设阶段只能拿到公司/机构节点。
    GRAPHITI_EXTRACTION_INSTRUCTIONS = os.environ.get(
        'GRAPHITI_EXTRACTION_INSTRUCTIONS',
        (
            '抽取要求（务必遵守）：\n'
            '1. 文中以「该机械工程师」「该工程师」「主角」「一名…」等泛指或匿名方式指代的主体，'
            '必须为其创建实体；若类型列表中存在人物类型，归入最贴切的类型，'
            'name 使用文中最具体的称呼（例如「主角」或「该机械工程师」）。\n'
            '2. 直属主管、资深工程师、HR、猎头、招聘官、同行、行业分析师、社区网友等角色，'
            '即使没有姓名，也要按文中描述创建各自的人物实体。\n'
            '3. 每个实体的 summary 用中文，写清它在本文中的身份与可核查事实。\n'
            '4. 每条关系的 fact 用中文完整句子，包含主客体与关键时间/数字。'
        ),
    )
    GRAPHITI_PAGE_SIZE = int(os.environ.get('GRAPHITI_PAGE_SIZE', '500'))
    # 这里曾有一个 GRAPHITI_USE_NATIVE_QUERIES：声明了默认值、文档里也当作开关，
    # 但代码里从来没有读过它——一个"看起来能改行为、实际什么也不改"的配置项
    # （与 reranker 那次"校验通过但没生效"同类，见掩盖台账 N-28）。已删除。
    GRAPHITI_SEARCH_TIMEOUT = int(os.environ.get('GRAPHITI_SEARCH_TIMEOUT', '600'))

    # Kuzu 执行并发：**单连接 + 单线程**。
    # Kuzu 是嵌入式 C++ 引擎，graphiti 的 `AsyncConnection` 默认会建 N 个连接 + N 个线程，
    # 且 `max_threads_per_query=0` = 查询内部不限制并行。本机实测（faulthandler dump）：
    # 回写写图谱时 `kuzu.Connection.execute` 以 access violation 直接杀死整个后端进程，
    # 进程没有任何 Python 异常、atexit 也没跑（见 `掩盖台账.md` N-24）。
    # 图谱规模是几十~几百节点，串行执行的开销可以忽略；崩溃的代价是整次运行。
    # 调大这两个值等于重新打开那次崩溃的路径，所以 0（不限制）在这里是非法值。
    GRAPHITI_MAX_CONCURRENT_QUERIES = int(
        os.environ.get('GRAPHITI_MAX_CONCURRENT_QUERIES', '1')
    )
    GRAPHITI_MAX_THREADS_PER_QUERY = int(
        os.environ.get('GRAPHITI_MAX_THREADS_PER_QUERY', '1')
    )
    if GRAPHITI_MAX_CONCURRENT_QUERIES < 1:
        raise ValueError(
            f"GRAPHITI_MAX_CONCURRENT_QUERIES={GRAPHITI_MAX_CONCURRENT_QUERIES} 非法："
            "至少 1 条连接（N-24 记录的是并行执行崩溃，不接受 0）"
        )
    if GRAPHITI_MAX_THREADS_PER_QUERY < 1:
        raise ValueError(
            f"GRAPHITI_MAX_THREADS_PER_QUERY={GRAPHITI_MAX_THREADS_PER_QUERY} 非法："
            "0 在 Kuzu 里表示查询内部不限制并行线程，这正是 N-24 崩溃的触发条件"
        )

    # 采访（interview_agents）等待 OASIS 环境回话的预算。
    # 历史上这里是硬编码 180 秒：一次批量采访 = 受访 Agent 数 × 平台数 次 LLM 调用，
    # 实测单次调用可达 30–60 秒，于是"10 次调用"在结构上必然超时（实测报错
    # "等待命令响应超时 (180.0秒)"）。现在按工作量给预算，并把算出来的值写进日志。
    INTERVIEW_TIMEOUT_SECONDS = int(os.environ.get('INTERVIEW_TIMEOUT_SECONDS', '600'))
    INTERVIEW_TIMEOUT_PER_CALL_SECONDS = int(
        os.environ.get('INTERVIEW_TIMEOUT_PER_CALL_SECONDS', '90')
    )

    # 模拟活动回写的批大小（每个平台累积多少条活动合并成一个 episode）。
    # 与云端保持一致（5）：调大它只是把"回写慢"掩盖成"批次少"，同时让单个
    # episode 变长、抽取更慢、结构化输出更容易被截断。要调必须显式配置并说明理由。
    GRAPH_MEMORY_BATCH_SIZE = int(os.environ.get('GRAPH_MEMORY_BATCH_SIZE', '5'))
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}
    
    # 文本处理配置
    DEFAULT_CHUNK_SIZE = 500  # 默认切块大小
    DEFAULT_CHUNK_OVERLAP = 50  # 默认重叠大小
    
    # OASIS模拟配置
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')
    
    # OASIS平台可用动作配置
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]
    
    # Report Agent配置
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))

    # 章节生成的输出上限。此前的 4096 会把"推理 + 正文"一起算进去：本端点
    # （deepseek-flash）返回 `reasoning_content`，而它计入 `completion_tokens`——
    # 实测一句「只回答两个字」就烧掉 54 个推理 token（本地输出上限探针）；
    # 报告第 3 章因此在 4096 处被截成 122 字（掩盖台账 N-30）。
    # 取值依据：实测最长章节 11105 字 ≈ 万级 token，加上推理开销仍需余量；
    # 端点实测接受 8192 / 16384 / 65536。这只是上限，不生成就不计费。
    # 截断必须失败（N-30），任何情况下不允许改回"接受半截文本"。
    REPORT_SECTION_MAX_TOKENS = int(
        os.environ.get('REPORT_SECTION_MAX_TOKENS', '32768')
    )
    
    @classmethod
    def validate(cls) -> list[str]:
        """验证必要配置"""
        errors: list[str] = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY 未配置")

        backend = (cls.ZEP_BACKEND or 'cloud').strip().lower()
        if backend == 'cloud':
            if not cls.ZEP_API_KEY:
                errors.append("ZEP_API_KEY 未配置（ZEP_BACKEND=cloud 时需要）")
        elif backend == 'graphiti':
            if not cls.GRAPHITI_DB_PATH:
                errors.append("GRAPHITI_DB_PATH 未配置")
            if cls.GRAPHITI_MIN_SCORE is not None and not (
                0 < cls.GRAPHITI_MIN_SCORE <= 1
            ):
                errors.append(
                    "GRAPHITI_MIN_SCORE 必须在 (0, 1] 区间内；"
                    "置 0 会关闭相关性过滤，使检索结果不可验证"
                )
        else:
            errors.append(
                f"ZEP_BACKEND 取值非法：{cls.ZEP_BACKEND}（应为 cloud 或 graphiti）"
            )

        if os.environ.get("ZEP_API_URL"):
            errors.append("ZEP_API_URL 不受支持；MiroFish 仅连接 Zep Cloud")
        if cls.REPORT_SECTION_MAX_TOKENS < 1:
            errors.append(
                "REPORT_SECTION_MAX_TOKENS 必须为正整数：0 或负数会让章节生成"
                "在第一次请求就被截断"
            )
        if cls.DEBUG:
            import warnings
            warnings.warn("Flask DEBUG mode is enabled. Do not use in production.", RuntimeWarning)
        return errors
