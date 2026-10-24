"""
OASIS 双平台并行模拟预设脚本
同时运行Twitter和Reddit模拟，读取相同的配置文件

功能特性:
- 双平台（Twitter + Reddit）并行模拟
- 完成模拟后不立即关闭环境，进入等待命令模式
- 支持通过IPC接收Interview命令
- 支持单个Agent采访和批量采访
- 支持远程关闭环境命令

使用方式:
    python run_parallel_simulation.py --config simulation_config.json
    python run_parallel_simulation.py --config simulation_config.json --no-wait  # 完成后立即关闭
    python run_parallel_simulation.py --config simulation_config.json --twitter-only
    python run_parallel_simulation.py --config simulation_config.json --reddit-only

日志结构:
    sim_xxx/
    ├── twitter/
    │   └── actions.jsonl    # Twitter 平台动作日志
    ├── reddit/
    │   └── actions.jsonl    # Reddit 平台动作日志
    ├── simulation.log       # 主模拟进程日志
    └── run_state.json       # 运行状态（API 查询用）
"""

# ============================================================
# 解决 Windows 编码问题：在所有 import 之前设置 UTF-8 编码
# 这是为了修复 OASIS 第三方库读取文件时未指定编码的问题
# ============================================================
import sys
import os

if sys.platform == 'win32':
    # 设置 Python 默认 I/O 编码为 UTF-8
    # 这会影响所有未指定编码的 open() 调用
    os.environ.setdefault('PYTHONUTF8', '1')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    
    # 重新配置标准输出流为 UTF-8（解决控制台中文乱码）
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    
    # 强制设置默认编码（影响 open() 函数的默认编码）
    # 注意：这需要在 Python 启动时就设置，运行时设置可能不生效
    # 所以我们还需要 monkey-patch 内置的 open 函数
    import builtins
    _original_open = builtins.open
    
    def _utf8_open(file, mode='r', buffering=-1, encoding=None, errors=None, 
                   newline=None, closefd=True, opener=None):
        """
        包装 open() 函数，对于文本模式默认使用 UTF-8 编码
        这可以修复第三方库（如 OASIS）读取文件时未指定编码的问题
        """
        # 只对文本模式（非二进制）且未指定编码的情况设置默认编码
        if encoding is None and 'b' not in mode:
            encoding = 'utf-8'
        return _original_open(file, mode, buffering, encoding, errors, 
                              newline, closefd, opener)
    
    builtins.open = _utf8_open

import argparse
import asyncio
import json
import logging
import multiprocessing
import random
import signal
import sqlite3
import time
import warnings
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


# 全局变量：用于信号处理
_shutdown_event = None
_cleanup_done = False

# 添加 backend 目录到路径
# 脚本固定位于 backend/scripts/ 目录
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, '..'))
_project_root = os.path.abspath(os.path.join(_backend_dir, '..'))
sys.path.insert(0, _scripts_dir)
sys.path.insert(0, _backend_dir)

from simulation_time_utils import (  # noqa: E402 - 依赖上面的 sys.path 设置
    EmptySimulationRun,
    active_agent_target,
    assert_run_produced_actions,
    resolve_simulation_start_hour,
    simulated_clock,
)
from twitter_recsys_model import (  # noqa: E402 - 依赖上面的 sys.path 设置
    batch_policy,
    embedding_policy,
    load_and_inject,
    pin_offline_runtime,
)

# 运行期不联网必须在 `import oasis`（→ transformers）之前钉死：它们在导入期读这些开关。
# twitter 的推荐时间线用**项目自有目录** `<项目>/models/twhin-bert-base/`（清单固定内容，
# 见 twitter_recsys_model.py 与掩盖台账 N-46/F17/F18），不经过 HF 缓存布局。
pin_offline_runtime()

# 加载项目根目录的 .env 文件（包含 LLM_API_KEY 等配置）
from dotenv import load_dotenv
_env_file = os.path.join(_project_root, '.env')
if os.path.exists(_env_file):
    load_dotenv(_env_file)
    print(f"已加载环境配置: {_env_file}")
else:
    # 尝试加载 backend/.env
    _backend_env = os.path.join(_backend_dir, '.env')
    if os.path.exists(_backend_env):
        load_dotenv(_backend_env)
        print(f"已加载环境配置: {_backend_env}")


class MaxTokensWarningFilter(logging.Filter):
    """过滤掉 camel-ai 关于 max_tokens 的警告（我们故意不设置 max_tokens，让模型自行决定）"""
    
    def filter(self, record):
        # 过滤掉包含 max_tokens 警告的日志
        if "max_tokens" in record.getMessage() and "Invalid or missing" in record.getMessage():
            return False
        return True


# 在模块加载时立即添加过滤器，确保在 camel 代码执行前生效
logging.getLogger().addFilter(MaxTokensWarningFilter())


def disable_oasis_logging():
    """
    压掉 OASIS 的逐条日志（每个 agent 的观察和动作），但**不压掉它的错误**。

    `Platform.update_rec_table()` 把推荐表更新的异常吞在自己里面
    （`platform.py:372-375`：`except Exception as e: twitter_log.error(e); return`），
    那条 ERROR 是唯一能说明"推荐时间线为什么没建起来"的证据。把 `social.rec` /
    `social.twitter` 也设成 CRITICAL 并摘掉 handler，等于把 OASIS 的失败改成静默：
    实测 18 场模拟 twitter 的 `rec` 全为 0，而日志里一个字都没有（台账 N-35/N-46）。
    """
    # 逐条日志：压掉
    chatty_loggers = [
        "social.agent",
        "oasis.env",
        "table",
    ]

    for logger_name in chatty_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)  # 只记录严重错误
        logger.handlers.clear()
        logger.propagate = False

    # 失败证据：只留 ERROR，并让它传播到根 logger（→ stderr → simulation.log）
    for logger_name in ("social.rec", "social.twitter"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.handlers.clear()
        logger.propagate = True


def init_logging_for_simulation(simulation_dir: str):
    """
    初始化模拟的日志配置
    
    Args:
        simulation_dir: 模拟目录路径
    """
    # 禁用 OASIS 的详细日志
    disable_oasis_logging()
    
    # 清理旧的 log 目录（如果存在）
    old_log_dir = os.path.join(simulation_dir, "log")
    if os.path.exists(old_log_dir):
        import shutil
        shutil.rmtree(old_log_dir, ignore_errors=True)


from action_logger import SimulationLogManager
from action_recorder import ActionRecorder

try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    import oasis
    from oasis import (
        ActionType,
        LLMAction,
        ManualAction,
        generate_twitter_agent_graph,
        generate_reddit_agent_graph
    )
except ImportError as e:
    print(f"错误: 缺少依赖 {e}")
    print("请先安装: pip install oasis-ai camel-ai")
    sys.exit(1)


# Twitter可用动作（不包含INTERVIEW，INTERVIEW只能通过ManualAction手动触发）
# REFRESH 必须在内：OASIS 用它把推荐时间线（rec 表）拉进 agent 的 observation。
# 缺了它，agent 看不到别人的帖子，只能一直 CREATE_POST——实测 35 agent × 11 轮
# 只产出 103 条发帖、0 赞 0 转发 0 关注（台账 N-35）。
TWITTER_ACTIONS = [
    ActionType.CREATE_POST,
    ActionType.LIKE_POST,
    ActionType.REPOST,
    ActionType.FOLLOW,
    ActionType.DO_NOTHING,
    ActionType.QUOTE_POST,
    ActionType.REFRESH,
]

# Reddit可用动作（不包含INTERVIEW，INTERVIEW只能通过ManualAction手动触发）
REDDIT_ACTIONS = [
    ActionType.LIKE_POST,
    ActionType.DISLIKE_POST,
    ActionType.CREATE_POST,
    ActionType.CREATE_COMMENT,
    ActionType.LIKE_COMMENT,
    ActionType.DISLIKE_COMMENT,
    ActionType.SEARCH_POSTS,
    ActionType.SEARCH_USER,
    ActionType.TREND,
    ActionType.REFRESH,
    ActionType.DO_NOTHING,
    ActionType.FOLLOW,
    ActionType.MUTE,
]


# IPC相关常量
IPC_COMMANDS_DIR = "ipc_commands"
IPC_RESPONSES_DIR = "ipc_responses"
ENV_STATUS_FILE = "env_status.json"

class CommandType:
    """命令类型常量"""
    INTERVIEW = "interview"
    BATCH_INTERVIEW = "batch_interview"
    CLOSE_ENV = "close_env"


class ParallelIPCHandler:
    """
    双平台IPC命令处理器
    
    管理两个平台的环境，处理Interview命令
    """
    
    def __init__(
        self,
        simulation_dir: str,
        twitter_env=None,
        twitter_agent_graph=None,
        reddit_env=None,
        reddit_agent_graph=None
    ):
        self.simulation_dir = simulation_dir
        self.twitter_env = twitter_env
        self.twitter_agent_graph = twitter_agent_graph
        self.reddit_env = reddit_env
        self.reddit_agent_graph = reddit_agent_graph
        
        self.commands_dir = os.path.join(simulation_dir, IPC_COMMANDS_DIR)
        self.responses_dir = os.path.join(simulation_dir, IPC_RESPONSES_DIR)
        self.status_file = os.path.join(simulation_dir, ENV_STATUS_FILE)
        
        # 确保目录存在
        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)
    
    def update_status(self, status: str):
        """更新环境状态"""
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump({
                "status": status,
                "twitter_available": self.twitter_env is not None,
                "reddit_available": self.reddit_env is not None,
                "timestamp": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
    
    def poll_command(self) -> Optional[Dict[str, Any]]:
        """轮询获取待处理命令"""
        if not os.path.exists(self.commands_dir):
            return None
        
        # 获取命令文件（按时间排序）
        command_files = []
        for filename in os.listdir(self.commands_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.commands_dir, filename)
                command_files.append((filepath, os.path.getmtime(filepath)))
        
        command_files.sort(key=lambda x: x[1])
        
        for filepath, _ in command_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
        
        return None
    
    def send_response(self, command_id: str, status: str, result: Dict = None, error: str = None):
        """发送响应"""
        response = {
            "command_id": command_id,
            "status": status,
            "result": result,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        response_file = os.path.join(self.responses_dir, f"{command_id}.json")
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        
        # 删除命令文件
        command_file = os.path.join(self.commands_dir, f"{command_id}.json")
        try:
            os.remove(command_file)
        except OSError:
            pass
    
    def _get_env_and_graph(self, platform: str):
        """
        获取指定平台的环境和agent_graph
        
        Args:
            platform: 平台名称 ("twitter" 或 "reddit")
            
        Returns:
            (env, agent_graph, platform_name) 或 (None, None, None)
        """
        if platform == "twitter" and self.twitter_env:
            return self.twitter_env, self.twitter_agent_graph, "twitter"
        elif platform == "reddit" and self.reddit_env:
            return self.reddit_env, self.reddit_agent_graph, "reddit"
        else:
            return None, None, None
    
    async def _interview_single_platform(self, agent_id: int, prompt: str, platform: str) -> Dict[str, Any]:
        """
        在单个平台上执行Interview
        
        Returns:
            包含结果的字典，或包含error的字典
        """
        env, agent_graph, actual_platform = self._get_env_and_graph(platform)
        
        if not env or not agent_graph:
            return {"platform": platform, "error": f"{platform}平台不可用"}
        
        try:
            agent = agent_graph.get_agent(agent_id)
            interview_action = ManualAction(
                action_type=ActionType.INTERVIEW,
                action_args={"prompt": prompt}
            )
            actions = {agent: interview_action}
            await env.step(actions)
            
            result = self._get_interview_result(agent_id, actual_platform)
            result["platform"] = actual_platform
            return result
            
        except Exception as e:
            return {"platform": platform, "error": str(e)}
    
    async def handle_interview(self, command_id: str, agent_id: int, prompt: str, platform: str = None) -> bool:
        """
        处理单个Agent采访命令
        
        Args:
            command_id: 命令ID
            agent_id: Agent ID
            prompt: 采访问题
            platform: 指定平台（可选）
                - "twitter": 只采访Twitter平台
                - "reddit": 只采访Reddit平台
                - None/不指定: 同时采访两个平台，返回整合结果
            
        Returns:
            True 表示成功，False 表示失败
        """
        # 如果指定了平台，只采访该平台
        if platform in ("twitter", "reddit"):
            result = await self._interview_single_platform(agent_id, prompt, platform)
            
            if "error" in result:
                self.send_response(command_id, "failed", error=result["error"])
                print(f"  Interview失败: agent_id={agent_id}, platform={platform}, error={result['error']}")
                return False
            else:
                self.send_response(command_id, "completed", result=result)
                print(f"  Interview完成: agent_id={agent_id}, platform={platform}")
                return True
        
        # 未指定平台：同时采访两个平台
        if not self.twitter_env and not self.reddit_env:
            self.send_response(command_id, "failed", error="没有可用的模拟环境")
            return False
        
        results = {
            "agent_id": agent_id,
            "prompt": prompt,
            "platforms": {}
        }
        success_count = 0
        
        # 并行采访两个平台
        tasks = []
        platforms_to_interview = []
        
        if self.twitter_env:
            tasks.append(self._interview_single_platform(agent_id, prompt, "twitter"))
            platforms_to_interview.append("twitter")
        
        if self.reddit_env:
            tasks.append(self._interview_single_platform(agent_id, prompt, "reddit"))
            platforms_to_interview.append("reddit")
        
        # 并行执行
        platform_results = await asyncio.gather(*tasks)
        
        for platform_name, platform_result in zip(platforms_to_interview, platform_results):
            results["platforms"][platform_name] = platform_result
            if "error" not in platform_result:
                success_count += 1
        
        if success_count > 0:
            self.send_response(command_id, "completed", result=results)
            print(f"  Interview完成: agent_id={agent_id}, 成功平台数={success_count}/{len(platforms_to_interview)}")
            return True
        else:
            errors = [f"{p}: {r.get('error', '未知错误')}" for p, r in results["platforms"].items()]
            self.send_response(command_id, "failed", error="; ".join(errors))
            print(f"  Interview失败: agent_id={agent_id}, 所有平台都失败")
            return False
    
    async def handle_batch_interview(self, command_id: str, interviews: List[Dict], platform: str = None) -> bool:
        """
        处理批量采访命令
        
        Args:
            command_id: 命令ID
            interviews: [{"agent_id": int, "prompt": str, "platform": str(optional)}, ...]
            platform: 默认平台（可被每个interview项覆盖）
                - "twitter": 只采访Twitter平台
                - "reddit": 只采访Reddit平台
                - None/不指定: 每个Agent同时采访两个平台
        """
        # 按平台分组
        twitter_interviews = []
        reddit_interviews = []
        both_platforms_interviews = []  # 需要同时采访两个平台的
        
        for interview in interviews:
            item_platform = interview.get("platform", platform)
            if item_platform == "twitter":
                twitter_interviews.append(interview)
            elif item_platform == "reddit":
                reddit_interviews.append(interview)
            else:
                # 未指定平台：两个平台都采访
                both_platforms_interviews.append(interview)
        
        # 把 both_platforms_interviews 拆分到两个平台
        if both_platforms_interviews:
            if self.twitter_env:
                twitter_interviews.extend(both_platforms_interviews)
            if self.reddit_env:
                reddit_interviews.extend(both_platforms_interviews)
        
        results = {}
        # 记账口径：requested = 请求的 Agent 数；succeeded = 真的拿到回答的 Agent 数；
        # missing = 明确没拿到回答的（含环境未运行、Agent 取不到、回答为空）
        missing: Dict[str, List[int]] = {}
        unavailable: List[Dict[str, Any]] = []
        # 处理Twitter平台的采访
        if twitter_interviews and self.twitter_env:
            try:
                twitter_actions = {}
                for interview in twitter_interviews:
                    agent_id = interview.get("agent_id")
                    prompt = interview.get("prompt", "")
                    try:
                        agent = self.twitter_agent_graph.get_agent(agent_id)
                        twitter_actions[agent] = ManualAction(
                            action_type=ActionType.INTERVIEW,
                            action_args={"prompt": prompt}
                        )
                    except Exception as e:
                        print(f"  错误: 无法获取Twitter Agent {agent_id}: {e}")
                        unavailable.append({"platform": "twitter", "agent_id": agent_id, "error": str(e)})
                
                if twitter_actions:
                    await self.twitter_env.step(twitter_actions)

                    for interview in twitter_interviews:
                        agent_id = interview.get("agent_id")
                        result = self._get_interview_result(agent_id, "twitter")
                        result["platform"] = "twitter"
                        if not result.get("response"):
                            # 没有回答就不是成功：单独记账，不计入 succeeded
                            missing.setdefault("twitter", []).append(agent_id)
                            continue
                        results[f"twitter_{agent_id}"] = result
            except Exception as e:
                print(f"  错误: Twitter批量Interview失败: {e}")
                platform_error = str(e)
                for interview in twitter_interviews:
                    missing.setdefault("twitter", []).append(interview.get("agent_id"))
                unavailable.append(
                    {"platform": "twitter", "agent_id": None, "error": platform_error}
                )

        # 处理Reddit平台的采访
        if reddit_interviews and self.reddit_env:
            try:
                reddit_actions = {}
                for interview in reddit_interviews:
                    agent_id = interview.get("agent_id")
                    prompt = interview.get("prompt", "")
                    try:
                        agent = self.reddit_agent_graph.get_agent(agent_id)
                        reddit_actions[agent] = ManualAction(
                            action_type=ActionType.INTERVIEW,
                            action_args={"prompt": prompt}
                        )
                    except Exception as e:
                        print(f"  错误: 无法获取Reddit Agent {agent_id}: {e}")
                        unavailable.append({"platform": "reddit", "agent_id": agent_id, "error": str(e)})
                
                if reddit_actions:
                    await self.reddit_env.step(reddit_actions)

                    for interview in reddit_interviews:
                        agent_id = interview.get("agent_id")
                        result = self._get_interview_result(agent_id, "reddit")
                        result["platform"] = "reddit"
                        if not result.get("response"):
                            missing.setdefault("reddit", []).append(agent_id)
                            continue
                        results[f"reddit_{agent_id}"] = result
            except Exception as e:
                print(f"  错误: Reddit批量Interview失败: {e}")
                for interview in reddit_interviews:
                    missing.setdefault("reddit", []).append(interview.get("agent_id"))
                unavailable.append(
                    {"platform": "reddit", "agent_id": None, "error": str(e)}
                )

        requested_count = len(interviews)
        succeeded_count = len(results)
        missing_count = sum(len(items) for items in missing.values())

        # 环境没在跑、或某个平台的 Agent 取不到时，请求数远大于成功数：
        # 这里必须把它说出来，不能让 len(results) 冒充成功数（见 AGENTS.md §1.1）
        if results:
            self.send_response(command_id, "completed", result={
                "requested_count": requested_count,
                "interviews_count": succeeded_count,
                "missing_count": missing_count,
                "missing_agents": missing,
                "unavailable": unavailable,
                "results": results,
            })
            print(
                f"  批量Interview完成: 成功 {succeeded_count}/{requested_count}，"
                f"未回答 {missing_count}，不可用 {len(unavailable)}"
            )
            return True

        self.send_response(command_id, "failed", result={
            "requested_count": requested_count,
            "interviews_count": 0,
            "missing_count": missing_count,
            "missing_agents": missing,
            "unavailable": unavailable,
        }, error=f"没有成功的采访（请求 {requested_count} 个，全部未回答或不可用）")
        return False
    
    def _get_interview_result(self, agent_id: int, platform: str) -> Dict[str, Any]:
        """从数据库获取最新的Interview结果。

        `response` 为 None 表示"这个 Agent 没有给出回答"——它**不是**成功。
        历史上调用方按请求数统计成功数，于是没回答的 Agent 也被算作"已采访"。
        """
        db_path = os.path.join(self.simulation_dir, f"{platform}_simulation.db")

        result = {
            "agent_id": agent_id,
            "response": None,
            "timestamp": None,
        }

        if not os.path.exists(db_path):
            print(
                f"  错误: Interview 结果无法读取（数据库不存在 {db_path}），"
                f"agent_id={agent_id} 记为未回答"
            )
            return result

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # 查询最新的Interview记录
            cursor.execute("""
                SELECT user_id, info, created_at
                FROM trace
                WHERE action = ? AND user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (ActionType.INTERVIEW.value, agent_id))

            row = cursor.fetchone()
            if row:
                user_id, info_json, created_at = row
                try:
                    info = json.loads(info_json) if info_json else {}
                    # 没有 response 字段时不要拿整个 info 当回答（那是元数据不是回答）
                    response = info.get("response") if isinstance(info, dict) else info
                    result["response"] = response
                    result["timestamp"] = created_at
                except json.JSONDecodeError:
                    result["response"] = info_json

            conn.close()

        except Exception as e:
            print(
                f"  错误: 读取 Interview 结果异常"
                f"（agent_id={agent_id}, platform={platform}）：{e}"
            )

        return result
    
    async def process_commands(self) -> bool:
        """
        处理所有待处理命令
        
        Returns:
            True 表示继续运行，False 表示应该退出
        """
        command = self.poll_command()
        if not command:
            return True
        
        command_id = command.get("command_id")
        command_type = command.get("command_type")
        args = command.get("args", {})
        
        print(f"\n收到IPC命令: {command_type}, id={command_id}")
        
        if command_type == CommandType.INTERVIEW:
            await self.handle_interview(
                command_id,
                args.get("agent_id", 0),
                args.get("prompt", ""),
                args.get("platform")
            )
            return True
            
        elif command_type == CommandType.BATCH_INTERVIEW:
            await self.handle_batch_interview(
                command_id,
                args.get("interviews", []),
                args.get("platform")
            )
            return True
            
        elif command_type == CommandType.CLOSE_ENV:
            print("收到关闭环境命令")
            self.send_response(command_id, "completed", result={"message": "环境即将关闭"})
            return False
        
        else:
            self.send_response(command_id, "failed", error=f"未知命令类型: {command_type}")
            return True


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)




def create_model(config: Dict[str, Any], use_boost: bool = False):
    """
    创建LLM模型
    
    支持双 LLM 配置，用于并行模拟时提速：
    - 通用配置：LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME
    - 加速配置（可选）：LLM_BOOST_API_KEY, LLM_BOOST_BASE_URL, LLM_BOOST_MODEL_NAME
    
    如果配置了加速 LLM，并行模拟时可以让不同平台使用不同的 API 服务商，提高并发能力。
    
    Args:
        config: 模拟配置字典
        use_boost: 是否使用加速 LLM 配置（如果可用）
    """
    # 检查是否有加速配置
    boost_api_key = os.environ.get("LLM_BOOST_API_KEY", "")
    boost_base_url = os.environ.get("LLM_BOOST_BASE_URL", "")
    boost_model = os.environ.get("LLM_BOOST_MODEL_NAME", "")
    has_boost_config = bool(boost_api_key)
    
    # 根据参数和配置情况选择使用哪个 LLM
    if use_boost and has_boost_config:
        # 使用加速配置
        llm_api_key = boost_api_key
        llm_base_url = boost_base_url
        llm_model = boost_model or os.environ.get("LLM_MODEL_NAME", "")
        config_label = "[加速LLM]"
    else:
        # 使用通用配置
        llm_api_key = os.environ.get("LLM_API_KEY", "")
        llm_base_url = os.environ.get("LLM_BASE_URL", "")
        llm_model = os.environ.get("LLM_MODEL_NAME", "")
        config_label = "[通用LLM]"
    
    # 如果 .env 中没有模型名，则使用 config 作为备用
    if not llm_model:
        llm_model = config.get("llm_model", "gpt-4o-mini")
    
    # 设置 camel-ai 所需的环境变量
    if llm_api_key:
        os.environ["OPENAI_API_KEY"] = llm_api_key
    
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("缺少 API Key 配置，请在项目根目录 .env 文件中设置 LLM_API_KEY")
    
    if llm_base_url:
        os.environ["OPENAI_API_BASE_URL"] = llm_base_url
    
    print(f"{config_label} model={llm_model}, base_url={llm_base_url[:40] if llm_base_url else '默认'}...")
    
    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=llm_model,
    )


def get_active_agents_for_round(
    env,
    config: Dict[str, Any],
    current_hour: int,
    round_num: int
) -> List:
    """根据时间和配置决定本轮激活哪些Agent"""
    time_config = config.get("time_config", {})
    agent_configs = config.get("agent_configs", [])
    
    base_min = time_config.get("agents_per_hour_min", 5)
    base_max = time_config.get("agents_per_hour_max", 20)
    
    peak_hours = time_config.get("peak_hours", [9, 10, 11, 14, 15, 20, 21, 22])
    off_peak_hours = time_config.get("off_peak_hours", [0, 1, 2, 3, 4, 5])
    
    if current_hour in peak_hours:
        multiplier = time_config.get("peak_activity_multiplier", 1.5)
    elif current_hour in off_peak_hours:
        multiplier = time_config.get("off_peak_activity_multiplier", 0.3)
    else:
        multiplier = 1.0
    
    candidates = []
    for cfg in agent_configs:
        agent_id = cfg.get("agent_id", 0)
        active_hours = cfg.get("active_hours", list(range(8, 23)))
        activity_level = cfg.get("activity_level", 0.5)
        
        if current_hour not in active_hours:
            continue
        
        if random.random() < activity_level:
            candidates.append(agent_id)
    
    # 低活跃时段不允许被乘法归零：候选非空时至少 1 个（N-20）
    target_count = active_agent_target(
        base_min=base_min,
        base_max=base_max,
        multiplier=multiplier,
        candidate_count=len(candidates),
    )
    
    selected_ids = random.sample(
        candidates, 
        min(target_count, len(candidates))
    ) if candidates else []
    
    active_agents = []
    for agent_id in selected_ids:
        try:
            agent = env.agent_graph.get_agent(agent_id)
            active_agents.append((agent_id, agent))
        except Exception:
            pass
    
    return active_agents


class PlatformSimulation:
    """平台模拟结果容器"""
    def __init__(self):
        self.env = None
        self.agent_graph = None
        self.total_actions = 0     # 已写入 actions.jsonl 的动作数
        self.agent_actions = 0     # 其中 Agent 真的做出来的（不含种子帖）
        self.rounds_run = 0


async def run_twitter_simulation(
    config: Dict[str, Any], 
    simulation_dir: str,
    main_logger: Optional[SimulationLogManager] = None,
    max_rounds: Optional[int] = None
) -> PlatformSimulation:
    """运行Twitter模拟
    
    Args:
        config: 模拟配置
        simulation_dir: 模拟目录
        main_logger: 主日志管理器
        max_rounds: 最大模拟轮数（可选，用于截断过长的模拟）
        
    Returns:
        PlatformSimulation: 包含env和agent_graph的结果对象
    
    动作记账由 `ActionRecorder` 承担（round_start / 动作 / round_end → twitter/actions.jsonl）。
    它不再是"可选参数"：以前记账器可以是 None，于是"没接上"和"没有动作"长得一样（N-21）。
    """
    result = PlatformSimulation()
    
    def log_info(msg):
        if main_logger:
            main_logger.info(f"[Twitter] {msg}")
        print(f"[Twitter] {msg}")
    
    log_info("初始化...")
    
    # Twitter 使用通用 LLM 配置
    model = create_model(config, use_boost=False)
    
    # OASIS Twitter使用CSV格式
    profile_path = os.path.join(simulation_dir, "twitter_profiles.csv")
    if not os.path.exists(profile_path):
        log_info(f"错误: Profile文件不存在: {profile_path}")
        return result
    
    result.agent_graph = await generate_twitter_agent_graph(
        profile_path=profile_path,
        model=model,
        available_actions=TWITTER_ACTIONS,
    )
    
    # 动作记账器：动作数、actions.jsonl、回写输入全部来自这一条路径（N-21）
    recorder = ActionRecorder("twitter", simulation_dir, config)
    recorder.bind_agent_graph(result.agent_graph)

    # 推荐时间线的模型：校验清单 + 本地加载 + 搬到 GPU + 注入，全部在建环境之前完成。
    # 任一步失败就抛 `RecsysModelUnavailable`——没有它 `rec` 表永远是空的，
    # agent 看不到别人的帖子，整场 twitter 只剩单向广播（台账 N-46）。
    # 设备必须进日志：产物里要能看出这一轮到底在 GPU 还是 CPU 上算（N-47）。
    model_started = time.time()
    model_dir, model_seconds, model_device = load_and_inject()
    log_info(
        f"推荐时间线模型已加载: {model_dir}"
        f"（设备 {model_device}，校验+加载 {time.time() - model_started:.1f}s，"
        f"其中 transformers {model_seconds:.1f}s）"
    )
    # 批大小策略必须进日志（N-48）：OASIS 写死 1000，8 GB 卡实测 OOM；
    # 现在按可用显存算（B），算不出就用实测安全档（A）。
    policy = batch_policy()
    log_info(
        f"推荐表批大小策略: {policy['mode']}，上限 {policy['preflight_batch']} 条/批"
        f"（OASIS 请求 {policy['requested']}；{policy.get('reason') or ''}"
        f"{'，可用显存 ' + str(policy.get('free_mb')) + ' MB' if policy.get('free_mb') else ''}）"
    )
    # 句向量策略也必须进日志（N-47）：随机 pooler 会让推荐表每次运行都不一样
    embedding = embedding_policy()
    log_info(
        f"推荐表句向量: {embedding['pooling']}"
        f"（确定性={embedding['deterministic']}，替换 {embedding['replaces']}）"
    )
    
    db_path = os.path.join(simulation_dir, "twitter_simulation.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    
    result.env = oasis.make(
        agent_graph=result.agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
        semaphore=30,  # 限制最大并发 LLM 请求数，防止 API 过载
    )    
    await result.env.reset()
    log_info("环境已启动")
    
    recorder.log_simulation_start()
    
    # 起点取 time_config 的活跃时段（历史上固定 00:00 → 短轮数整场空转，见 N-18/N-20）
    start_hour = resolve_simulation_start_hour(config)
    
    # 执行初始事件
    event_config = config.get("event_config", {})
    initial_posts = event_config.get("initial_posts", [])
    
    if initial_posts:
        initial_actions = {}
        for post in initial_posts:
            agent_id = post.get("poster_agent_id", 0)
            content = post.get("content", "")
            try:
                agent = result.env.agent_graph.get_agent(agent_id)
            except Exception as error:
                # 种子帖取不到发布者 = 这条种子事件凭空消失，模拟的前提被悄悄改掉。
                # 原来这里是 `except Exception: pass`（掩盖台账 N-23），不允许。
                raise RuntimeError(
                    f"初始帖子无法发布：agent_id={agent_id} 取不到 Agent（{error}）"
                ) from error
            initial_actions[agent] = ManualAction(
                action_type=ActionType.CREATE_POST,
                action_args={"content": content}
            )
        
        if initial_actions:
            await result.env.step(initial_actions)
            log_info(f"已发布 {len(initial_actions)} 条初始帖子")
    
    # round 0 = 种子帖：走和后面每一轮相同的记账路径（从库里数，不从意图数）
    seed_actions = recorder.record_round(0, start_hour)
    log_info(f"round 0 记账: {seed_actions} 条种子帖")
    
    # 主模拟循环
    time_config = config.get("time_config", {})
    total_hours = time_config.get("total_simulation_hours", 72)
    minutes_per_round = time_config.get("minutes_per_round", 30)
    total_rounds = (total_hours * 60) // minutes_per_round
    
    # 如果指定了最大轮数，则截断
    if max_rounds is not None and max_rounds > 0:
        original_rounds = total_rounds
        total_rounds = min(total_rounds, max_rounds)
        if total_rounds < original_rounds:
            log_info(f"轮数已截断: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")
    
    start_time = datetime.now()

    print(f"模拟起点: 第 1 天 {start_hour:02d}:00（来自 time_config.peak_hours）", flush=True)
    for round_num in range(total_rounds):
        # 检查是否收到退出信号
        if _shutdown_event and _shutdown_event.is_set():
            if main_logger:
                main_logger.info(f"收到退出信号，在第 {round_num + 1} 轮停止模拟")
            break
        
        simulated_day, simulated_hour = simulated_clock(
            round_num, start_hour=start_hour, minutes_per_round=minutes_per_round
        )
        
        active_agents = get_active_agents_for_round(
            result.env, config, simulated_hour, round_num
        )
        
        if active_agents:
            actions = {agent: LLMAction() for _, agent in active_agents}
            await result.env.step(actions)
        
        # 一轮一次记账：round_start → 库里新增的真实动作 → round_end。
        # 没有活跃 Agent 的轮次同样记账（0 条动作），否则"死时段"在日志里不存在
        round_action_count = recorder.record_round(round_num + 1, simulated_hour)

        # step 过就必须有推荐时间线：没有它 agent 看不到别人的帖子，
        # "安静"与"平台根本没在跑"必须区分开（台账 N-35/N-46）
        if active_agents:
            recorder.require_timeline(round_num + 1)

        if not active_agents:
            log_info(
                f"Day {simulated_day}, {simulated_hour:02d}:00 - "
                f"Round {round_num + 1}/{total_rounds} 无活跃 Agent（动作 {round_action_count}）"
            )
        elif (round_num + 1) % 20 == 0:
            progress = (round_num + 1) / total_rounds * 100
            log_info(
                f"Day {simulated_day}, {simulated_hour:02d}:00 - "
                f"Round {round_num + 1}/{total_rounds} ({progress:.1f}%) 动作 {round_action_count}"
            )
    
    # 注意：不关闭环境，保留给Interview使用
    
    recorder.log_simulation_end(total_rounds)
    
    result.total_actions = recorder.total_actions
    result.agent_actions = recorder.agent_actions
    result.rounds_run = total_rounds
    elapsed = (datetime.now() - start_time).total_seconds()
    log_info(
        f"模拟循环完成! 耗时: {elapsed:.1f}秒, 记账动作 {recorder.total_actions}"
        f"（种子帖 {recorder.seed_actions}，被过滤的环境行 {recorder.filtered_rows}）"
    )
    
    return result


async def run_reddit_simulation(
    config: Dict[str, Any], 
    simulation_dir: str,
    main_logger: Optional[SimulationLogManager] = None,
    max_rounds: Optional[int] = None
) -> PlatformSimulation:
    """运行Reddit模拟
    
    Args:
        config: 模拟配置
        simulation_dir: 模拟目录
        main_logger: 主日志管理器
        max_rounds: 最大模拟轮数（可选，用于截断过长的模拟）
        
    Returns:
        PlatformSimulation: 包含env和agent_graph的结果对象
    """
    result = PlatformSimulation()
    
    def log_info(msg):
        if main_logger:
            main_logger.info(f"[Reddit] {msg}")
        print(f"[Reddit] {msg}")
    
    log_info("初始化...")
    
    # Reddit 使用加速 LLM 配置（如果有的话，否则回退到通用配置）
    model = create_model(config, use_boost=True)
    
    profile_path = os.path.join(simulation_dir, "reddit_profiles.json")
    if not os.path.exists(profile_path):
        log_info(f"错误: Profile文件不存在: {profile_path}")
        return result
    
    result.agent_graph = await generate_reddit_agent_graph(
        profile_path=profile_path,
        model=model,
        available_actions=REDDIT_ACTIONS,
    )
    
    # 动作记账器：动作数、actions.jsonl、回写输入全部来自这一条路径（N-21）
    recorder = ActionRecorder("reddit", simulation_dir, config)
    recorder.bind_agent_graph(result.agent_graph)
    
    db_path = os.path.join(simulation_dir, "reddit_simulation.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    
    result.env = oasis.make(
        agent_graph=result.agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT,
        database_path=db_path,
        semaphore=30,  # 限制最大并发 LLM 请求数，防止 API 过载
    )
    
    await result.env.reset()
    log_info("环境已启动")
    
    recorder.log_simulation_start()
    
    # 起点取 time_config 的活跃时段（历史上固定 00:00 → 短轮数整场空转，见 N-18/N-20）
    start_hour = resolve_simulation_start_hour(config)
    
    # 执行初始事件
    event_config = config.get("event_config", {})
    initial_posts = event_config.get("initial_posts", [])
    
    if initial_posts:
        initial_actions = {}
        for post in initial_posts:
            agent_id = post.get("poster_agent_id", 0)
            content = post.get("content", "")
            try:
                agent = result.env.agent_graph.get_agent(agent_id)
            except Exception as error:
                # 种子帖取不到发布者 = 这条种子事件凭空消失，模拟的前提被悄悄改掉。
                # 原来这里是 `except Exception: pass`（掩盖台账 N-23），不允许。
                raise RuntimeError(
                    f"初始帖子无法发布：agent_id={agent_id} 取不到 Agent（{error}）"
                ) from error
            
            # Reddit 允许同一 Agent 连发多条种子帖
            if agent in initial_actions:
                if not isinstance(initial_actions[agent], list):
                    initial_actions[agent] = [initial_actions[agent]]
                initial_actions[agent].append(ManualAction(
                    action_type=ActionType.CREATE_POST,
                    action_args={"content": content}
                ))
            else:
                initial_actions[agent] = ManualAction(
                    action_type=ActionType.CREATE_POST,
                    action_args={"content": content}
                )
        
        if initial_actions:
            await result.env.step(initial_actions)
            log_info(f"已发布 {len(initial_actions)} 条初始帖子")
    
    # round 0 = 种子帖：走和后面每一轮相同的记账路径（从库里数，不从意图数）
    seed_actions = recorder.record_round(0, start_hour)
    log_info(f"round 0 记账: {seed_actions} 条种子帖")
    
    # 主模拟循环
    time_config = config.get("time_config", {})
    total_hours = time_config.get("total_simulation_hours", 72)
    minutes_per_round = time_config.get("minutes_per_round", 30)
    total_rounds = (total_hours * 60) // minutes_per_round
    
    # 如果指定了最大轮数，则截断
    if max_rounds is not None and max_rounds > 0:
        original_rounds = total_rounds
        total_rounds = min(total_rounds, max_rounds)
        if total_rounds < original_rounds:
            log_info(f"轮数已截断: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")
    
    start_time = datetime.now()

    print(f"模拟起点: 第 1 天 {start_hour:02d}:00（来自 time_config.peak_hours）", flush=True)
    for round_num in range(total_rounds):
        # 检查是否收到退出信号
        if _shutdown_event and _shutdown_event.is_set():
            if main_logger:
                main_logger.info(f"收到退出信号，在第 {round_num + 1} 轮停止模拟")
            break
        
        simulated_day, simulated_hour = simulated_clock(
            round_num, start_hour=start_hour, minutes_per_round=minutes_per_round
        )
        
        active_agents = get_active_agents_for_round(
            result.env, config, simulated_hour, round_num
        )
        
        if active_agents:
            actions = {agent: LLMAction() for _, agent in active_agents}
            await result.env.step(actions)
        
        # 一轮一次记账：round_start → 库里新增的真实动作 → round_end。
        # 没有活跃 Agent 的轮次同样记账（0 条动作），否则"死时段"在日志里不存在
        round_action_count = recorder.record_round(round_num + 1, simulated_hour)

        # step 过就必须有推荐时间线：没有它 agent 看不到别人的帖子，
        # "安静"与"平台根本没在跑"必须区分开（台账 N-35/N-46）
        if active_agents:
            recorder.require_timeline(round_num + 1)

        if not active_agents:
            log_info(
                f"Day {simulated_day}, {simulated_hour:02d}:00 - "
                f"Round {round_num + 1}/{total_rounds} 无活跃 Agent（动作 {round_action_count}）"
            )
        elif (round_num + 1) % 20 == 0:
            progress = (round_num + 1) / total_rounds * 100
            log_info(
                f"Day {simulated_day}, {simulated_hour:02d}:00 - "
                f"Round {round_num + 1}/{total_rounds} ({progress:.1f}%) 动作 {round_action_count}"
            )
    
    # 注意：不关闭环境，保留给Interview使用
    
    recorder.log_simulation_end(total_rounds)
    
    result.total_actions = recorder.total_actions
    result.agent_actions = recorder.agent_actions
    result.rounds_run = total_rounds
    elapsed = (datetime.now() - start_time).total_seconds()
    log_info(
        f"模拟循环完成! 耗时: {elapsed:.1f}秒, 记账动作 {recorder.total_actions}"
        f"（种子帖 {recorder.seed_actions}，被过滤的环境行 {recorder.filtered_rows}）"
    )
    
    return result


async def main():
    parser = argparse.ArgumentParser(description='OASIS双平台并行模拟')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='配置文件路径 (simulation_config.json)'
    )
    parser.add_argument(
        '--twitter-only',
        action='store_true',
        help='只运行Twitter模拟'
    )
    parser.add_argument(
        '--reddit-only',
        action='store_true',
        help='只运行Reddit模拟'
    )
    parser.add_argument(
        '--max-rounds',
        type=int,
        default=None,
        help='最大模拟轮数（可选，用于截断过长的模拟）'
    )
    parser.add_argument(
        '--no-wait',
        action='store_true',
        default=False,
        help='模拟完成后立即关闭环境，不进入等待命令模式'
    )
    
    args = parser.parse_args()
    
    # 在 main 函数开始时创建 shutdown 事件，确保整个程序都能响应退出信号
    global _shutdown_event
    _shutdown_event = asyncio.Event()
    
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    simulation_dir = os.path.dirname(args.config) or "."
    wait_for_commands = not args.no_wait
    
    # 初始化日志配置（禁用 OASIS 日志，清理旧文件）
    init_logging_for_simulation(simulation_dir)
    
    # 创建日志管理器（平台动作日志由各平台自己的 ActionRecorder 负责）
    log_manager = SimulationLogManager(simulation_dir)
    
    log_manager.info("=" * 60)
    log_manager.info("OASIS 双平台并行模拟")
    log_manager.info(f"配置文件: {args.config}")
    log_manager.info(f"模拟ID: {config.get('simulation_id', 'unknown')}")
    log_manager.info(f"等待命令模式: {'启用' if wait_for_commands else '禁用'}")
    log_manager.info("=" * 60)
    
    time_config = config.get("time_config", {})
    total_hours = time_config.get('total_simulation_hours', 72)
    minutes_per_round = time_config.get('minutes_per_round', 30)
    config_total_rounds = (total_hours * 60) // minutes_per_round
    
    log_manager.info(f"模拟参数:")
    log_manager.info(f"  - 总模拟时长: {total_hours}小时")
    log_manager.info(f"  - 每轮时间: {minutes_per_round}分钟")
    log_manager.info(f"  - 配置总轮数: {config_total_rounds}")
    if args.max_rounds:
        log_manager.info(f"  - 最大轮数限制: {args.max_rounds}")
        if args.max_rounds < config_total_rounds:
            log_manager.info(f"  - 实际执行轮数: {args.max_rounds} (已截断)")
    log_manager.info(f"  - Agent数量: {len(config.get('agent_configs', []))}")
    
    log_manager.info("日志结构:")
    log_manager.info(f"  - 主日志: simulation.log")
    log_manager.info(f"  - Twitter动作: twitter/actions.jsonl")
    log_manager.info(f"  - Reddit动作: reddit/actions.jsonl")
    log_manager.info("=" * 60)
    
    start_time = datetime.now()
    
    # 存储两个平台的模拟结果
    twitter_result: Optional[PlatformSimulation] = None
    reddit_result: Optional[PlatformSimulation] = None
    
    if args.twitter_only:
        twitter_result = await run_twitter_simulation(config, simulation_dir, log_manager, args.max_rounds)
    elif args.reddit_only:
        reddit_result = await run_reddit_simulation(config, simulation_dir, log_manager, args.max_rounds)
    else:
        # 并行运行（每个平台各自记账到自己的 actions.jsonl）
        results = await asyncio.gather(
            run_twitter_simulation(config, simulation_dir, log_manager, args.max_rounds),
            run_reddit_simulation(config, simulation_dir, log_manager, args.max_rounds),
        )
        twitter_result, reddit_result = results
    
    total_elapsed = (datetime.now() - start_time).total_seconds()
    log_manager.info("=" * 60)
    log_manager.info(f"模拟循环完成! 总耗时: {total_elapsed:.1f}秒")

    # 空产物即失败：整场 0 动作不允许进入"等待命令模式"装作成功（N-18）。
    # 判据用 agent_actions（不含 round 0 的种子帖）：种子帖是配置给的输入，
    # 只有它落地说明模拟没跑出任何东西。
    combined_actions = sum(
        result.agent_actions for result in (twitter_result, reddit_result) if result is not None
    )
    combined_seed_actions = sum(
        result.total_actions - result.agent_actions
        for result in (twitter_result, reddit_result) if result is not None
    )
    log_manager.info(
        f"动作账目: Agent 动作 {combined_actions} 条，种子帖 {combined_seed_actions} 条"
    )
    total_rounds_run = max(
        (getattr(result, "rounds_run", 0) for result in (twitter_result, reddit_result) if result is not None),
        default=0,
    )
    empty_run_error: Optional[str] = None
    try:
        assert_run_produced_actions(
            combined_actions,
            total_rounds=total_rounds_run or 1,
            minutes_per_round=int((config.get("time_config") or {}).get("minutes_per_round", 30)),
            start_hour=resolve_simulation_start_hour(config),
            peak_hours=(config.get("time_config") or {}).get("peak_hours", []),
            off_peak_hours=(config.get("time_config") or {}).get("off_peak_hours", []),
        )
    except EmptySimulationRun as error:
        # 先关环境、再以非零退出码结束（runner 会把日志尾部作为失败原因）
        log_manager.info(f"ERROR: {error}")
        empty_run_error = str(error)
    
    # 是否进入等待命令模式
    if wait_for_commands:
        log_manager.info("")
        log_manager.info("=" * 60)
        log_manager.info("进入等待命令模式 - 环境保持运行")
        log_manager.info("支持的命令: interview, batch_interview, close_env")
        log_manager.info("=" * 60)
        
        # 创建IPC处理器
        ipc_handler = ParallelIPCHandler(
            simulation_dir=simulation_dir,
            twitter_env=twitter_result.env if twitter_result else None,
            twitter_agent_graph=twitter_result.agent_graph if twitter_result else None,
            reddit_env=reddit_result.env if reddit_result else None,
            reddit_agent_graph=reddit_result.agent_graph if reddit_result else None
        )
        ipc_handler.update_status("alive")
        
        # 等待命令循环（使用全局 _shutdown_event）
        try:
            while not _shutdown_event.is_set():
                should_continue = await ipc_handler.process_commands()
                if not should_continue:
                    break
                # 使用 wait_for 替代 sleep，这样可以响应 shutdown_event
                try:
                    await asyncio.wait_for(_shutdown_event.wait(), timeout=0.5)
                    break  # 收到退出信号
                except asyncio.TimeoutError:
                    pass  # 超时继续循环
        except KeyboardInterrupt:
            print("\n收到中断信号")
        except asyncio.CancelledError:
            print("\n任务被取消")
        except Exception as e:
            print(f"\n命令处理出错: {e}")
        
        log_manager.info("\n关闭环境...")
        ipc_handler.update_status("stopped")
    
    # 关闭环境
    if twitter_result and twitter_result.env:
        await twitter_result.env.close()
        log_manager.info("[Twitter] 环境已关闭")
    
    if reddit_result and reddit_result.env:
        await reddit_result.env.close()
        log_manager.info("[Reddit] 环境已关闭")
    
    log_manager.info("=" * 60)
    log_manager.info(f"全部完成!")
    log_manager.info(f"日志文件:")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'simulation.log')}")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'twitter', 'actions.jsonl')}")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'reddit', 'actions.jsonl')}")
    log_manager.info("=" * 60)

    if empty_run_error:
        # 非零退出码 → runner 记为 FAILED（而不是一直 running）
        print(f"ERROR: {empty_run_error}", flush=True)
        sys.exit(1)


def setup_signal_handlers(loop=None):
    """
    设置信号处理器，确保收到 SIGTERM/SIGINT 时能够正确退出
    
    持久化模拟场景：模拟完成后不退出，等待 interview 命令
    当收到终止信号时，需要：
    1. 通知 asyncio 循环退出等待
    2. 让程序有机会正常清理资源（关闭数据库、环境等）
    3. 然后才退出
    """
    def signal_handler(signum, frame):
        global _cleanup_done
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\n收到 {sig_name} 信号，正在退出...")
        
        if not _cleanup_done:
            _cleanup_done = True
            # 设置事件通知 asyncio 循环退出（让循环有机会清理资源）
            if _shutdown_event:
                _shutdown_event.set()
        
        # 不要直接 sys.exit()，让 asyncio 循环正常退出并清理资源
        # 如果是重复收到信号，才强制退出
        else:
            print("强制退出...")
            sys.exit(1)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    setup_signal_handlers()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序被中断")
    except SystemExit:
        pass
    finally:
        # 清理 multiprocessing 资源跟踪器（防止退出时的警告）
        try:
            from multiprocessing import resource_tracker
            resource_tracker._resource_tracker._stop()
        except Exception:
            pass
        print("模拟进程已退出")
