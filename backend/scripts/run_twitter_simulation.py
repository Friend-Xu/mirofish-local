"""
OASIS Twitter模拟预设脚本
此脚本读取配置文件中的参数来执行模拟，实现全程自动化

功能特性:
- 完成模拟后不立即关闭环境，进入等待命令模式
- 支持通过IPC接收Interview命令
- 支持单个Agent采访和批量采访
- 支持远程关闭环境命令

使用方式:
    python run_twitter_simulation.py --config /path/to/simulation_config.json
    python run_twitter_simulation.py --config /path/to/simulation_config.json --no-wait  # 完成后立即关闭
"""

import argparse
import asyncio
import json
import logging
import os
import random
import signal
import time
import sys
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional

# 全局变量：用于信号处理
_shutdown_event = None
_cleanup_done = False

# 添加项目路径
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, '..'))
_project_root = os.path.abspath(os.path.join(_backend_dir, '..'))
sys.path.insert(0, _scripts_dir)
sys.path.insert(0, _backend_dir)

# 加载项目根目录的 .env 文件（包含 LLM_API_KEY 等配置）
from dotenv import load_dotenv
_env_file = os.path.join(_project_root, '.env')
if os.path.exists(_env_file):
    load_dotenv(_env_file)
else:
    _backend_env = os.path.join(_backend_dir, '.env')
    if os.path.exists(_backend_env):
        load_dotenv(_backend_env)


import re

from simulation_time_utils import (
    EmptySimulationRun,
    active_agent_target,
    assert_run_produced_actions,
    resolve_simulation_start_hour,
    simulated_clock,
)

# 记账只有一条路径：三个 run_*_simulation.py 共用 action_recorder（掩盖台账 N-21）
from action_recorder import ActionRecorder


class UnicodeFormatter(logging.Formatter):
    """自定义格式化器，将 Unicode 转义序列转换为可读字符"""
    
    UNICODE_ESCAPE_PATTERN = re.compile(r'\\u([0-9a-fA-F]{4})')
    
    def format(self, record):
        result = super().format(record)
        
        def replace_unicode(match):
            try:
                return chr(int(match.group(1), 16))
            except (ValueError, OverflowError):
                return match.group(0)
        
        return self.UNICODE_ESCAPE_PATTERN.sub(replace_unicode, result)


class MaxTokensWarningFilter(logging.Filter):
    """过滤掉 camel-ai 关于 max_tokens 的警告（我们故意不设置 max_tokens，让模型自行决定）"""
    
    def filter(self, record):
        # 过滤掉包含 max_tokens 警告的日志
        if "max_tokens" in record.getMessage() and "Invalid or missing" in record.getMessage():
            return False
        return True


# 在模块加载时立即添加过滤器，确保在 camel 代码执行前生效
logging.getLogger().addFilter(MaxTokensWarningFilter())


def setup_oasis_logging(log_dir: str):
    """配置 OASIS 的日志，使用固定名称的日志文件"""
    os.makedirs(log_dir, exist_ok=True)
    
    # 清理旧的日志文件
    for f in os.listdir(log_dir):
        old_log = os.path.join(log_dir, f)
        if os.path.isfile(old_log) and f.endswith('.log'):
            try:
                os.remove(old_log)
            except OSError:
                pass
    
    formatter = UnicodeFormatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")
    
    loggers_config = {
        "social.agent": os.path.join(log_dir, "social.agent.log"),
        "social.twitter": os.path.join(log_dir, "social.twitter.log"),
        "social.rec": os.path.join(log_dir, "social.rec.log"),
        "oasis.env": os.path.join(log_dir, "oasis.env.log"),
        "table": os.path.join(log_dir, "table.log"),
    }
    
    for logger_name, log_file in loggers_config.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.propagate = False


from twitter_recsys_model import (  # noqa: E402
    batch_policy,
    embedding_policy,
    load_and_inject,
    pin_offline_runtime,
)

# 运行期不联网必须在 `import oasis`（→ transformers）之前钉死：它们在导入期读这些开关。
# twitter 的推荐时间线用**项目自有目录** `<项目>/models/twhin-bert-base/`（清单固定内容，
# 见 twitter_recsys_model.py 与掩盖台账 N-46/F17/F18），不经过 HF 缓存布局。
pin_offline_runtime()


try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    import oasis
    from oasis import (
        ActionType,
        LLMAction,
        ManualAction,
        generate_twitter_agent_graph
    )
except ImportError as e:
    print(f"错误: 缺少依赖 {e}")
    print("请先安装: pip install oasis-ai camel-ai")
    sys.exit(1)


# IPC相关常量
IPC_COMMANDS_DIR = "ipc_commands"
IPC_RESPONSES_DIR = "ipc_responses"
ENV_STATUS_FILE = "env_status.json"

class CommandType:
    """命令类型常量"""
    INTERVIEW = "interview"
    BATCH_INTERVIEW = "batch_interview"
    CLOSE_ENV = "close_env"


class IPCHandler:
    """IPC命令处理器"""
    
    def __init__(self, simulation_dir: str, env, agent_graph):
        self.simulation_dir = simulation_dir
        self.env = env
        self.agent_graph = agent_graph
        self.commands_dir = os.path.join(simulation_dir, IPC_COMMANDS_DIR)
        self.responses_dir = os.path.join(simulation_dir, IPC_RESPONSES_DIR)
        self.status_file = os.path.join(simulation_dir, ENV_STATUS_FILE)
        self._running = True
        
        # 确保目录存在
        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)
    
    def update_status(self, status: str):
        """更新环境状态"""
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump({
                "status": status,
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
    
    async def handle_interview(self, command_id: str, agent_id: int, prompt: str) -> bool:
        """
        处理单个Agent采访命令
        
        Returns:
            True 表示成功，False 表示失败
        """
        try:
            # 获取Agent
            agent = self.agent_graph.get_agent(agent_id)
            
            # 创建Interview动作
            interview_action = ManualAction(
                action_type=ActionType.INTERVIEW,
                action_args={"prompt": prompt}
            )
            
            # 执行Interview
            actions = {agent: interview_action}
            await self.env.step(actions)
            
            # 从数据库获取结果
            result = self._get_interview_result(agent_id)
            
            if not result.get("response"):
                # 没拿到回答就不是成功：不能报 completed
                print(f"  错误: Interview 没有回答: agent_id={agent_id}")
                self.send_response(
                    command_id, "failed", result=result,
                    error=f"Agent {agent_id} 没有给出回答（trace 中没有 response）",
                )
                return False
            
            self.send_response(command_id, "completed", result=result)
            print(f"  Interview完成: agent_id={agent_id}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"  Interview失败: agent_id={agent_id}, error={error_msg}")
            self.send_response(command_id, "failed", error=error_msg)
            return False
    
    async def handle_batch_interview(self, command_id: str, interviews: List[Dict]) -> bool:
        """
        处理批量采访命令
        
        Args:
            interviews: [{"agent_id": int, "prompt": str}, ...]
        """
        try:
            # 构建动作字典
            actions = {}
            agent_prompts = {}  # 记录每个agent的prompt
            unavailable = []
            
            for interview in interviews:
                agent_id = interview.get("agent_id")
                prompt = interview.get("prompt", "")
                
                try:
                    agent = self.agent_graph.get_agent(agent_id)
                    actions[agent] = ManualAction(
                        action_type=ActionType.INTERVIEW,
                        action_args={"prompt": prompt}
                    )
                    agent_prompts[agent_id] = prompt
                except Exception as e:
                    print(f"  错误: 无法获取Agent {agent_id}: {e}")
                    unavailable.append({"agent_id": agent_id, "error": str(e)})
            
            if not actions:
                self.send_response(command_id, "failed", result={
                    "requested_count": len(interviews),
                    "unavailable": unavailable,
                }, error="没有有效的Agent")
                return False
            
            # 执行批量Interview
            await self.env.step(actions)
            
            # 获取所有结果：只有真的拿到 response 才算成功
            results = {}
            missing = {}
            for agent_id in agent_prompts.keys():
                result = self._get_interview_result(agent_id)
                if not result.get("response"):
                    missing[agent_id] = "没有回答"
                    continue
                results[agent_id] = result
            
            self.send_response(command_id, "completed", result={
                "requested_count": len(interviews),
                "interviews_count": len(results),
                "missing_count": len(missing),
                "missing_agents": missing,
                "unavailable": unavailable,
                "results": results
            })
            print(
                f"  批量Interview完成: 成功 {len(results)}/{len(interviews)}，"
                f"未回答 {len(missing)}，不可用 {len(unavailable)}"
            )
            return True
            
        except Exception as e:
            error_msg = str(e)
            print(f"  批量Interview失败: {error_msg}")
            self.send_response(command_id, "failed", error=error_msg)
            return False
    
    def _get_interview_result(self, agent_id: int) -> Dict[str, Any]:
        """从数据库获取最新的Interview结果"""
        db_path = os.path.join(self.simulation_dir, "twitter_simulation.db")
        
        result = {
            "agent_id": agent_id,
            "response": None,
            "timestamp": None
        }
        
        if not os.path.exists(db_path):
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
                    if response:
                        result["response"] = response
                        result["timestamp"] = created_at
                except json.JSONDecodeError:
                    result["response"] = info_json
            
            conn.close()
            
        except Exception as e:
            print(f"  错误: 读取Interview结果失败: {e}")
        
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
                args.get("prompt", "")
            )
            return True
            
        elif command_type == CommandType.BATCH_INTERVIEW:
            await self.handle_batch_interview(
                command_id,
                args.get("interviews", [])
            )
            return True
            
        elif command_type == CommandType.CLOSE_ENV:
            print("收到关闭环境命令")
            self.send_response(command_id, "completed", result={"message": "环境即将关闭"})
            return False
        
        else:
            self.send_response(command_id, "failed", error=f"未知命令类型: {command_type}")
            return True


class TwitterSimulationRunner:
    """Twitter模拟运行器"""
    
    # Twitter可用动作（不包含INTERVIEW，INTERVIEW只能通过ManualAction手动触发）
    # REFRESH 必须在内：OASIS 用它把推荐时间线（rec 表）拉进 agent 的 observation。
    # 没有它，agent 看不到别人的帖子，只能一直 CREATE_POST——实测 35 agent × 11 轮
    # 只产出 103 条发帖、0 赞 0 转发 0 关注（台账 N-35）。
    AVAILABLE_ACTIONS = [
        ActionType.CREATE_POST,
        ActionType.LIKE_POST,
        ActionType.REPOST,
        ActionType.FOLLOW,
        ActionType.DO_NOTHING,
        ActionType.QUOTE_POST,
        ActionType.REFRESH,
    ]
    
    def __init__(self, config_path: str, wait_for_commands: bool = True):
        """
        初始化模拟运行器
        
        Args:
            config_path: 配置文件路径 (simulation_config.json)
            wait_for_commands: 模拟完成后是否等待命令（默认True）
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.simulation_dir = os.path.dirname(config_path)
        self.wait_for_commands = wait_for_commands
        self.env = None
        self.agent_graph = None
        self.ipc_handler = None
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_profile_path(self) -> str:
        """获取Profile文件路径（OASIS Twitter使用CSV格式）"""
        return os.path.join(self.simulation_dir, "twitter_profiles.csv")
    
    def _get_db_path(self) -> str:
        """获取数据库路径"""
        return os.path.join(self.simulation_dir, "twitter_simulation.db")
    
    def _create_model(self):
        """
        创建LLM模型
        
        统一使用项目根目录 .env 文件中的配置（优先级最高）：
        - LLM_API_KEY: API密钥
        - LLM_BASE_URL: API基础URL
        - LLM_MODEL_NAME: 模型名称
        """
        # 优先从 .env 读取配置
        llm_api_key = os.environ.get("LLM_API_KEY", "")
        llm_base_url = os.environ.get("LLM_BASE_URL", "")
        llm_model = os.environ.get("LLM_MODEL_NAME", "")
        
        # 如果 .env 中没有，则使用 config 作为备用
        if not llm_model:
            llm_model = self.config.get("llm_model", "gpt-4o-mini")
        
        # 设置 camel-ai 所需的环境变量
        if llm_api_key:
            os.environ["OPENAI_API_KEY"] = llm_api_key
        
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("缺少 API Key 配置，请在项目根目录 .env 文件中设置 LLM_API_KEY")
        
        if llm_base_url:
            os.environ["OPENAI_API_BASE_URL"] = llm_base_url
        
        print(f"LLM配置: model={llm_model}, base_url={llm_base_url[:40] if llm_base_url else '默认'}...")
        
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=llm_model,
        )
    
    def _get_active_agents_for_round(
        self, 
        env, 
        current_hour: int,
        round_num: int
    ) -> List:
        """
        根据时间和配置决定本轮激活哪些Agent
        
        Args:
            env: OASIS环境
            current_hour: 当前模拟小时（0-23）
            round_num: 当前轮数
            
        Returns:
            激活的Agent列表
        """
        time_config = self.config.get("time_config", {})
        agent_configs = self.config.get("agent_configs", [])
        
        # 基础激活数量
        base_min = time_config.get("agents_per_hour_min", 5)
        base_max = time_config.get("agents_per_hour_max", 20)
        
        # 根据时段调整
        peak_hours = time_config.get("peak_hours", [9, 10, 11, 14, 15, 20, 21, 22])
        off_peak_hours = time_config.get("off_peak_hours", [0, 1, 2, 3, 4, 5])
        
        if current_hour in peak_hours:
            multiplier = time_config.get("peak_activity_multiplier", 1.5)
        elif current_hour in off_peak_hours:
            multiplier = time_config.get("off_peak_activity_multiplier", 0.3)
        else:
            multiplier = 1.0
        
        # 根据每个Agent的配置计算激活概率
        candidates = []
        for cfg in agent_configs:
            agent_id = cfg.get("agent_id", 0)
            active_hours = cfg.get("active_hours", list(range(8, 23)))
            activity_level = cfg.get("activity_level", 0.5)
            
            # 检查是否在活跃时间
            if current_hour not in active_hours:
                continue
            
            # 根据活跃度计算概率
            if random.random() < activity_level:
                candidates.append(agent_id)
        
        # 低活跃时段不允许被乘法归零：候选非空时至少 1 个（N-20）
        target_count = active_agent_target(
            base_min=base_min,
            base_max=base_max,
            multiplier=multiplier,
            candidate_count=len(candidates),
        )
        
        # 随机选择
        selected_ids = random.sample(
            candidates, 
            min(target_count, len(candidates))
        ) if candidates else []
        
        # 转换为Agent对象
        active_agents = []
        for agent_id in selected_ids:
            try:
                agent = env.agent_graph.get_agent(agent_id)
                active_agents.append((agent_id, agent))
            except Exception:
                pass
        
        return active_agents
    
    async def run(self, max_rounds: int = None):
        """运行Twitter模拟
        
        Args:
            max_rounds: 最大模拟轮数（可选，用于截断过长的模拟）
        """
        print("=" * 60)
        print("OASIS Twitter模拟")
        print(f"配置文件: {self.config_path}")
        print(f"模拟ID: {self.config.get('simulation_id', 'unknown')}")
        print(f"等待命令模式: {'启用' if self.wait_for_commands else '禁用'}")
        print("=" * 60)
        
        # 加载时间配置
        time_config = self.config.get("time_config", {})
        total_hours = time_config.get("total_simulation_hours", 72)
        minutes_per_round = time_config.get("minutes_per_round", 30)
        
        # 计算总轮数
        total_rounds = (total_hours * 60) // minutes_per_round
        
        # 如果指定了最大轮数，则截断
        if max_rounds is not None and max_rounds > 0:
            original_rounds = total_rounds
            total_rounds = min(total_rounds, max_rounds)
            if total_rounds < original_rounds:
                print(f"\n轮数已截断: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")
        
        print(f"\n模拟参数:")
        print(f"  - 总模拟时长: {total_hours}小时")
        print(f"  - 每轮时间: {minutes_per_round}分钟")
        print(f"  - 总轮数: {total_rounds}")
        if max_rounds:
            print(f"  - 最大轮数限制: {max_rounds}")
        print(f"  - Agent数量: {len(self.config.get('agent_configs', []))}")
        
        # 创建模型
        print("\n初始化LLM模型...")
        model = self._create_model()
        
        # 加载Agent图
        print("加载Agent Profile...")
        profile_path = self._get_profile_path()
        if not os.path.exists(profile_path):
            print(f"错误: Profile文件不存在: {profile_path}")
            return
        
        self.agent_graph = await generate_twitter_agent_graph(
            profile_path=profile_path,
            model=model,
            available_actions=self.AVAILABLE_ACTIONS,
        )
        
        # 数据库路径
        db_path = self._get_db_path()
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"已删除旧数据库: {db_path}")
        
        # 推荐时间线的模型：校验清单 + 本地加载 + 搬到 GPU + 注入，全部在建环境之前完成。
        # 任一步失败就抛 `RecsysModelUnavailable`——没有它 `rec` 表永远是空的，
        # agent 看不到别人的帖子，整场 twitter 只剩单向广播（台账 N-46）。
        # 设备必须进日志：产物里要能看出这一轮到底在 GPU 还是 CPU 上算（N-47）。
        model_started = time.time()
        model_dir, model_seconds, model_device = load_and_inject()
        print(
            f"推荐时间线模型已加载: {model_dir}"
            f"（设备 {model_device}，校验+加载 {time.time() - model_started:.1f}s，"
            f"其中 transformers {model_seconds:.1f}s）",
            flush=True,
        )
        # 批大小策略必须进日志（N-48）：OASIS 写死 1000，8 GB 卡实测 OOM；
        # 现在按可用显存算（B），算不出就用实测安全档（A）。
        policy = batch_policy()
        print(
            f"推荐表批大小策略: {policy['mode']}，上限 {policy['preflight_batch']} 条/批"
            f"（OASIS 请求 {policy['requested']}；{policy.get('reason') or ''}"
            f"{'，可用显存 ' + str(policy.get('free_mb')) + ' MB' if policy.get('free_mb') else ''}）",
            flush=True,
        )
        # 句向量策略也必须进日志（N-47）：随机 pooler 会让推荐表每次运行都不一样
        embedding = embedding_policy()
        print(
            f"推荐表句向量: {embedding['pooling']}"
            f"（确定性={embedding['deterministic']}，替换 {embedding['replaces']}）",
            flush=True,
        )

        # 创建环境
        print("创建OASIS环境...")
        self.env = oasis.make(
            agent_graph=self.agent_graph,
            platform=oasis.DefaultPlatformType.TWITTER,
            database_path=db_path,
            semaphore=30,  # 限制最大并发 LLM 请求数，防止 API 过载
        )
        
        await self.env.reset()
        print("环境初始化完成\n")
        
        # 动作记账器：动作数、actions.jsonl、回写输入全部来自这一条路径（N-21）
        recorder = ActionRecorder("twitter", self.simulation_dir, self.config)
        recorder.bind_agent_graph(self.agent_graph)
        recorder.log_simulation_start()
        
        # 初始化IPC处理器
        self.ipc_handler = IPCHandler(self.simulation_dir, self.env, self.agent_graph)
        self.ipc_handler.update_status("running")
        
        # 起点取 time_config 的活跃时段（历史上固定 00:00 → 短轮数整场空转，见 N-18/N-20）
        start_hour = resolve_simulation_start_hour(self.config)
        print(f"\n模拟起点: 第 1 天 {start_hour:02d}:00（来自 time_config.peak_hours）", flush=True)
        
        # 执行初始事件
        event_config = self.config.get("event_config", {})
        initial_posts = event_config.get("initial_posts", [])
        
        if initial_posts:
            print(f"执行初始事件 ({len(initial_posts)}条初始帖子)...")
            initial_actions = {}
            for post in initial_posts:
                agent_id = post.get("poster_agent_id", 0)
                content = post.get("content", "")
                try:
                    agent = self.env.agent_graph.get_agent(agent_id)
                except Exception as error:
                    # 种子帖取不到发布者 = 这条种子事件凭空消失，模拟的前提被悄悄改掉。
                    # 原来这里只打印一句警告（掩盖台账 N-23），不允许。
                    raise RuntimeError(
                        f"初始帖子无法发布：agent_id={agent_id} 取不到 Agent（{error}）"
                    ) from error
                initial_actions[agent] = ManualAction(
                    action_type=ActionType.CREATE_POST,
                    action_args={"content": content}
                )
            
            if initial_actions:
                await self.env.step(initial_actions)
                print(f"  已发布 {len(initial_actions)} 条初始帖子")
        
        # round 0 = 种子帖：走和后面每一轮相同的记账路径（从库里数，不从意图数）
        seed_actions = recorder.record_round(0, start_hour)
        print(f"  round 0 记账: {seed_actions} 条种子帖", flush=True)
        
        # 主模拟循环
        print("\n开始模拟循环...")
        start_time = datetime.now()
        
        for round_num in range(total_rounds):
            # 计算当前模拟时间
            simulated_day, simulated_hour = simulated_clock(
                round_num, start_hour=start_hour, minutes_per_round=minutes_per_round
            )
            
            # 获取本轮激活的Agent
            active_agents = self._get_active_agents_for_round(
                self.env, simulated_hour, round_num
            )
            
            # 每一轮都要可见（包括 0 个激活的"死时段"轮）：
            # 只打 10 的倍数会让"全程 0 动作"完全看不见
            print(f"  [Day {simulated_day}, {simulated_hour:02d}:00] "
                  f"Round {round_num + 1}/{total_rounds} "
                  f"- {len(active_agents)} agents active", flush=True)
            
            if active_agents:
                # 构建动作
                actions = {
                    agent: LLMAction()
                    for _, agent in active_agents
                }
                await self.env.step(actions)
            
            # 一轮一次记账：round_start → 库里新增的真实动作 → round_end。
            # 没有活跃 Agent 的轮次同样记账（0 条动作），否则"死时段"在日志里不存在。
            # 动作数取自同一条路径——不再"另外数一遍 trace"（N-21）。
            produced = recorder.record_round(round_num + 1, simulated_hour)
            print(f"      -> 实际动作: {produced}", flush=True)

            # step 过就必须有推荐时间线：没有它 agent 看不到别人的帖子，
            # "安静"与"平台根本没在跑"必须区分开（台账 N-35/N-46）
            if active_agents:
                recorder.require_timeline(round_num + 1)
        
        recorder.log_simulation_end(total_rounds)
        
        # 空产物即失败：没有 Agent 动作的运行不允许进入"等待命令模式"装作成功。
        # 判据用 agent_actions（不含 round 0 的种子帖——种子帖是配置给的输入）。
        assert_run_produced_actions(
            recorder.agent_actions,
            total_rounds=total_rounds,
            minutes_per_round=minutes_per_round,
            start_hour=start_hour,
            peak_hours=self.config.get("time_config", {}).get("peak_hours", []),
            off_peak_hours=self.config.get("time_config", {}).get("off_peak_hours", []),
        )
        
        total_elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n模拟循环完成!", flush=True)
        print(f"  - 总耗时: {total_elapsed:.1f}秒")
        print(f"  - 动作账目: {recorder.total_actions} 条已写入 actions.jsonl"
              f"（种子帖 {recorder.seed_actions}，环境行被过滤 {recorder.filtered_rows}）")
        print(f"  - 数据库: {db_path}")
        print(f"  - 动作日志: {os.path.join(self.simulation_dir, 'twitter', 'actions.jsonl')}")
        
        # 是否进入等待命令模式
        if self.wait_for_commands:
            print("\n" + "=" * 60)
            print("进入等待命令模式 - 环境保持运行")
            print("支持的命令: interview, batch_interview, close_env")
            print("=" * 60)
            
            self.ipc_handler.update_status("alive")
            
            # 等待命令循环（使用全局 _shutdown_event）
            try:
                while not _shutdown_event.is_set():
                    should_continue = await self.ipc_handler.process_commands()
                    if not should_continue:
                        break
                    try:
                        await asyncio.wait_for(_shutdown_event.wait(), timeout=0.5)
                        break  # 收到退出信号
                    except asyncio.TimeoutError:
                        pass
            except KeyboardInterrupt:
                print("\n收到中断信号")
            except asyncio.CancelledError:
                print("\n任务被取消")
            except Exception as e:
                print(f"\n命令处理出错: {e}")
            
            print("\n关闭环境...")
        
        # 关闭环境
        self.ipc_handler.update_status("stopped")
        await self.env.close()
        
        print("环境已关闭")
        print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description='OASIS Twitter模拟')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='配置文件路径 (simulation_config.json)'
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
    
    # 在 main 函数开始时创建 shutdown 事件
    global _shutdown_event
    _shutdown_event = asyncio.Event()
    
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    # 初始化日志配置（使用固定文件名，清理旧日志）
    simulation_dir = os.path.dirname(args.config) or "."
    setup_oasis_logging(os.path.join(simulation_dir, "log"))
    
    runner = TwitterSimulationRunner(
        config_path=args.config,
        wait_for_commands=not args.no_wait
    )
    try:
        await runner.run(max_rounds=args.max_rounds)
    except EmptySimulationRun as error:
        # 以非零退出码结束：runner 会把 simulation.log 的尾部作为失败原因
        print(f"\nERROR: {error}", flush=True)
        sys.exit(1)


def setup_signal_handlers():
    """
    设置信号处理器，确保收到 SIGTERM/SIGINT 时能够正确退出
    让程序有机会正常清理资源（关闭数据库、环境等）
    """
    def signal_handler(signum, frame):
        global _cleanup_done
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\n收到 {sig_name} 信号，正在退出...")
        if not _cleanup_done:
            _cleanup_done = True
            if _shutdown_event:
                _shutdown_event.set()
        else:
            # 重复收到信号才强制退出
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
        print("模拟进程已退出")
