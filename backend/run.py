"""
MiroFish Backend 启动入口
"""

import os
import sys

# 解决 Windows 控制台中文乱码问题：在所有导入之前设置 UTF-8 编码
if sys.platform == 'win32':
    # 设置环境变量确保 Python 使用 UTF-8
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    # 重新配置标准输出流为 UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.config import Config
from app.services.report_agent import ReportManager
from app.services.simulation_runner import SimulationRunner


def reconcile_previous_run() -> None:
    """启动和解：把"上次后端进程死在半路"这件事说出来。

    原生崩溃（实测 Kuzu access violation，见 `掩盖台账.md` N-24）不会留下 Python 异常，
    也不会跑 atexit：`run_state.json` 停在 `running`、模拟子进程变孤儿，没人会说出这件事。
    历史处理是把"长跑中途死"归因成"工具会话回收宿主进程"（M-05），失败就此消失。

    报告任务同理（N-32）：`reports/<id>/meta.json` 停在 `planning`、`progress.json` 停在
    `generating`，之后任何查询都看到"还在生成"。任务本身只存在内存里，进程一死就没了，
    能收尾的只有这两份状态文件。

    放在启动入口而不是 Flask 应用工厂：测试会反复调用 `create_app()`，
    和解会写真实运行目录——那会让一次 pytest 把正在跑的模拟判成失败。
    """

    try:
        stale_runs = SimulationRunner.reconcile_stale_runs()
    except Exception as error:  # noqa: BLE001 - 和解失败要记录，但不能挡住服务启动
        print(f"启动和解失败（模拟运行）: {error}")
    else:
        for item in stale_runs:
            print(f"启动和解: {item}")

    try:
        stale_reports = ReportManager.reconcile_stale_reports()
    except Exception as error:  # noqa: BLE001 - 同上，报告侧和解也不能挡住启动
        print(f"启动和解失败（报告任务）: {error}")
    else:
        for item in stale_reports:
            print(f"启动和解（报告）: {item}")


def main():
    """主函数"""
    # 验证配置
    errors = Config.validate()
    if errors:
        print("配置错误:")
        for err in errors:
            print(f"  - {err}")
        print("\n请检查 .env 文件中的配置")
        sys.exit(1)

    # 上次运行的收尾（崩溃/强杀留下的 running 状态与孤儿进程）
    reconcile_previous_run()
    
    # 创建应用
    app = create_app()
    
    # 获取运行配置
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5001))
    debug = Config.DEBUG
    
    # 启动服务
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    main()

