from argparse import ArgumentParser
import multiprocessing
import signal
import sys
import time
from tqdm import tqdm

from .emulator import EmulatorManager
from .evaluator import Evaluator
from .utils import setup_logger


logger = setup_logger()
MAX_CONSECUTIVE_RETRIES = 5


def get_parser():
    parser = ArgumentParser(description="Android任务评估工具")
    parser.add_argument(
        "--config", 
        "-c",
        type=str, 
        required=True,
        help="任务配置文件路径"
    )
    parser.add_argument(
        "--log_dir", 
        "-d",
        type=str, 
        default="debug/logs",
        help="日志输出目录"
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default="debug/task_stats.parquet",
        help="检查点文件路径，不指定则从头开始评估"
    )
    parser.add_argument(
        "--agent_name",
        type=str,
        default="m3a_gpt4v",
        help="agent名称"
    )
    parser.add_argument(
        "--repeat_times", 
        "-r",
        type=int, 
        default=3,
        help="每个任务执行的次数"
    )
    parser.add_argument(
        "--test_samples", 
        "-t",
        type=int, 
        default=None,
        help="测试的任务数量，不指定则测试所有任务"
    )
    parser.add_argument(
        "--exclude_pattern", 
        type=str, 
        default=None,
        help="要排除的任务的正则表达式模式，例如: 'Recipe.*|Task.*'"
    )
    parser.add_argument(
        "--max_restarts", 
        type=int, 
        default=10,
        help="模拟器最大重启次数"
    )
    parser.add_argument(
        "--sdk_path",
        type=str,
        default=None,
        help="Android SDK路径，默认使用ANDROID_HOME环境变量或~/Library/Android/sdk"
    )
    parser.add_argument(
        "--app_path",
        type=str,
        default=None,
        help="应用APK文件路径，默认使用ANDROID_APP_PATH环境变量或debug/app-release.apk"
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default=None,
        help="要加载的模拟器快照名称，不指定则不使用快照"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="任务执行失败时的最大重试次数"
    )
    return parser


def main():
    # 创建信号处理函数
    def signal_handler(signum, frame):
        logger.warning(f"收到信号 {signum}，准备退出程序")
        if 'evaluator' in locals():
            # 等待子进程完成当前任务的统计更新
            time.sleep(1)
            # 通知子进程保存统计
            evaluator.status_queue.put(("save_stats", None))
            # 等待子进程处理完成
            time.sleep(1)
        if 'manager' in locals():
            logger.info("关闭模拟器...")
            manager.stop_emulator()
        sys.exit(1)

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill

    try:
        parser = get_parser()
        args = parser.parse_args()
        
        # 创建状态队列
        status_queue = multiprocessing.Queue()
        
        # 首先创建Evaluator实例来检查是否有待执行的任务
        evaluator = Evaluator(
            config_file=args.config,
            log_dir=args.log_dir,
            agent_name=args.agent_name,
            status_queue=status_queue,
            checkpoint_file=args.checkpoint_file,
            n_trials=args.repeat_times,
            test_samples=args.test_samples,
            exclude_pattern=args.exclude_pattern,
            max_retries=args.max_retries
        )
        
        # 检查是否有待执行的任务
        if not evaluator.has_pending_tasks():
            logger.info("所有任务已完成足够的试验次数，无需启动模拟器")
            return
            
        # 创建并启动模拟器
        manager = EmulatorManager(
            sdk_path=args.sdk_path, 
            app_path=args.app_path,
            snapshot_name=args.snapshot
        )
        logger.info(f"使用Android SDK路径: {manager.sdk_path}")
        logger.info(f"使用应用文件路径: {manager.app_path}")
        if args.snapshot:
            logger.info(f"使用模拟器快照: {args.snapshot}")
        
        logger.info("Starting emulator...")
        if not manager.start_emulator():
            logger.error("模拟器启动失败，退出程序")
            sys.exit(1)
        logger.info("Emulator started")
        
        # 首次安装应用
        if not manager.install_app():
            logger.error("初始应用安装失败，退出程序")
            manager.stop_emulator()
            sys.exit(1)
        
        # 启动Evaluator进程
        evaluator.start()
        
        consecutive_failures = 0
        emulator_restarts = 0
        MAX_RESTARTS = args.max_restarts
        
        # 创建进度条
        pbar = None
        
        try:
            # 等待任务执行状态
            while True:
                status, data = status_queue.get()
                
                if status == "start":
                    # 初始化进度条
                    pbar = tqdm(
                        total=data["total_tasks"],
                        desc="评测进度",
                        unit="task",
                        position=0,
                        leave=True,
                        ncols=100,  # 设置进度条宽度
                        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                    )
                    start_time = data["start_time"]
                elif status == "progress":
                    # 更新进度条
                    if pbar:
                        pbar.update(1)
                        # 截断较长的任务名称，保留前20个字符
                        threshold_length = 7
                        task_name = data['current_task'][:threshold_length] + '...' if len(data['current_task']) > threshold_length else data['current_task']
                        task_desc = f"任务: {task_name} ({data['current_trial']}/{evaluator.n_trials})"
                        pbar.set_description_str(task_desc)
                elif status == "task_failed":
                    consecutive_failures += 1
                    logger.info(f"任务 {data} 失败，连续失败数: {consecutive_failures}")
                    
                    if consecutive_failures >= MAX_CONSECUTIVE_RETRIES:
                        if emulator_restarts >= MAX_RESTARTS:
                            logger.error(f"模拟器重启次数已达到上限 {MAX_RESTARTS}，终止程序")
                            break
                            
                        logger.warning(f"连续{MAX_CONSECUTIVE_RETRIES}个任务失败，重启模拟器 (重启次数: {emulator_restarts + 1}/{MAX_RESTARTS})")
                        manager.stop_emulator()
                        time.sleep(2)
                        
                        if not manager.start_emulator():
                            logger.error("模拟器重启失败，退出程序")
                            break
                        
                        if not manager.install_app():
                            logger.error("重启后应用安装失败，退出程序")
                            manager.stop_emulator()
                            break
                            
                        consecutive_failures = 0
                        emulator_restarts += 1
                        
                elif status == "task_success":
                    consecutive_failures = 0
                    logger.info(f"任务 {data} 成功")
                    
                elif status == "all_tasks_completed":
                    logger.info("所有任务执行完成，退出程序")
                    break
                elif status == "save_stats":
                    # 来自信号处理器的保存请求
                    evaluator.save_stats()
                    
        except Exception as e:
            logger.error(f"执行过程中出错: {str(e)}")
            raise
        finally:
            if pbar:
                pbar.close()
            
            # 关闭模拟器
            logger.info("Stopping emulator...")
            manager.stop_emulator()
            logger.info("Emulator stopped")
            
            # 如果是因为重启次数过多而退出，返回非零状态码
            if emulator_restarts >= MAX_RESTARTS:
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        if 'evaluator' in locals():
            logger.info("保存当前统计结果...")
            evaluator.save_stats()
        if 'manager' in locals():
            logger.info("关闭模拟器...")
            manager.stop_emulator()
        sys.exit(1)


if __name__ == "__main__":
    main()
