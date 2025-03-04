from argparse import ArgumentParser
import multiprocessing
import signal
import sys
import time
import os

from .emulator import EmulatorManager
from .evaluator import Evaluator
from .utils import setup_logger, create_progress, log_success, log_error, log_warning, log_emulator, log_task
from .safe_queue import SafeQueue

logger = setup_logger("main")
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
    # 添加信号处理状态标志
    shutdown_in_progress = False
    
    # 创建信号处理函数
    def signal_handler(signum, frame):
        nonlocal shutdown_in_progress
        if shutdown_in_progress:
            log_warning(logger, "关闭操作正在进行中，请耐心等待...")
            return
            
        shutdown_in_progress = True
        log_warning(logger, f"收到信号 {signum}，准备退出程序")
        
        try:
            if 'evaluator' in locals():
                # 通知子进程保存统计
                logger.info("正在保存统计信息...")
                evaluator.state_manager.request_shutdown()
                evaluator.save_stats()
                # 等待保存完成
                for _ in range(5):  # 最多等待5秒
                    if not evaluator.is_alive():
                        break
                    time.sleep(1)
                    
            if 'manager' in locals():
                log_emulator(logger, "正在关闭模拟器...")
                manager.safe_shutdown()
                
        except Exception as e:
            log_error(logger, f"关闭过程中出错: {str(e)}")
        finally:
            sys.exit(1)

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill

    try:
        parser = get_parser()
        args = parser.parse_args()
        
        # 创建状态队列
        status_queue = SafeQueue()
        
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
        
        log_emulator(logger, "Starting emulator...")
        if not manager.start_emulator():
            log_error(logger, "模拟器启动失败，退出程序")
            sys.exit(1)
        log_success(logger, "Emulator started")
        
        # 首次安装应用
        if not manager.install_app():
            log_error(logger, "初始应用安装失败，退出程序")
            manager.stop_emulator()
            sys.exit(1)
            
        # 注册父进程心跳
        status_queue.register_parent(os.getpid())
        
        # 创建评测器进程
        evaluator = Evaluator(
            config_file=args.config,
            log_dir=args.log_dir,
            agent_name=args.agent_name,
            status_queue=status_queue,  # 传递 SafeQueue 实例
            checkpoint_file=args.checkpoint_file,
            n_trials=args.repeat_times,
            test_samples=args.test_samples,
            exclude_pattern=args.exclude_pattern,
            max_retries=args.max_retries,
        )
        
        # 启动评测器进程
        evaluator_process = multiprocessing.Process(target=evaluator.run)
        evaluator_process.start()
        log_task(logger, f"Evaluator进程启动，进程ID: {evaluator_process.pid}")
        
        consecutive_failures = 0
        emulator_restarts = 0
        MAX_RESTARTS = args.max_restarts
        
        # 创建进度条
        progress = None
        task_id = None
        
        # 上次心跳检查时间
        last_heartbeat_check = time.time()
        last_heartbeat_success = time.time()
        HEARTBEAT_INTERVAL = 2.0  # 心跳检查间隔（秒）
        HEARTBEAT_TIMEOUT = 10.0  # 心跳超时时间（秒）
        MAX_HEARTBEAT_FAILURES = 3  # 最大连续心跳失败次数
        heartbeat_failures = 0  # 连续心跳失败计数
        
        # 等待子进程初始化
        time.sleep(1)
        
        try:
            # 等待任务执行状态
            while evaluator_process.is_alive():
                try:
                    # 更新父进程心跳
                    status_queue.update_parent_heartbeat()
                    
                    # 定期检查子进程心跳
                    current_time = time.time()
                    if current_time - last_heartbeat_check >= HEARTBEAT_INTERVAL:
                        if not status_queue.check_child_alive(HEARTBEAT_TIMEOUT):
                            heartbeat_failures += 1
                            log_warning(logger, f"子进程心跳检查失败 ({heartbeat_failures}/{MAX_HEARTBEAT_FAILURES})")
                            if heartbeat_failures >= MAX_HEARTBEAT_FAILURES:
                                log_error(logger, "子进程心跳连续失败次数超过限制，准备关闭...")
                                break
                        else:
                            heartbeat_failures = 0  # 重置失败计数
                            last_heartbeat_success = current_time
                        last_heartbeat_check = current_time
                        
                        # 如果太长时间没有成功的心跳，也退出
                        if current_time - last_heartbeat_success >= HEARTBEAT_TIMEOUT * 2:
                            log_error(logger, "子进程心跳长时间未恢复，准备关闭...")
                            break
                    
                    status_data = status_queue.safe_get(timeout=1)  # 缩短超时时间以便更频繁地检查心跳
                    if status_data is None:
                        continue
                        
                    status, data = status_data
                    
                    if status == "start":
                        # 初始化进度条
                        progress = create_progress()
                        task_id = progress.add_task(
                            description="评测进度",
                            total=data["total_tasks"]
                        )
                        progress.start()
                        
                    elif status == "progress" and progress is not None:
                        # 更新进度条
                        progress.update(
                            task_id,
                            completed=data["completed"],
                            description=f"评测进度 - 当前任务: {data['current_task']} ({data['current_trial']}/{args.repeat_times})"
                        )
                        
                    elif status == "task_success":
                        log_success(logger, f"任务 {data} 执行成功")
                        consecutive_failures = 0
                        
                    elif status == "task_failed":
                        log_error(logger, f"任务 {data} 执行失败")
                        consecutive_failures += 1
                        
                    elif status == "need_restart":
                        if emulator_restarts < MAX_RESTARTS:
                            log_emulator(logger, "正在重启模拟器...")
                            manager.restart_emulator()
                            emulator_restarts += 1
                            consecutive_failures = 0
                        else:
                            log_error(logger, f"模拟器重启次数超过限制 ({MAX_RESTARTS})，退出程序")
                            break
                            
                    elif status == "all_tasks_completed":
                        log_success(logger, "所有任务执行完成")
                        break
                        
                except Exception as e:
                    log_error(logger, f"主循环出错: {str(e)}")
                    break
                    
        finally:
            if progress is not None:
                progress.stop()
            
            # 等待子进程结束
            logger.info("等待子进程结束...")
            evaluator_process.join(timeout=5)
            if evaluator_process.is_alive():
                evaluator_process.terminate()
                evaluator_process.join(timeout=5)
                if evaluator_process.is_alive():
                    evaluator_process.kill()
            
            # 关闭模拟器
            log_emulator(logger, "正在关闭模拟器...")
            manager.safe_shutdown()
            
    except KeyboardInterrupt:
        log_warning(logger, "收到用户中断，准备退出程序")
        if 'evaluator_process' in locals() and evaluator_process.is_alive():
            evaluator_process.terminate()
        if 'manager' in locals():
            manager.safe_shutdown()
        sys.exit(1)
    except Exception as e:
        log_error(logger, f"程序执行出错: {str(e)}")
        if 'evaluator_process' in locals() and evaluator_process.is_alive():
            evaluator_process.terminate()
        if 'manager' in locals():
            manager.safe_shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
