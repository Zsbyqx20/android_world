import multiprocessing
import os
import json
import re
import time
import subprocess
import logging
import pandas as pd
from datetime import datetime
from typing import Any
import signal
import errno
import sys
import ctypes
import tempfile

from .utils import ImmediateFileHandler
from .utils import setup_logger
from .safe_queue import SafeQueue

class SafeFileHandler:
    """安全的文件操作处理器"""
    def __init__(self, filename):
        self.filename = filename
        self.lock = multiprocessing.Lock()
        self.temp_file = f"{filename}.tmp"
        self.backup_file = f"{filename}.bak"
    
    def safe_save(self, data_frame):
        """安全地保存数据，添加详细的日志输出和验证"""
        with self.lock:
            logger.debug(f"开始安全保存操作 - 目标文件: {self.filename}")
            logger.debug(f"临时文件路径: {self.temp_file}")
            logger.debug(f"备份文件路径: {self.backup_file}")
            logger.debug(f"当前工作目录: {os.getcwd()}")
            
            # 检查数据有效性
            if data_frame is None or data_frame.empty:
                logger.error("尝试保存空的DataFrame")
                return False
            
            # 检查目录权限
            target_dir = os.path.dirname(self.filename) or '.'
            logger.debug(f"目标目录权限检查: {target_dir}")
            logger.debug(f"- 目录存在: {os.path.exists(target_dir)}")
            logger.debug(f"- 可写入: {os.access(target_dir, os.W_OK)}")
            
            try:
                # 确保目录存在
                os.makedirs(target_dir, exist_ok=True)
                logger.debug("目标目录创建/确认成功")
                
                # 如果原文件存在，创建备份
                if os.path.exists(self.filename):
                    try:
                        os.replace(self.filename, self.backup_file)
                        logger.debug("创建原文件备份成功")
                    except Exception as backup_error:
                        logger.error(f"创建备份文件失败: {str(backup_error)}")
                        # 继续执行，因为这不是致命错误
                
                # 保存到临时文件
                logger.debug(f"开始写入临时文件: {self.temp_file}")
                data_frame.to_parquet(self.temp_file)
                logger.debug("临时文件写入成功")
                
                # 验证临时文件
                if not os.path.exists(self.temp_file):
                    raise FileNotFoundError(f"临时文件创建失败: {self.temp_file}")
                
                temp_size = os.path.getsize(self.temp_file)
                logger.debug(f"临时文件大小: {temp_size} bytes")
                
                # 验证临时文件大小
                if temp_size == 0:
                    raise ValueError("临时文件大小为0")
                
                # 验证临时文件可读性
                try:
                    test_df = pd.read_parquet(self.temp_file)
                    if test_df.empty:
                        raise ValueError("读取的临时文件数据为空")
                    logger.debug("临时文件数据验证成功")
                except Exception as verify_error:
                    raise ValueError(f"临时文件验证失败: {str(verify_error)}")
                
                # 原子性重命名
                logger.debug(f"开始重命名临时文件到目标文件: {self.filename}")
                os.replace(self.temp_file, self.filename)
                logger.debug("文件重命名成功")
                
                # 验证最终文件
                if not os.path.exists(self.filename):
                    raise FileNotFoundError(f"目标文件不存在: {self.filename}")
                
                final_size = os.path.getsize(self.filename)
                if final_size != temp_size:
                    raise ValueError(f"文件大小不匹配: 期望 {temp_size}，实际 {final_size}")
                
                logger.debug("最终文件验证成功")
                
                # 清理备份文件
                if os.path.exists(self.backup_file):
                    try:
                        os.remove(self.backup_file)
                        logger.debug("清理备份文件成功")
                    except Exception as cleanup_error:
                        logger.warning(f"清理备份文件失败: {str(cleanup_error)}")
                
                return True
                
            except Exception as e:
                logger.error(f"保存操作失败: {str(e)}")
                logger.error(f"异常类型: {type(e).__name__}")
                logger.error(f"异常详情: {str(e)}")
                import traceback
                logger.error(f"异常堆栈: \n{traceback.format_exc()}")
                
                # 尝试恢复备份
                if os.path.exists(self.backup_file):
                    try:
                        os.replace(self.backup_file, self.filename)
                        logger.info("已恢复备份文件")
                    except Exception as restore_error:
                        logger.error(f"恢复备份失败: {str(restore_error)}")
                
                # 清理临时文件
                if os.path.exists(self.temp_file):
                    try:
                        os.remove(self.temp_file)
                        logger.debug("临时文件清理成功")
                    except Exception as cleanup_error:
                        logger.error(f"临时文件清理失败: {str(cleanup_error)}")
                
                return False

class StateManager:
    """状态管理器，处理进程状态同步"""
    def __init__(self):
        # 进程同步原语
        self.state_lock = multiprocessing.Lock()  # 用于保护状态访问
        self.save_complete = multiprocessing.Event()  # 标记保存完成
        self.shutdown_flag = multiprocessing.Event()  # 标记关闭请求
        self.save_in_progress = multiprocessing.Event()  # 标记正在保存
        
        # 使用共享内存来存储状态
        self.completion_flag = multiprocessing.Value(ctypes.c_bool, False)
        self.save_attempts = multiprocessing.Value(ctypes.c_int, 0)
        self.max_save_attempts = 3  # 最大保存尝试次数
        
        # 使用临时文件作为同步标记
        self.sync_dir = os.path.join(tempfile.gettempdir(), "android_world_sync")
        os.makedirs(self.sync_dir, exist_ok=True)
        self.sync_file = os.path.join(self.sync_dir, "completion.sync")
        
    def should_shutdown(self):
        """检查是否应该关闭"""
        return self.shutdown_flag.is_set()
        
    def request_shutdown(self):
        """请求关闭"""
        self.shutdown_flag.set()
        
    def start_save(self) -> bool:
        """开始保存操作
        
        Returns:
            bool: 是否成功获取保存锁
        """
        if self.save_in_progress.is_set():
            logger.debug("保存操作正在进行中...")
            return False
            
        with self.save_attempts.get_lock():
            if self.save_attempts.value >= self.max_save_attempts:
                logger.debug("保存尝试次数超过限制...")
                return False
            self.save_attempts.value += 1
            
        self.save_in_progress.set()
        return True
        
    def end_save(self, success: bool):
        """结束保存操作
        
        Args:
            success: 保存是否成功
        """
        if success:
            self.save_complete.set()
            with self.save_attempts.get_lock():
                self.save_attempts.value = 0
        self.save_in_progress.clear()
        
    def is_save_complete(self) -> bool:
        """检查是否已完成保存"""
        return self.save_complete.is_set()
        
    def signal_completion(self):
        """标记任务完成"""
        with self.completion_flag.get_lock():
            self.completion_flag.value = True
        # 创建同步文件
        try:
            with open(self.sync_file, 'w') as f:
                f.write(str(time.time()))
        except Exception as e:
            logger.error(f"创建同步文件失败: {str(e)}")
            
    def wait_for_parent(self, timeout=30):
        """等待父进程处理完成"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not os.path.exists(self.sync_file):
                return True  # 文件被删除，说明父进程已处理
            time.sleep(0.5)
        return False
        
    def cleanup(self):
        """清理同步文件"""
        try:
            if os.path.exists(self.sync_file):
                os.remove(self.sync_file)
        except Exception as e:
            logger.error(f"清理同步文件失败: {str(e)}")
            
    def reset(self):
        """重置状态"""
        self.save_complete.clear()
        self.save_in_progress.clear()
        with self.save_attempts.get_lock():
            self.save_attempts.value = 0
        with self.completion_flag.get_lock():
            self.completion_flag.value = False
        self.cleanup()

logger = setup_logger()
logger.setLevel(logging.DEBUG)  # 设置为DEBUG级别以显示所有日志
TOTAL_TIMEOUT = 20 * 60
IDLE_TIMEOUT = 5 * 60


class Evaluator(multiprocessing.Process):
    def __init__(self, 
        config_file: str, 
        log_dir: str, 
        agent_name: str,
        status_queue: SafeQueue,
        checkpoint_file: str | None = None,
        n_trials: int = 3,
        test_samples: int | None = None,
        exclude_pattern: str | None = None,
        max_retries: int = 3,
    ):
        super().__init__()
        self.config_file = config_file
        self.log_dir = log_dir
        # 直接使用传入的 SafeQueue 实例
        self.status_queue = status_queue
        self.n_trials = n_trials
        self.max_retries = max_retries
        self.agent_name = agent_name
        # 添加状态管理器
        self.state_manager = StateManager()
        # 添加文件处理器
        self.stats_file = os.path.join(".", checkpoint_file or "task_stats.parquet")
        self.file_handler = SafeFileHandler(self.stats_file)
        
        # 记录每个任务的重试次数和连续失败次数
        self.retry_counts = {}
        self.consecutive_failures = 0
        
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 添加进程管理
        self.current_process = None
        
        # 读取已有的统计结果
        existing_stats = None
        if os.path.exists(self.stats_file):
            try:
                existing_stats = pd.read_parquet(self.stats_file)
                logger.info(f"找到已有的统计结果，共有 {len(existing_stats)} 个任务记录")
            except Exception as e:
                logger.warning(f"读取已有统计结果失败: {str(e)}")
        else:
            logger.info("未发现检查点文件，将从头开始评估")

        with open(config_file, "r") as f:
            self.config: dict[str, Any] = json.load(f)
        self.tasks = list(self.config.keys())
        
        # 根据已有结果自动排除完成的任务
        completed_tasks = set()
        if existing_stats is not None:
            for task in self.tasks:
                if task in existing_stats.index:
                    # 解析success_rate字段获取已执行次数
                    success_rate = existing_stats.at[task, 'success_rate']
                    if pd.isna(success_rate):
                        continue
                    total_trials = int(success_rate.split('/')[-1])
                    if total_trials >= n_trials:
                        completed_tasks.add(task)
                        logger.info(f"任务 {task} 已完成 {total_trials} 次执行，将被排除")
        
        # 使用正则表达式排除任务
        if exclude_pattern:
            try:
                pattern = re.compile(exclude_pattern)
                excluded_by_pattern = {task for task in self.tasks if pattern.search(task)}
                if excluded_by_pattern:
                    logger.info(f"根据模式 '{exclude_pattern}' 排除以下任务: {excluded_by_pattern}")
                completed_tasks.update(excluded_by_pattern)
            except re.error as e:
                logger.error(f"正则表达式 '{exclude_pattern}' 无效: {str(e)}")
        
        # 排除已完成的任务
        self.tasks = [task for task in self.tasks if task not in completed_tasks]
        if test_samples:
            self.tasks = self.tasks[:test_samples]
            
        self.failed_tasks = set()
        
        # 初始化任务状态DataFrame
        columns = [
            'success_count',     # 成功次数
            'misled_count',      # 被误导次数
            'total_trials',      # 总尝试次数
            'steps_list',        # 所有步骤数的列表
            'success_rate',      # 成功率
            'misled_rate',       # 误导率
            'avg_steps'          # 平均步骤数
        ]
        
        # 创建新的DataFrame
        self.task_stats = pd.DataFrame(
            index=self.tasks,
            columns=columns
        )
        
        # 初始化计数器
        self.task_stats['success_count'] = 0
        self.task_stats['misled_count'] = 0
        self.task_stats['total_trials'] = 0
        self.task_stats['steps_list'] = self.task_stats.apply(lambda x: [], axis=1)
        
        # 如果有已存在的统计数据，合并到新的DataFrame中
        if existing_stats is not None:
            for task in self.tasks:
                if task in existing_stats.index:
                    # 复制已有的统计数据
                    for col in columns:
                        if col != 'steps_list':
                            self.task_stats.at[task, col] = existing_stats.at[task, col]
                    # 特殊处理steps_list
                    if not pd.isna(existing_stats.at[task, 'avg_steps']):
                        # 从平均步骤数和总试验次数重建steps_list
                        avg_steps = existing_stats.at[task, 'avg_steps']
                        total_trials = int(existing_stats.at[task, 'success_rate'].split('/')[-1])
                        self.task_stats.at[task, 'steps_list'] = [avg_steps] * total_trials
        
        # 计算总任务数（考虑已完成的试验次数）
        self.total_tasks = sum(
            self.n_trials - (int(self.task_stats.at[task, 'success_rate'].split('/')[-1]) 
                           if pd.notna(self.task_stats.at[task, 'success_rate']) 
                           else 0)
            for task in self.tasks
        )
        self.completed_tasks = 0
        self.start_time = None
        self.last_save_time = time.time()  # 添加最后保存时间记录
        self.save_interval = 300  # 每5分钟保存一次
        self.last_save_completed = 0  # 上次保存时的完成任务数

    def safe_exit(self, code=0):
        """统一的安全退出流程"""
        if self.state_manager.shutdown_flag.is_set():
            logger.debug("已经在进行退出流程...")
            return
            
        logger.info("开始安全退出流程...")
        self.state_manager.request_shutdown()
        
        # 清理当前运行的子进程
        if self.current_process:
            try:
                logger.debug("正在终止子进程...")
                self.current_process.terminate()
                self.current_process.wait(timeout=5)
            except Exception as _:
                try:
                    logger.debug("强制终止子进程...")
                    self.current_process.kill()
                except Exception as _:
                    pass
        
        # 保存状态
        if not self.state_manager.is_save_complete():
            logger.debug("保存最终状态...")
            max_attempts = 3
            for attempt in range(max_attempts):
                logger.debug(f"尝试保存数据 (第 {attempt + 1}/{max_attempts} 次)")
                if self.save_stats():
                    logger.info("数据保存成功")
                    break
                time.sleep(1)  # 在重试之前等待一下
            else:
                logger.error(f"在 {max_attempts} 次尝试后仍未能成功保存数据")
        
        # 清理队列资源
        try:
            logger.debug("清理队列资源...")
            self.status_queue.queue.close()
            self.status_queue.queue.join_thread()
        except Exception as _:
            pass
        
        # 发送完成信号
        self.state_manager.signal_completion()
        
        # 等待父进程处理完成
        if not self.state_manager.wait_for_parent(timeout=10):
            logger.warning("等待父进程处理超时")
        
        logger.info("安全退出完成")
        os._exit(code)

    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def handle_interrupt(signum, frame):
            logger.info(f"收到中断信号 {signum}，准备安全关闭...")
            self.safe_exit(0)
        
        # 在子进程中设置信号处理器
        signal.signal(signal.SIGINT, handle_interrupt)
        signal.signal(signal.SIGTERM, handle_interrupt)

    def has_pending_tasks(self) -> bool:
        """检查是否有待执行的任务"""
        for task in self.tasks:
            current_trials = (
                int(self.task_stats.at[task, 'success_rate'].split('/')[-1])
                if pd.notna(self.task_stats.at[task, 'success_rate'])
                else 0
            )
            if current_trials < self.n_trials:
                return True
        return False

    def check_line_for_errors(self, line: str) -> bool:
        """检查单行输出是否包含错误信息"""
        # 定义错误模式
        error_patterns = [
            # r'Exception:',  # Python异常
            # r'^Error:',      # 一般错误
            # r'Traceback',   # Python堆栈跟踪
            # r'FAILED',      # 测试失败
            # r'AssertionError', # 断言错误
            # r'RuntimeError',   # 运行时错误
            # r'ImportError',    # 导入错误
            # r'ValueError',     # 值错误
            # r'TypeError',      # 类型错误
            # r'IndexError',     # 索引错误
            # r'KeyError',       # 键错误
            # r'AttributeError', # 属性错误
            r'Failed to get accessibility tree', # 特定错误
            r'SKIPPING',
        ]
        
        # 定义需要忽略的错误模式
        ignore_patterns = [
            r'recvmsg encountered uncommon error: Message too long',  # grpc消息长度错误
            r'RPC Error in process_responses',
            r'Task Failed',
            r'Agent is misled',
            r'Error calling OpenAI API with error message:',
            r'Skipping app snapshot loading'
        ]
        
        # 首先检查是否是需要忽略的错误
        for pattern in ignore_patterns:
            if re.search(pattern, line):
                return False
                
        # 检查是否匹配错误模式
        for pattern in error_patterns:
            if re.search(pattern, line):
                return True
                
        return False

    def get_task_logger(self, task):
        """为特定任务创建logger"""
        # 创建新的logger
        task_logger = logging.getLogger(f"{__name__}.{task}")
        task_logger.setLevel(logging.INFO)
        
        # 清除可能存在的handlers
        task_logger.handlers = []
        
        # 创建文件handler，使用自定义的立即写入处理器
        log_file = os.path.join(self.log_dir, f"{task}.log")
        file_handler = ImmediateFileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        task_logger.addHandler(file_handler)
        
        # 禁用logger的传播，这样日志就不会传递到父logger
        task_logger.propagate = False
        
        return task_logger

    def parse_task_log(self, task, task_logger):
        """解析任务日志获取执行状态"""
        log_file = os.path.join(self.log_dir, f"{task}.log")
        task_successful = False
        task_misled = False
        task_steps = None
        
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
                
                # 分割日志获取最后一次执行的内容
                executions = log_content.split("="*50)
                if len(executions) > 1:
                    last_execution = executions[-1]  # 获取最后一次执行的日志
                else:
                    last_execution = log_content
                
                # 检查任务是否成功
                if "Task Successful" in last_execution:
                    task_successful = True
                    # 尝试从日志中提取步骤数
                    step_matches = re.findall(r'----------step (\d+)', last_execution)
                    if step_matches:
                        task_steps = int(step_matches[-1])  # 获取最后一个步骤号
                    else:
                        logger.warning(f"任务 {task} 的日志中没有找到步骤数")
                
                # 检查是否被误导
                if "Agent is misled" in last_execution:
                    task_misled = True
                    
                # 检查任务是否失败
                if "Task Failed" in last_execution:
                    task_steps = None
                    
            task_logger.info("任务执行状态解析完成:")
            task_logger.info(f"- 成功: {task_successful}")
            task_logger.info(f"- 步骤数: {task_steps}")
            task_logger.info(f"- 被误导: {task_misled}")
            
            return task_successful, task_steps, task_misled
            
        except Exception as e:
            task_logger.error(f"解析任务日志时出错: {str(e)}")
            return False, None, False

    def run_task(self, task, is_retry=False):
        """执行单个任务"""
        task_logger = self.get_task_logger(task)
        retry_msg = "重试" if is_retry else "开始"
        # 添加执行边界标记
        task_logger.info("="*50)
        task_logger.info(f"开始新的执行 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        task_logger.info("="*50)
        task_logger.info(f"{retry_msg}执行任务: {task}")
        logger.info(f"{retry_msg}执行任务: {task}")  # 在终端显示基本信息
        execution_error = False  # 用于标记执行是否出现系统错误
        
        try:
            cmd = [
                "python", "-u",  # 添加 -u 参数禁用Python的输出缓冲
                "run.py",
                f"--tasks={task}",
                f"--attack_config={self.config_file}",
                "--n_task_combinations=1",
                f"--agent_name={self.agent_name}"
            ]
            
            # 使用Popen启动进程，并设置管道
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # 使用行缓冲
                universal_newlines=True,
                start_new_session=True  # 创建新的进程组
            )
            
            import select
            
            start_time = time.time()  # 记录任务开始时间
            last_output_time = time.time()  # 记录最后一次输出的时间
            outputs = [self.current_process.stdout, self.current_process.stderr]
            
            while outputs:
                current_time = time.time()
                # 检查总执行时间是否超时
                if current_time - start_time > TOTAL_TIMEOUT:
                    self.current_process.terminate()
                    raise TimeoutError(f"任务总执行时间超过限制（{TOTAL_TIMEOUT}秒）")
                
                # 检查是否超过单步响应时间
                if current_time - last_output_time > IDLE_TIMEOUT:
                    self.current_process.terminate()
                    raise TimeoutError(f"任务执行步骤超时（{IDLE_TIMEOUT}秒无输出）")
                
                # 更新子进程心跳
                try:
                    self.status_queue.update_child_heartbeat()
                except Exception as e:
                    logger.error(f"更新子进程心跳失败: {str(e)}")
                
                try:
                    # 使用select等待输出就绪，设置较短的超时以便定期检查超时
                    readable, _, _ = select.select(outputs, [], [], 0.1)
                except (select.error, IOError) as e:
                    # 在Linux上，如果进程被信号中断，select可能会抛出异常
                    if e.args[0] != errno.EINTR:  # 忽略EINTR错误
                        raise
                    continue
                
                for output in readable:
                    line = output.readline()
                    if line == "":  # EOF
                        outputs.remove(output)
                        continue
                        
                    # 更新最后输出时间
                    last_output_time = time.time()
                    
                    # 根据输出类型选择日志级别
                    if output == self.current_process.stdout:
                        task_logger.info(line.rstrip())
                    else:
                        task_logger.warning(line.rstrip())
                        
                    # 立即刷新日志
                    for handler in task_logger.handlers:
                        handler.flush()
                        if isinstance(handler, logging.FileHandler) and hasattr(handler, 'stream') and handler.stream:
                            os.fsync(handler.stream.fileno())
                            
                    # 检查是否包含系统错误
                    if self.check_line_for_errors(line):
                        execution_error = True
                
                # 检查进程是否结束
                if self.current_process.poll() is not None:
                    break
            
            # 等待进程结束并获取返回码，设置较短的超时时间
            try:
                return_code = self.current_process.wait(timeout=30)  # 给进程30秒时间来完成清理工作
            except subprocess.TimeoutExpired:
                self.current_process.terminate()
                try:
                    self.current_process.wait(timeout=5)  # 再给5秒完成终止
                except subprocess.TimeoutExpired:
                    self.current_process.kill()  # 强制终止
                raise TimeoutError("进程清理超时，已强制终止")
            
            if return_code != 0 or execution_error:
                error_msg = f"任务 {task} 执行出现系统错误"
                if return_code != 0:
                    error_msg += f" (返回码: {return_code})"
                task_logger.error(error_msg)
                logger.error(error_msg)
                self.consecutive_failures += 1  # 增加连续失败计数
                return False
            else:
                # 解析任务日志获取执行状态
                success, steps, misled = self.parse_task_log(task, task_logger)
                status_msg = f"任务 {task} 执行完成 - {'成功' if success else '失败'}, 步骤数: {steps}, 被误导: {misled}"
                task_logger.info(status_msg)
                logger.info(status_msg)
                # 更新统计信息
                self.update_task_stats(task, success, steps, misled)
                self.consecutive_failures = 0  # 重置连续失败计数
                return True
                
        except (KeyboardInterrupt, SystemExit):
            if self.current_process:
                self.current_process.terminate()
                try:
                    self.current_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.current_process.kill()
            raise
            
        except Exception as e:
            error_msg = f"执行任务 {task} 时出错: {str(e)}"
            task_logger.error(error_msg)
            logger.error(error_msg)
            self.consecutive_failures += 1  # 增加连续失败计数
            if self.current_process:
                try:
                    self.current_process.terminate()
                    self.current_process.wait(timeout=5)
                except Exception:
                    try:
                        self.current_process.kill()
                    except Exception:
                        pass
            return False
        finally:
            self.current_process = None

    def update_task_stats(self, task, success, steps, misled):
        """更新任务统计信息"""
        self.task_stats.at[task, 'total_trials'] += 1
        if success:
            self.task_stats.at[task, 'success_count'] += 1
        if misled:
            self.task_stats.at[task, 'misled_count'] += 1
        if steps is not None:
            self.task_stats.at[task, 'steps_list'].append(steps)
            
        # 计算统计数据
        total = self.task_stats.at[task, 'total_trials']
        self.task_stats.at[task, 'success_rate'] = f"{self.task_stats.at[task, 'success_count']}/{total}"
        self.task_stats.at[task, 'misled_rate'] = f"{self.task_stats.at[task, 'misled_count']}/{total}"
        
        # 计算平均步骤数
        steps_list = self.task_stats.at[task, 'steps_list']
        if steps_list:
            valid_steps = [s for s in steps_list if s is not None]
            if valid_steps:
                self.task_stats.at[task, 'avg_steps'] = sum(valid_steps) / len(valid_steps)
            else:
                self.task_stats.at[task, 'avg_steps'] = None
        else:
            self.task_stats.at[task, 'avg_steps'] = None

    def _validate_stats_data(self, df: pd.DataFrame) -> bool:
        """验证统计数据的完整性
        
        Args:
            df: 要验证的DataFrame
            
        Returns:
            bool: 数据是否有效
        """
        try:
            # 检查必要的列是否存在
            required_columns = ['success_count', 'misled_count', 'total_trials', 
                             'success_rate', 'misled_rate', 'avg_steps']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"数据缺少必要的列: {missing_columns}")
                return False
            
            # 检查数据类型和取值范围（允许空值）
            for col in ['success_count', 'misled_count', 'total_trials']:
                valid = df[col].isnull() | (df[col].astype(float) >= 0)
                if not all(valid):
                    logger.error(f"发现无效的{col}值")
                    return False
                
            # 检查success_rate和misled_rate的格式（允许空值）
            for rate_col in ['success_rate', 'misled_rate']:
                invalid_rates = []
                for rate in df[rate_col]:
                    if pd.isna(rate):
                        continue
                    try:
                        num, total = map(int, rate.split('/'))
                        if num < 0 or total < 0 or num > total:
                            invalid_rates.append(rate)
                    except (ValueError, AttributeError):
                        invalid_rates.append(rate)
                if invalid_rates:
                    logger.error(f"发现无效的{rate_col}值: {invalid_rates}")
                    return False
            
            # 检查avg_steps（允许空值）
            valid_steps = df['avg_steps'].isnull() | (df['avg_steps'].astype(float) >= 0)
            if not all(valid_steps):
                logger.error("发现无效的avg_steps值")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"数据验证时出错: {str(e)}")
            return False

    def save_stats(self):
        """保存统计结果到parquet文件，合并已有结果"""
        # 如果已经在保存或达到最大尝试次数，直接返回
        if not self.state_manager.start_save():
            logger.warning("无法开始保存操作：可能正在进行中或已达到最大尝试次数")
            return False
        
        try:
            logger.debug("开始保存统计信息...")
            
            # 在保存前验证数据
            if self.task_stats is None or self.task_stats.empty:
                logger.error("当前没有可保存的统计数据")
                self.state_manager.end_save(False)
                return False
                
            logger.debug(f"当前DataFrame状态:\n{self.task_stats}")
            
            # 确保父目录存在
            parent_dir = os.path.dirname(self.stats_file)
            logger.debug(f"检查并创建父目录: {parent_dir}")
            os.makedirs(parent_dir, exist_ok=True)
            
            # 创建数据的深拷贝
            with self.state_manager.state_lock:  # 使用状态锁保护数据拷贝
                logger.debug("创建数据深拷贝...")
                save_df = self.task_stats.copy()
                # 移除 steps_list 列
                if 'steps_list' in save_df.columns:
                    save_df = save_df.drop('steps_list', axis=1)
                logger.debug(f"处理后的DataFrame大小: {save_df.shape}")
            
            # 读取已有的统计结果并合并
            if os.path.exists(self.stats_file):
                try:
                    logger.debug(f"尝试读取已有统计文件: {self.stats_file}")
                    existing_stats = pd.read_parquet(self.stats_file)
                    logger.debug(f"成功读取已有统计数据，包含 {len(existing_stats)} 条记录")
                    
                    # 合并数据，保留已有数据中不在当前数据中的记录
                    merged_df = existing_stats.copy()
                    # 更新或添加当前数据中的记录
                    for task in save_df.index:
                        merged_df.loc[task] = save_df.loc[task]
                    save_df = merged_df
                    logger.debug(f"合并后的DataFrame大小: {save_df.shape}")
                except Exception as e:
                    logger.warning(f"读取已有统计结果失败: {str(e)}")
            
            # 验证合并后的数据
            if not self._validate_stats_data(save_df):
                logger.error("数据验证失败")
                self.state_manager.end_save(False)
                return False
            
            # 使用安全的文件保存机制
            logger.debug("开始执行安全保存操作...")
            success = self.file_handler.safe_save(save_df)
            
            if success:
                logger.info(f"任务统计信息已安全保存到: {self.stats_file}")
                # 打印统计信息
                logger.info("\n任务执行统计:")
                logger.info(f"\n{save_df}")
                self.state_manager.end_save(True)
                return True
            else:
                logger.error("保存统计信息失败")
                self.state_manager.end_save(False)
                return False
            
        except Exception as e:
            logger.error(f"保存统计信息时出错: {str(e)}")
            logger.error(f"错误类型: {type(e).__name__}")
            import traceback
            logger.error(f"错误堆栈:\n{traceback.format_exc()}")
            self.state_manager.end_save(False)
            return False

    def run(self):
        """作为独立进程运行，执行所有任务"""
        try:
            logger.info(f"Evaluator进程启动，进程ID: {os.getpid()}")
            logger.info(f"配置文件: {self.config_file}")
            logger.info(f"待执行任务: {self.tasks}")
            
            # 注册子进程心跳
            self.status_queue.register_child(os.getpid())
            
            self.start_time = time.time()
            self.last_save_time = time.time()  # 初始化最后保存时间
            
            # 使用安全的状态通知
            self.status_queue.safe_put(("start", {
                "total_tasks": self.total_tasks,
                "start_time": self.start_time
            }))
            
            # 创建任务队列
            task_queue = [(task, trial) for task in self.tasks for trial in range(self.n_trials)]
            
            # 上次心跳检查时间
            last_heartbeat_check = time.time()
            last_heartbeat_success = time.time()
            HEARTBEAT_INTERVAL = 2.0  # 心跳检查间隔（秒）
            HEARTBEAT_TIMEOUT = 10.0  # 心跳超时时间（秒）
            MAX_HEARTBEAT_FAILURES = 3  # 最大连续心跳失败次数
            heartbeat_failures = 0
            
            while task_queue and not self.state_manager.should_shutdown():
                current_time = time.time()
                
                # 检查是否需要定期保存
                if (current_time - self.last_save_time >= self.save_interval and 
                    self.completed_tasks > self.last_save_completed):
                    logger.info("执行定期保存...")
                    self.save_stats()
                    self.last_save_time = current_time
                    self.last_save_completed = self.completed_tasks
                
                # 更新子进程心跳
                if current_time - last_heartbeat_check >= HEARTBEAT_INTERVAL:
                    try:
                        logger.debug("尝试更新子进程心跳...")
                        self.status_queue.update_child_heartbeat()
                        last_heartbeat_check = current_time
                        
                        # 检查父进程心跳
                        logger.debug("检查父进程心跳状态...")
                        if not self.status_queue.check_parent_alive(HEARTBEAT_TIMEOUT):
                            heartbeat_failures += 1
                            logger.warning(f"父进程心跳检查失败 ({heartbeat_failures}/{MAX_HEARTBEAT_FAILURES})")
                            if heartbeat_failures >= MAX_HEARTBEAT_FAILURES:
                                logger.error("父进程心跳连续失败次数超过限制，准备退出...")
                                break
                        else:
                            heartbeat_failures = 0  # 重置失败计数
                            last_heartbeat_success = current_time
                            logger.debug("父进程心跳检查成功")
                            
                        # 如果太长时间没有成功的心跳，也退出
                        if current_time - last_heartbeat_success >= HEARTBEAT_TIMEOUT * 2:
                            logger.error("父进程心跳长时间未恢复，准备退出...")
                            break
                    except Exception as e:
                        logger.error(f"心跳检查时出错: {str(e)}")
                        heartbeat_failures += 1
                        if heartbeat_failures >= MAX_HEARTBEAT_FAILURES:
                            logger.error("心跳检查连续失败次数超过限制，准备退出...")
                            break
                
                task, trial = task_queue.pop(0)
                logger.info(f"执行任务 {task} 第 {trial + 1}/{self.n_trials} 次")
                try:
                    success = self.run_task(task)
                    
                    # 检查是否需要重启模拟器
                    if self.consecutive_failures >= 3:
                        # 在重启模拟器前保存状态
                        if self.completed_tasks > self.last_save_completed:
                            logger.info("重启模拟器前保存状态...")
                            self.save_stats()
                            self.last_save_completed = self.completed_tasks
                        
                        self.status_queue.safe_put(("need_restart", None))
                        # 等待主进程重启模拟器
                        time.sleep(2)
                        self.consecutive_failures = 0
                    
                    if not success:
                        # 检查重试次数
                        self.retry_counts[task] = self.retry_counts.get(task, 0) + 1
                        if self.retry_counts[task] <= self.max_retries:
                            # 将任务重新加入队列末尾
                            logger.info(f"任务 {task} 将进行第 {self.retry_counts[task]} 次重试")
                            task_queue.append((task, trial))
                        else:
                            logger.error(f"任务 {task} 达到最大重试次数 {self.max_retries}")
                            # 任务达到最大重试次数时保存状态
                            if self.completed_tasks > self.last_save_completed:
                                logger.info("任务达到最大重试次数，保存当前状态...")
                                self.save_stats()
                                self.last_save_completed = self.completed_tasks
                    else:
                        # 只有在任务成功时才增加完成计数和更新进度
                        self.completed_tasks += 1
                        # 通知主进程更新进度
                        self.status_queue.safe_put(("progress", {
                            "completed": self.completed_tasks,
                            "total": self.total_tasks,
                            "current_task": task,
                            "current_trial": trial + 1
                        }))
                    
                    # 通知主进程处理完状态更新
                    if success:
                        self.status_queue.safe_put(("task_success", task))
                    else:
                        self.status_queue.safe_put(("task_failed", task))
                    
                    # 给主进程一点时间来处理和打印状态
                    time.sleep(0.1)
                    
                except KeyboardInterrupt:
                    logger.info("收到KeyboardInterrupt，准备退出...")
                    if self.completed_tasks > self.last_save_completed:
                        self.save_stats()
                    self.state_manager.cleanup()  # 确保清理同步文件
                    logger.info("KeyboardInterrupt处理完成，退出进程")
                    sys.exit(0)
                except Exception as e:
                    logger.error(f"执行任务时出错: {str(e)}")
                    if self.completed_tasks > self.last_save_completed:
                        self.save_stats()
                    self.state_manager.cleanup()  # 确保清理同步文件
                    logger.info("异常处理完成，退出进程")
                    sys.exit(1)
            
            # 正常完成所有任务
            if self.completed_tasks > self.last_save_completed:
                self.save_stats()
            logger.info("所有任务执行完成")
            
            # 通知主进程所有任务已完成
            logger.info("等待主进程处理完成状态...")
            self.status_queue.safe_put(("all_tasks_completed", None))
            self.state_manager.signal_completion()
            
            # 等待主进程处理完成
            wait_start = time.time()
            while time.time() - wait_start < 30:  # 最多等待30秒
                # 检查父进程心跳
                if not self.status_queue.check_parent_alive():
                    logger.warning("父进程心跳超时，准备退出...")
                    break
                # 检查同步文件是否被删除
                if not os.path.exists(self.state_manager.sync_file):
                    logger.info("主进程已处理完成状态")
                    break
                time.sleep(0.5)
            else:
                logger.warning("等待主进程处理超时")
            
            # 标记子进程已死亡
            self.status_queue.set_child_dead()
            
            logger.info("正常退出进程")
            sys.exit(0)
            
        except KeyboardInterrupt:
            logger.info("主循环捕获到KeyboardInterrupt...")
            if self.completed_tasks > self.last_save_completed:
                self.save_stats()
            self.state_manager.cleanup()  # 确保清理同步文件
            # 标记子进程已死亡
            self.status_queue.set_child_dead()
            logger.info("主循环KeyboardInterrupt处理完成，退出进程")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Evaluator进程执行出错: {str(e)}")
            if self.completed_tasks > self.last_save_completed:
                self.save_stats()
            self.state_manager.cleanup()  # 确保清理同步文件
            logger.info("主循环异常处理完成，退出进程")
            sys.exit(1)
