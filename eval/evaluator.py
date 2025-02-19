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

from .utils import ImmediateFileHandler
from .utils import setup_logger


logger = setup_logger()
TOTAL_TIMEOUT = 20 * 60
IDLE_TIMEOUT = 5 * 60


class Evaluator(multiprocessing.Process):
    def __init__(self, 
        config_file: str, 
        log_dir: str, 
        agent_name: str,
        status_queue: multiprocessing.Queue, 
        checkpoint_file: str | None = None,
        n_trials: int = 3,
        test_samples: int | None = None,
        exclude_pattern: str | None = None,
        max_retries: int = 3,
    ):
        super().__init__()
        self.config_file = config_file
        self.log_dir = log_dir
        self.status_queue = status_queue
        self.n_trials = n_trials
        self.max_retries = max_retries
        self.agent_name = agent_name
        # 记录每个任务的重试次数和连续失败次数
        self.retry_counts = {}
        self.consecutive_failures = 0
        
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 读取已有的统计结果
        self.stats_file = os.path.join(".", checkpoint_file or "task_stats.parquet")
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
        # 检查是否已经完成足够的试验次数
        current_trials = (
            int(self.task_stats.at[task, 'success_rate'].split('/')[-1])
            if pd.notna(self.task_stats.at[task, 'success_rate'])
            else 0
        )
        if current_trials >= self.n_trials:
            logger.info(f"任务 {task} 已完成足够的试验次数 ({current_trials}/{self.n_trials})")
            return True

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
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # 使用行缓冲
                universal_newlines=True
            )
            
            import select
            
            start_time = time.time()  # 记录任务开始时间
            last_output_time = time.time()  # 记录最后一次输出的时间
            outputs = [process.stdout, process.stderr]
            
            while outputs:
                current_time = time.time()
                # 检查总执行时间是否超时
                if current_time - start_time > TOTAL_TIMEOUT:
                    process.kill()
                    raise TimeoutError(f"任务总执行时间超过限制（{TOTAL_TIMEOUT}秒）")
                
                # 检查是否超过单步响应时间
                if current_time - last_output_time > IDLE_TIMEOUT:
                    process.kill()
                    raise TimeoutError(f"任务执行步骤超时（{IDLE_TIMEOUT}秒无输出）")
                
                # 使用select等待输出就绪，设置较短的超时以便定期检查超时
                readable, _, _ = select.select(outputs, [], [], 0.1)
                
                for output in readable:
                    line = output.readline()
                    if line == "":  # EOF
                        outputs.remove(output)
                        continue
                        
                    # 更新最后输出时间
                    last_output_time = time.time()
                    
                    # 根据输出类型选择日志级别
                    if output == process.stdout:
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
                if process.poll() is not None:
                    break
            
            # 等待进程结束并获取返回码，设置较短的超时时间
            try:
                return_code = process.wait(timeout=30)  # 给进程30秒时间来完成清理工作
            except subprocess.TimeoutExpired:
                process.kill()
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
                
        except TimeoutError as e:
            error_msg = f"执行任务 {task} 时超时: {str(e)}"
            task_logger.error(error_msg)
            logger.error(error_msg)
            self.consecutive_failures += 1  # 增加连续失败计数
            return False
            
        except Exception as e:
            error_msg = f"执行任务 {task} 时出错: {str(e)}"
            task_logger.error(error_msg)
            logger.error(error_msg)
            self.consecutive_failures += 1  # 增加连续失败计数
            return False
            
        finally:
            # 确保进程被终止
            if process is not None:
                try:
                    process.kill()
                except Exception as _:
                    pass

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

    def save_stats(self):
        """保存统计结果到parquet文件，合并已有结果"""
        try:
            # 读取已有的统计结果
            existing_stats = None
            if os.path.exists(self.stats_file):
                try:
                    existing_stats = pd.read_parquet(self.stats_file)
                except Exception as e:
                    logger.warning(f"读取已有统计结果失败: {str(e)}")
            
            # 准备要保存的DataFrame
            save_df = self.task_stats.drop('steps_list', axis=1)
            
            # 如果有已存在的统计数据，合并结果
            if existing_stats is not None:
                # 更新现有数据
                for task in save_df.index:
                    existing_stats.loc[task] = save_df.loc[task]
                # 使用合并后的数据
                save_df = existing_stats
            
            # 保存到parquet文件
            save_df.to_parquet(self.stats_file)
            logger.info(f"任务统计信息已保存到: {self.stats_file}")
            
            # 打印统计信息
            logger.info("\n任务执行统计:")
            logger.info(f"\n{save_df}")
            
        except Exception as e:
            logger.error(f"保存统计信息时出错: {str(e)}")

    def run(self):
        """作为独立进程运行，执行所有任务"""
        try:
            logger.info(f"Evaluator进程启动，配置文件: {self.config_file}")
            logger.info(f"待执行任务: {self.tasks}")
            
            self.start_time = time.time()
            # 通知主进程开始执行
            self.status_queue.put(("start", {
                "total_tasks": self.total_tasks,
                "start_time": self.start_time
            }))
            
            # 创建任务队列
            task_queue = [(task, trial) for task in self.tasks for trial in range(self.n_trials)]
            
            while task_queue:
                task, trial = task_queue.pop(0)
                logger.info(f"执行任务 {task} 第 {trial + 1}/{self.n_trials} 次")
                try:
                    success = self.run_task(task)
                    
                    # 检查是否需要重启模拟器
                    if self.consecutive_failures >= 3:
                        self.status_queue.put(("need_restart", None))
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
                    else:
                        # 只有在任务成功时才增加完成计数和更新进度
                        self.completed_tasks += 1
                        # 通知主进程更新进度
                        self.status_queue.put(("progress", {
                            "completed": self.completed_tasks,
                            "total": self.total_tasks,
                            "current_task": task,
                            "current_trial": trial + 1
                        }))
                    
                    # 通知主进程处理完状态更新
                    if success:
                        self.status_queue.put(("task_success", task))
                    else:
                        self.status_queue.put(("task_failed", task))
                    
                    # 给主进程一点时间来处理和打印状态
                    time.sleep(0.1)
                except KeyboardInterrupt:
                    # 确保当前任务的统计信息被保存
                    self.save_stats()
                    raise
            
            # 保存统计结果
            self.save_stats()
            logger.info("所有任务执行完成")
            # 通知主进程所有任务已完成
            self.status_queue.put(("all_tasks_completed", None))
            
        except KeyboardInterrupt:
            # 确保统计信息被保存
            self.save_stats()
            raise
        except Exception as e:
            logger.error(f"Evaluator进程执行出错: {str(e)}")
            self.save_stats()
            raise
