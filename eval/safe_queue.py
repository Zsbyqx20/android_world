import multiprocessing
import time
import os
import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

class SafeQueue:
    """安全的队列封装，提供可靠的进程间通信和双向心跳检查"""
    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.lock = multiprocessing.Lock()
        self.put_event = multiprocessing.Event()
        # 心跳检查相关
        self.parent_alive = multiprocessing.Value('b', True)
        self.child_alive = multiprocessing.Value('b', True)
        self.last_parent_heartbeat = multiprocessing.Value('d', time.time())
        self.last_child_heartbeat = multiprocessing.Value('d', time.time())
        # 共享状态
        self.parent_pid = multiprocessing.Value('i', 0)
        self.child_pid = multiprocessing.Value('i', 0)
    
    def safe_put(self, item: Any, timeout: Optional[float] = None) -> bool:
        """安全地将数据放入队列
        
        Args:
            item: 要放入队列的数据
            timeout: 超时时间（秒）
            
        Returns:
            bool: 是否成功放入队列
        """
        with self.lock:
            try:
                self.queue.put(item, timeout=timeout)
                self.put_event.set()
                return True
            except Exception as e:
                logger.error(f"队列放入数据失败: {str(e)}")
                return False
    
    def safe_get(self, timeout: Optional[float] = None) -> Optional[Tuple[str, Any]]:
        """安全地从队列获取数据
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            Optional[Tuple[str, Any]]: 如果成功则返回(状态, 数据)元组，失败则返回None
        """
        try:
            return self.queue.get(timeout=timeout)
        except Exception as e:
            logger.debug(f"队列获取数据超时或失败: {str(e)}")
            return None
            
    def register_parent(self, pid: int):
        """注册父进程"""
        with self.parent_pid.get_lock():
            self.parent_pid.value = pid
            self.parent_alive.value = True
            self.last_parent_heartbeat.value = time.time()
            logger.debug(f"注册父进程 PID: {pid}")
            
    def register_child(self, pid: int):
        """注册子进程"""
        with self.child_pid.get_lock():
            self.child_pid.value = pid
            self.child_alive.value = True
            self.last_child_heartbeat.value = time.time()
            logger.debug(f"注册子进程 PID: {pid}")
            
    def update_parent_heartbeat(self):
        """更新父进程心跳时间戳"""
        with self.last_parent_heartbeat.get_lock():
            if self.parent_pid.value == 0:
                self.register_parent(os.getpid())
            current_time = time.time()
            old_time = self.last_parent_heartbeat.value
            self.last_parent_heartbeat.value = current_time
            logger.debug(f"更新父进程心跳 (PID: {self.parent_pid.value}, 旧时间: {old_time}, 新时间: {current_time})")
            
    def update_child_heartbeat(self):
        """更新子进程心跳时间戳"""
        with self.last_child_heartbeat.get_lock():
            if self.child_pid.value == 0:
                self.register_child(os.getpid())
            current_time = time.time()
            old_time = self.last_child_heartbeat.value
            self.last_child_heartbeat.value = current_time
            logger.debug(f"更新子进程心跳 (PID: {self.child_pid.value}, 旧时间: {old_time:.3f}, 新时间: {current_time:.3f}, 间隔: {current_time - old_time:.3f}秒)")
            
    def check_parent_alive(self, timeout: float = 10.0) -> bool:
        """检查父进程是否存活"""
        with self.parent_alive.get_lock():
            if not self.parent_alive.value:
                logger.debug("父进程已标记为死亡")
                return False
            with self.last_parent_heartbeat.get_lock():
                current_time = time.time()
                last_time = self.last_parent_heartbeat.value
                time_diff = current_time - last_time
                is_alive = time_diff < timeout
                logger.debug(f"检查父进程心跳 (PID: {self.parent_pid.value}, 当前时间: {current_time}, 上次心跳: {last_time}, 差值: {time_diff:.1f}秒, 存活: {is_alive})")
                if not is_alive:
                    self.parent_alive.value = False
                return is_alive
            
    def check_child_alive(self, timeout: float = 10.0) -> bool:
        """检查子进程是否存活"""
        with self.child_alive.get_lock():
            if not self.child_alive.value:
                logger.debug("子进程已标记为死亡")
                return False
            with self.last_child_heartbeat.get_lock():
                current_time = time.time()
                last_time = self.last_child_heartbeat.value
                time_diff = current_time - last_time
                is_alive = time_diff < timeout
                logger.debug(f"检查子进程心跳 (PID: {self.child_pid.value}, 当前时间: {current_time:.3f}, 上次心跳: {last_time:.3f}, 差值: {time_diff:.3f}秒, 存活: {is_alive})")
                if not is_alive:
                    self.child_alive.value = False
                return is_alive
            
    def set_parent_dead(self):
        """标记父进程已死亡"""
        with self.parent_alive.get_lock():
            self.parent_alive.value = False
            logger.debug(f"标记父进程已死亡 (PID: {self.parent_pid.value})")
            
    def set_child_dead(self):
        """标记子进程已死亡"""
        with self.child_alive.get_lock():
            self.child_alive.value = False
            logger.debug(f"标记子进程已死亡 (PID: {self.child_pid.value})")
            
    def is_parent_alive(self) -> bool:
        """获取父进程存活状态"""
        with self.parent_alive.get_lock():
            return self.parent_alive.value
            
    def is_child_alive(self) -> bool:
        """获取子进程存活状态"""
        with self.child_alive.get_lock():
            return self.child_alive.value 