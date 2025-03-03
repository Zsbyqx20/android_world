import multiprocessing
from typing import Any, Optional, Tuple

class SafeQueue:
    """安全的队列封装，提供可靠的进程间通信"""
    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.lock = multiprocessing.Lock()
        self.put_event = multiprocessing.Event()
    
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
            except Exception:
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
        except Exception:
            return None 