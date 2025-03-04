import logging
import os
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install
from rich.theme import Theme
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    MofNCompleteColumn
)

# 安装更好的异常追踪
install(show_locals=True)

# 自定义主题
custom_theme = Theme({
    "info": "white",
    "warning": "yellow",
    "error": "red",
    "success": "green",
    "task": "cyan",
    "emulator": "blue",
    "progress.percentage": "green",
    "progress.remaining": "cyan",
    "progress.description": "white"
})

# 创建控制台实例
console = Console(theme=custom_theme)

def create_progress() -> Progress:
    """创建自定义进度条"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    )

def setup_logger(name: str = None):
    """设置全局日志配置
    
    Args:
        name: 日志器名称，默认为 None 使用 root logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    logger.handlers = []
    
    # 创建 rich 处理器
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        show_time=True,
        show_path=False,
        enable_link_path=True
    )
    rich_handler.setLevel(logging.INFO)
    
    # 设置格式化器
    formatter = logging.Formatter("%(message)s")
    rich_handler.setFormatter(formatter)
    
    logger.addHandler(rich_handler)
    return logger

class ImmediateFileHandler(logging.FileHandler):
    """立即写入的文件处理器，用于任务日志"""
    def emit(self, record):
        super().emit(record)
        self.flush()
        # 强制写入到磁盘
        if self.stream:
            os.fsync(self.stream.fileno())

def log_success(logger: logging.Logger, message: str):
    """记录成功消息"""
    logger.info(f"[success]{message}[/success]")

def log_error(logger: logging.Logger, message: str):
    """记录错误消息"""
    logger.error(f"[error]{message}[/error]")

def log_warning(logger: logging.Logger, message: str):
    """记录警告消息"""
    logger.warning(f"[warning]{message}[/warning]")

def log_task(logger: logging.Logger, message: str):
    """记录任务相关消息"""
    logger.info(f"[task]{message}[/task]")

def log_emulator(logger: logging.Logger, message: str):
    """记录模拟器相关消息"""
    logger.info(f"[emulator]{message}[/emulator]")
