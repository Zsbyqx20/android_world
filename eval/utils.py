import logging
from typing import Literal
from termcolor import colored
import os


Color = Literal['grey', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']


class ColoredFormatter(logging.Formatter):
    """自定义的彩色日志格式化器"""
    
    # 定义不同日志级别的颜色
    COLORS: dict[str, Color] = {
        'DEBUG': 'grey',
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red',
    }
    
    def format(self, record):
        # 获取原始的日志消息
        message = super().format(record)
        
        # 根据日志级别选择颜色
        color = self.COLORS.get(record.levelname, 'white')
        
        # 为不同类型的消息添加不同的颜色处理
        if "任务执行失败" in message:
            return colored(message, 'red', attrs=['bold'])
        elif "成功" in message:
            return colored(message, 'green')
        elif "重试" in message:
            return colored(message, 'yellow')
        elif "开始执行任务" in message:
            return colored(message, 'cyan')
        elif "模拟器" in message:
            return colored(message, 'blue')
        else:
            return colored(message, color)


def setup_logger():
    """设置全局日志配置"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # 清除现有的处理器
    logger.handlers = []
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置彩色格式化器
    formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')  # 添加时间格式
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    return logger


class ImmediateFileHandler(logging.FileHandler):
    """立即写入的文件处理器"""
    def emit(self, record):
        super().emit(record)
        self.flush()
        # 强制写入到磁盘
        if self.stream:
            os.fsync(self.stream.fileno())
