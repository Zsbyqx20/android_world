import os
import subprocess
import sys
import time
import multiprocessing
from typing import Optional
from .utils import setup_logger

# 安全组件类
class SafeProcessManager:
    """安全的进程管理器"""
    def __init__(self):
        self.process_lock = multiprocessing.Lock()
        self.shutdown_event = multiprocessing.Event()
        self.process: Optional[subprocess.Popen] = None
    
    def start_process(self, cmd: list[str], **kwargs) -> Optional[subprocess.Popen]:
        with self.process_lock:
            if self.process is not None:
                return None
            try:
                self.process = subprocess.Popen(cmd, **kwargs)
                return self.process
            except Exception:
                return None
    
    def stop_process(self) -> bool:
        with self.process_lock:
            if self.process is None:
                return True
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)  # 等待进程终止
                except subprocess.TimeoutExpired:
                    self.process.kill()  # 强制终止
                return True
            except Exception:
                return False
            finally:
                self.process = None

class EmulatorStateManager:
    """模拟器状态管理器"""
    def __init__(self):
        self.state_lock = multiprocessing.Lock()
        self.ready_event = multiprocessing.Event()
        self._state = "stopped"
        
    def set_state(self, state: str):
        with self.state_lock:
            self._state = state
            if state == "ready":
                self.ready_event.set()
            elif state == "stopped":
                self.ready_event.clear()
            
    def get_state(self) -> str:
        with self.state_lock:
            return self._state
    
    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        return self.ready_event.wait(timeout=timeout)

class SafePathChecker:
    """安全的路径检查器"""
    def __init__(self):
        self.path_lock = multiprocessing.Lock()
    
    def check_path(self, path: str) -> bool:
        with self.path_lock:
            try:
                return os.path.exists(path)
            except Exception:
                return False
    
    def check_paths(self, paths: list[str]) -> bool:
        with self.path_lock:
            try:
                return all(os.path.exists(path) for path in paths)
            except Exception:
                return False

logger = setup_logger()
TOTAL_TIMEOUT = 20 * 60
IDLE_TIMEOUT = 5 * 60

class EmulatorManager:
    def __init__(
        self, 
        sdk_path: str | None = None, 
        app_path: str | None = None, 
        snapshot_name: str | None = None
    ):
        self.emulator_name = "AndroidWorldAvd"
        # 优先使用传入的sdk_path，其次使用环境变量，最后使用默认路径
        self.sdk_path = (
            sdk_path or 
            os.environ.get("ANDROID_HOME") or 
            os.path.expanduser("~/Library/Android/Sdk")
        )
        self.emulator_path = os.path.join(self.sdk_path, "emulator/emulator")
        self.process = None
        # 优先使用传入的app_path，其次使用环境变量，最后使用默认路径
        self.app_path = (
            app_path or
            os.environ.get("ANDROID_APP_PATH") or
            "debug/app-release.apk"
        )
        # snapshot相关配置
        self.snapshot_name = snapshot_name
        self.path_checker = SafePathChecker()
        self.state_manager = EmulatorStateManager()
        self.process_manager = SafeProcessManager()

    def check_paths(self):
        """检查SDK和APP路径是否有效"""
        paths_to_check = [
            (self.sdk_path, "Android SDK路径"),
            (self.emulator_path, "模拟器程序"),
            (self.app_path, "应用文件")
        ]
        
        for path, description in paths_to_check:
            if not self.path_checker.check_path(path):
                logger.error(f"找不到{description}: {path}")
                return False
        return True

    def is_emulator_active(self):
        """检查模拟器是否处于活动状态"""
        try:
            # 使用adb devices命令检查设备状态
            result = subprocess.run(
                ["adb", "devices"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                logger.error("检查模拟器状态时出错")
                return False
                
            # 检查输出中是否包含emulator并且状态为device
            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:  # 跳过第一行（标题行）
                if 'emulator' in line and 'device' in line:
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"检查模拟器状态时发生异常: {str(e)}")
            return False

    def wait_for_emulator_ready(self, timeout=60):
        """等待模拟器完全启动并就绪"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # 检查进程是否还在运行
                if not self.process_manager.process or self.process_manager.process.poll() is not None:
                    logger.error("模拟器进程已终止")
                    return False
                
                # 首先检查设备状态
                status = self.check_device_status()
                if status == 'offline':
                    logger.warning("模拟器处于offline状态，等待恢复...")
                    time.sleep(2)
                    continue
                elif status == 'not_found':
                    logger.warning("未检测到模拟器设备，等待设备连接...")
                    time.sleep(2)
                    continue
                    
                # 检查系统启动完成
                result = subprocess.run(
                    ["adb", "shell", "getprop", "sys.boot_completed"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5  # 添加超时限制
                )
                if result.stdout.strip() == "1":
                    logger.info("模拟器系统启动完成")
                    return True
            except subprocess.TimeoutExpired:
                logger.warning("检查系统启动状态超时")
            except Exception as e:
                logger.warning(f"等待模拟器就绪时发生错误: {str(e)}")
            time.sleep(2)
        
        logger.error(f"等待模拟器就绪超时（{timeout}秒）")
        return False

    def start_emulator(self):
        """启动Android模拟器"""
        # 首先检查路径
        if not self.check_paths():
            return False

        # 如果模拟器已经在运行，直接返回
        if self.is_emulator_active():
            logger.info("模拟器已经在运行")
            return True

        # 先尝试终止可能存在的僵尸进程
        self.stop_emulator()
        time.sleep(2)  # 等待之前的进程完全终止

        cmd = [
            self.emulator_path,
            "-avd", self.emulator_name,
            "-no-audio",
            "-grpc", "8554",
            "-gpu", "host"
        ]

        # 根据snapshot配置添加相应的参数
        if self.snapshot_name is not None:
            cmd.extend([
                "-snapshot", self.snapshot_name,
                "-no-snapshot-save"  # 不保存退出时的状态
            ])
        else:
            cmd.append("-no-snapshot")  # 不使用snapshot时，完全冷启动

        try:
            self.state_manager.set_state("starting")
            with open(os.devnull, 'w') as devnull:
                kwargs = {
                    'stdout': devnull,
                    'stderr': devnull,
                    'start_new_session': True
                }
                
                if sys.platform == 'darwin':
                    kwargs['preexec_fn'] = os.setpgrp
                
                process = self.process_manager.start_process(cmd, **kwargs)
                if process is None:
                    logger.error("启动模拟器进程失败")
                    self.state_manager.set_state("stopped")
                    return False
                
                logger.info(f"模拟器启动中，进程ID: {process.pid}")
                if self.snapshot_name is not None:
                    logger.info(f"正在加载快照: {self.snapshot_name}")
                
                # 等待模拟器就绪
                if self.wait_for_emulator_ready():
                    logger.info("模拟器启动成功并就绪")
                    self.state_manager.set_state("ready")
                    return True
                else:
                    logger.error("模拟器启动超时")
                    self.stop_emulator()
                    return False
                    
        except Exception as e:
            logger.error(f"启动模拟器时出错: {str(e)}")
            self.state_manager.set_state("stopped")
            return False

    def stop_emulator(self):
        """停止Android模拟器"""
        try:
            self.state_manager.set_state("stopping")
            # 首先尝试使用adb优雅地关闭模拟器
            if self.is_emulator_active():
                try:
                    subprocess.run(
                        ["adb", "emu", "kill"], 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        timeout=10  # 设置超时时间
                    )
                    
                    # 等待模拟器完全关闭
                    for _ in range(5):  # 最多等待10秒
                        if not self.is_emulator_active():
                            logger.info("模拟器已正常关闭")
                            self.state_manager.set_state("stopped")
                            return True
                        time.sleep(2)
                except subprocess.TimeoutExpired:
                    logger.warning("adb关闭模拟器超时")
                except Exception as e:
                    logger.warning(f"使用adb关闭模拟器失败: {str(e)}")
            
            # 如果adb关闭失败或者超时，使用进程管理器强制终止
            if self.process_manager.stop_process():
                logger.info("已强制终止模拟器进程")
            else:
                logger.warning("强制终止模拟器进程失败，尝试使用系统命令")
                # 最后尝试使用系统命令强制终止
                try:
                    subprocess.run(
                        ["pkill", "-f", f"emulator -avd {self.emulator_name}"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=5
                    )
                    subprocess.run(
                        ["pkill", "-f", "qemu-system-"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=5
                    )
                    logger.info("已使用系统命令终止所有模拟器进程")
                except Exception as e:
                    logger.error(f"终止模拟器进程失败: {str(e)}")
                    return False
            
            self.state_manager.set_state("stopped")
            return True
            
        except Exception as e:
            logger.error(f"停止模拟器时出错: {str(e)}")
            self.state_manager.set_state("stopped")
            return False

    def install_app(self):
        """安装应用"""
        try:
            logger.info("开始安装应用...")
            # 先尝试卸载已存在的应用
            uninstall_cmd = ["adb", "uninstall", "com.mobilellm.awattackerapplier"]
            subprocess.run(uninstall_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 安装应用
            install_cmd = ["adb", "install", "-r", self.app_path]
            result = subprocess.run(
                install_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode == 0 and "Success" in result.stdout:
                logger.info("应用安装成功")
                return True
            else:
                logger.error(f"应用安装失败: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"安装应用时出错: {str(e)}")
            return False

    def check_device_status(self):
        """检查设备状态，返回 'online', 'offline' 或 'not_found'"""
        try:
            result = subprocess.run(
                ["adb", "devices"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                logger.error("检查设备状态时出错")
                return 'not_found'
                
            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:  # 跳过第一行（标题行）
                if 'emulator' in line:
                    if 'offline' in line:
                        return 'offline'
                    elif 'device' in line:
                        return 'online'
                        
            return 'not_found'
            
        except Exception as e:
            logger.error(f"检查设备状态时发生异常: {str(e)}")
            return 'not_found'

    def restart_if_needed(self):
        """检查设备状态并在需要时重启模拟器"""
        status = self.check_device_status()
        if status != 'online':
            logger.warning(f"设备状态异常: {status}，尝试重启模拟器...")
            self.stop_emulator()
            time.sleep(2)
            if not self.start_emulator():
                logger.error("模拟器重启失败")
                return False
            if not self.install_app():
                logger.error("应用重新安装失败")
                return False
            logger.info("模拟器重启成功")
            return True
        return True

    def reload_snapshot(self):
        """重新加载快照"""
        if self.snapshot_name is None:
            return True
            
        logger.info(f"重新加载快照: {self.snapshot_name}")
        self.stop_emulator()
        time.sleep(2)  # 等待模拟器完全关闭
        
        if not self.start_emulator():
            logger.error("重新加载快照时启动模拟器失败")
            return False
            
        if not self.install_app():
            logger.error("重新加载快照后安装应用失败")
            return False
            
        logger.info("快照重新加载成功")
        return True

    def safe_shutdown(self):
        """安全关闭模拟器，确保所有资源都被正确释放"""
        # 检查当前状态，避免重复关闭
        current_state = self.state_manager.get_state()
        if current_state in ["stopped", "stopping"]:
            logger.debug(f"模拟器已经在关闭或已停止状态 ({current_state})，跳过关闭操作")
            return True
            
        logger.info("开始安全关闭模拟器...")
        try:
            # 设置关闭状态
            self.state_manager.set_state("stopping")
            
            # 尝试正常关闭
            success = self.stop_emulator()
            
            # 确保状态被设置为stopped
            self.state_manager.set_state("stopped")
            
            return success
        except Exception as e:
            logger.error(f"安全关闭模拟器时出错: {str(e)}")
            # 确保状态被设置为stopped
            self.state_manager.set_state("stopped")
            return False
