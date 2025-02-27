import os
import subprocess
import sys
import time
from .utils import setup_logger

logger = setup_logger()


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

    def check_paths(self):
        """检查SDK和APP路径是否有效"""
        if not os.path.exists(self.sdk_path):
            logger.error(f"找不到Android SDK路径: {self.sdk_path}")
            return False
        if not os.path.exists(self.emulator_path):
            logger.error(f"找不到模拟器程序: {self.emulator_path}")
            return False
        if not os.path.exists(self.app_path):
            logger.error(f"找不到应用文件: {self.app_path}")
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
                    text=True
                )
                if result.stdout.strip() == "1":
                    logger.info("模拟器系统启动完成")
                    return True
            except Exception as e:
                logger.warning(f"等待模拟器就绪时发生错误: {str(e)}")
                pass
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
            "-gpu","host"
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
            with open(os.devnull, 'w') as devnull:
                if sys.platform == 'darwin':
                    self.process = subprocess.Popen(
                        cmd,
                        stdout=devnull,
                        stderr=devnull,
                        preexec_fn=os.setpgrp,
                        start_new_session=True
                    )
                elif sys.platform == 'linux':
                    self.process = subprocess.Popen(
                        cmd,
                        stdout=devnull,
                        stderr=devnull,
                        start_new_session=True
                    )
                else:
                    raise NotImplementedError(f"不支持的平台: {sys.platform}")
            logger.info(f"模拟器启动中，进程ID: {self.process.pid}")
            if self.snapshot_name is not None:
                logger.info(f"正在加载快照: {self.snapshot_name}")
            
            # 等待模拟器就绪
            if self.wait_for_emulator_ready():
                logger.info("模拟器启动成功并就绪")
                return True
            else:
                logger.error("模拟器启动超时")
                self.stop_emulator()
                return False
                
        except Exception as e:
            logger.error(f"启动模拟器时出错: {str(e)}")
            return False

    def stop_emulator(self):
        """停止Android模拟器"""
        try:
            # 首先尝试使用adb优雅地关闭模拟器
            if self.is_emulator_active():
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
                        return True
                    time.sleep(2)
            
            # 如果adb关闭失败或者超时，使用pkill强制终止
            subprocess.run(
                ["pkill", "-f", f"emulator -avd {self.emulator_name}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # 确保所有相关进程都被终止
            subprocess.run(
                ["pkill", "-f", "qemu-system-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            logger.info("已强制终止所有模拟器进程")
            return True
            
        except Exception as e:
            logger.error(f"停止模拟器时出错: {str(e)}")
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
