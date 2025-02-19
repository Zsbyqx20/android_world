import logging
import time
from concurrent import futures
from typing import List, Any, cast

import grpc
import json
import portpicker
import numpy as np
import shlex

from android_env import env_interface
from android_env.wrappers import base_wrapper
from android_env.components import action_type as android_action_type_lib
from android_env.proto import adb_pb2
from android_world.attack.models import AttackConfig, attack_config_to_rule
import dm_env

from android_world.env import adb_utils
from android_world.attack.proto.accessibility_pb2 import (
    AccessibilityTree,
)
from android_world.attack.proto.accessibility_pb2_grpc import (
    add_AccessibilityServiceServicer_to_server,
)
from android_world.attack.proto.window_info_pb2_grpc import (
    add_WindowInfoServiceServicer_to_server,
)

from android_world.attack.service import WindowInfoServiceImpl, AccessibilityServiceImpl


def broadcast_start_service(env: env_interface.AndroidEnvInterface) -> None:
    response = adb_utils.send_android_intent(
        command="broadcast",
        action="com.mobilellm.awattackerapplier.START_SERVICE",
        env=env,
    )
    if response.status == adb_pb2.AdbResponse.Status.OK:
        output = response.generic.output.decode("utf-8")
        result_code = int(output.split("result=")[1].split(",")[0])
        if result_code == 1:  # RESULT_SUCCESS
            logging.info("Attacker applier started successfully.")
        elif result_code == 2:  # RESULT_PERMISSION_DENIED
            raise RuntimeError(
                "Please grant both overlay and accessibility permission to start attacker applier."
            )
        elif result_code == 3:  # RESULT_ALREADY_RUNNING
            logging.info("Attacker applier is already running.")
        elif result_code == 4:  # RESULT_ERROR
            raise RuntimeError(
                "Result Error happened when starting attacker applier."
            )
    else:
        raise RuntimeError(f"No valid adb response received: {response}.")


def broadcast_stop_service(env: env_interface.AndroidEnvInterface) -> None:
    response = adb_utils.send_android_intent(
        command="broadcast",
        action="com.mobilellm.awattackerapplier.STOP_SERVICE",
        env=env,
    )
    if response.status == adb_pb2.AdbResponse.Status.OK:
        output = response.generic.output.decode("utf-8")
        result_code = int(output.split("result=")[1].split(",")[0])
        if result_code == 1:  # RESULT_SUCCESS
            logging.info("Attacker applier stopped successfully.")
        elif result_code == 2:  # RESULT_ERROR
            raise RuntimeError(
                "Result Error happened when stopping attacker applier."
            )
        elif result_code == 3:  # RESULT_ALREADY_STOPPED
            logging.info("Attacker applier is not running.")
        elif result_code == 4:  # RESULT_ERROR
            raise RuntimeError(
                "Result Error happened when stopping attacker applier."
            )
    else:
        raise RuntimeError(f"No valid adb response received: {response}.")


def broadcast_clear_rules(env: env_interface.AndroidEnvInterface) -> None:
    response = adb_utils.send_android_intent(
        command="broadcast",
        action="com.mobilellm.awattackerapplier.CLEAR_RULES",
        env=env,
    )
    if response.status == adb_pb2.AdbResponse.Status.OK:
        output = response.generic.output.decode("utf-8")
        result_code = int(output.split("result=")[1].split(",")[0])
        if result_code == 1:
            logging.info("Clear rules successfully.")
        elif result_code == 3:
            raise RuntimeError("Service is still running, cannot clear rules.")
        elif result_code == 4:
            raise RuntimeError("Exception raised when clearing rules.")
    else:
        raise RuntimeError(f"No valid adb response received: {response}")


def broadcast_import_rules(env: env_interface.AndroidEnvInterface, ruleJsonStr: str) -> None:
    response = adb_utils.issue_generic_request(
        ['shell', 'am', 'broadcast', '-a', 'com.mobilellm.awattackerapplier.IMPORT_RULES', '--es', 'rules_json', shlex.quote(ruleJsonStr)],
        env
    )
    if response.status == adb_pb2.AdbResponse.Status.OK:
        output = response.generic.output.decode("utf-8")
        result_code = int(output.split("result=")[1].split(",")[0])
        if result_code == 1:
            logging.info("Import rules successfully.")
        elif result_code == 3:
            raise RuntimeError("Service is still running, cannot import rules.")
        elif result_code == 4:
            raise RuntimeError("Exception raised when importing rules.")
        elif result_code == 5:
            raise RuntimeError("Invalid params specified.")
    else:
        raise RuntimeError(f"No valid adb response received: {response}")    


class EnableNetworkingError(ValueError):
    """网络启用失败异常"""

    pass


class A11yAttackGrpcWrapper(base_wrapper.BaseWrapper):
    """Wraps AndroidEnv to provide accessibility and window information via gRPC."""

    def __init__(
        self,
        env: env_interface.AndroidEnvInterface,
        install_a11y_forwarding_app: bool,
        port: int | None = None,
        disable_other_network_traffic: bool = False,
        latest_forest_only: bool = False,
        add_latest_info_to_obs: bool = False,
        info_timeout: float | None = None,
        max_enable_networking_attempts: int = 10,
        start_a11y_service: bool = True,
        polling_mode: bool = False,
        attack_config: dict[str, AttackConfig] = {},
    ):
        """初始化wrapper

        Args:
            env: 要包装的环境
            port: gRPC服务器端口号，如果为None则自动分配
            disable_other_network_traffic: 是否禁用除gRPC外的其他网络流量
            latest_forest_only: 是否只保存最新的accessibility tree
            add_latest_info_to_obs: 是否将最新的accessibility信息添加到observation中
            info_timeout: 获取accessibility信息的超时时间(秒)
            max_enable_networking_attempts: 网络重连最大尝试次数
            start_a11y_service: 是否启动accessibility服务
            polling_mode: 是否使用轮询模式收集数据
        """
        super().__init__(env)

        # 保存配置
        self._polling_mode = polling_mode
        self._max_enable_networking_attempts = max_enable_networking_attempts
        self._reset_enable_networking_attempts()
        self._disable_other_network_traffic = disable_other_network_traffic
        self._should_accumulate = False
        self._accumulated_extras = None
        self._add_latest_info_to_obs = add_latest_info_to_obs
        self._info_timeout = info_timeout
        self._relaunch_count = 0
        self._current_task = cast(str, None)
        self._attack_config = attack_config

        if port:
            self._port = port
        else:
            self._port = portpicker.pick_unused_port()

        # 启动accessibility服务
        if start_a11y_service:
            self._start_a11y_services()
            time.sleep(1.0)

        # 获取设备ID
        device_id = self.get_device_id()
        logging.info(f"Using device: {device_id}")

        # 初始化服务器
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        # 添加认证凭证
        server_credentials = grpc.local_server_credentials(
            grpc.LocalConnectionType.LOCAL_TCP
        )

        # 初始化Accessibility服务
        self._a11y_servicer = AccessibilityServiceImpl(
            device_id=device_id,
            latest_forest_only=latest_forest_only,
            polling_mode=polling_mode,
        )
        add_AccessibilityServiceServicer_to_server(self._a11y_servicer, self._server)

        # 初始化Window Info服务
        self._window_info_servicer = WindowInfoServiceImpl(env)
        add_WindowInfoServiceServicer_to_server(
            self._window_info_servicer, self._server
        )

        # 配置服务器
        uri_address = f"0.0.0.0:{self._port}"
        self._server.add_secure_port(uri_address, server_credentials)

        # 启动服务器
        logging.info("Starting server on %s", uri_address)
        self._server.start()
        logging.info("Server started successfully")

    def _reset_enable_networking_attempts(self) -> None:
        """重置网络重连尝试次数"""
        self._enable_networking_attempts_left = self._max_enable_networking_attempts
        self._enabling_networking_future = None
        self._a11y_exception = None

    def get_port(self) -> int:
        """获取服务器端口"""
        return self._port

    def close(self) -> None:
        # 设置关闭标志
        self._window_info_servicer.shutdown()
        self._a11y_servicer.shutdown()
        self.reset_rules()

        # 关闭gRPC服务器
        self._server.stop(None)
        logging.info("gRPC server stopped")

        # 关闭环境
        super().close()

    def attempt_enable_networking(self) -> None:
        """尝试启用网络连接"""
        # 关闭飞行模式
        self.execute_adb_call(
            adb_pb2.AdbRequest(
                settings=adb_pb2.AdbRequest.SettingsRequest(
                    name_space=adb_pb2.AdbRequest.SettingsRequest.Namespace.GLOBAL,
                    put=adb_pb2.AdbRequest.SettingsRequest.Put(
                        key="airplane_mode_on", value="0"
                    ),
                )
            )
        )
        time.sleep(1.0)

        # 启用WiFi
        self.execute_adb_call(
            adb_pb2.AdbRequest(
                generic=adb_pb2.AdbRequest.GenericRequest(
                    args=[
                        "shell",
                        "svc",
                        "wifi",
                        "enable",
                    ]
                )
            )
        )
        time.sleep(1.0)

    def _configure_grpc(self) -> None:
        """配置gRPC网络设置"""
        if self._disable_other_network_traffic:
            # 限制网络流量
            self.execute_adb_call(
                adb_pb2.AdbRequest(
                    generic=adb_pb2.AdbRequest.GenericRequest(
                        args=[
                            "shell",
                            "su",
                            "0",
                            "iptables",
                            "-A",
                            "OUTPUT",
                            "-p",
                            "tcp",
                            "-d",
                            "10.0.2.2",
                            "--dport",
                            str(self._port),
                            "-j",
                            "ACCEPT",
                        ]
                    )
                )
            )
            time.sleep(3.0)
            self.execute_adb_call(
                adb_pb2.AdbRequest(
                    generic=adb_pb2.AdbRequest.GenericRequest(
                        args=[
                            "shell",
                            "su",
                            "0",
                            "iptables",
                            "-A",
                            "OUTPUT",
                            "-j",
                            "DROP",
                        ]
                    )
                )
            )
            time.sleep(3.0)

        # 配置代理
        self.execute_adb_call(
            adb_pb2.AdbRequest(
                settings=adb_pb2.AdbRequest.SettingsRequest(
                    name_space=adb_pb2.AdbRequest.SettingsRequest.Namespace.GLOBAL,
                    put=adb_pb2.AdbRequest.SettingsRequest.Put(
                        key="no_proxy", value=f"10.0.2.2:{self._port}"
                    ),
                )
            )
        )
        self.attempt_enable_networking()
        response = adb_utils.send_android_intent(
            command="broadcast",
            action="com.mobilellm.awattackerapplier.SET_GRPC_CONFIG",
            env=self._env,
            extras={
                "host": "auto",
                "port": ("int", self._port),
            },
        )
        if response.status != adb_pb2.AdbResponse.Status.OK:
            raise RuntimeError(
                f"Could not set gRPC config automatically by broadcast: {response}."
            )
        else:
            output = response.generic.output.decode("utf-8")
            result_code = int(output.split("result=")[1].split(",")[0])
            if result_code == 1:  # RESULT_SUCCESS
                logging.info("gRPC config set successfully.")
            elif result_code == 5:  # RESULT_INVALID_PARAMS
                raise Exception("Invalid params when setting gRPC config.")
            elif result_code == 3:  # RESULT_ALREADY_RUNNING
                logging.info("Service is running, cannot set gRPC config.")
            elif result_code == 4:  # RESULT_ERROR
                raise Exception("Error happened when setting gRPC config.")

    def reset(self) -> dm_env.TimeStep:
        """重置环境和网络设置"""
        self._reset_enable_networking_attempts()

        # 重置环境
        timestep = super().reset()

        # 暂停所有服务
        self._a11y_servicer.pause_and_clear()
        self._window_info_servicer.pause_and_clear()

        # 恢复所有服务
        self._a11y_servicer.resume()
        self._window_info_servicer.resume()

        # 如果环境重启，重新配置网络
        if self._env.stats()["relaunch_count"] > self._relaunch_count:
            adb_utils.launch_app("com.mobilellm.awattackerapplier", self._env)
            time.sleep(1.5)
            self._configure_grpc()
            self.reset_rules()
            self._relaunch_count = self._env.stats()["relaunch_count"]

        self._accumulated_extras = {}
        timeout = self._info_timeout or 0.0
        new_observation = self._fetch_task_extras_and_update_observation(
            timestep.observation, timeout
        )
        timestep = timestep._replace(observation=new_observation)
        return timestep

    def start_service(self):
        if self._current_task and self._current_task in self._attack_config:
            rules_json = {"version": "1.0", "rules": [attack_config_to_rule(self._attack_config[self._current_task]).model_dump()]}
            broadcast_import_rules(self._env, json.dumps(rules_json))
        else:
            raise ValueError(f"Task {self._current_task} not found in attack config. Shutting down...")
        broadcast_start_service(self._env)

    def reset_rules(self):
        broadcast_stop_service(self._env)
        broadcast_clear_rules(self._env)

    def step(self, action: Any) -> dm_env.TimeStep:
        """执行一步动作"""
        timeout = float(action.pop("wait_time", self._info_timeout or 0.0))
        timestep = super().step(action)
        new_observation = self._fetch_task_extras_and_update_observation(
            timestep.observation, timeout=timeout
        )
        timestep = timestep._replace(observation=new_observation)
        return timestep

    def _fetch_task_extras_and_update_observation(
        self, observation: dict[str, Any], timeout: float = 0.0
    ) -> dict[str, Any]:
        """获取task extras并更新observation"""
        if timeout > 0.0:
            observation = self._accumulate_and_return_a11y_info(
                timeout, get_env_observation=True
            )
            if not self._add_latest_info_to_obs:
                observation.pop("a11y_forest")
        else:
            new_obs = self._accumulate_and_return_a11y_info(get_env_observation=False)
            if self._add_latest_info_to_obs:
                observation.update(new_obs)
        return observation

    def _accumulate_and_return_a11y_info(
        self, timer: float | None = None, get_env_observation: bool = True
    ) -> dict[str, Any]:
        """累积并返回accessibility信息"""
        timer = timer or 0.0
        if timer > 0.0:
            time.sleep(timer)

        if get_env_observation:
            # 获取observation
            new_ts = super().step({
                "action_type": np.array(
                    android_action_type_lib.ActionType.REPEAT,
                    dtype=self._parent_action_spec["action_type"].dtype,
                ),
            })
            observation = new_ts.observation
        else:
            observation = {}

        extras = self.accumulate_new_extras()
        forests = self._extract_forests_from_extras(extras)
        if isinstance(forests, np.ndarray) and forests.size > 0:
            observation["a11y_forest"] = forests[-1]
        else:
            observation["a11y_forest"] = None
        return observation

    def accumulate_new_extras(self) -> dict[str, Any]:
        """累积并返回新的extras数据"""
        new_extras = self._fetch_task_extras()
        if self._should_accumulate:
            if self._accumulated_extras is None:
                self._accumulated_extras = {}

            for key in new_extras:
                if key in self._accumulated_extras:
                    self._accumulated_extras[key] = np.concatenate(
                        (self._accumulated_extras[key], new_extras[key]), axis=0
                    )
                else:
                    self._accumulated_extras[key] = new_extras[key]
        else:
            self._accumulated_extras = new_extras

        self._should_accumulate = True
        return self._accumulated_extras

    def _fetch_task_extras(self) -> dict[str, Any]:
        """获取task extras数据"""
        base_extras = super().task_extras(latest_only=False).copy()

        if (
            self._enabling_networking_future is None
            and "exception" in base_extras
            and base_extras["exception"].shape[0]
        ):
            self._a11y_exception = base_extras["exception"]
            logging.warning(
                "AccessibilityForwarder logged exceptions: %s", self._a11y_exception
            )

            if self._enable_networking_attempts_left > 0:
                logging.warning(
                    "Attempting to enable networking. %s attempts left.",
                    self._enable_networking_attempts_left - 1,
                )
                executor = futures.ThreadPoolExecutor(max_workers=1)
                self._enabling_networking_future = executor.submit(
                    self.attempt_enable_networking
                )
            else:
                raise EnableNetworkingError(
                    "Accessibility service failed multiple times with "
                    f"exception: {self._a11y_exception}."
                )

        if (
            self._enabling_networking_future is not None
            and self._enabling_networking_future.done()
        ):
            self._enabling_networking_future = None
            self._enable_networking_attempts_left -= 1
            logging.info("Finished enabling networking.")

        # 获取accessibility数据
        forests = self._a11y_servicer.gather_forests()
        if forests:
            base_extras.update(self._package_forests_to_extras(forests))
            self._reset_enable_networking_attempts()

        events = self._a11y_servicer.gather_events()
        if events:
            base_extras.update(self._package_events_to_extras(events))
            self._reset_enable_networking_attempts()

        return base_extras

    def task_extras(self, latest_only: bool = False) -> dict[str, Any]:
        """获取task extras"""
        if self._accumulated_extras is None:
            raise RuntimeError("You must call .reset() before calling .task_extras()")

        self._should_accumulate = False
        extras = self._accumulated_extras.copy()

        if latest_only:
            self._keep_latest_only(extras)

        return extras

    @staticmethod
    def _extract_forests_from_extras(extras: dict[str, Any]) -> List[bytes]:
        """从extras中提取forests数据"""
        if "forests" in extras:
            return extras["forests"]
        return []

    @staticmethod
    def _package_forests_to_extras(forests: List[bytes]) -> dict[str, Any]:
        """将forests数据打包成extras格式"""
        return {"forests": np.array(forests)}

    @staticmethod
    def _package_events_to_extras(events: List[bytes]) -> dict[str, Any]:
        """将events数据打包成extras格式"""
        return {"events": np.array(events)}

    @staticmethod
    def _keep_latest_only(extras: dict[str, Any]) -> None:
        """只保留最新的数据"""
        if "forests" in extras and extras["forests"].shape[0] > 0:
            extras["forests"] = extras["forests"][-1:]
        if "events" in extras and extras["events"].shape[0] > 0:
            extras["events"] = extras["events"][-1:]

    def _start_a11y_services(self) -> None:
        """启动accessibility服务并赋予权限。

        Raises:
            RuntimeError: 如果accessibility服务启动失败。
        """
        # 赋予悬浮窗权限
        response = adb_utils.issue_generic_request(
            [
                "shell",
                "appops",
                "set",
                "com.mobilellm.awattackerapplier",
                "SYSTEM_ALERT_WINDOW",
                "allow",
            ],
            self._env,
        )
        if response.status != adb_pb2.AdbResponse.Status.OK:
            raise RuntimeError(
                f"Could not grant SYSTEM_ALERT_WINDOW permission: {response}."
            )
        
        # 启动accessibility服务
        start_service_request = adb_pb2.AdbRequest(
            settings=adb_pb2.AdbRequest.SettingsRequest(
                name_space=adb_pb2.AdbRequest.SettingsRequest.Namespace.SECURE,
                put=adb_pb2.AdbRequest.SettingsRequest.Put(
                    key="enabled_accessibility_services",
                    value=(
                        "com.mobilellm.awattackerapplier/com.mobilellm.awattackerapplier.AWAccessibilityService"
                    ),
                ),
            )
        )
        start_service_response = self.execute_adb_call(start_service_request)
        if start_service_response.status != adb_pb2.AdbResponse.Status.OK:
            raise RuntimeError(
                "Could not start accessibility forwarder "
                "service: "
                f"{start_service_response}."
            )

    def get_ui_elements(self, attack_config: AttackConfig):
        # 获取accessibility tree
        if self._polling_mode:
            # 在轮询模式下，从累积的extras中获取
            extras = self.accumulate_new_extras()
            forests = self._extract_forests_from_extras(extras)
            if not isinstance(forests, np.ndarray) or forests.size == 0:
                raise RuntimeError("Could not get accessibility tree")
            forest_data = forests[-1]
        else:
            # 在按需模式下，直接请求新的forest
            forest_data = self._a11y_servicer.get_forest()
            if forest_data is None:
                raise RuntimeError("Could not get accessibility tree")

        activity, _ = adb_utils.get_current_activity(self._env)
        if activity is None:
            raise RuntimeError("Could not get current activity")
        package_name = activity.split('/')[0]
        activity_name = activity.split('/')[1].replace(package_name, '')
        config = attack_config
        if attack_config.packageName != package_name or attack_config.activityName != activity_name:
            config = None

        # 转换为UI elements
        from android_world.attack.nodes import attack_tree_to_ui_elements_new

        tree = AccessibilityTree()
        tree.ParseFromString(forest_data)

        elements, truth = attack_tree_to_ui_elements_new(
            tree, exclude_invisible_elements=True, attack_config=config
        )
        return elements, truth

    def get_device_id(self) -> str:
        try:
            # 使用 adb devices 命令获取设备列表
            response = adb_utils.issue_generic_request(["devices"], self._env)

            if response.status != adb_pb2.AdbResponse.Status.OK:
                logging.error(f"Failed to get device list: {response.error_message}")
                raise RuntimeError(
                    f"Failed to get device list: {response.error_message}"
                )

            # 解析设备列表输出
            output = response.generic.output.decode("utf-8")
            # 跳过第一行 "List of devices attached"
            devices = [
                line.split("\t")[0]
                for line in output.strip().split("\n")[1:]
                if line.strip() and "\t" in line
            ]

            if not devices:
                logging.error("No devices connected")
                raise RuntimeError("No devices connected")

            if len(devices) > 1:
                logging.error(f"Multiple devices connected: {devices}")
                raise RuntimeError(
                    "Multiple devices connected, please specify device ID"
                )

            device_id = devices[0]
            logging.info(f"Found single device: {device_id}")
            return device_id

        except Exception as e:
            logging.error(f"Failed to get device list: {e}")
            raise RuntimeError(f"Failed to get device list: {e}")

    def set_current_task(self, task_name: str) -> None:
        self._current_task = task_name
