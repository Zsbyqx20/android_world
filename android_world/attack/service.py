import logging
import time
import queue
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Generator

import grpc

from android_env import env_interface

from android_world.env import adb_utils
from android_world.attack.proto.accessibility_pb2 import (
    ClientResponse,
    ServerCommand,
    AccessibilityTree,
    GetAccessibilityTreeRequest,
)
from android_world.attack.proto.accessibility_pb2_grpc import (
    AccessibilityServiceServicer,
)
from android_world.attack.proto.window_info_pb2 import (
    WindowInfoRequest,
    WindowInfoResponse,
    WindowInfoSource,
    ResponseType,
)
from android_world.attack.proto.window_info_pb2_grpc import (
    WindowInfoServiceServicer,
)


@dataclass
class StreamInfo:
    """Information about a connected stream."""

    command_sender: queue.Queue
    device_id: str


class WindowInfoServiceImpl(WindowInfoServiceServicer):
    """Implementation of the Window Info gRPC service."""

    def __init__(self, env: env_interface.AndroidEnvInterface):
        """Initialize the service.

        Args:
            env: The Android environment interface.
        """
        self._env = env
        self._shutdown = False
        self._paused = False

    def shutdown(self):
        """Shutdown the service."""
        self._shutdown = True

    def pause_and_clear(self) -> None:
        """暂停服务。"""
        self._paused = True
        logging.info("Window info service paused")

    def resume(self) -> None:
        """恢复服务。"""
        self._paused = False
        logging.info("Window info service resumed")

    def GetCurrentWindowInfo(
        self,
        request: WindowInfoRequest,
        context: grpc.ServicerContext,
    ) -> WindowInfoResponse:
        """Get current window information."""
        device_id = request.device_id
        logging.info(f"Getting window info for device: {device_id}")

        try:
            # 检查连接状态
            if not context.is_active():
                logging.error("gRPC context is not active")
                return WindowInfoResponse(
                    success=False,
                    error_message="Connection lost",
                    type=ResponseType.WINDOW_INFO,
                )

            # 如果服务暂停或正在关闭，返回相应响应
            if self._paused:
                return WindowInfoResponse(
                    success=False,
                    error_message="Service is paused",
                    type=ResponseType.WINDOW_INFO,
                )
            if self._shutdown:
                return WindowInfoResponse(
                    success=True,
                    type=ResponseType.SERVER_STOP,
                )

            # 尝试通过ADB获取窗口信息
            package_name = None
            activity_name = None
            source = WindowInfoSource.PC_ADB

            # 使用adb_utils获取当前activity信息
            try:
                current_activity, _ = adb_utils.get_current_activity(self._env)
                if current_activity:
                    package_name, activity_name = current_activity.split("/")
                    activity_name = activity_name.replace(package_name, "")
                    logging.info(f"Got window info: {package_name}/{activity_name}")
            except Exception as e:
                logging.error(f"Failed to get window info via ADB: {e}")
                return WindowInfoResponse(
                    success=False,
                    error_message=str(e),
                    type=ResponseType.WINDOW_INFO,
                )

            if package_name and activity_name:
                return WindowInfoResponse(
                    success=True,
                    package_name=package_name,
                    activity_name=activity_name,
                    timestamp=int(time.time() * 1000),
                    source=source,
                    type=ResponseType.WINDOW_INFO,
                )
            else:
                return WindowInfoResponse(
                    success=False,
                    error_message="Failed to get window info",
                    type=ResponseType.WINDOW_INFO,
                )

        except Exception as e:
            return WindowInfoResponse(
                success=False,
                error_message=str(e),
                type=ResponseType.WINDOW_INFO,
            )


class AccessibilityServiceImpl(AccessibilityServiceServicer):
    """Implementation of the Accessibility gRPC service."""

    def __init__(
        self,
        device_id: str,
        latest_forest_only: bool = False,
        polling_mode: bool = False,
    ):
        """Initialize the service.

        Args:
            device_id: The ID of the device to serve.
            latest_forest_only: If True, only keep the latest accessibility tree.
            polling_mode: If True, use polling to collect data periodically. If False, collect on demand.
        """
        self._lock = threading.Lock()
        self._device_id = device_id
        self._latest_forest_only = latest_forest_only
        self._polling_mode = polling_mode
        self._forests: List[bytes] = []
        self._events: List[bytes] = []
        self._paused = False
        self._shutdown = False
        # Forest bookkeeping
        self._get_forest = threading.Event()  # Whether to request a forest
        self._forest_ready = threading.Event()  # Whether the forest is ready
        self._latest_forest: Optional[bytes] = None

        # Command queues for each device
        self._command_queues: Dict[str, queue.Queue] = {}
        self._command_responses: Dict[str, threading.Event] = {}

    def shutdown(self) -> None:
        self._shutdown = True
        with self._lock:
            self._forests.clear()
            self._events.clear()

    def pause_and_clear(self) -> None:
        """暂停服务并清除已收集的数据。"""
        self._paused = True
        with self._lock:
            self._forests.clear()
            self._events.clear()
        logging.info("Accessibility service paused and cleared")

    def resume(self) -> None:
        """恢复服务。"""
        self._paused = False
        logging.info("Accessibility service resumed")
        
        # # 主动获取一次accessibility tree，确保有数据
        # try:
        #     request = GetAccessibilityTreeRequest(device_id=self._device_id)
        #     tree = self.GetAccessibilityTree(request, None)  # context is None for local call
        #     if tree:
        #         with self._lock:
        #             self._latest_forest = tree.SerializeToString()
        #             if not self._latest_forest_only:
        #                 self._forests.append(self._latest_forest)
        #         logging.info("Successfully got initial accessibility tree after resume")
        #     else:
        #         logging.error("Failed to get initial accessibility tree after resume")
        # except Exception as e:
        #     logging.error(f"Error getting initial accessibility tree after resume: {e}")

    def handle_accessibility_data_update(self, device_id: str, data: bytes) -> None:
        """Handle incoming accessibility data updates.

        Args:
            device_id: The ID of the device sending the update.
            data: The raw accessibility data.
        """
        if self._paused:
            logging.info("Accessibility service is paused, ignoring update")
            return

        logging.debug(
            f"Updating accessibility data for device {device_id}: {len(data)} bytes"
        )

        with self._lock:
            self._latest_forest = data
            if self._latest_forest_only:
                self._forests = [data]
            else:
                self._forests.append(data)

            # If we were waiting for forest data, signal that it's ready
            if self._get_forest.is_set():
                self._forest_ready.set()

    def StreamAccessibility(
        self,
        request_iterator: Generator[ClientResponse, None, None],
        context: grpc.ServicerContext,
    ) -> Generator[ServerCommand, None, None]:
        """Handle bi-directional streaming of accessibility data."""
        logging.info("New streaming connection established")
        logging.info(f"Using device: {self._device_id}")

        # Create command queue and response event for this device
        command_queue = queue.Queue()
        response_event = threading.Event()
        with self._lock:
            self._command_queues[self._device_id] = command_queue
            self._command_responses[self._device_id] = response_event
            logging.info(f"Created command queue for device: {self._device_id}")

        def process_responses():
            """Process incoming responses."""
            try:
                for response in request_iterator:
                    # 检查是否已关闭
                    if self._shutdown:
                        logging.info(
                            "Service is shutting down, stopping response processing"
                        )
                        break

                    if response.device_id == "heartbeat":
                        logging.info("💓 Received heartbeat")
                        continue

                    if response.success:
                        self.handle_accessibility_data_update(
                            response.device_id, response.raw_output
                        )
                        # Signal that we got a response
                        if self._device_id in self._command_responses:
                            self._command_responses[self._device_id].set()
                    else:
                        logging.error(
                            f"Received error response: {response.error_message}"
                        )

                    if self._polling_mode:
                        command_queue.put(
                            ServerCommand(
                                device_id=self._device_id,
                                command=ServerCommand.CommandType.GET_ACCESSIBILITY_TREE,
                            )
                        )
            except grpc.RpcError as e:
                if not self._shutdown:
                    # 如果不是因为关闭导致的错误，才记录日志
                    logging.error("RPC Error in process_responses: %s", e)
            except Exception as e:
                if not self._shutdown:
                    logging.exception("Error in process_responses: %s", e)
            finally:
                # 发送停止信号
                command_queue.put(
                    ServerCommand(
                        device_id=self._device_id,
                        command=ServerCommand.CommandType.STOP,
                    )
                )
                # 清理资源
                with self._lock:
                    if self._device_id in self._command_queues:
                        del self._command_queues[self._device_id]
                    if self._device_id in self._command_responses:
                        del self._command_responses[self._device_id]
                logging.info("Response processing stopped")

        def collect_forest_periodically():
            """Periodically collect accessibility forest data."""
            while context.is_active():
                try:
                    # Request new forest data
                    command_queue.put(
                        ServerCommand(
                            device_id=self._device_id,
                            command=ServerCommand.CommandType.GET_ACCESSIBILITY_TREE,
                        )
                    )
                    time.sleep(1.0)
                except Exception as e:
                    logging.error(f"Error in forest collection: {e}")
                    break

        # Start response processing thread
        response_thread = threading.Thread(target=process_responses)
        response_thread.daemon = True
        response_thread.start()

        # Start periodic forest collection thread if in polling mode
        if self._polling_mode:
            collection_thread = threading.Thread(target=collect_forest_periodically)
            collection_thread.daemon = True
            collection_thread.start()

        try:
            # Send commands
            while context.is_active():
                try:
                    command = command_queue.get(timeout=1.0)
                    if command.command == ServerCommand.CommandType.STOP:
                        logging.info("Received stop command, ending stream")
                        break
                    yield command
                except queue.Empty:
                    continue
        finally:
            # Clean up device resources
            with self._lock:
                self._command_queues.pop(self._device_id, None)
                self._command_responses.pop(self._device_id, None)

    def get_forest(self) -> Optional[bytes]:
        """Issues a request to get the accessibility forest from the client."""
        if self._polling_mode:
            # In polling mode, just wait for the next forest
            self._get_forest.set()
            try:
                if not self._forest_ready.wait(timeout=5.0):
                    logging.error("Timeout waiting for forest data")
                    return None

                with self._lock:
                    if self._latest_forest:
                        forest = self._latest_forest
                        self._latest_forest = None
                        return forest
            finally:
                self._get_forest.clear()
                self._forest_ready.clear()
        else:
            # In on-demand mode, use GetAccessibilityTree RPC
            try:
                request = GetAccessibilityTreeRequest(device_id=self._device_id)
                tree = self.GetAccessibilityTree(
                    request, None
                )  # context is None for local call
                return tree.SerializeToString()
            except Exception as e:
                logging.error(f"Failed to get accessibility tree: {e}")
                return None

        return None

    def gather_forests(self) -> List[bytes]:
        """Get collected accessibility forests.

        Returns:
            List of raw forest data.
        """
        with self._lock:
            forests = self._forests.copy()
            self._forests.clear()
        return forests

    def gather_events(self) -> List[bytes]:
        """Get collected accessibility events.

        Returns:
            List of raw event data.
        """
        with self._lock:
            events = self._events.copy()
            self._events.clear()
        return events

    def GetAccessibilityTree(
        self,
        request: GetAccessibilityTreeRequest,
        context: grpc.ServicerContext,
    ) -> AccessibilityTree:
        """Get the current accessibility tree."""
        try:
            if request.device_id != self._device_id:
                logging.warning(
                    f"Requested device ID {request.device_id} does not match service device {self._device_id}"
                )

            logging.info(f"Getting accessibility tree for device: {self._device_id}")

            # 检查连接状态
            if context and not context.is_active():
                logging.error("gRPC context is not active")
                raise RuntimeError("Connection lost")

            # 如果服务暂停，返回错误
            if self._paused:
                logging.warning("Accessibility service is paused")
                raise RuntimeError("Service is paused")

            # 发送命令请求新的tree
            with self._lock:
                if self._device_id not in self._command_queues:
                    raise RuntimeError("No active connection for device")

                command_queue = self._command_queues[self._device_id]
                response_event = self._command_responses[self._device_id]
                logging.info(f"Found command queue for device: {self._device_id}")

            # Clear any previous response
            response_event.clear()

            # Send request
            command_queue.put(
                ServerCommand(
                    device_id=self._device_id,
                    command=ServerCommand.CommandType.GET_ACCESSIBILITY_TREE,
                )
            )
            logging.info(f"Sent command to device: {self._device_id}")

            # Wait for response
            if not response_event.wait(timeout=10.0):
                raise RuntimeError("Timeout waiting for tree data")

            with self._lock:
                if self._latest_forest:
                    tree = AccessibilityTree()
                    tree.ParseFromString(self._latest_forest)
                    self._latest_forest = None
                    logging.info(f"Successfully got tree for device: {self._device_id}")
                    return tree
                else:
                    raise RuntimeError("No tree data available")

        except Exception as e:
            logging.error(f"Failed to get accessibility tree: {e}")
            raise
