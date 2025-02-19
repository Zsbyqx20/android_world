import base64
import io
import json
import time
from typing import Any, Optional, List, Dict, Tuple
from PIL import Image
import numpy as np
import requests

# from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer

# from android_world.agents import m3a_utils
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils
import re


def image2PIL(image: np.ndarray) -> Image:
    image = Image.fromarray(image).convert("RGB")
    return image


def encode_image_from_ndarray(image_array: np.ndarray) -> str:
    # 将 np.ndarray 转换为 PIL 图像对象
    image = Image.fromarray(image_array)

    # 使用 BytesIO 将图像保存为字节流（JPEG 格式）
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")

    # 获取字节流中的 Base64 编码
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # 生成 Base64 编码后的图片 URL
    img_url = f"data:image/jpeg;base64,{img_base64}"

    return img_url


def formatting_input(
    task: str, history_step: List[str], history_action: List[str], image: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Formats input data into a structured message for further processing.

    Parameters:
    - task (str): The task or query the user is asking about.
    - history_step (List[str]): A list of historical steps in the conversation.
    - history_action (List[str]): A list of actions corresponding to the history steps.
    - round_num (int): The current round number (used to identify the image file).

    Returns:
    - List[Dict[str, Any]]: A list of messages formatted as dictionaries.

    Raises:
    - ValueError: If the lengths of `history_step` and `history_action` do not match.
    """
    # current_platform = identify_os()
    current_platform = "Mobile"
    platform_str = f"(Platform: {current_platform})\n"
    format_str = "(Answer in Action-Operation format.)\n"

    if len(history_step) != len(history_action):
        raise ValueError("Mismatch in lengths of history_step and history_action.")

    history_str = "\nHistory steps: "
    for index, (step, action) in enumerate(zip(history_step, history_action)):
        history_str += f"\n{index}. {step}\t{action}"

    query = f"Task: {task}{history_str}\n{platform_str}{format_str}"

    # Create image URL with base64 encoding
    img_url = encode_image_from_ndarray(image)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": img_url},
                },
            ],
        },
    ]
    return messages


def extract_grounded_operation(response: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts the grounded operation and action from the response text.

    Parameters:
    - response (str): The model's response text.

    Returns:
    - (step, action) (Tuple[Optional[str], Optional[str]]): Extracted step and action from the response.
    """
    grounded_pattern = r"Grounded Operation:\s*(.*)"
    action_pattern = r"Action:\s*(.*)"

    step = None
    action = None

    matches_history = re.search(grounded_pattern, response)
    matches_actions = re.search(action_pattern, response)
    if matches_history:
        step = matches_history.group(1)
    if matches_actions:
        action = matches_actions.group(1)

    return step, action


def is_balanced(s: str) -> bool:
    """
    Checks if the parentheses in a string are balanced.

    Parameters:
    - s (str): The string to check.

    Returns:
    - bool: True if parentheses are balanced, False otherwise.
    """
    stack = []
    mapping = {")": "(", "]": "[", "}": "{"}
    if "(" not in s:
        return False
    for char in s:
        if char in mapping.values():
            stack.append(char)
        elif char in mapping.keys():
            if not stack or mapping[char] != stack.pop():
                return False
    return not stack


def extract_operation(step: Optional[str]) -> Dict[str, Any]:
    """
    Extracts the operation and other details from the grounded operation step.

    Parameters:
    - step (Optional[str]): The grounded operation string.

    Returns:
    - Dict[str, Any]: A dictionary containing the operation details.
    """
    if step is None or not is_balanced(step):
        return {"operation": "NO_ACTION"}

    op, detail = step.split("(", 1)
    detail = "(" + detail
    others_pattern = r"(\w+)\s*=\s*([^,)]+)"
    others = re.findall(others_pattern, detail)
    Grounded_Operation = dict(others)
    Grounded_Operation = {key: value.strip("'\"") for key, value in others}
    boxes_pattern = r"box=\[\[(.*?)\]\]"
    boxes = re.findall(boxes_pattern, detail)
    if boxes:
        Grounded_Operation["box"] = list(map(int, boxes[0].split(",")))
    Grounded_Operation["operation"] = op.strip()

    return Grounded_Operation


# https://github.com/THUDM/CogAgent/blob/main/Action_space.md
# 动作空间


def step2action(
    operation: dict[str, Any], screen_size: Optional[Tuple[int, int]] = None
) -> Optional[dict[str, Any]]:
    result_action = {
        "action_type": None,
        "x": None,
        "y": None,
        "direction": None,
        "text": None,
        "app_name": None,
    }
    mapping = {
        "CLICK": "click",
        "DOUBLE_CLICK": "double_tap",
        "TYPE": "input_text",
        "LAUNCH": "open_app",
    }
    SCROLL = ("SCROLL_UP", "SCROLL_DOWN", "SCROLL_LEFT", "SCROLL_RIGHT")
    if operation["operation"] in mapping:
        result_action["action_type"] = mapping[operation["operation"]]
    elif operation["operation"] in SCROLL:
        result_action["action_type"] = "scroll"
        result_action["direction"] = operation["operation"].split("_")[1].lower()
    else:
        print("Unsuported operation:", operation["operation"])
        return result_action
    if "text" in operation:
        result_action["text"] = operation["text"]
    if "box" in operation:
        a, b, c, d = operation["box"]
        x, y = (a + c) // 2, (b + d) // 2
        x, y = int(x / 1000 * screen_size[0]), int(y / 1000 * screen_size[1])
        result_action["x"], result_action["y"] = x, y
    if "app" in operation:
        print("Open App:", operation["app"])
        result_action["app_name"] = operation["app"].strip("'").strip('"')
    return result_action


class CogAgent(base_agent.EnvironmentInteractingAgent):
    def __init__(
        self,
        env: interface.AsyncEnv,
        name: str = "cogagent",
        wait_after_action_seconds: float = 2.0,
    ):
        super().__init__(env, name)
        from gradio_client import Client

        self.model = Client("http://127.0.0.1:7890/")
        # self.model=None
        self.history = []
        self.wait_after_action_seconds = wait_after_action_seconds

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        # Hide the coordinates on screen which might affect the vision model.
        self.env.hide_automation_ui()
        self.history = []

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        step_data = {
            "raw_screenshot": None,
            "screen_size": None,
            "raw_response": None,
            "action": None,
            "step": None,
        }
        # Get the current state.
        print("----------step " + str(len(self.history) + 1))
        state = self.get_post_transition_state()
        step_data["raw_screenshot"] = state.pixels.copy()
        step_data["screen_size"] = self.env.logical_screen_size
        # Create messages for the chat API.
        messages = formatting_input(
            task=goal,
            history_step=[step["step"] for step in self.history],
            history_action=[step["action"] for step in self.history],
            image=step_data["raw_screenshot"],
        )
        # Generate a response from the chat API.
        q, i = messages[0]["content"][0]["text"], messages[0]["content"][1]["image_url"]
        try:
            response = self.model.predict(
                query=q,
                image=i,
                api_name="/my_predict",
            )
        except Exception as e:
            print("**********error**********")
            print(e)
            print("try again")
            time.sleep(1)
            response = self.model.predict(
                query=q,
                image=i,
                max_length=8192,
                api_name="/my_predict",
            )
        print("**********response**********")
        print(response)
        step_data["raw_response"] = response
        # Extract action and step from the response.
        step, action = extract_grounded_operation(response)
        step_data["action"], step_data["step"] = (action, step)

        # Grounding Operation to JsonAction
        operation = extract_operation(step)

        if operation["operation"] == "END":
            return base_agent.AgentInteractionResult(
                True,
                step_data,
            )

        result_action = step2action(operation, step_data["screen_size"])
        if result_action["action_type"] is None:
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(
                False,
                step_data,
            )
        else:
            converted_action = json_action.JSONAction(**result_action)
        self.env.execute_action(converted_action)
        time.sleep(self.wait_after_action_seconds)
        self.history.append(step_data)

        return base_agent.AgentInteractionResult(
            False,
            step_data,
        )
