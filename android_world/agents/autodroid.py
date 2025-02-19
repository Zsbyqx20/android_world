import time
from typing import Any, Optional
from android_world.agents import base_agent, m3a_utils
from android_world.agents import infer
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils
import re

Autodroid_Prompt = (
    "You are a smartphone assistant to help users complete tasks by interacting with mobile apps. Given a task, the previous UI actions, and the content of current UI state, your job is to decide whether the task is already finished by the previous actions, and if not, decide which UI element in current UI state should be interacted.\n"
    "Task to be completed:\n"
    "{task}\n"
    "Current UI state:\n"
    "{html_desc}\n"
    "Former UI actions you have taken:\n"
    "{action_history}\n"
    'Your answer should always use the following format: {{ "Steps": "...<steps usually involved to complete the above task on a smartphone>", "Analyses": "...<Analyses of the relations between the task, and relations between the previous UI actions and current UI state>", "Finished": "Yes/No", "Next step": "None or a <high level description of the next step>", "id": "an integer or -1 (if the task has been completed by previous UI actions)", "action": "touch or scroll up/down or set_text", "input_text": "N/A or ...<input text>" }}\n'
    """**Note that the id is the id number of the UI element to interact with. If you think the task has been completed by previous UI actions, the id should be -1. If 'Finished' is 'Yes', then the 'description' of 'Next step' is 'None', otherwise it is a high level description of the next step. If the 'action' is 'touch' or 'scroll up' or 'scroll down', the 'input_text' is N/A, otherwise it is the '<input text>'. Please do not output any content other than the JSON format. **\n"""
)


def extract_json(s: str) -> Optional[dict[str, Any]]:
    """Extracts JSON from string.

    Args:
      s: A string with a JSON in it. E.g., "{'hello': 'world'}" or from CoT:
        "let's think step-by-step, ..., {'hello': 'world'}".

    Returns:
      JSON object.
    """

    import ast

    pattern = r"\{.*\}"
    match = re.search(pattern, s, re.DOTALL)
    if match:
        try:
            return ast.literal_eval(match.group())
        except (SyntaxError, ValueError) as error:
            print("Cannot extract JSON, skipping due to error %s", error)
            return None
    else:
        return None


def _generate_ui_elements_description_list_full(
    ui_elements: list[representation_utils.UIElement],
    screen_width_height_px: tuple[int, int],
) -> str:
  """Generate description for a list of UIElement using full information.

  Args:
    ui_elements: UI elements for the current screen.
    screen_width_height_px: Logical screen size.

  Returns:
    Information for each UIElement.
  """
  tree_info = ''
  for index, ui_element in enumerate(ui_elements):
    if m3a_utils.validate_ui_element(ui_element, screen_width_height_px):
      tree_info += f'UI element {index}: {str(ui_element)}\n'
  return tree_info

def validate_ui_element(
    ui_element: representation_utils.UIElement,
    screen_width_height_px: tuple[int, int],
) -> bool:
    """Used to filter out invalid UI element."""
    screen_width, screen_height = screen_width_height_px

    # Filters out invisible element.
    if not ui_element.is_visible:
        return False

    # Filters out element with invalid bounding box.
    if ui_element.bbox_pixels:
        x_min = ui_element.bbox_pixels.x_min
        x_max = ui_element.bbox_pixels.x_max
        y_min = ui_element.bbox_pixels.y_min
        y_max = ui_element.bbox_pixels.y_max

        if (
            x_min >= x_max
            or x_min >= screen_width
            or x_max <= 0
            or y_min >= y_max
            or y_min >= screen_height
            or y_max <= 0
        ):
            return False
    if (
        not ui_element.text
        and not ui_element.content_description
        and not ui_element.is_clickable
    ):
        return False
    if (
        not ui_element.content_description
        and not ui_element.text
        and not ui_element.is_scrollable
    ):  # actionable?
        return False

    return True

def _action_gen_prompt(
    goal: str,
    history: list[str],
    ui_elements: str,
) -> str:
    if history:
        history = "\n".join(history)
    else:
        history = "You just started, no action has been performed yet."

    return Autodroid_Prompt.format(
        task=goal,
        html_desc=ui_elements,
        action_history=history,
    )


def extract_action(v: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    try:
        result_action = {
            "action_type": None,
            "index": None,
            "direction": None,
            "text": None,
        }

        if "Finished" in v.keys():
            whether_finished_answer = v["Finished"].lower() == "yes"
        elif "finished" in v.keys():
            whether_finished_answer = v["finished"].lower() == "yes"
        else:
            whether_finished_answer = False
        if whether_finished_answer:
            result_action["index"] = -1
            result_action["action_type"] = "N/A"
            result_action["text"] = "N/A"
        else:
            result_action["index"] = "N/A"

    except Exception as e:  # pylint: disable=broad-except
        print("Failed to extract action:{0}".format(str(e)))
        pass
    if result_action["index"] != -1:
        step_desc = v
        try:
            result_action["index"] = step_desc["id"]
            result_action["action_type"] = step_desc["action"]
            result_action["text"] = step_desc["input_text"]
            if result_action["index"] == "N/A":
                result_action["index"] = -1
            else:
                result_action["index"] = int(result_action["index"])
            assert result_action["action_type"] in [
                "touch",
                "scroll up",
                "scroll down",
                "set_text",
                "N/A",
            ]
            if result_action["action_type"] == "touch":
                result_action["action_type"] = "click"
            elif result_action["action_type"] == "scroll up":
                result_action["action_type"] = "scroll"
                result_action["direction"] = "up"
            elif result_action["action_type"] == "scroll down":
                result_action["action_type"] = "scroll"
                result_action["direction"] = "down"
            elif result_action["action_type"] == "set_text":
                result_action["action_type"] = "input_text"
        except Exception as e:  # pylint: disable=broad-except
            print("Failed to extract action:{0}".format(str(e)))
            result_action["index"] = -1
    return result_action


class Autodroid(base_agent.EnvironmentInteractingAgent):
    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.MultimodalLlmWrapper,
        name: str = "autodroid",
        wait_after_action_seconds: float = 2.0,
    ):
        super().__init__(env, name)
        self.llm = llm
        self.history = []
        self.wait_after_action_seconds = wait_after_action_seconds

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        # Hide the coordinates on screen which might affect the vision model.
        self.env.hide_automation_ui()
        self.history = []

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        step_data = {
            "before_ui_elements": [],
            "action_prompt": None,
            "action_output": None,
            "action_raw_response": None,
            "action_history": None,
        }
        print("----------step " + str(len(self.history) + 1))
        state = self.get_post_transition_state()
        logical_screen_size = self.env.logical_screen_size

        before_ui_elements = state.ui_elements
        step_data["before_ui_elements"] = before_ui_elements
        before_ui_elements_html_list=_generate_ui_elements_description_list_full(
            before_ui_elements, logical_screen_size
        )
        # Generate the prompt.
        action_prompt = _action_gen_prompt(
            goal=goal,
            history=[
                "Step " + str(i + 1) + "- " + step_info["action_history"]
                for i, step_info in enumerate(self.history)
            ],
            ui_elements=before_ui_elements_html_list,
        )
        step_data["action_prompt"] = action_prompt

        # Call LLM
        action_output, is_safe, raw_response = self.llm.predict_mm(action_prompt, [])
        # if is_safe == False:  # pylint: disable=singleton-comparison
        #  is_safe could be None
        #  pass
        if not raw_response:
            raise RuntimeError("Error calling LLM in action selection phase.")
        step_data["action_output"] = action_output
        step_data["action_raw_response"] = raw_response
        print("action_output:",action_output)
        # parse the json
        v = extract_json(action_output)
        if v is None:
            result_action = None
        else:
            result_action = extract_action(v)
        if result_action["index"] == -1:
            step_data["action_history"] = (
                "The task is finished, no further action is needed."
            )
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(
                True,
                step_data,
            )

        try:
            # target_view_idx = result_action["index"]
            # result_action["index"]=action_grounding_index[result_action["index"]]
            converted_action = json_action.JSONAction(**result_action)
            self.env.execute_action(converted_action)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print("Failed to execute action.")
            print(str(e))
            step_data["action_history"] = (
                "Can not execute the action, make sure to select the action with"
                " the required parameters (if any) in the correct JSON format!"
            )
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(
                False,
                step_data,
            )
        step_data["action_history"] = f'UI element {result_action["index"]}: {str(before_ui_elements[result_action["index"]])}\n'
        time.sleep(self.wait_after_action_seconds)
        self.history.append(step_data)

        return base_agent.AgentInteractionResult(
            False,
            step_data,
        )
