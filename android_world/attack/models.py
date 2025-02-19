import json
from pydantic import BaseModel, RootModel
from pathlib import Path
from typing import Literal

EXTRAS_TEXT_PLACEHOLDER = "THIS IS A IMPOSSIBLE TEXT PLACEHOLDER"

class Padding(BaseModel):
    left: float
    top: float
    right: float
    bottom: float


class OverlayStyle(BaseModel):
    x: float
    y: float
    width: float
    height: float
    text: str
    fontSize: float
    backgroundColor: str
    textColor: str
    horizontalAlign: str
    verticalAlign: str
    uiAutomatorCode: str
    padding: Padding


class Rule(BaseModel):
    name: str
    packageName: str
    activityName: str
    isEnabled: bool
    overlayStyles: list[OverlayStyle]
    tags: list[str] = []


class AttackConfigExtras(BaseModel):
    action: str = "IMPOSSIBLE"
    area: tuple[int, int, int, int] = (0, 0, 0, 0)
    relative_index: int = -1
    relative_text: str = EXTRAS_TEXT_PLACEHOLDER


class AttackSetting(BaseModel):
    level: Literal["simple", "medium", "hard"]
    action: str
    density: str


class AttackConfig(BaseModel):
    packageName: str
    activityName: str
    overlayStyles: list[OverlayStyle]
    extras: AttackConfigExtras


class AttackConfigValidator(RootModel[dict[str, AttackConfig]]):
    pass


def load_attack_config_legacy(attack_config: str):
    rules_map: dict[tuple[str, str], Rule] = {}
    if not attack_config:
        raise ValueError("Attack config is required for attacker applier to start.")
    attack_config_path = Path(attack_config)
    if not attack_config_path.exists():
        raise ValueError(f"Attack config file does not exist: {attack_config}")
    with attack_config_path.open("r") as f:
        attack_config = json.load(f)
    try:
        rules = [Rule(**rule) for rule in attack_config["rules"]] # type: ignore
        for rule in rules:
            rules_map[(rule.packageName, rule.activityName)] = rule
        return rules_map
    except Exception as e:
        raise ValueError(f"Error loading attack config: {e}")


def load_attack_config(attack_config: str) -> dict[str, AttackConfig]:
    if not attack_config:
        raise ValueError("Attack config is required for attacker applier to start.")
    attack_config_path = Path(attack_config)
    if not attack_config_path.exists():
        raise ValueError(f"Attack config file does not exist: {attack_config}")
    with attack_config_path.open("r") as f:
        attack_config = json.load(f)
    return AttackConfigValidator.model_validate(attack_config).root


def attack_config_to_rule(attack_config: AttackConfig):
    return Rule(
        name="attack rule",
        packageName=attack_config.packageName,
        activityName=attack_config.activityName,
        isEnabled=True,
        overlayStyles=attack_config.overlayStyles,
    )
