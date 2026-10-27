"""The only source of administrative capabilities; labels never participate."""

from stanford_ai.models import ActionKind, ExperimentConfig

ADMIN = frozenset({ActionKind.APPROVE_RULE, ActionKind.ALLOCATE_RESOURCE,
                   ActionKind.ASSIGN_TASK, ActionKind.DECIDE_APPEAL,
                   ActionKind.RESTRICT_ACCESS, ActionKind.CHANGE_SCHEDULE})
COMMON = frozenset(ActionKind) - ADMIN


def permitted(config: ExperimentConfig, group: str) -> list[str]:
    return sorted(COMMON | (ADMIN if not config.unequal or group == "A" else set()))


def matrix(config: ExperimentConfig) -> dict[str, list[str]]:
    return {group: permitted(config, group) for group in ("A", "B")}
