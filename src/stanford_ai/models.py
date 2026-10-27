"""Explicit experiment schema. Actor identity is supplied by the engine, never the LLM."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ActionKind(StrEnum):
    COMMUNICATE = "communicate"
    PROPOSE_RULE = "propose_rule"
    VOTE_RULE = "vote_rule"
    APPROVE_RULE = "approve_rule"
    REQUEST_RESOURCE = "request_resource"
    ALLOCATE_RESOURCE = "allocate_resource"
    ASSIGN_TASK = "assign_task"
    VOLUNTEER_TASK = "volunteer_task"
    COMPLETE_TASK = "complete_task"
    REQUEST_EXCEPTION = "request_exception"
    FILE_APPEAL = "file_appeal"
    DECIDE_APPEAL = "decide_appeal"
    NEGOTIATE = "negotiate"
    OFFER_AGREEMENT = "offer_agreement"
    ACCEPT_AGREEMENT = "accept_agreement"
    RESTRICT_ACCESS = "restrict_access"
    USE_ACCESS = "use_access"
    CHALLENGE = "challenge"
    CHANGE_SCHEDULE = "change_schedule"
    NOOP = "noop"


class Action(StrictModel):
    # All fields required (nullable where irrelevant), compatible with strict Structured Outputs.
    kind: ActionKind
    target: str | None
    reference: str | None
    amount: int | None = Field(ge=0, le=3)
    resource: Literal["leisure_credits"] | None
    zone: Literal["garden", "library", "workshop"] | None
    scope: Literal["A", "B", "all"] | None
    approve: bool | None
    private: bool
    message: str = Field(max_length=600)

    @classmethod
    def make(cls, kind: ActionKind | str, **kwargs) -> "Action":
        values = dict(target=None, reference=None, amount=None, resource=None, zone=None,
                      scope=None, approve=None, private=False, message="")
        return cls(kind=kind, **(values | kwargs))


class Personality(StrictModel):
    assertiveness: float = Field(ge=0, le=1)
    cooperativeness: float = Field(ge=0, le=1)
    conflict_avoidance: float = Field(ge=0, le=1)
    rule_adherence: float = Field(ge=0, le=1)
    risk_tolerance: float = Field(ge=0, le=1)
    empathy_orientation: float = Field(ge=0, le=1)


class Agent(StrictModel):
    id: str
    group: Literal["A", "B"]
    personality: Personality
    goals: list[str]
    memory: list[dict] = Field(default_factory=list)
    memory_summary: dict[str, int] = Field(default_factory=dict)
    relationships: dict[str, dict[str, int]] = Field(default_factory=dict)


class ExperimentConfig(StrictModel):
    condition: Literal["A", "B", "C", "D"] = "B"
    config_version: str = "1.0"
    backend: Literal["mock", "openai"] = "mock"
    model: str = "gpt-4.1-mini-2025-04-14"
    n_agents: int = Field(default=4, ge=2, le=100)
    days: int = Field(default=2, ge=1, le=100)
    rounds_per_day: int = Field(default=2, ge=1, le=20)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    temperature: float | None = Field(default=0.7, ge=0, le=2)
    max_output_tokens: int = Field(default=400, ge=128, le=4096)
    max_requests: int = Field(default=100, ge=1, le=100000)
    max_tokens: int = Field(default=250000, ge=1)
    budget_usd: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    # Standard uncached rates, user must confirm for the selected model before a paid run.
    input_usd_per_million: float = Field(default=0.40, ge=0, allow_inf_nan=False)
    output_usd_per_million: float = Field(default=1.60, ge=0, allow_inf_nan=False)
    pricing_confirmed: bool = False
    max_retries: int = Field(default=2, ge=0, le=5)
    retry_delay_seconds: float = Field(default=1, ge=0, le=30)
    timeout_seconds: float = Field(default=45, ge=1, le=120)
    memory_window: int = Field(default=12, ge=1, le=100)
    history_window: int = Field(default=12, ge=1, le=100)

    @model_validator(mode="after")
    def balanced(self):
        if self.n_agents % 2:
            raise ValueError("n_agents must be even, with equal group sizes")
        if self.backend == "openai" and (
            not self.pricing_confirmed or self.input_usd_per_million <= 0
            or self.output_usd_per_million <= 0
        ):
            raise ValueError("Confirm positive current model prices before using OpenAI")
        return self

    @property
    def unequal(self) -> bool:
        return self.condition in ("A", "B")

    @property
    def explicit_labels(self) -> bool:
        return self.condition in ("A", "C")


class World(StrictModel):
    day: int = 0
    round: int = 0
    agents: dict[str, Agent]
    pool: int = 0
    holdings: dict[str, int]
    allocated_today: dict[str, int]
    spent: int = 0
    daily_supply: int = 0
    rules: dict[str, dict] = Field(default_factory=dict)
    tasks: dict[str, dict] = Field(default_factory=dict)
    schedule: list[str] = Field(default_factory=lambda: ["library", "garden", "workshop"])
    restrictions: dict[str, dict[str, int]] = Field(default_factory=dict)
    requests: dict[str, dict] = Field(default_factory=dict)
    appeals: dict[str, dict] = Field(default_factory=dict)
    disputes: dict[str, dict] = Field(default_factory=dict)
    agreements: dict[str, dict] = Field(default_factory=dict)
    history: list[dict] = Field(default_factory=list)
