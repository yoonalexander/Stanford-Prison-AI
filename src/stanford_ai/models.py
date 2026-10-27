"""Explicit experiment schema. Actor identity is supplied by the engine, never the LLM."""

from enum import StrEnum
from typing import Literal
from urllib.parse import urlsplit

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


BackendName = Literal["mock", "openai", "ollama", "openai_compatible"]


class ModelProfile(StrictModel):
    """Connection/generation settings only; never placed in an agent's observations."""

    backend: BackendName
    model: str = Field(min_length=1)
    base_url: str | None = None
    api_key_env: str = Field(default="OPENAI_API_KEY", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    output_mode: Literal["json_schema", "json_object"] = "json_schema"
    token_limit_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    temperature: float | None = Field(default=0.7, ge=0, le=2)
    max_output_tokens: int = Field(default=400, ge=128, le=4096)
    local_context_tokens: int = Field(default=32768, ge=4096, le=262144)
    model_digest: str | None = None
    input_usd_per_million: float = Field(default=0, ge=0, allow_inf_nan=False)
    output_usd_per_million: float = Field(default=0, ge=0, allow_inf_nan=False)
    pricing_confirmed: bool = False
    timeout_seconds: float = Field(default=120, ge=1, le=600)

    @model_validator(mode="after")
    def connection(self):
        import ipaddress
        if self.base_url:
            url = urlsplit(self.base_url)
            if (url.scheme not in {"http", "https"} or not url.hostname or url.username
                    or url.password or url.query or url.fragment):
                raise ValueError("base_url must be an HTTP(S) endpoint without credentials/query/fragment")
            try:
                loopback = ipaddress.ip_address(url.hostname).is_loopback
            except ValueError:
                loopback = url.hostname.lower() == "localhost"
            if self.backend == "ollama" and not loopback:
                raise ValueError("Free Ollama backend requires a loopback server; cloud routing is disabled")
            if self.backend == "openai_compatible" and url.scheme != "https" and not loopback:
                raise ValueError("Remote compatible endpoints require HTTPS")
        if self.backend == "openai" and self.base_url not in (None, "https://api.openai.com/v1"):
            raise ValueError("Use openai_compatible for a custom endpoint")
        if self.backend == "openai_compatible" and not self.base_url:
            raise ValueError("An OpenAI-compatible endpoint requires base_url (including /v1 if needed)")
        if self.backend in {"mock", "ollama"}:
            if self.input_usd_per_million or self.output_usd_per_million:
                raise ValueError("Mock/local Ollama profiles have zero API rates")
        elif not self.pricing_confirmed:
            raise ValueError("Confirm current model prices before using a hosted endpoint")
        if self.backend == "openai" and (self.input_usd_per_million <= 0
                                          or self.output_usd_per_million <= 0):
            raise ValueError("Confirm positive current model prices before using OpenAI")
        if self.backend != "openai_compatible" and self.output_mode != "json_schema":
            raise ValueError("JSON-object mode is an explicit compatibility option only")
        return self

    @property
    def endpoint(self) -> str | None:
        return self.base_url or {"ollama": "http://127.0.0.1:11434",
                                 "openai": "https://api.openai.com/v1"}.get(self.backend)


class ExperimentConfig(StrictModel):
    condition: Literal["A", "B", "C", "D"] = "B"
    config_version: str = "1.1"
    backend: BackendName = "mock"
    model: str = "gpt-4.1-mini-2025-04-14"
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    output_mode: Literal["json_schema", "json_object"] = "json_schema"
    token_limit_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    local_context_tokens: int = Field(default=32768, ge=4096, le=262144)
    model_profiles: dict[str, ModelProfile] = Field(default_factory=dict)
    agent_models: dict[str, str] = Field(default_factory=dict)
    n_agents: int = Field(default=4, ge=2, le=100)
    days: int = Field(default=2, ge=1, le=100)
    rounds_per_day: int = Field(default=2, ge=1, le=20)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    temperature: float | None = Field(default=0.7, ge=0, le=2)
    max_output_tokens: int = Field(default=400, ge=128, le=4096)
    max_requests: int = Field(default=100, ge=1, le=100000)
    max_tokens: int = Field(default=250000, ge=1)
    budget_usd: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    # Standard uncached rates, user must confirm for the selected model before a paid run.
    input_usd_per_million: float = Field(default=0.40, ge=0, allow_inf_nan=False)
    output_usd_per_million: float = Field(default=1.60, ge=0, allow_inf_nan=False)
    pricing_confirmed: bool = False
    max_retries: int = Field(default=2, ge=0, le=5)
    retry_delay_seconds: float = Field(default=1, ge=0, le=30)
    timeout_seconds: float = Field(default=120, ge=1, le=600)
    memory_window: int = Field(default=12, ge=1, le=100)
    history_window: int = Field(default=12, ge=1, le=100)

    @model_validator(mode="after")
    def balanced(self):
        if self.n_agents % 2:
            raise ValueError("n_agents must be even, with equal group sizes")
        ids = {f"agent_{i + 1:02d}" for i in range(self.n_agents)}
        if set(self.agent_models) - ids:
            raise ValueError("agent_models contains an unknown agent ID for n_agents")
        if "default" in self.model_profiles:
            raise ValueError("The profile name 'default' is reserved")
        if set(self.agent_models.values()) - set(self.model_profiles):
            raise ValueError("agent_models references an undefined model profile")
        for actor in sorted(ids):
            self.profile_for(actor)  # Validate every actual route, including the default if used.
        return self

    def profile_for(self, actor: str) -> ModelProfile:
        if actor in self.agent_models:
            return self.model_profiles[self.agent_models[actor]]
        return ModelProfile(backend=self.backend, model=self.model, base_url=self.base_url,
            api_key_env=self.api_key_env, output_mode=self.output_mode,
            token_limit_field=self.token_limit_field, temperature=self.temperature,
            max_output_tokens=self.max_output_tokens, local_context_tokens=self.local_context_tokens,
            input_usd_per_million=0 if self.backend in {"mock", "ollama"} else self.input_usd_per_million,
            output_usd_per_million=0 if self.backend in {"mock", "ollama"} else self.output_usd_per_million,
            pricing_confirmed=self.pricing_confirmed, timeout_seconds=self.timeout_seconds)

    def resolved_models(self) -> dict[str, dict]:
        return {f"agent_{i + 1:02d}": self.profile_for(f"agent_{i + 1:02d}").model_dump(mode="json")
                | {"profile": self.agent_models.get(f"agent_{i + 1:02d}", "default")}
                for i in range(self.n_agents)}

    @property
    def evidence_class(self) -> str:
        synthetic = [p["backend"] == "mock" for p in self.resolved_models().values()]
        return "synthetic" if all(synthetic) else "mixed_synthetic" if any(synthetic) else "llm"

    @property
    def backend_summary(self) -> str:
        backends = {p["backend"] for p in self.resolved_models().values()}
        return next(iter(backends)) if len(backends) == 1 else "mixed"

    @property
    def model_summary(self) -> str:
        models = sorted({p["model"] for p in self.resolved_models().values()})
        return models[0] if len(models) == 1 else " + ".join(models)

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
