import json
from dataclasses import asdict, dataclass

from stanford_ai.agents.prompts import output_format
from stanford_ai.models import ExperimentConfig


class BudgetExhausted(Exception):
    pass


@dataclass
class Reservation:
    input_tokens: int
    output_tokens: int
    cost: float


@dataclass
class Ledger:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0
    accounted_tokens: int = 0
    accounted_cost_usd: float = 0
    unknown_usage_requests: int = 0

    def as_dict(self):
        return asdict(self)

    def reserve(self, cfg: ExperimentConfig, prompt: list[dict]) -> Reservation:
        # A deliberately conservative UTF-8 byte envelope, including schema + wire overhead.
        # No tools, images or hidden conversation state are sent by this backend.
        size = len(json.dumps({"input": prompt, "text": output_format()},
                              ensure_ascii=False).encode("utf-8")) + 4096
        output = cfg.max_output_tokens
        cost = (size * cfg.input_usd_per_million + output * cfg.output_usd_per_million) / 1e6
        if cfg.backend == "mock":
            size, output, cost = 0, 0, 0
        if self.requests + 1 > cfg.max_requests:
            raise BudgetExhausted("Request limit reached")
        if self.accounted_tokens + size + output > cfg.max_tokens:
            raise BudgetExhausted("Token limit cannot cover the next conservative reservation")
        if self.accounted_cost_usd + cost > cfg.budget_usd + 1e-12:
            raise BudgetExhausted("Dollar budget cannot cover the next conservative reservation")
        self.requests += 1
        self.accounted_tokens += size + output
        self.accounted_cost_usd += cost
        return Reservation(size, output, cost)

    def settle(self, cfg: ExperimentConfig, reservation: Reservation, input_tokens: int,
               output_tokens: int):
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Negative provider usage")
        cost = (input_tokens * cfg.input_usd_per_million
                + output_tokens * cfg.output_usd_per_million) / 1e6
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.estimated_cost_usd += cost
        self.accounted_tokens += input_tokens + output_tokens - (
            reservation.input_tokens + reservation.output_tokens)
        self.accounted_cost_usd += cost - reservation.cost
        if (input_tokens > reservation.input_tokens or output_tokens > reservation.output_tokens):
            raise BudgetExhausted("Provider exceeded conservative reservation; stopping for audit")
