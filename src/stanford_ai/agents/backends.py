"""OpenAI transport and an explicitly synthetic engineering-only test backend."""

import json
import random
from dataclasses import dataclass

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from stanford_ai.agents.prompts import output_format
from stanford_ai.models import Action, ExperimentConfig


@dataclass
class Completion:
    text: str
    raw: dict
    input_tokens: int | None
    output_tokens: int | None


class BackendFailure(Exception):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class OpenAIBackend:
    def __init__(self, config: ExperimentConfig, client=None):
        self.config = config
        self.client = client or OpenAI(max_retries=0, timeout=config.timeout_seconds)

    def complete(self, prompt: list[dict]) -> Completion:
        cfg = self.config
        kwargs = {"temperature": cfg.temperature} if cfg.temperature is not None else {}
        try:
            response = self.client.responses.create(
                model=cfg.model, input=prompt, text={"format": output_format()},
                max_output_tokens=cfg.max_output_tokens, store=False, **kwargs,
            )
        except RateLimitError as exc:
            raise BackendFailure("OpenAI rate limit (429)", retryable=True) from exc
        except APIStatusError as exc:
            # Avoid persisting exception bodies which might contain account details.
            raise BackendFailure(f"OpenAI HTTP {exc.status_code}") from exc
        except APIConnectionError as exc:
            raise BackendFailure("OpenAI connection/timeout; billing status unknown") from exc
        return Completion(response.output_text, response.model_dump(mode="json"),
                          response.usage.input_tokens if response.usage else None,
                          response.usage.output_tokens if response.usage else None)


class MockBackend:
    """Seeded fixture sampler. Its choices provide NO behavioural research evidence."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed + 49999)

    def complete(self, prompt: list[dict]) -> Completion:
        o = json.loads(prompt[-1]["content"])
        me = o["you"]["id"]
        others = [a["id"] for a in o["residents"] if a["id"] != me]
        target = self.rng.choice(others)
        options = [Action.make("noop"), Action.make("communicate", target=target,
                    private=self.rng.choice([True, False]), message="Can we coordinate today's work?"),
                   Action.make("request_resource", amount=1, resource="leisure_credits"),
                   Action.make("offer_agreement", target=target,
                               message="Let us coordinate use of shared materials.")]
        for ref, task in o["tasks"].items():
            if not task["completed"]:
                if me == task["assigned"] or me in task["helpers"]:
                    options.append(Action.make("complete_task", reference=ref))
                else:
                    options.append(Action.make("volunteer_task", reference=ref))
                if task["assigned"] is None:
                    options.append(Action.make("assign_task", reference=ref, target=target))
        options.append(Action.make("propose_rule", amount=self.rng.randint(0, 3), scope="all",
                                   resource="leisure_credits", message="Proposed daily credit cap."))
        for ref, rule in o["rules"].items():
            if rule["status"] == "proposed":
                options.append(Action.make("approve_rule", reference=ref, approve=True))
                if me not in rule["votes"]:
                    options.append(Action.make("vote_rule", reference=ref, approve=True))
        for ref, agreement in o["agreements"].items():
            if agreement["target"] == me and agreement["status"] == "offered":
                options.append(Action.make("accept_agreement", reference=ref))
        for agent_id in o["resources"]["holdings"]:
            if o["resources"]["pool"] > 0 and o["resources"]["allocated_today"][agent_id] < 3:
                options.append(Action.make("allocate_resource", target=agent_id, amount=1,
                                           resource="leisure_credits"))
        options.extend([Action.make("restrict_access", target=target, zone="garden"),
                        Action.make("change_schedule", zone="library")])
        options = [a for a in options if a.kind in o["available_actions"]]
        action = self.rng.choice(options)
        text = action.model_dump_json()
        return Completion(text, {"backend": "mock", "synthetic": True, "output_text": text},
                          0, 0)
