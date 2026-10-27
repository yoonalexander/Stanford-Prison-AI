import hashlib
import json
import random
import subprocess
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from pydantic import ValidationError

from stanford_ai import __version__
from stanford_ai.agents.backends import BackendFailure, MockBackend, OpenAIBackend
from stanford_ai.agents.prompts import PROMPT_VERSION, messages, output_format
from stanford_ai.environment.engine import Engine
from stanford_ai.environment.permissions import matrix
from stanford_ai.experiments.design import initial_world
from stanford_ai.metrics.compute import RUBRIC_VERSION, measure
from stanford_ai.models import Action, ExperimentConfig
from stanford_ai.simulation.budget import BudgetExhausted, Ledger
from stanford_ai.storage.sqlite import Store, snapshot


def provenance(config: ExperimentConfig) -> dict:
    package = Path(__file__).resolve().parents[1]
    hashes = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(package.rglob("*.py"))}
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package, capture_output=True, text=True,
                             timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git = "uncommitted"
    import importlib.metadata
    versions = {name: importlib.metadata.version(name)
                for name in ("openai", "pydantic", "streamlit", "pandas", "plotly")}
    return {"software_version": __version__, "git_commit": git, "source_sha256": hashes,
            "dependencies": versions, "prompt_version": PROMPT_VERSION,
            "rubric_version": RUBRIC_VERSION, "permission_matrix": matrix(config),
            "response_format": output_format(),
            "generation_parameters": {"temperature": config.temperature,
                                      "max_output_tokens": config.max_output_tokens},
            "mock_is_synthetic": config.backend == "mock"}


class Runner:
    def __init__(self, config: ExperimentConfig, store: Store, backend=None):
        self.config, self.store = config, store
        self.backend = backend
        self.ledger = Ledger()
        self.run_id = str(uuid.uuid4())
        self.engine = Engine(config, initial_world(config))

    def run(self, on_event=None, should_stop=None) -> str:
        cfg = self.config
        self.store.create(self.run_id, cfg.model_dump(mode="json"), provenance(cfg),
                          self.engine.world, self.ledger.as_dict())
        status, reason = "completed", None
        try:
            backend = self.backend or (MockBackend(cfg.seed) if cfg.backend == "mock"
                                       else OpenAIBackend(cfg))
            for day in range(1, cfg.days + 1):
                before = snapshot(self.engine.world)
                event = self.engine.begin_day(day)
                self._persist(event, before, on_event)
                for round_ in range(1, cfg.rounds_per_day + 1):
                    order = list(self.engine.world.agents)
                    random.Random(cfg.seed + day * 1009 + round_ * 9176).shuffle(order)
                    for actor in order:
                        if should_stop and should_stop():
                            status, reason = "stopped", "Stopped by user"
                            return self.run_id
                        before = snapshot(self.engine.world)
                        self.engine.world.round = round_
                        prompt = messages(cfg, self.engine.world, actor)
                        action = self._decide(backend, prompt, actor)
                        event = self.engine.apply(actor, action)
                        self._persist(event, before, on_event)
        except BudgetExhausted as exc:
            status, reason = "budget_exhausted", str(exc)
        except BackendFailure as exc:
            status, reason = "failed", str(exc)
        except KeyboardInterrupt:
            status, reason = "interrupted", "Interrupted by user"
        except BaseException as exc:
            status, reason = "failed", f"Unexpected {type(exc).__name__}; inspect local logs"
            raise
        finally:
            self.store.status(self.run_id, status, reason, self.ledger.as_dict())
        return self.run_id

    def _persist(self, event, before, callback):
        self.store.record(self.run_id, event, before, self.engine.world, measure(self.engine.world))
        if callback:
            callback(self.run_id, event, self.engine.world, self.ledger)

    def _decide(self, backend, prompt, actor) -> Action:
        cfg = self.config
        for attempt in range(cfg.max_retries + 1):
            reservation = self.ledger.reserve(cfg, prompt)
            ordinal = self.ledger.requests
            self.store.start_request(self.run_id, ordinal, actor, self.engine.world.day,
                                     self.engine.world.round, prompt, asdict(reservation),
                                     self.ledger.as_dict())
            result = {}
            retryable = False
            try:
                completion = backend.complete(prompt)
            except BaseException as exc:
                # No usage returned: retain the full reservation instead of assuming zero billing.
                self.ledger.unknown_usage_requests += 1
                result = {"status": "transport_error", "error_type": type(exc).__name__,
                          "usage_unknown": True}
                self.store.finish_request(self.run_id, ordinal, result, self.ledger.as_dict())
                if isinstance(exc, BackendFailure) and exc.retryable and attempt < cfg.max_retries:
                    time.sleep(min(30, cfg.retry_delay_seconds * 2**attempt))
                    continue
                raise
            result = {"raw": completion.raw, "text": completion.text,
                      "input_tokens": completion.input_tokens, "output_tokens": completion.output_tokens,
                      "usage_unknown": False}
            if completion.input_tokens is None or completion.output_tokens is None:
                self.ledger.unknown_usage_requests += 1
                result.update(status="missing_usage", usage_unknown=True)
                self.store.finish_request(self.run_id, ordinal, result, self.ledger.as_dict())
                raise BackendFailure("Provider returned no usage; billing status unknown")
            try:
                self.ledger.settle(cfg, reservation, completion.input_tokens, completion.output_tokens)
                action = Action.model_validate_json(completion.text, strict=True)
                result.update(status="parsed", structured_action=action.model_dump(mode="json"))
            except ValidationError as exc:
                result.update(status="invalid_output", validation_errors=json.loads(exc.json()))
                retryable = True
            except BudgetExhausted:
                result["status"] = "reservation_exceeded"
                raise
            finally:
                self.store.finish_request(self.run_id, ordinal, result, self.ledger.as_dict())
            if not retryable:
                return action
            if attempt < cfg.max_retries:
                time.sleep(min(30, cfg.retry_delay_seconds * 2**attempt))
        raise BackendFailure("Invalid/refused/incomplete model output after bounded identical-prompt retries")
