"""OpenAI transport and an explicitly synthetic engineering-only test backend."""

import json
import os
import random
from dataclasses import dataclass

import httpx
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from stanford_ai.agents.prompts import output_format
from stanford_ai.models import Action, ExperimentConfig, ModelProfile


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
    def __init__(self, config: ExperimentConfig | ModelProfile, client=None):
        self.config = config.profile_for("agent_01") if isinstance(config, ExperimentConfig) else config
        self.owns_client = client is None
        self.client = client or sdk_client(self.config)

    def close(self):
        if self.owns_client:
            self.client.close()

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


def sdk_client(profile: ModelProfile):
    key = os.getenv(profile.api_key_env)
    if not key:
        from urllib.parse import urlsplit
        host = urlsplit(profile.endpoint or "").hostname
        if profile.backend == "openai_compatible" and host in {"localhost", "127.0.0.1", "::1"}:
            key = "local-no-key"
        else:
            raise BackendFailure(f"Missing API credential environment variable: {profile.api_key_env}")
    return OpenAI(api_key=key, base_url=profile.endpoint, max_retries=0,
                  timeout=profile.timeout_seconds)


class OpenAICompatibleBackend(OpenAIBackend):
    """Explicit Chat Completions compatibility; never downgrade formats or switch models."""

    def complete(self, prompt: list[dict]) -> Completion:
        cfg = self.config
        kwargs = {"temperature": cfg.temperature} if cfg.temperature is not None else {}
        response_format = ({"type": "json_schema", "json_schema": {
            k: v for k, v in output_format().items() if k != "type"}}
            if cfg.output_mode == "json_schema" else {"type": "json_object"})
        try:
            response = self.client.chat.completions.create(
                model=cfg.model, messages=prompt, response_format=response_format,
                **{cfg.token_limit_field: cfg.max_output_tokens}, **kwargs)
        except RateLimitError as exc:
            raise BackendFailure("Compatible endpoint rate limit (429)", retryable=True) from exc
        except APIStatusError as exc:
            raise BackendFailure(f"Compatible endpoint HTTP {exc.status_code}") from exc
        except APIConnectionError as exc:
            raise BackendFailure("Compatible endpoint connection/timeout; billing status unknown") from exc
        content = response.choices[0].message.content if response.choices else ""
        return Completion(content or "", response.model_dump(mode="json"),
                          response.usage.prompt_tokens if response.usage else None,
                          response.usage.completion_tokens if response.usage else None)


def local_models(base_url="http://127.0.0.1:11434") -> list[dict]:
    # Reuse validation so discovery cannot accidentally contact a cloud endpoint in free mode.
    ModelProfile(backend="ollama", model="discovery", base_url=base_url)
    with httpx.Client(base_url=base_url.rstrip("/") + "/", timeout=5,
                      trust_env=False, follow_redirects=False) as client:
        response = client.get("api/tags")
        response.raise_for_status()
        return response.json()["models"]


class OllamaBackend:
    """Native local-only Ollama API: schema constraints, token counts, no API credentials."""

    def __init__(self, profile: ModelProfile, client=None):
        self.config = profile
        self.owns_client = client is None
        self.client = client or httpx.Client(
            base_url=profile.endpoint.rstrip("/") + "/", timeout=profile.timeout_seconds,
            trust_env=False, follow_redirects=False)
        self.runtime_info = {}

    def close(self):
        if self.owns_client:
            self.client.close()

    def prepare(self):
        try:
            tags = self._json("GET", "api/tags")["models"]
            requested = self.config.model
            entry = next((m for m in tags if m["name"] in {requested, requested + ":latest"}), None)
            if not entry:
                raise BackendFailure(f"Local model {requested} is not installed; no download/fallback attempted")
            details = self._json("POST", "api/show", {"model": entry["name"]})
            if (entry.get("remote_host") or entry.get("remote_model") or details.get("remote_host")
                    or details.get("remote_model") or not details.get("model_info")):
                raise BackendFailure("Free local backend refuses cloud-routed models")
            if self.config.model_digest and self.config.model_digest != entry["digest"]:
                raise BackendFailure("Installed model digest differs from the configured study model")
            lengths = [v for k, v in details["model_info"].items() if k.endswith(".context_length")]
            if lengths and self.config.local_context_tokens > min(lengths):
                raise BackendFailure("Configured local context exceeds this model's supported context")
            self.runtime_info = {"model": entry["name"], "digest": entry["digest"],
                "details": entry.get("details", {}), "size_bytes": entry.get("size"),
                "ollama_version": self._json("GET", "api/version").get("version"),
                "local_only": True, "num_ctx": self.config.local_context_tokens,
                "think": False}
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise BackendFailure("Local Ollama discovery failed; start Ollama and check the installed model") from exc

    def _json(self, method, path, data=None):
        response = self.client.request(method, path, json=data)
        response.raise_for_status()
        return response.json()

    def complete(self, prompt: list[dict]) -> Completion:
        cfg = self.config
        options = {"num_predict": cfg.max_output_tokens, "num_ctx": cfg.local_context_tokens}
        if cfg.temperature is not None:
            options["temperature"] = cfg.temperature
        try:
            raw = self._json("POST", "api/chat", {"model": self.runtime_info["model"],
                "messages": prompt, "format": output_format()["schema"], "stream": False,
                "think": False, "options": options})
        except httpx.HTTPStatusError as exc:
            raise BackendFailure(f"Local Ollama HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise BackendFailure("Local Ollama connection/timeout; no substitute action generated") from exc
        except ValueError as exc:
            raise BackendFailure("Local Ollama returned an invalid transport response") from exc
        return Completion(raw.get("message", {}).get("content", ""), raw,
                          raw.get("prompt_eval_count"), raw.get("eval_count"))


class BackendRouter:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.backends = {}
        self.runtime_info = {}

    def prepare(self):
        # Prepare every used profile before any agent decision, catching unavailable routes early.
        for actor in self.config.resolved_models():
            name = self.config.agent_models.get(actor, "default")
            if name in self.backends:
                continue
            profile = self.config.profile_for(actor)
            backend = {"mock": lambda: MockBackend(self.config.seed),
                       "openai": lambda: OpenAIBackend(profile),
                       "openai_compatible": lambda: OpenAICompatibleBackend(profile),
                       "ollama": lambda: OllamaBackend(profile)}[profile.backend]()
            self.backends[name] = backend
            if hasattr(backend, "prepare"):
                backend.prepare()
            self.runtime_info[name] = {"backend": profile.backend, "model": profile.model,
                                       "endpoint": profile.endpoint,
                                       **getattr(backend, "runtime_info", {})}

    def for_agent(self, actor):
        return self.backends[self.config.agent_models.get(actor, "default")]

    def close(self):
        for backend in self.backends.values():
            if hasattr(backend, "close"):
                backend.close()


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
