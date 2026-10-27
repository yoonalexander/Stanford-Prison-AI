import json

import httpx
import pytest
from openai import OpenAI
from pydantic import ValidationError

from stanford_ai.agents import backends
from stanford_ai.agents.backends import BackendFailure, OllamaBackend, OpenAICompatibleBackend
from stanford_ai.dashboard.setup import profile_rows
from stanford_ai.experiments.design import initial_world, load_config
from stanford_ai.experiments.export import export_run
from stanford_ai.metrics.compute import compare_runs
from stanford_ai.models import Action, ExperimentConfig, ModelProfile
from stanford_ai.simulation.budget import BudgetExhausted, Ledger
from stanford_ai.simulation.runner import Runner
from stanford_ai.storage.sqlite import Store


def local(model="local-test", **kwargs):
    return ModelProfile(backend="ollama", model=model, **kwargs)


def test_assignments_are_independent_of_labels_permissions_and_initialization():
    config = ExperimentConfig(model_profiles={"local": local()}, agent_models={"agent_02": "local"})
    assert config.profile_for("agent_02").backend == "ollama"
    assert config.profile_for("agent_01").backend == "mock"
    assert config.backend_summary == "mixed" and config.evidence_class == "mixed_synthetic"
    assert initial_world(config) == initial_world(ExperimentConfig())
    assert set(config.resolved_models()) == set(initial_world(config).agents)


@pytest.mark.parametrize("overrides", [
    {"agent_models": {"agent_99": "local"}},
    {"agent_models": {"agent_01": "missing"}},
    {"model_profiles": {"default": {"backend": "mock", "model": "fixture"}}},
])
def test_invalid_rosters_fail_before_any_decisions(overrides):
    with pytest.raises(ValidationError):
        ExperimentConfig(**overrides)


@pytest.mark.parametrize("url", ["https://ollama.com", "http://192.168.1.4:11434",
                                "http://user:secret@127.0.0.1:11434", "http://127.0.0.1:11434?key=secret"])
def test_free_local_endpoint_cannot_route_to_cloud_or_store_credentials(url):
    with pytest.raises(ValidationError):
        local(base_url=url)


def test_compatible_endpoints_require_explicit_url_and_pricing_confirmation():
    with pytest.raises(ValidationError):
        ModelProfile(backend="openai_compatible", model="test")
    with pytest.raises(ValidationError):
        ModelProfile(backend="openai_compatible", model="test", base_url="https://provider.example/v1")
    profile = ModelProfile(backend="openai_compatible", model="test",
        base_url="https://provider.example/v1", pricing_confirmed=True)
    assert profile.input_usd_per_million == 0  # Explicitly confirmed zero-rate hosted tier supported.


def ollama_transport(calls, *, cloud=False, digest="sha-test", usage=True):
    def handler(request):
        data = json.loads(request.content) if request.content else None
        calls.append((request.url.path, data))
        match request.url.path:
            case "/api/tags":
                body = {"models": [{"name": "local-test", "digest": digest, "size": 123,
                    "remote_host": "ollama.com" if cloud else None, "details": {"format": "gguf"}}]}
            case "/api/show":
                body = {"model_info": {"test.context_length": 131072}}
            case "/api/version":
                body = {"version": "test-version"}
            case "/api/chat":
                assert data["format"]["additionalProperties"] is False
                assert data["options"]["num_predict"] == 400 and data["think"] is False
                body = {"model": "local-test", "message": {"content": Action.make("noop").model_dump_json()},
                        "done": True}
                if usage:
                    body |= {"prompt_eval_count": 100, "eval_count": 50}
            case _:
                raise AssertionError(request.url.path)
        return httpx.Response(200, json=body)
    return httpx.MockTransport(handler)


def test_local_schema_metadata_and_real_token_accounting_at_zero_api_cost(tmp_path, monkeypatch):
    calls = []
    real_class = OllamaBackend
    client = httpx.Client(base_url="http://127.0.0.1:11434/", transport=ollama_transport(calls))
    monkeypatch.setattr(backends, "OllamaBackend", lambda p: real_class(p, client=client))
    config = ExperimentConfig(backend="ollama", model="local-test", n_agents=2, days=1,
                              rounds_per_day=1, budget_usd=0)
    with client, Store(tmp_path / "local.sqlite") as store:
        runner = Runner(config, store)
        runner.run()
        run = store.run(runner.run_id)
        assert run["status"] == "completed" and run["provenance"]["evidence_class"] == "llm"
        assert run["provenance"]["runtime_models"]["default"]["digest"] == "sha-test"
        assert runner.ledger.input_tokens == 200 and runner.ledger.output_tokens == 100
        assert runner.ledger.estimated_cost_usd == runner.ledger.accounted_cost_usd == 0
        assert runner.ledger.accounted_tokens == 300
        assert len([c for c in calls if c[0] == "/api/chat"]) == 2
        assert "/api/pull" not in [c[0] for c in calls]
        path = export_run(store, runner.run_id, tmp_path / "exports")
        assert "local-test" in (path / "actions.csv").read_text()
        assert compare_runs(store, [runner.run_id]).iloc[0].backend == "ollama"


@pytest.mark.parametrize("cloud,pin", [(True, None), (False, "different-digest")])
def test_cloud_or_changed_local_model_is_rejected_before_generation(cloud, pin):
    calls = []
    with httpx.Client(base_url="http://127.0.0.1:11434/", transport=ollama_transport(calls, cloud=cloud)) as client:
        backend = OllamaBackend(local(model_digest=pin), client=client)
        with pytest.raises(BackendFailure):
            backend.prepare()
        assert "/api/chat" not in [c[0] for c in calls]


def test_uninstalled_local_model_does_not_download_or_substitute():
    calls = []
    with httpx.Client(base_url="http://127.0.0.1:11434/", transport=ollama_transport(calls)) as client:
        backend = OllamaBackend(local(model="not-installed"), client=client)
        with pytest.raises(BackendFailure, match="not installed"):
            backend.prepare()
        assert len(calls) == 1


@pytest.mark.parametrize("mode,limit", [("json_schema", "max_tokens"), ("json_object", "max_completion_tokens")])
def test_compatible_api_uses_explicit_formats_and_records_raw_usage(mode, limit):
    payloads = []
    def handler(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        return httpx.Response(200, json={"id": "chat-test", "object": "chat.completion",
            "created": 1, "model": "reported-model-snapshot", "choices": [{"index": 0,
            "message": {"role": "assistant", "content": Action.make("noop").model_dump_json()},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                                                   "total_tokens": 150}})
    profile = ModelProfile(backend="openai_compatible", model="provider-model",
        base_url="https://provider.example/v1", pricing_confirmed=True, output_mode=mode,
        token_limit_field=limit, temperature=None)
    with OpenAI(api_key="test-key", base_url=profile.base_url,
                http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
        result = OpenAICompatibleBackend(profile, client=client).complete([
            {"role": "system", "content": "Return JSON."}])
    assert payloads[0]["response_format"]["type"] == mode
    assert payloads[0][limit] == 400 and "temperature" not in payloads[0]
    assert result.input_tokens == 100 and result.raw["model"] == "reported-model-snapshot"


def test_mixed_router_uses_agent_routes_rates_and_keeps_identity_out_of_prompts(tmp_path, monkeypatch):
    calls = []
    class RoutedBackend:
        def __init__(self, profile):
            self.profile = profile
        def complete(self, prompt):
            from stanford_ai.agents.backends import Completion
            actor = json.loads(prompt[-1]["content"])["you"]["id"]
            calls.append((actor, self.profile.model, prompt))
            return Completion(Action.make("noop").model_dump_json(),
                              {"model": self.profile.model}, 100, 50)
        def close(self):
            pass
    monkeypatch.setattr(backends, "OpenAIBackend", RoutedBackend)
    monkeypatch.setattr(backends, "OpenAICompatibleBackend", RoutedBackend)
    monkeypatch.setattr(backends, "OllamaBackend", RoutedBackend)
    cfg = ExperimentConfig(n_agents=4, days=1, rounds_per_day=1, model_profiles={
        "gpt": ModelProfile(backend="openai", model="PRIVATE_MODEL_A", pricing_confirmed=True,
            input_usd_per_million=1, output_usd_per_million=2),
        "other": ModelProfile(backend="openai_compatible", model="PRIVATE_MODEL_B",
            base_url="https://provider.example/v1", pricing_confirmed=True,
            input_usd_per_million=3, output_usd_per_million=4),
        "local": local("PRIVATE_MODEL_C")},
        agent_models={"agent_01": "gpt", "agent_02": "other", "agent_03": "local"})
    with Store(tmp_path / "mixed.sqlite") as store:
        runner = Runner(cfg, store)
        runner.run()
        assert store.run(runner.run_id)["status"] == "completed"
        assert {a: m for a, m, _ in calls} == {"agent_01": "PRIVATE_MODEL_A",
            "agent_02": "PRIVATE_MODEL_B", "agent_03": "PRIVATE_MODEL_C"}
        assert runner.ledger.estimated_cost_usd == pytest.approx(.0007)
        for actor, model, prompt in calls:
            assert "PRIVATE_MODEL" not in json.dumps(prompt)
        requests = store.requests(runner.run_id)
        assert all(r["result"]["model"]["model"] == cfg.profile_for(r["agent_id"]).model for r in requests)
        comparison = compare_runs(store, [runner.run_id]).iloc[0]
        assert comparison.backend == "mixed" and comparison.evidence_class == "mixed_synthetic"


def test_zero_dollar_budget_still_stops_paid_route_before_a_call(tmp_path):
    class NoCalls:
        calls = 0
        def complete(self, _):
            self.calls += 1
            raise AssertionError("A paid request must never be sent at zero budget")
    backend = NoCalls()
    cfg = ExperimentConfig(backend="openai", pricing_confirmed=True, budget_usd=0)
    with Store(tmp_path / "zero.sqlite") as store:
        runner = Runner(cfg, store, backend)
        runner.run()
        assert store.run(runner.run_id)["status"] == "budget_exhausted" and backend.calls == 0


def test_local_context_limit_does_not_silently_truncate_or_change_memory():
    cfg = ExperimentConfig(backend="ollama", model="local-test", local_context_tokens=4096)
    ledger = Ledger()
    with pytest.raises(BudgetExhausted, match="Local context"):
        ledger.reserve(cfg, [{"role": "user", "content": "x" * 5000}])
    assert ledger.requests == 0


def test_dashboard_profile_table_parses_nullable_fields_and_rejects_duplicates():
    rows = [{"name": "gemma", "backend": "ollama", "model": "gemma3:4b",
             "base_url": "", "temperature": float("nan"), "model_digest": ""}]
    assert profile_rows(rows)["gemma"].temperature is None
    with pytest.raises(ValueError, match="distinct"):
        profile_rows(rows + rows)


def test_free_and_mixed_local_examples_have_no_paid_routes():
    for filename in ["local_free.yaml", "mixed_local.yaml"]:
        config = load_config("configs/" + filename)
        assert config.evidence_class == "llm"
        assert {p["backend"] for p in config.resolved_models().values()} == {"ollama"}
        assert config.budget_usd == 0
