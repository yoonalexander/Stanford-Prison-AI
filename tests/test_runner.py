import json

import pytest

from stanford_ai.agents.backends import BackendFailure, Completion
from stanford_ai.experiments.export import export_run
from stanford_ai.metrics.compute import action_frame, classify, compare_runs, gini
from stanford_ai.models import Action, ExperimentConfig
from stanford_ai.simulation.budget import BudgetExhausted, Ledger
from stanford_ai.simulation.runner import Runner
from stanford_ai.storage.sqlite import Store, digest, snapshot


def test_mock_persistence_replay_exports_and_seed(tmp_path):
    cfg = ExperimentConfig(days=2, n_agents=4)
    with Store(tmp_path / "study.sqlite") as store:
        a = Runner(cfg, store)
        run_id = a.run()
        b = Runner(cfg, store)
        b.run()
        assert store.run(run_id)["status"] == "completed"
        assert len(store.requests(run_id)) == 16
        assert store.replay(run_id).history == store.replay(b.run_id).history
        assert snapshot(store.replay(run_id)) == snapshot(a.engine.world)
        previous = digest(store.run(run_id)["initial_state"])
        for row in store.events(run_id):
            assert row["before_hash"] == previous
            assert digest(snapshot(store.replay(run_id, row["seq"]))) == row["after_hash"]
            previous = row["after_hash"]
        output = export_run(store, run_id, tmp_path / "exports")
        raw = json.loads((output / "raw.json").read_text(encoding="utf-8"))
        assert len(raw["events"]) == 18
        assert len(raw["snapshots"]) == 19
        assert (output / "actions.csv").is_file()
        assert (output / "metrics.csv").is_file()
        assert raw["requests"][0]["prompt"][0]["role"] == "system"
        data = compare_runs(store, [run_id, b.run_id])
        assert len(data) == 2 and set(data.backend) == {"mock"}


def test_request_budget_stops_without_replacement_actions(tmp_path):
    cfg = ExperimentConfig(max_requests=2, days=10)
    with Store(tmp_path / "study.sqlite") as store:
        runner = Runner(cfg, store)
        runner.run()
        assert store.run(runner.run_id)["status"] == "budget_exhausted"
        assert len(store.requests(runner.run_id)) == 2
        assert len(action_frame(store.replay(runner.run_id))) == 2


class StubBackend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    def complete(self, _):
        self.calls += 1
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return Completion(value, {"output_text": value}, 20, 10)


def paid(**overrides):
    return ExperimentConfig(backend="openai", pricing_confirmed=True, days=1,
                            rounds_per_day=1, retry_delay_seconds=0, **overrides)


@pytest.mark.parametrize("budget", ["max_tokens", "budget_usd"])
def test_paid_reservation_rejects_before_transport(tmp_path, budget):
    cfg = paid(**{budget: 1 if budget == "max_tokens" else 0.000001})
    backend = StubBackend([])
    with Store(tmp_path / "study.sqlite") as store:
        runner = Runner(cfg, store, backend)
        runner.run()
        assert store.run(runner.run_id)["status"] == "budget_exhausted"
        assert backend.calls == 0


def test_invalid_outputs_logged_retried_without_prompt_changes(tmp_path):
    backend = StubBackend(["not json", "{}", "[]"])
    with Store(tmp_path / "study.sqlite") as store:
        runner = Runner(paid(), store, backend)
        runner.run()
        requests = store.requests(runner.run_id)
        assert store.run(runner.run_id)["status"] == "failed"
        assert len(requests) == 3
        assert requests[0]["prompt"] == requests[1]["prompt"] == requests[2]["prompt"]
        assert all(r["result"]["status"] == "invalid_output" for r in requests)
        assert runner.ledger.input_tokens == 60 and runner.ledger.output_tokens == 30
        assert not action_frame(store.replay(runner.run_id)).shape[0]


def test_unknown_billing_keeps_reservation_and_stops(tmp_path):
    backend = StubBackend([BackendFailure("timeout")])
    with Store(tmp_path / "study.sqlite") as store:
        runner = Runner(paid(), store, backend)
        runner.run()
        assert runner.ledger.unknown_usage_requests == 1
        assert runner.ledger.accounted_tokens > 400
        assert runner.ledger.input_tokens == 0
        assert store.run(runner.run_id)["status"] == "failed"


def test_response_without_usage_preserves_raw_output_and_stops(tmp_path):
    class MissingUsage:
        def complete(self, _):
            return Completion("{}", {"id": "resp_missing_usage", "output": []}, None, None)
    with Store(tmp_path / "study.sqlite") as store:
        runner = Runner(paid(), store, MissingUsage())
        runner.run()
        result = store.requests(runner.run_id)[0]["result"]
        assert result["status"] == "missing_usage"
        assert result["raw"]["id"] == "resp_missing_usage"
        assert runner.ledger.unknown_usage_requests == 1
        assert store.run(runner.run_id)["status"] == "failed"


def test_rate_limit_attempts_are_budgeted_and_logged(tmp_path):
    noop = Action.make("noop").model_dump_json()
    backend = StubBackend([BackendFailure("429", retryable=True), noop, noop, noop, noop])
    with Store(tmp_path / "study.sqlite") as store:
        runner = Runner(paid(), store, backend)
        runner.run()
        assert store.run(runner.run_id)["status"] == "completed"
        assert runner.ledger.requests == 5 and runner.ledger.unknown_usage_requests == 1
        assert runner.ledger.accounted_tokens > 120  # 4 actual completions + retained reservation.


def test_pause_stop_preserves_data_without_extra_calls(tmp_path):
    with Store(tmp_path / "study.sqlite") as store:
        runner = Runner(ExperimentConfig(), store)
        runner.run(should_stop=lambda: True)
        assert store.run(runner.run_id)["status"] == "stopped"
        assert len(store.events(runner.run_id)) == 1
        assert not store.requests(runner.run_id)


def test_recover_pending_request_retains_reserved_usage(tmp_path):
    from stanford_ai.experiments.design import initial_world
    with Store(tmp_path / "study.sqlite") as store:
        cfg = ExperimentConfig()
        ledger = Ledger(requests=1, accounted_tokens=100, accounted_cost_usd=0.1)
        store.create("abandoned", cfg.model_dump(), {}, initial_world(cfg), ledger.as_dict())
        store.start_request("abandoned", 1, "agent_01", 1, 1, [], {}, ledger.as_dict())
        store.recover("abandoned")
        result = store.run("abandoned")
        assert result["status"] == "interrupted"
        assert result["usage"]["accounted_tokens"] == 100
        assert result["usage"]["unknown_usage_requests"] == 1


def test_observable_classifications_and_gini(engine, groups):
    admin, resident = groups["A"][0], groups["B"][0]
    for zone in ["garden", "library", "workshop"]:
        event = engine.apply(admin, Action.make("restrict_access", target=resident, zone=zone))
    assert set(classify(event, engine.world)) == {"coercive", "authoritarian"}
    denied = engine.apply(resident, Action.make("restrict_access", target=admin, zone="garden"))
    assert classify(denied, engine.world) == ["neutral"]
    assert gini([1, 1, 1]) == 0 and gini([0, 0, 3]) == pytest.approx(2 / 3)


def test_reservation_settlement_releases_unused_capacity():
    cfg = paid()
    ledger = Ledger()
    reserved = ledger.reserve(cfg, [{"role": "user", "content": "hello"}])
    ledger.settle(cfg, reserved, 10, 5)
    assert ledger.accounted_tokens == 15
    assert ledger.accounted_cost_usd == pytest.approx((10 * .4 + 5 * 1.6) / 1e6)
    with pytest.raises(BudgetExhausted):
        Ledger().reserve(paid(max_requests=1, max_tokens=1), [])
