"""Use the real SDK over a mocked HTTP transport; never call the remote API."""

import json

import httpx
from openai import OpenAI

from stanford_ai.agents.backends import OpenAIBackend
from stanford_ai.models import Action, ExperimentConfig
from stanford_ai.simulation.runner import Runner
from stanford_ai.storage.sqlite import Store


def test_official_sdk_structured_output_transport_and_usage(tmp_path):
    payloads = []
    action = Action.make("communicate", message="Ready to coordinate today's activities.")

    def handler(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        assert request.url.path == "/v1/responses"
        assert payload["text"]["format"]["strict"] is True
        assert payload["text"]["format"]["schema"]["additionalProperties"] is False
        assert payload["store"] is False
        return httpx.Response(200, json={
            "id": "resp_test", "object": "response", "created_at": 1,
            "status": "completed", "model": payload["model"],
            "output": [{"type": "message", "id": "msg_test", "role": "assistant",
                        "status": "completed", "content": [{"type": "output_text",
                        "text": action.model_dump_json(), "annotations": []}]}],
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}},
        })

    cfg = ExperimentConfig(backend="openai", pricing_confirmed=True, n_agents=2,
                            days=1, rounds_per_day=1)
    client = OpenAI(api_key="test-not-a-real-key", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        with Store(tmp_path / "study.sqlite") as store:
            runner = Runner(cfg, store, OpenAIBackend(cfg, client=client))
            runner.run()
            assert store.run(runner.run_id)["status"] == "completed"
            assert len(payloads) == 2
            assert runner.ledger.input_tokens == 200
            assert runner.ledger.output_tokens == 100
            assert store.requests(runner.run_id)[0]["result"]["raw"]["id"] == "resp_test"
    finally:
        client.close()
