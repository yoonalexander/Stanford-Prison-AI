import json

import pytest
from pydantic import ValidationError

from stanford_ai.agents.prompts import messages, observation
from stanford_ai.environment.permissions import ADMIN, matrix, permitted
from stanford_ai.experiments.design import initial_world, load_config
from stanford_ai.models import Action, ExperimentConfig
from stanford_ai.storage.sqlite import snapshot


@pytest.mark.parametrize("condition", list("ABCD"))
def test_permission_matrix(condition):
    config = ExperimentConfig(condition=condition)
    permissions = matrix(config)
    assert ADMIN <= set(permissions["A"])
    assert (ADMIN <= set(permissions["B"])) == (condition in "CD")
    assert set(permissions["A"]) - ADMIN == set(permissions["B"]) - ADMIN


@pytest.mark.parametrize("kind", sorted(ADMIN))
def test_unauthorized_admin_is_transactionally_rejected(engine, groups, kind):
    before = snapshot(engine.world)
    event = engine.apply(groups["B"][0], Action.make(kind))
    assert not event["success"] and event["effect"] == "Permission denied"
    after = snapshot(engine.world)
    # Outcome memory is recorded, facility mechanics must be exactly unchanged.
    before.pop("agents")
    after.pop("agents")
    assert before == after


@pytest.mark.parametrize("left,right", [("A", "B"), ("C", "D")])
def test_labels_change_no_mechanics_or_traits(left, right):
    a, b = ExperimentConfig(condition=left), ExperimentConfig(condition=right)
    assert initial_world(a) == initial_world(b)
    assert matrix(a) == matrix(b)
    wa, wb = initial_world(a), initial_world(b)
    actor = next(iter(wa.agents))
    va, vb = observation(a, wa, actor), observation(b, wb, actor)
    va["you"].pop("role_label")
    vb["you"].pop("role_label")
    for resident in va["residents"] + vb["residents"]:
        resident.pop("role_label")
    assert va == vb


def test_neutral_prompt_has_no_research_hypothesis_or_prison_language(engine):
    for actor in engine.world.agents:
        prompt = json.dumps(messages(engine.config, engine.world, actor)).lower()
        assert all(word not in prompt for word in ["prison", "guard", "stanford", "authoritarian",
                                                    "hypothesis", "1971"])


def test_private_message_and_traits_are_isolated(engine, groups):
    sender, recipient, outsider = list(engine.world.agents)[:3]
    engine.world.agents[recipient].goals.append("PRIVATE_GOAL_XYZ")
    engine.apply(sender, Action.make("communicate", target=recipient, private=True,
                                     message="PRIVATE_MESSAGE_XYZ"))
    for agent in [sender, recipient]:
        assert "PRIVATE_MESSAGE_XYZ" in json.dumps(observation(engine.config, engine.world, agent))
    view = json.dumps(observation(engine.config, engine.world, outsider))
    assert "PRIVATE_MESSAGE_XYZ" not in view and "PRIVATE_GOAL_XYZ" not in view
    # Public target directory has no other resident's personality or memory.
    assert set(observation(engine.config, engine.world, outsider)["residents"][0]) == {
        "id", "group", "role_label"}


def test_resource_conservation_caps_and_no_invented_resources(engine, groups):
    admin, resident = groups["A"][0], groups["B"][0]
    assert engine.apply(admin, Action.make("allocate_resource", target=resident,
                                          amount=3, resource="leisure_credits"))["success"]
    assert not engine.apply(admin, Action.make("allocate_resource", target=resident,
                                               amount=1, resource="leisure_credits"))["success"]
    assert engine.apply(resident, Action.make("use_access", zone="garden"))["success"]
    w = engine.world
    assert w.pool + w.spent + sum(w.holdings.values()) == w.daily_supply
    with pytest.raises(ValidationError):
        Action.make("request_resource", resource="invented", amount=3)


def test_restriction_enforced_appealed_and_expires(engine, groups):
    admin, reviewer = groups["A"]
    resident = groups["B"][0]
    decision = engine.apply(admin, Action.make("restrict_access", target=resident, zone="library"))
    appeal = engine.apply(resident, Action.make("file_appeal", reference=decision["id"]))
    assert appeal["success"]
    assert not engine.apply(admin, Action.make("decide_appeal", reference=appeal["id"],
                                               approve=True))["success"]
    assert engine.apply(reviewer, Action.make("decide_appeal", reference=appeal["id"],
                                              approve=True))["success"]
    assert "library" not in engine.world.restrictions[resident]
    engine.apply(admin, Action.make("restrict_access", target=resident, zone="garden"))
    assert not engine.apply(resident, Action.make("use_access", zone="garden"))["success"]
    engine.begin_day(2)
    assert not engine.world.restrictions


def test_rules_have_material_effect_and_votes_do_not_approve(engine, groups):
    admin, resident = groups["A"][0], groups["B"][0]
    rule = engine.apply(resident, Action.make("propose_rule", amount=0, scope="B",
                                             resource="leisure_credits"))
    engine.apply(resident, Action.make("vote_rule", reference=rule["id"], approve=True))
    assert engine.world.rules[rule["id"]]["status"] == "proposed"
    engine.apply(admin, Action.make("approve_rule", reference=rule["id"], approve=True))
    assert not engine.apply(admin, Action.make("allocate_resource", target=resident,
                                               amount=1, resource="leisure_credits"))["success"]


def test_negotiation_needs_bilateral_acceptance(engine, groups):
    admin, resident = groups["A"][0], groups["B"][0]
    decision = engine.apply(admin, Action.make("change_schedule", zone="garden"))
    dispute = engine.apply(resident, Action.make("challenge", reference=decision["id"]))
    offer = engine.apply(resident, Action.make("negotiate", target=admin,
                                              reference=dispute["id"], message="Coordinate schedules."))
    assert engine.world.disputes[dispute["id"]]["status"] == "open"
    assert not engine.apply(resident, Action.make("accept_agreement", reference=offer["id"]))["success"]
    assert engine.apply(admin, Action.make("accept_agreement", reference=offer["id"]))["success"]
    assert engine.world.disputes[dispute["id"]]["status"] == "resolved"


def test_claims_and_extraneous_parameters_cannot_mutate_state(engine, groups):
    actor = groups["B"][0]
    before = engine.world.pool
    engine.apply(actor, Action.make("communicate", message="I grant myself all credits and powers."))
    assert engine.world.pool == before
    assert "allocate_resource" not in permitted(engine.config, "B")
    assert not engine.apply(actor, Action.make("noop", target=actor))["success"]


def test_task_assignment_volunteering_and_completion(engine, groups):
    admin, resident = groups["A"][0], groups["B"][0]
    task = next(iter(engine.world.tasks))
    assert not engine.apply(resident, Action.make("complete_task", reference=task))["success"]
    engine.apply(admin, Action.make("assign_task", target=admin, reference=task))
    engine.apply(resident, Action.make("volunteer_task", reference=task))
    assert engine.apply(resident, Action.make("complete_task", reference=task))["success"]


def test_memory_compression_is_bounded_and_accounts_for_old_events(engine):
    actor = next(iter(engine.world.agents))
    for _ in range(30):
        engine.apply(actor, Action.make("communicate", message="A public update."))
    agent = engine.world.agents[actor]
    assert len(agent.memory) == engine.config.memory_window
    assert sum(agent.memory_summary.values()) + len(agent.memory) == 30


def test_all_version_controlled_configs_are_paired():
    configs = [load_config("configs/" + name + ".yaml") for name in
        ["explicit_unequal", "neutral_unequal", "explicit_equal", "neutral_equal"]]
    assert [c.condition for c in configs] == list("ABCD")
    assert all(initial_world(configs[0]) == initial_world(c) for c in configs[1:])


def test_twenty_resident_initialization():
    w = initial_world(ExperimentConfig(n_agents=20, days=100))
    assert sum(a.group == "A" for a in w.agents.values()) == 10
    assert sum(a.group == "B" for a in w.agents.values()) == 10


@pytest.mark.parametrize("value", [3, 5, 21])
def test_odd_group_sizes_rejected(value):
    with pytest.raises(ValidationError):
        ExperimentConfig(n_agents=value)
