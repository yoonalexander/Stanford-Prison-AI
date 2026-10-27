import pytest

from stanford_ai.environment.engine import Engine
from stanford_ai.experiments.design import initial_world
from stanford_ai.models import ExperimentConfig


@pytest.fixture
def engine():
    config = ExperimentConfig(n_agents=4)
    engine = Engine(config, initial_world(config))
    engine.begin_day(1)
    return engine


@pytest.fixture
def groups(engine):
    return {g: [a.id for a in engine.world.agents.values() if a.group == g] for g in ("A", "B")}
