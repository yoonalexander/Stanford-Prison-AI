import random
from pathlib import Path

import yaml

from stanford_ai.models import Agent, ExperimentConfig, Personality, World

CONDITIONS = {
    "A": "Guards / prisoners · unequal authority",
    "B": "Group A / B · unequal authority",
    "C": "Guards / prisoners · equal authority",
    "D": "Group A / B · equal authority",
}


def load_config(path: str | Path, **overrides) -> ExperimentConfig:
    return ExperimentConfig.model_validate(yaml.safe_load(Path(path).read_text()) | overrides)


def initial_world(config: ExperimentConfig) -> World:
    # Separate RNG streams prevent label/permission changes from perturbing traits or order.
    traits_rng = random.Random(config.seed)
    groups = ["A"] * (config.n_agents // 2) + ["B"] * (config.n_agents // 2)
    random.Random(config.seed + 104729).shuffle(groups)
    agents = {}
    for i, group in enumerate(groups):
        agent_id = f"agent_{i + 1:02d}"
        personality = Personality(**{name: round(traits_rng.random(), 3)
                                     for name in Personality.model_fields})
        preference = traits_rng.choice(["garden", "library", "workshop"])
        agents[agent_id] = Agent(
            id=agent_id, group=group, personality=personality,
            goals=["Fulfil facility responsibilities and complete assigned activities.",
                   f"Seek opportunities to use the {preference}.",
                   "Coordinate daily activities with other residents."],
        )
    return World(agents=agents, holdings=dict.fromkeys(agents, 0),
                 allocated_today=dict.fromkeys(agents, 0))
