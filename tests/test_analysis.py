import pandas as pd
import pytest

from stanford_ai.experiments.analysis import factorial_contrasts


def frame():
    rows = []
    for seed in (1, 2, 3):
        for condition, outcome in zip("ABCD", [5, 3, 2, 0], strict=True):
            rows.append(dict(seed=seed, condition=condition, outcome=outcome,
                backend="mock", model="fixture", n_agents=4, days=2, rounds_per_day=2,
                control_fingerprint="same"))
    return pd.DataFrame(rows)


def test_factorial_contrasts_use_complete_paired_blocks():
    result = factorial_contrasts(frame(), "outcome", bootstrap_samples=100)
    assert result.estimate.tolist() == [3, 2, 0, 3]
    assert result.paired_blocks.tolist() == [3] * 4


def test_incomplete_blocks_and_single_block_intervals():
    data = frame()
    assert factorial_contrasts(data[data.condition != "D"], "outcome").empty
    assert factorial_contrasts(data[data.seed == 1], "outcome").bootstrap_low.isna().all()


def test_mixed_models_settings_and_duplicate_pairs_rejected():
    data = frame()
    data.loc[0, "control_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="Match"):
        factorial_contrasts(data, "outcome")
    data = frame()
    with pytest.raises(ValueError, match="exactly one"):
        factorial_contrasts(pd.concat([data, data.iloc[:1]]), "outcome")
