"""Exploratory paired 2x2 contrasts. Seeds are paired run blocks, not LLM determinism."""

import random

import pandas as pd


def factorial_contrasts(data: pd.DataFrame, metric: str, bootstrap_samples=2000) -> pd.DataFrame:
    """Require complete A/B/C/D blocks; never pool mocks/models/settings implicitly."""
    columns = ["contrast", "paired_blocks", "estimate", "bootstrap_low", "bootstrap_high"]
    if data.empty:
        return pd.DataFrame(columns=columns)
    controls = ["backend", "model", "n_agents", "days", "rounds_per_day", "control_fingerprint"]
    if any(data[c].nunique() != 1 for c in controls):
        raise ValueError("Match backend, model, size, duration and generation/budget/memory settings first")
    if data.duplicated(["seed", "condition"]).any():
        raise ValueError("Select exactly one run per seed and condition for paired contrasts")
    pivot = data.pivot(index="seed", columns="condition", values=metric).reindex(columns=list("ABCD"))
    pivot = pivot.dropna()
    if pivot.empty:
        return pd.DataFrame(columns=columns)
    effects = {
        "authority: unequal minus equal": (pivot.A + pivot.B - pivot.C - pivot.D) / 2,
        "labels: explicit minus neutral": (pivot.A + pivot.C - pivot.B - pivot.D) / 2,
        "interaction: (A-C) minus (B-D)": pivot.A - pivot.C - pivot.B + pivot.D,
        "neutral labels: B minus D": pivot.B - pivot.D,
    }
    rng = random.Random(2026)
    rows = []
    for name, series in effects.items():
        values = series.tolist()
        resampled = sorted(sum(rng.choices(values, k=len(values))) / len(values)
                           for _ in range(bootstrap_samples))
        rows.append({"contrast": name, "paired_blocks": len(values), "estimate": series.mean(),
                     # A one-block CI is uninformative: show missing, not false precision.
                     "bootstrap_low": resampled[int(.025 * bootstrap_samples)] if len(values) > 1 else None,
                     "bootstrap_high": resampled[int(.975 * bootstrap_samples)] if len(values) > 1 else None})
    return pd.DataFrame(rows, columns=columns)
