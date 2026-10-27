"""Observable proxies only. No inferred motives, emotions, or consciousness."""

from collections import Counter

import pandas as pd

from stanford_ai.models import World

RUBRIC_VERSION = "1.0"
RESTRICTIVE = {"restrict_access", "approve_rule"}
COOPERATIVE = {"accept_agreement", "volunteer_task", "complete_task"}
RESISTANT = {"challenge", "file_appeal", "request_exception"}
ASSERTIVE = {"propose_rule", "assign_task", "change_schedule", "negotiate"}


def is_restrictive(event: dict, world: World) -> bool:
    a = event.get("action")
    if not event["success"] or not a:
        return False
    if a["kind"] == "restrict_access":
        return True
    if a["kind"] == "approve_rule" and a["approve"]:
        rule = world.rules.get(a["reference"], {})
        return rule.get("cap", 3) < 3
    return False


def classify(event: dict, world: World, restrictive_count: int | None = None) -> list[str]:
    """Multi-label, action-based descriptors; 'authoritarian' is a repeated-action proxy."""
    if not event["success"] or not event.get("action"):
        return ["neutral"]
    kind = event["action"]["kind"]
    labels = []
    if kind in COOPERATIVE:
        labels.append("cooperative")
    if kind in RESISTANT:
        labels.append("resistant")
    if kind in ASSERTIVE:
        labels.append("assertive")
    if is_restrictive(event, world):
        labels.append("coercive")
        if restrictive_count is None:
            restrictive_count = sum(e["actor"] == event["actor"]
                and event["day"] - 2 <= e["day"] <= event["day"] and e["id"] <= event["id"]
                and is_restrictive(e, world) for e in world.history)
        if restrictive_count >= 3:
            labels.append("authoritarian")
    return labels or ["neutral"]


def classification_map(world: World) -> dict[str, list[str]]:
    from collections import defaultdict, deque
    recent = defaultdict(deque)
    result = {}
    for event in world.history:
        queue = recent[event["actor"]]
        while queue and queue[0] < event["day"] - 2:
            queue.popleft()
        if is_restrictive(event, world):
            queue.append(event["day"])
        result[event["id"]] = classify(event, world, len(queue))
    return result


def gini(values: list[int]) -> float:
    if not values or sum(values) == 0:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    return sum((2 * i - n - 1) * x for i, x in enumerate(ordered, 1)) / (n * sum(ordered))


def measure(world: World) -> dict:
    actions = [e for e in world.history if e.get("action")]
    successful = [e for e in actions if e["success"]]
    counts = Counter(e["action"]["kind"] for e in successful)
    restrictive = sum(is_restrictive(e, world) for e in successful)
    accepted = [a for a in world.agreements.values() if a["status"] == "accepted"]
    cross = sum(world.agents[a["proposer"]].group != world.agents[a["target"]].group
                for a in accepted)
    classifications = classification_map(world)
    labels = Counter(label for e in successful for label in classifications[e["id"]])
    active_caps = [(3 - r["cap"]) / 3 for r in world.rules.values() if r["status"] == "active"]
    reviews = Counter(a["status"] for a in world.appeals.values())
    open_disputes = sum(d["status"] == "open" for d in world.disputes.values())
    # Escalation proxy: restrictive response by a challenged actor after the challenge, same day.
    escalations = sum(any(e["actor"] == d["other"] and e["id"] > ref
                         and e["day"] == next(x["day"] for x in world.history if x["id"] == ref)
                         and is_restrictive(e, world) for e in successful)
                      for ref, d in world.disputes.items())
    return {
        "seq": len(world.history), "day": world.day, "round": world.round,
        "attempts": len(actions), "successful_actions": len(successful),
        "failed_actions": len(actions) - len(successful),
        "rules_proposed": counts["propose_rule"],
        "rules_approved": sum(e["action"]["kind"] == "approve_rule" and e["action"]["approve"]
                              for e in successful),
        "mean_active_rule_restrictiveness": sum(active_caps) / len(active_caps) if active_caps else 0,
        "sanctions": counts["restrict_access"], "restrictive_decisions": restrictive,
        "resource_gini": gini(list(world.allocated_today.values())),
        "appeals_approved": reviews["approved"], "appeals_denied": reviews["denied"],
        "successful_negotiations": sum(a["dispute"] is not None for a in accepted),
        "agreements_accepted": len(accepted),
        "shared_resource_agreements": sum(a.get("resource") is not None for a in accepted),
        "cross_group_collaboration": cross,
        "voluntary_assistance": counts["volunteer_task"],
        "cooperation": labels["cooperative"], "conflict": counts["challenge"] + counts["file_appeal"],
        "disputed_decisions": counts["challenge"], "challenges_to_authority": counts["challenge"],
        "unresolved_conflicts": open_disputes, "escalations": escalations,
        "communication_count": counts["communicate"],
        "communication_characters": sum(len(e["action"]["message"]) for e in successful
                                          if e["action"]["kind"] == "communicate"),
        "private_messages": sum(e["action"]["private"] for e in successful),
        "rubric_version": RUBRIC_VERSION,
    }


def action_frame(world: World) -> pd.DataFrame:
    rows = []
    classifications = classification_map(world)
    for e in world.history:
        if e.get("action"):
            a = e["action"]
            rows.append({k: e[k] for k in ("id", "day", "round", "actor", "group", "success", "effect")}
                        | {"action": a["kind"], "target": a["target"], "message": a["message"],
                           "private": a["private"], "restrictive": int(is_restrictive(e, world)),
                           "cooperative": int(e["success"] and a["kind"] in COOPERATIVE),
                           "resistant": int(e["success"] and a["kind"] in RESISTANT),
                           "classifications": ", ".join(classifications[e["id"]])})
    return pd.DataFrame(rows, columns=["id", "day", "round", "actor", "group", "success", "effect",
                                      "action", "target", "message", "private", "restrictive",
                                      "cooperative", "resistant", "classifications"])


def compare_runs(store, run_ids: list[str]) -> pd.DataFrame:
    from stanford_ai.storage.sqlite import digest
    rows = []
    for run_id in run_ids:
        run = store.run(run_id)
        metrics = store.metrics(run_id)
        if not metrics:
            continue
        cfg, final = run["config"], metrics[-1]
        controls = {k: v for k, v in cfg.items() if k not in {"condition", "seed"}}
        attempts = final["attempts"]
        rows.append({"run_id": run_id, "condition": cfg["condition"], "backend": cfg["backend"],
                     "model": cfg["model"], "seed": cfg["seed"], "status": run["status"],
                     "labels": "explicit" if cfg["condition"] in ("A", "C") else "neutral",
                     "authority": "unequal" if cfg["condition"] in ("A", "B") else "equal",
                     "days": cfg["days"], "n_agents": cfg["n_agents"],
                     "control_fingerprint": digest(controls),
                     "rounds_per_day": cfg["rounds_per_day"],
                     "restrictive_per_100_turns": 100 * final["restrictive_decisions"] / max(1, attempts),
                     "cooperation_per_100_turns": 100 * final["cooperation"] / max(1, attempts),
                     "conflict_per_100_turns": 100 * final["conflict"] / max(1, attempts),
                     "attempts": attempts, "resource_gini": final["resource_gini"]})
    return pd.DataFrame(rows)
