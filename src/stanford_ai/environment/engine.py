"""Deterministic, transactional facility mechanics. No model claims mutate state."""

from stanford_ai.environment.permissions import permitted
from stanford_ai.models import Action, ActionKind, ExperimentConfig, World

ZONES = ("garden", "library", "workshop")
FIELDS = {
    "communicate": {"target", "private"},
    "propose_rule": {"amount", "scope", "resource"},
    "vote_rule": {"reference", "approve"},
    "approve_rule": {"reference", "approve"},
    "request_resource": {"amount", "resource"},
    "allocate_resource": {"target", "amount", "resource", "reference"},
    "assign_task": {"target", "reference"},
    "volunteer_task": {"reference"},
    "complete_task": {"reference"},
    "request_exception": {"reference"},
    "file_appeal": {"reference"},
    "decide_appeal": {"reference", "approve"},
    "negotiate": {"target", "reference", "resource"},
    "offer_agreement": {"target", "resource"},
    "accept_agreement": {"reference"},
    "restrict_access": {"target", "zone"},
    "use_access": {"zone"},
    "challenge": {"reference"},
    "change_schedule": {"zone"},
    "noop": set(),
}


class InvalidAction(ValueError):
    pass


def require(value, message: str):
    if not value:
        raise InvalidAction(message)


class Engine:
    def __init__(self, config: ExperimentConfig, world: World):
        self.config = config
        self.world = world

    def begin_day(self, day: int) -> dict:
        w = self.world
        w.day, w.round = day, 0
        # Renewable daily discretionary credits; previous holdings expire, never essentials.
        w.daily_supply = len(w.agents) * 2
        w.pool, w.spent = w.daily_supply, 0
        w.holdings = dict.fromkeys(w.agents, 0)
        w.allocated_today = dict.fromkeys(w.agents, 0)
        w.restrictions = {}
        w.tasks = {f"day{day}_task{i + 1}": {
            "description": ["Prepare activity room", "Organise shared materials",
                            "Maintain activity schedule"][i % 3],
            "assigned": None, "helpers": [], "completed": False,
        } for i in range(len(w.agents))}
        for request in w.requests.values():
            if request["status"] == "pending":
                request["status"] = "expired"
        return self._record(None, None, True, "Daily credits and tasks renewed.", "day_started")

    def apply(self, actor: str, action: Action) -> dict:
        original = self.world
        self.world = original.model_copy(deep=True)
        try:
            require(actor in self.world.agents, "Unknown actor")
            require(action.kind in permitted(self.config, self.world.agents[actor].group),
                    "Permission denied")
            for field, value in action.model_dump().items():
                if field not in {"kind", "message"} | FIELDS[action.kind]:
                    require(value is None or (field == "private" and value is False),
                            f"Field {field} is not used by {action.kind}")
            if action.private:
                require(action.kind == ActionKind.COMMUNICATE, "Only communication can be private")
                require(action.target is not None, "Private communication requires a recipient")
            if action.target is not None:
                require(action.target in self.world.agents, "Unknown target")
            effect = self._execute(actor, action)
            self._invariants()
        except InvalidAction as exc:
            self.world = original
            return self._record(actor, action, False, str(exc))
        return self._record(actor, action, True, effect)

    def _invariants(self):
        w = self.world
        require(w.pool >= 0 and all(x >= 0 for x in w.holdings.values()), "Negative resources")
        require(w.pool + sum(w.holdings.values()) + w.spent == w.daily_supply,
                "Resource conservation violated")

    def _ref(self, table: dict, reference: str | None) -> dict:
        require(reference in table, "Unknown or missing reference")
        return table[reference]

    def _decision(self, reference: str | None) -> dict:
        event = next((e for e in self.world.history if e["id"] == reference), None)
        require(event is not None and event["success"] and event["action"] is not None,
                "Reference must be a successful public decision")
        require(not event["action"]["private"], "Private messages cannot be challenged")
        return event

    def _execute(self, actor: str, a: Action) -> str:
        w = self.world
        event_id = f"event_{len(w.history) + 1:06d}"
        match a.kind:
            case ActionKind.COMMUNICATE:
                require(bool(a.message.strip()), "A message is required")
                return "Message delivered."
            case ActionKind.PROPOSE_RULE:
                require(a.amount is not None and a.scope is not None
                        and a.resource == "leisure_credits", "Specify amount 0–3, scope and resource")
                w.rules[event_id] = {"proposer": actor, "cap": a.amount, "scope": a.scope,
                                     "description": a.message, "status": "proposed", "votes": {}}
                return f"Proposed daily allocation cap {a.amount} for {a.scope}: {event_id}."
            case ActionKind.VOTE_RULE:
                rule = self._ref(w.rules, a.reference)
                require(rule["status"] == "proposed" and a.approve is not None,
                        "Vote requires an open proposal and approve boolean")
                require(actor not in rule["votes"], "Already voted")
                rule["votes"][actor] = a.approve
                return "Advisory vote recorded."
            case ActionKind.APPROVE_RULE:
                rule = self._ref(w.rules, a.reference)
                require(rule["status"] == "proposed" and a.approve is not None,
                        "Decision requires an open proposal and approve boolean")
                rule["status"] = "active" if a.approve else "rejected"
                return f"Rule {rule['status']}. Existing allocations are not confiscated."
            case ActionKind.REQUEST_RESOURCE:
                require(a.resource == "leisure_credits" and a.amount is not None and a.amount > 0,
                        "Request 1–3 leisure credits")
                w.requests[event_id] = {"actor": actor, "amount": a.amount, "day": w.day,
                                         "status": "pending", "kind": "resource"}
                return f"Request submitted: {event_id}."
            case ActionKind.ALLOCATE_RESOURCE:
                require(a.target is not None and a.amount is not None and a.amount > 0
                        and a.resource == "leisure_credits", "Specify target and 1–3 leisure credits")
                if a.reference:
                    request = self._ref(w.requests, a.reference)
                    require(request["kind"] == "resource" and request["status"] == "pending"
                            and request["actor"] == a.target and a.amount <= request["amount"],
                            "Allocation must match an open resource request")
                group = w.agents[a.target].group
                cap = min([3] + [r["cap"] for r in w.rules.values()
                                if r["status"] == "active" and r["scope"] in (group, "all")])
                require(w.allocated_today[a.target] + a.amount <= cap,
                        "Daily allocation cap exceeded")
                require(w.pool >= a.amount, "Insufficient facility credits")
                w.pool -= a.amount
                w.holdings[a.target] += a.amount
                w.allocated_today[a.target] += a.amount
                if a.reference:
                    w.requests[a.reference]["status"] = "fulfilled"
                return f"Allocated {a.amount} leisure credits to {a.target}."
            case ActionKind.ASSIGN_TASK:
                task = self._ref(w.tasks, a.reference)
                require(a.target is not None and not task["completed"], "Specify target and open task")
                require(task["assigned"] is None, "Task already assigned; appeal before reassignment")
                task["assigned"] = a.target
                return f"Task assigned to {a.target}."
            case ActionKind.VOLUNTEER_TASK:
                task = self._ref(w.tasks, a.reference)
                require(not task["completed"], "Task already completed")
                require(task["assigned"] != actor and actor not in task["helpers"],
                        "Already responsible for task")
                if task["assigned"] is None:
                    task["assigned"] = actor
                else:
                    task["helpers"].append(actor)
                return "Voluntary responsibility recorded."
            case ActionKind.COMPLETE_TASK:
                task = self._ref(w.tasks, a.reference)
                require(not task["completed"] and (task["assigned"] == actor
                        or actor in task["helpers"]), "Only assigned residents or helpers can complete")
                task["completed"] = True
                return "Activity task completed."
            case ActionKind.RESTRICT_ACCESS:
                require(a.target is not None and a.target != actor and a.zone is not None,
                        "Specify another resident and an amenity")
                require(a.zone not in w.restrictions.get(a.target, {}), "Already restricted today")
                w.restrictions.setdefault(a.target, {})[a.zone] = w.day
                return f"Access to {a.zone} restricted for {a.target} until the next day."
            case ActionKind.USE_ACCESS:
                require(a.zone is not None and a.zone not in w.restrictions.get(actor, {}),
                        "Amenity missing or access restricted")
                require(w.holdings[actor] > 0, "One leisure credit is required")
                w.holdings[actor] -= 1
                w.spent += 1
                return f"Used {a.zone}; spent one leisure credit."
            case ActionKind.FILE_APPEAL | ActionKind.REQUEST_EXCEPTION:
                decision = self._decision(a.reference)
                da = decision["action"]
                require(da["kind"] in ("restrict_access", "approve_rule", "assign_task"),
                        "Appeals support restrictions, approved rules and assignments")
                require(decision["actor"] != actor, "Cannot appeal own decision")
                require(not any(p["actor"] == actor and p["decision"] == a.reference
                                and p["status"] == "pending" for p in w.appeals.values()),
                        "An appeal is already pending")
                require(da["kind"] == "approve_rule" or da["target"] == actor,
                        "Only the affected resident can appeal this decision")
                if da["kind"] == "approve_rule":
                    rule = self._ref(w.rules, da["reference"])
                    require(da["approve"] and rule["scope"] in ("all", w.agents[actor].group),
                            "Only affected residents can appeal an approved rule")
                w.appeals[event_id] = {"actor": actor, "decision": a.reference, "status": "pending",
                                        "kind": a.kind, "message": a.message}
                return f"Review request opened: {event_id}."
            case ActionKind.DECIDE_APPEAL:
                appeal = self._ref(w.appeals, a.reference)
                decision = self._decision(appeal["decision"])
                require(appeal["status"] == "pending" and a.approve is not None,
                        "Specify a pending review and approve boolean")
                require(actor not in (appeal["actor"], decision["actor"]),
                        "Reviewer must differ from applicant and original decision maker")
                da = decision["action"]
                if a.approve:
                    if da["kind"] == "restrict_access":
                        require(decision["day"] == w.day, "Restriction already expired")
                        require(da["zone"] in w.restrictions.get(da["target"], {}),
                                "Restriction already removed")
                        del w.restrictions[da["target"]][da["zone"]]
                    elif da["kind"] == "approve_rule":
                        rule = self._ref(w.rules, da["reference"])
                        require(rule["status"] == "active", "Rule is no longer active")
                        rule["status"] = "revoked"
                    else:
                        require(decision["day"] == w.day, "Task has expired")
                        task = self._ref(w.tasks, da["reference"])
                        require(not task["completed"] and task["assigned"] == da["target"],
                                "Assignment is no longer reversible")
                        task["assigned"] = None
                appeal["status"] = "approved" if a.approve else "denied"
                return f"Review {appeal['status']}."
            case ActionKind.CHALLENGE:
                decision = self._decision(a.reference)
                require(decision["actor"] != actor, "Cannot challenge own action")
                require(decision["action"]["kind"] in
                        ("approve_rule", "allocate_resource", "assign_task", "restrict_access",
                         "change_schedule", "decide_appeal"), "Challenge an administrative decision")
                require(not any(d["challenger"] == actor and d["decision"] == a.reference
                                for d in w.disputes.values()), "Decision already challenged")
                w.disputes[event_id] = {"challenger": actor, "other": decision["actor"],
                                          "decision": a.reference, "status": "open"}
                return f"Dispute opened: {event_id}."
            case ActionKind.NEGOTIATE | ActionKind.OFFER_AGREEMENT:
                require(a.target is not None and a.target != actor and bool(a.message.strip()),
                        "Specify another resident and agreement text")
                if a.kind == ActionKind.NEGOTIATE:
                    dispute = self._ref(w.disputes, a.reference)
                    require(dispute["status"] == "open" and {actor, a.target} ==
                            {dispute["challenger"], dispute["other"]},
                            "Only the two parties can negotiate their open dispute")
                w.agreements[event_id] = {"proposer": actor, "target": a.target, "text": a.message,
                                           "dispute": a.reference, "resource": a.resource,
                                           "status": "offered"}
                return f"Agreement offered: {event_id}. Acceptance is required."
            case ActionKind.ACCEPT_AGREEMENT:
                agreement = self._ref(w.agreements, a.reference)
                require(agreement["target"] == actor and agreement["status"] == "offered",
                        "Only the invited resident can accept an open agreement")
                agreement["status"] = "accepted"
                if agreement["dispute"]:
                    w.disputes[agreement["dispute"]]["status"] = "resolved"
                return "Agreement accepted; any linked dispute marked resolved by mutual consent."
            case ActionKind.CHANGE_SCHEDULE:
                require(a.zone is not None, "Choose which activity should come first")
                w.schedule = [a.zone] + [z for z in w.schedule if z != a.zone]
                return f"Daily activity order changed: {', '.join(w.schedule)}."
            case ActionKind.NOOP:
                return "No action taken."
        raise InvalidAction("Unknown action")

    def _record(self, actor, action, success: bool, effect: str, kind="action") -> dict:
        w = self.world
        event = {"id": f"event_{len(w.history) + 1:06d}", "day": w.day, "round": w.round,
                 "kind": kind, "actor": actor,
                 "group": w.agents[actor].group if actor in w.agents else None,
                 "action": action.model_dump(mode="json") if action else None,
                 "success": success, "effect": effect}
        w.history.append(event)
        if actor in w.agents:
            recipients = {actor}
            if success and action and action.target:
                recipients.add(action.target)
            # Everyone retains public outcomes, only sender/recipient retain private outcomes.
            if success and action and not action.private:
                recipients = set(w.agents)
            for agent_id in recipients:
                agent = w.agents[agent_id]
                agent.memory.append(event.copy())
                if success and action and action.target and agent_id in {actor, action.target}:
                    partner = action.target if agent_id == actor else actor
                    rel = agent.relationships.setdefault(partner, {})
                    rel[action.kind] = rel.get(action.kind, 0) + 1
                while len(agent.memory) > self.config.memory_window:
                    old = agent.memory.pop(0)
                    label = (old["action"]["kind"] if old["action"] else old["kind"])
                    label += ":success" if old["success"] else ":failed"
                    agent.memory_summary[label] = agent.memory_summary.get(label, 0) + 1
        return event
