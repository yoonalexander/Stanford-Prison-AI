import argparse
import json
import os

from dotenv import load_dotenv

from stanford_ai.experiments.design import load_config
from stanford_ai.experiments.export import export_run
from stanford_ai.metrics.compute import compare_runs
from stanford_ai.models import ExperimentConfig
from stanford_ai.simulation.runner import Runner
from stanford_ai.storage.sqlite import Store


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Stanford AI computational behaviour experiments")
    parser.add_argument("--db", default=os.getenv("STANFORD_AI_DB", "outputs/experiments.sqlite"))
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config")
    for name, typ in [("condition", str), ("backend", str), ("model", str), ("n-agents", int),
                      ("days", int), ("rounds-per-day", int), ("seed", int), ("max-requests", int),
                      ("max-tokens", int), ("budget-usd", float), ("input-usd-per-million", float),
                      ("output-usd-per-million", float), ("max-output-tokens", int)]:
        run.add_argument(f"--{name}", type=typ)
    run.add_argument("--confirm-pricing", action="store_true", default=None)
    run.add_argument("--repeats", type=int, default=1, help="Each repeat has its OWN budget; increments seed")
    export = sub.add_parser("export")
    export.add_argument("run_id")
    export.add_argument("--out", default="outputs/exports")
    replay = sub.add_parser("replay")
    replay.add_argument("run_id")
    replay.add_argument("--seq", type=int)
    sub.add_parser("list")
    compare = sub.add_parser("compare")
    compare.add_argument("run_ids", nargs="+")
    recover = sub.add_parser("recover")
    recover.add_argument("run_id")
    args = parser.parse_args()
    with Store(args.db) as store:
        if args.command == "run":
            if not 1 <= args.repeats <= 100:
                parser.error("--repeats must be 1..100; per-run budgets multiply")
            overrides = {k: v for k, v in vars(args).items() if k in ExperimentConfig.model_fields
                         and v is not None}
            if args.confirm_pricing:
                overrides["pricing_confirmed"] = True
            cfg = load_config(args.config, **overrides) if args.config else ExperimentConfig(**overrides)
            for repeat in range(args.repeats):
                config = ExperimentConfig.model_validate(cfg.model_dump() | {"seed": cfg.seed + repeat})
                run_id = Runner(config, store).run()
                result = store.run(run_id)
                print(json.dumps({"id": run_id, "status": result["status"], "reason": result["reason"],
                                  "usage": result["usage"]}, indent=2))
                if result["status"] != "completed":
                    raise SystemExit(2)
        elif args.command == "export":
            print(export_run(store, args.run_id, args.out))
        elif args.command == "replay":
            print(store.replay(args.run_id, args.seq).model_dump_json(indent=2))
        elif args.command == "list":
            for r in store.runs():
                print(r["id"], r["status"], r["config"]["condition"], r["config"]["backend"])
        elif args.command == "compare":
            print(compare_runs(store, args.run_ids).to_csv(index=False))
        elif args.command == "recover":
            store.recover(args.run_id)
            print("Marked interrupted. Reserved usage retained; no API calls made.")


if __name__ == "__main__":
    main()
