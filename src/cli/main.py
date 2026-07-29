"""geoverse CLI entrypoint: python -m src.cli.main <command> ...

Thin wrapper - all real logic lives in src/catalog, src/processing, src/ai.
"""

import argparse
import json
import sys


def cmd_ingest(args: argparse.Namespace) -> None:
    from src.catalog.scanner import mark_processed, scan_raw_datasets

    records = mark_processed(scan_raw_datasets())
    for r in records:
        print(f"{r.id:45s} {r.source.value:12s} {r.status.value:10s} {r.acquisition_date or ''}")


def cmd_preprocess(args: argparse.Namespace) -> None:
    print(
        f"preprocess is a placeholder - wire src/processing/coregistration.py "
        f"for AOI={args.aoi!r} once a valid overlapping AOI is confirmed",
        file=sys.stderr,
    )


def cmd_train(args: argparse.Namespace) -> None:
    print(
        f"train is a placeholder - wire src/ai/{args.model}/train.py "
        f"once datasets/exports/{args.aoi} exists",
        file=sys.stderr,
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    from src.ai.objectives.registry import get_objective

    objective = get_objective(args.objective)
    print(json.dumps({"objective": objective.name, "metrics": objective.metrics}, indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    print("predict is a placeholder - wire src/ai/classic/infer.py or hybrid_unet inference", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="geoverse")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="scan datasets/raw and print the catalog")
    p_ingest.set_defaults(func=cmd_ingest)

    p_preprocess = sub.add_parser("preprocess", help="co-register + tile an AOI")
    p_preprocess.add_argument("--aoi", required=True)
    p_preprocess.set_defaults(func=cmd_preprocess)

    p_train = sub.add_parser("train", help="train a segmentation model")
    p_train.add_argument("--aoi", required=True)
    p_train.add_argument("--model", choices=["classic", "quantum"], default="classic")
    p_train.set_defaults(func=cmd_train)

    p_evaluate = sub.add_parser("evaluate", help="show metrics for an objective")
    p_evaluate.add_argument("--objective", default="flood-segmentation")
    p_evaluate.set_defaults(func=cmd_evaluate)

    p_predict = sub.add_parser("predict", help="run inference on an AOI/date")
    p_predict.add_argument("--aoi", required=True)
    p_predict.add_argument("--date", required=False)
    p_predict.set_defaults(func=cmd_predict)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
