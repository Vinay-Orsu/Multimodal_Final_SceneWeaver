from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from driftguard.pipeline import load_config, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="driftguard video pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run end-to-end pipeline")
    run.add_argument("--config", type=str, default="configs/default.yaml")
    run.add_argument("--models_config", type=str, default="configs/models.yaml")
    run.add_argument("--prompts_config", type=str, default="configs/prompts.yaml")
    run.add_argument("--storyline", type=str, default="")
    run.add_argument("--storyline_file", type=str, default="")
    run.add_argument("--dry_run", action="store_true")
    return p


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        overlays = [Path(args.models_config), Path(args.prompts_config)]
        cfg = load_config(Path(args.config), overlays=overlays)
        run_dir = run_pipeline(
            cfg,
            storyline=args.storyline or None,
            storyline_file=Path(args.storyline_file) if args.storyline_file else None,
            dry_run_override=True if args.dry_run else None,
        )
        print(run_dir.as_posix())


if __name__ == "__main__":
    main()
