from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from driftguard.pipeline import load_config, run_pipeline  # noqa: E402
from driftguard.utils.io import read_text  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate video windows using driftguard pipeline.")
    parser.add_argument("--dry_run", action="store_true", help="Force dry-run output (prompt text artifacts).")
    parser.add_argument("--real_run", action="store_true", help="Force real generation (requires generation.enabled and model runtime).")
    args = parser.parse_args()

    if args.dry_run and args.real_run:
        raise ValueError("Use only one of --dry_run or --real_run.")

    cfg = load_config(ROOT / "configs/default.yaml", overlays=[ROOT / "configs/models.yaml", ROOT / "configs/prompts.yaml"])
    storyline = read_text(ROOT / "data/examples/storyline.txt")

    override = None
    if args.dry_run:
        override = True
    elif args.real_run:
        override = False

    run_dir = run_pipeline(cfg, storyline=storyline, dry_run_override=override)
    print(run_dir.as_posix())


if __name__ == "__main__":
    main()
