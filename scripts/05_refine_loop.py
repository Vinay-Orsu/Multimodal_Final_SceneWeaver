from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from driftguard.pipeline import load_config, run_pipeline  # noqa: E402
from driftguard.utils.io import read_text  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "configs/default.yaml", overlays=[ROOT / "configs/models.yaml", ROOT / "configs/prompts.yaml"])
    storyline = read_text(ROOT / "data/examples/storyline.txt")
    # In this basic version, refine is integrated inside run_pipeline.
    run_dir = run_pipeline(cfg, storyline=storyline, dry_run_override=True)
    print(f"refined_run={run_dir.as_posix()}")


if __name__ == "__main__":
    main()
