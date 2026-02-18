from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from driftguard.pipeline import load_config  # noqa: E402
from driftguard.planning.storyboard import make_storyboard  # noqa: E402
from driftguard.utils.io import read_text, write_json  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "configs/default.yaml", overlays=[ROOT / "configs/models.yaml", ROOT / "configs/prompts.yaml"])
    storyline = read_text(ROOT / "data/examples/storyline.txt")
    story_cfg = cfg["story"]
    windows = make_storyboard(
        storyline=storyline,
        total_minutes=float(story_cfg.get("total_minutes", 0.5)),
        window_seconds=int(story_cfg.get("window_seconds", 10)),
    )
    write_json(
        ROOT / "data/processed/storyboard.json",
        {"windows": [w.__dict__ for w in windows], "num_windows": len(windows)},
    )
    print(f"windows={len(windows)}")


if __name__ == "__main__":
    main()
