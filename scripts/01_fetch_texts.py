from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from driftguard.pipeline import load_config  # noqa: E402
from driftguard.retrieval.fetch import FetchConfig, fetch_texts  # noqa: E402
from driftguard.utils.io import write_text  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "configs/default.yaml", overlays=[ROOT / "configs/models.yaml", ROOT / "configs/prompts.yaml"])
    r = cfg.get("retrieval", {})
    urls = r.get("urls", [])
    docs = fetch_texts(
        urls=urls,
        config=FetchConfig(whitelist_domains=r.get("whitelist_domains", []), timeout_sec=20),
    )
    out_dir = ROOT / "data/raw"
    for i, doc in enumerate(docs):
        write_text(out_dir / f"doc_{i:03d}.txt", doc)
    print(f"fetched_docs={len(docs)}")


if __name__ == "__main__":
    main()
