from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from driftguard.retrieval.clean import normalize_text  # noqa: E402
from driftguard.retrieval.chunk import chunk_text  # noqa: E402
from driftguard.retrieval.index import SimpleIndex  # noqa: E402
from driftguard.utils.io import load_lines, write_json  # noqa: E402


def main() -> None:
    raw_dir = ROOT / "data/raw"
    chunks = []
    for p in sorted(raw_dir.glob("*.txt")):
        text = normalize_text(p.read_text(encoding="utf-8"))
        chunks.extend(chunk_text(text, source=p.stem))
    idx = SimpleIndex.build(chunks)
    write_json(ROOT / "data/processed/index_meta.json", {"num_chunks": len(idx.chunks)})
    print(f"chunks={len(idx.chunks)}")


if __name__ == "__main__":
    main()
