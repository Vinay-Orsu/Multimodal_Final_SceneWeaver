from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from driftguard.retrieval.chunk import chunk_text


def test_chunking_basic_overlap():
    text = "a" * 1000
    chunks = chunk_text(text=text, source="unit", chunk_size_chars=300, chunk_overlap_chars=50)
    assert len(chunks) >= 3
    assert chunks[0].chunk_id == "unit_0000"
    assert chunks[1].metadata["start"] < chunks[0].metadata["end"]


def test_chunking_rejects_invalid_overlap():
    try:
        chunk_text("abc", "u", chunk_size_chars=100, chunk_overlap_chars=100)
    except ValueError:
        assert True
        return
    assert False
