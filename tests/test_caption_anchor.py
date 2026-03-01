import numpy as np

from memory_module.captioner import Captioner, CaptionerConfig


def test_caption_stub_produces_anchor():
    # Simple 3-frame dummy clip
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    cap = Captioner(CaptionerConfig(model_id="__stub__", stub_fallback=True))
    cap.load()
    captions, summary, dupes = cap.caption_frames(frames)

    assert len(captions) == 3
    assert isinstance(summary, str) and len(summary) > 0
    assert dupes is False
