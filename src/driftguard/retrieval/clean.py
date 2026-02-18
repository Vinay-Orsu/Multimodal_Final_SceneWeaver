from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text
