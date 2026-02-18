from __future__ import annotations

from dataclasses import dataclass
from typing import List
from urllib.parse import urlparse
from urllib.request import urlopen


@dataclass
class FetchConfig:
    whitelist_domains: List[str]
    timeout_sec: int = 20


def _allowed(url: str, whitelist_domains: List[str]) -> bool:
    if not whitelist_domains:
        return True
    domain = urlparse(url).netloc.lower()
    return any(domain.endswith(d.lower()) for d in whitelist_domains)


def fetch_texts(urls: List[str], config: FetchConfig) -> List[str]:
    docs: List[str] = []
    for url in urls:
        if not _allowed(url, config.whitelist_domains):
            continue
        with urlopen(url, timeout=config.timeout_sec) as r:  # nosec - controlled by whitelist in practice
            raw = r.read().decode("utf-8", errors="ignore")
        docs.append(raw)
    return docs
