from __future__ import annotations

import random


def seed_for_window(base_seed: int, window_index: int, regen_try: int = 0) -> int:
    rng = random.Random(base_seed + (window_index * 1009) + regen_try)
    return rng.randint(1, 2**31 - 1)
