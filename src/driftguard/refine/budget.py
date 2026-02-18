from __future__ import annotations


class RegenBudget:
    def __init__(self, max_regens_per_window: int):
        self.max_regens_per_window = max(0, max_regens_per_window)

    def attempts(self) -> range:
        return range(self.max_regens_per_window + 1)
