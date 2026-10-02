"""A reward model for tests: scores come from a JSON file ``{"<episode>": score}``.

Use it as ``--model tests.rewards.fake_scorer:make --pretrained scores.json``.
"""

import json
from pathlib import Path


class FakeScorer:
    def __init__(self, scores: dict[str, float]) -> None:
        self.name = "fake:test"
        self.scores = scores
        self.closed = False

    def score(self, root: Path, episode_index: int) -> float:
        return float(self.scores.get(str(episode_index), 0.5))

    def close(self) -> None:
        self.closed = True


def make(*, camera: str | None, device: str, pretrained: str | None) -> FakeScorer:
    scores = json.loads(Path(pretrained).read_text()) if pretrained else {}
    return FakeScorer(scores)
