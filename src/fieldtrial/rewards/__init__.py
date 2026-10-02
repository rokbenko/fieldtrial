"""Reward models that score recorded episodes (docs/guides/reward-models.md).

A scorer turns one episode of a LeRobot v3.0 dataset into a score in [0, 1]. Scores are
suggestions only: a person's label is the label. The built-in scorers use LeRobot's reward
models and need the ``rewards`` extra (``pip install 'fieldtrial[rewards]'``, Python 3.12+):

- ``robometer``: Robometer's predicted progress at the episode's last frame (0 to 1).
- ``topreward``: TOPReward's probability that the instruction was completed,
  ``exp(log P("True"))``.

Any other scorer can be named as ``package.module:factory``; ``factory(camera=..., device=...,
pretrained=...)`` must return an object with a ``name``, ``score(root, episode_index)`` and
``close()``.
"""

import importlib
from pathlib import Path
from typing import Protocol, runtime_checkable

BUILTIN = ("robometer", "topreward")


class ScorerError(RuntimeError):
    """A scorer cannot be created or cannot score an episode."""


@runtime_checkable
class EpisodeScorer(Protocol):
    """Scores episodes of a LeRobot dataset."""

    name: str
    """Recorded with every score, for example ``robometer:lerobot/Robometer-4B``."""

    def score(self, root: Path, episode_index: int) -> float:
        """A score in [0, 1] for one episode of the dataset at ``root``."""
        ...

    def close(self) -> None:
        """Release the model."""
        ...


def make_scorer(
    model: str,
    *,
    camera: str | None = None,
    device: str = "cuda",
    pretrained: str | None = None,
) -> EpisodeScorer:
    """A scorer by name (``robometer``, ``topreward``) or ``package.module:factory``."""
    if model in BUILTIN:
        from fieldtrial.rewards import lerobot_models  # imports LeRobot only when created

        cls = (
            lerobot_models.RobometerScorer
            if model == "robometer"
            else lerobot_models.TOPRewardScorer
        )
        return cls(camera=camera, device=device, pretrained=pretrained)
    if ":" not in model:
        raise ScorerError(
            f"unknown reward model {model!r}; use one of {', '.join(BUILTIN)} or "
            "package.module:factory"
        )
    module_name, _, attr = model.partition(":")
    try:
        factory = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ScorerError(f"cannot load reward model {model!r}: {exc}") from exc
    try:
        scorer = factory(camera=camera, device=device, pretrained=pretrained)
    except Exception as exc:  # a third-party factory may raise anything
        raise ScorerError(f"creating reward model {model!r} failed: {exc}") from exc
    if not isinstance(scorer, EpisodeScorer):
        raise ScorerError(f"{model} did not return a scorer (name, score, close)")
    return scorer
