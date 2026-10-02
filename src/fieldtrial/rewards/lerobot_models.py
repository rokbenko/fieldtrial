"""LeRobot reward models as episode scorers (the ``rewards`` extra).

Written against LeRobot 0.6.1 (``lerobot.rewards``): the same configs, models and encoder
steps that LeRobot's own ``compute_rabc_weights.py`` scripts use. LeRobot, torch and
transformers are imported only when a scorer is created. These classes are not exercised
in fieldtrial's test suite, which has no GPU and no model weights; the scoring service is
tested with a fake scorer instead.
"""

import math
from pathlib import Path
from typing import Any

from fieldtrial.rewards import ScorerError


def _lerobot() -> dict[str, Any]:  # pragma: no cover - needs the rewards extra
    try:
        import torch
        from lerobot.datasets import LeRobotDataset
        from lerobot.lerobot_types import TransitionKey
    except ImportError as exc:
        raise ScorerError(
            "reward models need the rewards extra (Python 3.12+): pip install 'fieldtrial[rewards]'"
        ) from exc
    return {"torch": torch, "LeRobotDataset": LeRobotDataset, "TransitionKey": TransitionKey}


class _Base:  # pragma: no cover - needs the rewards extra
    """Shared dataset handling: evenly spaced frames of one camera, and the task string."""

    name = ""

    def __init__(self, camera: str | None, device: str, frames: int) -> None:
        self._lib = _lerobot()
        self._camera = camera
        self._device = device
        self._frames = frames
        self._datasets: dict[Path, Any] = {}

    def _dataset(self, root: Path) -> Any:
        if root not in self._datasets:
            self._datasets[root] = self._lib["LeRobotDataset"](
                repo_id=f"local/{root.name}", root=root, download_videos=False
            )
        return self._datasets[root]

    def _episode(self, root: Path, episode_index: int, key: str) -> tuple[Any, str]:
        torch = self._lib["torch"]
        ds = self._dataset(root)
        ep = ds.meta.episodes[episode_index]
        start, end = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        count = end - start
        if count <= 0:
            raise ScorerError(f"episode {episode_index} has no frames")
        k = min(self._frames, count)
        picks = sorted({round(i * (count - 1) / max(k - 1, 1)) for i in range(k)})
        first = ds[start]
        if key not in first:
            cameras = [name for name in first if name.startswith("observation.images.")]
            raise ScorerError(f"the dataset has no camera {key!r}; it has {cameras}")
        frames = torch.stack([ds[start + i][key] for i in picks])
        task = first.get("task")
        return frames, task if isinstance(task, str) and task else "perform the task"

    def _reward(self, model: Any, encoder: Any, frames: Any, task: str, key: str) -> float:
        torch = self._lib["torch"]
        tk = self._lib["TransitionKey"]
        encoded = encoder(
            {
                tk.OBSERVATION: {key: frames.unsqueeze(0)},  # (1, T, C, H, W)
                tk.COMPLEMENTARY_DATA: {"task": task},
            }
        )
        batch = {
            k: v.to(self._device) if isinstance(v, torch.Tensor) else v
            for k, v in encoded[tk.OBSERVATION].items()
        }
        with torch.no_grad():
            reward = model.compute_reward(batch)
        return float(reward.reshape(-1)[-1].item())

    def close(self) -> None:
        self._datasets.clear()
        self._model = None
        torch = self._lib["torch"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class RobometerScorer(_Base):  # pragma: no cover - needs the rewards extra
    """Robometer's progress prediction at the last of up to 8 evenly spaced frames."""

    def __init__(
        self, *, camera: str | None, device: str = "cuda", pretrained: str | None = None
    ) -> None:
        super().__init__(camera, device, frames=8)
        from lerobot.rewards.robometer.configuration_robometer import RobometerConfig
        from lerobot.rewards.robometer.modeling_robometer import RobometerRewardModel
        from lerobot.rewards.robometer.processor_robometer import RobometerEncoderProcessorStep

        path = pretrained or "lerobot/Robometer-4B"
        config = RobometerConfig(pretrained_path=path, device=device, reward_output="progress")
        if camera:
            config.image_key = camera
        self._key = config.image_key
        self._model = RobometerRewardModel.from_pretrained(path, config=config).to(device).eval()
        self._encoder = RobometerEncoderProcessorStep(
            base_model_id=config.base_model_id,
            image_key=config.image_key,
            task_key=config.task_key,
            default_task=config.default_task,
            max_frames=self._frames,
            use_multi_image=config.use_multi_image,
            use_per_frame_progress_token=config.use_per_frame_progress_token,
        )
        self.name = f"robometer:{path}"

    def score(self, root: Path, episode_index: int) -> float:
        """Predicted progress at the episode's last frame, in [0, 1]."""
        frames, task = self._episode(root, episode_index, self._key)
        progress = self._reward(self._model, self._encoder, frames, task, self._key)
        return min(max(progress, 0.0), 1.0)


class TOPRewardScorer(_Base):  # pragma: no cover - needs the rewards extra
    """TOPReward's probability that the instruction was completed, over up to 16 frames."""

    def __init__(
        self, *, camera: str | None, device: str = "cuda", pretrained: str | None = None
    ) -> None:
        super().__init__(camera, device, frames=16)
        from lerobot.rewards.topreward.configuration_topreward import TOPRewardConfig
        from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel
        from lerobot.rewards.topreward.processor_topreward import TOPRewardEncoderProcessorStep

        if pretrained:
            model = TOPRewardModel.from_pretrained(pretrained)
            config = model.config
            config.device = device
        else:
            config = TOPRewardConfig(device=device)
            model = TOPRewardModel(config)
        if camera:
            config.image_key = camera
        self._key = config.image_key
        self._model = model.to(device).eval()
        self._encoder = TOPRewardEncoderProcessorStep(
            vlm_name=config.vlm_name,
            image_key=config.image_key,
            task_key=config.task_key,
            default_task=config.default_task,
            max_frames=self._frames,
            fps=config.fps,
            prompt_prefix=config.prompt_prefix,
            prompt_suffix_template=config.prompt_suffix_template,
            add_chat_template=config.add_chat_template,
            max_length=config.max_input_length,
        )
        self.name = f"topreward:{pretrained or config.vlm_name}"

    def score(self, root: Path, episode_index: int) -> float:
        """Probability that the instruction was completed, ``exp(log P("True"))``."""
        frames, task = self._episode(root, episode_index, self._key)
        log_prob = self._reward(self._model, self._encoder, frames, task, self._key)
        return min(max(math.exp(min(log_prob, 0.0)), 0.0), 1.0)
