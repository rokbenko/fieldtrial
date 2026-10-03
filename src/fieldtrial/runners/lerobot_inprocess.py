"""The ``lerobot`` runner: LeRobot policies run inside fieldtrial (docs/guides/lerobot.md).

The robot is connected once, when the first trial is prepared, and stays connected while
the server runs. Each arm's policy is loaded with LeRobot (``policy.path``, optionally
``policy.revision``) and driven by LeRobot's own inference engines (``serving.inference``:
``sync`` or ``rtc``), so switching arms never needs the operator. Every trial is recorded as
one episode of a LeRobot v3.0 dataset, with the study's instruction as its task, so the
dataset says nothing about the arm. The episode index and trial id are written to
``fieldtrial_episodes.json`` in the dataset folder, and the trial is linked to its episode
automatically.

Arms that load the same checkpoint share its weights. ``keep_loaded`` sets how many
checkpoints stay in memory. The dataset is finalized after every episode (so it is valid
and readable between trials) and reopened for the next one.

LeRobot is imported only when a trial is first prepared. This module mirrors the setup
of LeRobot 0.6.1's ``build_rollout_context`` (robot, features, dataset, policy,
processors, inference engine) and reuses ``RolloutStrategy``'s helpers for the control
loop; it refuses other LeRobot versions.
"""

import gc
import json
import logging
import re
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fieldtrial.design.runner_config import LeRobotRunnerConfig
from fieldtrial.runners.base import (
    ArmSpec,
    EpisodeRef,
    RunArtifacts,
    RunnerCapabilities,
    RunnerError,
    RunnerStatus,
    Termination,
    TrialContext,
)

log = logging.getLogger("fieldtrial.runners.lerobot")

SIDECAR = "fieldtrial_episodes.json"
MIN_VERSION = (0, 6, 1)
BELOW_VERSION = (0, 7)
JOIN_S = 10.0


def check_version(version: str) -> None:
    """Refuse LeRobot versions this runner was not written for."""
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    parts = tuple(int(p) for p in match.groups()) if match else None
    if parts is None or not (parts >= MIN_VERSION and parts[:2] < BELOW_VERSION):
        raise RunnerError(
            f"the lerobot runner needs LeRobot >=0.6.1,<0.7 (found {version}); "
            "install it with pip install 'fieldtrial[lerobot-runner]'"
        )


def import_lerobot() -> Any:
    """The ``lerobot`` package, after checking its version."""
    try:
        import lerobot
    except ImportError as exc:
        raise RunnerError(
            "the lerobot runner needs LeRobot (Python 3.12+): "
            "pip install 'fieldtrial[lerobot-runner]'"
        ) from exc
    check_version(str(getattr(lerobot, "__version__", "unknown")))
    return lerobot


def default_root(config: LeRobotRunnerConfig, study: str, folder: Path) -> tuple[str, Path]:
    """The dataset's repo id and folder."""
    repo_id = config.dataset.repo_id or f"local/{study}"
    name = repo_id.split("/")[1]
    root = Path(config.dataset.root) if config.dataset.root else Path("lerobot") / name
    return repo_id, (root if root.is_absolute() else folder / root).resolve()


def read_sidecar(root: Path) -> list[dict[str, Any]]:
    """The episodes fieldtrial recorded in a dataset (episode index, trial id, sequence)."""
    path = root / SIDECAR
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunnerError(f"cannot read {path}: {exc}") from exc
    return list(data.get("episodes", []))


def _write_sidecar(root: Path, study: str, episodes: list[dict[str, Any]]) -> None:
    path = root / SIDECAR
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"fieldtrial": 1, "study": study, "episodes": episodes}, indent=1) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


@dataclass
class _Runtime:
    """The runtime settings LeRobot's strategy helpers read (``ctx.runtime.cfg``)."""

    fps: float
    interpolation_multiplier: int
    use_torch_compile: bool = False
    display_data: bool = False


@dataclass
class _Checkpoint:
    """A loaded policy with its processors, shared by the arms that use it."""

    policy: Any
    preprocessor: Any
    postprocessor: Any
    ordered_action_keys: list[str]
    rtc_default: Any
    size_mb: float


@dataclass
class _Active:
    """The arm whose policy drives the robot."""

    blind_code: str
    key: tuple[str, str | None]
    strategy: Any
    context: Any


@dataclass
class _Episode:
    frames: int = 0
    overruns: int = 0
    elapsed_s: float = 0.0
    timed_out: bool = False
    error: str | None = None


class _Session:
    """The connected robot, its dataset and the loaded policies."""

    def __init__(self, config: LeRobotRunnerConfig, *, study: str, folder: Path, task: str) -> None:
        self.config = config
        self.study = study
        self.task = task
        self.lr = import_lerobot()
        import draccus
        from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
        from lerobot.processor import make_default_processors
        from lerobot.robots import RobotConfig, make_robot_from_config
        from lerobot.rollout.robot_wrapper import ThreadSafeRobot
        from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features

        self.repo_id, self.root = default_root(config, study, folder)
        try:
            robot_config = draccus.decode(RobotConfig, dict(config.robot))
        except Exception as exc:
            raise RunnerError(f"runners.lerobot.robot: {exc}") from exc
        self.processors = make_default_processors()
        teleop_p, _, robot_obs_p = self.processors
        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()
        try:
            initial = self.robot.get_observation()
            self.initial_position = {k: v for k, v in initial.items() if k.endswith(".pos")}
            self.wrapper = ThreadSafeRobot(self.robot)
            observation_hw = {
                k: v
                for k, v in self.robot.observation_features.items()
                if isinstance(v, tuple) or (v is float and k.endswith((".pos", ".vel")))
            }
            self.action_hw = {
                k: v for k, v in self.robot.action_features.items() if k.endswith((".pos", ".vel"))
            }
            video = config.dataset.video
            action_features = aggregate_pipeline_dataset_features(
                pipeline=teleop_p,
                initial_features=create_initial_features(action=self.action_hw),
                use_videos=video,
            )
            observation_features = aggregate_pipeline_dataset_features(
                pipeline=robot_obs_p,
                initial_features=create_initial_features(observation=observation_hw),
                use_videos=video,
            )
            self.features = combine_feature_dicts(action_features, observation_features)
            self.hw_features = hw_to_dataset_features(observation_hw, "observation")
            self.cameras = {
                k for k, v in self.robot.observation_features.items() if isinstance(v, tuple)
            }
            self.dataset = self._open_dataset()
            self.episodes = read_sidecar(self.root)
            self.device: str | None = config.device
        except Exception:
            self.robot.disconnect()
            raise

    # --- dataset ----------------------------------------------------------------------

    def _writer_kwargs(self) -> dict[str, Any]:
        return {
            "image_writer_threads": 4 * max(len(self.cameras), 1),
            "streaming_encoding": self.config.dataset.streaming_encoding,
        }

    def _open_dataset(self) -> Any:
        from lerobot.datasets import LeRobotDataset

        if (self.root / "meta" / "info.json").exists():
            return LeRobotDataset.resume(self.repo_id, root=self.root, **self._writer_kwargs())
        return LeRobotDataset.create(
            self.repo_id,
            self.config.fps,
            root=self.root,
            robot_type=self.robot.name,
            features=self.features,
            use_videos=self.config.dataset.video,
            **self._writer_kwargs(),
        )

    def save(self, trial: TrialContext, frames: int) -> tuple[EpisodeRef | None, float]:
        """Save the trial's episode, finalize the dataset and reopen it for the next one."""
        started = time.perf_counter()
        if frames == 0:
            self.dataset.clear_episode_buffer()
            return None, 0.0
        index = int(self.dataset.num_episodes)
        self.dataset.save_episode()
        self.dataset.finalize()
        self.episodes.append({"episode_index": index, "trial_id": trial.trial_id, "seq": trial.seq})
        _write_sidecar(self.root, self.study, self.episodes)
        info = json.loads((self.root / "meta" / "info.json").read_text(encoding="utf-8"))
        self.dataset = self._open_dataset()
        ref = EpisodeRef(
            root=str(self.root),
            codebase_version=str(info.get("codebase_version", "")),
            episode_index=index,
            length=frames,
        )
        return ref, time.perf_counter() - started

    # --- policies ---------------------------------------------------------------------

    def resolve_device(self, policy_config: Any) -> str:
        """One device for every arm: ``device``, else the first policy's, else auto."""
        if self.device is None:
            from lerobot.utils.device_utils import (
                auto_select_torch_device,
                is_torch_device_available,
            )

            device = getattr(policy_config, "device", None)
            if not device or not is_torch_device_available(device):
                device = auto_select_torch_device().type
            self.device = str(device)
        return self.device

    def load(self, path: str, revision: str | None) -> _Checkpoint:
        """Load a checkpoint and its processors (as ``build_rollout_context`` does)."""
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies import get_policy_class, make_pre_post_processors

        policy_config = PreTrainedConfig.from_pretrained(path, revision=revision)
        policy_config.pretrained_path = path
        if getattr(policy_config, "use_peft", False):
            raise RunnerError("PEFT adapters are not supported by the lerobot runner")
        device = self.resolve_device(policy_config)
        policy = get_policy_class(policy_config.type).from_pretrained(
            path, config=policy_config, revision=revision
        )
        policy = policy.to(device)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=path,
            pretrained_revision=revision,
            # What lerobot-rollout passes for a new dataset. The recording dataset's own
            # stats would differ between arms loaded before and after some episodes.
            dataset_stats={},
            preprocessor_overrides={
                "device_processor": {"device": device},
                "rename_observations_processor": {"rename_map": self.config.rename_map},
            },
        )
        self._check_cameras(policy_config)
        size = sum(p.numel() * p.element_size() for p in policy.parameters()) / 2**20
        return _Checkpoint(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            ordered_action_keys=_action_order(
                getattr(policy_config, "action_feature_names", None), list(self.action_hw)
            ),
            rtc_default=getattr(policy.config, "rtc_config", None),
            size_mb=float(size),
        )

    def _check_cameras(self, policy_config: Any) -> None:
        if self.config.rename_map:
            return
        from lerobot.configs import FeatureType

        expected = {
            k for k, v in policy_config.input_features.items() if v.type == FeatureType.VISUAL
        }
        provided = {f"observation.images.{k}" for k in self.cameras}
        if not (expected <= provided or provided <= expected):
            raise RunnerError(
                f"the policy expects cameras {sorted(expected)} but the robot provides "
                f"{sorted(provided)}; map them with runners.lerobot.rename_map"
            )

    def activate(self, checkpoint: _Checkpoint, serving: dict[str, Any]) -> tuple[Any, Any]:
        """A started strategy and rollout context for one arm's serving settings."""
        import draccus
        from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep
        from lerobot.rollout import (
            DatasetContext,
            HardwareContext,
            InferenceEngineConfig,
            PolicyContext,
            ProcessorContext,
            RolloutContext,
            RTCInferenceConfig,
            RuntimeContext,
            SyncInferenceConfig,
            create_inference_engine,
        )
        from lerobot.rollout.inference.rtc import supports_rtc_inference

        try:
            inference = draccus.decode(
                InferenceEngineConfig, dict(serving.get("inference") or {"type": "sync"})
            )
        except Exception as exc:
            raise RunnerError(f"serving.inference: {exc}") from exc
        policy = checkpoint.policy
        if isinstance(inference, RTCInferenceConfig):
            if not supports_rtc_inference(policy):
                raise RunnerError("this policy does not support RTC inference; use type: sync")
            policy.config.rtc_config = inference.rtc
        else:
            policy.config.rtc_config = checkpoint.rtc_default
        if hasattr(policy, "init_rtc_processor"):
            policy.init_rtc_processor()
        if isinstance(inference, SyncInferenceConfig) and any(
            isinstance(step, RelativeActionsProcessorStep) and step.enabled
            for step in getattr(checkpoint.preprocessor, "steps", ())
        ):
            raise RunnerError("this policy uses relative actions; use serving.inference type: rtc")
        multiplier = int(serving.get("interpolation_multiplier", 1))
        shutdown = threading.Event()
        engine = create_inference_engine(
            inference,
            policy=policy,
            preprocessor=checkpoint.preprocessor,
            postprocessor=checkpoint.postprocessor,
            robot_wrapper=self.wrapper,
            hw_features=self.hw_features,
            dataset_features=self.features,
            ordered_action_keys=checkpoint.ordered_action_keys,
            task=self.task,
            fps=self.config.fps,
            device=self.device,
            use_torch_compile=False,
            shutdown_event=shutdown,
        )
        teleop_p, robot_action_p, robot_obs_p = self.processors
        context = RolloutContext(
            runtime=RuntimeContext(
                cfg=_Runtime(fps=self.config.fps, interpolation_multiplier=multiplier),
                shutdown_event=shutdown,
            ),
            hardware=HardwareContext(
                robot_wrapper=self.wrapper, teleop=None, initial_position=self.initial_position
            ),
            policy=PolicyContext(
                policy=policy,
                preprocessor=checkpoint.preprocessor,
                postprocessor=checkpoint.postprocessor,
                inference=engine,
            ),
            processors=ProcessorContext(
                teleop_action_processor=teleop_p,
                robot_action_processor=robot_action_p,
                robot_observation_processor=robot_obs_p,
            ),
            data=DatasetContext(
                dataset=None,
                dataset_features=self.features,
                hw_features=self.hw_features,
                ordered_action_keys=checkpoint.ordered_action_keys,
            ),
        )
        strategy = _strategy_class()(None)
        strategy.setup(context)
        engine.pause()
        return strategy, context

    # --- control loop -----------------------------------------------------------------

    def episode(
        self, strategy: Any, context: Any, stop: threading.Event, timeout_s: float | None
    ) -> _Episode:
        """Run the policy until ``stop`` is set or the time limit; record every frame."""
        from lerobot.rollout.strategies.core import send_next_action
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import build_dataset_frame
        from lerobot.utils.robot_utils import precise_sleep

        out = _Episode()
        engine = context.policy.inference
        interpolator = strategy._interpolator
        engine.reset()
        interpolator.reset()
        engine.resume()
        interval = interpolator.get_control_interval(self.config.fps)
        # With interpolation the robot is commanded ``multiplier`` times per dataset frame;
        # recording every tick would put more frames than ``fps`` into each second.
        multiplier = max(int(context.runtime.cfg.interpolation_multiplier), 1)
        ticks = 0
        started = time.perf_counter()
        try:
            while not stop.is_set():
                tick = time.perf_counter()
                if timeout_s is not None and tick - started >= timeout_s:
                    out.timed_out = True
                    break
                if engine.failed or context.runtime.shutdown_event.is_set():
                    out.error = "the inference engine failed"
                    break
                obs = self.wrapper.get_observation()
                processed = strategy._process_observation_and_notify(context.processors, obs)
                action = send_next_action(processed, obs, context, interpolator)
                if action is not None and ticks % multiplier == 0:
                    frame = build_dataset_frame(self.features, processed, prefix=OBS_STR)
                    frame |= build_dataset_frame(self.features, action, prefix=ACTION)
                    self.dataset.add_frame({**frame, "task": self.task})
                    out.frames += 1
                if action is not None:
                    ticks += 1
                spare = interval - (time.perf_counter() - tick)
                if spare < 0:
                    out.overruns += 1
                precise_sleep(max(spare, 0.0))
        except Exception as exc:
            log.exception("lerobot runner: the control loop stopped")
            out.error = f"{type(exc).__name__}: {exc}"
        finally:
            engine.pause()
            out.elapsed_s = time.perf_counter() - started
        # A failure during the last tick, just before Stop, is checked only here.
        if out.error is None and engine.failed:
            out.error = "the inference engine failed"
        return out

    def reset_pose(self, strategy: Any, context: Any) -> None:
        """Move the robot back to the pose it had when it was connected."""
        strategy._return_to_initial_position(context.hardware, duration_s=self.config.reset_s)

    def close(self, active: _Active | None) -> None:
        """Stop the engine, finalize the dataset, return the robot and disconnect it."""
        try:
            self.dataset.finalize()
        finally:
            if active is not None:
                active.strategy._teardown_hardware(
                    active.context.hardware,
                    return_to_initial_position=self.config.return_to_initial_position,
                )
            elif self.robot.is_connected:
                self.robot.disconnect()


def _action_order(policy_names: list[str] | None, robot_names: list[str]) -> list[str]:
    """The order of the policy's action vector (LeRobot's ``_resolve_action_key_order``)."""
    if not policy_names:
        return robot_names
    names = list(policy_names)
    if len(names) != len(robot_names) or set(names) != set(robot_names):
        log.warning("policy action names do not match the robot's; using the robot's order")
        return robot_names
    return names


_STRATEGY: Any = None


def _strategy_class() -> Any:
    """A ``RolloutStrategy`` that runs one trial at a time (defined once LeRobot is loaded)."""
    global _STRATEGY
    if _STRATEGY is None:
        from lerobot.rollout import RolloutStrategy

        class TrialStrategy(RolloutStrategy):  # type: ignore[misc]
            """Engine set-up from LeRobot; the loop is :meth:`_Session.episode`."""

            def setup(self, ctx: Any) -> None:
                self._init_engine(ctx)

            def run(self, ctx: Any) -> None:
                raise NotImplementedError("fieldtrial runs episodes itself")

            def teardown(self, ctx: Any) -> None:
                if self._engine is not None:
                    self._engine.stop()

        _STRATEGY = TrialStrategy
    return _STRATEGY


class LeRobotRunner:
    """Runs each arm's LeRobot policy on a robot connected once (``runner: lerobot``)."""

    def __init__(
        self,
        config: LeRobotRunnerConfig,
        *,
        study: str,
        folder: Path,
        instruction: str | None = None,
    ) -> None:
        self.config = config
        self.study = study
        self.folder = Path(folder)
        self.task = instruction or ""
        self.capabilities = RunnerCapabilities(can_switch_arms=True, can_stop=True)
        self._session: _Session | None = None
        self._checkpoints: OrderedDict[tuple[str, str | None], _Checkpoint] = OrderedDict()
        self._active: _Active | None = None
        self._pending_load_s = 0.0
        self._pending_loaded = False
        self._trial: TrialContext | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._result: _Episode | None = None
        self._state = RunnerStatus("idle")
        self._error: str | None = None
        self._lock = threading.Lock()

    def _set(self, state: RunnerStatus) -> None:
        self._state = state

    # --- loading ----------------------------------------------------------------------

    def _connect(self) -> _Session:
        if self._session is None:
            self._set(RunnerStatus("loading", "connecting the robot"))
            try:
                self._session = _Session(
                    self.config, study=self.study, folder=self.folder, task=self.task
                )
            except RunnerError:
                raise
            except Exception as exc:
                log.exception("lerobot runner: connecting the robot failed")
                raise RunnerError(f"connecting the robot failed: {exc}") from exc
        return self._session

    def _checkpoint(self, session: _Session, arm: ArmSpec) -> tuple[_Checkpoint, bool]:
        """The arm's checkpoint, and whether it had to be loaded now."""
        key = (str(arm.policy["path"]), arm.policy.get("revision"))
        if key in self._checkpoints:
            self._checkpoints.move_to_end(key)
            return self._checkpoints[key], False
        keep = self.config.keep_loaded
        if keep != "all":
            while len(self._checkpoints) >= keep:
                self._checkpoints.popitem(last=False)
            gc.collect()
            _empty_cuda_cache()
        self._set(RunnerStatus("loading", f"loading arm {arm.blind_code}"))
        try:
            checkpoint = session.load(*key)
        except RunnerError:
            raise
        except Exception as exc:
            # The message may name the checkpoint, so it goes to the server log only.
            log.exception("lerobot runner: loading arm %s failed", arm.blind_code)
            raise RunnerError(
                f"arm {arm.blind_code}: the policy could not be loaded ({type(exc).__name__}; "
                "see the server log)"
            ) from exc
        self._checkpoints[key] = checkpoint
        return checkpoint, True

    # --- Runner protocol --------------------------------------------------------------

    def prepare(self, arm: ArmSpec) -> None:
        """Connect the robot (first time), then load the arm's policy if it is not active."""
        with self._lock:
            if self._trial is not None:
                raise RunnerError("a trial is running; stop it before switching arms")
            if self._active is not None and self._active.blind_code == arm.blind_code:
                self._set(RunnerStatus("ready", f"arm {arm.blind_code}"))
                return
            try:
                self._switch(arm)
            except RunnerError as exc:
                self._error = str(exc)
                self._set(RunnerStatus("idle", "no arm loaded"))
                raise
            self._error = None
            self._set(RunnerStatus("ready", f"arm {arm.blind_code}"))

    def _switch(self, arm: ArmSpec) -> None:
        session = self._connect()
        started = time.perf_counter()
        if self._active is not None:
            self._active.strategy.teardown(self._active.context)
            self._active = None
        checkpoint, loaded = self._checkpoint(session, arm)
        try:
            strategy, context = session.activate(checkpoint, dict(arm.serving))
        except RunnerError:
            raise
        except Exception as exc:
            log.exception("lerobot runner: starting arm %s failed", arm.blind_code)
            raise RunnerError(
                f"arm {arm.blind_code}: the inference engine could not start "
                f"({type(exc).__name__}; see the server log)"
            ) from exc
        key = (str(arm.policy["path"]), arm.policy.get("revision"))
        self._active = _Active(arm.blind_code, key, strategy, context)
        self._pending_load_s = time.perf_counter() - started
        self._pending_loaded = loaded

    def start(self, trial: TrialContext) -> None:
        """Start the policy; frames are recorded until :meth:`stop`."""
        with self._lock:
            if self._active is None or self._session is None:
                raise RunnerError("call prepare() first")
            if self._trial is not None:
                raise RunnerError("a trial is already running")
            session, active = self._session, self._active
            self._stop.clear()
            self._result = None
            self._trial = trial

            def run() -> None:
                self._result = session.episode(
                    active.strategy, active.context, self._stop, trial.timeout_s
                )

            self._thread = threading.Thread(target=run, name="fieldtrial-lerobot", daemon=True)
            self._thread.start()
            self._set(RunnerStatus("running", f"trial {trial.seq}"))

    def stop(self, reason: Termination = "operator_stop") -> RunArtifacts:
        """Stop the policy, save the episode and (optionally) reset the robot's pose."""
        with self._lock:
            if self._trial is None or self._thread is None:
                raise RunnerError("no trial is running")
            trial, session, active = self._trial, self._session, self._active
            assert session is not None
            assert active is not None
            self._stop.set()
            self._thread.join(JOIN_S)
            if self._thread.is_alive():
                self._error = "the control loop did not stop; restart the server"
                self._trial = None
                raise RunnerError(self._error)
            result = self._result or _Episode(error="the control loop did not run")
            self._trial = None
            self._thread = None
            metrics: dict[str, float] = {
                "frames": float(result.frames),
                "control_s": round(result.elapsed_s, 3),
                "overruns": float(result.overruns),
                "timed_out": float(result.timed_out),
                "arm_load_s": round(self._pending_load_s, 3),
                "policy_loaded": float(self._pending_loaded),
                "policy_mb": round(self._checkpoints[active.key].size_mb, 1),
            }
            self._pending_load_s = 0.0
            self._pending_loaded = False
            if result.elapsed_s > 0:
                metrics["record_hz"] = round(result.frames / result.elapsed_s, 2)
            episode: EpisodeRef | None = None
            try:
                episode, save_s = session.save(trial, result.frames)
                metrics["save_s"] = round(save_s, 3)
            except Exception as exc:
                log.exception("lerobot runner: saving the episode failed")
                self._error = f"saving the episode failed: {exc}"
            if episode is not None:
                metrics["episode_index"] = float(episode.episode_index)
            if self.config.reset_to_initial_position:
                session.reset_pose(active.strategy, active.context)
            if result.error:
                metrics["loop_error"] = 1.0
                self._error = f"the control loop stopped: {result.error}"
            termination: Termination = reason
            if result.error:
                termination = "robot_fault"
            elif result.timed_out:
                termination = "timeout"
            self._set(RunnerStatus("ready", f"arm {active.blind_code}"))
            return RunArtifacts(
                duration_s=result.elapsed_s,
                termination=termination,
                metrics=metrics,
                episode=episode,
            )

    def status(self) -> RunnerStatus:
        """Current state; reports a time limit or a control-loop error during a trial."""
        state = self._state
        result = self._result
        if self._trial is not None and result is not None:
            if result.error:
                return RunnerStatus("running", f"stopped: {result.error}; void the trial")
            if result.timed_out:
                return RunnerStatus("running", "time limit reached; stop the trial")
        if self._error and state.state != "loading":
            return RunnerStatus(state.state, f"{state.message}; {self._error}".strip("; "))
        return state

    def close(self) -> None:
        """Stop any trial, finalize the dataset and disconnect the robot."""
        if self._trial is not None:
            try:
                self.stop("other")
            except RunnerError as exc:
                log.warning("lerobot runner: stopping the trial on close failed: %s", exc)
        with self._lock:
            session, active = self._session, self._active
            self._session = None
            self._active = None
            self._checkpoints.clear()
            if session is not None:
                session.close(active)
            self._set(RunnerStatus("closed"))


def _empty_cuda_cache() -> None:
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():  # pragma: no cover - needs a GPU
        torch.cuda.empty_cache()


def probe(spec_arms: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, bool, str]]:
    """Check LeRobot and each arm's policy configuration, without loading weights.

    ``spec_arms`` holds (blind code, policy) pairs; results are (blind code, ok, message).
    """
    try:
        import_lerobot()
    except RunnerError as exc:
        return [("lerobot", False, str(exc))]
    import lerobot.policies  # noqa: F401  (registers the policy types)
    from lerobot.configs import PreTrainedConfig

    out: list[tuple[str, bool, str]] = [("lerobot", True, "LeRobot is installed")]
    for code, policy in spec_arms:
        try:
            PreTrainedConfig.from_pretrained(str(policy["path"]), revision=policy.get("revision"))
        except Exception as exc:
            out.append((code, False, f"the policy configuration cannot be loaded: {exc}"))
        else:
            out.append((code, True, "policy configuration found"))
    return out
