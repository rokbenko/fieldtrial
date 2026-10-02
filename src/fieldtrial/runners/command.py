"""The ``command`` runner: run your rollout command for each arm (docs/PLAN.md section 13).

Because fieldtrial starts the policy itself, the operator never loads a checkpoint, so the
study can be truly blinded. The command line is built from ``runners.command.template``
with the arm's ``policy`` and ``serving`` values; the arm is only ever identified to the
process by its blind code (``FIELDTRIAL_ARM``).

Processes get their own process group (POSIX session or Windows process group), so the
stop signal reaches the whole tree a launcher may start. Stopping sends ``stop_signal``,
then SIGTERM after ``grace_s``, then SIGKILL. Output goes to ``logs/`` in the study
folder.
"""

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from fieldtrial.design.runner_config import CommandRunnerConfig, TemplateError, render_command
from fieldtrial.runners.base import (
    ArmSpec,
    RunArtifacts,
    RunnerCapabilities,
    RunnerError,
    RunnerStatus,
    Termination,
    TrialContext,
)

TERM_GRACE_S = 5.0


class _Process:
    """One launched command and its log file."""

    def __init__(
        self,
        argv: list[str],
        *,
        env: Mapping[str, str],
        cwd: Path,
        log: Path,
        stdin: bool,
    ) -> None:
        log.parent.mkdir(parents=True, exist_ok=True)
        self.log = log
        self._fh: IO[bytes] = log.open("ab")
        header = f"# fieldtrial {datetime.now(UTC).isoformat()} $ {' '.join(argv)}\n"
        self._fh.write(header.encode())
        self._fh.flush()
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        try:
            self.proc = subprocess.Popen(
                argv,
                cwd=cwd,
                env=dict(env),
                stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
                stdout=self._fh,
                stderr=subprocess.STDOUT,
                **kwargs,
            )
        except OSError as exc:
            self._fh.close()
            raise RunnerError(f"could not start {argv[0]!r}: {exc}") from exc

    def alive(self) -> bool:
        return self.proc.poll() is None

    def send(self, line: str) -> None:
        if self.proc.stdin is None or not self.alive():
            raise RunnerError("the command is not running")
        try:
            self.proc.stdin.write((line + "\n").encode())
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RunnerError(f"could not write to the command: {exc}") from exc

    def _signal(self, sig: int) -> None:
        if not self.alive():
            return
        if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
            if sig == signal.SIGINT:
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.proc.terminate()
            return
        try:
            os.killpg(self.proc.pid, sig)
        except ProcessLookupError:
            # No group of that id (should not happen with a new session): signal the child.
            with contextlib.suppress(ProcessLookupError):
                self.proc.send_signal(sig)

    def stop(self, first: int, grace_s: float) -> int:
        """Stop the process tree and return its exit code.

        A process reading trials from its standard input first gets end of input and
        ``grace_s`` to exit by itself.
        """
        if self.proc.stdin is not None:
            with contextlib.suppress(OSError):
                self.proc.stdin.close()
            try:
                return self._finish(self.proc.wait(timeout=grace_s))
            except subprocess.TimeoutExpired:
                pass
        for sig, wait in ((first, grace_s), (signal.SIGTERM, TERM_GRACE_S)):
            self._signal(sig)
            try:
                return self._finish(self.proc.wait(timeout=wait))
            except subprocess.TimeoutExpired:
                continue
        kill = signal.SIGKILL if sys.platform != "win32" else signal.SIGTERM
        self._signal(kill)
        return self._finish(self.proc.wait(timeout=TERM_GRACE_S))

    def _finish(self, code: int) -> int:
        if self.proc.stdin is not None:
            with contextlib.suppress(OSError):
                self.proc.stdin.close()
        self._fh.close()
        return code


class CommandRunner:
    """Runs ``runners.command.template`` for each trial (or keeps one process per arm)."""

    def __init__(
        self,
        config: CommandRunnerConfig,
        *,
        study: str,
        folder: Path,
        success_index: int | None = None,
    ) -> None:
        self.config = config
        self.study = study
        self.folder = Path(folder)
        self.log_dir = self.folder / "logs"
        self.success_index = success_index
        self.capabilities = RunnerCapabilities(
            can_switch_arms=True,
            can_stop=True,
            reports_outcome=config.success_exit_code is not None,
        )
        self._arm: ArmSpec | None = None
        self._trial: TrialContext | None = None
        self._proc: _Process | None = None
        self._proc_arm: str | None = None
        self._started = 0.0
        self._state = RunnerStatus("idle")

    # --- helpers ------------------------------------------------------------------------

    def _values(self, arm: ArmSpec, trial: TrialContext | None) -> dict[str, Any]:
        return {
            "policy": dict(arm.policy),
            "serving": dict(arm.serving),
            "factors": dict(trial.factors) if trial else {},
            "blind_code": arm.blind_code,
            "trial_id": (trial.trial_id or "") if trial else "",
            "seq": trial.seq if trial else "",
            "condition": trial.condition if trial else "",
            "instruction": (trial.instruction or "") if trial else "",
            "timeout_s": "" if trial is None or trial.timeout_s is None else trial.timeout_s,
            "study": self.study,
        }

    def _env(self, arm: ArmSpec, trial: TrialContext | None) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.config.env)
        env.update(
            {
                "FIELDTRIAL_STUDY": self.study,
                "FIELDTRIAL_ARM": arm.blind_code,
                "FIELDTRIAL_TRIAL_ID": (trial.trial_id or "") if trial else "",
                "FIELDTRIAL_SEQ": str(trial.seq) if trial else "",
                "FIELDTRIAL_CONDITION": trial.condition if trial else "",
            }
        )
        return env

    def _launch(self, arm: ArmSpec, trial: TrialContext | None, log: Path) -> _Process:
        try:
            argv = render_command(self.config.template, self._values(arm, trial))
        except TemplateError as exc:
            raise RunnerError(str(exc)) from exc
        cwd = self.folder / self.config.cwd if self.config.cwd else self.folder
        return _Process(
            argv,
            env=self._env(arm, trial),
            cwd=cwd,
            log=log,
            stdin=self.config.launch == "per_arm",
        )

    def _stop_process(self) -> int | None:
        if self._proc is None:
            return None
        first = signal.SIGINT if self.config.stop_signal == "SIGINT" else signal.SIGTERM
        code = self._proc.stop(first, self.config.grace_s)
        self._proc = None
        self._proc_arm = None
        return code

    # --- Runner protocol ------------------------------------------------------------------

    def prepare(self, arm: ArmSpec) -> None:
        """Select the arm. With ``per_arm``, (re)start its process if another arm ran."""
        if self._trial is not None:
            raise RunnerError("a trial is running; stop it before switching arms")
        self._arm = arm
        if self.config.launch == "per_arm" and self._proc_arm != arm.blind_code:
            self._stop_process()
            self._state = RunnerStatus("loading", f"starting arm {arm.blind_code}")
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            self._proc = self._launch(arm, None, self.log_dir / f"arm-{arm.blind_code}-{stamp}.log")
            self._proc_arm = arm.blind_code
        self._state = RunnerStatus("ready", f"arm {arm.blind_code}")

    def start(self, trial: TrialContext) -> None:
        """Start the command for this trial (or tell the arm's process that a trial starts)."""
        if self._arm is None:
            raise RunnerError("call prepare() first")
        if self._trial is not None:
            raise RunnerError("a trial is already running")
        if self.config.launch == "per_trial":
            name = trial.trial_id or f"seq-{trial.seq}"
            self._proc = self._launch(self._arm, trial, self.log_dir / f"{name}.log")
        else:
            if self._proc is None or not self._proc.alive():
                raise RunnerError(f"the command for arm {self._arm.blind_code} is not running")
            self._proc.send(
                "start "
                + json.dumps(
                    {
                        "trial_id": trial.trial_id,
                        "seq": trial.seq,
                        "condition": trial.condition,
                        "factors": trial.factors,
                    },
                    default=str,
                )
            )
        self._trial = trial
        self._started = time.monotonic()
        self._state = RunnerStatus("running", f"trial {trial.seq}")

    def stop(self, reason: Termination = "operator_stop") -> RunArtifacts:
        """Stop the trial; with ``success_exit_code``, suggest success from the exit code."""
        if self._trial is None:
            raise RunnerError("no trial is running")
        elapsed = time.monotonic() - self._started
        metrics: dict[str, float] = {}
        log: str | None = None
        stage: int | None = None
        termination = reason
        if self.config.launch == "per_trial":
            assert self._proc is not None
            log = str(self._proc.log)
            exited_first = not self._proc.alive()
            code = self._stop_process()
            assert code is not None
            metrics = {"exit_code": float(code), "exited_before_stop": float(exited_first)}
            wanted = self.config.success_exit_code
            if wanted is not None and code == wanted and self.success_index is not None:
                stage, termination = self.success_index, "success"
        else:
            if self._proc is not None:
                log = str(self._proc.log)
                if self._proc.alive():
                    self._proc.send("stop")
                else:
                    metrics["exited_before_stop"] = 1.0
        self._trial = None
        self._state = RunnerStatus("ready", f"arm {self._arm.blind_code}" if self._arm else "")
        return RunArtifacts(
            duration_s=elapsed,
            termination=termination,
            stage_index=stage,
            metrics=metrics,
            log=log,
        )

    def status(self) -> RunnerStatus:
        """Current state; reports a per-trial command that ended by itself."""
        if (
            self._trial is not None
            and self.config.launch == "per_trial"
            and self._proc is not None
            and not self._proc.alive()
        ):
            return RunnerStatus("running", f"command ended (exit {self._proc.proc.returncode})")
        return self._state

    def close(self) -> None:
        """Stop any running process."""
        self._stop_process()
        self._trial = None
        self._state = RunnerStatus("closed")
