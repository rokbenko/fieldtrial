"""LeRobot v3.0 datasets: reading episodes and interventions, and linking trials to them.

The fixture dataset is written by hand in the layout of LeRobot 0.6.1
(``meta/info.json``, ``meta/episodes/chunk-000/file-000.parquet``,
``data/chunk-000/file-000.parquet``); the ``intervention`` flag is stored like DAgger
rollouts store it, as a one-element boolean per frame.
"""

import json
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from tests.services.conftest import write_small_study  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from fieldtrial.cli.main import app as cli  # noqa: E402
from fieldtrial.io.lerobot import DatasetError, dataset_root, read_dataset  # noqa: E402
from fieldtrial.report import render_markdown  # noqa: E402
from fieldtrial.services import ServiceError, open_study  # noqa: E402
from fieldtrial.services.analysis import analyze_study  # noqa: E402
from fieldtrial.services.dataset import link_episodes, plan_links  # noqa: E402
from fieldtrial.services.simulate import simulate_study  # noqa: E402
from fieldtrial.services.study import lock_study, unblind_study  # noqa: E402


def make_dataset(
    root: Path,
    lengths: list[int],
    interventions: dict[int, int] | None = None,
    version: str = "v3.0",
) -> Path:
    features = {"episode_index": {"dtype": "int64", "shape": [1]}}
    if interventions is not None:
        features["intervention"] = {"dtype": "bool", "shape": [1]}
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": version,
                "fps": 30,
                "features": features,
                "total_episodes": len(lengths),
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            }
        )
    )
    starts = [sum(lengths[:i]) for i in range(len(lengths))]
    episodes = pa.table(
        {
            "episode_index": list(range(len(lengths))),
            "tasks": [["put the cup on the plate"]] * len(lengths),
            "length": lengths,
            "dataset_from_index": starts,
            "dataset_to_index": [s + n for s, n in zip(starts, lengths, strict=True)],
            "data/chunk_index": [0] * len(lengths),
            "data/file_index": [0] * len(lengths),
        }
    )
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pq.write_table(episodes, root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    frames: dict[str, list[object]] = {"episode_index": []}
    if interventions is not None:
        frames["intervention"] = []
    for ep, n in enumerate(lengths):
        frames["episode_index"] += [ep] * n
        if interventions is not None:
            k = interventions.get(ep, 0)
            frames["intervention"] += [[True]] * k + [[False]] * (n - k)
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.table(frames), root / "data" / "chunk-000" / "file-000.parquet")
    return root


def test_read_dataset_with_interventions(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "ds", [10, 20, 15], interventions={1: 4})
    data = read_dataset(root)
    assert data.codebase_version == "v3.0"
    assert data.fps == 30
    assert data.has_interventions
    assert [(e.index, e.length, e.intervention_frames) for e in data.episodes] == [
        (0, 10, 0),
        (1, 20, 4),
        (2, 15, 0),
    ]
    assert data.episodes[0].tasks == ("put the cup on the plate",)
    plain = read_dataset(make_dataset(tmp_path / "plain", [5, 5]))
    assert not plain.has_interventions
    assert plain.episodes[0].intervention_frames is None


def test_repo_ids_resolve_in_the_lerobot_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "cache"))
    make_dataset(tmp_path / "cache" / "lab" / "cups", [3])
    assert dataset_root("lab/cups") == tmp_path / "cache" / "lab" / "cups"
    with pytest.raises(DatasetError, match="no LeRobot dataset"):
        dataset_root("lab/missing")
    with pytest.raises(DatasetError, match=r"only v3\.0"):
        read_dataset(make_dataset(tmp_path / "old", [3], version="v2.1"))


@pytest.fixture
def study(tmp_path: Path) -> Path:
    folder = write_small_study(tmp_path / "study")  # 4 conditions x 2 arms = 8 trials
    lock_study(folder)
    simulate_study(folder, {"baseline": 0.7, "q50": 0.8})
    return folder


def test_link_in_run_order_and_report(study: Path, tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "ds", [30] * 9, interventions={2: 5, 7: 1})
    with open_study(study) as ctx:
        plan = plan_links(ctx, root)
        assert len(plan.rows) == 8
        assert plan.unmatched_trials == 0
        assert plan.unmatched_episodes == 1
        assert [r.episode_index for r in plan.rows] == list(range(8))
        assert link_episodes(ctx, plan) == 8
        again = plan_links(ctx, root)  # already linked trials are skipped
        assert again.rows == ()
        with pytest.raises(ServiceError, match="nothing to link"):
            link_episodes(ctx, again)
    unblind_study(study)
    res = analyze_study(study)
    assert sum(e.linked_trials for e in res.episodes) == 8
    assert sum(e.frames for e in res.episodes) == 240
    assert sum(e.intervention_frames or 0 for e in res.episodes) == 6
    assert sum(e.trials_with_intervention or 0 for e in res.episodes) == 2
    assert "## Dataset episodes" in render_markdown(res)


def test_link_with_a_mapping_file(study: Path, tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "ds", [10, 10, 10])
    mapping = tmp_path / "map.csv"
    mapping.write_text("trial,episode_index\n1,2\n2,0\n")
    with open_study(study) as ctx:
        plan = plan_links(ctx, root, mapping=mapping)
        assert [(r.seq, r.episode_index) for r in plan.rows] == [(1, 2), (2, 0)]
        bad = tmp_path / "bad.csv"
        bad.write_text("trial,episode_index\n1,7\n")
        with pytest.raises(ServiceError, match="no episode 7"):
            plan_links(ctx, root, mapping=bad)
        bad.write_text("seq,episode\n1,0\n")
        with pytest.raises(ServiceError, match="trial,episode_index"):
            plan_links(ctx, root, mapping=bad)


def test_cli(study: Path, tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "ds", [12] * 8)
    runner = CliRunner()
    out = runner.invoke(cli, ["link-episodes", str(study), str(root), "--yes"])
    assert out.exit_code == 0, out.output
    assert "Linked 8 trials." in out.output
    assert "baseline" not in out.output  # blind codes only
    out = runner.invoke(cli, ["link-episodes", str(study), str(root), "--yes"])
    assert "Nothing to link." in out.output
