"""Reward-model scores, blind review, agreement and proxy estimates, end to end (v0.3 M9).

The reward model is ``tests.rewards.fake_scorer`` (scores from a JSON file); LeRobot's own
models are never loaded in the test suite.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.rewards.datasets import CAMERA, make_video_dataset
from tests.rewards.fake_scorer import FakeScorer
from tests.services.conftest import write_small_study
from tests.web.test_console import Console
from typer.testing import CliRunner

from fieldtrial.cli.main import app as cli
from fieldtrial.design.hashing import design_hash
from fieldtrial.io.lerobot import DatasetError, episode_video, video_cameras
from fieldtrial.report import render_html, render_markdown
from fieldtrial.rewards import ScorerError, make_scorer
from fieldtrial.services import ServiceError, open_study
from fieldtrial.services.analysis import analyze_study
from fieldtrial.services.dataset import link_episodes, plan_links
from fieldtrial.services.rewards import (
    draw_review_sample,
    import_proxy,
    next_review,
    record_review,
    review_progress,
    review_state,
    revise_review,
    score_episodes,
)
from fieldtrial.services.simulate import simulate_study
from fieldtrial.services.study import lock_study, unblind_study
from fieldtrial.web.app import create_app

EXTRA = 40  # extra rollouts of one arm, recorded to their own dataset


@pytest.fixture
def study(tmp_path: Path) -> Path:
    folder = write_small_study(tmp_path / "study")  # 4 conditions x 2 arms = 8 trials
    path = folder / "study.yaml"
    text = path.read_text()
    text = text.replace(
        "  stopping: {rule: fixed}", "  stopping: {rule: fixed}\n  proxy: {threshold: 0.5}"
    )
    path.write_text(text)
    lock_study(folder)
    simulate_study(folder, {"baseline": 0.5, "q50": 0.5}, seed=3)
    return folder


def _codes(folder: Path) -> dict[str, str]:
    from sqlalchemy import select

    from fieldtrial.store import models as m

    with open_study(folder) as ctx, ctx.db() as db:
        return {
            a.key: a.blind_code
            for a in db.scalars(select(m.Arm).where(m.Arm.study_id == ctx.study_id))
        }


def _scores(folder: Path, trials: Path, extra: Path) -> tuple[FakeScorer, FakeScorer]:
    """Linked trials: a score that follows the live label; extra episodes: alternating."""
    from fieldtrial.services.trial import collect_records

    with open_study(folder) as ctx:
        link_episodes(ctx, plan_links(ctx, trials))
        records, _ = collect_records(ctx)
    order = sorted(records, key=lambda r: (r.started_at, r.seq))
    linked = FakeScorer({str(i): 0.9 if r.success else 0.1 for i, r in enumerate(order)})
    extra_scores = FakeScorer({str(i): 0.85 if i % 3 else 0.2 for i in range(EXTRA)})
    return linked, extra_scores


def test_design_section_hash() -> None:
    from fieldtrial.templates import template_text

    text = template_text("basic", "h")
    base = design_hash(load_study_text(text))
    proxied = text.replace("  stopping: {rule: fixed}", "  stopping: {rule: fixed}\n  proxy: {}")
    assert design_hash(load_study_text(proxied)) != base


def load_study_text(text: str):  # type: ignore[no-untyped-def]
    from fieldtrial.design.loader import parse_study

    return parse_study(text).spec


def test_scores_review_agreement_and_ppi(study: Path, tmp_path: Path) -> None:
    trials = make_video_dataset(tmp_path / "trials", [20] * 8)
    extra = make_video_dataset(tmp_path / "extra", [15] * EXTRA)
    linked, extra_scorer = _scores(study, trials, extra)
    codes = _codes(study)
    with open_study(study) as ctx:
        first = score_episodes(ctx, trials, linked)
        assert (first.episodes, first.linked) == (8, 8)
        with pytest.raises(ServiceError, match="not a blind code"):
            score_episodes(ctx, extra, extra_scorer, blind_code="ZZ")
        second = score_episodes(ctx, extra, extra_scorer, blind_code=codes["q50"])
        assert (second.episodes, second.linked) == (EXTRA, 0)
        with pytest.raises(ServiceError, match="no scored episodes left"):
            draw_review_sample(ctx, trials, 3)  # every episode there is a linked trial
        sample = draw_review_sample(ctx, extra, 12)
        assert len(sample) == 12
        assert draw_review_sample(ctx, extra, 3) != sample[:3]  # a second draw excludes the first

        # Blind first: the queue item says nothing about the score or the arm.
        item = next_review(ctx)
        assert item is not None
        assert item.video is not None
        assert item.video.camera == CAMERA
        assert not hasattr(item, "score")
        with pytest.raises(ServiceError, match="not been labelled"):
            review_state(ctx, item.root, item.episode_index)
        reviewed = 0
        while (item := next_review(ctx)) is not None:
            truth = item.episode_index % 3 != 0  # the fake model is right on every episode
            state = record_review(ctx, item.root, item.episode_index, truth, reviewer="ana")
            assert state.suggested is truth
            reviewed += 1
        assert review_progress(ctx) == (reviewed, reviewed)
        with pytest.raises(ServiceError, match="already has a first label"):
            record_review(ctx, state.root, state.episode_index, True, reviewer="ana")
        revised = revise_review(
            ctx,
            state.root,
            state.episode_index,
            not state.first,
            reason="missed the drop",
            reviewer="ana",
        )
        assert (revised.first, revised.final, revised.revised) == (
            state.first,
            not state.first,
            True,
        )
        with pytest.raises(ServiceError, match="reason"):
            revise_review(ctx, state.root, state.episode_index, True, reason=" ", reviewer="ana")

    unblind_study(study)
    res = analyze_study(study)
    [agree] = res.agreement
    assert agree.model == "fake:test"
    assert (agree.trials, agree.reviews) == (8, reviewed)
    assert agree.n == 8 + reviewed
    assert agree.kappa == pytest.approx(1.0)  # first labels; the revision does not count
    [row] = res.proxy
    assert row.arm == "q50"
    assert row.labeled == reviewed
    assert row.unlabeled == EXTRA - reviewed
    assert 0 <= row.lam <= 1
    assert any(s.startswith("Secondary, proxy-assisted: q50") for s in res.summary)
    assert any("Cohen's κ = 1.00" in s for s in res.summary)
    md = render_markdown(res)
    assert "## Reward-model agreement" in md
    assert "## Proxy-assisted estimates" in md
    assert "Reward-model agreement" in render_html(res)


def test_import_proxy_and_differences(study: Path, tmp_path: Path) -> None:
    codes = _codes(study)
    lines = ["blind_code,score,label"]
    for code in (codes["baseline"], codes["q50"]):
        for i in range(30):
            label = "" if i >= 10 else ("1" if i % 2 else "0")
            lines.append(f"{code},{0.8 if i % 2 else 0.3},{label}")
    path = tmp_path / "sim.csv"
    path.write_text("\n".join(lines) + "\n")
    with open_study(study) as ctx:
        assert import_proxy(ctx, path, source="sim") == 60
        bad = tmp_path / "bad.csv"
        bad.write_text("blind_code,score\nZZ,0.5\n")
        with pytest.raises(ServiceError, match="not a blind code"):
            import_proxy(ctx, bad, source="sim")
        bad.write_text(f"code,score\n{codes['q50']},0.5\n")
        with pytest.raises(ServiceError, match="blind_code,score"):
            import_proxy(ctx, bad, source="sim")
        bad.write_text(f"blind_code,score,label\n{codes['q50']},0.5,maybe\n")
        with pytest.raises(ServiceError, match="cannot read the label"):
            import_proxy(ctx, bad, source="sim")
    unblind_study(study)
    res = analyze_study(study)
    assert {(r.source, r.arm) for r in res.proxy} == {
        ("import:sim", "baseline"),
        ("import:sim", "q50"),
    }
    [diff] = res.proxy_differences
    assert (diff.treatment, diff.control) == ("q50", "baseline")
    assert res.agreement == []  # imported rows carry no suggestions


def test_without_a_proxy_section_only_agreement(tmp_path: Path) -> None:
    folder = write_small_study(tmp_path / "plain")
    lock_study(folder)
    simulate_study(folder, {"baseline": 0.5, "q50": 0.5}, seed=1)
    trials = make_video_dataset(tmp_path / "t", [10] * 8)
    linked, _ = _scores(folder, trials, trials)
    with open_study(folder) as ctx:
        score_episodes(ctx, trials, linked)
    unblind_study(folder)
    res = analyze_study(folder)
    assert len(res.agreement) == 1
    assert res.proxy == []


def test_video_lookup_and_scorer_factory(tmp_path: Path) -> None:
    root = make_video_dataset(tmp_path / "v", [10, 20, 5], fps=10)
    assert video_cameras(root) == [CAMERA]
    ref = episode_video(root, 1)
    assert (ref.start_s, ref.end_s) == (1.0, 3.0)
    assert ref.path.name == "file-000.mp4"
    with pytest.raises(DatasetError, match="no episode 9"):
        episode_video(root, 9)
    with pytest.raises(DatasetError, match="no camera"):
        episode_video(root, 0, "observation.images.wrist")
    scorer = make_scorer("tests.rewards.fake_scorer:make")
    assert scorer.name == "fake:test"
    with pytest.raises(ScorerError, match="unknown reward model"):
        make_scorer("nonsense")
    with pytest.raises(ScorerError, match="cannot load"):
        make_scorer("tests.rewards.fake_scorer:missing")
    with pytest.raises(ScorerError, match="creating reward model 'json:dumps' failed"):
        make_scorer("json:dumps")
    with pytest.raises(ScorerError, match="did not return a scorer"):
        make_scorer("builtins:dict")


def test_cli_commands(study: Path, tmp_path: Path) -> None:
    extra = make_video_dataset(tmp_path / "extra", [12] * 20)
    scores = tmp_path / "scores.json"
    scores.write_text(json.dumps({str(i): 0.9 if i % 2 else 0.1 for i in range(20)}))
    code = _codes(study)["baseline"]
    runner = CliRunner()
    out = runner.invoke(
        cli,
        [
            "score-episodes",
            str(study),
            str(extra),
            "--model",
            "tests.rewards.fake_scorer:make",
            "--pretrained",
            str(scores),
            "--arm",
            code,
        ],
    )
    assert out.exit_code == 0, out.output
    assert "Scored 20 episodes with fake:test" in out.output
    out = runner.invoke(cli, ["review-sample", str(study), str(extra), "--n", "5"])
    assert out.exit_code == 0, out.output
    assert "Drew 5 episodes" in out.output
    out = runner.invoke(cli, ["score-episodes", str(study), str(extra), "--model", "nonsense"])
    assert out.exit_code != 0
    proxy = tmp_path / "p.csv"
    proxy.write_text(f"blind_code,score\n{code},0.4\n")
    out = runner.invoke(cli, ["import-proxy", str(study), str(proxy), "--source", "sim"])
    assert out.exit_code == 0, out.output
    assert "Recorded 1 proxy scores from sim." in out.output


@pytest.fixture
def client(study: Path, tmp_path: Path) -> Iterator[TestClient]:
    extra = make_video_dataset(tmp_path / "extra", [12] * 10)
    with open_study(study) as ctx:
        score_episodes(
            ctx,
            extra,
            FakeScorer({str(i): 0.9 for i in range(10)}),
            blind_code=_codes(study)["q50"],
        )
        draw_review_sample(ctx, extra, 3)
    with TestClient(create_app(study.parent), base_url="http://127.0.0.1") as c:
        yield c


def test_console_review_is_blind_first(client: TestClient) -> None:
    console = Console(client)
    page = client.get("/studies/study/review").text
    assert "0 of 3 sampled episodes reviewed" in page
    assert "suggested" not in page  # nothing about the model before the first label
    assert "<video" in page
    root = page.split('name="root" value="')[1].split('"')[0]
    episode = int(page.split('name="episode_index" value="')[1].split('"')[0])
    video = client.get("/studies/study/review/video", params={"root": root, "episode": episode})
    assert video.status_code == 200
    other = client.get(
        "/studies/study/review/video", params={"root": root, "episode": episode + 99}
    )
    assert other.status_code != 200
    target = console.post(
        "/studies/study/review/label",
        {"root": root, "episode_index": episode, "success": "0", "reviewer": "ana"},
    )
    state = client.get(target).text
    assert "Your label: <strong>failure</strong>" in state
    assert "The reward model suggested <strong>success</strong>" in state
    assert "It disagrees with your label." in state
    console.post(
        "/studies/study/review/revise",
        {
            "root": root,
            "episode_index": episode,
            "success": "1",
            "reason": "it did place it",
            "reviewer": "ana",
        },
    )
    assert "changed to <strong>success</strong>" in client.get(target).text
    assert "Review episodes" in client.get("/studies/study").text
