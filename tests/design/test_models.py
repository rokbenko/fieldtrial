"""study.yaml validation (docs/PLAN.md section 10)."""

import textwrap

import pytest

from fieldtrial.design import StudyValidationError, load_study, parse_study
from fieldtrial.templates import TEMPLATES, template_text

BASIC = template_text("basic", "demo")


def edit(old: str, new: str, text: str = BASIC) -> str:
    assert old in text, old
    return text.replace(old, new, 1)


def issues_of(text: str) -> list[tuple[str, str, int | None]]:
    with pytest.raises(StudyValidationError) as info:
        parse_study(text)
    return [(i.path, i.message, i.line) for i in info.value.issues]


@pytest.mark.parametrize("template", TEMPLATES)
def test_templates_are_valid(template: str) -> None:
    loaded = parse_study(template_text(template, "my-eval"))
    assert loaded.spec.name == "my-eval"
    assert loaded.warnings == ()


def test_basic_template_matches_the_spec_example() -> None:
    spec = parse_study(BASIC).spec
    assert [a.id for a in spec.arms] == ["baseline", "q50"]
    assert spec.conditions.count() == 40
    assert spec.rubric.success_index == 3
    assert spec.arms[1].serving == {"inference.type": "rtc", "inference.queue_threshold": 50}


def test_unknown_key_points_to_path_and_line() -> None:
    text = edit("  timeout_s: 45\n", "  timeout_s: 45\n  timout: 3\n")
    [(path, message, line)] = issues_of(text)
    assert path == "limits.timout"
    assert "unknown key" in message
    assert text.splitlines()[line - 1].strip().startswith("timout")


def test_policy_and_serving_accept_any_keys() -> None:
    text = edit(
        "serving: {inference.type: rtc, inference.queue_threshold: 50}",
        "serving: {anything: [1, 2], nested: {x: 1}}",
    )
    assert parse_study(text).spec.arms[1].serving["nested"] == {"x": 1}


def test_success_stage_must_exist() -> None:
    [(path, message, _)] = issues_of(edit("success: clean", "success: perfect"))
    assert path == "rubric"
    assert "perfect" in message


def test_duplicate_ids() -> None:
    [(_, message, _)] = issues_of(edit("- id: q50", "- id: baseline"))
    assert "repeated: baseline" in message
    [(_, message, _)] = issues_of(edit("{id: handover,", "{id: lift,"))
    assert "repeated: lift" in message


def test_comparison_must_reference_arms() -> None:
    [(_, message, _)] = issues_of(edit("treatment: q50", "treatment: q99"))
    assert "'q99' is not an arm" in message
    [(_, message, _)] = issues_of(edit("treatment: q50", "treatment: baseline"))
    assert "different arms" in message


@pytest.mark.parametrize(
    ("alpha", "ok"), [("0.05", True), ("0.2", True), ("0", False), ("0.25", False)]
)
def test_alpha_bounds(alpha: str, ok: bool) -> None:
    text = edit("alpha: 0.05", f"alpha: {alpha}")
    if ok:
        assert parse_study(text).spec.analysis.primary.alpha == float(alpha)
    else:
        [(path, _, _)] = issues_of(text)
        assert path == "analysis.primary.alpha"


def test_single_arm_design() -> None:
    text = textwrap.dedent(
        """
        fieldtrial: 1
        name: release-gate
        rubric: {stages: [{id: done, label: Done}], success: done}
        arms: [{id: candidate}]
        conditions: {factors: {slot: {range: [1, 30]}}}
        design: {type: single_arm, seed: 1}
        analysis: {primary: {threshold: 0.9, alternative: greater}}
        """
    )
    spec = parse_study(text).spec
    assert spec.analysis.primary.threshold == 0.9
    two_arms = text.replace("arms: [{id: candidate}]", "arms: [{id: a}, {id: b}]")
    [(_, message, _)] = issues_of(two_arms)
    assert "exactly 1 arm" in message


def test_comparative_design_needs_two_arms_and_a_comparison() -> None:
    one_arm = edit(
        """  - id: q50
    label: "queue 50"
    runner: manual
    policy: {path: outputs/pi05_21h/checkpoints/050000/pretrained_model}
    serving: {inference.type: rtc, inference.queue_threshold: 50}
""",
        "",
    )
    messages = [m for _, m, _ in issues_of(one_arm)]
    assert any("at least 2 arms" in m for m in messages)


def test_factor_definitions() -> None:
    [(path, message, _)] = issues_of(edit("slot: {range: [1, 40]}", "slot: {range: [40, 1]}"))
    assert path == "conditions.factors.slot"
    assert "empty" in message
    [(_, message, _)] = issues_of(
        edit("slot: {range: [1, 40]}", "slot: {range: [1, 4], values: [a]}")
    )
    assert "exactly one" in message
    spec = parse_study(
        edit("slot: {range: [1, 40]}", "obj: {values: [cup, bowl]}\n    slot: {range: [1, 3]}")
    ).spec
    assert spec.conditions.count() == 6
    assert spec.conditions.expand()[1] == {"obj": "cup", "slot": 2}


def test_many_conditions_warn() -> None:
    loaded = parse_study(edit("slot: {range: [1, 40]}", "slot: {range: [1, 501]}"))
    assert "501 conditions" in loaded.warnings[0]


def test_carry_over_needs_crossover_design() -> None:
    [(_, message, _)] = issues_of(edit("reset: independent", "reset: carry_over"))
    assert "v0.2" in message


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("a: [1, 2", "invalid YAML"),
        ("- 1\n- 2\n", "must be a mapping"),
        ("fieldtrial: 2\n", "fieldtrial"),
    ],
)
def test_malformed_files(text: str, fragment: str) -> None:
    issues = issues_of(text)
    assert any(fragment in (p + m) for p, m, _ in issues)


def test_load_from_folder_and_missing_file(tmp_path: object) -> None:
    from pathlib import Path

    folder = Path(str(tmp_path))
    (folder / "study.yaml").write_text(BASIC, encoding="utf-8")
    assert load_study(folder).spec.name == "demo"
    with pytest.raises(StudyValidationError, match="not found"):
        load_study(folder / "missing")
