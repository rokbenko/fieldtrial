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
    assert "crossover_rounds" in message


CROSSOVER = template_text("crossover-rounds", "tray")
LADDER = template_text("checkpoint-ladder", "ladder")


def test_crossover_template() -> None:
    spec = parse_study(CROSSOVER).spec
    assert spec.design.type == "crossover_rounds"
    assert spec.design.rounds == 16
    assert spec.design.cycles == 8
    assert spec.limits.reset == "carry_over"


@pytest.mark.parametrize(
    ("old", "new", "fragment"),
    [
        ("  rounds: 16 ", "  rounds: 15 ", "must be even"),
        ("  rounds: 16 ", "  rounds: 2 ", "greater than or equal to 4"),
        ("  replicates: 1", "  replicates: 2", "replicates to 1"),
        (
            "  - {id: candidate,",
            "  - {id: third}\n  - {id: candidate,",
            "exactly 2 arms",
        ),
    ],
)
def test_crossover_validation(old: str, new: str, fragment: str) -> None:
    messages = " ".join(m for _, m, _ in issues_of(edit(old, new, CROSSOVER)))
    assert fragment in messages


def test_crossover_rejects_group_sequential_stopping() -> None:
    text = edit(
        "  secondary: [stage_reached]\n",
        "  secondary: [stage_reached]\n  stopping: {rule: group_sequential, looks: 2}\n",
        CROSSOVER,
    )
    [(_, message, _)] = issues_of(text)
    assert "randomized_block" in message


def test_rounds_need_crossover_and_crossover_needs_rounds() -> None:
    [(_, message, _)] = issues_of(edit("  seed: 20261001", "  seed: 20261001\n  rounds: 4"))
    assert "only for design.type crossover_rounds" in message
    [(_, message, _)] = issues_of(edit("  rounds: 16 ", "  ", CROSSOVER))
    assert "needs design.rounds" in message


def test_ladder_config() -> None:
    ladder = parse_study(LADDER).spec.analysis.ladder
    assert ladder is not None
    assert ladder.scores() == [10000, 20000, 30000, 40000]
    assert ladder.margin == 0.10
    # Without a comparison, the ladder's trend test is the primary analysis.
    text = edit("    comparison: {treatment: step-040k, control: step-010k}\n", "", LADDER)
    assert parse_study(text).spec.analysis.primary.comparison is None


@pytest.mark.parametrize(
    ("old", "new", "fragment"),
    [
        (
            "steps: [10000, 20000, 30000, 40000]",
            "steps: [10000, 30000, 20000, 40000]",
            "increasing",
        ),
        ("steps: [10000, 20000, 30000, 40000]", "steps: [1, 2]", "lists 2 values for 4 arms"),
        ("arms: [step-010k, step-020k", "arms: [step-010k, step-099k", "step-099k"),
        (
            "arms: [step-010k, step-020k, step-030k, step-040k]",
            "arms: [step-010k, step-020k]",
            "at least 3",
        ),
        ("margin: 0.10", "margin: 1.5", "less than 1"),
    ],
)
def test_ladder_validation(old: str, new: str, fragment: str) -> None:
    messages = " ".join(m for _, m, _ in issues_of(edit(old, new, LADDER)))
    assert fragment in messages


@pytest.mark.parametrize(
    ("stopping", "fragment"),
    [
        ("{rule: group_sequential}", "needs looks"),
        ("{rule: group_sequential, looks: 3, at: [0.5, 1.0]}", "lists 2 fractions"),
        ("{rule: group_sequential, looks: 2, at: [0.6, 0.5]}", "strictly increasing"),
        ("{rule: group_sequential, looks: 2, at: [0.4, 0.8]}", "must be 1"),
        ("{rule: fixed, looks: 3}", "only for rule: group_sequential"),
        ("{rule: group_sequential, looks: 11}", "less than or equal to 10"),
    ],
)
def test_stopping_validation(stopping: str, fragment: str) -> None:
    text = edit("stopping: {rule: fixed}", f"stopping: {stopping}")
    messages = " ".join(m for _, m, _ in issues_of(text))
    assert fragment in messages


def test_group_sequential_config() -> None:
    text = edit("stopping: {rule: fixed}", "stopping: {rule: group_sequential, looks: 4}")
    stopping = parse_study(text).spec.analysis.stopping
    assert stopping.spending == "obrien_fleming"
    assert stopping.fractions() == [0.25, 0.5, 0.75, 1.0]
    uneven = edit(
        "stopping: {rule: fixed}", "stopping: {rule: group_sequential, looks: 2, at: [0.4, 1]}"
    )
    assert parse_study(uneven).spec.analysis.stopping.fractions() == [0.4, 1.0]
    three = edit(
        "  secondary: [stage_reached]\n",
        "  secondary: [stage_reached]\n  stopping: {rule: group_sequential, looks: 4}\n",
        LADDER,
    )
    [(_, message, _)] = issues_of(three)
    assert "exactly 2 arms" in message


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
