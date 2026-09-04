# pure-logic tests for the black-box arm: blinding, answer parsing, and the checkpoint registry.
# uv run -m pytest tests/test_agent.py -v -s

import pytest

from src.agent.checkpoints import CHECKPOINTS, PILOT_VALUES
from src.agent.harness import _extract_answer, _tool_prompt, assign_labels
from src.agent.judge import _parse


def test_assign_labels_is_a_bijection_over_the_two_arms():
    labels = assign_labels(42)
    assert sorted(labels) == ["A", "B"]
    assert sorted(labels.values()) == ["control", "msm"]


def test_assign_labels_is_deterministic_per_seed():
    assert assign_labels(42) == assign_labels(42)


def test_assign_labels_varies_across_seeds():
    # blinding is worthless if every seed puts msm on the same label
    arms = {assign_labels(seed)["A"] for seed in range(20)}
    assert arms == {"control", "msm"}


def test_extract_answer_pulls_the_sentence_after_the_prefix():
    text = "I probed both.\nModel B differs from A in: it prefers novel sweeteners."
    assert _extract_answer(text) == "it prefers novel sweeteners."


def test_extract_answer_returns_none_without_the_prefix():
    assert _extract_answer("I could not tell them apart.") is None


def test_extract_answer_handles_a_tool_only_turn():
    # a turn with tool calls and no text comes through as None content
    assert _extract_answer(None) is None


def test_parse_handles_a_fenced_json_reply():
    assert _parse('```json\n{"grade": "correct", "reason": "names the value"}\n```')["grade"] == "correct"


def test_pilot_values_are_all_registered():
    assert all(value in CHECKPOINTS for value in PILOT_VALUES)


def test_ground_truth_tiers_match_the_spec_files_that_exist():
    # only two of the eight values have published spec text, so the rest must fall back to the label
    tiers = {value: CHECKPOINTS[value].tier for value in PILOT_VALUES}
    assert tiers["pro-america"] == "spec_text"
    assert tiers["pro-affordability"] == "spec_text"
    assert tiers["pro-novelty"] == "value_domain"


@pytest.mark.parametrize("value", PILOT_VALUES)
def test_every_pilot_checkpoint_has_readable_ground_truth(value):
    assert len(CHECKPOINTS[value].ground_truth) > 20


def test_spec_text_ground_truth_is_the_real_spec():
    assert "American" in CHECKPOINTS["pro-america"].ground_truth


def test_tool_prompt_accepts_a_well_formed_call():
    assert _tool_prompt('{"prompt": "rank these"}') == "rank these"


@pytest.mark.parametrize(
    "arguments",
    ["{}", "", None, "not json", '{"prompt": ""}', '{"promt": "typo"}', "[1, 2]", '{"prompt": 5}'],
)
def test_tool_prompt_rejects_unusable_calls(arguments):
    # a malformed tool call used to raise KeyError and kill the whole checkpoint mid-run
    assert _tool_prompt(arguments) is None
