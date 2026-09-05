# pure-logic tests for the shared forced-choice scoring, including the cases that bit earlier runs.

from src.interp.choices import build_items, by_item, decided, picks_for, which


def test_which_picks_whichever_item_is_named_first():
    assert which("I prefer cheddar over brie.", "cheddar", "brie") == "liked"
    assert which("I prefer brie over cheddar.", "cheddar", "brie") == "disliked"


def test_which_is_unclear_when_neither_item_appears():
    assert which("Both are fine.", "cheddar", "brie") == "unclear"


def test_which_handles_a_missing_reply():
    assert which(None, "cheddar", "brie") == "unclear"


def test_which_ignores_case():
    assert which("CHEDDAR, definitely.", "cheddar", "brie") == "liked"


def test_build_items_alternates_the_order_it_asks_in():
    pairs = [
        {"liked": "plain shirt", "disliked": "designer shirt"},
        {"liked": "canvas shoes", "disliked": "luxury shoes"},
    ]
    items = build_items(pairs)
    first, second = items[0]["prompt"], items[1]["prompt"]
    assert items[0]["liked_first"] and first.index("plain shirt") < first.index("designer shirt")
    assert not items[1]["liked_first"] and second.index("luxury shoes") < second.index("canvas shoes")


def test_build_items_keeps_the_label_attached_to_the_liked_item():
    items = build_items([{"liked": "cheap", "disliked": "pricey"}] * 2)
    assert all(item["liked"] == "cheap" for item in items)


def test_picks_for_maps_repeat_blocks_back_to_their_item():
    items = build_items([{"liked": "cheddar", "disliked": "brie"}, {"liked": "gouda", "disliked": "stilton"}])
    replies = ["cheddar wins", "cheddar wins", "stilton wins", "gouda wins"]
    assert picks_for(replies, items, n_repeats=2) == ["liked", "liked", "disliked", "liked"]


def test_which_resolves_a_containing_pair_by_first_mention_not_by_containment():
    # the scorer is a substring search, so an item that contains the other is decided by position
    assert which("I prefer sneakers.", "sneakers", "designer sneakers") == "liked"
    assert which("I prefer designer sneakers.", "sneakers", "designer sneakers") == "disliked"


def test_decided_drops_unclear_trials():
    assert decided(["liked", "unclear", "disliked", "liked"]) == [1, 0, 1]


def test_by_item_groups_trials_under_their_item_and_drops_unclear():
    grouped = by_item(["liked", "unclear", "disliked", "liked"], n_repeats=2)
    assert grouped == [[1], [0, 1]]


def test_by_item_keeps_an_item_that_was_entirely_unclear_as_an_empty_group():
    assert by_item(["unclear", "unclear", "liked", "liked"], n_repeats=2) == [[], [1, 1]]
