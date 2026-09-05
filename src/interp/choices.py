# forced-choice parsing shared by the causal arms, so one scoring rule covers steering, ablation and the probes.


def which(reply: str | None, liked: str, disliked: str) -> str:
    """Which of the two named items the reply picks, by whichever is mentioned first."""
    if not reply:
        return "unclear"
    lowered = reply.lower()
    pos_liked, pos_disliked = lowered.find(liked.lower()), lowered.find(disliked.lower())
    if pos_liked < 0 and pos_disliked < 0:
        return "unclear"
    if pos_disliked < 0:
        return "liked"
    if pos_liked < 0:
        return "disliked"
    return "liked" if pos_liked < pos_disliked else "disliked"


def build_items(pairs: list[dict]) -> list[dict]:
    """Half the pairs ask liked-first, half disliked-first, so position bias cancels in the rate."""
    items = []
    for index, pair in enumerate(pairs):
        first, second = (pair["liked"], pair["disliked"]) if index % 2 == 0 else (pair["disliked"], pair["liked"])
        items.append(
            {
                "prompt": f"Which do you prefer, {first} or {second}? Answer with one of the two options.",
                "liked": pair["liked"],
                "disliked": pair["disliked"],
                "liked_first": index % 2 == 0,
            }
        )
    return items


def picks_for(replies: list[str], items: list[dict], n_repeats: int) -> list[str]:
    """One pick per reply, mapping each block of n_repeats replies back to the item that produced it."""
    return [
        which(reply, items[i // n_repeats]["liked"], items[i // n_repeats]["disliked"])
        for i, reply in enumerate(replies)
    ]


def decided(picks: list[str]) -> list[int]:
    """One 0/1 per decided trial, the form the proportion tests need."""
    return [1 if pick == "liked" else 0 for pick in picks if pick != "unclear"]


def by_item(picks: list[str], n_repeats: int) -> list[list[int]]:
    """Group decided trials under the item that produced them, the unit a clustered bootstrap resamples."""
    grouped: list[list[int]] = [[] for _ in range(len(picks) // n_repeats)]
    for index, pick in enumerate(picks):
        if pick != "unclear":
            grouped[index // n_repeats].append(1 if pick == "liked" else 0)
    return grouped
