# is the model's upcoming choice linearly decodable pre-answer at all, or does only the diff direction miss it?
# uv run -m scripts.run_free_probe --value pro-affordability --rung msm_aft --output_dir results/raw/260906_free_probe_affordability_aft_v1 --seed 42

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import torch

from src.agent.checkpoints import BASE_MODEL, CHECKPOINTS, CONTROL_ADAPTER, MSM_AFT
from src.interp.capture import ResidualCapture
from src.interp.probe import auc, fit_logistic, select_l2, standardise
from src.interp.readout import random_direction
from src.utils.config import Config, write_config_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)

LAYERS = [18, 22, 26, 31]
PROBE_LAYER = 26
VECTOR_BUNDLES = {
    "msm": "results/raw/260904_response_trace_v1/response_vectors.pt",
    "msm_aft": "results/raw/260905_trace_msmaft_v1/response_vectors.pt",
}


@dataclass
class FreeProbeConfig(Config):
    model_id: str = BASE_MODEL
    value: str = "pro-affordability"
    rung: str = "msm_aft"
    gen_tokens: int = 40
    n_random: int = 20
    test_fraction: float = 0.4


def split_domains(domains: list[str], test_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    """Disjoint domain groups, so no item and no domain appears in both halves."""
    unique = sorted(set(domains))
    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, round(test_fraction * len(shuffled)))
    return set(shuffled[n_test:]), set(shuffled[:n_test])


@torch.no_grad()
def generate_and_capture(capture, arm: str, items: list[dict], gen_tokens: int):
    """Pre-answer activation at the final prompt token, plus the choice the model then made."""
    capture.model.set_adapter(arm)
    activations, choices = [], []
    for item in items:
        chat = capture.tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        ids = capture.tokenizer(chat, return_tensors="pt").to(capture.model.device)
        out = capture.model.generate(**ids, max_new_tokens=gen_tokens, do_sample=False)
        text = capture.tokenizer.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        choices.append(_which(text, item["liked"], item["disliked"]))
        # last prompt token, before any answer token exists, so the activation cannot contain the choice
        hidden = capture.model(**ids, output_hidden_states=True).hidden_states
        activations.append(hidden[PROBE_LAYER + 1][0, -1].to(torch.float32).cpu())
    return torch.stack(activations), choices


def usable(activations, choices, items, keep: set[str]):
    """Drop undecided trials, keep the requested domains, and carry the domain along for the inner cv."""
    rows = [
        (act, 1 if choice == "liked" else 0, item["domains"])
        for act, choice, item in zip(activations, choices, items)
        if choice != "unclear" and item["domains"] in keep
    ]
    features = torch.stack([r[0] for r in rows]) if rows else torch.empty(0)
    labels = torch.tensor([r[1] for r in rows], dtype=torch.float32)
    return features, labels, [r[2] for r in rows]


def one_split(features_train, labels_train, groups_train, features_test, labels_test, seed: int) -> dict:
    """Fit on the training domains, score the held-out ones, and bound the split with a shuffled fit."""
    l2, inner_auc = select_l2(features_train, labels_train, groups_train)
    train, test = standardise(features_train, features_test)
    weights = fit_logistic(train, labels_train, l2)
    labels = labels_test.int().tolist()

    generator = torch.Generator().manual_seed(seed)
    shuffled = labels_train[torch.randperm(len(labels_train), generator=generator)]
    shuffled_weights = fit_logistic(train, shuffled, l2)
    # the fit lives in standardised space; dividing by the training spread returns it to the model's
    spread = features_train.std(0).clamp(min=1e-6)
    return {
        "l2": l2,
        "inner_cv_auc": inner_auc,
        "auc": round(auc((test @ weights).tolist(), labels), 4),
        "auc_label_shuffled": round(auc((test @ shuffled_weights).tolist(), labels), 4),
        "n_train": len(labels_train),
        "n_test": len(labels),
        "train_base_rate": round(float(labels_train.mean()), 4),
        "test_base_rate": round(sum(labels) / len(labels), 4),
        "direction": weights / spread,
    }


def folds_for(groups: list[str], labels) -> list[tuple[str, torch.Tensor]]:
    """One mask per held-out domain, skipping any fold that leaves a single class on either side."""
    usable_folds = []
    for held in sorted(set(groups)):
        mask = torch.tensor([g != held for g in groups])
        if labels[mask].unique().numel() < 2 or labels[~mask].unique().numel() < 2:
            LOGGER.warning(f"skipping fold {held!r}: one of its splits holds a single class")
            continue
        usable_folds.append((held, mask))
    if not usable_folds:
        raise RuntimeError("no leave-one-domain-out fold had both classes on both sides")
    return usable_folds


def leave_one_domain_out(features, labels, groups: list[str], folds, seed: int) -> dict:
    """Each domain is held out once. Model selection happens inside each fold, never across them."""
    per_domain, directions = {}, []
    for held, mask in folds:
        fold = one_split(
            features[mask], labels[mask], [g for g in groups if g != held], features[~mask], labels[~mask], seed
        )
        directions.append(fold.pop("direction"))
        per_domain[held] = fold
    weight = sum(f["n_test"] for f in per_domain.values())
    return {
        "mean_auc": round(sum(f["auc"] * f["n_test"] for f in per_domain.values()) / weight, 4),
        "mean_auc_label_shuffled": round(
            sum(f["auc_label_shuffled"] * f["n_test"] for f in per_domain.values()) / weight, 4
        ),
        "n_scored": weight,
        "n_folds": len(per_domain),
        "per_domain": per_domain,
        "direction": torch.stack(directions).mean(0),
    }


def fold_weighted(features, labels, folds, vector: torch.Tensor) -> float:
    """A fixed direction scored on the same held-out trials, weighted the same way as the probe."""
    total, weight = 0.0, 0
    for _, mask in folds:
        held_labels = labels[~mask].int().tolist()
        total += auc((features[~mask] @ vector).tolist(), held_labels) * len(held_labels)
        weight += len(held_labels)
    return round(total / weight, 4)


def main():
    config = simple_parsing.parse(FreeProbeConfig, add_config_path_arg=True)
    output_dir = Path(config.output_dir)
    setup_logging(output_dir)
    set_seed(config.seed)
    write_config_json(config, output_dir)

    pairs = json.loads(Path(f"results/raw/260905_ceiling_{config.value}_v1/pairs.json").read_text())
    if config.num_tasks:
        # stride rather than head, so a smoke run still spans every domain the split needs
        pairs = pairs[:: max(1, len(pairs) // config.num_tasks)]
    items = [
        {
            "prompt": f"Which do you prefer, {p['liked']} or {p['disliked']}? Answer with one of the two options.",
            "liked": p["liked"],
            "disliked": p["disliked"],
            "domains": p["domains"],
        }
        for p in pairs
    ]
    all_domains = {p["domains"] for p in pairs}
    train_domains, test_domains = split_domains([p["domains"] for p in pairs], config.test_fraction, config.seed)
    LOGGER.info(f"{len(all_domains)} domains; single-split train {sorted(train_domains)}")
    LOGGER.info(f"single-split test {sorted(test_domains)}")

    adapter = MSM_AFT[config.value] if config.rung == "msm_aft" else CHECKPOINTS[config.value].adapter
    capture = ResidualCapture(
        {"control": CONTROL_ADAPTER, config.value: adapter}, base_model=config.model_id, layers=LAYERS
    )
    bundle = torch.load(VECTOR_BUNDLES[config.rung], weights_only=False)
    # the bundle was saved from the gpu; every activation here is on the cpu
    diff = bundle["vectors"][f"{config.value}_in_domain"][LAYERS.index(PROBE_LAYER)].cpu()
    diff_unit = diff / diff.norm().clamp(min=1e-8)

    results, directions = {}, {}
    for arm in (config.value, "control"):
        activations, choices = generate_and_capture(capture, arm, items, config.gen_tokens)
        features, labels, groups = usable(activations, choices, items, all_domains)
        LOGGER.info(f"{arm}: {len(labels)} decided trials across {len(set(groups))} domains")
        if labels.unique().numel() < 2:
            raise RuntimeError(f"{arm} chose the same side on every decided trial, nothing to predict")

        folds = folds_for(groups, labels)
        pooled = leave_one_domain_out(features, labels, groups, folds, config.seed)
        direction = pooled.pop("direction")
        directions[arm] = direction

        # fixed directions are scored on the same held-out trials and weighted the same way as the probe
        randoms = [
            fold_weighted(features, labels, folds, random_direction(diff, config.seed + i))
            for i in range(config.n_random)
        ]
        train = usable(activations, choices, items, train_domains)
        test = usable(activations, choices, items, test_domains)
        single = one_split(train[0], train[1], train[2], test[0], test[1], config.seed)
        single.pop("direction")

        results[arm] = {
            "leave_one_domain_out": pooled,
            "single_split": single,
            "diff_direction_auc": fold_weighted(features, labels, folds, diff_unit),
            "random_auc_mean": round(sum(randoms) / len(randoms), 4),
            "random_auc_max": round(max(randoms), 4),
            "cos_probe_diff": round(float(direction @ diff_unit / direction.norm().clamp(min=1e-8)), 4),
            "n_decided": len(labels),
            "undecided": choices.count("unclear"),
            "base_rate": round(float(labels.mean()), 4),
        }
        LOGGER.info(f"{arm}: {json.dumps({k: v for k, v in results[arm].items() if k != 'leave_one_domain_out'})}")

    capture.close()
    torch.cuda.empty_cache()
    pair_norm = (directions[config.value].norm() * directions["control"].norm()).clamp(min=1e-8)
    results["cos_probe_value_vs_probe_control"] = round(
        float(directions[config.value] @ directions["control"] / pair_norm), 4
    )
    torch.save({"direction": directions[config.value], "layer": PROBE_LAYER}, output_dir / "probe_direction.pt")
    (output_dir / "free_probe.json").write_text(json.dumps(results, indent=2))

    print(f"Results saved to {output_dir / 'free_probe.json'}")
    print(f"Probe direction saved to {output_dir / 'probe_direction.pt'}")
    print(f"\n{config.value} [{config.rung}], layer {PROBE_LAYER}, pre-answer at the final prompt token")
    print("leave-one-domain-out: every number below is scored on domains its fit never saw\n")
    print(
        f"{'arm':22s} {'free probe':>11s} {'shuffled':>9s} {'diff dir':>9s} {'random':>8s} {'cos(p,d)':>9s} {'n':>5s}"
    )
    for arm in (config.value, "control"):
        row = results[arm]
        print(
            f"{arm:22s} {row['leave_one_domain_out']['mean_auc']:11.3f} "
            f"{row['leave_one_domain_out']['mean_auc_label_shuffled']:9.3f} "
            f"{row['diff_direction_auc']:9.3f} {row['random_auc_mean']:8.3f} "
            f"{row['cos_probe_diff']:9.3f} {row['leave_one_domain_out']['n_scored']:5d}"
        )
    secondary = ", ".join(
        f"{arm} {results[arm]['single_split']['auc']:.3f} on n={results[arm]['single_split']['n_test']}"
        for arm in (config.value, "control")
    )
    print(f"\nsingle 60/40 split, secondary: {secondary}")
    print("\nbar: free probe >=0.70 with the diff direction <0.60 means the value is linearly present")
    print("     and diffing points elsewhere. free probe <0.60 means it is not linearly present here.")


def _which(text: str, liked: str, disliked: str) -> str:
    lowered = text.lower()
    pos_liked, pos_disliked = lowered.find(liked.lower()), lowered.find(disliked.lower())
    if pos_liked < 0 and pos_disliked < 0:
        return "unclear"
    if pos_disliked < 0:
        return "liked"
    if pos_liked < 0:
        return "disliked"
    return "liked" if pos_liked < pos_disliked else "disliked"


main()
