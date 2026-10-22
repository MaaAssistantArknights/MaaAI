"""Reproducible exposure budgets: preserve historical length mix, balance within it."""

import json
import numpy as np
from file_utils import sha
from workspace import ROOT
from collections import Counter, defaultdict

def bucket(row):
    """按文字长度或数字类别归入已定义的曝光桶。"""
    return (
        "number"
        if row["scene"] == "depot_quantity"
        else "short"
        if len(row["label"]) < 7
        else "long"
    )


def schedule(rows, kind, strategy, count, seed=20260922, policy_path=None):
    """按固定策略和种子生成训练呈现顺序及曝光统计。"""
    if strategy not in {"historical", "balanced"}:
        raise ValueError("unknown recipe")
    config = policy_path or ROOT / "configs/sampling.json"
    policy = json.loads(config.read_text())
    scene_sampling = kind == "char" or strategy == "balanced"
    weights = policy[
        ("historical_char" if strategy == "historical" else "candidate_char")
        if kind == "char"
        else (
            "historical_word" if strategy == "historical" else "candidate_word_scenes"
        )
    ]
    pools = defaultdict(list)
    for i, row in enumerate(rows):
        pools[row["scene"] if scene_sampling else bucket(row)].append(i)
    if any(not pools[k] for k in weights):
        raise ValueError("required sampling bucket empty")
    strata = {
        key: {
            family: [i for i in pool if rows[i]["scene"] == family]
            for family in sorted({rows[i]["scene"] for i in pool})
        }
        for key, pool in pools.items()
    }
    rng = np.random.default_rng(seed)
    buckets = []
    used = Counter()
    # Largest-deficit apportionment preserves each share to within one sample.
    for n in range(count):
        key = max(weights, key=lambda k: (n + 1) * weights[k] - used[k])
        buckets.append(key)
        used[key] += 1
    rng.shuffle(buckets)
    indices = []
    for key in buckets:
        pool = pools[key]
        if kind == "word" and strategy == "balanced":
            families = list(strata[key])
            family = families[int(rng.integers(len(families)))]
            pool = strata[key][family]
        indices.append(pool[int(rng.integers(len(pool)))])
    # Sampling is with replacement: stored image count is not training exposure.
    # Record both draws and unique images to interpret data-scale experiments.
    selected = [rows[i] for i in indices]
    return np.asarray(indices, dtype="int64"), {
        "strategy": strategy,
        "policy_sha256": sha(config),
        "seed": seed,
        "presentations": count,
        "buckets": dict(used),
        "families": dict(Counter(r["scene"] for r in selected)),
        "special_character_presentations": {
            c: sum(c in r["label"] for r in selected) for c in "★《》Ⅱ0O :"
        },
        "unique_images": len(set(indices)),
    }
