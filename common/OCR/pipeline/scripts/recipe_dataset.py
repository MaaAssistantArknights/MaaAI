"""Versioned recognition data with visible-text targets, explicit mixtures and replay."""

import argparse
import copy
import io
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from core_training_data import pixel_hash, ref, render, sample_params, write
from file_utils import sha
from label_contract import inspect_label, load_encoder
from multilingual_data import choose_font, coverage, operator_candidates
from PIL import Image
from recipe_settings import DEFAULT_RECIPE, check_profile, load_recipe
from text_targets import partition, transform
from training_preflight import audit_rows
from recruitment_tags import recruitment_tags
from workspace import RELEASE, ROOT, RT
from routes import CLIENT, LOCALES as LANGS
from dataset_contract import load_contract

# Image budgets describe the frozen generated pool, not the eventual number
# of optimizer exposures. The sampling policy separately controls that mix.


def grammar(seed=20260922):
    """以固定种子构造任务场景所需的文字语法样本。"""
    rng = random.Random(seed)
    ascii_chars = (
        "".join((ROOT / "configs/ascii_keys.txt").read_text().splitlines()) + " "
    )
    # Finite, auditable display forms; old number.py formatted some decimals as scientific notation.
    out = {
        "kind": "explicit_recognition_grammar",
        "symbol_replay": [],
        "timer": [],
        "padded_number": [],
        "spaced_ascii": [],
        "ascii_replay": [],
        "battle_cost": [str(i) for i in range(100)],
    }
    for n in range(2000):
        out["symbol_replay"] += [f"★{n}", f"《{n}》", f"{n}Ⅱ", f"{n}II"]
        out["timer"].append(f"{n // 60:02}:{n % 60:02}")
        out["padded_number"].append(f"{n:04}")
        out["spaced_ascii"].append(
            "".join(rng.choices("0123456789ABCDEF", k=2))
            + " "
            + "".join(rng.choices("0123456789ABCDEF", k=3))
        )
        out["ascii_replay"].append(
            "".join(rng.choices(ascii_chars, k=rng.randint(2, 14))).strip()
        )
    for lang, units in {
        "zh-CN": ["万", "亿"],
        "zh-TW": ["萬", "億"],
        "en": ["K", "M"],
        "ja": ["万", "億"],
        "ko": ["만", "억"],
    }.items():
        # Keep both integer and decimal strings; units stay at the displayed-language stage.
        out["quantity_" + lang] = [str(n) for n in range(1000)] + [
            f"{n // 10}.{n % 10}{unit}" for unit in units for n in range(10, 2000)
        ]
    return out


def replay(label, params, recipe):
    """凭标签和逐图参数重新渲染，验证生成图可重放。"""
    im = render(label, params, recipe)
    if params.get("jpeg_quality"):
        f = io.BytesIO()
        im.save(f, format="JPEG", quality=params["jpeg_quality"])
        f.seek(0)
        im = Image.open(f).convert("RGB")
    return im


def build(
    output,
    pilot=False,
    expansion_base=None,
    profile_path=None,
    style_recipe=None,
    corpus_path=None,
    expansion_routes=None,
    recipe_config=DEFAULT_RECIPE,
):
    """按版本配方生成六路线图文数据并审计来源与分区。"""
    output.mkdir(parents=True, exist_ok=False)
    settings = load_recipe(recipe_config)
    word_families = settings["pool_per_scene"]["word"]
    char_families = settings["pool_per_scene"]["char"]
    profile_path = profile_path or ROOT / settings["model_profile"]
    profile = json.loads(profile_path.read_text())
    contract = ROOT / profile["contract"]
    limits = load_contract(contract)
    check_profile(settings, profile)
    recipe = copy.deepcopy(json.loads(style_recipe.read_text()))
    recipe.update(
        version=profile["version"],
        seed=settings["base_seed"],
        profile_sha256=sha(profile_path),
        contract_sha256=sha(contract),
        contract=profile["contract"],
        executable_recipe_id=settings["id"],
        executable_recipe_sha256=sha(recipe_config),
        source_selection="exact game field or explicit grammar; visible-text-v1; old dev/holdout remains reserved",
        training_scope="recipe_validation",
        full_dataset_acceptance=False,
    )
    corpus = [json.loads(s) for s in corpus_path.read_text().splitlines()]
    gp = output / "grammar.json"
    g = grammar(settings["grammar_seed"])
    write(gp, g)
    enc = {}
    meta = {}
    for model, dictionary, length in [
        ("v6", None, limits["encoder_max_length"]),
        ("v5-korean", RT / profile["languages"]["ko"]["dictionary"], limits["encoder_max_length"]),
        ("v6-ascii", None, limits["char_encoder_max_length"]),
    ]:
        enc[model], meta[model] = load_encoder(model, length, dictionary)
    pools = {}
    excluded = []
    fallback = profile["english_whole_run_fallback"]

    def add(key, rows, model, style):
        """将一组候选文字筛选后加入指定路线和场景池。"""
        recipe["styles"][key] = style
        seen = {}
        for row in rows:
            if not isinstance(row.get("label"), str) or not row["label"]:
                excluded.append(
                    {
                        "pool": key,
                        "source": row["source"],
                        "problems": ["empty_source_label"],
                    }
                )
                continue
            item = {**row, **transform(row["label"])}
            font, changed = choose_font(
                item["label"], key.split("/")[0], style["font"], fallback
            )
            literal = key.startswith("char/") and key.split("/")[1] in {
                "ascii_replay",
                "spaced_ascii",
            }
            problems = inspect_label(
                item["label"],
                enc[model],
                limits["char_ctc_time_steps" if model == "v6-ascii" else "ctc_time_steps"],
                literal_ascii=literal,
            )["problems"]
            if any(ord(c) not in coverage(font) for c in item["label"]):
                problems.append("font_missing_glyph")
            if problems:
                excluded.append(
                    {
                        "pool": key,
                        "source": item["source"],
                        "label": item["source_label"],
                        "problems": problems,
                    }
                )
                continue
            # Conservative grouping across canonical-equivalent spellings and old hash partitions.
            item.update(
                font=font,
                font_fallback=changed,
                literal_ascii=literal,
                partition=partition(item["source_label"]),
                source_missing_glyphs=row.get("missing_glyphs", []),
                missing_glyphs=[],
            )
            # If multiple game fields render to the same visible text, retain
            # the stricter partition so the same label cannot leak into train.
            old = seen.get(item["label"])
            if (
                old
                and {"train": 0, "dev": 1, "reserved": 2}[old["partition"]]
                >= {"train": 0, "dev": 1, "reserved": 2}[item["partition"]]
            ):
                continue
            seen[item["label"]] = item
        pools[key] = {
            part: [r for r in seen.values() if r["partition"] == part]
            for part in ["train", "dev", "reserved"]
        }
        if any(not pools[key][part] for part in ["train", "dev"]):
            raise ValueError("empty pool " + key)

    for language in LANGS:
        src = [r for r in corpus if r["language"] == language]
        model = "v5-korean" if language == "ko" else "v6"
        operators, rejected = operator_candidates(src)
        excluded.extend({"pool": language + "/operator_name", **r} for r in rejected)
        for family in word_families:
            key = language + "/" + family
            style = copy.deepcopy(
                recipe["styles"][
                    language
                    + "/"
                    + (
                        "item_name"
                        if family in {"skill_replay", "symbol_replay"}
                        else family
                    )
                ]
            )
            if family == "recruitment":
                rows = [
                    r
                    for r in src
                    if r["scene"] == "recruitment"
                    and r["label"] in recruitment_tags(language)
                ]
            elif family == "item_name":
                rows = [r for r in src if r["scene"] == "inventory"]
            elif family == "operator_name":
                rows = operators
            elif family == "skill_replay":
                rows = [
                    r
                    for r in src
                    if r.get("category") in {"building_skill", "operator_skill"}
                ]
            else:
                name = "quantity_" + language if family == "depot_quantity" else family
                rows = [
                    {"label": s, "source": ref(gp, f"/{name}/{i}")}
                    for i, s in enumerate(g[name])
                ]
                if family == "depot_quantity":
                    style["font"] = src[0][
                        "font"
                    ]  # Existing numeric face does not cover regional units.
            add(key, rows, model, style)
            print(
                json.dumps(
                    {
                        "event": "pool_checked",
                        "pool": key,
                        "labels": sum(map(len, pools[key].values())),
                    }
                ),
                flush=True,
            )
            recipe["styles"][key + "/fallback"] = {**style, "font": fallback}
    stage = RT / "data/game/cn/stage_table.json"
    stage_doc = json.loads(stage.read_text())
    for family in char_families:
        style = copy.deepcopy(
            recipe["styles"]["battle_cost" if family == "battle_cost" else "stage_code"]
        )
        rows = (
            [
                {
                    "label": v["code"],
                    "source": ref(
                        stage,
                        "/stages/" + k.replace("~", "~0").replace("/", "~1") + "/code",
                    ),
                }
                for k, v in stage_doc["stages"].items()
                if v.get("code") and v["code"].isascii()
            ]
            if family == "stage_code"
            else [
                {"label": s, "source": ref(gp, f"/{family}/{i}")}
                for i, s in enumerate(g[family])
            ]
        )
        add("char/" + family, rows, "v6-ascii", style)
    recipe.update(
        encoder_metadata=meta,
        pool_counts={
            k: {p: len(v) for p, v in parts.items()} for k, parts in pools.items()
        },
        excluded_labels=excluded,
        appearance_mixture={"normal_proxy": 0.7, "style_variation": 0.2, "stress": 0.1},
        scope={
            "recognition": "five-language common recognition and ASCII subset; original detector unchanged",
            "units": "released parser accepts localized 万/萬/만, 亿/億/억, K/M; numeric examples are grammar, not claims of exact game frequencies",
            "symbols": "shape learning and compatibility replay, not fictional game item names",
        },
        source_selection_hashes={
            str(p.relative_to(RT)): sha(p)
            for p in [
                RELEASE / "resource/battle_data.json",
                *(
                    RT / "data/assets/word/game" / loc / "building_data.json"
                    for loc in LANGS.values()
                ),
            ]
        },
    )
    # Expansion reuses the base pixels and all development examples. Only add
    # training rows on declared routes, so a scale comparison holds evaluation fixed.
    inherited = []
    if expansion_base:
        expansion_routes = set(expansion_routes or [*LANGS, "char"])
        unknown = expansion_routes - {*LANGS, "char"}
        if unknown:
            raise ValueError("unknown expansion route: " + ",".join(sorted(unknown)))
        if pilot:
            raise ValueError("expansion must use the complete frozen base")
        audit = json.loads((expansion_base / "audit.json").read_text())
        if not audit["passed"] or audit["manifest_sha256"] != sha(
            expansion_base / "manifest.jsonl"
        ):
            raise ValueError("invalid frozen base")
        inherited = [
            json.loads(s)
            for s in (expansion_base / "manifest.jsonl").read_text().splitlines()
        ]
        recipe.update(
            version="recognition-scale-final-v1"
            if expansion_routes != {*LANGS, "char"}
            else "recognition-scale-20k-v1",
            seed=settings["expansion_seed"],
            nested_base={
                "path": str(expansion_base.resolve()),
                "manifest_sha256": sha(expansion_base / "manifest.jsonl"),
            },
            expansion_routes=sorted(expansion_routes),
            expansion="add one original-size training pool only for declared routes; unchanged development images; prefer unseen training labels, same styles and proportions",
        )
    write(output / "recipe.json", recipe)
    recipe_sha = sha(output / "recipe.json")
    rows = []
    pixels = {}
    rng = random.Random(
        settings["expansion_seed"] if expansion_base else settings["base_seed"]
    )
    for original in inherited:
        source = expansion_base / original["image"]
        dest = output / original["image"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if sha(source) != original["sha256"]:
            raise ValueError("inherited image changed")
        os.link(source, dest)
        row = copy.deepcopy(original)
        row.update(
            recipe_sha256=recipe_sha,
            inherited_from={
                "manifest": str((expansion_base / "manifest.jsonl").resolve()),
                "id": original["id"],
            },
        )
        rows.append(row)
        pixels[row["pixel_sha256"]] = row["label"]
    for language in [*LANGS, "char"]:
        families = char_families if language == "char" else word_families
        for part in ["train", "dev"]:
            if expansion_base and language not in expansion_routes:
                continue
            folder = output / language / part
            folder.mkdir(parents=True, exist_ok=bool(expansion_base))
            if expansion_base and part == "dev":
                continue
            for family, n in families.items():
                pool = pools[language + "/" + family][part]
                count = (
                    10
                    if pilot
                    else n
                    if part == "train"
                    else n // settings["development_divisor"]
                )
                if family == "depot_quantity":
                    strata = defaultdict(list)
                    for item in pool:
                        strata[
                            "integer" if item["label"].isdigit() else item["label"][-1]
                        ].append(item)
                    pool = [
                        item
                        for group in __import__("itertools").zip_longest(
                            *strata.values()
                        )
                        for item in group
                        if item is not None
                    ]
                if expansion_base:
                    existing_labels = {
                        r["label"]
                        for r in inherited
                        if r["image"].startswith(f"{language}/{part}/")
                        and r["scene"] == family
                    }
                    pool = sorted(
                        pool, key=lambda r: (r["label"] in existing_labels, r["label"])
                    )
                for i in range(count):
                    src = pool[i % len(pool)] if i < len(pool) else rng.choice(pool)
                    # A bounded retry can vary appearance while keeping the
                    # chosen source text and its exact label fixed.
                    for retry in range(50):
                        tier = (
                            ["normal_proxy"] * 7 + ["style_variation"] * 2 + ["stress"]
                        )
                        tier = tier[i % 10]
                        params = sample_params(family, rng, recipe, part)
                        params["style"] = (
                            language
                            + "/"
                            + family
                            + ("/fallback" if src["font_fallback"] else "")
                        )
                        params["jpeg_quality"] = 0
                        if family == "operator_name":
                            params.update(
                                ink=rng.randint(30, 70),
                                background=rng.randint(210, 240),
                            )
                        if tier == "style_variation":
                            params.update(
                                font_size_delta=rng.choice([-2, 2]),
                                horizontal_scale_delta=rng.choice([-0.05, 0.05]),
                            )
                        if tier == "stress":
                            params.update(capture_scale=0.75, blur=0.3, jpeg_quality=85)
                        im = replay(src["label"], params, recipe)
                        digest = pixel_hash(im)
                        if digest not in pixels:
                            break
                    else:
                        raise ValueError("unique render budget exhausted")
                    pixels[digest] = src["label"]
                    rid = (
                        f"{language}-{part}-"
                        + ("expand-" if expansion_base else "")
                        + f"{family}-{i:05d}"
                    )
                    path = folder / (rid + ".png")
                    im.save(path)
                    row = {k: v for k, v in src.items() if k != "partition"}
                    row.update(
                        id=rid,
                        image=str(path.relative_to(output)),
                        sha256=sha(path),
                        pixel_sha256=digest,
                        language="zh-CN" if language == "char" else language,
                        client=CLIENT[language],
                        scene=family,
                        route="CharOCR" if language == "char" else "WordOCR",
                        split=part,
                        text_partition=part,
                        source_session=None,
                        training_eligible=False,
                        bounded_trial_eligible=True,
                        formal_evaluation_eligible=False,
                        trial_contract_sha256=sha(contract),
                        appearance_status="declared_synthetic",
                        appearance_tier=tier,
                        task_mapping_status="recipe-declared recognition use",
                        ctc_time_steps=limits["char_ctc_time_steps" if language == "char" else "ctc_time_steps"],
                        render=params,
                        recipe_sha256=recipe_sha,
                        leakage_group=(
                            "item-template:"
                            + recipe["templates"][params["template"]]["sha256"]
                        )
                        if params["template"] is not None
                        else "generated-instance:" + digest,
                    )
                    rows.append(row)
            print(
                json.dumps(
                    {
                        "event": "rendered",
                        "language": language,
                        "part": part,
                        "samples": sum(
                            r["image"].startswith(f"{language}/{part}/") for r in rows
                        ),
                    }
                ),
                flush=True,
            )
    mp = output / "manifest.jsonl"
    mp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    for language in [*LANGS, "char"]:
        for part in ["train", "dev"]:
            selected = [
                {**r, "image": str((output / r["image"]).resolve())}
                for r in rows
                if r["image"].startswith(f"{language}/{part}/")
            ]
            (output / language / part / "manifest.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected)
            )
    finalize(output, rows, recipe, enc)


def finalize(output, rows=None, recipe=None, enc=None):
    """复核全量标签、图像和来源后写出清单及审计摘要。"""
    mp = output / "manifest.jsonl"
    recipe = recipe or json.loads((output / "recipe.json").read_text())
    contract = ROOT / recipe["contract"]
    if sha(contract) != recipe["contract_sha256"]:
        raise ValueError("dataset contract changed")
    recipe_sha = sha(output / "recipe.json")
    rows = rows or [json.loads(line) for line in mp.read_text().splitlines()]
    if enc is None:
        enc = {
            m: load_encoder(m, v["configured_max_length"], v["dictionary_path"])[0]
            for m, v in recipe["encoder_metadata"].items()
        }
    pixels = {r["pixel_sha256"] for r in rows}
    result, _ = audit_rows(rows, output, enc, "training_trial", contract)
    write(output / "preflight.json", result)
    if not result["passed"]:
        raise ValueError("data admission failed; inspect preflight")
    # Hashes detect changed files; re-rendering additionally checks that saved
    # parameters actually reproduce the pixels used for training.
    for row in rows:
        if (
            pixel_hash(replay(row["label"], row["render"], recipe))
            != row["pixel_sha256"]
        ):
            raise ValueError("replay mismatch")
    groups = {}
    for r in rows:
        key = (
            ("char" if r["route"] == "CharOCR" else r["language"])
            + "/"
            + r["split"]
            + "/"
            + r["scene"]
        )
        v = groups.setdefault(key, {"images": 0, "labels": set()})
        v["images"] += 1
        v["labels"].add(r["label"])
    result = {
        "passed": True,
        "admission": "bounded_trial_only",
        "manifest_sha256": sha(mp),
        "recipe_sha256": recipe_sha,
        "samples": len(rows),
        "exact_replays": len(rows),
        "unique_pixels": len(pixels),
        "groups": {
            k: {"images": v["images"], "unique_labels": len(v["labels"])}
            for k, v in groups.items()
        },
        "formal_acceptance": False,
        "source_code_sha256": sha(Path(__file__)),
    }
    write(output / "audit.json", result)
    print("passed", len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--profile", type=Path)
    p.add_argument("--recipe-config", type=Path, default=DEFAULT_RECIPE)
    p.add_argument("--style-recipe", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--expansion-base", type=Path)
    p.add_argument("--expansion-route", action="append", choices=[*LANGS, "char"])
    p.add_argument("--audit-only", action="store_true")
    a = p.parse_args()
    finalize(a.output) if a.audit_only else build(
        a.output,
        a.pilot,
        a.expansion_base,
        a.profile,
        a.style_recipe,
        a.corpus,
        a.expansion_route,
        a.recipe_config,
    )
