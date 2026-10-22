"""Validate the versioned recipe used by the generator and formal plan."""

import json
from pathlib import Path
from routes import CLIENT

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE = ROOT / "configs/dataset/recipe.json"
ROUTES = {"zh-CN", "zh-TW", "en", "ja", "ko", "char"}


def load_recipe(path=DEFAULT_RECIPE):
    """检查版本配方的路线、数量和采样策略完整性。"""
    path = Path(path).resolve()
    recipe = json.loads(path.read_text())
    if (
        recipe.get("status") != "executable_recipe"
        or set(recipe.get("routes", {})) != ROUTES
    ):
        raise ValueError("unknown or incomplete executable OCR recipe")
    scenes = recipe["pool_per_scene"]
    word_total = sum(scenes["word"].values())
    char_total = sum(scenes["char"].values())
    if not word_total or not char_total:
        raise ValueError("empty route pool in executable recipe")
    if any(
        not isinstance(n, int) or n <= 0 or n % recipe["development_divisor"]
        for pool in scenes.values()
        for n in pool.values()
    ):
        raise ValueError("recipe scene counts must divide into the development pool")
    policy_path = (ROOT / recipe["sampling_policy"]).resolve()
    if not policy_path.is_relative_to(ROOT.resolve()) or not policy_path.is_file():
        raise ValueError("recipe sampling policy must be an existing repository file")
    if {recipe["routes"][route]["sampling"] for route in ROUTES} - {
        "historical",
        "balanced",
    }:
        raise ValueError("unknown route sampling strategy")
    for route in ROUTES:
        expected = char_total if route == "char" else word_total
        if recipe["routes"][route]["base_train_images"] != expected:
            raise ValueError("route pool count differs from executable scene counts")
        formal = recipe["routes"][route].get("formal_train_images", expected)
        if formal != (2 * expected if route == "char" else expected):
            raise ValueError("formal pool count differs from declared expansion")
    return recipe


def check_profile(recipe, profile):
    """核对数据配方与模型配置选用同一模型路线。"""
    if set(profile) != {"version", "contract", "languages", "english_whole_run_fallback", "char"}:
        raise ValueError("unknown or missing training profile fields")
    if set(profile["languages"]) != set(CLIENT) - {"char"}:
        raise ValueError("training profile must declare all five languages")
    for route in ROUTES:
        selected = profile["char"] if route == "char" else profile["languages"][route]
        if not set(selected) <= {"model", "dictionary"}:
            raise ValueError("unknown model route fields: " + route)
        if selected["model"] != recipe["routes"][route]["model"]:
            raise ValueError("recipe model and encoder profile differ: " + route)


def check_plan_recipe(recipe, route, sampling, rows, *, formal):
    """检查正式计划的路线数量与曝光策略符合配方。"""
    if formal and sampling != recipe["routes"][route]["sampling"]:
        raise ValueError("formal sampling differs from versioned recipe")
    if not formal:
        return
    selected = [
        r
        for r in rows
        if (
            r["route"] == "CharOCR"
            if route == "char"
            else r["route"] == "WordOCR" and r["language"] == route
        )
    ]
    train = [r for r in selected if r["split"] == "train"]
    expected = recipe["routes"][route].get(
        "formal_train_images", recipe["routes"][route]["base_train_images"]
    )
    # The reviewed Char 0/O repair is an explicit additive dataset, not a
    # different base pool or an implicit permission to change other routes.
    approved_char_addition = (
        route == "char"
        and len(train) > expected
        and sum(r.get("appearance_status") == "verified_real_crop" for r in train)
        == len(train) - expected
    )
    if len(train) != expected and not approved_char_addition:
        raise ValueError("formal training pool count differs from recipe")
