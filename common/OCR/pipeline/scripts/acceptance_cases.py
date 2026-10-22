"""Compare original task behavior against explicit truth, preserving shared failures."""

import hashlib
import json
from collections import Counter


def canonical(row):
    """只保留影响任务结果的字段，供业务变化比较和摘要哈希使用。"""
    # Preserve shop order, which determines purchasing priority. Scores are diagnostic.
    result = {k: v for k, v in row.items() if k not in ["results", "drops"]}
    if "results" in row:
        result["results"] = [
            {k: v for k, v in r.items() if k != "score"} for r in row["results"]
        ]
        if row["mode"] in ["recruit", "depot"]:
            result["results"] = sorted(
                result["results"], key=lambda r: json.dumps(r, sort_keys=True)
            )
    if "drops" in row:
        result["drops"] = sorted(
            row["drops"], key=lambda r: json.dumps(r, sort_keys=True)
        )
    return result


def business_digest(row):
    """将人工复核的例外绑定到候选业务输出的精确哈希。"""
    data = json.dumps(canonical(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def check_truth(case, pred):
    """按人工真值逐项检查分析结果，不把双方同错算作通过。"""
    truth = case["truth"]
    checks = []
    for key in ["stage_code", "difficulty", "stars", "times", "analyze_ok"]:
        if key in truth:
            checks.append(
                {
                    "field": key,
                    "expected": truth[key],
                    "actual": pred.get(key),
                    "match": pred.get(key) == truth[key],
                }
            )
    if "texts" in truth:
        texts = [r["text"] for r in pred.get("results", []) if "text" in r]
        match = (
            texts == truth["texts"]
            if truth.get("ordered")
            else Counter(texts) == Counter(truth["texts"])
        )
        checks.append(
            {
                "field": "texts",
                "expected": truth["texts"],
                "actual": texts,
                "match": match,
            }
        )
        if not truth["texts"] and pred.get("results"):
            checks[-1]["match"] = False
    # Bare quantities check only a multiset. Use boxed fields (or item-bound
    # drops) when correctness depends on which item received each quantity.
    if "quantities" in truth:
        quantities = [
            str(r["quantity"]) for r in pred.get("results", []) if "quantity" in r
        ]
        expected = [str(value) for value in truth["quantities"]]
        checks.append(
            {
                "field": "quantities",
                "expected": expected,
                "actual": quantities,
                "match": Counter(quantities) == Counter(expected),
            }
        )
    if "drops" in truth:
        key = lambda r: (r.get("item_id"), str(r.get("quantity")), r.get("type"))
        expected = Counter(key(r) for r in truth["drops"])
        actual = Counter(key(r) for r in pred.get("drops", []))
        checks.append(
            {
                "field": "drops",
                "expected": truth["drops"],
                "actual": pred.get("drops", []),
                "match": actual == expected,
            }
        )
    for field in truth.get("fields", []):
        x, y, w, h = field["box_xywh"]
        cx = x + w / 2
        cy = y + h / 2
        matched = [
            r
            for r in pred.get("results", [])
            if r["rect"][0] <= cx < r["rect"][0] + r["rect"][2]
            and r["rect"][1] <= cy < r["rect"][1] + r["rect"][3]
        ]
        actual = [str(r["quantity"]) for r in matched]
        checks.append(
            {
                "field": field["id"],
                "expected": [field["label"]],
                "actual": actual,
                "match": actual == [field["label"]],
                "known_baseline_omission": field["id"]
                in case.get("known_omissions", []),
            }
        )
    return checks


def compare_cases(cases, official, candidate):
    """在相同案例上比较官方版与候选版的业务结果和分析状态。"""
    def index(rows):
        """按案例 ID 建立索引，同时拒绝重复预测。"""
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate predictions")
        return {r["id"]: r for r in rows}

    base = index(official)
    new = index(candidate)
    ids = {r["id"] for r in cases}
    if set(base) != ids or set(new) != ids:
        raise ValueError("prediction ids differ from frozen cases")
    if not cases or len(ids) != len(cases):
        raise ValueError("empty or duplicate cases")
    result = []
    for c in cases:
        a = base[c["id"]]
        b = new[c["id"]]
        if a.get("mode") != c["mode"] or b.get("mode") != c["mode"]:
            raise ValueError("task mode differs")
        old = check_truth(c, a)
        fresh = check_truth(c, b)
        if not old:
            raise ValueError("case has no asserted truth")
        regressions = [
            y["field"] for x, y in zip(old, fresh) if x["match"] and not y["match"]
        ]
        # The analyzer can fail while still returning partially populated text.
        # Preserve that business failure even when truth asserts only the text.
        status_regression = (
            a.get("analyze_ok") is True and b.get("analyze_ok") is not True
        )
        changed = canonical(a) != canonical(b)
        approval = c.get("approved_business_change")
        reviewed = (
            changed
            and isinstance(approval, dict)
            and isinstance(approval.get("reason"), str)
            and bool(approval["reason"].strip())
            and approval.get("candidate_canonical_sha256") == business_digest(b)
        )
        result.append(
            {
                "id": c["id"],
                "route": c["route"],
                "mode": c["mode"],
                "source_scope": c["source_scope"],
                "official_checks": old,
                "candidate_checks": fresh,
                "new_truth_regressions": regressions,
                "new_analyzer_failure": status_regression,
                "whole_output_equal": not changed,
                "business_change_reviewed": reviewed,
                "candidate_canonical_sha256": business_digest(b),
                "stale_business_change_approval": approval is not None and not changed,
                "candidate_all_labeled_fields_correct": all(x["match"] for x in fresh),
            }
        )
    return {
        "cases": result,
        "new_truth_regressions": [
            r["id"] for r in result if r["new_truth_regressions"]
        ],
        "new_analyzer_failures": [r["id"] for r in result if r["new_analyzer_failure"]],
        "changed_business_outputs": [
            r["id"] for r in result if not r["whole_output_equal"]
        ],
        "unreviewed_business_changes": [
            r["id"] for r in result
            if (not r["whole_output_equal"] and not r["business_change_reviewed"])
            or r["stale_business_change_approval"]
        ],
        "formal_acceptance": False,
    }
