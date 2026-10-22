"""Explicit pixel provenance with a narrow reader for three frozen old IDs."""

EVIDENCE_KINDS = {"synthetic", "reviewed_real"}
LEGACY_KINDS = {"final": "synthetic", "v3": "synthetic", "real": "reviewed_real"}


def source_evidence_kind(prefix, declared=None):
    """确定评判来源是真实复核图还是合成图，拒绝未知来源。"""
    legacy = LEGACY_KINDS.get(prefix)
    if declared is None and legacy is None:
        raise ValueError("evidence_kind is required for a new source prefix: " + prefix)
    kind = declared or legacy
    if kind not in EVIDENCE_KINDS or (legacy == "reviewed_real" and kind != legacy):
        raise ValueError("invalid or conflicting evidence_kind: " + prefix)
    return kind


def raw_evidence_kind(row):
    """从单条冻结案例读取并校验来源类别。"""
    identifier = row["id"]
    if not identifier.startswith("raw/"):
        raise ValueError("raw evidence kind requested for non-raw case: " + identifier)
    pieces = identifier.split("/")
    prefix = pieces[1] if len(pieces) > 2 else ""
    try:
        return source_evidence_kind(prefix, row.get("evidence_kind"))
    except ValueError as error:
        if row.get("evidence_kind") is None and prefix not in LEGACY_KINDS:
            raise ValueError("raw case needs evidence_kind: " + identifier) from error
        raise
