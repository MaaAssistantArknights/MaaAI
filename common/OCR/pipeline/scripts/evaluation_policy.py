"""Pure release-quality decision for an already collected evaluation summary."""


def evaluation_failed(cfg, summary):
    """保留速度记录，并按评估用途执行已测质量门槛。"""
    policy = cfg.get("timing_policy", "record_only")
    if policy not in {"record_only", "enforce"}:
        raise ValueError("unknown timing policy")
    if cfg["purpose"] != "evaluation":
        return False
    frames = summary.get("frames") or {}
    return bool(
        (policy == "enforce" and summary["failed_timing_routes"])
        or frames.get("new_truth_regressions")
        or frames.get("new_analyzer_failures")
        or frames.get("unreviewed_business_changes")
        or any(
            not result["quality_eligible"]
            for result in summary["recognition"].values()
        )
    )
