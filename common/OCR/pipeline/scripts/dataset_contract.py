"""One record contract for generated and reviewed OCR training examples."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

POLICY = Path(__file__).resolve().parents[1] / "configs/dataset/split-policy.json"


def load_contract(path):
    """校验生成器与训练器共同使用的输入尺寸、标签长度和场景容量。"""
    contract = json.loads(Path(path).read_text())
    for key in ("input_shape", "char_input_shape"):
        shape = contract[key]
        if (len(shape) != 3 or shape[0] != 3
                or any(type(value) is not int or value <= 0 for value in shape)):
            raise ValueError("invalid input shape: " + key)
    for key in ("encoder_max_length", "char_encoder_max_length", "ctc_time_steps", "char_ctc_time_steps"):
        if type(contract[key]) is not int or contract[key] < 1:
            raise ValueError("invalid label capacity: " + key)
    for family in contract["families"].values():
        if family["route"] not in {"WordOCR", "CharOCR"}:
            raise ValueError("unknown family route")
        expected = contract["char_ctc_time_steps" if family["route"] == "CharOCR" else "ctc_time_steps"]
        if family["ctc_time_steps"] != expected:
            raise ValueError("family capacity differs from route capacity")
    return contract


@lru_cache(maxsize=1)
def split_policy():
    """读取固定的文字分区与主题留出规则。"""
    return json.loads(POLICY.read_text())


def split(label):
    """按文字哈希稳定分配训练或开发分区。"""
    bucket = int(hashlib.sha256(label.encode()).hexdigest()[:8], 16) % 10
    for name, assigned in split_policy()["hash_text_split"].items():
        if bucket in assigned:
            return "reserved" if name == "eval" else name
    raise ValueError("split policy does not cover the text hash bucket")


def training_partition(row):
    """要求训练器与审计器对样本的训练／开发分区判断一致。"""
    partition = row.get("text_partition")
    if partition not in {"train", "dev"} or row.get("split") != partition:
        raise ValueError("split and text_partition must match train or dev")
    return partition
