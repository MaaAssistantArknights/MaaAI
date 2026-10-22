"""Read the exact recruitment vocabulary from the pinned MAA release."""

import json
from functools import lru_cache

from workspace import RELEASE
from routes import CLIENT


@lru_cache(maxsize=5)
def recruitment_tags(language):
    """从固定 MAA 发布资源读取任务实际接受的公招标签。"""
    client = CLIENT[language]
    relative = ("resource/recruitment.json" if client == "Official"
                else f"resource/global/{client}/resource/recruitment.json")
    document = json.loads((RELEASE / relative).read_text())
    return {document["tags"][tag] for operator in document["operators"] for tag in operator["tags"]}
