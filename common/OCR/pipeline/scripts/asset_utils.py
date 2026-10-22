"""Hash-checked source assets and Unicode font coverage."""

from functools import lru_cache

from file_utils import sha
from fontTools.ttLib import TTFont
from workspace import RT


def asset_path(asset, runtime=RT):
    """验证素材位于运行目录且哈希匹配，返回可读取路径。"""
    path = (runtime / asset["path"]).resolve()
    if not path.is_relative_to(runtime.resolve()):
        raise ValueError("asset escapes runtime")
    if sha(path) != asset["sha256"]:
        raise ValueError(f"asset hash mismatch: {path}")
    return path


@lru_cache(maxsize=32)
def cmap(path):
    """读取并缓存字体支持的字符集合。"""
    with TTFont(path) as font:
        return frozenset(font.getBestCmap() or {})
