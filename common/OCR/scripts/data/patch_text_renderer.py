"""给 text_renderer 打 Pillow 10+ 兼容补丁。

Pillow 10 移除了 FreeTypeFont.getoffset / getsize，text_renderer（2021 年）仍在
使用，会导致渲染时抛异常并被 tenacity 无限重试。用 getbbox 等价替换。
"""
import argparse
from pathlib import Path

PATCHES = [
    (
        "        offset = font.getoffset(word)\n"
        "        size = font.getsize(word)\n"
        "        size = (size[0] - offset[0], size[1] - offset[1])\n"
        "        return size",
        "        # Pillow 10+ removed getoffset/getsize, use getbbox instead\n"
        "        # 标点等字符可能产生负尺寸，钳制到至少 1 避免生成崩溃\n"
        "        bbox = font.getbbox(word)\n"
        "        size = (max(bbox[2] - 2 * bbox[0], 1), max(bbox[3] - 2 * bbox[1], 1))\n"
        "        return size",
    ),
    (
        "        offset = font.getoffset(word)\n",
        "        bbox = font.getbbox(word)\n"
        "        offset = (bbox[0], bbox[1])\n",
    ),
    (
        "            size = font.getsize(c)\n",
        "            bbox = font.getbbox(c)\n"
        "            size = (max(bbox[2] - bbox[0], 1), max(bbox[3] - bbox[1], 1))\n",
    ),
    (
        "            c_offset = font.getoffset(c)\n",
        "            c_offset = (bbox[0], bbox[1])\n",
    ),
]

MARKER = "# Pillow 10+ removed getoffset/getsize, use getbbox instead"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "renderer",
        type=Path,
        nargs="?",
        default=Path("game_data/text_renderer/textrenderer/renderer.py"),
    )
    args = parser.parse_args()

    src = args.renderer.read_text(encoding="utf-8")
    if MARKER in src:
        print(f"{args.renderer}: already patched")
        return

    for old, new in PATCHES:
        if old not in src:
            raise SystemExit(f"{args.renderer}: patch target not found: {old!r}")
        src = src.replace(old, new, 1)

    args.renderer.write_text(src, encoding="utf-8")
    print(f"{args.renderer}: patched")


if __name__ == "__main__":
    main()
