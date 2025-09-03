"""
Prepare fonts (step 4) in pure Python.

Downloads Source Han Sans based on --font-lang, unzips into --fonts-dir,
and writes a fonts list file (fonts.txt) for later rendering steps.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

try:
    # Python 3.11+
    from urllib.request import urlopen
except Exception:  # pragma: no cover
    urlopen = None  # type: ignore


SOURCE_HAN_SANS_BASE = (
    "https://github.com/adobe-fonts/source-han-sans/releases/download/2.004R/SourceHanSans{lang}.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and index Source Han Sans fonts")
    parser.add_argument(
        "--font-lang",
        type=str,
        default="CN",
        choices=["CN", "TW", "JP", "KR"],
        help="Font language pack",
    )
    parser.add_argument(
        "--fonts-dir",
        type=Path,
        default=Path("fonts"),
        help="Directory to store fonts",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Network timeout in seconds",
    )
    return parser.parse_args()


def download_to_file(url: str, out_path: Path, timeout: int) -> None:
    if urlopen is None:
        raise RuntimeError("urllib not available for downloads")
    with urlopen(url, timeout=timeout) as resp:  # type: ignore[arg-type]
        out_path.write_bytes(resp.read())


def ensure_fonts(font_lang: str, fonts_dir: Path, timeout: int) -> Path:
    fonts_dir.mkdir(parents=True, exist_ok=True)
    url = SOURCE_HAN_SANS_BASE.format(lang=font_lang)
    zip_path = fonts_dir / f"SourceHanSans{font_lang}.zip"

    if not zip_path.exists():
        print(f"Downloading fonts: {url}")
        download_to_file(url, zip_path, timeout)
    else:
        print(f"Fonts archive already exists: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(fonts_dir)
    print(f"Fonts extracted into: {fonts_dir}")

    subset_dir = fonts_dir / "SubsetOTF" / font_lang
    if not subset_dir.exists():
        raise RuntimeError(f"Expected fonts directory not found: {subset_dir}")

    fonts_list = fonts_dir / "fonts.txt"
    files = sorted(p for p in subset_dir.iterdir() if p.is_file())
    with fonts_list.open("w", encoding="utf-8") as f:
        for p in files:
            f.write(str(p.resolve()) + "\n")
    print(f"Indexed {len(files)} fonts -> {fonts_list}")
    return fonts_list


def main() -> None:
    args = parse_args()
    ensure_fonts(args.font_lang, args.fonts_dir, args.timeout)


if __name__ == "__main__":
    main()


