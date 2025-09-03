"""
Fetch Arknights game data (step 3) in pure Python.

Downloads the GitHub repository archive and extracts it to --gamedata-dir.
Avoids requiring git on the system.
"""

from __future__ import annotations

import argparse
import io
import shutil
import tempfile
import zipfile
from pathlib import Path

try:
    # Python 3.11+
    from urllib.request import urlopen
except Exception:  # pragma: no cover
    urlopen = None  # type: ignore


ARKNIGHTS_REPO_OWNER = "Kengxxiao"
ARKNIGHTS_REPO_NAME = "ArknightsGameData"
ARKNIGHTS_ZIP_MASTER = (
    f"https://codeload.github.com/{ARKNIGHTS_REPO_OWNER}/{ARKNIGHTS_REPO_NAME}/zip/refs/heads/master"
)
ARKNIGHTS_ZIP_MAIN = (
    f"https://codeload.github.com/{ARKNIGHTS_REPO_OWNER}/{ARKNIGHTS_REPO_NAME}/zip/refs/heads/main"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract Arknights game data")
    parser.add_argument(
        "--gamedata-dir",
        type=Path,
        default=Path("ArknightsGameData"),
        help="Directory for Arknights game data",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-download game data even if directory exists",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Network timeout in seconds",
    )
    return parser.parse_args()


def download_to_bytes(url: str, timeout: int) -> bytes:
    if urlopen is None:
        raise RuntimeError("urllib not available for downloads")
    with urlopen(url, timeout=timeout) as resp:  # type: ignore[arg-type]
        return resp.read()


def extract_zip_bytes(data: bytes, dest_dir: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest_dir)


def replace_dir_with_extracted(extracted_root: Path, final_dir: Path) -> None:
    entries = list(extracted_root.iterdir())
    if len(entries) != 1 or not entries[0].is_dir():
        raise RuntimeError("Unexpected archive structure for ArknightsGameData")
    source_dir = entries[0]
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.move(str(source_dir), str(final_dir))


def main() -> None:
    args = parse_args()
    if args.gamedata_dir.exists() and not args.refresh:
        print(f"ArknightsGameData already present at: {args.gamedata_dir}")
        return

    print("Downloading ArknightsGameData archive...")
    data = None
    errors: list[str] = []
    for url in (ARKNIGHTS_ZIP_MASTER, ARKNIGHTS_ZIP_MAIN):
        try:
            data = download_to_bytes(url, timeout=args.timeout)
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url} -> {exc}")
    if data is None:
        raise RuntimeError("Failed to download ArknightsGameData. Tried: \n" + "\n".join(errors))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        extract_zip_bytes(data, tmp_path)
        replace_dir_with_extracted(tmp_path, args.gamedata_dir)

    print(f"ArknightsGameData ready at: {args.gamedata_dir}")


if __name__ == "__main__":
    main()


