"""Code and external assets have separate roots; no machine-specific defaults."""
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
value = os.environ.get("MAA_OCR_RUNTIME")
if not value:
    raise RuntimeError("Set MAA_OCR_RUNTIME or use pipeline.py --runtime PATH")
RT = Path(value).expanduser().resolve()
if not RT.is_dir():
    raise RuntimeError("Runtime directory does not exist: " + str(RT))
SOURCES_LOCK = json.loads((ROOT / "configs/sources.lock.json").read_text())
RELEASE_TAG = SOURCES_LOCK["release"]["tag"]
RELEASE = RT / "releases" / RELEASE_TAG
