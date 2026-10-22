import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

# Pipeline imports require an external runtime, even pure contract tests.
# Default to an empty temporary root; do not depend on the author's workstation.
import atexit,os,tempfile
_test_runtime = tempfile.TemporaryDirectory(prefix="maa-ocr-tests-")
atexit.register(_test_runtime.cleanup)
os.environ.setdefault("MAA_OCR_RUNTIME", _test_runtime.name)
sys.path.insert(0, str(ROOT / "pipeline/scripts"))
