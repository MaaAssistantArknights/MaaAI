## PP-OCRv5 Training Pipeline (Arknights Data) — Scratch

This is a minimal scaffold describing how to prepare data and train a PP-OCRv5 recognition model using Arknights data. It mirrors the early steps from the v3 script, but intentionally stops before cloning `text_renderer` (data rendering will be documented later).

### Overview
- **Goal**: Train a PP-OCRv5 recognition model for Arknights UIs/text.
- **High-level pipeline**:
  1. Prepare environment and working directory
  2. Configure generation parameters (images count, locale, font locale)
  3. Fetch Arknights game data repo
  4. Prepare fonts (Source Han Sans) for the chosen locale
  5. (Later) Fetch pretrained models
  6. (Later) Prepare wording/keys and numeric corpora
  7. (Later) Clone text_renderer and its requirements
  8. (Later) Render synthetic text images
  9. (Later) Split train/val and rename to PP-OCR format
  10. (Later) Train PP-OCRv5 model

This document covers steps 1–4 only. Use the provided Python scripts for steps 3 and 4.

### Prerequisites
- Python 3.8+ (recommend a virtual environment)
- Git
- A shell environment (PowerShell, Git Bash, or WSL on Windows)
- Optional but recommended: a proxy for faster downloads in mainland China

### 1) Prepare workspace
From the project root, switch to this directory:

```bash
cd common/pp-OCRv5
```

Create and activate a virtual environment (example with `venv`):

```bash
python -m venv .venv
# Windows PowerShell
. .venv\\Scripts\\Activate.ps1
# or Git Bash / WSL
source .venv/bin/activate
```

### 2) Configure parameters
Define how many images you plan to generate later and which client/locale to target. These are only variables for now; the actual rendering steps will come later.

```bash
# total images to generate (used later)
NUM_IMG=200000

# Which language to generate: zh_CN | zh_TW | ja_JP | ko_KR
CLIENT="zh_CN"

# Font language pack to use: CN | TW | JP | KR
FONT_LANG="CN"

echo "num_img: $NUM_IMG, client: $CLIENT, fontLang: $FONT_LANG"
```

Notes:
- Keep `CLIENT` and `FONT_LANG` consistent (e.g., `zh_CN` with `CN`).
- You can adjust these later before rendering.

### 3) Fetch Arknights game data (Python)
Download Arknights game data (via GitHub zip) without requiring git:

```bash
# from common/pp-OCRv5
python fetch_gamedata.py \
  --gamedata-dir ArknightsGameData \
  --timeout 60
# add --refresh to force re-download
```

### 4) Prepare fonts (Python)
Download Source Han Sans and index `fonts.txt` for the selected `FONT_LANG`:

```bash
# from common/pp-OCRv5
python prepare_fonts.py \
  --font-lang "$FONT_LANG" \
  --fonts-dir fonts \
  --timeout 60
```

Outputs:
- `ArknightsGameData/` populated with the repo contents
- `fonts/` with unzipped Source Han Sans and `fonts/fonts.txt`

At this point, stop here. Subsequent steps (cloning `text_renderer`, fetching pretrained models, generating corpora, rendering images, splitting/renaming, and training) will be added in the next iteration of this README.

### Next (not covered here)
- Clone `text_renderer` and install its requirements
- Download pretrained models for PP-OCRv5
- Prepare wording/number corpora using helper scripts
- Render short/long/number images, split train/val, and rename for PP-OCR
- Launch PP-OCRv5 training


