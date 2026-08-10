# Game data

This directory contains external resources used to build OCR datasets. Downloaded
repositories and font files are intentionally excluded from Git.

- `ArknightsGamedata/`: game text source from ArknightsAssets (cn/en/jp/kr/tw/bili)
- `text_renderer/`: synthetic text image generator
- `fonts/`: downloaded Source Han Sans files

Run `scripts/generate_dataset.sh` from any working directory to fetch the
dependencies automatically.
