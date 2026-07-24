# Game data

This directory contains external resources used to build OCR datasets. Downloaded
repositories and font files are intentionally excluded from Git.

- `ArknightsGameData/`: game text source used by the legacy generation script
- `ArknightsGamedata/`: optional current data from ArknightsAssets
- `text_renderer/`: synthetic text image generator
- `fonts/`: downloaded Source Han Sans files

Run `scripts/generate_dataset.sh` from any working directory to fetch the legacy
dependencies automatically.
