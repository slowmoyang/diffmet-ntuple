#!/usr/bin/env fish
set -xg PROJECT_PREFIX (dirname (readlink -m (status --current-filename)))
set -xga PYTHONPATH {$PROJECT_PREFIX}
micromamba activate diffmet-ntuple-py311
