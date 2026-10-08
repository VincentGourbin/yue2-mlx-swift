#!/bin/zsh
# Third-party tools used to measure karaoke timing and instrumental voice leaks (not shipped):
# Demucs (htdemucs) vocal separation, torchaudio MMS_FA forced alignment, Whisper via stable-ts,
# librosa pYIN, CMU dict syllables. Creates .venv-karaoke (gitignored). See Scripts/lyrics-eval/README.md.
set -e
cd "$(git -C "${0:A:h}" rev-parse --show-toplevel)"
uv venv -q --python 3.12 .venv-karaoke
uv pip install -q --python .venv-karaoke/bin/python torch torchaudio demucs librosa soundfile "numpy<2.3" pronouncing stable-ts
echo "ready: .venv-karaoke"
