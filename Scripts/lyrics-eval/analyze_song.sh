#!/bin/zsh
# analyze_song.sh <song_dir with audio.wav, score.abc, request.json> <lang>
set -e
R=$(git -C ${0:A:h} rev-parse --show-toplevel); K=${0:A:h}; D=${1:A}; LANG_=$2; PY=$R/.venv-karaoke/bin/python
cd $R; source .local-runs/env.sh 2>/dev/null || true
python3 -c "import json;print(json.load(open('$D/request.json'))['lyrics'])" > $D/lyrics.txt
.xcodebuild/Build/Products/Release/yue2 transcribe --audio $D/audio.wav --out $D/transcription >/dev/null 2>&1
cd $K
$PY separate.py $D/audio.wav $D/sep 2>/dev/null | tail -1
$PY ground_truth.py $D/sep/vocals.wav $D/lyrics.txt $D/gt.json 2>&1 | grep -v -i warn | tail -1
$PY mms_only.py $D/audio.wav $D/lyrics.txt $D/mms-mix.json 2>&1 | tail -1
$PY whisper_align.py $D/audio.wav $D/lyrics.txt $D/whisper.json $LANG_ 2>&1 | grep -v -iE "warn|%\|" | tail -1
