#!/bin/zsh
# leak.sh <run dir>... : SheetSage2 vocal-track notes + Demucs vocal energy ratio
R=$(git -C ${0:A:h} rev-parse --show-toplevel); K=${0:A:h}; cd $R; source .local-runs/env.sh 2>/dev/null
for d in "$@"; do
  [ -f $d/transcription/events.json ] || .xcodebuild/Build/Products/Release/yue2 transcribe --audio $d/audio.wav --out $d/transcription >/dev/null 2>&1
  [ -f $d/sep/separation.json ] || .venv-karaoke/bin/python -I $K/separate.py $d/audio.wav $d/sep >/dev/null 2>&1
  python3 -c "
import json,collections
e=json.load(open('$d/transcription/events.json')); v=[x['time'] for x in e for m in x.get('melody',[]) if m['track']==0]
s=json.load(open('$d/sep/separation.json')); b=collections.Counter(int(t//5) for t in v)
print(f'{\"$d\".split(\"/\")[-1]:22s} SheetSage2 vocal notes {len(v):3d} | Demucs vocal energy {s[\"vocal_energy_ratio\"]:.2f} | vocal notes per 5 s:', ' '.join(str(b.get(k,0)) for k in range(13)))"
done
