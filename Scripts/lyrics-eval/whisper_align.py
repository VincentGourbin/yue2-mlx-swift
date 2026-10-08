"""Independent word alignment with Whisper (stable-ts forced alignment). Usage: whisper_align.py audio lyrics.txt out.json [lang]"""
import json
import re
import sys

import stable_whisper

audio, lyrics_path, out = sys.argv[1:4]
lang = sys.argv[4] if len(sys.argv) > 4 else 'en'
text = ' '.join(l.strip() for l in open(lyrics_path).read().splitlines()
                if l.strip() and not re.fullmatch(r'\[[^\]]+\]', l.strip()))
model = stable_whisper.load_model('medium', device='cpu')
res = model.align(audio, text, language=lang)
words = [{'text': w.word.strip(), 't0': w.start, 't1': w.end} for s in res.segments for w in s.words if w.word.strip()]
json.dump({'words': words}, open(out, 'w'), indent=1)
print(len(words), 'words')
