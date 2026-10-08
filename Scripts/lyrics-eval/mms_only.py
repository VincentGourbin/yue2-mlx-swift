"""MMS_FA word alignment on any audio (mix or stem). Usage: mms_only.py audio.wav lyrics.txt out.json"""
import json
import sys

import librosa
from ground_truth import mms_words

y, sr = librosa.load(sys.argv[1], sr=22050, mono=True)
words = mms_words(y, sr, open(sys.argv[2]).read())
json.dump({'words': words}, open(sys.argv[3], 'w'), indent=1)
print(len(words), 'words')
