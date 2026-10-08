"""Voice-likeness of a Demucs vocal stem: share of loud voiced time, and pitch instability within held notes.
A piano holds its pitch within a few cents; a singing voice wanders/vibrates (tens of cents).
Usage: vibrato.py vocals.wav [...]"""
import sys

import librosa
import numpy as np

for path in sys.argv[1:]:
    y, sr = librosa.load(path, sr=22050, mono=True)
    hop = 256
    f0, voiced, _ = librosa.pyin(y, fmin=80, fmax=1100, sr=sr, frame_length=2048, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    loud = rms > np.percentile(rms, 99) * 10 ** (-25 / 20)
    m = np.where(voiced & loud, librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), np.nan)
    # within-note wobble: std of (pitch - 150 ms running median) over runs >= 250 ms
    wob, run = [], []
    k = int(0.15 * sr / hop) | 1
    for v in list(m) + [np.nan]:
        if np.isnan(v):
            if len(run) * hop / sr >= 0.25:
                r = np.array(run)
                med = np.array([np.median(r[max(0, i - k // 2):i + k // 2 + 1]) for i in range(len(r))])
                wob.append(np.std(r - med) * 100)
            run = []
        else:
            run.append(v)
    wob = np.array(wob)
    print(f"{path.split('karaoke-probe.noindex/')[-1]:40s} loud-voiced {np.mean(~np.isnan(m)):5.1%}  "
          f"held runs {len(wob):3d}  wobble median {np.median(wob) if len(wob) else 0:5.1f} cents  "
          f"runs>12c {np.mean(wob > 12) if len(wob) else 0:4.0%}")
