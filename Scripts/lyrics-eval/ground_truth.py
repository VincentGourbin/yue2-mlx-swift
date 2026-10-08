"""Third-party ground truth on a Demucs vocal stem:
 - pYIN f0 -> sung note segments (onset, offset, midi); held notes stay one segment while the pitch holds
 - torchaudio MMS_FA forced alignment of the known lyrics -> word start/end
Usage: ground_truth.py vocals.wav lyrics.txt out.json"""
import json
import re
import sys
import unicodedata

import librosa
import numpy as np
import torch
import torchaudio
from torchaudio.pipelines import MMS_FA


def pyin_notes(y, sr):
    hop = 256
    f0, voiced, prob = librosa.pyin(y, fmin=80, fmax=1100, sr=sr, frame_length=2048, hop_length=hop)
    t = librosa.times_like(f0, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    loud = rms > (np.percentile(rms, 95) * 10 ** (-30 / 20))
    midi = np.where(voiced & loud, librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), np.nan)
    # Segment: break on unvoiced gaps >= 60 ms or a pitch move > 0.8 semitone held >= 50 ms.
    notes, cur = [], None
    gap = 0
    for i, m in enumerate(midi):
        if np.isnan(m):
            gap += 1
            if cur and gap * hop / sr >= 0.06:
                notes.append(cur)
                cur = None
            continue
        gap = 0
        if cur is None:
            cur = {'i0': i, 'i1': i, 'p': [m]}
            continue
        ref = np.median(cur['p'][-8:])
        if abs(m - ref) > 0.8:
            look = midi[i:i + max(1, int(0.05 * sr / hop))]
            look = look[~np.isnan(look)]
            if len(look) and abs(np.median(look) - ref) > 0.8:
                notes.append(cur)
                cur = {'i0': i, 'i1': i, 'p': [m]}
                continue
        cur['i1'] = i
        cur['p'].append(m)
    if cur:
        notes.append(cur)
    out = []
    for n in notes:
        d = (n['i1'] - n['i0'] + 1) * hop / sr
        if d < 0.08:
            continue
        out.append({'t0': float(t[n['i0']]), 't1': float(t[n['i1']] + hop / sr), 'midi': float(np.median(n['p']))})
    return out, {'voiced_fraction': float(np.mean(~np.isnan(midi)))}


def mms_words(y, sr, lyrics):
    words = []
    for line in lyrics.splitlines():
        s = line.strip()
        if not s or re.fullmatch(r'\[[^\]]+\]', s):
            continue
        for w in s.split():
            plain = ''.join(c for c in unicodedata.normalize('NFKD', w.lower()) if not unicodedata.combining(c))
            norm = re.sub(r"[^a-z']", '', plain.replace('œ', 'oe').replace('æ', 'ae').replace('’', "'"))
            if norm:
                words.append((w, norm))
    wav = torch.from_numpy(librosa.resample(y, orig_sr=sr, target_sr=MMS_FA.sample_rate))[None]
    model = MMS_FA.get_model(with_star=False).eval()
    tok, aligner = MMS_FA.get_tokenizer(), MMS_FA.get_aligner()
    with torch.inference_mode():
        emission, _ = model(wav)
        spans = aligner(emission[0], tok([n for _, n in words]))
    ratio = wav.shape[1] / emission.shape[1] / MMS_FA.sample_rate
    out = []
    for (w, _), sp in zip(words, spans):
        out.append({'text': w, 't0': sp[0].start * ratio, 't1': sp[-1].end * ratio,
                    'score': float(np.mean([s.score for s in sp]))})
    return out


def main(vocals, lyrics_path, out_path):
    y, sr = librosa.load(vocals, sr=22050, mono=True)
    notes, stats = pyin_notes(y, sr)
    words = mms_words(y, sr, open(lyrics_path).read())
    json.dump({'notes': notes, 'words': words, **stats}, open(out_path, 'w'), indent=1)
    print(f'{len(notes)} sung segments, voiced {stats["voiced_fraction"]:.0%}, {len(words)} words aligned')


if __name__ == '__main__':
    main(*sys.argv[1:4])
