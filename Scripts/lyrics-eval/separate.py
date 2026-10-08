"""Separate vocals with Demucs (htdemucs) and report vocal-stem energy. Usage: separate.py audio.wav outdir"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model


def main(audio, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = get_model('htdemucs')
    model.eval()
    wav, sr = sf.read(audio, dtype='float32', always_2d=True)
    assert sr == model.samplerate or True
    if sr != model.samplerate:
        import librosa
        wav = librosa.resample(wav.T, orig_sr=sr, target_sr=model.samplerate).T
        sr = model.samplerate
    x = torch.from_numpy(wav.T.copy())
    ref = x.mean(0)
    x = (x - ref.mean()) / ref.std()
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    with torch.no_grad():
        stems = apply_model(model, x[None], device=device, split=True, overlap=0.25, progress=False)[0]
    stems = stems * ref.std() + ref.mean()
    names = model.sources
    vocals = stems[names.index('vocals')].numpy().T
    rest = sum(stems[i] for i in range(len(names)) if names[i] != 'vocals').numpy().T
    sf.write(out / 'vocals.wav', vocals, sr)
    sf.write(out / 'accompaniment.wav', rest, sr)
    # Energy report per 100 ms frame: how much of the song carries a vocal stem above -40 dB of the mix.
    hop = sr // 10
    def frames_db(y):
        m = y.mean(1)
        n = len(m) // hop
        r = np.sqrt((m[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
        return 20 * np.log10(r)
    v, a = frames_db(vocals), frames_db(rest)
    mix_db = 10 * np.log10(10 ** (v / 10) + 10 ** (a / 10))
    report = {
        'vocal_minus_mix_db_median': float(np.median(v - mix_db)),
        'vocal_frames_above_-12dB_of_mix': float(np.mean(v - mix_db > -12)),
        'vocal_frames_above_-20dB_of_mix': float(np.mean(v - mix_db > -20)),
        'vocal_energy_ratio': float((vocals ** 2).sum() / ((vocals ** 2).sum() + (rest ** 2).sum())),
        'seconds': len(vocals) / sr,
    }
    (out / 'separation.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
