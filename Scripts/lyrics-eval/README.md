# Lyrics timing and instrumental leak evaluation

Measurement tools behind `yue2 karaoke` and `yue2 generate --instrumental` (2026-10-07). They use
third-party models that are **not** shipped with the engine; set them up with
`Scripts/setup-lyrics-eval-env.sh` (creates `.venv-karaoke`, gitignored).

| Script | What it measures |
|---|---|
| `separate.py audio.wav outdir` | Demucs `htdemucs` vocal stem + vocal energy ratio (≈ 0.00 on a clean instrumental, 0.24+ when a voice leaks) |
| `ground_truth.py vocals.wav lyrics.txt out.json` | pYIN sung segments + MMS_FA (torchaudio) word alignment of the known lyrics |
| `mms_only.py audio.wav lyrics.txt out.json` | MMS_FA on the mix (matches the stem within 100 ms on 98 % of words) |
| `whisper_align.py audio lyrics.txt out.json [lang]` | independent Whisper (stable-ts) alignment, coarser than MMS |
| `karaoke.py` + `yabc.py` | Python reference of `LyricTimeline` (CMU dict syllables) |
| `eval_modes.py score.abc lyrics.txt events.json ref.json name=anchors.json…` | word-start error of score-only / score+anchor modes against a reference |
| `adherence.py score.abc run_dir…` | share of planned notes SheetSage2 hears in a render |
| `vibrato.py vocals.wav…` | pitch wobble of held runs (voice ≈ 13 cents, piano ≈ 7) |
| `analyze_song.sh song_dir en\|fr` | all of the above for one `yue2 generate --out` directory |
| `leak.sh run_dir…` | SheetSage2 sung-note count + Demucs vocal energy per render |
| `segments.py song_dir tokenizer.json out_dir` | prompt token → score bar / lyric word map, for `yue2 attention-probe` (hidden command: per-head attention mass on those segments per semantic frame) |
| `attention_eval.py mass.npy segments.json timeline.json mms.json` | how well the alignment heads track bars (vs a SheetSage2 timeline) and words (vs MMS) |
| `head_anchors.py`, `head_grid.py` | Python reference of `AttentionTimeline`: word starts and bar clock read from the heads |

Tools found misleading on YuE2 renders: the AudioSet AST classifier (0.03 "singing" on a sung song)
and Essentia `voice_instrumental` (calls a piano lead "voice"). Results and method:
`docs/knowledge/benchmarks/instrumental-and-karaoke-2026-10-07.md`.
