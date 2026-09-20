# Benchmarks

Mesures reproductibles du port YuE2 → MLX Swift sur Apple Silicon.

## Comment lire ce fichier

Source unique des chiffres : `swift-mlx-profiler` (`ProfilingSession.generateReport()`,
`MLXProfiler.shared.getLLMMetrics()`/`getTTSMetrics()`), jamais recalculés ou chronométrés
à la main (plan §0.3/§3.3). Une ligne = un run de `yue2 generate --profile` (ou `yue2 profile
plan|semantic` pour une mesure Jalon 2 partielle, sans NAR/VAE). `NAR s`/`VAE s`/`Audio s`/`RTF`
restent vides pour une mesure LM seule.

## Comment contribuer une ligne

1. Reproduis avec la commande citée dans l'entrée `tasks/JOURNAL.md` de la fiche correspondante
   (build Release, GPU refroidi — cf. `CLAUDE.md` § Performance work).
2. Ajoute une ligne au tableau : date, SHA court du commit, quantification de la voie AR, `cot`,
   débits et durées copiés tels quels de `TTSMetrics.summary`/`generateReport()`, chemin du
   `trace.json` produit (`--out <dir>`, jamais committé — gitignoré sous `.local-runs/`).
3. Ne modifie jamais une ligne existante : une nouvelle mesure est une nouvelle ligne.

| Date | Commit | Quant AR | cot | Tokens ABC | Tokens sém. | ABC tok/s | Sém. tok/s | NAR s | VAE s | Audio s | RTF | Pic mémoire | Trace |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-18 | HEAD (ce commit, T-2.11) | none (bf16) | full | 572 | 300 | 59.6 | 72.0 | — | — | — | — | 8447 MB (process), 7446 MB (MLX active) | `.local-runs/prof/trace.json` |
| 2026-09-18 | HEAD (ce commit, T-3.7) | none (bf16) | full | 64 (tronqué) | 100 (tronqué) | 25.4 | 61.6 | 1.3 | 0.4 | 4.0 | 1.20 | 9968 MB (process), 9369 MB (MLX active) | `.local-runs/p1/trace.json` |
| 2026-09-18 | HEAD (ce commit, T-3.7) | none (bf16) | full | 572 | 1574 | 65.0 | 67.1 | 51.5 | 2.0 | 63.0 | 0.82 | 14242 MB (process), 12791 MB (MLX active) | `.local-runs/song-full-profiled/trace.json` |
| 2026-09-18 | e5aa4f2 (T-4.0, référence froide) | none (bf16) | full | 64 (tronqué) | 100 (tronqué) | — | — | 0.6 | 0.1 | 4.0 | — | 9969 MB (process) | `.local-runs/bench/2026-09-18.txt` (abc 1.0s, sem 1.3s ; 120s refroidissement, A/A 0% dispersion) |
| 2026-09-18 | c37f5c1 (T-4.1, O4 confirmé) | none (bf16) | full | 64 (tronqué) | 100 (tronqué) | — | — | 0.6 | 0.1 | 4.0 | — | 9971 MB (process) | `bench-song.sh` (abc 1.0s, sem 1.4s ; stable vs référence T-4.0, 0% dispersion) |
| 2026-09-18 | HEAD (T-4.2, O2 actif) | none (bf16) | full | 64 (tronqué) | 100 (tronqué) | — | — | 0.5 | 0.1 | 4.0 | — | 9858 MB (process) | `bench-song.sh` (abc 1.0s, sem 1.4s ; NAR −17% vs T-4.1, dispersion A/A 3.4%) |
| 2026-09-18 | HEAD (T-4.2, O2 actif) | none (bf16) | full | 572 | 1574 | 65.1 | 67.8 | 50.9 | 2.0 | 63.0 | 0.83 | 14153 MB (process), 12791 MB (MLX active) | `.local-runs/song-full-o2/trace.json` (chanson complète, NAR −1.2% vs T-3.7 51.5s — gain du plan « 10-20s » ne se vérifie pas à cette longueur de préfixe) |
| 2026-09-18 | HEAD (T-4.4, O7 : compile SnakeBeta) | none (bf16) | full | 572 | 1574 | — | — | 47.6 | 1.4 | 63.0 | — | 14121 MB (process), 13108 MB (MLX active) | `.local-runs/song-full-o7/trace.json` (chanson complète ; VAE −30% vs 2.0s, seuil de 5% largement dépassé ; NAR à 47.6s dans ce run précis, à ne pas attribuer à O7 qui ne touche pas le NAR — variance de mesure entre runs non refroidis identiquement) |
| 2026-09-18 | 98892af (T-4.5, référence bf16 avant O5) | none (bf16) | full | 572 | 1574 | — | 68.7 | 45.4 | 1.4 | 63.0 | — | 14119 MB (process), 13108 MB (MLX active) | `.local-runs/song-full-bf16/trace.json` (chanson complète, réglages par défaut ; abc 8.6s, sem 22.9s) |
| 2026-09-18 | 98892af (T-4.5, O5 : quant AR qint8) | qint8 (AR seul, groupe 64) | full | 572 | 1536 | — | 96.6 | 45.2 | 1.4 | 61.4 | — | 12857 MB (process), 11654 MB (MLX active) | `.local-runs/song-full-qint8/trace.json` (chanson complète ; AR abc+sem 31.5s→22.0s, −30.2% ; NAR/VAE inchangés, confirmant l'exclusion `nar_*` ; nombre de frames sémantiques différent de la ligne bf16 — tirage stochastique, pas une régression, cf. `tasks/JOURNAL.md`) |
| 2026-09-18 | 98892af (T-4.5, O5 : chanson courte A/B/B/A) | none (bf16) puis qint8 | full | 64 (tronqué) | 100 (tronqué) | — | — | 0.5 | 0.1 | 4.0 | — | ≈9900 MB (bf16) / ≈8500 MB (qint8) | `Scripts/bench-song.sh` : A1 abc=1.0 sem=1.3, B1 abc=0.9 sem=0.9, B2 abc=0.7 sem=0.9, A2 abc=1.0 sem=1.3 (dispersion A/A 0%, B/B 8.7% — bruit sub-seconde) ; NAR/VAE identiques dans les 4 points |
