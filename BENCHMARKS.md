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
| 2026-09-26 | HEAD (0.2.2 proposée ; sémantique forcée à 1750 tokens = 70,0 s) | none (bf16), MLX pur (profil Mac) | full | ≈570 | 1750 | — | — | 53.5 / 50.9 | 1.5 | 70.0 | — | 14804 MB (process) | `.local-runs/abba-bf16-plain{1,2}/trace.json` (abc 8.6/8.9 s, sem 25.7/25.9 s ; total 89.4/87.2 s ; 120 s refroidissement, A/B/B/A, contrôle ±2.5 %) |
| 2026-09-26 | HEAD (idem) | none (bf16), optimisé (résidence, VAE fp16/256, profil mobile adaptatif) | full | ≈570 | 1750 | — | — | 56.3 / 60.5 | 1.3 | 70.0 | — | 5172 MB (process) | `.local-runs/abba-bf16-opt{1,2}/trace.json` (total 98.2/134.7 s ; phase plan à 30 % GPU / 110 % CPU : récupération du cache au seuil, bf16 hors budget mobile) |
| 2026-09-26 | HEAD (idem) | qint8-all + head, MLX pur | full | ≈570 | 1750 | — | — | 57.8 / 60.8 | 1.5 | 70.0 | — | 11519 MB (process) | `.local-runs/abba-q8-plain{1,2}/trace.json` (abc 6.0/5.8 s, sem 18.1 s ; total 83.5/86.2 s, contrôle ±3 %) |
| 2026-09-26 | HEAD (idem) | qint8-all + head, optimisé | full | ≈570 | 1750 | — | — | 59.9 / 60.5 | 1.2 | 70.0 | — | 4324 MB (process) | `.local-runs/abba-q8-opt{1,2}/trace.json` (total 84.9/85.6 s : résidence sans coût mesurable) |
| 2026-09-26 | HEAD (idem) | int4-mixed + head (NAR qint8), MLX pur | full | ≈570 | 1750 | — | — | 59.1 / 60.6 | 1.5 | 70.0 | — | 10464 MB (process) | `.local-runs/abba-q4mixed-plain{1,2}/trace.json` (abc 2.9/3.0 s, sem 13.0/13.7 s ; total 76.6/79.0 s, contrôle ±3 %) |
| 2026-09-26 | HEAD (idem) | int4-mixed + head, optimisé | full | ≈570 | 1750 | — | — | 59.7 / 76.3 | 1.2 | 70.0 | — | 3266-3519 MB (process) | `.local-runs/abba-q4mixed-opt{1,2}/trace.json` (total 76.6/109.5 s ; run 2 hors bande sur toutes les phases, trace sans anomalie, contrôle suivant à 79.0 s — état GPU ponctuel, non reproduit) |
| 2026-09-26 | 8313269 (objectif RTF < 1 sur 70 s) | int4 (AR seul) + head, NAR bf16, MLX pur | full | ≈570 | 1750 | — | — | 49.4 / 51.3 | 1.5 | 70.0 | 0.96 / 0.99 | ≈ 12 GB (process) | `.local-runs/bench.noindex/rtf-int4ar-bf16nar-32{,b}/trace.json` (abc 3.5 s, sem 13.0 s ; total **67.5 / 69.3 s**, 32 pas ; 120 s refroidissement, contrôle ±2.7 %) |
| 2026-09-26 | 8313269 (idem, 24 pas d'ODE) | int4 + head, NAR bf16, MLX pur | full | ≈570 | 1750 | — | — | 37.2 | 1.5 | 70.0 | 0.79 | ≈ 12 GB (process) | `.local-runs/bench.noindex/rtf-int4ar-bf16nar-24/trace.json` (total 55.3 s ; qualité à écouter) |
| 2026-09-26 | 8313269 (idem, 16 pas d'ODE) | int4 + head, NAR bf16, MLX pur | full | ≈570 | 1750 | — | — | 24.8 | 1.6 | 70.0 | 0.63 | ≈ 12 GB (process) | `.local-runs/rtf-int4ar-bf16nar-ode16/trace.json` (total 43.9 s ; qualité à écouter) |
| 2026-09-26 | HEAD (cache KV préalloué, T-4.3 bis) | int4-mixed + head, MLX pur | full | ≈570 | 1750 | — | — | 59.9 | 1.6 | 70.0 | 1.07 | ≈ 10.5 GB (process) | `.local-runs/bench.noindex/kv-q4mixed/trace.json` (abc 2.7 s, sem 10.7 s — phases AR 15.9 → 13.4 s, −16 % ; total 75.0 s vs 76.6/79.0 s) |
| 2026-09-26 | HEAD (idem) | qint8-all + head, NAR packé | full | ≈570 | 1750 | — | — | 66.9 | 1.6 | 70.0 | 1.30 | ≈ 11.5 GB | `.local-runs/bench.noindex/kv-q8all-packed/trace.json` (abc 5.5 s, sem 17.1 s ; NAR hors bande vs 57.8-60.8 s plus tôt — WindowServer 44 % au départ) |
| 2026-09-26 | HEAD (idem, `--nar-compute dequantized`) | qint8-all + head, NAR dé-quantifié bf16 | full | ≈570 | 1750 | — | — | 55.1 / 53.1 | 1.6 | 70.0 | 1.11 / 1.08 | ≈ 13 GB | `.local-runs/bench.noindex/kv-q8all-dequant{,-b}/trace.json` (abc 5.6/5.4 s, sem 15.5/15.2 s ; total **77.9 / 75.4 s** ; `mediaanalysisd` à 271 % pendant le premier) |
| 2026-09-26 | HEAD (sans code, 20 pas d'ODE) | qint8-all + head, NAR packé | full | ≈570 | 1750 | — | — | 36.4 | 1.5 | 70.0 | 0.92 | ≈ 11.5 GB | `.local-runs/bench.noindex/q8rtf-q8all-ode20/trace.json` (abc 6.4 s, sem 20.0 s ; total 64.4 s ; qualité à écouter) |
| 2026-09-26 | HEAD (six références, `--reference`) | 4bit-fast : int4-mixed+head, NAR dé-quantifié, tout résident | full | ≈570 | 1750 | — | — | 51.1 | 1.1 | 70.0 | 0.94 | 8875 MB | `.local-runs/bench.noindex/ref-20260926-2300-4bit-fast-1/trace.json` (abc 2.7 s, sem 10.7 s ; total 65.7 s) |
| 2026-09-26 | HEAD (idem) | 4bit-lean : int4-mixed+head, résidence, packé, fp16, mobile | full | ≈570 | 1750 | — | — | 61.5 | 1.2 | 70.0 | 1.11 | 3395 MB | `…/ref-20260926-2300-4bit-lean-1/trace.json` (abc 4.2 s, sem 10.8 s ; total 77.7 s) |
| 2026-09-26 | HEAD (idem) | 8bit-fast : qint8-all+head, NAR dé-quantifié, tout résident | full | ≈570 | 1750 | — | — | 54.0 | 1.1 | 70.0 | 1.09 | 9936 MB | `…/ref-20260926-2300-8bit-fast-1/trace.json` (abc 5.8 s, sem 15.5 s ; total 76.5 s ; 24 pas : 67.3 s, `fill-q8all-dequant-ode24`) |
| 2026-09-26 | HEAD (idem) | 8bit-lean : qint8-all+head, résidence, packé, fp16, mobile | full | ≈570 | 1750 | — | — | 57.1 | 1.1 | 70.0 | 1.13 | 4378 MB | `…/ref-20260926-2300-8bit-lean-1/trace.json` (abc 5.6 s, sem 15.3 s ; total 79.3 s) |
| 2026-09-26 | HEAD (idem) | 16bit-fast : bf16, tout résident, VAE fp16/1024 | full | ≈570 | 1750 | — | — | 51.3 | 1.1 | 70.0 | 1.21 | 11966 MB | `…/ref-20260926-2300-16bit-fast-1/trace.json` (abc 8.7 s, sem 23.8 s ; total 84.9 s) |
| 2026-09-26 | HEAD (idem) | 16bit-lean : bf16, résidence, VAE fp16/256, caches Mac | full | ≈570 | 1750 | — | — | 52.2 | 1.2 | 70.0 | 1.21 | 6920 MB | `…/ref-20260926-2300-16bit-lean-1/trace.json` (abc 8.4 s, sem 23.1 s ; total 84.9 s) |
| 2026-09-26 | HEAD (cache KV préalloué) | int4 (AR) + head, NAR bf16 | full | ≈570 | 1750 | — | — | 50.4 | 1.6 | 70.0 | 0.94 | ≈ 12 GB | `.local-runs/bench.noindex/fill-int4ar-bf16nar-32/trace.json` (abc 3.4 s, sem 10.7 s ; total **66.1 s**) |
