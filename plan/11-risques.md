## 11. Risques

| Risque | Impact | Mitigation |
|---|---|---|
| NAR trop lent sur Metal (64 forwards de 5 400 tokens) | chanson de 3,6 min en > 10 min | profiler dès 3.5 ; O8, O6 ; réduire `ode_steps` en option (qualité à écouter) ; noter que la référence 4090 est 71 s au total |
| SDPA MLX matérialise les scores pour L_nar ≈ 5 400 requêtes | 2,2 Go transitoires par couche | O6 (tuilage requêtes 1024) |
| Dérive bf16 MPS vs Metal sur 64 pas d'ODE | parité `solve` real hors tolérance alors que velocity est OK | tolérance `solve` plus large (5e-2) ; juger surtout la parité velocity + écoute ; comparer aussi à un run CPU fp32 court (4 pas, 16 frames) |
| swift-transformers et la regex Qwen (`(?i:…)`, `\p{L}`) | tokens différents sur des caractères rares | corpus tokenizer étendu (chinois, accents, emoji, retours à la ligne multiples) ; si écart, fallback : port de tiktoken (BPE par rangs) sur le modèle de `flux TekkenTokenizer.swift` |
| `update(parameters:)` no-op silencieux | parité sur des poids aléatoires | garde de couverture (§8) obligatoire |
| Licence CC-BY-NC-4.0 des poids | pas d'usage commercial | README + `ModelCatalog.license` ; G-6 avant toute publication |
| Version de transformers dans `.venv-ref` (4.57.6 requis, `trust_remote_code`) | scripts de référence cassés | pins stricts (1.1) ; pytest upstream vert avant toute fixture |
| Mémoire GPU wired par défaut (≈ 72 Go) | non bloquant ici (< 15 Go) | documenter `sysctl iogpu.wired_limit_mb` dans README seulement si nécessaire |

