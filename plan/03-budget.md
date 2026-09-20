## 3. Budget mémoire et performance attendue (M3 Max 96 Go)

### 3.1 Mémoire

| Composant | Résident | Note |
|---|---|---|
| LM bf16 (628 tenseurs) | 7,26 Go | dont voie AR ≈ 3,2 Go, voie NAR ≈ 2,8 Go, embed + lm_head 1,5 Go, `pe` 0,1 Go |
| Cache KV AR (phase sémantique, ≤ 24 576 tokens) | ≤ 2,8 Go | 28 couches × 2 × 8 têtes × 128 × L × 2 o ; typiquement L ≈ 7-12 k ⇒ 0,8-1,4 Go |
| Cache K/V du préfixe (NAR) | ≈ idem | réutilisable depuis la phase sémantique (§7, opt. O2) |
| Activations NAR (L_nar ≈ 5 400) | < 2 Go | MLP intermédiaire 5400×6144×2 o = 66 Mo ; SDPA fusionné (pas de matrice de scores 5400×12800×16 = 2,2 Go) — si MLX matérialise les scores, tuiler les requêtes (§7, O6) |
| VAE décodeur fp32 | 0,25 Go | activations tuilées : tuile de 1 056 frames ⇒ ≈ 0,5 Go par tenseur à la dernière couche |
| Total pic estimé | **≈ 12-15 Go** | large ; aucune stratégie d'unload n'est nécessaire, mais `Memory.clearCache()` entre étapes reste obligatoire |
| LM 8-bit voie AR (option J4) | ≈ 5,5 Go | lm_head + embed non quantifiés par défaut |

### 3.2 Débit estimé (ordres de grandeur à MESURER au Jalon 3 — ne pas les citer comme résultats)

- **AR décodage** : ≈ 3,6 Go lus par token en bf16 (voie AR 1,41 G params + lm_head 0,38 G) ⇒ borne mémoire ≈ 100 tok/s, attendu **40-70 tok/s**. Avec lm_head restreint (§7 O1) : 1,79 → 1,48 G params/token. Chanson de 3,6 min = ≈ 5 400 tokens sémantiques + ≈ 1 500 tokens ABC ⇒ **≈ 2-3 min**. Référence 4090 : 139 tok/s.
- **NAR** : 64 × forward de 1,41 G params sur ≈ 5 400 tokens ≈ 1 PFLOP de matmuls + ≈ 1 PFLOP d'attention ⇒ à 8-12 TFLOPS effectifs : **≈ 3-4 min**. C'est l'étape à profiler en premier.
- **VAE** : ≈ 10-20 s (fp32, convs k=7 sur 10 M d'échantillons). Référence 4090 : 3,6 s.
- **Total attendu ≈ 6-8 min pour 3,6 min d'audio** (vs 71 s sur 4090). Acceptable pour une v1 ; le Jalon 4 vise < 5 min.

### 3.3 Tableau de mesures à remplir (colonnes de `BENCHMARKS.md`, source unique = `swift-mlx-profiler` `TTSMetrics`)

`| Date | Commit | Quant AR | cot | Tokens ABC | Tokens sém. | ABC tok/s | Sém. tok/s | NAR s | VAE s | Audio s | RTF | Pic mémoire | Trace |`

