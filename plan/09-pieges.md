## 9. Checklist de pièges (contrat de revue — cocher les 16 à chaque tâche)

1. **RoPE non entrelacé** (`traditional: false`) et **q_norm/k_norm avant RoPE** — l'oubli donne des logits plausibles mais faux.
2. **K/V en cache = après k_norm et RoPE** ; la réutilisation NAR (O2) suppose les mêmes positions absolues.
3. **Positions NAR = `L_ar + i`** pour RoPE, **mais `pe[i]` local** pour l'embedding sinusoïdal ; les deux lignes de padding (`LATENT_START/END`) comptent dans `L_nar` et reçoivent `vae2llm(0) = bias`.
4. **Sigmoid du timestep en bf16** ; `logit(t)` en Double clampé ±20 (t = 1 ⇒ raw = 20).
5. **État ODE en bf16** en real (le résultat fp32 n'est qu'un cast final) ; en tiny fp32 des deux côtés.
6. **Ordre des processeurs de logits** (§2.3) et **scores fp32** (bf16 seulement en `legacy_off`) ; top-p garde 1 (3 en legacy_off) ; CFG combiné **en bf16 avant** upcast.
7. **Pénalité fenêtrée = fréquence** (`penalty^count`), fenêtre sur les tokens **générés** seulement, `min_tokens` compare `step` (0-based) ; le token END n'entre pas dans `history`.
8. **Eval tenseur par tenseur** dans les boucles de transformation de poids (fusion weight-norm, quantification) — sinon OOM silencieux ; jamais `eval(model.parameters())` global avant que tout soit chargé.
9. **weight_norm : norme sur toutes les dims sauf 0, AVANT la conversion de layout** ; ConvTranspose : `g` est par canal **d'entrée** ; layout `[in,out,k] → [out,k,in]`.
10. **`QuantizedLinear.weight.dtype` est uint32 packé** : cast des activations via un `computeDType` explicite, jamais `asType(weight.dtype)`.
11. **VAE fp32 strict** ; **pas de tanh final** ; conv finale **sans biais** ; clamp [-1, 1] seulement à l'export.
12. **head_dim 128 standard** — pas de padding SDPA à prévoir ; mais **aucun masque** en NAR (bidirectionnel + préfixe complet) et masque causal en AR.
13. **`xcodebuild test` ne transmet pas l'environnement** (`TEST_RUNNER_` uniquement) et **relance un worker crashé en affichant ✔** : utiliser `Scripts/run-tests.sh`, et `xcrun xctest` pour le code de sortie.
14. **Deadlock ABBA mlx-swift** (`compile` × `vjp`) ⇒ parallélisation swift-testing à 1 ; ne jamais lancer un gradient pendant une inférence.
15. **Pas de `swift build` pour produire un binaire** (metallib) ; `exact` sur mlx-swift ; jamais `branch:`.
16. **Ne jamais committer un chemin absolu**, un vrai poids, une fixture real, ni modifier une fixture tiny à la main ; le tokenizer `tokenizer.json` doit avoir `added_tokens` **vide**.

