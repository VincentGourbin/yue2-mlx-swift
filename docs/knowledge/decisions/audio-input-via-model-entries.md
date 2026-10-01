# Décision — entrée audio par les entrées natives du modèle (score imposé, SDEdit, continuation)

**Contexte** (2026-10-01, issue #1 reprise, demande de Vincent : « scénarios utilisateurs crédibles », implémentation backend de bout en bout). YuE2 n'a aucune compréhension audio à l'inférence : MERT2 et SheetSage2 ne servent qu'à l'entraînement, l'encodeur VAE ne produit que des latents acoustiques sans sens pour la voie AR. La seule entrée audio « honnête » passe donc par les artefacts que le modèle consomme déjà : la partition ABC (`SongRequest.abc`), les jetons sémantiques (passé de l'AR) et les latents (état de départ du solveur).

**Décision** : trois scénarios, tous câblés dans `YuE2Core` (API) et `yue2` (CLI), aucun nouveau modèle.

| Scénario | Entrée | Mécanisme | Coût |
|---|---|---|---|
| Mélodie fredonnée → chanson | enregistrement mono | `MelodyTranscriber` (YIN, grille de doubles-croches au BPM donné, tonalité Krumhansl, un accord diatonique par mesure) → ABC dans le dialecte du modèle → `generate --abc-file` (planificateur court-circuité) | CPU, < 1 s ; génération normale sans phase ABC |
| Variation | répertoire `generate --out` | `NAREdit.variation` : départ `bruit·t + latents·(1−t)`, `round((1−s)·pas)` pas sautés, mêmes jetons | NAR × s (0,4 → 6,2 s contre 13,9 s) |
| Régénérer à partir d'ici | idem + instant | `generateSemantic(continuation:)` (jetons gardés = passé de l'AR) puis `NAREdit.keep` (frames gardées réimposées après chaque pas, masque RePaint) | AR sur la partie neuve + NAR complet |

**Mesuré** (M3 Max, 4bit-fast, fredonnement synthétique de 8 mesures à 100 BPM, 25 notes) : transcription exacte (25/25 notes, rythme, tonalité G, dièses portés par l'armure) ; chanson de 20 s qui suit la partition (sémantique 3,1 s, NAR 13,9 s) ; variation s = 0,4 : écart relatif 0,38 sur les latents, même durée ; régénération depuis 10 s : 250 jetons et 250 frames gardés bit-exacts, 246/249 jetons neufs différents, saut maximal à la couture 0,16 contre 0,13 ailleurs (pas de discontinuité). Preuves d'écoute : fichiers attachés à la release v1.3.0.

**Points réglés en chemin** :
- YIN : s'arrêter au franchissement du seuil (pente descendante) lit une période trop courte, un demi-ton trop haut sur un son chanté ; il faut suivre le creux jusqu'à son minimum local, sur la courbe CMNDF complète.
- Quantification : les segments perdent l'attaque et la chute (≈ 30 ms) ; un silence plus court que 0,6 double-croche (respiration, consonne) tient la note jusqu'à la suivante.
- Longueur de la régénération : reconditionné sur sa première moitié, l'AR prolonge volontiers la chanson ×2,5 (20 s → 50 s). Par défaut la longueur de la source est conservée (`RegenerateLength.keepSource`), `.free` / `--free-length` rend la main au modèle.
- Un vibrato de test « multiplié » (`sin(2π·f·v(t)·t)`) balaie la fréquence de plus en plus large avec t ; l'intégrer dans la phase.

**Non couvert** : chansons multi-chunks (> 1750 frames) pour les deux éditions ; harmonisation (un accord diatonique par mesure, ex æquo tranché par l'ordre I ii iii IV V vi) ; polyphonie ; détection du tempo (BPM fourni par l'appelant). Le remix audio → audio par l'encodeur VAE (`remix-experimental`) reste hors plan : rien ne relie des latents réels aux jetons sémantiques.
