# Instrumental et karaoké : mesures du 2026-10-07

Deux usages des paroles qui posaient problème : produire un morceau **sans voix**, et **horodater les paroles** d'un morceau chanté (karaoké). Tout part d'un fait : la partition ABC que planifie YuE2 sépare `V: Vocal` de `V: Ins`, et le rendu la suit (SheetSage2 retranscrit le rendu de `city_lights` presque note pour note : 98 % des attaques entendues tombent sur la partition).

Outils tiers (non embarqués) : `Scripts/lyrics-eval/` (Demucs htdemucs, MMS_FA de torchaudio, Whisper/stable-ts, pYIN). Morceaux : `city_lights` (EN, ballade, 87 BPM, graine 831001), `fr_ville` (FR, chanson, 92 BPM, `M:2/4 L:1/16`), `en_fast` (EN, pop rock, 128 BPM). Rendus bf16, réglages par défaut, Mac M3 Max ; ces mesures portent sur la qualité, pas sur la vitesse.

## Instrumental

Recette amont (`skills/yue2-music/instrumental` de m-a-p) : planifier, déplacer toutes les notes de `Vocal` vers `Ins` (Vocal ne garde que silences et accords, priorité à la voix en cas de chevauchement), rendre cette partition imposée avec les seules étiquettes de section comme paroles et un style « no vocals… ». Le README amont avertit que des fuites de voix restent possibles.

**Détecteur de fuite.** Deux mesures concordent et suffisent : le nombre de notes que SheetSage2 range sur la piste chantée (track 0) et l'énergie de la piste vocale Demucs. Propre : 0 note, énergie ≤ 0,04. Fuite : 25 à 33 notes, énergie 0,24 à 0,46. Chanson chantée : 66 notes, 0,63. Trompeurs : AST AudioSet (0,03 « singing » sur la chanson chantée) et Essentia `voice_instrumental` (appelle « voix » un piano solo, 0,8-0,95 sur des rendus propres que Demucs, SheetSage2 et la stabilité de hauteur, 7 cents contre 13 pour une voix, donnent instrumentaux).

| Configuration (partition de `city_lights`) | Rendus sans voix |
|---|---|
| Partition d'origine, paroles vides, style « no vocals » | 0/1 (voix partout) |
| Partition d'origine + branche négative « vocals, singing… », CFG 1,5 | 0/2 (voix partout) |
| Transfert, paroles vides | 0/1 |
| Transfert, étiquettes de section, CFG 1,0 | 3/4 |
| Transfert, étiquettes, CFG 1,5 | 3/4 |
| **Transfert, étiquettes, branche négative, CFG 1,5** | **6/8** |

Le transfert est indispensable ; la branche négative (`negativeStyle`, nouvelle option expérimentale : la branche CFG négative reçoit `[Tags]` avec ces mots au lieu de l'instruction seule) aide sans suffire. Quand ça fuit, c'est **toujours le couplet** (10-30 s), jamais le refrain : une voix fredonne la mélodie déplacée. D'où le contrôle après rendu : SheetSage2 (3 s pour une minute) compte les notes chantées et, au-delà de 4, on relance la graine suivante (2 relances par défaut) ; à ~25 % de fuite par graine, il reste ≈ 2 % de risque. La mélodie reste suivie : 73-96 % des notes planifiées entendues, sans dégradation par la branche négative.

Implémenté : `ABCScore` (lecteur du dialecte), `Instrumental.transfer/renderRequest`, `yue2 generate --instrumental [--instrumental-retries N]` (écrit `planned.abc`, `instrumental.json`). Bout en bout, graine 5, plan du modèle : 62 notes déplacées, 0 note chantée, Demucs 0,02. Écart à l'amont : sur un chevauchement, nous laissons tomber tout le matériau `Ins` de la mesure (l'amont garde les morceaux de notes non recouverts).

## Karaoké

Prédiction : notes de `Vocal`, liaisons fusionnées (une note tenue = une note, jamais une nouvelle syllabe), placées sur la grille de temps que SheetSage2 transcrit du rendu ; syllabes alignées aux notes par programmation dynamique (mélisme bon marché sur la dernière syllabe du vers, cher au milieu). Référence : MMS_FA sur le mix (identique à 98 % à MMS sur la voix isolée Demucs ; plus proche des attaques réelles que Whisper : 66 ms contre 100 ms). Débuts de mots, part à ≤ 300 ms de MMS :

| Mode | city | fr | en_fast |
|---|---|---|---|
| Tempo `Q:` nominal depuis t = 0 | 0 % (≈ 3 s de décalage) | — | — |
| **Partition + grille SheetSage2** | 81 % | 84 % | 96 % |
| Partition + ancres Whisper | 89 % | 98 % | 96 % |
| Partition + ancres MMS | 98 % | 100 % | 100 % |

Débuts de vers : médiane 38-75 ms, ≤ 215 ms partout. La partition donne l'attaque musicale et la fin des notes tenues ; un aligneur forcé ne sert qu'à choisir **sur quelle note** démarre un mot quand un mélisme est ambigu (la DP seule se trompe d'une note dans ces cas). Recaler même Whisper, plus grossier, sur la partition l'améliore (77 → 89 % sur city).

Fins de notes tenues en fin de vers : la voix dure 90 à 290 ms de plus que la partition en moyenne (0,4-0,56 s sur 4 vers d'`en_fast`) ; SheetSage2 ne corrige pas (il quantifie comme la partition). À calibrer sur plus de morceaux avant d'ajouter une extension.

Pièges : `L:` n'est pas toujours `1/32` (`fr_ville` est en `M:2/4 L:1/16`) ; pYIN fusionne les notes répétées à la même hauteur en un segment « tenu » (ce n'est pas la synthèse qui tient) ; le compteur de syllabes anglais à règles du Swift diffère du dictionnaire CMU sur quelques mots (96-100 % de mots identiques au prototype Python, même précision).

Implémenté : `LyricTimeline` (YuE2Core, sans dépendance à SheetSage2Core : grille et attaques en entrée), `yue2 karaoke --song <dir>` (transcrit l'audio ou reprend `--events`, langue déduite du style, `--anchors` expérimental) → `karaoke.json` (vers, mots, syllabes, notes par syllabe). `yue2 transcribe` écrit désormais `events.json` comme son aide l'annonçait.

## Minutage par les têtes d'alignement du LM (2026-10-08)

Objectif de Vincent : que l'app se recale simplement, à la lecture, sur les temps des mesures, des notes et des mots d'une chanson qu'elle vient de générer, sans recalcul ni modèle de plus. Constat de départ : l'horloge nominale du plan (`Q:` depuis t = 0) tient à 0,15-0,23 s sur deux chansons, mais pas sur `city_lights` : le rendu y saute presque un temps d'intro, et tout reste décalé de 2,4 s. L'app coupe aussi la chanson à la durée demandée : « Matin de pluie » s'arrête au 5e vers sur 8, « Sur le quai » au 12e sur 18.

**Découverte.** En écrivant chaque trame sémantique (40 ms), le LM regarde son prompt, et quelques têtes AR suivent la chanson, comme les « alignment heads » de Whisper :
- quatre têtes regardent la mesure jouée : couches/têtes 6/8, 10/3, 18/7 et 18/6. Elles la trouvent dans 70-87 % des trames, et à une mesure près dans 94-100 %. Elles ont une avance constante sur la musique : 0,17, 0,30, 0,28 et 0,52 s ; moyennées puis passées dans un chemin monotone, 0,33 s ;
- une tête regarde le **mot suivant** : 14/10. Son attention arrive sur le mot k+1 quand le mot k commence.

Les têtes ont été trouvées sur fr et en_fast, puis tenues telles quelles sur six autres chansons : quatre générées sur l'iPhone en int4 (jeu d'essai de l'app), deux générées de bout en bout. L'int4 ne change rien : mêmes têtes, même avance.

**Mesure.** Débuts de mots à ≤ 300 ms (puis médiane) de MMS sur la voix Demucs, vers effectivement chantés seulement :

| Chanson | Partition + grille SheetSage2 | Attention seule (`AttentionTimeline`) |
|---|---|---|
| pluie (iPhone, FR) | 93 % | 89 % (79 ms) |
| quai (iPhone, FR, 75 s) | 83 % | 87 % (78 ms) |
| berceuse (iPhone, melody) | **10 %** | 78 % (80 ms) |
| sifflet (iPhone, partition de 5 mesures) | **12 %** | 71 % (122 ms) |
| fr | 84 % | 97 % (78 ms) |
| en_fast | 94 % | 94 % (59 ms) |
| indie (bout en bout) | **10 %** (« oh oh » non écrits en intro) | 87 % (70 ms) |
| rock (bout en bout) | 98 % | 88 % (65 ms) |

Mesures : 60-160 ms d'écart médian à SheetSage2. SheetSage2 reste un peu plus fin quand la voix suit la partition. Il s'effondre quand elle la quitte : berceuse, sifflet (le chant continue après la partition) et indie. L'attention tient partout.

**Recette.**
- Passe forcée sur les couches 0-18 après la phase sémantique, en réutilisant le cache KV de la génération : 1 à 2 s sur M3 Max pour 40-75 s de musique.
- Découpage du prompt jeton → mesure / mot par offsets d'octets exacts. Chaque caractère de la forme interne d'un jeton BPE vaut un octet ; c'est ce qui survit aux accents coupés entre jetons.
- Chemin monotone à fin libre (une chanson coupée n'atteint pas la dernière mesure).
- Les mots sont posés sur les notes de la partition par la DP, tirés vers les ancres de la tête « mot ». Un mot à plus de 0,75 s de son ancre quitte la partition (`onScore: false`).
- L'horloge des mesures est recalée sur les mots voisins (± 2 mesures). À ≤ 100 ms près, cela fait passer pluie de 36 à 57 % et en_fast de 72 à 85 %.

**Pièges.**
- Sans notes entendues, le calage de la mesure 1 prenait le décalage −64 (bug corrigé ; les égalités vont maintenant vers 0).
- MMS sur le mix est peu fiable sur les chansons de l'app (score de confiance médian 0,4-0,6). Il faut l'appliquer à la voix Demucs, et seulement aux vers chantés (Whisper libre pour savoir lesquels).
- Les partitions imposées de l'app changent de mesure en cours (`M:5/8`). `ABCScore` les lit désormais mesure par mesure ; le transfert instrumental les refuse.

Implémenté :
- `AttentionTimeline` (YuE2Core) ;
- `YuE2ForCausalLM.prefixAttention` ;
- `YuE2Tokenizer.byteCount(of:)` ;
- format v1 de `LyricTimeline` : `bars`, `notes`, mesure/temps partout, `onScore`, `duration` ;
- `yue2 generate --timeline`, `yue2 karaoke` (attention par défaut, `--sheetsage` sinon) ;
- commande cachée `yue2 attention-probe` et scripts `Scripts/lyrics-eval/{segments,attention_eval,head_anchors,head_grid}.py`.

Expériences : `.local-runs/karaoke-probe.noindex/{attn,app,e2e-timeline}`. Passage de relais à l'app : `../handover-timeline-2026-10-08.md`.

## Ouvert

- ~~Aligneur embarqué~~ : remplacé par les têtes d'alignement du LM (2026-10-08). MMS (Meta, poids CC-BY-NC 4.0) ne sert plus que de référence de mesure.
- Coût de la passe d'attention sur iPhone (non mesuré ; 19 couches sur prompt + chanson, cache réutilisé).
- Pourquoi le couplet fuit et pas le refrain (registre de la mélodie déplacée ? étiquette `[Verse]` ?).
- Extension des fins de vers, calibrée sur un corpus plus large ; écoute par Vincent des rendus instrumentaux (`.local-runs/karaoke-probe.noindex/neg-c1.5-s*`, `e2e-instrumental`).

## Reprises courtes : ordre du minutage et densité de la partition (2026-10-09, ASK Q14/Q15)

**Ordre.** Sans ancre de la tête « mot » (toujours pour le dernier mot, parfois pour l'avant-dernier quand la chanson finit sans chant), un mot restait sur sa note de partition, même quand celle-ci tombait avant le mot précédent : sur « Air sifflé » (partition de 5 mesures, 45 s d'audio), « Jusqu'à demain » était daté à 10,3 s après une ligne à 26 s. Désormais, un tel mot suit le précédent hors partition, avec 0,2 s par syllabe. Une syllabe finit au plus tard au début de la suivante : 6 chansons sur 8 avaient un chevauchement, dû à une fin de mélisme partagée. Débuts de mots inchangés sur les 8 chansons (indie 87 → 89 %). Essayé et écarté : un plancher sur l'attention normalisée (ε de 0,05 à 0,5), pour que les trames sans regard sur les paroles ne décident pas du chemin. Aucun effet, car la tête ne passe jamais nettement au dernier mot.

**Densité.** Reprise d'un riff instrumental de 19,5 s (`s20261008195029`) : 58 notes de `Vocal` en doubles croches et croches, silences au milieu des mesures, sous 4 vers (28 syllabes). Mac, int4-mixed + tête, fp16, phase sémantique et minutage seuls ; transcription Whisper libre pour savoir ce qui est chanté.

| Partition | Graines | `wordsOnScore` ≥ 0,9 | Chant (Whisper) |
|---|---|---|---|
| riff, CFG 1,0 / 1,5 | 8 + 8 | 4 / 4, dont 1 / 1 qui entasse tous les vers au début | 0 sur 5 intelligible (avec 4 rendus de l'app) |
| une note par temps, `% intro` | 8 | 2 : couplet en place 8 fois, refrain après la partition 6 fois | 1 sur 1 dans l'ordre |
| une note par temps, `% verse` / `% chorus` | 8 | 7 (dont 1 coupé à 10 s) | 4 sur 4 dans l'ordre |

Ce qui ressort :
- Les plans du modèle donnent 0,9 à 1,2 note par syllabe (1,8 sur une berceuse mélismatique réussie) et ne respectent pas eux-mêmes les noms de sections des paroles (`% intro` ajouté, `% pre-chorus` inventé, `[Bridge]` → `% outro`).
- Renommer les sections n'aide qu'une fois la densité corrigée.
- `wordsOnScore` attrape les paroles chantées plus loin ou après la partition. Il ne voit ni un rendu qui entasse tous les vers au début, ni un chant inintelligible sur la bonne mesure : deux rendus « justes » au minutage étaient muets ou incompréhensibles pour Whisper.

Expériences : scratchpad de session `q15/sweep/` ; écoutes dans `~/Desktop/PocketAnthem-ecoute/2026-10-09-riff-simplifie/`.
