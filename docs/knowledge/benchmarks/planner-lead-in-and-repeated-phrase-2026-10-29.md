---
type: benchmark
date: 2026-10-29
asks: [Q16, Q17]
---

# Intro du planificateur et phrase répétée (ASK Q16, Q17 b) — 2026-10-29

## Q16 : longueur de l'intro planifiée

Demande de l'iPhone `s20261017142348` (« Nouvelle version Pocket Anthem », techno 128 BPM, `cot full`, `[Intro]` 2 lignes, `[Verse]` 2, `[Chorus]` 2). Plans seuls sur le M3 Max dans le réglage du téléphone (`--quant int4-mixed --quant-head --precision fp16` : le plan du téléphone se reproduit à l'octet près, 11,5 s par plan), 10 graines par condition (410101 = celle du téléphone, puis 1 à 9). Lecture des plans par `yue2 score`. Dossier `.local-runs/bench.noindex/q16-intro/`.

| Condition | Plans qui chantent | Mesures avant la voix (plans chantés) | Notes chantées |
|---|---|---|---|
| I — la demande telle quelle | 7/10 | 0, 12, 18, 20, 25, 32, 57 | 30-52 (56 syllabes) |
| In — « male voice » au lieu de « rare male voice » | 6/10 | 0, 12, 18, 20, 32, 57 | 23-52 |
| C — refrain en premier (contournement de l'app) | 7/10 | 12, 12, 17, 18, 21, 25, 30 | 22-88 (51 syllabes) |
| Cn — C, « male voice » | 7/10 | 12, 12, 17, 18, 25, 25, 26 | 11-116 |
| Cs — C + « short intro » dans le style | 9/10 | 12, 12, 17, 17, 18, 24, 26, 34, 41 | 29-116 |
| Ib — I, planificateur bf16 | 4/10 | 0, 13, 16, 18 | 29-48 |
| PC — C, partition imposée ouvrant sur `% chorus` | 10/10 | 0-2 | 28-76, médiane 34 |
| PI — I, partition imposée ouvrant sur `% intro` + `V: Vocal` | 3/10 | 19, 34, 36 | 35-49 |

- Une section `% intro` ne porte jamais la voix : 0 note chantée dans les intros de I, In, C, Cs ; une seule note isolée sous `% intro` imposé (PI). Les lignes d'un `[Intro]` passent dans la section chantée suivante (rebaptisée `% verse`, 4 à 6 phrases au lieu de 2) : sur le téléphone, la ligne datée à 38,0 s est « Le curseur clignote sur l'écran », première ligne de l'`[Intro]`.
- L'intro n'est bornée par rien : l'invite ne porte aucune durée. Médiane 18 mesures (34 s) sur les 23 plans chantés de I, C et Cs ; plans de 26 à 220 mesures, 3 à 4 sur 10 coupés par le plafond de 4 096 jetons.
- Ni le refrain en premier, ni « short intro », ni « male voice » ne raccourcissent l'intro ; la graine décide (In reproduit I graine pour graine). Le planificateur bf16 n'est pas meilleur (4/10 chantent) : la quantification n'est pas en cause, c'est ce style techno qui pousse vers l'instrumental.
- Le préfixe `% chorus` fait chanter dès la mesure 1-3, mais le plan chante moins (médiane 34 notes pour 51 syllabes, contre 49 sans préfixe) : des lignes risquent de tomber.
- 3 à 6 plans sur 10 n'ont aucune note chantée sur ce style (int4 comme bf16) : `layout.firstSungBar == nil` les détecte avant la phase sémantique.

### Rendus des leviers (`4bit-lean`, Whisper medium libre)

Voxtral indisponible ce jour-là (son modèle est sur le Lexar, non monté) : transcription Whisper medium sans invite (`condition_on_previous_text=False`, pour qu'un texte répété ne se propage pas d'un segment à l'autre). Sur la référence « Maison » du 26, Whisper redonne les lectures Voxtral (couplet 2 : 1,5/4 en R, 3/4 en V).

| Rendu | Voix : `firstSungSeconds` → timeline | Durée : `seconds` → rendu | Entendu |
|---|---|---|---|
| plan du téléphone, `trimming(leadIn: 4, tail: 4)` | 8,0 → 2,9 s | 60,0 → 60,6 s | 6/6 lignes dans l'ordre |
| C-8 raccourci (intro d'1 mesure) | 2,6 → 2,6 s | 69,4 → 68,3 s | 6/6 dans l'ordre |
| C-9 raccourci (intro de 2 mesures) | 4,5 → 4,4 s | 71,3 → 71,4 s | 6/6, un refrain de plus |
| I-4 raccourci (intro de 4 mesures) | 8,2 → 8,3 s | 60,0 → 63,6 s | 6/6 dans l'ordre |
| PC-6 (`% chorus` imposé) | 1,2 → 0,6 s | 63,8 → 61,4 s | couplet perdu (4/6) |
| PC-410101 | 0 → 0 s | 48,8 → 47,7 s | 6/6, mais 4 lignes en 11 s |

- Raccourcir le plan du modèle : 4/4 propres. Le plan du téléphone, qui ne tenait qu'une ligne dans le plafond de 45 s, chante ses six lignes en 60 s.
- Les secondes nominales du `layout` tiennent sur ces rendus : la voix n'entre jamais plus de 0,1 s après `firstSungSeconds` (une fois 5 s avant : le rendu n'a pas joué toute l'intro écrite), et la durée reste entre −2,4 et +3,6 s de `seconds`. Une marge de 5 s suffit pour plafonner.
- `% chorus` imposé : à éviter (1/2).

## Q17 (b) : phrase identique sur des sections de textes différents

Deux témoins en `8bit-lean`, 4 graines par levier : « Système urbain » (`s20261026122324`, 4 mesures, 11 notes, imposée cinq fois ; graines 236350 = téléphone, 357068, 424242, 777001) et « Maison intelligente » (R du banc `maison-repeat`, 5 mesures, 14 noires ; graines 357068, 424242, 777001, 777002). Whisper medium libre, comme plus haut. Dossier `.local-runs/bench.noindex/q17-dedup/` (`*.json` les demandes, `*/audio.whisper.txt` les transcriptions).

| Levier | Système urbain : couplets à leur place | Maison : couplet 2 à sa place |
|---|---|---|
| la demande telle quelle (UB / R) | 0/4, le refrain aux cinq passes | 1,5/4 |
| Z — silences des couplets réécrits (`z16` → `z8z8`, `z4z4z4z4`) | 0/4 (une graine invente des paroles) | 1/4 net, couplet 2 présent 3/4 |
| G — CFG 1,5, négatif = même style, paroles vides | 1/4 net, 2 partiels | 2/4 nets, présent 4/4 |
| V — couplets une tierce mineure plus haut | 0,5/4, un rendu à 95 s | 3/4 (banc du 26) |
| V + G | 1,5/4 | 4/4 |

- **Pas de recopie** : entre passes, ≈ 1 % de jetons sémantiques communs à décalage d'une section, jamais plus de 3 identiques d'affilée (7 rendus du 26). Le texte répété est régénéré ; un blocage de n-grammes de jetons ne servirait à rien.
- **Une même mélodie appelle le même texte** : la réécriture de notation (Z) laisse la musique intacte et ne change rien ; le modèle lit l'ABC musicalement. Le moteur ne peut pas distinguer deux passes d'une phrase identique sans changer ce qui est joué : hors contrat.
- **Le guidage sur les paroles** (G) est le seul réglage moteur qui ait un effet sur les deux témoins : il fait apparaître les mots des couplets, mais la voix les répète parfois à la place d'un refrain. Coût : deux branches sur toute la phase sémantique, ×1,4 à ×2 sur le Mac. Aucun code : `cfgScale` + `negativeStyle` = le style.
- **« Système urbain » résiste à tout** (2 réussites entières sur 16 rendus de leviers, G et V + G, plus des partiels) ; « Maison », dont le couplet 1 garde toujours son texte, cède à une mélodie différente. La différence entre les deux n'est pas expliquée (phrase plus creuse : une mesure vide sur quatre).
- Les voix chantent plus vite que la partition sur « Maison » (≈ 8 s par section au lieu de 12 s) et remplissent la fin par des reprises, avec ou sans levier.
