# Relais : pipeline profond de l'AR pour PocketAnthem (2026-11-27)

Destinataire : l'agent qui intègre le moteur dans l'app `yue2-ios`. Le moteur se travaille ici, l'intégration chez cet agent.

## Ce qui change dans le moteur (1.13.0)

- Pendant la partition et la phase sémantique, le réseau du pas t+1 part sur le GPU avant que l'hôte lise le jeton t. Le GPU n'attend donc plus l'échantillonnage entre deux pas.
- **Rien à changer côté app.** Ce n'est ni une option ni une API : un `pin-engine.sh v1.13.0` suffit, une fois la version publiée.
- **Les chansons ne changent pas.** Jetons de partition, jetons sémantiques, cache KV et latents sont identiques au bit près à la 1.12.2. Seule la vitesse bouge.
- **La porte GPU est vérifiée à chaque pas, comme avant.** Pause, arrière-plan et annulation restent pris en compte au pas suivant.
  - Les soumissions se font toujours d'un bloc, juste après le passage de la porte et sans attente intermédiaire. Seul l'ordre change : tirage du pas t, puis réseau du pas t+1, avant la lecture du jeton.
  - Quand la porte se ferme, au plus un pas est en vol, comme avant. Ce pas inclut maintenant le réseau suivant, soit environ 50-60 ms estimés sur l'iPhone en `8bit-lean`.
  - Le test pause / arrière-plan / kill / reprise sur l'iPhone reste à refaire avant de livrer.

## Mesures sur le Mac (M3 Max)

Chanson de 70 s en `8bit-lean`, A/B/B/A ×3 avec refroidissement :

| Phase | Avant | Après | Gain |
|---|---|---|---|
| Sémantique | 115,3 jetons/s | 125,5 jetons/s | +8,8 % |
| Partition | 108,8 jetons/s | 117,2 jetons/s | +7,8 % |

Au banc, en `4bit-lean` : −12 % par jeton.

## À mesurer sur l'iPhone (le moteur ne peut pas le faire)

1. **Le gain réel du pipeline profond.** Comparer le temps des phases partition et sémantique en 1.12.2 et en 1.13.0 :
   - même chanson, même graine, `8bit-lean` ;
   - téléphone froid, deux passages chacun.

   On s'attend à moins que sur le Mac, parce que la lecture des poids domine là-bas. Les jetons doivent être identiques (`semantic.npy`) : si ce n'est pas le cas, c'est un bug à remonter.
2. **Le rapport 4 bits / 8 bits de la phase sémantique** (`4bit-lean` contre `8bit-lean`), pour savoir combien coûte vraiment le passage au profil 8 bits sur le téléphone.
3. **La mémoire du cache KV sur les chansons longues.** Elle prend environ 112 Ko par jeton et par séquence, ×2 sous CFG, soit environ 1,2 Go pour une chanson de 3 min.

   Si une chanson longue bute sur la porte mémoire, il faut le dire. C'est le seul cas où une compression du KV (type TurboQuant) vaudrait le chantier.

## Ce qui a été essayé et écarté (inutile de le redemander)

- **Fusion des projections q/k/v et gate/up** : aucun gain.
- **Décodage compilé** : aucun gain.
- **Mélange 4/8 bits des couches de l'AR** : la qualité baisse au rythme où la vitesse monte. Le couplet se perd déjà avec 7 couches sur 28 en 4 bits.
- Le détail est dans `docs/knowledge/log.md`, aux entrées du 2026-11-26 et du 2026-11-27.
