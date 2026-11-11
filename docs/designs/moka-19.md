# Un modèle Moka 19×19 distillé de KataGo

*   **Author:** corenting
*   **Status:** In Progress (pilote terminé)
*   **Date:** 2026-11-10

--------------------------------------------------------------------------------

## Overview

Le moteur sait déjà jouer en 19×19 : règles, recherche, parité et mémoire
tournent sur `synthetic-19` (`zig-wasm-nxn.md`, `zig-search.md`). Il lui
manque un réseau qui joue. Ce design refait la recette de Moka v2 en 19×19,
au même format : un tronc v2 (48 canaux, goulot de 24, 16 blocs), les mêmes
12 plans d'entrée, les mêmes têtes denses, le même `.bin` INT8. Le moteur et
le runtime de Moka restent donc la référence de parité, sans une ligne de
Zig à changer.

Le professeur est le réseau KataGo b18c384nbt, natif en 19×19, piloté par
son moteur d'analyse. KataGo joue contre lui-même sous les règles de Moka et
étiquette chaque position. Le pipeline Python de `pipeline/`, figé en 9×9,
devient paramétré par la taille du plateau et par l'architecture. Un pilote
mesure le débit et la courbe d'apprentissage avant tout gros corpus.

## Hypothèses

Elles viennent de la proposition et n'ont pas été confirmées. Le travail
avance avec elles :

1.  **Format Moka inchangé :** têtes denses, 12 plans, `.bin` INT8. Le
    runtime de Moka reste la référence de parité.
2.  **Calcul local :** tout tourne sur cette machine, un Ryzen 16 cœurs /
    32 threads avec une Radeon gfx1030 de 16 Go (famille RX 6800 / 6900 XT)
    et 31 Go de RAM.
3.  **Budget navigateur :** environ 800 Ko de poids et environ 3 ms par
    inférence.

## The Problem

*   **Les têtes deviennent l'essentiel du modèle.** Avec le tronc de v2,
    `policy.linear` passe à 362 × 1 444 en 19×19 : 523 000 poids, près de
    70 % du réseau. Le fichier fait 776 368 octets, contre 260 048 en 9×9.
*   **Le pipeline était figé en 9×9 et en v1.** Taille, limites de coups,
    dimensions du réseau et validation du manifeste étaient des constantes.
    Il ne savait pas exprimer le tronc de v2.
*   **Le professeur ONNX ne voit pas les échelles.** `katago.py` met les
    plans d'échelle de KataGo à zéro. C'était tolérable en 9×9, pas en
    19×19.
*   **Les données changent d'échelle.** Une position en f32 pèse 17 Ko et
    `train.py` chargeait tout en RAM, avec une augmentation position par
    position sur le CPU.
*   **`train.py` n'avait pas la recette de v2 :** pas de cible de visites,
    pas de résultat de partie, pas de tête auxiliaire, pas de schedule.

## Goals

*   Un pipeline paramétré par N et par l'architecture, identique au bit
    près en 9×9.
*   Les règles Python et le réseau PyTorch vérifiés contre les fixtures de
    l'oracle de chaque modèle, 9×9 et 19×19.
*   Un professeur KataGo natif, échelles comprises, avec des étiquettes
    complètes : politique brute et visites, valeur brute et de recherche,
    score, propriété.
*   Un auto-jeu sous les règles de Moka, des shards compacts, un
    entraînement sur GPU AMD.
*   Un pilote qui donne le débit réel et la courbe d'apprentissage.
*   Le coût d'inférence d'un v2 19×19 dans le moteur, mesuré.

## Non-Goals

*   Changer le format, les plans d'entrée ou le moteur Zig (phase 2).
*   Le corpus on-policy (voir Future Improvements).
*   Publier un modèle 19×19 : le pilote ne produit pas un modèle à jouer.

## Technical Plan

*   **Pipeline par N (`pipeline/`).** `BoardSpec` porte la taille et les
    limites de Moka (19×19 : 540 coups au plus, passe à partir du coup 89,
    `oracle/constants-19.ts`). `Architecture` décrit le réseau comme
    l'objet `architecture` de `go-model.json`, avec les contrôles de
    `tools/gen_model_layout.zig` ; les presets `v1` et `v2` s'instancient
    en 9×9 ou en 19×19. Un checkpoint sans architecture se charge en v1.
*   **Parité de l'oracle (`tests/test_oracle_parity.py`).** Chaque partie
    des fixtures de `moka-9`, `moka-9-v2` et `synthetic-19` est rejouée par
    `board.py` : features au bit près et coups légaux. Le réseau PyTorch
    reproduit les sorties du runtime de Moka à la précision du f32.
*   **Professeur (`analysis.py`).** `katago analysis` reçoit une requête
    JSON par position : les coups de la partie, les règles de Moka (ko
    simple, score de zone, pas de suicide, komi 7) et le nombre de visites.
    La configuration (`pipeline/katago/analysis.cfg`) rend toutes les
    valeurs du point de vue du joueur au trait. Les réponses réelles de
    KataGo 1.18.1 rangées dans `tests/fixtures/katago/` servent de tests.
*   **ROCm (`pipeline/katago/katago-rocm`).** Le build ROCm de KataGo est
    lancé avec les bibliothèques ROCm de la roue torch, liées sous leur
    soname : aucune installation ROCm système n'est nécessaire.
*   **Auto-jeu (`selfplay.py`).** 64 parties à la fois, une requête en vol
    par partie. Un quart des positions reçoit une recherche de 64 visites,
    les autres une recherche de 8 visites qui fait avancer la partie
    (playout cap randomization) ; toutes gardent la politique et la valeur
    brutes. Les premiers coups suivent la politique brute, les suivants les
    visites à température 0,6, puis le meilleur coup, sauf 6 % de coups de
    la politique brute. Pas de passe avant le coup 89, ni dans les coups
    joués ni dans les cibles. Abandon autorisé dans 70 % des parties
    (|v| ≥ 0,94 pendant 6 positions). Le résultat d'une partie jouée
    jusqu'au bout est le signe du score de KataGo à la dernière position.
*   **Shards (`shards.py`).** Un répertoire de `.npy` par shard : les
    11 plans binaires bit-packés, le komi en scalaire, les deux politiques
    en f16, la propriété en int8. Environ 2,4 Ko par position en 19×19,
    contre 17 Ko pour les seules features en f32.
*   **Entraînement (`train.py`).** Les tenseurs restent sur le GPU quand
    ils tiennent ; les plans sont décodés et une des 8 symétries est
    appliquée par position, sur le GPU. Cibles : politique 50 % visites,
    50 % politique brute là où il y a une recherche ; valeur 70 % recherche,
    30 % résultat. Deux têtes d'entraînement seulement, propriété et
    score, qui ne sont pas exportées. AdamW, warmup puis cosinus. Un départ
    à chaud depuis le 9×9 garde le tronc et crée de nouvelles têtes. La
    sélection reste faite sur le snapshot INT8.
*   **Coût dans le moteur (`synthetic-19-v2`).** `pnpm models` génère un
    second modèle synthétique, avec le tronc de `vendor/moka-v2` et des
    têtes 19×19 générées par la recette de `synthetic-19`. Il a la forme
    exacte d'un v2 19×19 entraîné. `build.zig` le déclare avec 81 pages
    pour `infer.wasm` et 581 pour `search.wasm` (les plus petites tailles
    qui laissent 25 % de marge), et le même arbre que `synthetic-19`.

## Critères

Fixés avant le premier match, comme pour v2, et repris tels quels de la
proposition :

*   poids sous le budget (environ 800 Ko) ;
*   parité Zig / Moka ;
*   200 parties à 1 s contre un profil human SL de KataGo, au rang cible
    fixé après le pilote. Ce rang est `rank_5k`, où le pilote fait 59,5 %
    [52,7 ; 66,3] avec la fin de partie corrigée de la recherche (voir
    Banc 19×19). Un candidat doit y faire au moins 67 % (+123 Elo contre
    le profil, environ +55 sur le pilote), au-dessus de l'intervalle du
    pilote. Au-delà d'environ 75 %, le rang de référence passe au profil
    suivant (`rank_3k`, pas encore joué). Le seuil vaut pour les
    conditions du banc : 6 matchs en parallèle, aucun entraînement à côté,
    la même recherche. Un changement de la recherche demande de rejouer le
    pilote, comme pour le compte de fin de partie : le seuil était de 60 %
    sur les 52,5 % de l'ancienne recherche ;
*   au moins 55 victoires sur 100 contre la version précédente.

L'arbitre est le score de KataGo, pas le retrait des pierres mortes de
Moka (`force-zig-v2.md`).

## Résultats

### Outillage

*   **PyTorch ROCm.** torch 2.14.1+rocm7.2 voit la Radeon (gfx1030) par HIP.
    Un pas d'entraînement de v2 en 19×19, lots de 256 : 1 677 positions/s
    en f32, 2 810 en fp16, 4 185 en bf16. Contrairement à ce que supposait
    la proposition, le bf16 est le plus rapide sur cette carte.
*   **`cudnn.benchmark` sous ROCm.** Il lance la recherche exhaustive de
    MIOpen pour chaque forme de convolution : plus de dix minutes avant le
    premier pas, contre 6 s avec les heuristiques par défaut.
    `device.configure` le laisse coupé sous ROCm.
*   **KataGo** (`katago benchmark`, b18c384nbt, 19×19) :

| Build | Threads | Évaluations/s | Lot moyen |
|---|---:|---:|---:|
| OpenCL | 16 | 312 | — |
| OpenCL | 48 | 358 | 23 |
| ROCm | 16 | 1 147 | 7,9 |
| ROCm | 32 | 1 358 | 15,3 |
| ROCm | 64 | 1 551 | 29,9 |

### Professeur et règles

*   Les réponses réelles confirment : les règles de Moka en JSON sont
    acceptées et une clé inconnue est une erreur ; `pass` est un coup
    valide ; `analyzeTurns` donne une réponse par tour ; `rawWinrate` est
    présent ; la propriété est du point de vue du joueur au trait ; l'indice
    de la politique correspond à celui de Moka.
*   Les visites des enfants peuvent dépasser celles de la racine : KataGo
    cherche un coup par classe de symétrie et le recopie sous chaque coup
    équivalent. La distribution des visites est normalisée.
*   Sur 1 446 positions d'auto-jeu, dont 12 avec un ko, KataGo ne donne
    jamais de masse à un coup que Moka refuse, et chaque coup joué se
    rejoue dans `board.py`.

### Entraînement

*   Le trainer apprend : sur les 13 061 positions d'un premier auto-jeu,
    60 époques font passer la loss de politique d'entraînement de 5,88 à
    2,29 et l'accord top-1 d'entraînement de 1 % à 59 % (27 % sur les
    parties de validation, avec seulement 64 parties en tout).
*   L'export de v2 en 19×19 fait 776 368 octets et passe
    `tools/gen_model_layout.zig`.

### Pilote

Commandes : `go-selfplay --positions 250000 --games-in-flight 64
--shard-rows 5000 --seed 2` vers `data/pilot-19` (non versionné), puis
quatre entraînements v2 d'environ 2 M d'échantillons chacun, et un jeu de
test produit après coup.

**Auto-jeu.**

*   253 468 positions, 1 211 parties, 63 431 recherches de 64 visites
    (25,0 %), en 2 086 s. Débit sur des fenêtres de 5 minutes : 122 à
    130 positions/s après la première, soit environ 125 positions/s une
    fois chaud ; 1 M de positions demandent environ 2 h 15.
*   Aucune partie perdue. 815 abandons (67 %), 396 fins par deux passes.
    Résultats : Noir 599, Blanc 539, jigo 73. Le komi entier (7) sous le
    score de zone permet le jigo : 18 % des parties jouées jusqu'au bout.
*   Longueur moyenne 209 coups (médiane 183, de 18 à 405). 84 parties
    (7 %) font moins de 80 coups, toutes abandonnées après un combat de
    coin issu de l'ouverture tirée de la politique brute.
*   594 Mo sur disque, 2,3 Ko par position.
*   Mémoire de KataGo : 2,1 Go au départ, 4,9 Go après 35 minutes, avec
    une croissance qui ralentit de 240 à 17 Mo par minute. La courbe est
    celle du remplissage du cache du réseau (2^20 entrées, politique et
    propriété en 19×19), pas celle d'une fuite, mais le plafond n'est pas
    mesuré. Le redémarrage par shard n'est pas nécessaire à cette
    échelle.

**Entraînements.** Lots de 256, bf16, AdamW à 1e-3 sans weight decay,
cosinus, tronc v2 (758 889 paramètres). La validation des entraînements
(les deux derniers shards) s'est révélée biaisée : ce sont les 19 parties
encore en cours quand l'auto-jeu a cessé d'en lancer, toutes finies par
deux passes, de 345 coups en moyenne. Elle n'a servi qu'à choisir
l'époque (la dernière ou presque dans les quatre courses).

Les mesures viennent donc d'un jeu de test à part : `go-selfplay
--positions 20000 --seed 3` vers `data/pilot-19-test`, avec le code
versionné. Toute partie lancée y est finie : 25 129 positions, 123 parties
(84 abandons, 39 fins par deux passes, 204 coups en moyenne, comme le
corpus), aucun échec, aucun repli (`fallbacks` à 0). Le KL est l'entropie
croisée de la politique moins l'entropie de la cible (1,572). La valeur est
comparée à celle du professeur seule ; prédire 0 donne une MSE de 0,414.
Tout est mesuré sur le snapshot INT8, à l'époque retenue :

| Corpus | Parties | Époques | Durée | KL | Accord top-1 | MSE valeur | Corr. valeur |
|---|---:|---:|---:|---:|---:|---:|---:|
| quart (12 shards) | 307 | 32 | 539 s | 1,539 | 40,6 % | 0,625 | 0,07 |
| moitié (24 shards) | 604 | 16 | 499 s | 1,410 | 41,9 % | 0,542 | 0,15 |
| tout (48 shards) | 1 192 | 8 | 525 s | 1,347 | 43,2 % | 0,455 | 0,26 |
| tout, tronc de v2 9×9 | 1 192 | 8 | 491 s | **1,229** | **45,3 %** | **0,406** | **0,36** |

*   **La politique apprend et profite des données.** À budget
    d'échantillons égal, le KL passe de 1,54 à 1,35 quand le corpus
    quadruple. La course sur tout le corpus n'a pas convergé en
    8 époques, et l'écart entre entraînement et test reste faible
    (44,7 % contre 43,2 % d'accord).
*   **La valeur est en retard.** Partie de zéro, elle fait moins bien que
    la constante 0 sur les trois corpus. Elle progresse avec les données
    (corrélation de 0,07 à 0,26), mais sa MSE d'entraînement tombe à
    0,09, 0,11 et 0,19 : la couche dense (2 canaux × 361 points → 32 → 1)
    apprend les parties par cœur plus vite qu'elle n'apprend à évaluer.
    Même constat sur les shards 24 à 47, jamais vus par le quart ni par la
    moitié (588 parties) : corrélation 0,13 et 0,12.
*   **Ce n'est pas un défaut des étiquettes.** Sur le jeu de test, la
    propriété du professeur vaut +0,71 sous les pierres du joueur au
    trait et −0,71 sous celles de l'adversaire, dans le bon sens pour
    99,5 % des positions : features et réponses ont la même perspective.
*   **Le tronc apprend le territoire.** La tête auxiliaire de propriété
    généralise : MSE de 0,070 (tout) et 0,050 (départ à chaud) contre
    0,442 pour la constante. Sa somme, komi compris, suit mieux la valeur
    du professeur que la tête de valeur elle-même (corrélation 0,37
    contre 0,26 ; 0,43 contre 0,36 au départ à chaud).
*   **Le tronc de v2 9×9 aide.** Le départ à chaud est en tête sur toutes
    les mesures. C'est la seule course dont la valeur rejoint la
    constante, sans la battre nettement ; elle sur-apprend encore (MSE
    d'entraînement 0,17).
*   **Débit :** 3 690 positions/s sur le chemin déterministe (le quart),
    environ 4 050 sur l'autre. Le chemin déterministe tourne donc sur la
    Radeon ; sa reproductibilité au bit près n'est pas vérifiée.
*   Le jeu de test compte 123 parties ; les positions d'une même partie
    sont corrélées. Les écarts de politique sont réguliers d'une course à
    l'autre ; ceux de valeur sont à confirmer sur plus de parties.

**Ce que le pilote dit pour la suite.**

1.  Partir du tronc de v2 9×9, pas de zéro.
2.  La politique justifie un corpus plus grand : environ 2 h 15 par
    million de positions avec ce professeur.
3.  La valeur est le risque. Avec 1 200 parties, la tête dense de 19×19
    retient les parties plus vite qu'elle n'évalue la position, et le
    tronc en sait plus qu'elle (la somme de la propriété la devance).
    Avant un corpus de dix fois plus de parties, essayer à coût nul :
    weight decay, moins de positions par partie, poids de la valeur. Si
    elle reste derrière la propriété, c'est l'argument de la phase 2 : une
    tête de valeur indépendante de N, par pooling comme celle de KataGo.
    Le weight decay et les poids sont essayés (voir Réglages de la
    valeur) ; moins de positions par partie ne l'est pas.
4.  Le banc 19×19 est venu après le pilote (voir Banc 19×19).
5.  Le jeu de validation d'un prochain entraînement doit être produit à
    part, comme ce jeu de test, et non pris dans les derniers shards.

### Banc 19×19

**Outillage.**

*   **`match/dist/match-19.js`** (`pnpm match:19`) est le banc 9×9
    recompilé par esbuild avec `oracle/constants-19.ts` à la place des
    constantes de Moka. Il garde les mêmes joueurs, les mêmes ouvertures de
    4 coups tirées de la politique, jouées deux fois en échangeant les
    couleurs, et le même arbitrage par KataGo. Il n'a pas de modèle par
    défaut ; `match.js` reste en 9×9 avec Moka par défaut.
*   **Modèles locaux.** Un export du pipeline se joue sans être versionné.
    Il va dans `models/local/<id>/`, ignoré par git, puis
    `zig build models wasm search -Dlocal-model=<id>:synthetic-19-v2` le
    construit avec la mémoire du modèle enregistré dans `zig-out/local/`.
    Les scripts et les tests qui listent `zig-out/models/` ne le voient
    pas.
*   **Contrôle de parité.** Avant chaque match, chaque moteur Zig est
    comparé au runtime de Moka sur les 9 premières positions de la ligne
    gloutonne. Le match s'arrête si un logit diffère de plus de 0,05 ou la
    valeur de plus de 0,01.
*   **Adversaire.** KataGo 1.18.1 ROCm, réseau human SL
    `b18c384nbt-humanv0` avec `match/katago-human.cfg`. Les 16 visites du
    réseau b18 ne décident que des passes. Chaque processus prend environ
    1 Go de RAM.
*   **Une tâche GPU à la fois.** Huit processus KataGo ROCm à côté d'un
    entraînement saturent les files matérielles de la carte (« Runlist is
    getting oversubscribed » dans le journal du noyau). Des fautes de page
    GPU tuent ensuite KataGo puis l'entraînement. Six processus à côté d'un
    entraînement sont passés, mais l'entraînement tombe alors de 3 600 à
    1 500 à 2 000 positions/s.

**Le pilote contre les profils humains.** Le modèle est `pilot-all-warm`
(départ à chaud du tronc de v2 9×9), exporté en `moka-19-pilot`. Son sha256
est `7258fc2109054ccb35f1d8eec6d1b2e181c765ccbe8eef3c1f44f019c38927eb`.
`zig-policy` joue le meilleur coup de la politique brute ; `zig-reuse`
cherche 1 s par coup en gardant l'arbre.

| Joueur | Profil | Parties | Score du pilote [IC 95 %] |
|---|---|---:|---|
| `zig-policy` | `rank_20k` | 2 | 2 victoires (+76 et +210) |
| `zig-policy` | `rank_10k` | 20 | 77,5 % [59,2 ; 95,8] |
| `zig-policy` | `rank_5k` | 20 | 20,0 % [2,5 ; 37,5] |
| `zig-policy` | `rank_1k` | 20 | 0 % |
| `zig-reuse`, 1 s | `rank_5k` | 20 | 52,5 % [30,6 ; 74,4] |
| `zig-policy` | `rank_5k` | 200 | **18,5 % [13,1 ; 23,9]** |
| `zig-reuse`, 1 s, ancienne fin | `rank_5k` | 200 | 52,5 % [45,6 ; 59,4] |
| `zig-reuse`, 1 s | `rank_5k` | 20 | 60,0 % [38,5 ; 81,5] |
| `zig-reuse`, 1 s | `rank_5k` | 200 | **59,5 % [52,7 ; 66,3]** |

Les lignes en gras sont la référence (6 matchs en parallèle, sans
entraînement à côté). « Ancienne fin » est la recherche d'avant le
correctif du compte de fin de partie (voir Arbitrage), les lignes
suivantes la recherche corrigée. Les fichiers de résultats restent dans
`bench/results/`, non versionnés.

*   **La politique seule** se place entre `rank_10k` et `rank_5k` :
    −258 Elo [−328 ; −201] contre `rank_5k`.
*   **La recherche aide déjà**, malgré une tête de valeur à peine
    meilleure que la constante : de 18,5 % à 59,5 %, environ +325 Elo.
    `zig-reuse` fait 412 visites par coup à 1 s, soit 2,4 ms par visite
    arbre compris.
*   **`rank_5k` reste le rang de référence** : le pilote y est proche de
    l'équilibre, loin des deux bornes, si bien qu'un progrès comme un
    recul s'y mesure (voir Critères).
*   **Arbitrage.** Avec la politique seule, KataGo et le retrait des
    pierres mortes de Moka désignent le même vainqueur dans 197 parties
    sur 200. Avec l'ancienne recherche, ils divergeaient dans 19 parties
    sur 200, toujours au profit de Zig selon Moka : 17 défaites et une
    nulle selon KataGo étaient des victoires selon Moka, une défaite une
    nulle, avec un écart de 4 à 69 points. Compté par Moka, `zig-reuse`
    aurait fait 61,5 %.
*   **La cause est le compte de fin de la recherche.** Avant de compter,
    Moka retire les pierres mortes au sens de Benson (dans la zone
    vivante en passant de l'adversaire), puis chaque camp à son tour joue
    la plus grosse capture immédiate, ou passe, jusqu'à deux passes
    (`getAutomaticallyDeadMoves`). Le score
    final de KataGo ne joue pas ces captures. La recherche comptait donc
    ses fins de partie comme si Zig avait pris des pierres qu'il n'a
    jamais prises, et passait au lieu de les capturer. Dans les 19
    parties, des pierres restent en atari à la fin ; dans 17, le compte
    brut, sans rien retirer, est exactement celui de KataGo. Sur les 400
    parties des deux lignes de 200, le retrait des seules pierres mortes
    au sens de Benson désigne le vainqueur de KataGo dans les 400, le
    compte complet de Moka dans 378.
*   **Le correctif** (`searchScore` dans `src/search/tree.zig`,
    `getSearchScore` dans `match/search.ts`) : en 19×19, la recherche
    compte une partie finie avec ce seul retrait, sans les captures
    d'après-partie. Le 9×9 garde le compte de Moka. Rejouées sur CPU, au
    dernier passe de Zig des 19 parties, l'ancienne recherche passe 17
    fois et la nouvelle jamais ; dans 20 parties gagnées par Zig, les deux
    passent 19 fois. Sur 200 parties, KataGo et Moka ne divergent plus
    que dans une, gagnée par Zig selon KataGo.
*   **Le gain au match est cohérent, pas prouvé seul.** Les deux séries
    de 200 jouent les mêmes ouvertures avec les mêmes couleurs (graine 1),
    mais une même ouverture donne presque deux tirages indépendants : une
    partie perdue la première fois est gagnée la seconde dans 60 % des
    cas, une partie gagnée l'est encore dans 57 %. Le solde est de
    +14 victoires (60 parties améliorées, 46 dégradées, test du signe
    p = 0,21). C'est ce qu'on attend de 19 parties perdues au compte qui
    redeviennent des parties ordinaires (environ 11 victoires, +5,5
    points), pas une preuve à elle seule.
*   **Moins de visites, seulement en fin de partie.** De 470 à 412
    visites par coup sur 200 parties, mais le nombre de visites avant le
    coup 300 ne change pas (401, 377 et 431 par tranche de 100 coups
    avant, 402, 374 et 418 après). Après le coup 300, il passe de 1 323
    à 678 : une fin de partie se compte sans inférence, et l'ancienne
    recherche en atteignait plus en passant tôt (explication probable,
    pas mesurée).

### Réglages de la valeur

Les essais « à coût nul » du pilote (point 3), sur le même corpus de
48 shards. Chaque course reprend la recette de `pilot-all-warm` (départ à
chaud, 8 époques, lots de 256, AdamW à 1e-3, graine 1) et change un ou
plusieurs réglages. Les défauts sont : weight decay 0, cible de valeur à
30 % le résultat de la partie et 70 % le professeur (`--outcome-mix 0.3`),
poids de la valeur 1, poids de la propriété 0,5. La validation, qui
choisit l'époque, est un jeu produit à part (`go-selfplay --positions
20000 --seed 4` vers `data/pilot-19-val`, 25 334 positions). Toutes les
courses gardent la dernière époque. Les mesures sont celles du jeu de
test du pilote, sur le snapshot INT8 :

| Course | Réglage | KL | Accord top-1 | MSE valeur | Corr. valeur | Signe |
|---|---|---:|---:|---:|---:|---:|
| base | aucun | 1,229 | 45,3 % | 0,406 | 0,355 | 58,9 % |
| seed2 | `--seed 2` | 1,215 | 45,8 % | 0,368 | 0,417 | 59,6 % |
| wd01 | `--weight-decay 0.1` | 1,210 | 46,1 % | 0,381 | 0,397 | 60,4 % |
| om0 | `--outcome-mix 0` | 1,222 | 45,2 % | 0,384 | 0,391 | 60,4 % |
| vw05 | `--value-weight 0.5` | 1,219 | 45,6 % | 0,399 | 0,366 | 58,7 % |
| own15 | `--ownership-weight 1.5` | 1,247 | 45,0 % | 0,385 | 0,404 | 60,1 % |
| combo | wd01 + om0 | 1,209 | 45,4 % | 0,380 | 0,395 | 61,0 % |
| combo-own | wd01 + om0 + own15 | 1,219 | 45,1 % | 0,365 | 0,420 | 60,6 % |
| seed3 | `--seed 3` | 1,258 | 44,9 % | 0,395 | 0,366 | 58,5 % |
| combo-s2 | combo, `--seed 2` | 1,201 | 46,0 % | 0,371 | 0,415 | 62,4 % |
| combo-s3 | combo, `--seed 3` | 1,238 | 45,3 % | 0,361 | 0,428 | 61,3 % |

Sur trois graines, la base fait en moyenne KL 1,234, MSE 0,390,
corrélation 0,379 et signe 59,0 % ; combo fait KL 1,216, MSE 0,371,
corrélation 0,413 et signe 61,6 %.

*   **Reproductibilité.** La course de base redonne les poids de
    `pilot-all-warm` au bit près (écart maximal 0,0), sur le chemin non
    déterministe et avec un autre jeu de validation : la même époque est
    retenue. Refaire une course ne mesure donc pas le bruit ; il faut
    changer de graine.
*   **Le bruit d'une graine est large.** Avec la seule graine changée, la
    base va de 1,215 à 1,258 en KL et de 0,368 à 0,406 en MSE de valeur :
    plus que l'effet de tout réglage pris sur une seule graine. Le premier
    balayage, tout en graine 1, ne départage donc rien seul.
*   **À graine égale, combo fait mieux que la base.** La graine fixe
    l'ordre des données, le tirage des symétries et l'initialisation des
    nouvelles têtes, ce qui apparie les courses. Sur les trois paires, combo baisse le KL (de 0,014 à 0,020)
    et améliore le signe de la valeur (de 2,1 à 2,8 points). La MSE baisse
    dans deux paires sur trois (0,026 et 0,034) et monte de 0,003 dans la
    troisième. Trois paires de même sens restent un indice, pas une
    preuve.
*   **Le poids de la propriété coûte à la politique** : 1,247 seul en
    graine 1, et de 1,209 à 1,219 quand on l'ajoute à combo, pour un gain
    de valeur que la graine seule donne aussi.
*   **La valeur reste derrière la propriété.** La meilleure corrélation
    (0,428) atteint à peine celle de la somme de la propriété de
    `pilot-all-warm` (0,43). Le jeu de test compte 123 parties, et ces
    écarts sont à la taille de son bruit.
*   **Pour le prochain corpus : combo** (`--weight-decay 0.1
    --outcome-mix 0`). Il ne perd sur aucune paire, et la cible du
    professeur seule est celle d'une distillation. Le poids de la
    propriété reste à 0,5. Le gain se vérifiera au match contre
    `rank_5k`.

### Moteur

`pnpm check:all` passe : 190 tests, dont la parité de l'inférence, la
mémoire constante et la recherche de `synthetic-19-v2`, et 16 tests e2e.
Le modèle et ses données de référence se régénèrent octet pour octet
(`pnpm models:check`, `pnpm oracle:check`). En 19×19, l'oracle ne produit
qu'un jeu de validation : la parité Python saute le jeu de calibration.

Relevé du 2026-11-10 (`pnpm bench --model synthetic-19-v2`, puis
`--model synthetic-19` ; fichiers `bench/results/2026-11-10T12-51-19-488Z.json`
et `2026-11-10T12-52-52-493Z.json`, non versionnés), sur un AMD Ryzen 9
5950X, Chromium 156.0.8078.4, Node 26.13.0. Médiane des p50 de 5 tours, en
ms, à `count = 1` :

| | `synthetic-19-v2` | `synthetic-19` |
| --- | --- | --- |
| Tronc | v2 : 48/24, 16 blocs | v1 |
| sha256 | `ffaf7b7aed50c3dbb2469fc9ab432796c5a82e47ca921b83b3319566488569dc` | `55f33968b08da4b255dbf04a6f468efe1691cce4bc258498e8f8c73dd44ebb44` |
| Moka, Worker | 77,785 | 29,020 |
| Moka, sans Worker | 77,610 | 29,030 |
| Zig, Worker | 2,965 | 1,135 |
| Zig, sans Worker | 2,900 | 1,100 |
| Zig, tronc seul | 2,800 | 1,000 |
| Rapport (Moka / Zig, Worker) | 26,23 | 25,57 |
| Part des têtes | 3,4 % | 9,1 % |

*   Le tronc de v2 coûte 2,8 fois celui de v1 en 19×19 ; les têtes
    denses, malgré leurs 523 000 poids, ne font que 3 % du temps.
*   **Budget :** 2,9 ms par inférence, juste sous les 3 ms de
    l'hypothèse 3, sur cette machine. `synthetic-19` faisait 1,98 ms sur
    le Ryzen 7 8845HS du relevé de `zig-wasm-nxn.md`, contre 1,10 ms ici :
    à ce rapport, un v2 19×19 y prendrait environ 5 ms (extrapolé, pas
    mesuré).
*   En une seconde de recherche, cela fait environ 340 évaluations sur
    cette machine, sans compter le coût de l'arbre.

### Site

`moka-19` est la graine 2 de la ronde `moka-19-r1`, publiée dans
`models/moka-19/` (provenance et licence KataGo à côté) avant le match
contre `rank_5k`, remis à plus tard : sa force en partie n'est pas mesurée.

*   Le site a deux éditions, chacune dans son dossier de `site/public` :
    9×9 (Moka v2) à la racine, 19×19 (`moka-19`) dans `19x19/`. Chacune a
    ses bundles, construits avec ses constantes (`scripts/build-js.mjs`),
    son wasm et son modèle ; la page, la feuille de style et les licences
    sont communes.
*   **Arbitrage.** Après deux passes, la page marque les pierres que
    l'estimation de Moka juge mortes, et le joueur corrige : un clic sur un
    groupe le marque mort ou vivant, le score suit, puis il accepte. Undo
    revient à la partie. Moka ne sait pas compter une fin de partie 19×19
    disputée (voir Banc 19×19, Arbitrage), et le modèle exporté n'a pas de
    tête de propriété : le joueur reste l'arbitre, en 9×9 comme en 19×19.

## Tests

*   `pipeline/tests/test_pipeline.py` : presets, aller-retour d'export en
    19×19, validation du manifeste calquée sur Zig, ancien checkpoint chargé
    en v1, tirage de coups légaux.
*   `pipeline/tests/test_oracle_parity.py` : règles, features, coups légaux
    et sorties du réseau contre les fixtures de l'oracle de chaque modèle.
    Il saute un modèle dont les fixtures ne sont pas générées.
*   `pipeline/tests/test_analysis.py` : requêtes et réponses réelles de
    KataGo 1.18.1, et le client contre un rejoueur de ces réponses (sans
    GPU).
*   `pipeline/tests/test_selfplay.py` : l'auto-jeu contre un faux KataGo en
    9×9 ; les shards se relisent, chaque ligne correspond à sa position
    dans la partie, les résultats et les cibles sans passe précoce sont
    vérifiés.
*   `pipeline/tests/test_train_smoke.py` : cibles des shards, décodeur et
    symétries du GPU contre les références CPU, entraînement avec têtes
    auxiliaires, départ à chaud du 9×9 vers le 19×19.
*   En 9×9, rien ne change : l'export d'un modèle semé, le re-export des
    poids publiés et leurs sorties gardent les mêmes empreintes qu'avant
    le design.

## Future Improvements

*   **Corpus on-policy.** v2 a montré que l'auto-jeu de b18 seul ne suffit
    pas (19/100 contre b6c96, 50/100 avec le corpus on-policy). Il faut le
    même en 19×19 : l'élève contre des profils human SL, b18 qui étiquette.
*   **Compte de fin de la recherche en 9×9.** Le correctif du 19×19
    (voir Banc 19×19, Arbitrage) ne touche pas le 9×9, où la recherche
    compte toujours avec les captures d'après-partie de Moka. Le même
    changement s'y mesurerait sur le banc de v2, sans réentraîner.
*   **Redémarrage de KataGo par shard.** Prévu par la proposition, pas
    implémenté : pendant le pilote, la mémoire suit le remplissage du cache
    du réseau et ralentit vers un plafond (voir Pilote).
    `pipeline/scripts/round-19.sh` relance KataGo par tranche de 250 000
    positions, la taille du pilote, ce qui suffit tant qu'une tranche reste
    sous environ 6 Go.
*   **`katago selfplay` et un convertisseur** de ses npz vers les 12 plans
    de Moka, si l'étiquetage par le moteur d'analyse ne suffit pas.
*   **Chemin déterministe sur GPU.** Il tourne sur la Radeon (le quart du
    pilote) ; deux courses identiques n'ont pas été comparées au bit près.
*   **Phase 2.** Têtes convolutives indépendantes de N (environ 250 Ko au
    lieu de 776 Ko) et plans d'échelle dans l'encodage, côté `rules.zig` et
    côté Python. Le pilote y ajoute un argument : avec 1 200 parties, la
    tête de valeur dense atteint à peine la constante 0.
*   **Licence.** Les réseaux kata1 sont sous une licence de type MIT dont la
    notice est à reproduire ; elle ne dit rien de la distillation.
