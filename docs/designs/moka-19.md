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
    fixé après le pilote ;
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
quatre entraînements v2 d'environ 2 M d'échantillons chacun. La
validation est fixe : les deux derniers shards, 6 559 positions de
19 parties, jamais vues par aucun entraînement.

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
cosinus, tronc v2 (758 889 paramètres). Le KL est l'entropie croisée de
la politique moins l'entropie de la cible (1,777 sur la validation).
Valeur : MSE contre la valeur du professeur seule (prédire 0 donne
0,596), et sa corrélation. Snapshot INT8, meilleure époque :

| Corpus | Parties | Époques | Durée | KL | Accord top-1 | MSE valeur | Corr. valeur |
|---|---:|---:|---:|---:|---:|---:|---:|
| quart (12 shards) | 307 | 32 | 539 s | 1,612 | 38,6 % | 0,691 | 0,30 |
| moitié (24 shards) | 604 | 16 | 499 s | 1,424 | 40,8 % | 0,703 | 0,24 |
| tout (48 shards) | 1 192 | 8 | 525 s | 1,381 | 42,2 % | 0,639 | 0,31 |
| tout, tronc de v2 9×9 | 1 192 | 8 | 491 s | **1,254** | **43,8 %** | **0,473** | **0,51** |

*   **La politique apprend et profite des données.** À budget
    d'échantillons égal, le KL passe de 1,61 à 1,38 quand le corpus
    quadruple. La course sur tout le corpus n'a pas convergé en
    8 époques, et l'écart entre entraînement et validation reste faible
    (44,7 % contre 42,2 % d'accord).
*   **La tête de valeur ne généralise pas.** Partie de zéro, elle fait
    moins bien que la constante 0 sur des parties jamais vues : sur les
    shards 24 à 47 (588 parties, 123 370 positions), le quart et la
    moitié donnent une MSE de 0,58 et 0,56 contre 0,40 pour la constante,
    une corrélation de 0,13 et 0,12, le bon signe sur 55 % des positions.
    Leur MSE de valeur d'entraînement tombe à 0,09 et 0,11 : la couche dense
    (2 canaux × 361 points → 32 → 1) apprend les parties par cœur dès la
    deuxième époque. La politique, elle, garde le même niveau sur ces
    588 parties (KL 1,55 et 1,42, accord 40,6 % et 41,6 %).
*   **Ce n'est pas un défaut des étiquettes.** Sur ces 588 parties, la
    propriété du professeur vaut +0,71 sous les pierres du joueur au
    trait et −0,71 sous celles de l'adversaire, dans le bon sens pour
    99,5 % des positions : features et réponses ont la même perspective.
*   **Le tronc apprend le territoire.** La tête auxiliaire de propriété
    généralise sur ces parties : MSE 0,071 contre 0,451 pour la constante.
    Sa somme, komi compris, suit mieux la valeur du professeur que la
    tête de valeur elle-même (corrélation 0,31 contre 0,12).
*   **Le tronc de v2 9×9 aide.** Le départ à chaud est en tête sur toutes
    les mesures, et c'est la seule course dont la valeur bat la
    constante. Sa valeur sur-apprend encore (MSE d'entraînement 0,17).
*   **Débit :** 3 690 positions/s sur le chemin déterministe (le quart),
    environ 4 050 sur l'autre. Le chemin déterministe tourne donc sur la
    Radeon ; sa reproductibilité au bit près n'est pas vérifiée.
*   La validation ne compte que 19 parties : les écarts de valeur entre
    les trois courses parties de zéro sont dans le bruit. Celui du départ à
    chaud est plus net (corrélation 0,51 contre 0,24 à 0,31), mais reste à
    confirmer sur plus de parties.

**Ce que le pilote dit pour la suite.**

1.  Partir du tronc de v2 9×9, pas de zéro.
2.  La politique justifie un corpus plus grand : environ 2 h 15 par
    million de positions avec ce professeur.
3.  La valeur est le risque. Avec 1 200 parties, la tête dense de 19×19
    retient les parties au lieu d'évaluer la position. Avant un corpus de
    dix fois plus de parties, essayer à coût nul : weight decay, moins de
    positions par partie, poids de la valeur. Si elle ne généralise
    toujours pas, c'est l'argument de la phase 2 : une tête de valeur
    indépendante de N, par pooling comme celle de KataGo.
4.  Aucun critère de match n'est évalué : `match/katago.ts` est encore
    limité au 9×9 (voir Future Improvements).

### Moteur

`pnpm test` passe (190 tests), dont la parité de l'inférence, la mémoire
constante et la recherche de `synthetic-19-v2`.

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
*   **`match/katago.ts` paramétré par N.** Il prend encore `GO_BOARD_SIZE`
    des constantes 9×9 de Moka ; les critères de match en dépendent.
*   **Redémarrage de KataGo par shard.** Prévu par la proposition, pas
    implémenté : pendant le pilote, la mémoire suit le remplissage du cache
    du réseau et ralentit vers un plafond (voir Pilote). À revoir si une
    course de plusieurs heures dépasse environ 6 Go.
*   **`katago selfplay` et un convertisseur** de ses npz vers les 12 plans
    de Moka, si l'étiquetage par le moteur d'analyse ne suffit pas.
*   **Chemin déterministe sur GPU.** Il tourne sur la Radeon (le quart du
    pilote) ; deux courses identiques n'ont pas été comparées au bit près.
*   **Phase 2.** Têtes convolutives indépendantes de N (environ 250 Ko au
    lieu de 776 Ko) et plans d'échelle dans l'encodage, côté `rules.zig` et
    côté Python. Le pilote y ajoute un argument : la tête de valeur dense
    ne généralise pas avec 1 200 parties.
*   **Licence.** Les réseaux kata1 sont sous une licence de type MIT dont la
    notice est à reproduire ; elle ne dit rien de la distillation.
