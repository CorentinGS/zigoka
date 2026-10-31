# Le modèle Moka v2 dans le moteur Zig

*   **Author:** corenting
*   **Status:** Implemented
*   **Date:** 2026-10-31

--------------------------------------------------------------------------------

## Overview

La PR [millionco/moka#3](https://github.com/millionco/moka/pull/3) propose
Moka v2, un réseau 9×9 distillé du réseau KataGo b18. Il garde le format, les
entrées, les têtes et le runtime de v1, mais le tronc grandit : 48 canaux au
lieu de 32, goulot de 24 au lieu de 16, 16 blocs résiduels au lieu de 12.
D'après la PR, il gagne 85 parties sur 100 contre v1, avec 64 visites de la
recherche Python de Moka.

Ce design ajoute v2 au moteur sous l'identifiant `moka-9-v2`, adapte les
noyaux à ses formes, et mesure en match ce que valent un meilleur réseau et
un moteur plus rapide l'un contre l'autre.

## The Problem

*   Le moteur ne connaissait que deux modèles, `moka-9` et `synthetic-19`.
*   Les noyaux traitaient deux positions à la fois seulement jusqu'à 16
    sorties. Les convolutions 3×3 de v2 en ont 24, et toutes passaient donc
    en une position à la fois.
*   On ne savait pas lequel gagne : le réseau v1 avec la recherche rapide de
    `search.wasm`, ou le réseau v2, plus fort mais environ 3 fois plus cher
    par visite.

## Goals

*   `moka-9-v2` construit, testé et mesuré comme `moka-9` : oracle, parité bit
    à bit avec Moka instrumenté pour accumuler en f32, `search.wasm` et sa
    parité avec `match/search.ts`.
*   Des noyaux plus rapides sur les formes de v2, sans perte sur v1 et sans
    changer un seul bit des sorties.
*   Des matchs en 9×9 qui opposent v1 et v2, dans ce moteur et dans le runtime
    de Moka.

## Non-Goals

*   Publier v2 dans `dist/` : `dist/` reste le modèle de la branche
    principale de Moka. La PR n'est pas fusionnée.
*   Changer la recherche, ou reproduire la recherche Python de Moka (symétries
    complètes, 64 visites) qui donne les résultats de la PR.
*   Ajouter v2 à la page de démo ou à la vidéo.

## Technical Plan

*   **Vendorisation.** `vendor/moka-v2/` contient le manifeste, les poids, le
    README, le journal, la licence et les notices du commit `1ad518a` de la
    branche `moka-v2`, avec leurs empreintes dans `VENDORED.md`. La PR ne
    touche pas `src/` : chaque fichier de `src/` y a la même empreinte que
    dans `vendor/moka/`, donc les règles, l'encodage et le runtime de
    `vendor/moka/src/` servent aussi v2. `pnpm oracle` vérifie les deux
    dossiers.
*   **Build.** `build.zig` déclare `moka-9-v2` : 27 pages pour `infer.wasm`
    (la plus petite taille qui laisse 25 % de marge), 346 pages pour
    `search.wasm` depuis la réutilisation de l'arbre, avec le même arbre que `moka-9` (8 192 nœuds,
    524 288 arêtes).
*   **Oracle.** v2 est un modèle entraîné, comme `moka-9` : le contrôle de
    santé des modèles générés ne s'y applique pas. La sonde du journal reste
    propre à `moka-9`, puisque ses coups viennent du journal de v1.
*   **Noyaux.** `kernels.convolve` découpe en tuiles de P positions × L
    groupes de 4 sorties (`TILES` dans `src/infer/kernels.zig`) chaque
    convolution dont le nombre de sorties est multiple de 4. Une forme
    absente de `TILES` prend une position par tuile ; une sortie qui n'est
    pas multiple de 4, comme la convolution de valeur à 2 sorties, garde la
    boucle d'une position à la fois. Chaque vecteur de poids chargé sert les P
    positions, et les P × L accumulateurs restent dans les registres. Chaque
    sortie s'accumule toujours dans le même ordre (biais, puis taps, puis
    canaux), donc les résultats ne changent pas d'un bit.
*   **Matchs.** `match/match.ts` prend `--first-model` et `--second-model`
    (`moka-9` par défaut) et charge les poids et les WASM de chaque modèle
    depuis `zig-out/`. Les ouvertures sont toujours tirées de la politique de
    `moka-9`, pour garder les mêmes ouvertures d'un match à l'autre.

## Résultats

### Noyaux

Une inférence en Node, 100 positions du banc, médiane de 15 tours :

| Tuiles des couches à 24 sorties | Tuiles des couches à 48 sorties | `moka-9-v2` |
|---|---|---|
| 1 × 24 (avant) | 1 × 48 (avant) | 805 µs |
| 2 × 24 | 1 × 48 | 905 µs |
| 2 × 24 | 2 × 48 | 925 µs |
| 3 × 12 | 1 × 48 | 676 µs |
| 3 × 12 | 3 × 12 | 651 µs |

Apparier deux positions sur 24 sorties ralentit : il faut 12 accumulateurs,
et V8 en déverse une partie hors des registres. Au-delà d'environ
9 accumulateurs par tuile, toutes les formes essayées sont plus lentes. Sur
v1, des tuiles de 2 × 16 sur les couches à 32 sorties donnent 247 → 232 µs.

Banc Chromium (`pnpm bench --model moka-9-v2 --verdict`), médiane des p50
d'un aller-retour `infer` via le Worker, avant et après les tuiles :

| | Moka | Zig avant | Zig après | Rapport après |
|---|---|---|---|---|
| `moka-9` | 5,765 ms | 0,265 ms | 0,255 ms | 22,61 |
| `moka-9-v2` | 15,005 ms | 0,815 ms | 0,660 ms (−19 %) | 22,73 |

Avant les tuiles, v2 coûtait à Zig 3,1 fois v1, contre 2,6 fois à Moka, et
le rapport tombait de 21,75 à 18,80. Après, il est le même sur les deux
modèles.

### Matchs

Match en Node, 200 parties (100 ouvertures de 4 coups jouées dans les deux
couleurs), 100 ms par coup, 4 parties en parallèle. `zig-search` est la
recherche de `search.wasm` ; `moka-puct` est `match/search.ts` sur le runtime
de Moka. Les visites sont des médianes sur les coups 0 à 59 :

| Match | Résultat du premier | Score [IC 95 %] | Elo [IC 95 %] | Visites médianes |
|---|---|---|---|---|
| `zig-search` v2 contre `zig-search` v1 | 154 V, 30 D, 16 N | 81,0 % [75,6 ; 86,4] | +252 [+196 ; +322] | 138 contre 380 |
| `zig-search` v1 contre `moka-puct` v2 | 130 V, 57 D, 13 N | 68,3 % [61,8 ; 74,7] | +133 [+84 ; +188] | 300 contre 4 |
| `zig-search` v2 contre `moka-puct` v2 | 175 V, 6 D, 19 N | 92,3 % [88,5 ; 96,0] | +430 [+355 ; +550] | 90 contre 4 |
| `moka-puct` v2 contre `moka-puct` v1 | 121 V, 59 D, 20 N | 65,5 % [58,9 ; 72,1] | +111 [+63 ; +165] | 5 contre 15 |
| `zig-search` v1 contre `moka-puct` v1 | 188 V, 6 D, 6 N | 95,5 % [92,6 ; 98,4] | +531 [+440 ; +713] | 390 contre 15 |

*   À temps égal, v2 vaut mieux que v1 dans les deux moteurs, alors qu'il a
    environ 3 fois moins de visites : +252 Elo dans Zig, +111 dans Moka.
*   La vitesse pèse plus que le réseau : `zig-search` avec v1 bat
    `moka-puct` avec v2 (+133 Elo). À 100 ms, Moka ne fait que 4 visites
    avec v2.
*   La meilleure combinaison est v2 dans `search.wasm`.

Limites :

*   `moka-puct` dépasse son temps d'au plus une inférence (jusqu'à 164 ms
    par coup avec v2), ce qui l'avantage ;
*   à 4 parties en parallèle, les visites de `zig-search` v1 varient de 300 à
    390 selon l'adversaire, à cause du partage du processeur ;
*   ces Elo décrivent cette recherche PUCT simple à 100 ms, sur cette machine.
    Ils ne se comparent pas aux 85 victoires sur 100 de la PR, obtenues avec
    la recherche Python de Moka (64 visites, symétries complètes).

## Tests

*   `zig build test` construit aussi les tests natifs pour `moka-9-v2` :
    poids déquantifiés identiques bit à bit à ceux de Moka, plus petite taille
    mémoire avec 25 % de marge, lignes sentinelles à zéro après la passe
    avant. Les tests de noyaux couvrent les formes de v2 (3×3 12 → 48 et
    24 → 24, 1×1 48 → 24, 24 → 48 et 48 → 4), bit à bit contre une référence
    scalaire.
*   `tests/infer-parity.test.mjs` parcourt les modèles de `zig-out/models/` :
    `moka-9-v2` passe la règle de parité sur ses 5 239 positions de
    calibration et ses 1 318 positions de validation. La tolérance reste
    celle de `oracle/tolerance.json`, calibrée sur `moka-9` seulement, et
    v2 n'a ni snapshot de sorties ni constantes de layout épinglées dans
    `src/test_root.zig`.
*   `tests/search-parity.test.ts` couvre `moka-9-v2` : règles, visites de
    chaque enfant de la racine, lots, réutilisation de l'arbre, partie finie
    refusée et arbre plein.
*   `tests/wasm-checks.test.mjs` contrôle les quatre WASM de `moka-9-v2`.

## Future Improvements

*   **Licence des poids.** v1 et v2 sont distillés de réseaux KataGo, et Moka
    publie les deux sous MIT. Ce dépôt ne fait que reprendre les fichiers de
    Moka ; la licence des réseaux KataGo eux-mêmes n'est pas vérifiée ici.
*   **Publier v2** dans `dist/` si la PR est fusionnée : le `worker.js` publié
    refuse aujourd'hui tout autre `.bin` que celui de v1.
