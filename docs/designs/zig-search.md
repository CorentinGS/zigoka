# Recherche PUCT dans le moteur Zig

*   **Author:** corenting
*   **Status:** Implemented
*   **Date:** 2026-10-31

--------------------------------------------------------------------------------

## Overview

Ce design suit [`zig-wasm-moka.md`](zig-wasm-moka.md), dont les Future
Improvements prévoyaient « les règles dans le moteur, un PUCT avec un arbre de
taille fixe, un protocole de lots ». Il livre `search.wasm` : le réseau du
moteur d'inférence, les règles de Moka et une recherche PUCT dans un seul
module, sans allocation. `infer.wasm` ne change pas : même ABI, mêmes
13 pages, mêmes tests.

## The Problem

`match/search.ts` fait la recherche en TypeScript sur les règles de Moka et
appelle `infer.wasm` une fois par position. Mesuré en Node sur le
AMD Ryzen 7 8845HS :

| | 9×9 | 19×19 |
|---|---|---|
| Inférence Zig, une position | 244 µs | 1 130 µs |
| `getLegalMoves` de Moka, une position | 45 µs | 275 µs |
| `encodeStudentFeatures` de Moka | 5,5 µs | 16 µs |
| Visite complète de `match/search.ts` | 313 µs | — |

À 9×9, les règles et l'arbre en JavaScript prennent 22 % d'une visite. Moka
teste chaque point en jouant le coup sur une copie du plateau, ce qui coûte à
peu près le carré de la surface : à 19×19, les seuls coups légaux ajoutent
24 % au temps d'une inférence.

## Goals

*   Les règles de Moka en Zig, pour N = 9 et N = 19, avec les mêmes réponses
    que `vendor/moka/src/game.ts` : coups légaux, coup joué ou refusé, ko,
    entrées du réseau bit à bit, score, pierres mortes.
*   La recherche de `match/search.ts` en Zig, avec les mêmes visites, enfant
    par enfant, pour un même nombre de visites.
*   Un arbre de taille fixe, alloué statiquement ; quand il est plein, la
    recherche s'arrête et rend le meilleur coup trouvé.
*   L'évaluation de plusieurs feuilles par appel du réseau (lots), mesurée.

## Non-Goals

*   Réutiliser l'arbre d'un coup à l'autre, ou changer l'algorithme : la
    parité avec `match/search.ts` passe d'abord.
*   Paralléliser la recherche sur plusieurs threads.
*   Publier `search.wasm` dans `dist/` : il sert la démo et les matchs.

## Technical Plan

```
JS (js/search-engine.ts)                      search.wasm
  writePosition(state)  ── position_ptr ──▶   Position (plateau, ko, compteurs,
                                                         deux derniers coups)
  search_start()        ──────────────────▶   racine : règles, entrées, réseau
  boucle jusqu'à l'échéance :
    search_run(cible, lot) ───────────────▶   sélection PUCT, règles, entrées,
                                              network.forward, expansion, remontée
  search_best_move()    ◀──────────────────   coup le plus visité
```

*   **Règles** (`src/rules/rules.zig`, générique en N). Un `Position` tient le
    plateau, le ko, le nombre de coups, les passes consécutives, la couleur à
    jouer et les deux derniers coups : c'est tout ce que lisent les règles et
    les entrées du réseau. Deux calculs diffèrent de Moka sans changer les
    réponses :
    *   la légalité d'un point vide se lit sur les groupes voisins : un voisin
        vide, un groupe ami à plus d'une liberté ou un groupe adverse à une
        liberté rendent le coup légal, sinon c'est un suicide ;
    *   dans l'après-match des pierres mortes, la meilleure capture immédiate
        se calcule avec les mêmes comptes de libertés au lieu de jouer chaque
        point.

    Les pierres mortes sortent comme un ensemble trié ; Moka les rend dans
    l'ordre de découverte, et ses appelants ne s'en servent que comme
    ensemble.
*   **Recherche** (`src/search/tree.zig`). Mêmes constantes que
    `match/search.ts` : exploration 1,5, première visite à la valeur du nœud
    moins 0,2, priors par softmax en f64 sur les coups candidats, pas de passe
    avant `MIN_PASS_MOVE_COUNT` sauf s'il est le seul coup, valeur terminale au
    signe du score après retrait des pierres mortes. Nœuds et arêtes vivent
    dans des tableaux statiques dont la taille vient de `build.zig` :

    | Modèle | Nœuds | Arêtes | Mémoire de `search.wasm` |
    |---|---|---|---|
    | `moka-9` | 8 192 | 524 288 | 332 pages (21,8 Mo) |
    | `synthetic-19` | 4 096 | 819 200 | 580 pages (38 Mo) |

    Une feuille réserve de quoi s'étendre (un nœud et `N² + 1` arêtes) avant
    d'être créée ; sans place, `search_run` rend `pool_full`.
*   **Temps** : WASM freestanding n'a pas d'horloge. `js/search-engine.ts`
    appelle `search_run` par tranches de visites : chaque tranche vise la
    moitié du temps restant, d'après la vitesse de la tranche précédente.
*   **Lots** : `search_run(cible, lot)` collecte jusqu'à `lot` feuilles avant
    d'appeler le réseau. Une visite en cours compte comme une défaite pour
    écarter les descentes suivantes du même chemin. Ces visites virtuelles
    vivent dans des compteurs séparés, pour qu'avec un lot de 1 la recherche
    refasse exactement les opérations flottantes de `match/search.ts`.

## Résultats

Visites par coup en Node, 12 positions de parties tirées de la politique :

| | `match/search.ts` + `infer.wasm` | `search.wasm`, lot 1 | lot 2 | lot 4 | lot 8 |
|---|---|---|---|---|---|
| 9×9, 100 ms | 331 | 477 (+44 %) | 488 | 417 | 450 |
| 19×19, 300 ms | 205 | 263 (+28 %) | 263 | 263 | 264 |

Les lots n'apportent rien : `infer(count)` enchaîne les positions une à une,
et les noyaux réutilisent déjà chaque poids chargé sur deux positions du
plateau. Prendre des positions de plateaux différents revient à traiter plus
de positions d'un même plateau, ce qui était déjà plus lent avec des blocs de
3 ou 4 positions. Les écarts entre lots à 9×9 viennent de la forme de l'arbre
(collisions, nœuds terminaux), pas du réseau. Le lot reste à 1 par défaut.

Match en Node, 200 parties (100 ouvertures jouées dans les deux couleurs),
4 parties en parallèle, `zig-search` (`search.wasm`) contre `zig-puct`
(`match/search.ts` avec `infer.wasm`), même réseau :

| Temps par coup | Résultat de `zig-search` | Score [IC 95 %] | Elo [IC 95 %] | Visites médianes, coups 0 à 59 |
|---|---|---|---|---|
| 100 ms | 99 V, 82 D, 19 N | 54,3 % [47,3 ; 61,2] | +30 [−18 ; +79] | 357 contre 298 |
| 10 ms | 108 V, 77 D, 15 N | 57,8 % [50,9 ; 64,6] | +54 [+6 ; +104] | 35 contre 29 |

Avec 4 parties en parallèle, le gain de visites tombe à +20 % (contre +44 %
sur un seul thread). À 100 ms, l'écart de force n'est pas significatif : vers
300 visites, cette recherche gagne peu à en avoir plus. À 10 ms, où chaque
visite compte davantage, il l'est à peine. Le gain attendu est surtout en
19×19, où une visite coûte 4 à 5 fois plus cher.

Les moyennes de visites par coup ne servent pas ici : en fin de partie,
l'arbre ne contient presque plus que des nœuds terminaux, une visite n'y coûte
que quelques microsecondes, et un seul coup peut dépasser 500 000 visites. Pour
la même raison, la tranche de visites suivante se calcule avec le coût d'une
visite borné par celui de l'évaluation de la racine : sinon, une tranche de
visites terminales fait viser trop de visites, et le coup déborde de son temps
(jusqu'à 795 ms pour 100 ms avant cette borne).

## Tests

*   `tests/search-parity.test.ts`, pour `moka-9` et `synthetic-19` :
    *   règles : à chaque position de parties aléatoires et de parties tirées
        de la politique, coups légaux, entrées du réseau octet par octet, fin
        de partie, score, pierres mortes (une position sur 5 en 9×9, sur 60
        en 19×19, et chaque position finale), puis le coup joué et un point au
        hasard, légal ou non ;
    *   recherche : sur des positions de parties tirées de la politique, les
        visites de chaque enfant de la racine et le coup choisi sont ceux de
        `match/search.ts` avec le réseau Zig ;
    *   lots de 2, 4 et 8 : le nombre de visites est respecté et le coup est
        légal ;
    *   une partie finie est refusée, et un arbre plein arrête la recherche.
*   `tests/wasm-checks.test.mjs` contrôle aussi `search.wasm` : SIMD, mémoire
    fixe, pas de `memory.grow`, exports exacts, aucun import.
*   `src/rules/rules.zig` a ses tests natifs (capture, ko, suicide, légalité).

## Future Improvements

*   **Paralléliser l'évaluation.** C'est le seul levier qui rendrait les lots
    utiles : plusieurs moteurs évaluent chacun une feuille du lot en même
    temps. En WASM, cela demande des threads sur une mémoire partagée, donc
    `SharedArrayBuffer` et une page isolée (COOP/COEP), ou un Worker par
    moteur avec une copie des entrées et des sorties à chaque lot.
*   **Réutiliser l'arbre** du coup précédent : le sous-arbre du coup joué
    garde ses visites.
*   **Un vrai réseau 19×19.** Les règles et la recherche sont prêtes ; seul
    le modèle manque.
