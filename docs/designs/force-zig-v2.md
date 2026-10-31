# Mesurer la force de Zig v2

*   **Author:** corenting
*   **Status:** Implemented
*   **Date:** 2026-10-31

--------------------------------------------------------------------------------

## Overview

Le meilleur joueur du dépôt, `zig-reuse` avec le modèle `moka-9-v2`, gagne
plus de 90 % de ses parties contre Moka à temps égal. À ce niveau, un match
ne dit plus grand-chose : l'intervalle de l'Elo fait près de 200 points. Ce
design ajoute deux mesures qui situent ce joueur : un handicap de temps
contre Moka d'origine, et des matchs contre KataGo, soit avec son réseau
normal, soit avec son réseau human SL qui imite un rang humain.

## The Problem

*   Contre `moka-puct` v1 à 100 ms par coup, `zig-reuse` v2 fait 92,5 %
    [88,8 ; 96,2], soit +436 Elo [+361 ; +559]. Contre `moka-policy` v1,
    97,3 %. L'écart est trop grand pour être mesuré.
*   On ne savait pas situer le moteur hors du dépôt : ni contre une IA de
    référence, ni en rang humain.

## Goals

*   Donner à chaque joueur son propre temps par coup.
*   Faire jouer KataGo par GTP, sous les règles de Moka, à nombre de visites
    fixe ou en imitant un rang humain.
*   Arbitrer les parties contre KataGo avec le score de KataGo.

## Non-Goals

*   Fournir KataGo, ses réseaux ou un pilote OpenCL.
*   Jouer KataGo dans le navigateur ou dans la vidéo.
*   Donner un rang humain au 19×19.

## Technical Plan

*   **Temps par joueur.** `--first-time` et `--second-time` remplacent
    `--time` pour un joueur. `getTimeMs` dans `match/games.ts` choisit le
    temps de chaque côté, et le nom du joueur porte son temps quand il
    diffère (`zig-reuse@moka-9-v2/10ms`). Dans le nom du fichier de
    résultats, le `/` devient `-`. Le match du navigateur passe aussi par
    `getTimeMs`, mais sa page n'a qu'un temps.
*   **Joueurs asynchrones.** Un joueur peut rendre une promesse de coup, et
    `playGame` attend chaque coup. Les joueurs du navigateur restent
    synchrones.
*   **Client GTP.** `match/katago.ts` lance `katago gtp` et lui passe les
    règles de Moka par `-override-config` : ko simple, score par zone, pas
    de taxe, suicide interdit, pas de bouton, pas de bonus de handicap
    (`whiteHandicapBonus=0`), komi 7 et plateau 9×9 par GTP. Les journaux de
    KataGo vont dans le dossier temporaire (`logDir`). L'abandon est coupé
    dans les deux configs ; un `resign` arrête donc le match sur une erreur.
    Avant chaque `genmove`, il rejoue sur le plateau de KataGo les coups qui
    manquent, ou repart d'un plateau vide si l'historique diverge. Les
    coordonnées GTP sautent la lettre I et comptent les lignes depuis le bas.
*   **Configs.** `match/katago-gtp.cfg` donne exactement `maxVisits` visites
    par coup : pas d'abandon, pas de réflexion pendant le tour adverse, pas
    de recherche réduite après une passe ou quand KataGo gagne.
    `match/katago-human.cfg` reprend `gtp_human5k_example.cfg` de KataGo sans
    délai ni abandon : le coup vient de la politique human SL brute, et les
    visites du réseau normal décident seulement quand passer. Dans les deux
    cas, `maxVisits` vient de `--katago-visits`, passé par `-override-config`,
    qui l'emporte sur la valeur des configs.
*   **Arbitrage.** Quand KataGo joue, le banc demande `final_score` en fin de
    partie et en fait le résultat. Le score de Moka reste dans `mokaScore`,
    et le résumé compte les parties où les deux arbitres désignent un vainqueur
    différent (`summary.disagreements`, `{ count, games }`). Sans cela, Zig peut gagner sur une erreur du score de Moka (voir
    plus bas).
*   **Lancement.** `--first katago` ou `--second katago`, avec
    `--katago-binary`, `--katago-model`, `--katago-visits` (1 par défaut,
    16 avec `--katago-profile`), et pour l'imitation humaine
    `--katago-human-model` et `--katago-profile` (`rank_20k` à `rank_9d`).
    Les chemins prennent par défaut `KATAGO_BINARY`, `KATAGO_MODEL` et
    `KATAGO_HUMAN_MODEL`. Le profil est passé tel quel à KataGo, sans
    contrôle ; un réseau human SL sans profil n'est pas utilisé. Chaque
    worker lance son propre processus KataGo. Le résumé nomme le joueur
    `katago-<visites>v` ou `katago-<profil>`.

## Résultats

Matériel : AMD Ryzen 7 8845HS, KataGo 1.18.1 OpenCL sur la Radeon 780M
intégrée (Mesa rusticl, `RUSTICL_ENABLE=radeonsi`), réseau normal
`b10c512h8nbt3tflrs-fson-silu-rsnh`, réseau human SL `b18c384nbt-humanv0`.
Les ouvertures de 4 coups sont jouées deux fois en échangeant les couleurs.

### Handicap de temps contre Moka v1

`zig-reuse` v2 contre `moka-puct` v1 à 100 ms par coup, 200 parties par
ligne :

| Temps de Zig | Score de Zig [IC 95 %] | Elo [IC 95 %] | Visites par coup (Zig contre Moka) |
|---|---|---|---|
| 100 ms | 92,5 % [88,8 ; 96,2] | +436 [+361 ; +559] | 173 contre 15 |
| 10 ms | 81,8 % [76,4 ; 87,1] | +260 [+204 ; +332] | 17 contre 15 |
| 5 ms | 72,0 % [65,8 ; 78,2] | +164 [+114 ; +222] | 8 contre 14 |
| 2 ms | 54,8 % [47,9 ; 61,6] | +33 [−15 ; +82] | 2,6 contre 14 |

Les lignes à 10, 5 et 2 ms n'ont pas de fichier dans `bench/results/` : le
nom du fichier portait encore le `/` du temps, et l'écriture échouait après
l'affichage du résumé.

Zig v2 avec environ 45 fois moins de temps vaut Moka v1. Ce facteur
additionne la vitesse du moteur et le meilleur réseau : à 10 ms, avec à peu
près autant de visites que Moka, Zig v2 gagne encore 82 % de ses parties.

### Contre KataGo

`zig-reuse` v2 contre KataGo, 100 parties par ligne, 2 parties en parallèle :

| Adversaire | Temps de Zig | Score de Zig | Résultat de Zig |
|---|---|---|---|
| KataGo b10, 1 visite | 100 ms | 8,0 % [2,7 ; 13,3] | 0 V, 84 D, 16 N |
| KataGo b10, 1 visite | 400 ms | 3,0 % [0,0 ; 6,3] | 0 V, 94 D, 6 N |

Le réseau de KataGo, sans aucune recherche, bat Zig v2. Quatre fois plus de
temps, soit 2 358 visites par coup au lieu de 187, n'aide pas : c'est le
réseau v2 qui limite, pas la recherche. Ces deux matchs ont tourné avant
l'arbitrage par KataGo : leurs fichiers ont `score` pour Moka et
`katagoScore` pour KataGo, et les chiffres ci-dessus viennent de
`katagoScore`. Les deux arbitres désignent le même vainqueur dans les
200 parties.

### Contre un rang humain imité

`zig-reuse` v2 à 100 ms contre KataGo human SL, 100 parties par ligne,
3 parties en parallèle. Ces deux matchs aussi ont tourné avant l'arbitrage
par KataGo ; les deux scores ont été enregistrés :

| Profil | Score, arbitre Moka [IC 95 %] | Score, arbitre KataGo | Désaccords |
|---|---|---|---|
| `rank_1k` | 68,0 % [58,9 ; 77,1] | 65,0 % | 3 / 100 |
| `rank_3d` | 56,0 % [46,3 ; 65,7] | 51,0 % | 5 / 100 |

Dans les 8 désaccords, l'arbitre Moka donne toujours la victoire à Zig, avec
parfois de gros écarts (Moka +21, KataGo −3). Plusieurs de ces parties
s'arrêtent après 30 à 47 coups. La recherche de Zig évalue les fins de partie
avec le retrait automatique des pierres mortes de Moka : quand KataGo passe,
Zig passe aussi si ce retrait le donne gagnant, même dans une position qu'il
juge mal. Le score de KataGo est le bon arbitre de ces matchs.

Zig v2 à 100 ms joue à égalité avec le profil `rank_3d` en 9×9, et bat
`rank_1k`. Ce rang surestime sans doute sa force :

*   De `rank_1k` à `rank_3d`, le score ne baisse que d'environ 100 Elo pour
    4 rangs. KataGo prévient que les profils dan, sans recherche,
    n'atteignent pas la force des joueurs qu'ils imitent.
*   Les profils viennent de parties humaines en ligne, surtout en 19×19. Un
    rang en 9×9 ne se transpose pas en 19×19.

L'estimation la plus prudente place Zig v2 autour d'un faible dan en 9×9,
nettement sous le réseau seul de KataGo.

## Tests

*   `tests/katago.test.ts` : coordonnées GTP (lettre I absente, lignes depuis
    le bas, passe), aller-retour de chaque point, refus des coordonnées hors
    du plateau 9×9, et lecture des scores GTP (`B+3.5`, `W+0.5`, `0`).
*   Les matchs contre KataGo ne font pas partie de `pnpm test` : ils
    demandent KataGo et ses réseaux.

## Future Improvements

*   **Score de fin de partie.** La recherche de Zig pourrait refuser de
    passer tant que le retrait automatique des pierres mortes est incertain,
    pour ne plus gagner ni perdre sur une erreur de score.
*   **Réseau.** Comme plus de recherche n'aide pas contre KataGo, le prochain
    gain viendra d'un meilleur réseau, pas d'un moteur plus rapide.
*   **Échelle de visites.** Comparer Zig v2 à KataGo à visites fixes demande
    un adversaire plus faible que b10 à 1 visite, par exemple un plus petit
    réseau KataGo.
