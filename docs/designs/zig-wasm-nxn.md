# Moteur Zig/WASM en N×N : mesure en 19×19 contre le runtime de Moka

*   **Author:** corenting
*   **Status:** Implemented
*   **Date:** 2026-10-30

--------------------------------------------------------------------------------

## Overview

Ce design prolonge [`docs/designs/zig-wasm-moka.md`](zig-wasm-moka.md),
appelé ici « le premier design ». Celui-ci a livré un moteur d'inférence
Zig/WASM, sans allocation, qui remplace le Worker 9×9 de Moka. Dans Chromium,
sur la machine de mesure, un aller-retour `infer` via le Worker prend
0,365 ms, contre 5,900 ms pour Moka, soit un rapport d'environ 16 (« le 16× »
dans la suite).

Le projet veut savoir si une alternative WASM au JavaScript de Moka garde un
avantage de vitesse quand on passe du 9×9 au 19×19 ; le critère chiffré est
NX6. Il veut ensuite pouvoir intégrer le moteur dans d'autres pages, comme
hôte JS/WASM avec un `infer.wasm` par modèle (voir Non-Goals). Ce design est
l'étape qui fait tourner le moteur en 19×19 et le compare au runtime
JavaScript de Moka (`vendor/moka/src/runtime.ts`). Il mesure deux choses :

*   **la vitesse** de l'inférence en 19×19, dans les deux moteurs ;
*   **la parité numérique** : sur les mêmes entrées, Zig calcule les mêmes
    logits, bit à bit, que la référence `f32Accumulate` (définie plus bas), et
    la même valeur à 2 ulp près.

Il ne mesure pas la force de jeu. Aucun modèle 19×19 entraîné n'existe : le
modèle 19×19 de ce design, `synthetic-19`, est **synthétique**. Il reprend
tel quel le tronc entraîné du modèle 9×9 de Moka (la couche d'entrée, les
12 blocs résiduels et le pooling, dont aucun poids ne dépend de la taille du
plateau). Seuls les tenseurs de tête dont la forme dépend de N² sont
générés. Ses sorties n'ont aucun sens pour jouer ; elles servent à
chronométrer et à vérifier la parité.

La recherche PUCT et les matchs entre moteurs ne font pas partie de ce
design. Ils feront l'objet du design suivant, écrit pour N×N. Ce design lui
fournit le temps par inférence en 19×19, et un moteur dont la taille de
plateau n'est plus fixée à 9.

### Termes

*   **« Moka »** désigne le projet ; **« le runtime de Moka »** désigne
    `vendor/moka/src/runtime.ts` ; **« les règles de Moka »** désignent
    `vendor/moka/src/game.ts` et `constants.ts`.
*   Un **modèle** est désigné par un identifiant : `moka-9` (le modèle 9×9
    publié par Moka) et `synthetic-19` (le modèle synthétique de ce design).
*   L'**écart** entre deux sorties est leur différence absolue maximale. Le
    mot « dérive » n'est pas employé.
*   Le **tronc** est la couche d'entrée et les 12 blocs, blocs globaux compris
    (pooling global et ses deux couches denses). Les **têtes** sont la
    politique et la valeur : leurs convolutions, leurs couches denses et
    `tanh`.
*   Tous les chemins sont relatifs à la racine du dépôt. Les tailles de
    fichiers et de tampons sont en octets décimaux (Ko, Mo) ; la mémoire WASM
    est en pages de 64 Kio.

### Repris du premier design

Ce document s'appuie sur les éléments suivants du premier design, rappelés
ici pour qu'il se lise seul.

*   **Le moteur et son ABI.** `infer.wasm` (build `ReleaseFast`) et
    `infer-safe.wasm` (même moteur, build `ReleaseSafe`, avec les contrôles de
    bornes) exportent :
    *   `memory` ; `weights_ptr()` et `weights_capacity()`, où le JavaScript
        copie le `.bin` ; `expected_sha256_ptr()`, le sha256 attendu, figé au
        build ; `load_weights()`, qui déquantifie les poids ;
    *   `input_ptr()`, `policy_ptr()`, `value_ptr()` et `max_batch()` (8) ;
    *   `infer(count)`, qui calcule `count` positions (1 à 8) et renvoie un
        statut : 0 réussi, 1 poids non chargés, 2 `count` invalide, 3 sortie
        non finie.

    Le JavaScript écrit la position k à `input_ptr + k × input_stride`
    flottants : les `feature_count` valeurs de l'encodage de Moka
    (N² intersections × 12 plans). `input_stride` vaut (N² + 1) × 12 : la
    dernière rangée, qui reste à zéro, est une sentinelle interne au moteur.
    Les logits de la position k sont à `policy_ptr + k × policy_moves` (f32),
    sa valeur à `value_ptr + k` (f64).
*   **`Geometry(N)`** (`src/geometry.zig`) : le type Zig qui donne, pour un
    plateau N×N connu à la compilation, la surface, la sentinelle et les
    voisins de chaque intersection. Les noyaux de convolution le prennent en
    paramètre.
*   **`f32Accumulate`** : un mode de l'oracle
    (`oracle/moka-runtime-instrumented.ts`, copie instrumentée du runtime de
    Moka). Dans les convolutions seulement, il arrondit en f32
    (`Math.fround`) après chaque produit et chaque somme, comme le fait Zig.
    Le pooling et les couches denses restent en f64 dans les deux moteurs :
    `linear` accumule en f64 et stocke en f32, la valeur passe par `tanh` en
    f64. C'est pourquoi les logits de Zig peuvent être identiques bit à bit à
    ceux de ce mode, alors que
    le **mode normal** (le runtime de Moka tel quel, qui accumule tout en f64)
    diffère de lui de quelques ulp.
*   **G1** : les mesures de vitesse se font sur la machine de l'auteur (Linux,
    AMD Ryzen 7 8845HS), dans Chromium ; chaque résultat note le processeur
    et la version de Chromium.
*   **G3** : sur `moka-9`, l'écart entre Zig et le mode normal reste sous une
    tolérance égale à 4 fois l'écart maximal mesuré entre le mode normal et
    `f32Accumulate` sur le jeu de calibration, avec des plafonds de 2e-3 pour
    les logits et de 1e-3 pour la valeur (au-delà, le projet s'arrête, sans
    relever le seuil). Le meilleur coup légal est le même, sauf quasi-égalité :
    `|logit(coup de Moka) − logit(coup de Zig)| ≤ 2 × tolérance des logits`,
    les deux logits pris dans les sorties de Moka. Aucune sortie n'est `NaN`
    ni infinie. La tolérance est dans `oracle/tolerance.json`.
*   **G4** : le protocole de vitesse. Médiane, sur 5 tours alternés, du p50
    d'un aller-retour `infer` via le Worker, après 10 inférences de chauffe et
    sur 100 inférences mesurées, page servie avec COOP/COEP. Le premier design
    exigeait un rapport d'au moins 2 entre Moka et Zig, avec cette règle
    d'arrêt : en dessous, le projet s'arrête.
*   **G5** : zéro allocation. Mémoire de taille fixe, taille initiale égale à
    la taille maximale, aucune instruction `memory.grow`, taille constante sur
    1 000 inférences.
*   **G6** : intégrité des poids. Le Worker calcule le sha256 des octets reçus
    et refuse une longueur ou un sha256 différents de ceux figés au build ;
    les poids déquantifiés sont ceux de Moka bit à bit.
*   **D1 à D4** : les outils de diagnostic d'un écart bit à bit, spécifiés par
    le premier design (fichiers D1 à D4) mais **jamais créés**, car la parité
    9×9 a passé du premier coup. Noyaux scalaires de référence (`src/infer/reference.zig`), build `wasm-debug` qui
    expose les sorties de chaque couche, `dumpLayers` dans l'oracle, et
    `tests/infer-layers.test.mjs`, qui trouve la première couche qui diffère.
*   **Le Worker de Moka du banc** : `vendor/moka/src/worker.ts`, bundlé tel
    quel dans `bench/dist/moka-worker.js` et piloté par
    `vendor/moka/src/client.ts`. Ni l'un ni l'autre ne dépend de la taille du
    plateau : ils chargent le manifeste et le `.bin` qu'on leur donne, et
    `runtime.ts` lit `architecture.boardSize`. Ils tournent donc en 19×19
    sans modification.

## The Problem

**Le 16× ne permet pas de conclure pour le 19×19.** Il est mesuré sur un
plateau de 81 intersections, avec un modèle de 113 648 octets. En 19×19, il y
a 361 intersections :

*   le calcul par inférence est multiplié par environ 4,5 (361 / 81) ;
*   les tampons du moteur passent d'environ 606 Ko à environ 3,4 Mo
    (estimation), et sa mémoire de 13 pages à 66 ;
*   la tête de politique, dont la forme dépend de N², représente environ 84 %
    des paramètres du modèle 19×19.

Deux effets pourraient réduire l'écart de vitesse entre les moteurs :

*   les couches denses de Zig (des blocs globaux et des têtes) sont
    scalaires, en f64. Leur part en 9×9 n'a
    pas été mesurée ; en 19×19, une estimation faite pendant la préparation
    de ce design (non mesurée) la place entre 20 et 28 % du temps de Zig ;
*   les défauts de cache, côté Zig, et le travail du ramasse-miettes, côté
    Moka, augmentent quand les poids et les activations ne tiennent plus dans
    quelques centaines de Ko.

Sans mesure, l'avantage de WASM en 19×19 reste une supposition.

**Le moteur Zig gère déjà N ; les outils qui l'entourent sont figés en
9×9.** Le moteur lit la taille du plateau dans le manifeste au build
(`src/infer/buffers.zig:9`), calcule la forme des têtes à partir de N²
(`tools/gen_model_layout.zig:148-196`), et ses noyaux prennent `Geometry(N)`
en paramètre. Le runtime de Moka lit lui aussi `architecture.boardSize`. En
revanche :

*   le build ne connaît qu'un modèle et fixe la mémoire à 13 pages
    (`build.zig:4`, `:11-12`) ;
*   le JavaScript, l'oracle, le banc et les tests écrivent en dur les valeurs
    9×9 (972 entrées, 82 coups, 113 648 octets) ;
*   l'oracle produit ses positions avec les règles de Moka, figées en 9×9
    dans `vendor/moka/src/constants.ts`, que ce dépôt ne modifie pas ;
*   le format des données de référence stocke les coups sur un octet, alors
    qu'il y a 362 coups en 19×19 ;
*   le banc compare les sorties avec la tolérance de G3, calibrée sur les
    logits 9×9 entraînés.

**Ce design revient sur trois choix du premier design.** Le premier design
classe le 19×19 en Non-Goal, refuse de garder en 19×19 des têtes dont la
forme dépend de N² (Alternative H), et prévoit des têtes indépendantes de N
pour un vrai modèle 19×19 (Future Improvements). Ce design garde pourtant ces
têtes, pour une seule raison : le runtime de Moka n'en connaît pas d'autres,
et sans elles aucune comparaison n'est possible. La tête de politique devient
`[362, 1444]`. Le temps d'inférence reste représentatif d'un réseau de cette
taille pour le tronc, qui fait l'essentiel des multiplications-additions
(environ 25,7 millions, contre environ 0,55 million dans les têtes,
estimation). La part des têtes dans le temps, mesurée en J4, dit de combien
les têtes synthétiques pèsent sur le temps complet. Le temps d'initialisation, la mémoire et la
taille du `.bin` 19×19 ne le sont pas : un vrai modèle 19×19 n'aurait pas ces
têtes.

**La comparaison a une limite connue.** Le runtime de Moka alloue plus de
50 tableaux par inférence et accumule en f64. Ce design compare Zig à ce
runtime, et pas à tout JavaScript. Une réécriture optimisée en TypeScript du
runtime de Moka (le « témoin TypeScript » des Future Improvements du premier
design) dirait si l'avantage tient à WASM ou à ce runtime ; elle reste hors
de ce design.

## Goals

Les identifiants commencent par **NX**, pour les distinguer des G1 à G8 du
premier design. Les jalons J0 à J5 sont définis dans le Technical Plan.

*   **NX1 — Chaque étape prend un identifiant de modèle (J1).** Le build, les
    tests, l'oracle et le banc reçoivent un identifiant de modèle. La taille
    du plateau vient du manifeste du modèle et n'est jamais une option
    séparée. Aucune valeur 9×9 n'est écrite en dur, sauf dans les tests
    réservés à `moka-9` : les deux tests Zig de « Parité et non-régression »,
    G3 et le snapshot, les 13 pages et le seuil gzip, et les tests e2e.
*   **NX2 — Modèle synthétique déterministe et sain.**
    *   **(J2)** `synthetic-19` reprend le tronc de `moka-9` octet pour octet ;
        ses têtes sont générées avec une graine fixe. Deux générations donnent
        des fichiers identiques octet pour octet. Le modèle est accepté par
        `GoModelRuntime.create` de Moka, avec des tenseurs `float32` et des
        blocs d'échelles alignés sur 4 octets.
    *   **(J3)** Le modèle est accepté par `zig build gen`. Sur le jeu 19×19,
        toutes les sorties sont finies, aucune activation des têtes n'est
        subnormale au sens du f32, et les sorties ne sont pas dégénérées
        (règles exactes dans « Contrôle de santé »). Les subnormaux du tronc
        sont comptés et enregistrés.
*   **NX3 — Oracle 19×19 déterministe (J3).** Les positions 19×19 sont
    produites par les règles de Moka non modifiées, avec des constantes 19×19
    substituées au moment du bundle. Deux générations donnent des fichiers
    identiques octet pour octet. `vendor/` n'est pas modifié.
*   **NX4 — Parité en 19×19 (J3).** Sur le jeu 19×19 de `synthetic-19`, pour
    `infer.wasm` et `infer-safe.wasm` exécutés sous Node :
    *   les logits sont identiques bit à bit à ceux de `f32Accumulate` ;
    *   la valeur diffère de celle de `f32Accumulate` d'au plus 2 ulp de f64 ;
    *   aucune sortie n'est `NaN` ni infinie.

    Il n'y a ni tolérance ni contrôle du meilleur coup en 19×19 : avec des
    têtes générées, ils compareraient Moka f32 à Moka f64 et ne pourraient
    pas échouer à cause de Zig.
*   **NX5 — Zéro allocation en 19×19 (J3).** L'artefact `synthetic-19`
    respecte G5 avec sa propre taille de mémoire (66 pages).
*   **NX6 — Vitesse en 19×19 (J5).** Avec le protocole de G4, sur la machine
    de G1, dans une seule session qui mesure `synthetic-19` et `moka-9` :
    *   **verdict :** le rapport (p50 du Worker de Moka) / (p50 du Worker de
        Zig) sur `synthetic-19`. Sous 2, la règle d'arrêt de G4 s'applique :
        le passage au 19×19 s'arrête, et le design PUCT n'est pas écrit pour
        N×N. Ce rapport porte sur l'inférence complète. On suppose, sans le
        mesurer, qu'il minore le rapport sur le tronc seul, car les têtes sont
        la partie la plus scalaire de Zig ;
    *   **livrable pour PUCT :** deux temps de Zig en ms, l'inférence complète
        via le Worker et le tronc seul. **Le chiffre représentatif est le
        temps du tronc** ; le temps complet en est une borne haute ;
    *   **contrôle `moka-9` :** le rapport sur `moka-9` mesuré dans la même
        session ;
    *   le fichier de résultats note l'identifiant, le sha256 et la taille du
        plateau de chaque modèle, et la machine.
*   **NX7 — Le 9×9 ne change pas (J1, puis à chaque jalon).**
    *   Sur les deux jeux de `moka-9`, les sorties d'`infer.wasm` et
        d'`infer-safe.wasm` sont identiques bit à bit à celles relevées avant
        toute modification (J0), sur des entrées elles aussi identiques.
    *   G3 passe avec l'`oracle/tolerance.json` actuel, sans recalibration.
    *   L'artefact `moka-9` garde 13 pages, sa taille après gzip reste sous
        32 Kio, et les tests e2e passent avec des messages d'erreur identiques
        au caractère près.
*   **NX8 — Un contrat N×N pour le design PUCT (J5).** Ce document décrit ce
    que l'étape suivante peut supposer : N lu dans le manifeste au build, dont
    dérivent toutes les tailles du moteur, une ABI qui exporte ses dimensions, `infer(count)`
    jusqu'à 8 positions, le format v2 des données de référence, et les temps
    de NX6.

## Non-Goals

*   **La force de jeu.** Aucun accord sur le meilleur coup en 19×19 n'est
    mesuré : les têtes de `synthetic-19` sont générées.
*   **La recherche PUCT, les matchs, le protocole de lots côté JS et les
    symétries.** Ils feront l'objet du design suivant.
*   **Les règles et l'encodage en Zig, et des règles de référence N×N.** Les
    règles de Moka avec des constantes substituées servent ici à produire des
    entrées plausibles, rien de plus : elles ne sont pas testées en 19×19 et
    ne sont pas une référence pour le design PUCT. Les positions de test des
    règles (bords, captures, ko, décompte) appartiennent à ce design-là.
*   **Un vrai modèle 19×19**, des têtes indépendantes de N, ou un réseau plus
    large que 32 canaux. Le résultat de ce design ne prédit pas la vitesse
    d'un réseau plus large : au-delà de 32 canaux, le noyau ne garde plus ses
    accumulateurs en registres.
*   **Un choix de taille à l'exécution**, ou un binaire qui contient plusieurs
    modèles. Chaque modèle donne son propre `infer.wasm`.
*   **Un Worker 19×19 publié.** Le seul utilisateur de l'artefact 19×19 est le
    banc. `dist/` reste le Worker 9×9 publié. Les tests e2e, les contrôles CSP
    et les tests des messages d'erreur restent en 9×9.
*   **Présenter le temps d'initialisation, la mémoire ou la taille du `.bin`
    19×19 comme représentatifs.**
*   **Le témoin TypeScript.** Il reste dans les Future Improvements du premier
    design.
*   **Mesurer le tronc côté Moka.** Il faudrait livrer au banc un Worker de
    Moka modifié, et la comparaison ne porterait plus sur le runtime de Moka.
*   **La vectorisation des couches denses.** Elle se décidera sur un vrai
    modèle 19×19, pas sur la part mesurée avec des têtes synthétiques.
*   **Un paquet Zig réutilisable, sans variables globales.** La réutilisation
    visée ici est un hôte JS/WASM, avec un `infer.wasm` par modèle. Les
    variables globales du moteur restent. Un paquet Zig se décidera avec
    PUCT.
*   **Les activations INT8 ou INT16, `relaxed-simd`, les threads WASM, les
    navigateurs autres que Chromium, et le mobile.**

## Mesures enregistrées

Section remplie pendant l'implémentation, dans le commit du jalon indiqué.

*   **J3 :**
    *   contrôle de santé : les valeurs `health` du jeu de validation de
        chaque modèle ;
    *   durée de `pnpm check:all` et de chaque génération de l'oracle.

    Relevé du 2026-10-30, sur un AMD Ryzen 7 8845HS, Node 24.22.1 :

    | Jeu de validation | `moka-9` (1 481 positions) | `synthetic-19` (241 positions) |
    | --- | --- | --- |
    | `activationCount` | 166 205 225 | 120 371 065 |
    | `nonFiniteCount` | 0 | 0 |
    | `trunkSubnormalCount` | 0 | 0 |
    | `headSubnormalCount` | 0 | 0 |
    | `deadValueHiddenPositionCount` | 0 | 0 |
    | `saturatedValueFraction` | 0,097 | 0 |
    | `medianLegalLogitRange` | 9,31 | 49,0 |

    *   `synthetic-19` passe le contrôle avec `POLICY_SCALE_FACTOR` et
        `VALUE_SCALE_FACTOR` à 1. L'étendue médiane des logits légaux est
        environ 5 fois celle de `moka-9` : c'est ce que prévoit la réserve
        de « Hypothèse de l'échelle », puisque les entrées des têtes sont
        positives. Le critère bloquant ne porte que sur une étendue nulle.
    *   La parité de NX4 passe au premier essai : les outils D1 à D4 (F46)
        ne sont pas créés, et le plugin de F31 suffit (F36 n'est pas créé).
    *   Durées : génération de l'oracle `moka-9` 2 min 7 s, `synthetic-19`
        19 s (contre 15 s estimées) ; `pnpm check:all` depuis un état sans
        `zig-out/` ni données de référence, 4 min 35 s (3 min 43 s en J1).
*   **J5 :** NX6. Pour `synthetic-19` et `moka-9` : les p50 de Moka et de
    Zig avec et sans Worker, le p50 du tronc, le rapport, la part des têtes,
    la machine, la version de Chromium et le sha256 de chaque modèle.

    Relevé du 2026-10-30 (`pnpm bench --model synthetic-19 --verdict`, fichier
    `bench/results/2026-10-30T18-38-44-464Z.json`, non versionné), sur un
    AMD Ryzen 7 8845HS, Chromium 156.0.8078.4, Node 24.22.1. Chaque valeur est
    la médiane des p50 de 5 tours, en ms, à `count = 1` :

    | | `synthetic-19` | `moka-9` |
    | --- | --- | --- |
    | Taille du plateau | 19 | 9 |
    | sha256 | `55f33968b08da4b255dbf04a6f468efe1691cce4bc258498e8f8c73dd44ebb44` | `35c1e51cc9518a21e2f81cded7f69f41d22c80ca05522ed5e4056cb26d9ed053` |
    | Moka, Worker | 28,830 | 6,400 |
    | Moka, sans Worker | 28,590 | 6,240 |
    | Zig, Worker | 1,990 | 0,380 |
    | Zig, sans Worker | 1,980 | 0,335 |
    | Zig, tronc seul | 1,535 | 0,310 |
    | Rapport (Moka / Zig, Worker) | 14,49 | 16,84 |
    | Part des têtes | 22,5 % | 7,5 % |
    | Dérive de Moka contre `baseline-<id>.json` | sans référence | −16,5 % |

    *   **Verdict : NX6 passe**, avec un rapport de 14,49 pour un seuil de 2.
        La règle d'arrêt de G4 ne s'applique pas : le design PUCT peut être
        écrit pour N×N.
    *   **Livrable pour PUCT :** en 19×19, 1,535 ms pour le tronc seul (le
        chiffre représentatif) et 1,990 ms pour l'inférence complète via le
        Worker (la borne haute).
    *   Le contrôle `moka-9` donne 16,84, proche des 16,16 de la phase 1 du
        premier design. La dérive de Moka reste sous le seuil d'alerte de
        20 %.
    *   La part des têtes n'est pas stable d'une session à l'autre : le run
        de J4, sur le même code, donnait 28,9 % pour `synthetic-19` et
        15,8 % pour `moka-9` (rapports 15,36 et 15,46). Elle est la
        différence de deux p50 proches, et l'écart de moins de 0,1 ms en
        9×9 est du même ordre que le bruit entre tours. Seul son ordre de
        grandeur sert : les têtes pèsent plus en 19×19, ce qui va dans le
        sens de l'hypothèse de NX6 (le rapport sur l'inférence complète
        minore celui du tronc), sans la mesurer.
    *   Le rapport compare ce moteur au runtime TypeScript de Moka, et non à
        tout JavaScript ; les têtes de `synthetic-19` sont générées, ce qui
        ne change pas le coût du calcul mais ne dit rien de la force de jeu.
*   **Comparaison 9×9 / 19×19** (après J5). Facteur entre `moka-9` et
    `synthetic-19`, dans la session de J5 et, entre parenthèses, dans celle
    de J4, sur le même code :

    | Médiane des p50 | 9×9 | 19×19 | Facteur 9→19 |
    | --- | --- | --- | --- |
    | Moka, Worker | 6,40 ms (6,88) | 28,83 ms (30,34) | 4,50× (4,41×) |
    | Zig, Worker | 0,380 ms (0,445) | 1,990 ms (1,975) | 5,24× (4,44×) |
    | Zig, sans Worker | 0,335 ms (0,380) | 1,980 ms (1,975) | 5,91× (5,20×) |
    | Zig, tronc seul | 0,310 ms (0,320) | 1,535 ms (1,405) | 4,95× (4,39×) |
    | Zig, têtes (sans Worker − tronc) | 0,025 ms (0,06) | 0,445 ms (0,57) | 10 à 18× |
    | Rapport Moka / Zig | 16,84 (15,46) | 14,49 (15,36) | |

    Nombre de multiplications-additions, compté sur les manifestes : la
    surface est multipliée par 4,46 (361/81) ; le tronc passe de 5,76 M à
    25,6 M (4,46×) ; les têtes passent de 47 k à 615 k (13×), dont
    `policy.linear` de 26 568 (82×324) à 522 728 (362×1444).

    *   Le tronc suit la surface dans les deux moteurs. Rien ne signale un
        effet de cache en 19×19.
    *   Les têtes font 2,3 % des opérations en 19×19, mais 22 à 29 % du
        temps de Zig. `dense.linear` faisait une seule chaîne d'additions
        f64 par sortie, pour garder l'ordre des sommes de Moka : les
        522 728 additions de `policy.linear` dépendaient chacune de la
        précédente.
*   **Têtes en lignes parallèles** (après J5). `dense.linear` calcule
    maintenant 4 sorties à la fois dans un vecteur f64 ; chaque sortie
    garde l'ordre d'additions de Moka, donc les sorties restent identiques
    bit à bit (snapshot de J0 et parité de NX4 inchangés). 8 lignes ne
    font pas mieux que 4 dans Node, 2 lignes font moins bien.

    Deux sessions consécutives, l'une avec le changement, l'autre sans,
    même machine et même Chromium (fichiers
    `bench/results/2026-10-30T19-51-18-358Z.json` et
    `2026-10-30T19-53-06-944Z.json`, non versionnés) :

    | Médiane des p50 | 19×19 avant | 19×19 après | 9×9 avant | 9×9 après |
    | --- | --- | --- | --- | --- |
    | Moka, Worker | 26,720 ms | 26,790 ms | 5,905 ms | 5,800 ms |
    | Zig, Worker | 1,740 ms | 1,550 ms | 0,345 ms | 0,345 ms |
    | Zig, sans Worker | 1,655 ms | 1,470 ms | 0,310 ms | 0,300 ms |
    | Zig, tronc seul | 1,285 ms | 1,280 ms | 0,290 ms | 0,290 ms |
    | Rapport Moka / Zig | 15,36 | 17,28 | 17,12 | 16,81 |

    *   En 19×19, les têtes passent d'environ 0,37 ms à 0,19 ms, et
        l'inférence complète via le Worker gagne 11 %. Le 9×9 ne change pas
        au-delà du bruit.
    *   Dans ces deux sessions, Moka est 23 à 24 % plus rapide que
        `baseline-moka-9.json`, et le tronc de Zig 17 % plus rapide qu'en
        J5 : la machine allait plus vite. Seule la comparaison entre ces deux
        sessions vaut, pas avec les chiffres de J5.
    *   Il reste environ 0,19 ms de têtes pour 0,6 M d'opérations. Chaque
        colonne lit encore 4 poids isolés, un par ligne : c'est l'objet de
        l'entrée suivante.
*   **Poids des têtes en `[in][out]`** (après J5). Au chargement,
    `weights.transposeLinear` déquantifie une seconde fois chaque tenseur
    linéaire depuis le `.bin`, directement dans l'ordre `[in][out]`. Il
    n'utilise pas de tampon : le tampon de réordonnancement est dimensionné
    pour les convolutions, et y loger `policy.linear` aurait fait dépasser
    13 pages à `moka-9` (NX7). `dense.linear` parcourt les entrées dans
    l'ordre et met à jour toutes les sommes par vecteurs de 4 f64, avec des
    lectures contiguës de 4 f32. Chaque somme garde l'ordre d'additions de
    Moka : sorties identiques bit à bit. 8 voies ne font pas mieux que 4
    dans Node, 2 voies font moins bien. Sans vecteurs explicites, la même
    boucle n'est pas vectorisée et ne gagne rien.

    Deux sessions consécutives, l'une avec le changement, l'autre avec la
    version à 4 lignes (fichiers `bench/results/2026-10-30T20-03-19-144Z.json`
    et `2026-10-30T20-05-04-842Z.json`, non versionnés) :

    | Médiane des p50 | 19×19 avant | 19×19 après | 9×9 avant | 9×9 après |
    | --- | --- | --- | --- | --- |
    | Moka, Worker | 26,325 ms | 26,245 ms | 5,675 ms | 5,665 ms |
    | Zig, Worker | 1,495 ms | 1,415 ms | 0,330 ms | 0,335 ms |
    | Zig, sans Worker | 1,460 ms | 1,355 ms | 0,300 ms | 0,295 ms |
    | Zig, tronc seul | 1,265 ms | 1,260 ms | 0,285 ms | 0,285 ms |
    | Rapport Moka / Zig | 17,61 | 18,55 | 17,20 | 16,91 |

    *   En 19×19, les têtes passent d'environ 0,195 ms à 0,095 ms, et
        l'inférence complète via le Worker gagne 5 %. Depuis la version à
        une chaîne par sortie, les têtes sont passées d'environ 0,37 ms à
        0,095 ms. Le 9×9 ne change pas au-delà du bruit.
    *   Le temps d'initialisation de Zig ne change pas de façon visible
        (8 à 15 ms dans les deux sessions).
    *   Moka est 26 % plus rapide que `baseline-moka-9.json` dans ces deux
        sessions ; comme pour l'entrée précédente, seule la comparaison
        entre ces deux sessions vaut.
*   **Convolutions par paires de positions** (après J5). Dans
    `kernels.convolve`, les convolutions à 16 sorties au plus (les deux
    3×3 16→16 et la 1×1 32→16 de chaque bloc, 82 % des opérations du
    tronc pour les 3×3) calculent 2 positions à la fois : chaque vecteur de
    poids lu sert aux deux. Chaque sortie garde l'ordre d'additions
    précédent (tap, puis canal d'entrée), donc les sorties restent
    identiques bit à bit ; la dernière position d'un plateau impair passe
    par l'ancien chemin. Dans Node, 3 et 4 positions font moins bien que 2
    (probablement des registres débordés par V8), et étendre les paires
    aux convolutions à 32 sorties ne gagne rien.

    Deux sessions consécutives, l'une avec le changement, l'autre sans
    (fichiers `bench/results/2026-10-30T20-13-14-289Z.json` et
    `2026-10-30T20-15-00-467Z.json`, non versionnés) :

    | Médiane des p50 | 19×19 avant | 19×19 après | 9×9 avant | 9×9 après |
    | --- | --- | --- | --- | --- |
    | Moka, Worker | 26,230 ms | 26,215 ms | 5,670 ms | 5,655 ms |
    | Zig, Worker | 1,420 ms | 1,130 ms | 0,325 ms | 0,265 ms |
    | Zig, sans Worker | 1,360 ms | 1,080 ms | 0,295 ms | 0,230 ms |
    | Zig, tronc seul | 1,260 ms | 0,985 ms | 0,285 ms | 0,225 ms |
    | Rapport Moka / Zig | 18,47 | 23,20 | 17,45 | 21,34 |

    *   Le tronc gagne 22 % en 19×19 et 21 % en 9×9 ; l'inférence via le
        Worker gagne 20 % et 18 %.
    *   L'`infer.wasm` de `moka-9` passe de 23,5 Ko (7,5 Ko après gzip) à
        24,1 Ko (8,0 Ko), sous le plafond de 32 Kio. Les deux changements
        des têtes l'avaient déjà fait passer de 19,2 Ko (6,7 Ko) à 23,5 Ko.

## Open Questions

*   **Le plugin esbuild qui substitue les constantes fonctionne-t-il ?** Il
    n'existe pas encore. Il est construit en premier en J3 ; le repli et sa
    preuve d'équivalence sont décrits dans « Oracle N×N ».
*   **Durée de `pnpm check:all`.** L'oracle 19×19 compte environ
    240 positions ; à environ 60 ms par position pour les deux passes de Moka
    (mode normal et `f32Accumulate`), une génération prend environ 15 s, et
    `oracle:check` la refait. `check:all` passerait d'environ 4 minutes à
    environ 5 (estimation, mesurée en J3).

## Critical User Journeys (CUJs)

### CUJ 1 : Mesurer le moteur en 19×19

*   **Given** : un clone du dépôt sans artefacts construits, avec les outils
    de `mise.toml` installés.
*   **When** : le développeur lance `pnpm check:all`, puis
    `pnpm bench --model synthetic-19 --verdict`.
*   **Then** : `check:all` construit et teste `moka-9` et `synthetic-19`, et
    vérifie la parité des deux. Le banc écrit un fichier de résultats qui
    nomme chaque modèle, son sha256, la taille du plateau et la machine, et
    qui donne le rapport des p50 sur `synthetic-19` et sur `moka-9`, les temps
    de Zig (complet et tronc) et la part des têtes. La commande échoue si le
    rapport sur `synthetic-19` est inférieur à 2.

### CUJ 2 : Un intégrateur 9×9 met à jour ce dépôt

*   **Given** : une page qui utilise `dist/worker.js` et `dist/infer.wasm`
    avec le client et le modèle de Moka.
*   **When** : l'intégrateur remplace ces deux fichiers par ceux de cette
    version.
*   **Then** : rien ne change pour la page. Les sorties sont identiques bit à
    bit, les messages d'erreur le sont au caractère près, et `dist/` ne
    contient que le 9×9.

### CUJ 3 : Le design PUCT part de ce contrat

*   **Given** : ce design livré, avec la mesure de NX6.
*   **When** : l'auteur du design PUCT dimensionne sa recherche.
*   **Then** : il dispose du temps de Zig par inférence en 19×19 (complet et
    tronc) et d'une ABI qui donne ses dimensions. Il écrit lui-même ses
    règles N×N et leurs positions de test.

--------------------------------------------------------------------------------

## Technical Plan (Architecture)

### Vue d'ensemble

```text
                     build.zig : registre des modèles
          moka-9 (vendor/moka/model)       synthetic-19 (models/synthetic-19)
                       │                               ▲
                       │      oracle/generate-synthetic-model.ts (J2)
                       │      (tronc de moka-9 + têtes générées)
                       ▼                               │
   zig build models ─► zig-out/models/<id>/go-model.{json,bin}
   zig build wasm   ─► zig-out/<id>/infer.wasm, infer-safe.wasm
   zig build wasm-profile ─► zig-out/<id>/infer-profile.wasm (banc seulement)
                       │
       ┌───────────────┼────────────────────────────┐
       ▼               ▼                            ▼
  oracle (Node)    tests Node                  banc Chromium
  generate-9.js :  parité NX4/NX7,             bench/dist/<id>/ :
  règles de Moka   wasm-checks,                modèle, positions, sorties
  telles quelles ; valeurs des exports         attendues ; Workers de Moka
  generate-19.js :                             et de Zig partagés
  constantes 19×19
       │
       ▼
  oracle/fixtures/<id>/{calibration,validation,tensors}  (format v2)
```

Le moteur Zig change peu, et seulement en trois points : quatre exports de
dimensions, `forward(k)` découpé en tronc et têtes, et un artefact de
profilage pour le banc. Le reste du travail porte sur les outils qui
l'entourent : chaque étape reçoit un identifiant de modèle, et la taille du
plateau se lit dans le manifeste de ce modèle.

### Registre des modèles

`build.zig` contient la seule liste des modèles :

| Identifiant | Dossier du manifeste et du `.bin` | Pages mémoire | Enregistré en |
| --- | --- | --- | --- |
| `moka-9` | `vendor/moka/model` | 13 | J1 |
| `synthetic-19` | `models/synthetic-19` | 66 | J3 |

*   Les 66 pages viennent de la règle du premier design (25 % de marge
    au-dessus des tampons et de la pile) : environ 3,4 Mo de tampons et
    64 Kio de pile (estimation). L'assertion de build existante refuse un
    nombre trop petit, et le test « the configured page count is the smallest
    one with 25% headroom » un nombre trop grand ; les deux tournent pour
    chaque modèle.
*   `zig build wasm` et `zig build wasm-safe` construisent tous les modèles
    enregistrés. L'option `-Dmodel=<id>` restreint le build à un seul ; un
    identifiant inconnu échoue avec la liste des identifiants.
*   `zig build models` copie le manifeste et le `.bin` de chaque modèle dans
    `zig-out/models/<id>/`. L'oracle, les tests Node et `build-js` lisent les
    modèles là. Une seule exception : le générateur du modèle synthétique lit
    `moka-9` dans `vendor/moka/model/`, car il tourne avant tout build.
*   Les artefacts vont dans `zig-out/<id>/` au lieu de `zig-out/bin/`.
    `dist/infer.wasm` est la copie de `zig-out/moka-9/infer.wasm`.

### Modèle synthétique

`oracle/generate-synthetic-model.ts` lit `moka-9` et écrit
`models/synthetic-19/go-model.json` et `go-model.bin`. Dans ce qui suit,
N = 19, A = N² = 361, et `rng()` est la suite de `oracle/mulberry32.ts`
initialisée avec la graine 19, qui renvoie un nombre dans [0, 1[.
`pick(n)` vaut `floor(rng() × n)`.

*   **Tenseurs repris tels quels :** tous ceux dont la forme ne dépend pas de
    N (couche d'entrée, 12 blocs, pooling, convolutions des têtes,
    `value.hidden.bias`, `value.output.weight` et `value.output.bias`). Leurs
    octets (valeurs, échelles, biais) sont copiés sans changement.
*   **Tenseurs générés**, dans cet ordre de tirage :
    1.  `policy.linear.weight` `[A + 1, 4A]` (INT8, une échelle par ligne) et
        `policy.linear.bias` `[A + 1]` (f32). Pour chaque ligne j de 0 à A :
        *   la ligne source s de `moka-9` est la ligne de passe (81) si
            j = A, sinon `pick(81)` ;
        *   les 4A valeurs INT8 de la ligne j sont tirées, dans l'ordre, par
            `source[s][pick(324)]` : uniformément, avec remise, parmi les
            324 valeurs de la ligne source ;
        *   l'échelle de la ligne j est
            `fround(échelle[s] × sqrt(324 / 4A) × POLICY_SCALE_FACTOR)` ;
            son biais est `biais[s]`, sans changement.
    2.  `value.hidden.weight` `[32, 2A]` (INT8, une échelle par ligne). Pour
        chaque ligne i de 0 à 31, la ligne source est la ligne i de `moka-9` ;
        les 2A valeurs sont `source[i][pick(162)]`, et l'échelle est
        `fround(échelle[i] × sqrt(162 / 2A) × VALUE_SCALE_FACTOR)`.
        `POLICY_SCALE_FACTOR` et `VALUE_SCALE_FACTOR` sont deux constantes du
        générateur, qui valent 1.
*   **Hypothèse de l'échelle.** Le facteur `sqrt(entrées 9×9 / entrées
    19×19)` garde l'amplitude des sorties si les termes de la somme sont
    centrés et indépendants. Les entrées des têtes sont positives (après
    ReLU), donc une composante moyenne croît plutôt comme le nombre
    d'entrées. Le contrôle de santé vérifie ce qui compte pour la mesure ; la
    saturation et l'étendue des logits sont enregistrées pour juger
    l'hypothèse.
*   **Fichier `.bin` :** les blocs (données, puis échelles) sont écrits dans
    l'ordre croissant de leur décalage dans le `.bin` de `moka-9`. Chaque bloc
    commence à un multiple de 4, avec des octets nuls de bourrage si besoin.
    Taille : 629 968 octets (aucun bourrage n'est nécessaire avec ces
    formes).
*   **Manifeste :** celui de `moka-9`, avec, pour chaque tenseur, `shape`,
    `dataOffset` et `scaleOffset` recalculés, et `architecture.boardSize`
    (19), `architecture.policyMoveCount` (362), `weightsBytes` et `sha256`
    mis à jour. Il est écrit par `JSON.stringify(manifeste, null, 2)` suivi
    d'un saut de ligne, avec les clés dans l'ordre du manifeste de `moka-9`.
*   **Versionné**, avec un `PROVENANCE.md` qui cite Moka et sa licence MIT, la
    graine, la commande de régénération, et la valeur des deux constantes
    d'échelle. `pnpm models:check` régénère dans un dossier temporaire et
    compare le manifeste et le `.bin` octet pour octet ; `PROVENANCE.md`,
    écrit à la main, n'est pas comparé.

### ABI : quatre exports de dimensions

`infer.wasm` exporte en plus quatre fonctions sans argument qui renvoient un
`u32` :

| Export | Valeur | Source dans le moteur |
| --- | --- | --- |
| `board_size()` | N | `model_layout.CONFIG.board_size` |
| `feature_count()` | N² × 12 | `buffers.Board.AREA × CONFIG.input_planes` |
| `policy_moves()` | N² + 1 | `model_layout.CONFIG.policy_moves` |
| `input_stride()` | (N² + 1) × 12 | `buffers.INPUT_STRIDE` |

Ces exports s'ajoutent à l'ABI du premier design sans en changer aucun.

*   `js/engine.ts` lit `feature_count()` et `policy_moves()` au lieu de ses
    constantes. Les messages d'erreur sont construits à partir de ces
    valeurs : pour `moka-9`, leur texte reste identique au caractère près.
*   Un seul `worker.js` sert tous les modèles : il charge déjà le `infer.wasm`
    de son propre dossier (`new URL("./infer.wasm", import.meta.url)`).

### Oracle N×N

**Règles 19×19 : substitution au moment du bundle.** Les règles de Moka
(`vendor/moka/src/game.ts`) prennent toutes leurs tailles dans
`./constants`, et `constants.ts` ne contient que des littéraux.

*   `oracle/constants-19.ts` réexporte tout `vendor/moka/src/constants.ts` et
    redéfinit cinq valeurs :
    *   `GO_BOARD_SIZE = 19`, `GO_BOARD_AREA = 361`, `GO_PASS_MOVE = 361` ;
    *   `GO_MAXIMUM_MOVE_COUNT = 540` (120 × 4,5) et
        `GO_MINIMUM_PASS_MOVE_COUNT = 89` (20 × 361 / 81, arrondi). Les
        parties 19×19 s'arrêtent à 60 coups : `GO_MAXIMUM_MOVE_COUNT` n'est
        jamais atteint, et `GO_MINIMUM_PASS_MOVE_COUNT` interdit toute passe.
        Les parties 19×19 ne contiennent donc pas de passe, et les plans de
        passe ne sont jamais exercés en 19×19.

    Les autres constantes ne dépendent pas de la taille : indices de plans,
    couleurs, komi (7) et sa normalisation (15), qui sont des entrées du
    réseau entraîné.
*   `scripts/build-js.mjs` produit deux bundles de l'oracle :
    `oracle/dist/generate-9.js`, tel qu'aujourd'hui, et
    `oracle/dist/generate-19.js`. Dans ce dernier, un plugin esbuild
    (`onResolve`) remplace par `oracle/constants-19.ts` tout import dont le
    chemin, résolu et complété par `.ts` s'il n'a pas d'extension, est
    exactement `vendor/moka/src/constants.ts`, sauf quand il vient de
    `oracle/constants-19.ts`. `game.ts` importe `"./constants"`, sans
    extension (ligne 27). Les imports de
    `vendor/moka/src/runtime-constants.ts` ne sont pas touchés. Un alias
    simple ne suffit pas : il ne traite pas les chemins relatifs.
*   **Contrôle de comportement.** Au démarrage, chaque bundle vérifie que
    `createGameState().board.length` vaut `boardSize²` du modèle demandé, et
    échoue sinon avec les deux valeurs. Ce contrôle prouve que la
    substitution a atteint `game.ts`, et refuse un modèle de la mauvaise
    taille.
*   **Repli** si le plugin ne fonctionne pas : `oracle/game-nxn.ts`, une
    copie de `game.ts` qui reçoit ses constantes en paramètre. Avant tout
    usage en 19×19, la copie doit prouver qu'elle se comporte comme
    l'original : `generate-9.js` construit sur la copie, avec les constantes
    9×9, doit produire pour `moka-9` un dossier identique octet pour octet à
    celui du bundle d'origine. Ce contrôle est ajouté à `oracle:check` tant
    que le repli est en place.

**Jeux de positions, par taille de plateau :**

| Jeu | 9×9 (`moka-9`) | 19×19 (`synthetic-19`) |
| --- | --- | --- |
| Calibration | 50 parties aléatoires (graines 1 à 50) | aucun |
| Validation | 20 parties jouées par le modèle (graines 101 à 120), 6 premiers coups aléatoires | 4 parties aléatoires (graines 101 à 104), arrêtées après 60 coups, environ 240 positions |
| Sondes | « journal » et « smoke » | « smoke » aux points 4-4 : 60, 72, 288, 300 |

*   Il n'y a pas de calibration en 19×19, puisque aucune tolérance n'y est
    calibrée. Les parties 19×19 sont aléatoires (coups légaux tirés au
    hasard) : jouées par le modèle, elles dépendraient de têtes qui n'ont pas
    de sens.
*   La sonde **« smoke »** joue quatre coups fixés, en alternant les couleurs
    comme le fait `playMove` (noir, blanc, noir, blanc), aux quatre points
    4-4, puis enregistre la position obtenue ; elle vérifie que l'oracle
    produit des pierres dans les quatre quarts du plateau. La sonde **« journal »** rejoue une
    position du journal de Moka et vérifie la réponse que Moka y avait
    donnée ; elle n'a pas de sens pour une tête générée. La même règle vaut
    pour l'auto-contrôle de démarrage de l'oracle, qui lance les deux sondes :
    en 19×19, il ne lance que « smoke ».

**Format v2 des données de référence, pour tous les modèles :**

*   dossier `oracle/fixtures/<id>/`, avec `calibration/` (pour `moka-9`
    seulement), `validation/` et `tensors/` (les poids déquantifiés lus par
    les tests Zig) ;
*   `chosen-move.u16` remplace `chosen-move.u8` ; les autres fichiers ne
    changent pas ;
*   `index.json` reçoit `formatVersion: 2`, `modelId`, `modelSha256`,
    `boardSize`, `featureCount`, `moveCount` et `health` (ci-dessous), en plus
    de `positions` et `games`. Les lecteurs prennent les tailles là, et plus
    dans des constantes.

**Contrôle de santé (NX2).** Il porte sur ce qui fausserait la mesure de
vitesse ou la parité, et rien d'autre.

*   **Où :** un crochet `onTensor(name, values)` sur l'instance
    `f32Accumulate` du runtime instrumenté, qui tourne déjà sur chaque
    position : aucune passe de plus. Le crochet reçoit la sortie de chaque
    couche, après son activation quand elle en a une : couche d'entrée ; pour
    chaque bloc, réduction, première et seconde convolutions, expansion, et
    sortie après addition et ReLU ; pooling global, puis ses deux couches
    denses (`global.hidden` après ReLU, `global.output`) dans les blocs
    globaux ; convolution de politique, logits ; convolution de valeur,
    couche cachée après ReLU, sortie avant `tanh`.
*   **Calcul :** pour chaque jeu de chaque modèle, sur toutes ses positions,
    l'oracle compte les activations, les sorties non finies, et les valeurs
    subnormales : v ≠ 0 et |v| < 2⁻¹²⁶, le plus petit f32 normal. Les
    subnormaux sont comptés à part pour le tronc et pour les têtes. Chaque
    jeu a un compteur neuf ; les sondes de démarrage ne sont pas comptées.
    Les diagnostics sont calculés sur les sorties `f32Accumulate`.
*   **Critères bloquants :**
    *   aucune sortie non finie ;
    *   aucun subnormal dans les têtes (convolutions de tête, logits, couche
        cachée de valeur). Seules les têtes sont générées, et seules elles
        sont réglables ;
    *   des sorties non dégénérées, pour que la parité de NX4 teste vraiment
        le calcul de Zig : sur chaque position, au moins une unité de
        `value.hidden` est non nulle après ReLU ; moins de 10 % des positions
        ont `|valeur| ≥ 0,999` ; l'étendue médiane des logits légaux est non
        nulle.
*   **Diagnostics, enregistrés sans seuil :**
    *   la fraction de subnormaux du tronc. Le tronc porte les poids entraînés
        de Moka : un subnormal y est un comportement du vrai réseau en 19×19,
        et il ralentirait Zig (f32) plus que Moka (f64), donc désavantagerait
        Zig dans le verdict ;
    *   la fraction de positions où `|valeur| ≥ 0,999` (saturation de
        `tanh`) ;
    *   la médiane, par position, de l'étendue `max − min` des logits des
        coups légaux.
*   **Clés de `index.json` :** `health.activationCount`,
    `health.nonFiniteCount`, `health.trunkSubnormalCount`,
    `health.headSubnormalCount`, `health.deadValueHiddenPositionCount`,
    `health.saturatedValueFraction`, `health.medianLegalLogitRange`.
*   **Si un critère bloquant échoue sur `synthetic-19` :** l'oracle échoue
    avec la valeur et la limite. L'auteur ajuste alors à la main
    `POLICY_SCALE_FACTOR` ou `VALUE_SCALE_FACTOR` dans le générateur, note la
    nouvelle valeur et la raison dans `PROVENANCE.md`, et régénère le modèle ;
    ou il arrête le passage au 19×19. Le générateur ne lit jamais les
    résultats de l'oracle.

**Tolérance.** `oracle/tolerance.json` ne change pas et ne vaut que pour
`moka-9`. `pnpm calibrate` lit `oracle/fixtures/moka-9/calibration/`.

### Parité et non-régression

*   **Snapshot, avant toute modification (J0) :**
    `oracle/snapshots/moka-9.json` note, pour chaque jeu (`calibration`,
    `validation`) et pour chaque artefact (`infer.wasm`, `infer-safe.wasm`),
    le sha256 des entrées (`features.f32`), des logits (f32) et des valeurs
    (f64). Le sha256 des artefacts n'y est pas : les artefacts changent
    forcément en J1 (nouveaux exports). Le schéma est donné avec le fichier
    F4.
*   **Règle de parité commune.** Une sortie est conforme à sa référence si ses
    logits sont identiques bit à bit et si sa valeur en diffère d'au plus
    2 ulp de f64. Zig a pour référence `f32Accumulate`, et Moka le mode
    normal. La marge de 2 ulp vient de `tanh`, qui n'est pas garanti
    identique entre deux moteurs JavaScript et Zig. Un seul module, `bench/parity-rule.ts`,
    l'applique sous Node, dans le banc et dans les tests e2e.
*   **`tests/infer-parity.test.mjs`** tourne pour chaque modèle et chaque
    artefact :
    *   la règle de parité, sur chaque jeu, devient un critère (c'est
        aujourd'hui un simple diagnostic) ;
    *   pour `moka-9` seulement : G3 (tolérance et meilleur coup), et les
        sha256 des entrées et des sorties égaux à ceux du snapshot (NX7) ;
    *   pour tous les modèles : mémoire constante sur 1 000 inférences,
        `infer(8)` égal à 8 × `infer(1)`, chemins d'erreur, et valeur des
        exports de dimensions comparée au manifeste et à `index.json`.
*   **Tests Zig natifs :** `zig build test` lance les tests du moteur pour
    chaque modèle. Les tests de géométrie et ceux de l'outil de génération ne
    dépendent pas du modèle et tournent une fois. Deux tests sont réservés à
    `moka-9`, car ils comparent à des valeurs 9×9 écrites en dur : « buffer
    sizes follow the memory table » (`buffers.zig`) et « the generated layout
    matches the Moka model » (`test_root.zig`).
*   **Échec du bit à bit en 19×19 :** les outils D1 à D4 sont créés à ce
    moment (F46), puis lancés avec `-Dmodel=synthetic-19`.

### Banc

*   `pnpm bench --model <id>` (par défaut `moka-9`). `build-js` prépare
    `bench/dist/<id>/` pour chaque modèle :
    *   `go-model.json` et `go-model.bin`, `infer.wasm`, une copie de
        `dist/worker.js` (qui charge l'`infer.wasm` de son dossier), et
        `infer-profile.wasm` à partir de J4 ;
    *   les 100 premières positions du jeu de validation (`positions.f32`), et
        leurs sorties attendues : `logits-normal.f32`, `value-normal.f64`,
        `logits-f32acc.f32`, `value-f32acc.f64` ;
    *   `bench-inputs.json` : `modelId`, `modelSha256`, `boardSize`,
        `featureCount`, `moveCount`, `positionCount`. La page et `run.mjs`
        lisent les dimensions et le sha256 là.

    Un jeu de moins de 100 positions fait échouer `build-js` avec le nombre de
    positions trouvé. Le Worker de Moka reste unique, à
    `bench/dist/moka-worker.js` : la page lui transmet, par `client.ts`, le
    manifeste et les octets du modèle demandé.
*   **Contrôle des sorties.** Le banc ne chronomètre qu'un moteur dont les
    sorties sont conformes. Chaque moteur est comparé à sa propre référence,
    avec la règle de parité commune : Moka au mode normal, Zig à
    `f32Accumulate`. Le banc n'a donc plus besoin de tolérance.
    **Repli**, si le runtime de Moka sous Chromium ne reproduit pas le mode
    normal sous Node : le contrôle de Moka ne vérifie plus que des sorties
    finies et le nombre de logits, et l'écart maximal avec le mode normal est
    enregistré dans le fichier de résultats. Zig garde sa règle bit à bit.
*   **Protocole de G4**, inchangé : 5 tours, avec et sans Worker, 10 inférences
    de chauffe et 100 mesurées, COOP/COEP, `simd128`.
*   **Tronc seul.** Un troisième artefact, `infer-profile.wasm`, construit
    seulement pour le banc, exporte en plus `forward_trunk(count)`, qui fait
    le même contrôle que `infer` puis s'arrête après le tronc. L'étape
    `stage=trunk` du banc l'instancie directement (sans Worker) et charge les
    poids. Elle crée un moteur de banc dont `infer(features)` fait
    `input.set(features)` puis `forward_trunk(1)`, et échoue si le statut
    n'est pas 0. Ce moteur est chronométré par la même boucle que Zig sans
    Worker ; la seule différence de coût JS est l'absence de lecture des
    sorties, ce qui est noté avec la part des têtes. Comme `forward_trunk` ne produit pas de
    sortie, la page appelle ensuite `infer(1)` sur la première position avec
    la même instance, et contrôle sa sortie.
*   **Part des têtes :** (p50 de Zig sans Worker − p50 du tronc) / p50 de Zig
    sans Worker. Les deux p50 sont mesurés sans Worker, à `count = 1`, dans
    les mêmes tours, et chacun est la médiane de ses 5 tours.
*   **Verdict :** `--verdict` mesure `synthetic-19` et `moka-9` dans la même
    session. Dans chaque tour, les deux modèles sont mesurés l'un après
    l'autre, dans un ordre qui alterne d'un tour à l'autre ; pour chaque
    modèle, les runtimes alternent comme aujourd'hui, et l'étape tronc suit
    Zig sans Worker. NX6 s'applique au modèle demandé.
*   La référence de Moka par modèle devient `bench/results/baseline-<id>.json` ;
    `baseline-moka.json` est renommé `baseline-moka-9.json`, sans changement
    de contenu.
*   `dist/` ne contient jamais `infer-profile.wasm` ni rien de 19×19.

### Jalons

Chaque jalon est un commit, et `pnpm check:all` passe à chaque commit. La
phrase vaut pour un clone vierge, sans `zig-out/` ni données de référence.

| Jalon | Contenu | Objectifs atteints |
| --- | --- | --- |
| J0 | Harnais de test commun, snapshot des sorties 9×9 | référence de NX7 |
| J1 | Registre avec `moka-9` seul, exports de dimensions, format v2, tailles lues partout, banc par modèle | NX1, NX7 |
| J2 | Générateur et fichiers de `synthetic-19`, non enregistré | NX2 (déterminisme) |
| J3 | Oracle 19×19, contrôle de santé, enregistrement de `synthetic-19` | NX2 (santé), NX3, NX4, NX5 ; durée de l'oracle mesurée |
| J4 | Artefact de profilage, étape tronc, verdict du banc | CUJ 1 exécutable |
| J5 | Mesure du verdict, documents | NX6, NX8 |

`synthetic-19` n'est enregistré qu'en J3, dans le commit qui crée aussi son
oracle : avant, aucun consommateur n'aurait ses données de référence. Si NX4
échoue en J3, le commit n'est pas fait, et les outils D1 à D4 sont créés
(F46) pour trouver la première couche qui diffère.

### Contrat pour le design PUCT (NX8)

*   N vient du manifeste du modèle, lu au build par `gen_model_layout`, qui
    vérifie aussi `policyMoveCount = N² + 1` (F7). Toutes les tailles du
    moteur en dérivent, par `model_layout.CONFIG` et `Geometry(N)` ; aucune
    n'est écrite à la main.
*   L'ABI donne ses dimensions (`board_size`, `feature_count`,
    `policy_moves`, `input_stride`) ; `infer(count)` accepte 1 à 8 positions.
*   Le format v2 des données de référence accepte toute taille de plateau qui
    a un bundle d'oracle.
*   Le temps de Zig en 19×19, complet et tronc seul, en ms, fixe le nombre de
    visites possible par coup.

Les règles de Moka à constantes substituées ne font pas partie du contrat.

## Alternatives Considered

Les alternatives sont numérotées de 1 à 15, pour les distinguer des
alternatives A à Q du premier design, que ce document cite.

### Poids du modèle 19×19

#### Alternative 1 : poids entièrement aléatoires

*   **Pros** : aucun lien avec les poids de Moka ; une loi normale et une
    graine suffisent.
*   **Cons** : sans normalisation, 12 blocs de ReLU et d'additions résiduelles
    font croître ou s'éteindre les activations selon l'échelle choisie. Les
    sorties deviennent infinies, ou égales aux biais, ce qui rend la parité
    triviale, ou la valeur sature `tanh`. Des activations subnormales
    ralentiraient le f32 de Zig plus que le f64 de Moka et fausseraient la
    mesure de vitesse. Régler l'échelle de chaque tenseur sur celle de
    `moka-9` réduit le risque sans le supprimer, car la structure des poids
    entraînés est perdue.
*   **Reason for Rejection** : le tronc entraîné, repris tel quel, garde des
    activations proches de celles d'une vraie partie, sans aucun réglage.
    Seules les têtes, dont la forme change avec N, doivent être générées.

#### Alternative 2 : têtes indépendantes de N, façon KataGo

*   **Pros** : plus proches d'un vrai modèle 19×19 ; la mémoire et la taille
    du `.bin` deviendraient représentatives.
*   **Cons** : le runtime de Moka ne connaît pas ces têtes ; la comparaison
    devient impossible, et il n'y a plus de référence pour la parité.
*   **Reason for Rejection** : sans ces têtes, il n'y a pas de comparaison
    avec Moka. Elles restent dans les Future Improvements, pour un vrai
    modèle 19×19.

#### Alternative 3 : régénérer le modèle synthétique à chaque build

*   **Pros** : aucun binaire d'environ 630 Ko dans git.
*   **Cons** : `zig build` dépendrait de Node ; le sha256 de `synthetic-19`
    ne serait stable qu'à condition que le générateur le soit, sans rien pour
    le vérifier.
*   **Reason for Rejection** : versionner le modèle et vérifier sa
    régénération avec `pnpm models:check` garde `zig build` autonome.

### Positions 19×19

#### Alternative 4 : positions tirées au hasard plan par plan

*   **Pros** : aucune règle 19×19 ; génération rapide.
*   **Cons** : des plans d'entrée indépendants décrivent des positions
    impossibles (libertés incohérentes avec les pierres, ko sans capture). Le
    tronc entraîné n'a jamais vu de telles entrées, et ses activations ne
    ressembleraient plus à celles d'une partie.
*   **Reason for Rejection** : les entrées doivent rester plausibles pour que
    la mesure le soit, et les règles de Moka en produisent dès que leurs
    constantes changent.

#### Alternative 5 : pierres posées au hasard

*   **Pros** : des positions qui ressemblent à des parties, sans jouer.
*   **Cons** : des groupes sans liberté restent sur le plateau ; les plans de
    libertés et de ko ne correspondent à aucune partie.
*   **Reason for Rejection** : mêmes défauts que l'alternative 4, moins
    visibles.

#### Alternative 6 : copie paramétrée de `game.ts` dans `oracle/`

*   **Pros** : ne dépend d'aucun mécanisme d'esbuild ; la taille devient un
    vrai paramètre.
*   **Cons** : une copie de 600 lignes à maintenir, qui peut s'écarter de
    `vendor/moka/src/game.ts` ; il faut prouver qu'elle se comporte comme
    l'original en 9×9.
*   **Reason for Rejection** : écartée comme solution principale, car la
    substitution garde le code de Moka octet pour octet. Gardée comme repli
    (F36) si le plugin ne fonctionne pas, avec la preuve d'équivalence
    décrite dans « Oracle N×N ».

### Dimensions et registre

#### Alternative 7 : lire les dimensions dans le manifeste, côté Worker

*   **Pros** : aucun nouvel export.
*   **Cons** : le premier design exclut volontairement que le Worker lise le
    manifeste ; ses contrôles portent sur le `.bin`, son sha256 et sa
    longueur. Une seconde source de dimensions pourrait contredire le
    `infer.wasm` chargé.
*   **Reason for Rejection** : le `infer.wasm` connaît ses dimensions ; les
    exporter garde une seule source.

#### Alternative 8 : injecter les dimensions dans `worker.js` au bundle

*   **Pros** : aucun export ajouté ; constantes connues dès le chargement.
*   **Cons** : un `worker.js` par modèle, et rien n'empêche d'associer un
    `worker.js` à l'`infer.wasm` d'un autre modèle.
*   **Reason for Rejection** : avec les exports, un seul `worker.js` sert tous
    les modèles, et ses dimensions viennent toujours de l'`infer.wasm` chargé.

#### Alternative 9 : taille de plateau choisie à l'exécution

*   **Pros** : un seul `infer.wasm` pour toutes les tailles.
*   **Cons** : les tampons statiques devraient être dimensionnés pour le plus
    grand plateau, ce qui reste sans allocation mais coûte la mémoire du
    19×19 à une page 9×9 ; les boucles perdent leurs bornes connues à la
    compilation, dont dépendent la vectorisation et le déroulage ; le sha256
    des poids attendus ne pourrait plus être figé au build.
*   **Reason for Rejection** : la mémoire de G5 et l'intégrité des poids de G6
    supposent un modèle connu au build.

#### Alternative 10 : liste des modèles dupliquée côté JavaScript

*   **Pros** : l'oracle et le banc trouvent les modèles sans passer par
    `zig build`.
*   **Cons** : deux listes à garder d'accord (identifiants, dossiers).
*   **Reason for Rejection** : `zig build models` installe chaque modèle dans
    `zig-out/models/<id>/`, que les outils JS parcourent.

#### Alternative 11 : nombre de pages calculé par `build.zig` ou par l'outil de génération

*   **Pros** : aucune valeur à écrire à la main.
*   **Cons** : `build.zig` devrait refaire le calcul des tampons de
    `buffers.zig` à partir du manifeste, ce qui duplique la logique. L'outil
    de génération ne peut pas fournir la valeur : les options de mémoire du
    lien sont fixées avant qu'il tourne.
*   **Reason for Rejection** : une valeur par modèle dans le registre suffit ;
    l'assertion de build et le test « plus petit nombre de pages » la refusent
    si elle est fausse, avec le nombre d'octets nécessaires.

### Périmètre et mesure

#### Alternative 12 : une démo PUCT en 9×9 d'abord

*   **Pros** : avec de vrais poids, une recherche plus profonde se voit en
    jouant. Le journal de Moka juge la recherche par lots « far too slow »
    dans le navigateur (`vendor/moka/experiment-log.md:1203`).
*   **Cons** : elle ne dit rien du 19×19, qui est le but du projet ; écrite
    avant ce design, elle serait dimensionnée pour le 9×9.
*   **Reason for Rejection** : ce design fixe d'abord le temps par inférence
    en 19×19 et rend le moteur indépendant de N, pour que la recherche soit
    écrite une fois, pour N×N.

#### Alternative 13 : un banc sans parité

*   **Pros** : pas d'oracle 19×19, donc un `check:all` plus court d'environ
    une minute (estimation).
*   **Cons** : rien ne garantirait que le moteur chronométré calcule la bonne
    chose ; un tampon trop petit ou une tête mal orientée passerait inaperçu.
*   **Reason for Rejection** : un temps mesuré sur un moteur faux ne vaut rien,
    et le banc refuse déjà de chronométrer un moteur dont les sorties sont
    fausses.

#### Alternative 14 : séparer tronc et têtes dans un banc natif

*   **Pros** : une horloge plus fine dans Zig ; aucun artefact WASM de plus.
*   **Cons** : le temps natif (AVX2) ne dit rien de la répartition en WASM,
    où les couches f64 scalaires et les noyaux `simd128` n'ont pas les mêmes
    coûts relatifs.
*   **Reason for Rejection** : `infer-profile.wasm` mesure la répartition dans
    Chromium, là où tourne le moteur, au prix d'un artefact réservé au banc.

#### Alternative 15 : un témoin `synthetic-9`

*   **Pros** : un modèle 9×9 aux têtes générées par la même recette
    séparerait l'effet de la taille du plateau de l'effet des têtes
    générées.
*   **Cons** : un modèle, un oracle et une tolérance de plus.
*   **Reason for Rejection** : en 9×9, les têtes générées ont la forme de
    celles de `moka-9`. Le témoin aurait donc la même vitesse que `moka-9`,
    déjà mesuré. Son contrôle du meilleur coup et sa tolérance compareraient
    Moka f32 à Moka f64, sans tester Zig.

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Mémoire.** L'artefact `synthetic-19` demande 66 pages (4 224 Kio),
    contre 13 (832 Kio) pour `moka-9`. Il ne sert qu'au banc et aux tests.
*   **Taille du dépôt.** `models/synthetic-19/` ajoute environ 630 Ko.
*   **Durée des contrôles.** `check:all` passerait d'environ 4 à environ
    5 minutes (Open Questions).
*   **Compatibilité.** Les intégrateurs 9×9 ne voient aucun changement
    (CUJ 2, NX7). L'ABI ne fait que s'étendre.
*   **Retour arrière.** Chaque jalon est un commit qui passe `check:all` ; J1
    se vérifie seul avec `moka-9`, avant tout code 19×19.
*   **Sécurité.** Aucune nouvelle surface n'est livrée : `dist/` ne change
    que par les quatre exports de dimensions, et l'artefact 19×19 comme
    `infer-profile.wasm` n'y vont jamais. Les listes exactes de fichiers de
    `dist/` et d'exports, vérifiées par les tests, le garantissent. Les
    contrôles de G6 (longueur et sha256 des poids) ne changent pas.

### Risques et parades

| Risque | Parade |
| --- | --- |
| Le plugin esbuild ne substitue pas les constantes | Le contrôle de comportement de l'oracle échoue ; on passe au repli (`oracle/game-nxn.ts`), avec sa preuve d'équivalence en 9×9 |
| Zig n'est pas identique bit à bit à `f32Accumulate` en 19×19 | Création des outils D1 à D4 (F46), puis diagnostic avec `-Dmodel=synthetic-19` ; J3 n'est pas commité tant que NX4 échoue |
| `synthetic-19` échoue au contrôle de santé | Ajustement à la main d'une constante d'échelle notée dans `PROVENANCE.md`, ou arrêt du passage au 19×19 |
| Le runtime de Moka sous Chromium ne reproduit pas bit à bit le mode normal sous Node | Le banc refuse de chronométrer. Le contrôle de Moka ne vérifie plus que des sorties finies et le nombre de logits, et l'écart maximal avec le mode normal est enregistré dans le fichier de résultats ; Zig garde sa règle bit à bit (« Banc ») |
| Rapport de NX6 sous 2 | Règle d'arrêt de G4 : le passage au 19×19 s'arrête |

--------------------------------------------------------------------------------

## Additional Resources

*   [`docs/designs/zig-wasm-moka.md`](zig-wasm-moka.md) : le premier design,
    ses mesures enregistrées, son Alternative H et ses Future Improvements.
*   [`concepts_brief_nxn.md`](../../concepts_brief_nxn.md) : le brief de ce
    design, avec l'estimation de la part des couches denses.
*   `vendor/moka/src/runtime.ts` (lecture de `boardSize`, à partir de la
    ligne 287), `vendor/moka/src/constants.ts` et `game.ts` (import unique des
    constantes, lignes 1-27).
*   `vendor/moka/experiment-log.md`, lignes 1169-1203 (recherche par lots,
    trop lente pour le navigateur) et 1670-1786 (recherche du site).

## Future Improvements

*   **Têtes indépendantes de N** et un vrai modèle 19×19 (Alternative 2).
*   **Vectoriser les couches denses**, sur ce vrai modèle, si leur part du
    temps de Zig le justifie : accumulation f64 sur deux voies, poids
    réordonnés en `[entrée][sortie]`, sans changer l'ordre des additions de
    chaque sortie.
*   **Une variante plus large** (96 ou 128 canaux), avec un noyau découpé par
    blocs de sorties.
*   **Le témoin TypeScript** du premier design, pour que la comparaison porte
    sur JavaScript, et pas seulement sur le runtime de Moka.
*   **Le design PUCT N×N**, à partir du contrat de NX8.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

Les fichiers sont numérotés F1, F2, …, pour les distinguer des fichiers 1 à
38 du premier design. Ils sont groupés par jalon ; chaque jalon est un commit
qui passe `pnpm check:all`.

### Target Files to Create/Modify

#### J0 : harnais commun et snapshot du 9×9

###### F1. `./tests/inference-harness.mjs` (Create)

*   **Type / Scope** : module de test Node (sans `.test` dans le nom, pour
    que `node --test "tests/*.test.mjs"` ne le lance pas).
*   **Proposed Changes** :
    *   reprend, sans changer leur comportement, les fonctions de
        `tests/infer-parity.test.mjs:50-105` (`instantiate`, `writeInputs`,
        `readOutputs`, `runFixtureSet` en lots de 8, `getUlpDistance` en f64)
        et `selectHighestLegalMove` (ligne 107) ;
    *   les tailles deviennent des paramètres (`featureCount`, `inputStride`,
        `moveCount`) au lieu des variables du module, et
        `selectHighestLegalMove` reçoit `passMove = moveCount - 1`. En J0, le
        test passe les valeurs 9×9 ; F17 les lit dans les exports en J1 ;
    *   exporte `sha256Hex(bytes)`, utilisé par F3 et F17 ;
    *   en J1, `getUlpDistance` passe dans `bench/parity-rule.ts` (F22), et F1
        l'importe.

###### F2. `./tests/infer-parity.test.mjs` (Modify)

*   **Type / Scope** : test Node.
*   **Proposed Changes** : importe F1 au lieu de ses propres fonctions. Aucun
    test ne change.

###### F3. `./scripts/snapshot-outputs.mjs` (Create)

*   **Type / Scope** : script Node.
*   **Proposed Changes** :
    `node scripts/snapshot-outputs.mjs --wasm <dossier> --fixtures <dossier> --out <fichier>`.
    Pour `infer.wasm` et `infer-safe.wasm` du dossier `--wasm`, et pour les
    jeux `calibration` et `validation`, il lance l'inférence avec F1 et écrit
    les sha256 du schéma de F4. Il ne lit que `features.f32` et le champ
    `positions` d'`index.json`, communs aux formats v1 et v2.
    `featuresSha256` porte sur le fichier `features.f32` entier.
    `logitsSha256` et `valuesSha256` portent sur les sorties de toutes les
    positions, concaténées dans l'ordre d'`index.json` (f32 pour les logits,
    f64 pour les valeurs, petit-boutiste).

###### F4. `./oracle/snapshots/moka-9.json` (Create)

*   **Type / Scope** : données de référence versionnées.
*   **Proposed Changes** : produit pendant J0, après F1 à F3, par
    `pnpm build && pnpm oracle && node scripts/snapshot-outputs.mjs --wasm zig-out/bin --fixtures oracle/fixtures --out oracle/snapshots/moka-9.json`.
    Schéma (les valeurs sont des exemples) :

    ```json
    {
      "modelId": "moka-9",
      "sets": {
        "calibration": {
          "positions": 0,
          "featuresSha256": "…",
          "outputs": {
            "infer.wasm": { "logitsSha256": "…", "valuesSha256": "…" },
            "infer-safe.wasm": { "logitsSha256": "…", "valuesSha256": "…" }
          }
        },
        "validation": { "…": "même forme" }
      }
    }
    ```

#### J1 : registre, exports, format v2, aucune taille 9×9 en dur

###### F5. `./build.zig` (Modify)

*   **Type / Scope** : build.
*   **Proposed Changes** :
    *   un tableau `MODELS` d'entrées `{ id, directory, memory_pages }`, qui
        ne contient en J1 que `moka-9` (`vendor/moka/model`, 13 pages) ;
    *   option `-Dmodel=<id>` : ne traite que ce modèle ; un identifiant
        inconnu échoue avec `unknown model <id>; known models: <liste>` ;
    *   pour chaque modèle, une étape de génération produit son
        `model_layout.zig`, installé dans `gen/<id>/model_layout.zig` ;
    *   `wasm` et `wasm-safe` installent `zig-out/<id>/infer.wasm` et
        `zig-out/<id>/infer-safe.wasm`, avec
        `initial_memory = max_memory = memory_pages × 64 Kio` ;
    *   nouvelle étape `models` : installe le manifeste et le `.bin` de chaque
        modèle dans `zig-out/models/<id>/` ;
    *   `test` : les tests de géométrie (`geometry_tests`) et de l'outil de
        génération (`tool_tests`, qui intègrent le manifeste de `moka-9`)
        tournent une fois. Pour chaque modèle, un module `src_tests`
        (racine `src/test_root.zig`) reçoit les imports de ce modèle :
        `model_layout` (sa génération), `geometry`, `memory_config` (ses
        pages et la pile de 64 Kio), `go_model_bin` (son `.bin`) et
        `build_options`, qui porte `fixtures_dir = <fixtures>/<id>` et
        `model_id = <id>` ;
    *   `-Dfixtures` désigne désormais la racine, par défaut
        `oracle/fixtures`.

###### F6. `./src/infer/exports.zig` (Modify)

*   **Type / Scope** : ABI WASM.
*   **Proposed Changes** : exports `board_size()`, `feature_count()`,
    `policy_moves()` et `input_stride()`, tous en `u32`, avec les sources du
    tableau « ABI : quatre exports de dimensions ».

###### F7. `./tools/gen_model_layout.zig` (Modify)

*   **Type / Scope** : outil de build.
*   **Proposed Changes** :
    *   `parseConfig` refuse un manifeste où `policyMoveCount` diffère de
        `boardSize² + 1`, avec la nouvelle étiquette
        `error.PolicyMoveCountMismatch`. Le contrôle a lieu avant celui des
        tenseurs, dont les formes attendues en dépendent ;
    *   dans la boucle des tenseurs, après les contrôles de bornes existants,
        un tenseur `float32` dont `dataOffset` n'est pas un multiple de 4, ou
        un bloc d'échelles dont `scaleOffset` ne l'est pas, est refusé avec
        `error.MisalignedTensor` et un message qui nomme le tenseur. Placé
        après les bornes, ce contrôle laisse « an offset past weightsBytes is
        rejected » (`dataOffset` 113 646) recevoir `error.OutOfBounds` ;
    *   trois tests avec `expectRejected` : `policyMoveCount` à 81, un
        `dataOffset` `float32` décalé de 2 octets dans les bornes, un
        `scaleOffset` décalé de 2 octets dans les bornes.

    Le manifeste de `moka-9` passe ces contrôles : tous ses tenseurs `float32`
    et ses blocs d'échelles commencent à un multiple de 4.

###### F8. `./src/infer/buffers.zig` (Modify)

*   **Type / Scope** : tampons statiques.
*   **Proposed Changes** : « buffer sizes follow the memory table » renvoie
    `error.SkipZigTest` si `build_options.model_id` n'est pas `moka-9`. Le
    test du plus petit nombre de pages tourne pour tous les modèles.

###### F9. `./src/test_root.zig` (Modify)

*   **Type / Scope** : tests natifs.
*   **Proposed Changes** : « the generated layout matches the Moka model »
    renvoie `error.SkipZigTest` hors de `moka-9`. Le contrôle
    `WEIGHTS_F32_LEN` existe déjà dans ce test (`test_root.zig:19-20`) : il en
    sort, sans être dupliqué, pour former un nouveau test qui tourne avec
    tous les modèles. Ce test vérifie aussi
    `CONFIG.policy_moves == board_size² + 1`.

###### F10. `./src/infer/weights.zig` (Modify)

*   **Type / Scope** : tests natifs.
*   **Proposed Changes** : la limite de lecture de `readReferenceTensor`
    (ligne 83) passe de 1 Mio à 16 Mio : `policy.linear.weight` déquantifié
    pèse 2 090 912 octets en 19×19. Le chemin ne change pas, puisque
    `fixtures_dir` vaut déjà `<fixtures>/<id>`. La limite de `network.zig:119`
    (64 Mio) suffit.

###### F11. `./src/infer/network.zig` (Modify)

*   **Type / Scope** : moteur.
*   **Proposed Changes** : `forward(k)` est découpé en
    `pub fn forwardTrunk(k: usize) void` (le tronc) et
    `fn forwardHeads(k: usize) bool` (les têtes et le contrôle des sorties
    finies). `k` reste un `usize`, comme dans `forward`.
    `forward(k)` reste, et renvoie `forwardTrunk(k)` puis `forwardHeads(k)` :
    même ordre de calcul, mêmes sorties.

###### F12. `./src/infer/kernels.zig` (Modify)

*   **Type / Scope** : tests natifs.
*   **Proposed Changes** : `expectSameAsScalar` reçoit la taille du plateau en
    paramètre au lieu de `Geometry(9)`. Les six appels existants (formes
    `taps × entrées × sorties` : 9×12×32, 9×16×16, 1×32×16, 1×16×32,
    1×32×4, 1×32×2) sont faits pour N = 9 et N = 19.

###### F13. `./js/engine.ts` (Modify)

*   **Type / Scope** : colle JS du Worker.
*   **Proposed Changes** : `FEATURE_COUNT` et `MOVE_COUNT` disparaissent.
    L'interface `InferExports` reçoit les quatre exports ; `createEngine` lit
    `feature_count()` et `policy_moves()`. Les messages gardent leur texte,
    avec ces valeurs à la place des constantes : pour `moka-9`, ils restent
    identiques au caractère près.

###### F14. `./oracle/fixture-files.ts` (Create)

*   **Type / Scope** : module de l'oracle.
*   **Proposed Changes** :
    *   reçoit, sans changement, `packFloat32`, `packFloat64`, `packUint16` et
        `compareDirectories` (`oracle/generate-fixtures.ts:327-358, 490-508`),
        pour que F15, F26 et F36 les importent sans lancer le `main()` de
        l'oracle ;
    *   ajoute `compareFiles(expectedDirectory, actualDirectory, names)`, qui
        compare octet pour octet les fichiers nommés et échoue avec le nom du
        premier qui diffère.

###### F15. `./oracle/generate-fixtures.ts` (Modify)

*   **Type / Scope** : oracle.
*   **Proposed Changes** :
    *   option `--model <id>` (obligatoire) : lit le modèle dans
        `<models-dir>/<id>/`, écrit dans `oracle/fixtures/<id>/`. L'option
        `--models-dir` vaut par défaut `zig-out/models` ;
    *   au démarrage, contrôle de comportement :
        `createGameState().board.length` doit valoir `boardSize²` du
        manifeste, sinon échec avec les deux valeurs ;
    *   les tailles restent dérivées de `GO_BOARD_AREA` et de
        `GO_STUDENT_INPUT_PLANE_COUNT`. Les jeux et les sondes viennent d'une
        table indexée par `GO_BOARD_SIZE` (9 en J1 ; 19 ajouté par F33) ; la
        sonde « journal », au démarrage comme dans le jeu de validation, ne
        tourne que pour `moka-9` ;
    *   format v2 : `chosen-move.u16`, et les champs `formatVersion`,
        `modelId`, `modelSha256`, `boardSize`, `featureCount`, `moveCount`
        dans `index.json` ; `tensors/` écrit sous `oracle/fixtures/<id>/` ;
    *   `--check` régénère dans `oracle/fixtures/.check/<id>/` et compare avec
        `compareDirectories` (F14) ; le contrôle des fichiers copiés dans
        `vendor/` ne change pas.

###### F16. `./scripts/run-oracle.mjs` (Create)

*   **Type / Scope** : script Node.
*   **Proposed Changes** :
    *   supprime les anciens dossiers `oracle/fixtures/calibration`,
        `validation` et `tensors` s'ils existent (format v1, non versionnés) ;
    *   liste les dossiers de `--models-dir` (par défaut `zig-out/models/`),
        `moka-9` en premier, lit `boardSize` dans chaque manifeste, et lance
        `node oracle/dist/generate-<boardSize>.js --model <id> --models-dir <dossier>`,
        avec `--check` s'il est passé ;
    *   `--model <id>` restreint à un modèle ;
    *   une taille sans bundle échoue avec
        `no oracle bundle for board size <N> (model <id>)`.

###### F17. `./tests/inference-harness.mjs` et `./tests/infer-parity.test.mjs` (Modify)

*   **Type / Scope** : tests Node.
*   **Proposed Changes** :
    *   F1 lit les tailles dans les exports et dans `index.json`, et importe
        la règle de parité commune de `bench/parity-rule.ts` (F22) ;
    *   `infer-parity.test.mjs` boucle sur tous les modèles de
        `zig-out/models/`, et sur leurs artefacts de `zig-out/<id>/`. Un modèle
        sans données de référence fait échouer le test avec
        `model <id> has no reference data; run pnpm oracle` ; seul `build-js`
        saute un tel modèle. Les constantes
        `FEATURE_COUNT`, `INPUT_STRIDE`, `MOVE_COUNT`, `PASS_MOVE` et
        `WEIGHTS_BYTES` (lignes 11-16) disparaissent ; le poids est lu dans
        `zig-out/models/<id>/go-model.bin` ; les coups sont lus en `u16` ;
        `VALUE_ULP_LIMIT = 2n` et `MAX_BATCH = 8` restent ;
    *   la règle de parité commune, sur chaque jeu, devient un critère (le
        test échoue au premier écart, avec la position et l'indice) ;
    *   pour `moka-9` seulement : G3 avec `oracle/tolerance.json`, et sha256
        des entrées et des sorties égaux à ceux de F4 ;
    *   pour chaque modèle et artefact, valeur des exports :
        `board_size()` égale à `architecture.boardSize`, `policy_moves()` à
        `architecture.policyMoveCount` et à `moveCount`, `feature_count()` à
        `featureCount` et à `board_size()² × 12`, `input_stride()` à
        `(board_size()² + 1) × 12`, `weights_capacity()` à `weightsBytes`,
        `max_batch()` à 8.

###### F18. `./oracle/calibrate-tolerance.ts` (Modify)

*   **Type / Scope** : oracle.
*   **Proposed Changes** : lit `oracle/fixtures/moka-9/calibration/` ;
    `MOVE_COUNT` disparaît au profit de `moveCount` d'`index.json`. Le
    message de dépassement des plafonds ne dit plus « before any Zig code is
    written » ; il dit que la tolérance n'est pas modifiée et que G3 est à
    revoir. Le fichier écrit reste `oracle/tolerance.json`.

###### F19. `./scripts/build-js.mjs` (Modify)

*   **Type / Scope** : bundles et copies.
*   **Proposed Changes** :
    *   `copyModelFiles` copie dans `dist/` le modèle de
        `zig-out/models/moka-9/` au lieu de `vendor/moka/model/` ;
        `copyWasmArtifact` copie `zig-out/moka-9/infer.wasm` ;
    *   le bundle Node de l'oracle devient `oracle/dist/generate-9.js` ;
    *   `copyTolerance` disparaît. `writeBenchmarkInputs` écrit, pour chaque
        modèle de `zig-out/models/`, le dossier `bench/dist/<id>/` du
        Technical Plan (« Banc »), avec les tailles d'`index.json` ; les
        constantes `BENCHMARK_PLANE_COUNT`, `BOARD_AREA`, `POLICY_MOVE_COUNT`
        et `FEATURE_COUNT` (lignes 90-93) disparaissent de ce fichier. Un
        modèle sans données de référence est sauté, comme aujourd'hui ; un jeu
        de moins de 100 positions fait échouer le script avec le nombre
        trouvé ;
    *   `dist/worker.js` est aussi copié dans chaque `bench/dist/<id>/`, pour
        que le Worker de Zig y trouve son `infer.wasm` ;
    *   chaque copie ne fait rien si sa source manque, comme
        `copyWasmArtifact` aujourd'hui. C'est le cas du modèle de `dist/`, de
        l'`infer.wasm` de `dist/` et de `bench/dist/<id>/`, de `worker.js` et
        des données du banc. `pnpm build` passe ainsi sur un clone vierge, et
        `pnpm oracle` peut construire ses bundles avant d'avoir des données.

###### F20. `./tests/wasm-checks.test.mjs` (Modify)

*   **Type / Scope** : test Node.
*   **Proposed Changes** : boucle sur les modèles et leurs artefacts. La liste
    exacte des noms exportés (`EXPECTED_EXPORTS`) reçoit les quatre exports
    de dimensions. Pour tous : au moins 128 instructions `v128`, taille
    initiale égale à la taille maximale, aucun `memory.grow`. Pour `moka-9`
    seulement, deux assertions : 13 pages (nouvelle), taille après gzip sous
    32 Kio. Un nouveau test vérifie que `dist/` contient exactement
    `go-model.bin`, `go-model.json`, `infer.wasm` et `worker.js`.

###### F21. `./tests/e2e/worker.spec.ts` (Modify)

*   **Type / Scope** : tests Playwright, 9×9 seulement.
*   **Proposed Changes** :
    *   le cas nominal lit `/bench/dist/moka-9/positions.f32` dans la page ;
    *   `page.evaluate` renvoie les sorties brutes
        (`Array.from(policyLogits)`, `value`) ;
    *   le test lit `bench/dist/moka-9/logits-f32acc.f32` et
        `value-f32acc.f64` avec `fs` côté Node, puis applique
        `findParityMismatch` au lieu de la tolérance ;
    *   `tests/e2e/page.ts` ne change pas ;
    *   les constantes 9×9 et les messages attendus ne changent pas : ce test
        est réservé à `moka-9`.

###### F22. `./bench/parity-rule.ts` (Create) ; `./bench/bench.ts`, `./bench/run.mjs` (Modify)

*   **Type / Scope** : banc, première partie.
*   **Proposed Changes** :
    *   `parity-rule.ts` exporte
        `findParityMismatch(logits, values, expectedLogits, expectedValues)`,
        et `getUlpDistance` (déplacé de F1), sans dépendance à Node ni au DOM.
        La fonction renvoie `{ position, index, kind }` pour la première
        sortie qui viole la règle de parité commune : `index` est l'indice du
        logit, ou `null` pour la valeur, et `kind` vaut `"logit"` ou
        `"value"`. Elle renvoie `null` si toutes les sorties sont conformes.
        F17, F21 et `bench.ts` l'importent ;
    *   `bench.ts` lit `model=<id>` dans l'URL (par défaut `moka-9`), puis
        `/bench/dist/<id>/bench-inputs.json`. `POSITION_COUNT`,
        `FEATURE_COUNT` et `MOVE_COUNT` (lignes 5-7) viennent de ce fichier ;
        `MANIFEST_URL`, `WEIGHTS_URL` et `ZIG_WASM_URL` pointent sous
        `/bench/dist/<id>/`, `ZIG_WORKER_URL` vers `/bench/dist/<id>/worker.js`,
        `MOKA_WORKER_URL` reste `/bench/dist/moka-worker.js` ;
    *   `assertMatchesReference` applique la règle de parité commune contre
        `logits-normal.f32` et `value-normal.f64` pour Moka, contre
        `logits-f32acc.f32` et `value-f32acc.f64` pour Zig ; la lecture de
        `tolerance.json` et `valueOffset` disparaissent ;
    *   `run.mjs` reçoit une fonction `readValueOption(name)` pour
        `--model <id>`, à côté de `readOption`, qui reste booléenne. Il
        passe `model=<id>` à chaque mesure, lit `modelSha256` et `boardSize`
        dans `bench-inputs.json`, les écrit une fois en tête du fichier de
        résultats, et lit la référence dans `baseline-<id>.json`.

    `bench/index.html` ne change pas : il charge `dist/bench.js` et ne lit
    aucune taille.

###### F23. `./bench/results/baseline-moka.json` → `./bench/results/baseline-moka-9.json` (Move)

*   **Type / Scope** : données versionnées.
*   **Proposed Changes** : déplacement sans changement de contenu.

###### F24. `./tests/build-errors.test.mjs` (Create)

*   **Type / Scope** : test Node.
*   **Proposed Changes** :
    *   `zig build wasm -Dmodel=nope` échoue, et sa sortie d'erreur contient
        `unknown model nope` et `moka-9` ;
    *   `node scripts/run-oracle.mjs --models-dir <tmp>` échoue, avec
        `no oracle bundle for board size 13 (model bad)`. Ici `<tmp>` est un
        dossier créé par `fs.mkdtempSync`, qui contient
        `bad/go-model.json`, copie du manifeste de `moka-9` où
        `architecture.boardSize` vaut 13.

###### F25. `./package.json` (Modify)

*   **Type / Scope** : scripts.
*   **Proposed Changes** :

    | Script | Commande |
    | --- | --- |
    | `build` | `zig build models wasm wasm-safe && pnpm build:js` |
    | `oracle` | `zig build models && pnpm build:js && node scripts/run-oracle.mjs` |
    | `oracle:check` | `zig build models && pnpm build:js && node scripts/run-oracle.mjs --check` |
    | `calibrate` | `pnpm build:js && node oracle/dist/calibrate.js` |
    | `test` | `zig build test && zig build models wasm wasm-safe && node --test "tests/*.test.mjs" "tests/*.test.ts"` |
    | `bench` | `zig build models wasm && pnpm build:js && node bench/run.mjs` |
    | `check:all` | inchangé |

#### J2 : générateur et fichiers de `synthetic-19`

###### F26. `./oracle/generate-synthetic-model.ts` (Create)

*   **Type / Scope** : générateur Node.
*   **Proposed Changes** :
    *   écrit `models/synthetic-19/go-model.json` et `go-model.bin` selon la
        recette du Technical Plan, avec les constantes `POLICY_SCALE_FACTOR`
        et `VALUE_SCALE_FACTOR` (1) ;
    *   lit `moka-9` dans `vendor/moka/model/` et refuse un tenseur que la
        recette ne classe ni comme repris ni comme généré ;
    *   après l'écriture, charge le résultat avec `GoModelRuntime.create` de
        Moka et lance une inférence sur un vecteur d'entrée nul ; un échec, ou
        une sortie non finie, fait échouer le générateur ;
    *   `--check` régénère dans un dossier créé par
        `fs.mkdtempSync(path.join(os.tmpdir(), "synthetic-19-"))`. Il compare
        `go-model.json` et `go-model.bin` à ceux de `models/synthetic-19/`
        avec `compareFiles` (F14), puis supprime le dossier. `PROVENANCE.md`,
        écrit à la main, n'est pas comparé ;
    *   réutilise `oracle/mulberry32.ts`.

###### F27. `./models/synthetic-19/` (Create)

*   **Type / Scope** : modèle versionné.
*   **Proposed Changes** : `go-model.json`, `go-model.bin` et `PROVENANCE.md`
    (origine du tronc, licence MIT de Moka, graine 19, valeur des deux
    constantes d'échelle, commande `pnpm models`).

###### F28. `./scripts/build-js.mjs` et `./package.json` (Modify)

*   **Type / Scope** : intégration.
*   **Proposed Changes** :
    *   bundle Node `oracle/dist/generate-synthetic-model.js` ;
    *   scripts `models` = `pnpm build:js && node oracle/dist/generate-synthetic-model.js`
        et `models:check` = `pnpm build:js && node oracle/dist/generate-synthetic-model.js --check` ;
    *   `check:all` = `pnpm typecheck && pnpm gen && pnpm build && pnpm models:check && pnpm oracle && pnpm oracle:check && pnpm test && pnpm test:e2e`.
        `models:check` vient après `pnpm build`, dont `build:js` doit passer
        sur un clone vierge (F19). `pnpm typecheck` y entre parce qu'esbuild
        ne vérifie pas les types.

    `synthetic-19` n'entre pas encore dans `MODELS` : aucun autre
    consommateur ne le voit.

###### F29. `./THIRD_PARTY_NOTICES.md` (Modify)

*   **Type / Scope** : licences.
*   **Proposed Changes** : ajouter une entrée pour `models/synthetic-19/`, qui
    contient les poids du tronc de Moka, sous licence MIT.

#### J3 : oracle 19×19, contrôle de santé, enregistrement

###### F30. `./oracle/constants-19.ts` (Create)

*   **Type / Scope** : constantes de règles 19×19.
*   **Proposed Changes** : `export *` depuis `vendor/moka/src/constants.ts`,
    et redéfinition des cinq valeurs du Technical Plan.

###### F31. `./scripts/build-js.mjs` (Modify)

*   **Type / Scope** : bundles.
*   **Proposed Changes** : bundle `oracle/dist/generate-19.js`, avec un plugin
    `onResolve` :
    1.  filtre `/(^|\/)constants(\.ts)?$/` ;
    2.  `candidate = path.resolve(args.resolveDir, args.path)` ; si
        `path.extname(candidate)` est vide, ajouter `.ts` ;
    3.  si `candidate === path.resolve("vendor/moka/src/constants.ts")` et
        `args.importer !== path.resolve("oracle/constants-19.ts")`, renvoyer
        `{ path: path.resolve("oracle/constants-19.ts") }` ;
    4.  sinon, ne rien renvoyer : esbuild résout normalement.

    `./runtime-constants` passe le filtre mais pas l'étape 3.

###### F32. `./oracle/health.ts` et `./oracle/moka-runtime-instrumented.ts` (Create, Modify)

*   **Type / Scope** : oracle.
*   **Proposed Changes** :
    *   `moka-runtime-instrumented.ts` reçoit l'option
        `onTensor(name, values)`, appelée sur les tenseurs listés dans
        « Contrôle de santé ». Sans l'option, les sorties ne changent pas ;
        `assertNormalParity` et le test bit à bit de F17 le vérifient ;
    *   `health.ts` exporte `createHealthCounter()`, dont les méthodes sont
        `observe(name, values)` (appelée par `onTensor`),
        `observeOutputs(logits, value, legalMoves)` et `toJSON()`, qui
        renvoie les clés `health.*` de « Contrôle de santé » ;
    *   `health.ts` exporte aussi `assertHealthy(modelId, health)`. Elle
        applique les critères bloquants et lève une `Error` avec le message
        `model <id> failed the health check: <clé> = <valeur>, limit <limite>`.
        Le `main()` de l'oracle convertit cette erreur en échec du
        processus.

###### F33. `./oracle/generate-fixtures.ts` (Modify)

*   **Type / Scope** : oracle et tests.
*   **Proposed Changes** :
    *   la table des jeux de `generate-fixtures.ts` reçoit l'entrée 19
        (Technical Plan, « Jeux de positions ») ;
    *   `generate-fixtures.ts` crée un compteur neuf par jeu, branche
        `onTensor` sur l'instance `f32Accumulate`, écrit `health` dans
        `index.json` de chaque jeu, et appelle `assertHealthy` pour
        `synthetic-19`. Pour `moka-9`, `health` est écrit mais pas jugé.

###### F34. `./tests/oracle-health.test.ts` (Create)

*   **Type / Scope** : test Node.
*   **Proposed Changes** :
    *   `assertHealthy` refuse, avec le message de F32 :
        *   une sortie `NaN` ;
        *   un subnormal dans les têtes ;
        *   une position où toutes les unités de `value.hidden` sont nulles ;
        *   10 % de valeurs saturées ;
        *   une étendue médiane des logits nulle ;
    *   il accepte des subnormaux dans le tronc.

###### F35. `./build.zig` (Modify)

*   **Type / Scope** : build.
*   **Proposed Changes** : `synthetic-19` (`models/synthetic-19`, 66 pages)
    entre dans `MODELS`.

###### F36. `./oracle/game-nxn.ts` (Create, seulement si le plugin échoue)

*   **Type / Scope** : repli de l'oracle.
*   **Déclencheur** : `generate-19.js`, construit avec le plugin de F31,
    échoue au contrôle de comportement.
*   **Proposed Changes** :
    *   `oracle/game-nxn.ts` est une copie de `vendor/moka/src/game.ts` qui
        exporte `createRules(constants)`. Cette fonction renvoie les fonctions
        que `generate-fixtures.ts` importe aujourd'hui des règles de Moka ;
    *   `generate-fixtures.ts` reçoit ses règles par un nouveau module
        `oracle/rules.ts`. `scripts/build-js.mjs` construit `generate-9.js` et
        `generate-19.js` avec un `oracle/rules.ts` qui appelle `createRules`
        avec les constantes 9×9 ou 19×19. Il construit aussi
        `oracle/dist/generate-9-vendor.js`, avec un `oracle/rules.ts` qui
        réexporte `vendor/moka/src/game.ts` ;
    *   `package.json` : `oracle:check` lance en plus
        `node oracle/dist/generate-9-vendor.js --model moka-9 --out oracle/fixtures/.vendor/moka-9`.
        Il compare ensuite le résultat à `oracle/fixtures/moka-9/` avec
        `compareDirectories` (F14). `generate-fixtures.ts` accepte `--out`
        (par défaut `oracle/fixtures/<id>`).

#### J4 : artefact de profilage et verdict du banc

###### F37. `./src/infer/exports.zig` (Modify)

*   **Type / Scope** : ABI WASM.
*   **Proposed Changes** : `Status` devient `pub`, pour que F38 renvoie les
    mêmes codes.

###### F38. `./src/infer/profile_exports.zig` (Create)

*   **Type / Scope** : racine de l'artefact de profilage.
*   **Proposed Changes** :
    *   redéclare `pub const panic = std.debug.no_panic`, qui ne vaut que
        dans le fichier racine ;
    *   force l'analyse des exports d'`exports.zig` par
        `comptime { _ = @import("exports.zig"); }` ;
    *   importe lui-même `buffers.zig`, `weights.zig` et `network.zig` ;
    *   `export fn forward_trunk(count: u32) u32` contrôle les mêmes choses
        qu'`infer`, dans le même ordre (`exports.zig:51-52`) :
        *   `@intFromEnum(Status.bad_batch_count)` si `count` n'est pas entre
            1 et `MAX_BATCH` ;
        *   puis `@intFromEnum(Status.weights_not_loaded)` si `weights.loaded`
            est faux ;
        *   sinon, appelle `network.forwardTrunk(k)` pour chaque `k` de
            `0..count` et renvoie `@intFromEnum(Status.ok)`.

###### F39. `./build.zig` (Modify)

*   **Type / Scope** : build.
*   **Proposed Changes** : `addInferWasm` reçoit la racine en paramètre
    (`src/infer/exports.zig` pour `wasm` et `wasm-safe`). Nouvelle étape
    `wasm-profile`, qui installe `zig-out/<id>/infer-profile.wasm`
    (`ReleaseFast`, racine `src/infer/profile_exports.zig`) pour chaque
    modèle.

###### F40. `./scripts/build-js.mjs` et `./package.json` (Modify)

*   **Type / Scope** : intégration.
*   **Proposed Changes** : `writeBenchmarkInputs` copie
    `zig-out/<id>/infer-profile.wasm`, si le fichier existe, dans
    `bench/dist/<id>/`. Les scripts
    deviennent `build` = `zig build models wasm wasm-safe wasm-profile && pnpm build:js`,
    `test` = `zig build test && zig build models wasm wasm-safe wasm-profile && node --test "tests/*.test.mjs" "tests/*.test.ts"`
    et `bench` = `zig build models wasm wasm-profile && pnpm build:js && node bench/run.mjs`.

###### F41. `./tests/wasm-checks.test.mjs` (Modify)

*   **Type / Scope** : test Node.
*   **Proposed Changes** : pour `infer-profile.wasm`, la liste exacte des
    noms exportés est celle d'`infer.wasm` plus `forward_trunk` ; les
    contrôles `v128`, taille initiale égale à la taille maximale et
    `memory.grow` s'appliquent ; ni le nombre de pages ni la taille après
    gzip. Le test de la liste exacte de `dist/` (F20) garantit déjà
    qu'`infer-profile.wasm` n'y est pas.

###### F42. `./bench/bench.ts` et `./bench/run.mjs` (Modify)

*   **Type / Scope** : banc, seconde partie.
*   **Proposed Changes** :
    *   `bench.ts` : paramètre `stage=trunk`, valable seulement avec
        `runtime=zig&worker=0`. La page instancie
        `/bench/dist/<id>/infer-profile.wasm` sans `createEngine`, puis copie et
        charge les poids. Elle crée le moteur de banc du Technical Plan
        (« Tronc seul »), dont `infer(features)` fait `input.set(features)`
        puis `forward_trunk(1)`, et le chronomètre par la même boucle que Zig
        sans Worker. Elle contrôle ensuite `infer(1)` sur la première position
        avec la même instance ;
    *   `run.mjs --model <id> --verdict` : `<id>` est le modèle jugé
        (`synthetic-19` dans CUJ 1), et `moka-9` est toujours mesuré comme
        contrôle, dans l'ordre du Technical Plan. Avec `--model moka-9`, seul
        `moka-9` est mesuré, comme aujourd'hui. Le fichier reste
        `bench/results/<horodatage>.json` :

        ```json
        {
          "date": "2026-11-02T10:00:00.000Z",
          "machine": { "cpu": "…", "browserVersion": "…", "node": "…" },
          "runs": [{ "model": "synthetic-19", "round": 1, "runtime": "zig", "worker": "trunk", "p50Ms": 0 }],
          "models": {
            "<id>": {
              "modelSha256": "…",
              "boardSize": 19,
              "summary": { "moka": { "1": { "medianP50Ms": 0 }, "0": { "medianP50Ms": 0 } }, "zig": { "1": { "medianP50Ms": 0 }, "0": { "medianP50Ms": 0 }, "trunk": { "medianP50Ms": 0 } } },
              "ratio": 0,
              "headShare": 0,
              "baselineDriftPercent": null
            }
          },
          "verdict": { "model": "synthetic-19", "threshold": 2, "ratio": 0, "passed": true }
        }
        ```

        Les valeurs sont des exemples. Les règles du fichier :
        *   chaque run reçoit `model` ; ceux de l'étape tronc ont
            `runtime: "zig"` et `worker: "trunk"` ;
        *   `summary` est `buildSummary` appliqué aux runs du modèle ;
        *   `ratio` = `summary.moka["1"].medianP50Ms` /
            `summary.zig["1"].medianP50Ms` ;
        *   `headShare` suit la formule du Technical Plan ;
        *   `baselineDriftPercent` est calculé contre `baseline-<id>.json`, et
            vaut `null` sans ce fichier. Le contrôle de dérive existant
            s'applique à chaque modèle qui a une référence ;
        *   `passed` = `ratio ≥ 2` pour le modèle jugé. La commande sort avec
            un code non nul si `passed` est faux.

###### F47. `./tests/profile-exports.test.mjs` (Create)

*   **Type / Scope** : test Node.
*   **Proposed Changes** : pour chaque modèle, sur
    `zig-out/<id>/infer-profile.wasm`, `forward_trunk` renvoie :
    *   2 pour `count` 0 et 9, avant comme après `load_weights` ;
    *   1 pour `count` 1 avant `load_weights` ;
    *   0 pour `count` 1 et 8 après `load_weights`.

    Après `forward_trunk(1)`, `infer(1)` renvoie des sorties conformes à la
    règle de parité commune sur la première position du jeu de validation.

#### J5 : verdict et documents

###### F43. `./docs/designs/zig-wasm-nxn.md` (Modify)

*   **Type / Scope** : design.
*   **Proposed Changes** : compléter « Mesures enregistrées » avec le
    résultat de `pnpm bench --model synthetic-19 --verdict` (la partie J3 est
    remplie dans le commit de J3) ; passer le statut à « Implemented ».

###### F44. `./docs/designs/zig-wasm-moka.md` (Modify)

*   **Type / Scope** : design.
*   **Proposed Changes** :
    *   sous l'en-tête : « Partiellement amendé par `zig-wasm-nxn.md` » ;
    *   Non-Goal « Modèle 19×19 » (lignes 184-187) : remplacer « Les têtes et
        les tests des noyaux à N = 19 viendront avec le design du modèle
        19×19 » par « Les tests des noyaux à N = 19 et un modèle synthétique
        19×19 aux têtes dépendantes de N² sont ajoutés par
        `zig-wasm-nxn.md` » ;
    *   Alternative H (ligne 808) et Future Improvements (ligne 1005) : un
        renvoi d'une ligne vers `zig-wasm-nxn.md`.

###### F45. `./README.md` (Modify)

*   **Type / Scope** : documentation.
*   **Proposed Changes** :
    *   ajouter la taille du plateau à chaque temps mesuré ;
    *   une section « 19×19 » avec le résultat de NX6 et ses limites (modèle
        synthétique, comparaison avec le runtime de Moka et non avec tout
        JavaScript) ;
    *   tableau des commandes (ligne 42 et suivantes) : `pnpm calibrate` porte
        sur `moka-9` ; ajouter `pnpm models`, `pnpm models:check` et l'option
        `--model <id>` de `pnpm bench` ;
    *   ligne 46 : la ligne de `check:all` cite la nouvelle chaîne (F28) ;
    *   « Quand Moka publie de nouveaux poids » (lignes 151-164) : ajouter
        `pnpm models` après l'étape 1, puisque le tronc de `synthetic-19`
        vient de `moka-9` ;
    *   lignes 160-161 : la tolérance n'est plus copiée dans `bench/dist/` ;
    *   lignes 174-183 : préciser que 13 pages et 82 logits valent pour
        `moka-9`, et donner les 66 pages et 362 logits de `synthetic-19`.

#### Outils de diagnostic (seulement si NX4 échoue)

###### F46. Outils D1 à D4 du premier design (Create, seulement si NX4 échoue)

*   **Type / Scope** : diagnostic, hors des artefacts publiés.
*   **Proposed Changes** : reprend les fichiers D1 à D4 du premier design
    (`zig-wasm-moka.md`, « Outils de diagnostic »), avec ces adaptations
    N×N :
    *   D1 (`src/infer/reference.zig`) réutilise `scalarReference` de
        `kernels.zig`, déjà paramétré par `Geometry(N)` après F12 ;
    *   D2 (`wasm-debug`) passe par le registre et installe
        `zig-out/<id>/infer-debug.wasm` ;
    *   D3 (`dumpLayers`) écrit dans `oracle/fixtures/<id>/layers/`, avec
        `boardSize²` lignes par couche, lu dans `index.json` ;
    *   D4 (`tests/infer-layers.test.mjs`) boucle sur les modèles comme F17.

### Verification Plan

*   **J0** : pendant J0, après F1 à F3,
    `pnpm build && pnpm oracle && node scripts/snapshot-outputs.mjs --wasm zig-out/bin --fixtures oracle/fixtures --out oracle/snapshots/moka-9.json`,
    puis `pnpm test` (F2 ne change aucun résultat).
*   **Compilation/Build** : `pnpm gen`, `pnpm build`.
*   **Unit Tests** : `zig build test`, puis
    `node --test "tests/*.test.mjs" "tests/*.test.ts"`, dont
    `tests/build-errors.test.mjs`, `tests/oracle-health.test.ts` et
    `tests/profile-exports.test.mjs`.
*   **Oracle et modèle** : `pnpm models:check` (à partir de J2),
    `pnpm oracle`, `pnpm oracle:check`.
*   **Static Analysis & Linting** : `zig fmt --check src tools build.zig`,
    `pnpm typecheck`.
*   **E2E/Integration** : `pnpm test:e2e` (9×9 seulement).
*   **Chaque jalon** : `pnpm check:all` passe sur un clone vierge ; à partir
    de J1, F17 vérifie NX7 contre F4.
*   **J1** : `pnpm bench --model moka-9` termine et accepte les sorties des
    deux moteurs.
*   **J3** : noter la durée de `pnpm check:all` et de chaque génération de
    l'oracle, et les valeurs `health`, dans « Mesures enregistrées ».
*   **J4** : `pnpm bench --model synthetic-19 --verdict` termine et écrit le
    fichier de F42 ; le verdict n'est pas encore jugé.
*   **J5** : `pnpm bench --model synthetic-19 --verdict` sur la machine de G1.

--------------------------------------------------------------------------------

*Drafted with EGM*
