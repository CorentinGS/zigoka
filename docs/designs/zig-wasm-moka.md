# Moteur d'inférence Zig/WASM sans allocation pour le modèle Moka 9×9

*   **Author:** corenting
*   **Status:** Approved
*   **Date:** 2026-10-30

--------------------------------------------------------------------------------

## Overview

Ce projet est un dépôt séparé de [Moka](https://github.com/millionco/moka), un
réseau de neurones de Go 9×9 publié par la société Million sous licence MIT.
Moka s'exécute dans le navigateur, en TypeScript. Son modèle compte 105 353
paramètres, et ses poids INT8 sont dans un fichier `go-model.bin` de
113 648 octets.

Le projet exécute l'inférence de ce modèle avec un moteur écrit en Zig et
compilé en WebAssembly (`wasm32-freestanding`, instructions `simd128`). Le
moteur ne fait aucune allocation dynamique : toute sa mémoire de travail est
réservée à la compilation.

**Zig est une contrainte du projet, pas une conséquence du problème.**
L'auteur veut un moteur Zig sans allocation qui pourra évoluer vers un modèle
19×19. Les autres approches (optimiser le TypeScript de Moka, Rust, un runtime
généraliste) sont discutées dans les Alternatives.

Le projet livre un Web Worker (`worker.js` + `infer.wasm`) qui remplace celui
de Moka et accepte exactement les mêmes messages. Un intégrateur garde le
client de Moka (`GoModelWorkerClient`), ses règles et ses poids, et change
seulement l'URL du Worker. Ce qui reste à la charge de l'intégrateur est
décrit dans le CUJ 2 : servir `worker.js` et `infer.wasm` depuis l'origine de
la page, et autoriser la compilation WASM dans la politique de sécurité du
contenu (CSP) de `worker.js`. Le périmètre exclu est décrit dans les
Non-Goals.

La référence est Moka au commit
[`a418039e`](https://github.com/millionco/moka/tree/a418039e9d823779e3e6d0ea17a244230abc8379).
Les fichiers utilisés seront copiés dans `vendor/moka/` avec leur licence.
Dans ce document, les chemins comme `src/runtime.ts:263` sont relatifs à la
racine de Moka à ce commit.

## The Problem

**Ce que la v1 apporte : lever un risque.** Le paquet publié de Moka
(`dist/worker.js` + `go-model.bin`) appelle `infer` une fois par coup
(`README.md:86-89`). Chaque appel coûte environ 3 ms. Moka juge ce coût « bien
en dessous du budget d'interaction », pour une cadence de 500 ms par coup
(`experiment-log.md:165-169`). Pour cet usage, un moteur plus rapide ne
change rien de visible.

La v1 sert donc à répondre à une question : **un moteur `simd128` en un seul
appel par inférence est-il au moins 2 fois plus rapide que le JavaScript de
Moka, dans un Worker ?** Moka n'a jamais pu le montrer (voir plus bas). Une
réponse positive débloque deux usages où la latence compte :

*   **La recherche du site de Moka.** Le site joue chaque coup avec une
    recherche à 28 visites, avec la moyenne des 8 symétries à la racine. C'est
    le seul gain de force reproductible du dernier cycle de Moka
    (`experiment-log.md:1747-1755`, `:1783`). Une réponse prend 252 à 254 ms à
    chaud.
    *   Une réponse coûte environ 35 inférences : 8 pour la racine, une par
        symétrie, et environ 27 pour les autres visites. À 3,2 ms chacune,
        cela fait environ 112 ms, soit 44 % du temps de réponse. Ce calcul est
        une estimation : il n'a pas été mesuré.
    *   Un moteur 2 fois plus rapide réduirait donc le temps de réponse
        d'environ 22 % (loi d'Amdahl). Le même budget de temps permettrait
        aussi plus de visites.
    *   Ce site utilise un autre fichier de poids (mixte INT8/INT16, 138 616
        octets, `experiment-log.md:1573`) et son propre Worker. Aucun des deux
        n'est dans le dépôt de Moka. Les préconditions pour les prendre en
        charge sont listées dans Future Improvements.
*   **Un futur modèle 19×19**, avec environ 4,5 fois plus d'intersections et
    probablement plus de canaux.

Si G4 échoue, le projet s'arrête après la phase 1, qui ne coûte que le
minimum nécessaire pour mesurer. Si G4 passe mais que le site n'est jamais
migré, la v1 reste un Worker plus rapide pour le paquet publié, sans gain
visible pour les joueurs.

**Limites du code actuel.** Le moteur TypeScript de Moka (`src/runtime.ts`) :

*   calcule tout en scalaire, sans instructions SIMD ;
*   crée un nouveau `Float32Array` à chaque couche, soit plus de 50 allocations
    par inférence ;
*   teste les bords du plateau dans les boucles sur la fenêtre 3×3 de chaque
    convolution (`src/runtime.ts:113`, `:120`).

Les auteurs de Moka ont déjà essayé le WASM (`wasm/src/lib.rs`). Ce crate
n'implémente que la ReLU et une couche linéaire, et il n'a jamais été plus
rapide que le moteur JS. Le journal avance comme cause probable les copies et
les passages répétés entre JS et WASM (`experiment-log.md:188`).

**Pourquoi on ne peut pas encore mesurer le gain.** Le banc de Moka envoie des
entrées de 810 valeurs au lieu de 972 (`web/constants.ts:33`). Le réseau lit
au-delà de la fin du tableau et calcule sur des `NaN`. Le chiffre du README
(8,8 ms en médiane, sur Apple Silicon) n'est donc pas une base fiable. Les
3,2 ms du journal viennent d'un autre protocole, sur une machine que le
journal ne décrit pas (`experiment-log.md:144-165`). Ils ne servent ici que
d'ordre de grandeur. La phase 1 corrige le banc et mesure Moka à nouveau.

## Goals

### Étapes

| Phase | Étape | Contenu | Critère de sortie |
| --- | --- | --- | --- |
| 1 (prototype) | S0 | outillage, copie de Moka, oracle, calibration de la tolérance, banc corrigé, mesure de Moka | données de référence déterministes, tolérance sous les plafonds, référence G1 enregistrée |
| 1 | S1 | squelette Zig, géométrie, outil de build | `infer.wasm` construit à partir du manifeste figé |
| 1 | S2 | chargement et déquantification des poids | poids identiques bit à bit à ceux de Moka |
| 1 | S3 | noyaux, passe avant, Worker minimal, parité | G3 atteint, puis **verdict G4** |
| 2 (consolidation, seulement si G4 passe) | S4 | contrôles d'erreur, tests e2e (dont la CSP qui s'applique au Worker), vérifications de l'artefact, build `ReleaseSafe`, préparation INT16, CI, README | `pnpm check:all` au vert sur la machine de développement ; la CI GitHub s'ajoute si un dépôt distant existe |

### Objectifs

*   **G1 — Mesure de référence (S0).** On mesure le p50, le p95 et le temps
    d'initialisation du moteur TypeScript de Moka, avec et sans Worker, sur
    des entrées valides de 972 valeurs.
    *   La mesure se fait sur **la machine de mesure** : la machine Linux de
        l'auteur, sous Chromium. Le CPU et la version de Chromium sont
        enregistrés avec les résultats.
    *   Le résultat de cette mesure est appelé **la référence G1**.
*   **G2 — Oracle déterministe (S0).** Des données de référence sont produites
    à partir du code de Moka. Deux générations donnent des fichiers identiques
    octet pour octet.
*   **G3 — Parité de l'inférence (S3).** Sur le jeu de validation, et pour le
    `.wasm` exécuté sous Node :
    *   l'écart absolu maximal des logits et de la valeur reste sous les seuils
        calibrés en S0 sur le jeu de calibration (plafonds : 2e-3 et 1e-3) ;
    *   le meilleur coup légal est le même. Une différence n'est acceptée
        qu'en cas de quasi-égalité : selon les logits de Moka, le coup choisi
        par Moka et celui choisi par Zig sont à au plus
        `2 × tolérance des logits` l'un de l'autre, puisque chacun des deux
        logits peut dériver de la tolérance ;
    *   aucune sortie n'est `NaN` ni infinie.
*   **G4 — Vitesse (S3).** Le p50 de l'aller-retour `infer` (une inférence
    unique, via le Worker) avec le moteur Zig est au plus la moitié de celui de
    la référence G1, mesurée de nouveau dans la même session Chromium.
    *   **Critère d'arrêt :** si le seuil n'est pas atteint à la fin de S3, le
        projet s'arrête et la phase 2 n'a pas lieu.
    *   **Mesures informatives, sans effet sur la décision :** inférence sans
        Worker, temps d'initialisation, latence des 10 premières inférences
        après `ready`, et durée d'une rafale de 35 appels `infer` successifs,
        soit le coût estimé d'un coup du site à 28 visites.
    *   **Limite connue :** le Worker coûte environ 0,3 ms fixes par
        aller-retour (`experiment-log.md:165`). Si la référence G1 descend à
        1 ms, le moteur Zig doit tenir en 0,5 ms aller-retour compris, soit
        environ 0,2 ms de calcul. Ce cas déclenche le critère d'arrêt : le
        seuil n'est pas révisé.
    *   Le verdict ne vaut que pour la machine de mesure.
*   **G5 — Zéro allocation, vérifiable (S3, S4).** Le `.wasm` ne contient aucun
    allocateur, et sa mémoire linéaire a une taille fixe. La taille de cette
    mémoire ne change pas pendant 1 000 inférences, et le binaire ne contient
    aucune instruction `memory.grow`.
*   **G6 — Intégrité des poids (S2, S3).** Le moteur charge le `go-model.bin`
    d'origine, sans modification. Le Worker calcule lui-même le sha256 des
    octets reçus, et refuse tout fichier dont la longueur ou le sha256 diffère
    de celui du build. Ce contrôle ne dépend pas de l'émetteur du message.
    Après déquantification, les poids f32 sont identiques bit à bit à ceux de
    Moka.
*   **G7 — Préparation de l'extension INT16 (S4).** L'outil de build accepte des
    tenseurs `int16`, vérifiés sur un tenseur synthétique. Le modèle du site
    demande en plus les préconditions listées dans Future Improvements.
*   **G8 — Build reproductible (S1, S4).** La version de Zig est fixée en S1.
    En S4, la CI lance les contrôles de la section Build et la parité.

## Non-Goals

*   **Règles, encodage des plans et décompte des points en Zig.** Ils restent
    ceux de Moka. Les porter n'accélère pas l'inférence. Sur le thread
    principal, cela imposerait en plus une initialisation asynchrone, donc un
    écart d'API. Le seul usage qui justifierait ce portage est une recherche
    dans le Worker. Il fera l'objet d'un design séparé.
*   **Améliorer la force de jeu.** Ce design ne mesure que la latence par
    inférence.
*   **Protocole de lots (`inferBatch`) côté JS.** Rien ne l'appelle en v1. Il
    viendra avec le design de la recherche. L'ABI Zig accepte déjà jusqu'à 8
    positions par appel.
*   **Le modèle mixte INT8/INT16 du site de Moka en v1.** Ni ses poids ni une
    référence ne sont disponibles (voir Future Improvements).
*   **Activations quantifiées en INT8 ou INT16**, et la calibration de leurs
    échelles. Moka ne fournit pas ces échelles. C'est un sujet de v2, à juger
    sur la part des positions où le meilleur coup légal reste le même qu'en
    v1.
*   **Modèle 19×19.** La géométrie et les noyaux sont écrits de façon
    générique en N, ce qui ne coûte rien. Seule la géométrie est testée à
    N = 19. Les têtes et les tests des noyaux à N = 19 viendront avec le
    design du modèle 19×19.
*   **Recherche MCTS/PUCT.**
*   **Les autres formats que Moka accepte** (définis dans « Architecture du
    réseau »). L'outil de build les refuse explicitement.
*   **Zéro allocation côté JavaScript.** La garantie couvre la mémoire WASM
    uniquement.
*   **Navigateurs autres que Chromium, et mobile.** Ils ne sont ni mesurés ni
    testés en v1.
*   **Build natif AVX2 comme cible de performance.** Le build natif sert
    uniquement aux tests.

## Open Questions

*   **Seuils de tolérance définitifs (G3).** Ils seront fixés en S0, selon la
    règle de « Oracle et tolérance ».
*   **Plafond de taille de `infer.wasm`.** La proposition est 32 Kio après
    gzip, contre environ 4 Kio pour le runtime de Moka. Elle sera confirmée
    à la fin de S3. Le build de S1 (119 octets, sans noyaux) ne dit rien de
    la taille finale.
*   **Taille de la mémoire linéaire et de la pile.** Les valeurs proposées sont
    1 Mio et 64 Kio. Elles seront confirmées en S2, d'après la taille réelle
    des variables globales : le nombre de pages est le plus petit qui laisse
    au moins 25 % de marge au-dessus des tampons et de la pile.
*   **Version exacte de Zig.** Résolu en S1 : Zig 0.17.0, la dernière
    version stable au 2026-10-30. Elle est écrite dans `mise.toml` et dans
    `minimum_zig_version` de `build.zig.zon`.
*   **Quelle CSP Chromium applique au Worker.** D'après la spécification
    HTML, un Worker dédié chargé en http(s) reçoit la CSP des en-têtes de
    `worker.js`, et non celle de la page. Ce point sera vérifié en S4 par les
    tests e2e (fichier 36), et le résultat sera écrit dans le CUJ 2 et le
    README. Il ne sert qu'à l'intégration, donc pas au verdict G4.

## Critical User Journeys (CUJs)

### CUJ 1 : Mesurer les deux moteurs côte à côte

*   **Given** : le dépôt est cloné, la version de Zig fixée est installée via
    mise, et Chromium est disponible pour Playwright.
*   **When** : le développeur lance `pnpm bench`.
*   **Then** : le banc décrit dans « Banc de mesure » écrit un tableau
    p50/p95/initialisation pour chaque moteur, avec et sans Worker.

### CUJ 2 : Remplacer le Worker dans une page qui utilise Moka

*   **Given** : une page suit le README de Moka. Elle crée
    `new GoModelWorkerClient(new Worker("/moka/worker.js", { type: "module" }))`
    et sert `go-model.json` et `go-model.bin`.
*   **When** : l'intégrateur sert `worker.js` et `infer.wasm` dans un même
    dossier, sur l'origine de la page (un Worker doit avoir la même origine
    que la page), et remplace l'URL du Worker par celle du projet. Si le
    serveur envoie une CSP avec `worker.js`, elle doit autoriser
    `'wasm-unsafe-eval'` dans `script-src`. Pour un Worker chargé en http(s),
    c'est cette CSP qui s'applique, et non celle de la page (à confirmer dans
    Chromium, voir Open Questions).
*   **Then** :
    *   `initialize` réussit ;
    *   `infer` renvoie `{ policyLogits: Float32Array(82), value }` dans le même
        format qu'avant, avec des valeurs dans la tolérance de G3. Un test e2e
        le vérifie dans Chromium ;
    *   les règles de Moka (`playMove`, `selectHighestLegalMove`…) fonctionnent
        sans changement.

### CUJ 3 : Refuser des poids qui ne correspondent pas

*   **Given** : le `.bin` fourni a été réentraîné ou modifié. Son manifeste a
    pu être mis à jour en conséquence.
*   **When** : `initialize` est appelé, par le client de Moka ou par un autre
    émetteur.
*   **Then** :
    *   le Worker refuse avec un message qui nomme la cause : la longueur
        attendue, ou le sha256 attendu ;
    *   aucune inférence n'est possible ensuite ;
    *   le moteur ne renvoie jamais de résultat calculé sur de mauvais poids.

### CUJ 4 : Se replier sur le Worker de Moka

*   **Given** : le navigateur n'a pas `simd128`, ou la CSP de `worker.js`
    bloque la compilation WASM.
*   **When** : `initialize` est appelé.
*   **Then** : la promesse est rejetée avec un message qui nomme la cause.
    L'intégrateur attrape ce rejet et crée le Worker de Moka à la place. L'API
    étant la même, le reste de la page ne change pas. Le README du projet
    montre ce code.

--------------------------------------------------------------------------------

## Technical Plan (Architecture)

### Vue d'ensemble

```text
 Thread principal (page)                         Web Worker (ce projet)
 ─────────────────────────                       ─────────────────────────────
 code de la page                                 worker.js
   │  règles, encodage : code de Moka            │  au chargement : télécharge
   │  (inchangé)                                 │     et compile infer.wasm
   ▼                                             │  "initialize" : contrôles,
 GoModelWorkerClient de Moka (inchangé)          │     copie du .bin dans WASM
   fetch manifeste + .bin, sha256                │  "infer" : copie des plans,
   infer(plans) ─────────── postMessage ───────► ▼     infer(1), copie des logits
             ◄────────────── postMessage ──────  infer.wasm (simd128)
                                                   poids f32 statiques,
                                                   tampons d'activation statiques
```

Le projet ne fournit que la partie droite du schéma. Le code JS se limite à
charger `infer.wasm`, à copier les octets à la frontière et à répondre aux
messages de Moka.

Le Worker de Moka (`src/worker.ts`) fonctionne ainsi :

*   à `initialize`, il crée un `GoModelRuntime` à partir du manifeste et du
    `.bin` reçus, puis répond `ready` ;
*   à `infer`, il reçoit un `Float32Array` de 972 valeurs, lance la passe avant
    et répond `result` en transférant les logits ;
*   toute exception devient un message `error`.

### Architecture du réseau

Le réseau est décrit par `model/go-model.json`. Les noms des tenseurs sont
ceux du manifeste.

```text
entrée      81 intersections × 12 plans                       (972 valeurs)
stem        conv 3×3 12→32 + ReLU                             stem.*
12 blocs (numérotés 0 à 11), chacun :
  reduce    conv 1×1 32→16 + ReLU
  first     conv 3×3 16→16 + ReLU
  [blocs 3, 7, 11] pooling global :
            moyenne et max de chaque canal sur 81 intersections (32 valeurs)
            → linéaire 32→8 + ReLU → linéaire 8→16
            → les 16 valeurs sont ajoutées comme biais à chaque intersection
              (pas de ReLU après cet ajout)
  second    conv 3×3 16→16 + ReLU
  expand    conv 1×1 16→32, ajouté au tronc, + ReLU
tête politique  conv 1×1 32→4 + ReLU → aplati (324) → linéaire 324→82
                (82 logits : coups 0 à 80, puis la passe en 81)
tête valeur     conv 1×1 32→2 + ReLU → aplati (162) → linéaire 162→32 + ReLU
                → linéaire 32→1 → tanh
```

**Plans d'entrée.** Ils sont produits par `encodeStudentFeatures` de Moka.
Le mot « Student » vient de ce que Moka a été distillé depuis KataGo. Les
plans sont rangés par intersection : la valeur du plan `p` à l'intersection
`i` est à l'indice `i × 12 + p` (`src/game.ts:215`).

**Format.** Les poids des convolutions et des couches linéaires sont en
`int8`, avec une échelle f32 par canal de sortie. Les biais sont en f32. Les
poids des convolutions sont rangés `[out][kh][kw][in]`. Les blocs sont de
type `nested-bottleneck`.

Moka accepte aussi des formats que ce moteur refuse :

*   `int4` : deux poids par octet ;
*   les échelles groupées (`quantizationGroupSize`) : une échelle par groupe
    de poids au lieu d'une par canal ;
*   les blocs « standard » : deux convolutions 3×3 sans goulot.

### Paramétrage

*   **`Geometry(comptime N)`** décrit le plateau :
    *   `AREA = N*N` et `SENTINEL = AREA` ;
    *   une table qui donne, pour chaque intersection, ses 9 voisins 3×3 dans
        l'ordre ligne par ligne, avec `SENTINEL` pour un voisin hors plateau.

    Avec les noyaux de convolution, c'est la seule partie paramétrée par N.
*   **`ModelConfig`** décrit le réseau. Il est généré au build à partir de
    `go-model.json` : canaux, blocs, intervalle du pooling global, décalages,
    formes et types des tenseurs, taille du `.bin` et sha256. Il n'est jamais
    écrit à la main.

### `infer.wasm` : mémoire

Toute la mémoire est faite de variables globales statiques, dont la taille est
connue à la compilation. Il n'y a pas d'allocateur. Les tailles sont données
en octets (1 f32 = 4 octets).

| Tampon | Taille (octets) | Rôle |
| --- | ---: | --- |
| `weights_bin` | 113 648 | copie brute du `.bin`, écrite par JS |
| `weights_f32` | 421 412 | 105 353 paramètres déquantifiés, ré-agencés pour SIMD |
| `reorder_tmp` | 13 824 | copie d'un tenseur de convolution pendant le ré-agencement (le plus gros, `stem.weight`, a 3 456 valeurs) |
| `input` | 31 488 | 8 positions × 82 lignes × 12 plans f32 (ligne 81 = sentinelle nulle) |
| `trunk` | 10 496 | 82 × 32 f32 (ligne 81 toujours nulle) |
| `bottleneck_a`, `bottleneck_b` | 5 248 chacun | 82 × 16 f32 (ligne 81 toujours nulle) |
| `pool_features`, `pool_hidden`, `pool_bias` | 128 + 32 + 64 | pooling global |
| `policy_features`, `value_features`, `value_hidden` | 1 296 + 648 + 128 | têtes |
| `policy_out` | 2 624 | 8 × 82 logits f32 |
| `value_out` | 64 | 8 valeurs f64 |

Toutes ces tailles découlent de `ModelConfig`, sauf la capacité de 8
positions, qui est un choix : elle coûte environ 30 Ko et servira à une future
recherche (les 8 symétries d'une position). Le Worker appelle toujours
`infer(1)`. La taille de la mémoire linéaire et celle de la pile sont des
questions ouvertes.

### `infer.wasm` : chargement des poids

1.  **Au chargement de `worker.js`**, avant tout message, le Worker télécharge
    `infer.wasm` (`fetch`, puis `arrayBuffer`) et le compile
    (`WebAssembly.compile`). Le téléchargement se fait ainsi en parallèle de
    ceux du client, et aucun type MIME particulier n'est exigé.
2.  **À `initialize`**, le Worker fait ses contrôles dans cet ordre. Le plan
    d'implémentation renvoie à cette liste au lieu de la répéter. Les messages
    qui distinguent SIMD, CSP et téléchargement arrivent en phase 2 ; en
    phase 1, tout échec de compilation donne « Failed to load infer.wasm. ».
    1.  **Compilation** : le résultat de l'étape 1 est attendu.
    2.  **Cause d'un échec** : sans SIMD et sous une CSP qui bloque, la
        compilation lève la même `CompileError`. Le Worker distingue donc les
        cas ainsi :
        *   si le `fetch` a échoué, c'est l'erreur de téléchargement ;
        *   sinon, si `WebAssembly.validate` sur les octets d'`infer.wasm`
            renvoie `false`, le navigateur n'a pas `simd128` ;
        *   sinon, la compilation a été bloquée par la CSP, qui ne contrôle
            pas `validate`.
    3.  **Instanciation.**
    4.  **Longueur** : la taille du `.bin` doit être égale à
        `weights_capacity()`. Ce contrôle protège la mémoire avant la copie.
    5.  **Hash** : le Worker calcule le sha256 des octets reçus
        (`crypto.subtle.digest`, comme `src/utils/get-buffer-sha256.ts` de
        Moka) et le compare aux 32 octets de `expected_sha256_ptr()`, figés au
        build. Le champ `manifest.sha256` n'est pas lu.
3.  Le Worker copie le `.bin` dans `weights_bin` et appelle `load_weights()`.
    Le contrôle de longueur de l'étape 2.4 est le seul : Zig lit toujours
    `weights_capacity()` octets.
4.  Zig déquantifie chaque tenseur entier : `w = f32(q) * scale[out]`. Ce
    calcul donne le même résultat bit à bit que Moka (preuve en annexe A1). Les
    biais f32 sont recopiés tels quels.
5.  Zig ré-agence les poids des convolutions pour vectoriser sur les canaux de
    sortie :
    *   3×3 : `[out][kh][kw][in]` devient `[kh·kw][in][out]` ;
    *   1×1 : `[out][in]` devient `[in][out]`.

    Les couches linéaires gardent la disposition `[out][in]`. Le
    ré-agencement se fait en place dans `weights_f32`, tenseur par tenseur :
    chaque tenseur est copié dans `reorder_tmp`, puis réécrit à sa place dans
    la nouvelle disposition. Un second tampon complet de 421 Ko ne tiendrait
    pas dans la mémoire fixe.

### `infer.wasm` : calcul et ordre des opérations

Le design repose sur un principe : **le noyau SIMD fait exactement les mêmes
opérations f32, dans le même ordre, qu'une version scalaire de référence.**
Seule la précision des accumulateurs diffère de Moka, qui accumule en f64.
Grâce à ce principe, on peut mesurer la dérive numérique et localiser un bug.

*   **Convolutions.** Pour chaque intersection, l'accumulateur de chaque canal
    de sortie part du biais. On parcourt ensuite les 9 voisins dans l'ordre
    ligne par ligne de Moka, puis les canaux d'entrée dans l'ordre, avec
    `acc = acc + x * w` en f32.
    *   Le SIMD porte sur les canaux de sortie : 4 vecteurs `f32x4` pour 16
        canaux, 8 pour 32. La valeur d'entrée `x` est diffusée dans un vecteur,
        multipliée par un vecteur de poids, puis ajoutée.
    *   Chaque voie calcule donc la même suite d'opérations que la version
        scalaire. Il n'y a ni somme horizontale ni FMA, puisque le `simd128` de
        base n'a pas de FMA. `relaxed-simd` n'est pas utilisé.
    *   Un voisin hors plateau lit la ligne sentinelle nulle. Le résultat est
        identique au saut des bords dans Moka (annexe A2).
*   **ReLU.** Elle reproduit `Math.max(0, x)` de Moka (`src/runtime.ts:46`) :
    *   elle renvoie `x` si `x > 0` ;
    *   elle renvoie `NaN` si `x` est `NaN`, sans le masquer ;
    *   sinon, elle renvoie `+0`.

    Elle s'écrit avec des comparaisons explicites, et non avec `@max` de Zig,
    qui renvoie l'autre opérande quand l'un vaut `NaN`. Il en va de même pour
    le maximum du pooling global.
*   **Bloc résiduel.** Il suit `src/runtime.ts:303-380` et le schéma de
    « Architecture du réseau » :
    *   `reduce` (1×1, ReLU) lit `trunk` et écrit `a` ;
    *   `first` (3×3, ReLU) écrit `b` ;
    *   le pooling global ajoute son biais à `b`, sans ReLU ensuite ;
    *   `second` (3×3) écrit `a`. Moka applique la ReLU juste avant `expand`
        (`src/runtime.ts:368-370`) ; Zig l'applique au stockage, ce qui donne
        les mêmes valeurs puisque `a` n'est lu que par `expand` ;
    *   `expand` (1×1) calcule `acc = biais + Σ x·w` en f32, puis écrit
        `trunk = relu(trunk + acc)`, en place, comme `addAndApplyRelu`
        (`src/runtime.ts:52-58`, `:380`).

    Le calcul en place est sûr parce que `expand` ne lit que l'intersection
    qu'il écrit. Toutes les opérations élément par élément s'arrêtent à
    `AREA`, et la ligne sentinelle n'est jamais écrite.
*   **Pooling global et couches linéaires.** Ils se calculent en scalaire f64,
    avec les mêmes arrondis que Moka (annexe A3). Ils comptent pour moins de
    1 % des multiplications d'une inférence. Le f64 ne coûte donc rien de
    mesurable.
*   **Têtes.** Elles suivent le schéma de « Architecture du réseau ». Les
    sorties de la conv 1×1 sont aplaties par intersection (`i·4 + c`). `tanh`
    se calcule en f64, et la valeur sort en f64, comme le `number` JS de Moka.
*   **Contrôle des sorties.** Après chaque position, Zig vérifie que les 82
    logits et la valeur sont finis. Sinon, `infer` renvoie un code d'erreur.
    C'est ainsi qu'une entrée contenant un `NaN` est détectée.

### `infer.wasm` : fonctions exportées (ABI C)

```text
memory                          // mémoire linéaire exportée (export_memory)
weights_ptr() -> [*]u8          weights_capacity() -> u32   // 113 648
expected_sha256_ptr() -> [*]const u8                        // 32 octets, figés au build
load_weights() -> void           // lit weights_capacity() octets
input_ptr() -> [*]f32           policy_ptr() -> [*]f32      value_ptr() -> [*]f64
max_batch() -> u32              // 8
infer(count: u32) -> Status     // 1 ≤ count ≤ max_batch()

Status (u32) : ok = 0, weights_not_loaded = 1, bad_batch_count = 2,
               non_finite_output = 3
```

*   **Entrées.** La position `k` occupe 82 × 12 = 984 f32 à partir de
    `input_ptr() + k × 984`. Le JS y copie les 972 valeurs de Moka telles
    quelles. Les 12 dernières valeurs forment la ligne sentinelle : elles
    restent nulles et ne sont jamais écrites.
*   **Sorties.** Les sorties de la position `k` sont `policy_ptr() + k × 82`
    et `value_ptr() + k`.

Si le diagnostic bit à bit révèle une divergence, et seulement dans ce cas, un
build de débogage (`-Dlayer-dump=true`) ajoute `layer_dump_ptr()`. Cet export donne
accès aux activations de chaque couche pour la première position (voir
« Outils de diagnostic » dans le plan). L'artefact publié ne le contient
jamais.

### Protocole du Worker

Le Worker utilise les messages de Moka (`src/types.ts:62-91`). Il reçoit
`initialize` et `infer`, et il émet `ready`, `result` et `error`. Il garde
les mêmes champs et les mêmes transferts de tampons.

Les logits sont copiés de la mémoire WASM vers un nouveau `Float32Array`
avant d'être transférés, comme le fait déjà Moka.

Les messages sont traités dans l'ordre d'arrivée. Un `infer` reçu pendant un
`initialize` attend la fin de celui-ci.

**`initialize` remplace toujours l'instance.** Au début de chaque
`initialize`, le Worker abandonne l'instance précédente, avant tout contrôle.
Si un contrôle échoue, aucun poids n'est actif, et un `infer` reçoit
« Go model worker is not initialized. » (CUJ 3). Moka, lui, garde l'ancien
runtime quand un `initialize` échoue (`src/worker.ts:12`) ; ce Worker s'en
écarte volontairement, pour qu'aucun `infer` ne tourne avec des poids qui ne
sont pas ceux qu'on vient d'envoyer.

**Erreurs.** Le comportement diffère de celui de Moka dans un cas : quand une
entrée invalide donne des sorties `NaN`, Moka les renvoie sans erreur, alors
que ce Worker renvoie une erreur. Les nouveaux messages sont en anglais,
comme ceux de Moka :

| Cause | Message |
| --- | --- |
| pas de `simd128` | « WebAssembly SIMD is not supported in this browser. » |
| compilation bloquée | « WebAssembly compilation was blocked; the Content Security Policy of worker.js must allow 'wasm-unsafe-eval'. » |
| téléchargement impossible | « Failed to load infer.wasm. » |
| longueur des poids | « Model weights have the wrong length (expected 113648 bytes). » |
| sha256 des poids | « Model weights do not match this engine build (expected sha256 35c1e51c…). » |
| longueur de l'entrée | « Input features must have 972 values. » |
| sortie non finie | « Model produced a non-finite output. » |
| `infer` sans `initialize` réussi | « Go model worker is not initialized. » (message de Moka) |
| autre `Status` (`weights_not_loaded`, `bad_batch_count`) | « Go model inference failed. » (message de repli de Moka) |

Les deux derniers `Status` ne se produisent pas si le Worker est correct : il
appelle toujours `infer(1)`, et seulement après `load_weights()`. Ils restent
testés directement sur le `.wasm` (fichier 28).

Si WASM s'arrête sur un `trap`, quelle qu'en soit la cause (un `@panic`, un
accès hors de la mémoire linéaire), l'état de l'instance n'est plus fiable. Le
Worker considère alors le moteur comme hors service : il renvoie une erreur à
la requête en cours et à toutes les suivantes, jusqu'à un nouvel `initialize`,
qui crée une instance neuve.

Ces erreurs passent par le message `error` de Moka, et le client de Moka les
transforme en promesse rejetée (CUJ 4).

### Build

`build.zig` définit les étapes suivantes.

*   **`zig build gen`** exécute un outil Zig côté hôte, qui a le droit
    d'allouer.
    *   Il lit `vendor/moka/model/go-model.json`.
    *   Il accepte les tenseurs `int8` et `float32` avec une échelle par canal
        de sortie, et les tenseurs `int16` à partir de S4.
    *   Il refuse tous les autres formats et versions.
    *   Il recalcule le sha256 du `.bin`, et le build échoue s'il ne correspond
        pas au manifeste.
    *   Il écrit `model_layout.zig` (`ModelConfig`).
*   **`zig build test`** lance les tests natifs.
*   **`zig build wasm`** produit `infer.wasm` : `wasm32-freestanding` avec
    `simd128`, en `ReleaseFast`, avec une mémoire de taille fixe.
*   **En phase 2, `zig build wasm-safe`** produit `infer-safe.wasm`, en
    `ReleaseSafe`. En `ReleaseFast`, un accès hors bornes est un comportement
    indéfini, et non un `trap`. La parité est donc aussi vérifiée sur ce
    build.
*   **En cas de divergence seulement, `zig build wasm-debug`** produit
    `infer-debug.wasm`, avec l'export de vidage.
*   **Contrôles de CI (phase 2) :**
    *   le formatage ;
    *   les tests natifs ;
    *   le build ;
    *   la présence d'instructions `v128` ;
    *   une mémoire avec `initial == maximum` ;
    *   le plafond de taille ;
    *   la parité, sur `infer.wasm` et sur `infer-safe.wasm`.

Le code JS (Worker, banc, oracle) est en TypeScript, empaqueté avec esbuild.
Les dépendances sont gérées avec pnpm.

### Oracle et tolérance (S0)

*   **Copie de Moka.** Moka est copié dans `vendor/moka/` au commit
    `a418039e`, avec sa licence.
*   **Version instrumentée de `runtime.ts`.** Elle se trouve dans `oracle/`.
    C'est une copie, et non une modification de `vendor/` : elle est d'abord
    versionnée sans changement, puis modifiée dans un commit à part, si bien
    que git montre le diff. Elle ajoute l'option `f32Accumulate`, qui arrondit
    en f32 après chaque produit et chaque somme des convolutions
    (`Math.fround`). L'option `dumpLayers` ne s'ajoute qu'en cas de
    divergence.
*   **Deux jeux de données de référence**, produits par un générateur Node
    avec les règles de Moka.
    *   **Calibration** : 50 parties aléatoires à graine fixe.
    *   **Validation** : 20 parties de Moka contre elle-même. Les 6 premiers
        coups sont tirés au hasard, puis Moka joue `selectHighestLegalMove`.
        S'y ajoutent les sondes fixes : la partie du journal où Noir joue 20,
        24, 56 et 60 (`experiment-log.md:1561`), et la position de
        `tests/model-smoke.mjs`. Les réponses de Moka sont recalculées avec
        `go-model.bin`, car celles du journal viennent du modèle du site.
    *   Pour chaque position, le générateur enregistre les plans, les logits
        et la valeur, en mode normal et en mode `f32Accumulate`, ainsi que les
        coups légaux, le nombre de coups joués et le coup choisi par Moka. Il
        enregistre aussi une fois les tenseurs déquantifiés par Moka.
    *   **Format** : fichiers binaires little-endian, avec un index JSON. Ce
        format conserve les valeurs bit à bit et prend environ 4 fois moins de
        place que du texte.
    *   **Déterminisme (G2)** : `pnpm oracle:check` génère les données une
        seconde fois dans un dossier temporaire et les compare octet pour
        octet. Il vérifie aussi les sha256 de `vendor/moka/VENDORED.md`.
*   **Ce que vérifie l'oracle :**
    *   **Sur les deux jeux**, Zig et Moka en mode `f32Accumulate` sont
        identiques bit à bit sur les logits. Ce test distingue un bug d'une
        dérive numérique. **C'est un diagnostic, pas une condition de G3 ni de
        G4** : un échec ne bloque pas le verdict, mais il doit être expliqué
        avant la phase 2, avec les outils de diagnostic si besoin.
    *   **Pour la valeur**, un écart de 2 ulp est accepté. `tanh` n'a pas la
        même implémentation en Zig et dans V8, et chacune peut se tromper de
        moins de 1 ulp.
    *   **Tolérance de G3.** L'écart entre Moka normal et Moka `f32Accumulate`
        est mesuré sur le jeu de calibration. La tolérance de G3 vaut 4 fois
        l'écart maximal observé.
    *   **Plafonds.** Si cette tolérance dépasse 2e-3 sur les logits ou 1e-3
        sur la valeur, on ne relève pas le seuil : on arrête le projet et on
        revient à la conception.
    *   **G3 se vérifie sur le jeu de validation.** Ce test montre que la
        dérive f32 mesurée sur des parties aléatoires vaut aussi pour des
        parties jouées par Moka.

### Banc de mesure (S0, S3)

*   **La page `bench/`** est reprise de `web/benchmark.ts` de Moka et corrigée :
    *   elle utilise 100 positions de 972 valeurs, tirées du jeu de validation ;
    *   avant de chronométrer, elle compare les 100 sorties du moteur à celles
        de Moka en mode normal, avec la tolérance de G3. Un écart fait échouer
        la mesure : un moteur faux ne peut pas gagner G4 ;
    *   une sortie `NaN` fait échouer la mesure.
*   **Paramètres d'URL.** Les deux moteurs passent par le même client, celui de
    Moka. Le moteur se choisit avec `runtime=moka|zig`, et l'usage du Worker
    avec `worker=1|0`. Avec `worker=0`, le moteur est appelé directement sur
    le thread principal : `GoModelRuntime.create` pour Moka, une instance
    `infer.wasm` pour Zig.
*   **Serveur.** Le serveur local envoie les en-têtes COOP/COEP. Sans
    isolation cross-origin, Chromium arrondit `performance.now()` à environ
    100 µs, soit 10 % d'une mesure de 1 ms.
*   **Le script Playwright `bench/run.mjs`** lance Chromium sans affichage
    (headless), avec le même binaire pour tous les moteurs.
    *   Il fait 5 tours. Chaque tour mesure les deux moteurs dans un ordre
        alterné (Moka puis Zig, puis Zig puis Moka au tour suivant).
    *   Chaque mesure comprend 10 inférences de chauffe, enregistrées à titre
        informatif, puis 100 inférences mesurées. En mode `worker=1`, elle
        ajoute 5 rafales de 35 appels successifs, à titre informatif.
    *   Le verdict G4 se calcule sur la médiane des p50 des 5 tours, en mode
        `worker=1`.

## Alternatives Considered

### Approche générale

#### Alternative A : optimiser le runtime TypeScript de Moka sur place

*   **Pros** : deux des trois limites du code actuel se corrigent en
    TypeScript : tampons préalloués, table de voisins avec ligne sentinelle, et
    poids ré-agencés. Le runtime resterait d'environ 4 Kio, et le site en
    profiterait directement, puisque `web/runtime.ts` est une copie de
    `src/runtime.ts`. Le changement pourrait aussi être proposé à Moka.
*   **Cons** : JavaScript n'a pas d'instructions SIMD. Ce moteur ne pourrait
    pas servir de base au modèle 19×19 en Zig.
*   **Reason for Rejection** : le projet impose Zig (voir Overview). Mesurer ce
    moteur à titre de témoin reste utile pour une éventuelle proposition à
    Moka (voir Future Improvements).

#### Alternative B : un autre langage pour le WASM SIMD (Rust, C)

*   **Pros** : Moka a déjà un crate Rust et sa chaîne de build
    (`wasm/Cargo.toml`). Rust (`core::arch::wasm32`) et C (clang) offrent les
    mêmes instructions `simd128`.
*   **Cons** : aucun avantage technique sur Zig pour ce moteur. Le `comptime`
    de Zig génère directement `ModelConfig` et `Geometry(N)`.
*   **Reason for Rejection** : le projet impose Zig. Une proposition à Moka
    devra justifier l'ajout d'une troisième chaîne d'outils (TS, Python, Zig),
    ou bien porter le moteur en Rust.

#### Alternative C : un runtime généraliste (`onnxruntime-web`)

*   **Pros** : déjà une dépendance de Moka (`package.json:46`), et un backend
    SIMD maintenu par d'autres.
*   **Cons** : le runtime pèse plusieurs Mo, contre environ 4 Kio pour celui de
    Moka. Il alloue, et il ne permet de contrôler ni l'ordre des opérations ni
    la parité.
*   **Reason for Rejection** : incompatible avec G5, et la taille est hors de
    proportion pour un modèle de 114 Ko.

### Précision et calcul

#### Alternative D : activations quantifiées (INT8 ou INT16) dès la v1

*   **Pros** :
    *   en INT8 : produit scalaire entier, quatre fois moins d'octets par
        activation, et la vitesse la plus élevée en théorie ;
    *   en INT16 : moitié moins de mémoire que le f32, et le produit
        `i32x4.dot_i16x8_s` existe dans le `simd128` de base.
*   **Cons** : Moka ne fournit pas d'échelles d'activation, il faudrait donc
    une étape de calibration. Le `simd128` de base n'a pas de produit scalaire
    INT8. Ce produit n'existe que dans `relaxed-simd`, dont les résultats
    peuvent varier d'un matériel à l'autre.
*   **Reason for Rejection** : les sorties ne seraient plus comparables bit à
    bit, et l'oracle ne pourrait plus distinguer un bug du bruit de
    quantification. Reporté en v2.

#### Alternative E : garder les poids en INT8 et les déquantifier dans la boucle interne

*   **Pros** : environ 114 Ko de poids en mémoire au lieu de 421 Ko.
*   **Cons** : le `simd128` de base n'a pas de produit INT8 × f32. La boucle
    interne demanderait donc plus d'instructions. Le calcul `Σ(q·x)·scale` ne
    donne pas les mêmes arrondis que celui de Moka, qui calcule
    `Σ(q·scale)·x`.
*   **Reason for Rejection** : la parité bit à bit serait perdue, et 421 Ko
    tiennent dans la mémoire statique.

#### Alternative F : suivre l'ordre de calcul de Moka (accumulation f64, ou vectorisation sur les canaux d'entrée)

*   **Pros** :
    *   accumuler en f64 donnerait des sorties presque identiques à celles de
        Moka ;
    *   vectoriser sur les canaux d'entrée garderait la disposition de Moka
        (`[out][kh][kw][in]`), sans ré-agencement des poids au chargement.
*   **Cons** :
    *   `f64x2` traite deux valeurs par instruction au lieu de quatre ;
    *   vectoriser sur les canaux d'entrée demande une somme horizontale par
        sortie, qui change l'ordre des additions par rapport à la version
        scalaire.
*   **Reason for Rejection** : le f64 perdrait une grande partie du gain visé
    par G4 (2 fois). La somme horizontale ferait perdre l'égalité bit à bit
    avec le mode `f32Accumulate`, qui sert de diagnostic. On mesure plutôt la
    dérive f32 avec ce mode.

#### Alternative G : tampon bordé (N+2)² au lieu d'une table de voisins

*   **Pros** : on obtient les voisins par un décalage d'adresse constant.
*   **Cons** : les activations ne sont plus rangées comme dans Moka (81
    intersections consécutives). Le vidage des couches et la lecture des têtes
    deviennent plus compliqués.
*   **Reason for Rejection** : la table de voisins avec une ligne sentinelle
    donne le même calcul sans test de bord, et garde la disposition de Moka.

### Paramétrage et poids

#### Alternative H : `comptime N` aussi pour les têtes

*   **Pros** : tout le réseau semble prêt pour le 19×19.
*   **Cons** : les têtes dépendent de N² (`[82, 324]` = `[N²+1, 4N²]`). Une
    tête de politique 19×19 de même structure pèserait environ 5 fois le
    modèle entier. Aucun poids 19×19 n'existe pour tester ce code.
*   **Reason for Rejection** : un modèle 19×19 aura presque sûrement d'autres
    têtes. On ne paramètre que `Geometry(N)` et les noyaux.

#### Alternative I : un autre traitement du `.bin` et du manifeste

Quatre variantes ont été écartées :

*   **Script Python qui convertit le `.bin` vers un format prêt pour SIMD.**
    Le sha256 vérifié par le client ne correspondrait plus au fichier
    d'origine, il faudrait tenir deux fichiers synchronisés, et Python
    entrerait dans la chaîne de build. Le ré-agencement au chargement ne coûte
    qu'une passe sur 105 000 valeurs.
*   **Lecture de `go-model.json` dans Zig à l'exécution.** Il faudrait un
    analyseur JSON sans allocation. Avec des tampons statiques (G5), une autre
    architecture demande de toute façon un nouveau build.
*   **Poids intégrés au `.wasm` (`@embedFile`).** `infer.wasm` grossirait
    d'environ 114 Ko, et le contrôle sha256 du client de Moka ne
    s'appliquerait plus. L'intégrateur sert déjà le `.bin` (CUJ 2).
*   **Comparaison du manifeste au moteur champ par champ.** Elle donnerait un
    message plus précis, mais demanderait un contrat de plus, et refuserait un
    manifeste dont seuls les champs descriptifs diffèrent. Le hash des octets
    suffit à prouver la compatibilité, et le message cite le hash attendu.

### Périmètre et intégration

#### Alternative J : porter les règles, l'encodage et le décompte des points dans un `rules.wasm` sur le thread principal

*   **Pros** : une plus grande part du moteur serait en Zig, et les règles
    deviendraient synchrones après l'initialisation.
*   **Cons** : aucun effet sur la latence de `infer`, la seule grandeur de G4.
    Il faut en plus une initialisation asynchrone (`await initRules()`), un
    second artefact, une ABI d'état, et reproduire exactement environ 500
    lignes de `src/game.ts`.
*   **Reason for Rejection** : le seul usage qui justifierait ce portage est
    une recherche dans le Worker. Elle aurait besoin des règles dans
    `infer.wasm`, et non sur le thread principal. Ce sera un design séparé.

#### Alternative K : porter les règles dans le Worker, avec `infer.wasm`

*   **Pros** : les règles seraient déjà au bon endroit pour une future
    recherche.
*   **Cons** : les fonctions de règles deviendraient asynchrones et
    attendraient derrière l'inférence. L'interface paraîtrait figée pendant le
    calcul.
*   **Reason for Rejection** : cela rompt la compatibilité avec l'API de Moka,
    alors que rien ne l'utilise en v1. À reprendre dans le design de la
    recherche.

#### Alternative L : protocole de lots `inferBatch` dès la v1

*   **Pros** : un seul message pour plusieurs positions.
*   **Cons** : rien ne l'appelle en v1. La recherche de Moka évalue une feuille
    à la fois (`experiment-log.md:1678`, `:1979`). Un lot ne servirait qu'aux
    8 symétries de la racine, pour un gain d'environ 2 ms par coup.
*   **Reason for Rejection** : le contrat public s'élargirait sans usage. L'ABI
    Zig garde `infer(count)`, et le protocole JS viendra avec la recherche,
    aligné sur celui du site de Moka.

#### Alternative M : viser le modèle INT8/INT16 du site dès la v1

*   **Pros** : c'est l'artefact que les joueurs utilisent.
*   **Cons** : ni ses poids ni son Worker ne sont dans le dépôt de Moka, et
    leur licence n'est pas connue. `runtime.ts` n'accepte que les formats
    `int4` et `int8` (`src/runtime.ts:37`), donc il n'existe aucune référence
    pour vérifier la parité.
*   **Reason for Rejection** : impossible à vérifier aujourd'hui. Préparé par
    G7.

#### Alternative N : un repli sans `simd128` dans ce projet

*   **Pros** : le Worker fonctionnerait sur tous les navigateurs. Deux formes
    sont possibles : intégrer le moteur TypeScript de Moka au Worker, ou
    publier un second build Zig scalaire, identique bit à bit.
*   **Cons** : le code TS de Moka deviendrait une partie du produit, avec deux
    comportements numériques dans le même Worker, ou bien il faudrait
    construire et tester un second artefact. Rien n'indique qu'un build
    scalaire serait plus rapide que le JS : l'essai `wasm/src/lib.rs` ne l'a
    jamais été.
*   **Reason for Rejection** : comme l'API est la même, l'intégrateur se replie
    lui-même sur le Worker de Moka (CUJ 4).

#### Alternative O : forker Moka au lieu d'un projet séparé

*   **Pros** : le travail correspond à une expérience que Moka prévoit
    lui-même, « un chemin WASM fusionné convolution/résidu »
    (`experiment-log.md:204`).
*   **Cons** : le projet dépendrait de l'outillage de Moka (`vite-plus` en
    version `latest`, et sa commande d'empaquetage `vp pack`) et de son rythme
    de changement.
*   **Reason for Rejection** : un dépôt séparé, avec Moka copié à un commit
    figé, garde le build reproductible. Le travail pourra être proposé à Moka
    plus tard.

#### Alternative P : réutiliser `wasm/src/lib.rs` de Moka

*   **Pros** : du code existant.
*   **Cons** : il alloue (`Vec`), il est scalaire, il n'a pas de convolution,
    et il est appelé une fois par couche.
*   **Reason for Rejection** : il n'implémente pas le réseau. On garde
    seulement l'hypothèse du journal : le réseau entier doit passer en un seul
    appel exporté.

### Mesure

#### Alternative Q : un autre critère de vitesse pour G4

Trois critères ont été écartés au profit de l'aller-retour `infer` isolé :

*   **Une rafale de 35 appels successifs**, soit un coup du site. Elle se
    mesure à conditions égales, puisque chaque appel reste un message. Elle
    est mesurée à titre informatif. Elle n'est pas le critère : G4 reprend la
    question posée par la v1, la vitesse d'une inférence. La rafale ne fait
    qu'estimer un coup du site, qui utilise un autre modèle et un autre
    Worker.
*   **Le calcul seul, sans Worker.** Il isole la vitesse du noyau, mais ne
    garantit aucun gain pour l'intégrateur, qui passe toujours par le Worker.
    Il reste une mesure informative.
*   **Les 8,8 ms du README.** Ce chiffre a été mesuré sur une entrée invalide
    (810 valeurs au lieu de 972, sorties `NaN`) et sur une autre machine.

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Performance.** Le seuil G4 décide de la poursuite ou de l'arrêt du
    projet. Le temps d'initialisation (déquantification et ré-agencement
    d'environ 105 000 valeurs) est une mesure informative (G4).
*   **Mémoire.** La taille est fixe, avec un plafond provisoire de 1 Mio (voir
    Open Questions). La mémoire ne peut pas grandir, et un test le vérifie.
*   **Taille de l'artefact.** `infer.wasm` doit rester sous un plafond, mesuré
    après gzip. La valeur proposée est 32 Kio (voir Open Questions). Un test de
    CI le vérifie en phase 2.
*   **Sécurité et intégrité.**
    *   Le moteur refuse tout poids dont la longueur ou le sha256 des octets ne
        correspond pas à son build, quel que soit l'émetteur.
    *   Il refuse toute entrée qui n'a pas 972 valeurs.
    *   Il ne lit aucune donnée hors des tampons dont la taille est contrôlée.
*   **Compatibilité.**
    *   Le navigateur doit gérer `simd128` (à partir de Chrome 91, Firefox 89
        et Safari 16.4).
    *   Si une CSP s'applique à `worker.js` (en-têtes de sa réponse), elle
        doit autoriser `'wasm-unsafe-eval'` dans `script-src`.
    *   `worker.js` et `infer.wasm` sont servis depuis l'origine de la page.
    *   Le moteur est mesuré et testé dans Chromium seulement.
    *   Sinon, `initialize` est refusé, et l'intégrateur se replie sur le
        Worker de Moka (CUJ 4).
*   **Retour arrière.** Il suffit de remettre l'URL du Worker de Moka.

### Risques et parades

| Risque | Effet | Parade | Où c'est traité |
| --- | --- | --- | --- |
| Le seuil 2× est hors d'atteinte. | Le projet s'arrête. | La phase 1 ne contient que le minimum nécessaire pour mesurer. | G4, phase 1 |
| La dérive f32 dépasse les plafonds. | La parité n'a plus de sens. | Arrêt en S0, sans relever le seuil. | Oracle et tolérance |
| Moka publie de nouveaux poids. | `initialize` est refusé avec le hash attendu. | Nouveau build (`zig build gen`), expliqué dans le README. | CUJ 3, README |
| Un accès hors bornes en `ReleaseFast` est un comportement indéfini. | Corruption silencieuse. | La parité est aussi vérifiée sur un build `ReleaseSafe`, et l'invariant de la ligne sentinelle est testé. | Build, phase 2 |
| Une CSP bloque le WASM. | `initialize` est refusé. | Message dédié, README, repli sur le Worker de Moka. | CUJ 2, CUJ 4 |
| Une entrée est invalide (longueur ou `NaN`). | Résultat faux ou `NaN`. | Contrôle de longueur et contrôle des sorties. | Protocole du Worker |
| La première inférence est lente : V8 compile d'abord le WASM avec son compilateur de base, puis l'optimise. | Le premier coup est plus lent que la mesure à chaud. | Latence des 10 premières inférences publiée à titre informatif. | Banc de mesure |
| Les autres navigateurs se comportent autrement. | Gain non garanti hors de Chromium. | Écrit dans les NFR. | NFR Compatibilité |

--------------------------------------------------------------------------------

### Additional Resources

*   Moka : [github.com/millionco/moka](https://github.com/millionco/moka), au
    commit `a418039e9d823779e3e6d0ea17a244230abc8379`.
    *   `README.md` ;
    *   `model/go-model.json` ;
    *   `src/runtime.ts`, `src/types.ts`, `src/client.ts`, `src/worker.ts`,
        `src/game.ts`, `src/utils/get-buffer-sha256.ts` ;
    *   `web/benchmark.ts`, `web/constants.ts` ;
    *   `wasm/src/lib.rs` ;
    *   `experiment-log.md` : coût du Worker et essai WASM (lignes 144-205),
        modèle et recherche du site (lignes 1559-1783 et 1979).

### Future Improvements

*   **Modèle mixte INT8/INT16 du site de Moka.** L'outil de build acceptera ses
    tenseurs (G7). Il manque trois préconditions :
    1.  obtenir les poids (138 616 octets, sha256 `4a58dcfd…`) avec une licence
        qui permet de les réutiliser ;
    2.  disposer d'une référence pour la parité : le Worker du site, ou une
        extension INT16 de l'oracle validée sur l'export Python de Moka ;
    3.  documenter le protocole de lots du site (`experiment-log.md:1674`).
*   **Recherche dans le Worker.** Un design séparé couvrira :
    *   les règles dans `infer.wasm` ;
    *   un PUCT avec un arbre de taille fixe ;
    *   un protocole de lots aligné sur celui du site.
*   **Témoin « TypeScript optimisé ».** Mesurer `runtime.ts` avec des tampons
    préalloués et une table de voisins, avant toute proposition à Moka
    (Alternative A).
*   **v2 en INT8.** Calibrer les échelles d'activation, puis évaluer un noyau
    INT16 ou INT8 sur son taux d'accord avec la v1.
*   **Têtes et modèle 19×19.** Concevoir des têtes indépendantes de N, par
    exemple par pooling global. Ajouter alors une référence scalaire des noyaux
    et leurs tests à N = 19.
*   **Mesure sur mobile et autres navigateurs.** Ajouter au banc un téléphone
    de milieu de gamme, Firefox et WebKit. Moka exige cette mesure sur mobile
    pour son propre WASM (`experiment-log.md:196`, `:205`).
*   **`relaxed-simd`.** Mesurer le gain du FMA, en acceptant de perdre l'égalité
    bit à bit.

### Appendix: Numerical Justifications

#### A1. La déquantification est identique bit à bit à celle de Moka

`q` est un entier d'au plus 16 bits, donc `f32(q)` est exact, puisque
|q| < 2^24. `scale` est un f32 (24 bits de mantisse). Le produit des
mantisses tient donc dans 40 bits, et il est exact en f64 (53 bits).

Moka calcule `q * scale` en f64 (`number` JS), puis arrondit une seule fois
en f32 en stockant le résultat dans un `Float32Array` (`src/runtime.ts:263`).
Une multiplication f32 en Zig donne directement l'arrondi correct du même
produit exact. Les deux résultats sont donc identiques.

#### A2. La ligne sentinelle donne le même résultat que le saut de Moka, au signe de zéro près

Un voisin hors plateau lit `x = 0` dans la ligne sentinelle. Le terme ajouté
`0 × w` vaut ±0, et `acc + (±0)` laisse `acc` inchangé, sauf éventuellement le
signe d'un `acc` nul. La ReLU renvoie toujours `+0` pour une entrée nulle,
comme `Math.max(0, x)` (`src/runtime.ts:46`). Le signe disparaît donc à la
première ReLU, et il est sans effet sur les produits suivants.

#### A3. Arrondis du pooling et des couches linéaires

*   **Moyenne.** Moka fait `pooled[c] += value / 81` sur un `Float32Array`
    (`src/runtime.ts:82`) : calcul en f64, puis arrondi en f32 à chaque pas.
    Zig reproduit `pooled = f32(f64(pooled) + f64(v) / 81)`.
*   **Maximum.** Comme dans Moka (`src/runtime.ts:83`), il part de `-inf`.
*   **Couches linéaires.** L'accumulateur f64 part du biais et ajoute des
    produits de deux f32, qui sont exacts en f64. Le résultat est stocké en
    f32, comme le fait `applyLinear` de Moka (`src/runtime.ts:145-165`).

--------------------------------------------------------------------------------

## Detailed Implementation Plan

Les chemins sont relatifs à la racine du dépôt (`./`). Les fichiers sont
groupés par phase puis par étape. Une étape ne commence que lorsque la
précédente est vérifiée. **La phase 2 ne commence que si le verdict G4 est
positif.** Les fichiers marqués *(généré)* sont produits par le build et
ignorés par git. Les numéros désignent les entrées de cette liste : une entrée
peut couvrir plusieurs fichiers, et « fichier 28 » se lit « entrée 28 ».
« La référence G1 » désigne `bench/results/baseline-moka.json`.

### Target Files to Create/Modify

#### Phase 1 : prototype et verdict

##### S0 : prérequis, oracle, mesure de référence

###### 1. `./mise.toml` (Create)

*   **Type / Scope** : configuration des outils.
*   **Proposed Changes** : fixe `node` et `pnpm` à des versions exactes, sans
    plage. `node` est en version majeure 24 (LTS active en 2026-10), et au
    moins 22.18 dans tous les cas : c'est la première version qui supprime
    les types TypeScript sans option. La cible esbuild de l'oracle suit la
    même version majeure (`target: node24`). La version exacte de `zig` est
    ajoutée au début de S1 (voir Open Questions) : c'est 0.17.0.

###### 2. `./.gitignore` (Create)

*   **Type / Scope** : configuration git.
*   **Proposed Changes** : ignore `zig-out/`, `.zig-cache/`, `node_modules/`,
    `dist/`, `bench/dist/`, `oracle/dist/`, `tests/e2e/dist/`,
    `oracle/fixtures/`, et `bench/results/*.json` sauf
    `bench/results/baseline-*.json`. Le dépôt est initialisé avec `git init`
    à cette étape.

###### 3. `./LICENSE` (Create)

*   **Type / Scope** : licence.
*   **Proposed Changes** : MIT, comme Moka, pour que la redistribution du code
    et des poids copiés reste simple. Choix confirmé par l'auteur le
    2026-10-30.

###### 4. `./THIRD_PARTY_NOTICES.md` (Create)

*   **Type / Scope** : attribution.
*   **Proposed Changes** : cite Moka (MIT, « Copyright (c) 2026 Million
    Software, Inc. », `LICENSE:3`) avec l'URL et le
    commit copiés, puis reprend les notices de
    `vendor/moka/THIRD_PARTY_NOTICES.md`.

###### 5. `./vendor/moka/` (Create, copie)

*   **Type / Scope** : code de référence en lecture seule.
*   **Proposed Changes** : copie, au commit `a418039e`, des éléments suivants.
    Ces fichiers ne sont jamais modifiés.
    *   `src/` en entier ;
    *   `model/go-model.json` et `model/go-model.bin` ;
    *   pour référence seulement : `web/benchmark.ts`, `web/constants.ts`,
        `wasm/src/lib.rs`, `README.md` et `experiment-log.md` ;
    *   `LICENSE` et `THIRD_PARTY_NOTICES.md`.

###### 6. `./vendor/moka/VENDORED.md` (Create)

*   **Type / Scope** : traçabilité.
*   **Proposed Changes** : indique l'URL et le commit d'origine, puis un bloc
    de code au format de `sha256sum` (`<hash>  <chemin relatif à
    vendor/moka/>`, une ligne par fichier copié). `pnpm oracle:check`
    (fichier 12) lit ce bloc et vérifie chaque hash.

###### 7. `./package.json` (Create)

*   **Type / Scope** : configuration Node.
*   **Proposed Changes** : nom `go-network`, `"type": "module"`,
    `"private": true`.
    *   Dépendances de développement, à des versions exactes : `esbuild`,
        `typescript`, `@types/node` et `@playwright/test`. `pnpm-lock.yaml`
        est versionné.
    *   Scripts. Ceux de S0 ne dépendent que de `build:js`, et donc pas de
        Zig, qui n'arrive qu'en S1 :

        | Script | Commande |
        | --- | --- |
        | `setup:browser` | `playwright install chromium` |
        | `build:js` | `node scripts/build-js.mjs` |
        | `gen` | `zig build gen` |
        | `build` | `zig build wasm`, puis `pnpm build:js` |
        | `oracle` | `pnpm build:js`, puis `node oracle/dist/generate.js` |
        | `oracle:check` | `pnpm build:js`, puis `node oracle/dist/generate.js --check` |
        | `calibrate` | `pnpm build:js`, puis `node oracle/dist/calibrate.js` |
        | `test` | `zig build test`, puis `zig build wasm`, puis `node --test "tests/*.test.mjs" "tests/*.test.ts"` |
        | `bench` | `pnpm build:js`, puis `node bench/run.mjs` |

        Les options se passent sans `--` (`pnpm bench --baseline`), car pnpm
        transmet les arguments au script.

###### 8. `./tsconfig.json` (Create)

*   **Type / Scope** : configuration TypeScript.
*   **Proposed Changes** : sert à `tsc --noEmit`, tandis qu'esbuild produit
    les bundles. Options :
    *   `strict`, `target: ES2022`, `module: ESNext`,
        `moduleResolution: bundler`, `noEmit: true` ;
    *   `lib: ["ES2022", "DOM"]`, sans `WebWorker`, qui entre en conflit avec
        `DOM`. Le code du Worker reste typé, car `Window.postMessage(message,
        options?: WindowPostMessageOptions)` accepte `{ transfer }` ;
    *   `erasableSyntaxOnly` et `verbatimModuleSyntax`, pour que `tsc` refuse
        ce que Node ne sait pas supprimer (`enum`, `namespace`) ;
    *   `types: ["node"]`, pour l'oracle et les tests ;
    *   `allowImportingTsExtensions: true`, pour que `tests/*.test.ts`, lancés
        par Node avec la suppression des types, importent `js/*.ts` avec leur
        extension.

    Il inclut `js/`, `oracle/`, `bench/`, `tests/` et `vendor/moka/src/`, pour
    que les types globaux des messages de Moka (`src/types.ts`) soient
    disponibles. `vendor/` étant figé, une erreur `strict` qui viendrait de
    Moka ne se corrige pas : la règle concernée est désactivée dans
    `tsconfig.json`, et ce choix est noté dans `VENDORED.md`.

###### 9. `./scripts/build-js.mjs` (Create)

*   **Type / Scope** : script de build JS.
*   **Proposed Changes** : appels esbuild qui produisent :
    *   pour le navigateur (`format: esm`, `platform: browser`,
        `target: es2022`, `bundle: true`). Le Worker doit être en ESM, car il
        utilise `import.meta.url` et se crée avec `{ type: "module" }` :
        *   `dist/worker.js`, à partir de `js/worker.ts`, seulement si ce
            fichier existe : il n'est créé qu'en S3 (fichier 27) ;
        *   `bench/dist/bench.js`, à partir de `bench/bench.ts` ;
        *   `bench/dist/moka-worker.js`, à partir de
            `vendor/moka/src/worker.ts` ;
    *   pour Node (`format: esm`, `platform: node`, `target: node24`) :
        `oracle/dist/generate.js` et `oracle/dist/calibrate.js`.

    Le script copie aussi `vendor/moka/model/*` dans `dist/`, et
    `zig-out/bin/infer.wasm` seulement s'il existe : en S0, il n'existe pas
    encore. Si `oracle/fixtures/validation/` existe, il écrit les entrées du
    banc : `bench/dist/positions.f32` (100 positions × 972 valeurs) et
    `bench/dist/expected.f32` (100 × 82 logits, puis 100 valeurs, en mode
    normal). Il copie aussi `oracle/tolerance.json` seulement s'il existe :
    `pnpm calibrate` lance `build:js` avant de le créer. La page de mesure
    échoue si l'un de ces trois fichiers manque.

    Les entrées du banc n'existent donc qu'après un `build:js` lancé après
    `pnpm oracle` et `pnpm calibrate`. C'est le cas dans l'ordre de S0 du
    Verification Plan, puisque `pnpm bench` relance `build:js`.

    Un bundler est nécessaire parce que les imports de `src/` de Moka n'ont
    pas d'extension, et que Node ne peut donc pas les exécuter directement.

###### 10. `./oracle/mulberry32.ts` (Create)

*   **Type / Scope** : utilitaire.
*   **Proposed Changes** : générateur pseudo-aléatoire mulberry32,
    déterministe pour une graine `u32`, qui renvoie des valeurs dans [0, 1).

###### 11. `./oracle/moka-runtime-instrumented.ts` (Create)

*   **Type / Scope** : référence numérique instrumentée.
*   **Proposed Changes** :
    *   le premier commit est une copie de `vendor/moka/src/runtime.ts`, avec
        en tête un commentaire qui indique la source et le commit. Seul le
        chemin d'import change : `./runtime-constants` devient
        `../vendor/moka/src/runtime-constants` ;
    *   un second commit ajoute :
        *   l'option `f32Accumulate`, qui applique `Math.fround` après chaque
            produit et chaque somme des convolutions uniquement. Le pooling et
            les couches linéaires restent en f64, comme dans Zig. Elle se
            passe à la création :
            `GoModelRuntime.create(manifest, weights, { f32Accumulate })`,
            avec `false` par défaut ;
        *   la méthode `getDequantizedTensors(): Record<string,
            GoModelFloatTensor>` (type global de `src/types.ts:28-31`), qui
            renvoie le champ privé `tensors` (`src/runtime.ts:169`).
    *   **Contrôle** : le générateur (fichier 12) vérifie au démarrage que le
        mode normal donne, sur les sondes, des sorties identiques bit à bit à
        celles du `GoModelRuntime` de `vendor/`. Sinon, il s'arrête avec le
        code 1.

###### 12. `./oracle/generate-fixtures.ts` (Create)

*   **Type / Scope** : générateur de données de référence.
*   **Proposed Changes** : importe les règles et l'encodage depuis
    `vendor/moka/src/index.ts`, et les constantes (`GO_PASS_MOVE`,
    `GO_MINIMUM_PASS_MOVE_COUNT`, `GO_BOARD_AREA`,
    `GO_STUDENT_INPUT_PLANE_COUNT`, qui vaut 12 ; 972 = 81 × 12) depuis
    `src/constants.ts`, au lieu de recopier ces nombres. Le `.bin` lu par `fs.readFileSync` est découpé en
    `ArrayBuffer` avec `byteOffset` et `byteLength`, comme dans
    `tests/model-smoke.mjs:8-11`, avant `GoModelRuntime.create`.
    *   **Positions enregistrées :** chaque position où un coup va être
        joué, position initiale comprise. La position terminale
        (`isGameOver` vrai) n'est pas enregistrée.
    *   **Tirage aléatoire :** `n` est le nombre de coups permis, c'est-à-dire
        `getLegalMoves` sans la passe tant que `moveCount < 20`. Le coup tiré
        est l'indice `Math.floor(rng() × n)`. Si `n = 0`, le coup est la
        passe, comme le fait `selectHighestLegalMove` (`src/game.ts:171`,
        `:177-178`).
    *   **Calibration :** 50 parties, graines 1 à 50.
        *   Chaque coup est tiré au hasard selon la règle ci-dessus.
        *   La partie s'arrête quand `isGameOver` devient vrai.
    *   **Validation :** 20 parties, graines 101 à 120.
        *   Les 6 premiers coups sont tirés au hasard selon la même règle.
        *   Moka joue ensuite `selectHighestLegalMove` avec ses logits en mode
            normal.
        *   La partie s'arrête à `isGameOver`, qui couvre aussi la limite de
            Moka (`GO_MAXIMUM_MOVE_COUNT = 120`, `src/constants.ts:13`).
    *   **Sondes**, ajoutées au jeu de validation :
        *   la partie du journal (`experiment-log.md:1561`) : Noir joue 20,
            puis Blanc joue le coup de Moka, et ainsi de suite pour 24, 56 et
            60. Les 4 positions où Blanc doit répondre sont enregistrées. Le
            générateur calcule lui-même la réponse de Moka avec
            `go-model.bin`. Si un coup de Noir devient illégal (Moka a déjà
            joué sur cette intersection), le générateur s'arrête avec le
            code 1 : la sonde est alors à redéfinir ;
        *   la position de `tests/model-smoke.mjs:13-20` : 20, 24, 56 et 60
            joués en alternant les couleurs.
    *   **Pour chaque position :**
        *   les 972 plans (`encodeStudentFeatures`) ;
        *   les logits et la valeur, dans les deux modes ;
        *   `moveCount` ;
        *   un masque `u8[82]` des coups légaux : `getLegalMoves` pour les 81
            intersections, et la passe à 1 seulement si `moveCount ≥ 20`
            (`GO_MINIMUM_PASS_MOVE_COUNT`) ;
        *   le coup choisi par `selectHighestLegalMove` en mode normal.
    *   **Une seule fois :** tous les tenseurs déquantifiés par Moka.
    *   **Format** *(généré)*. Un dossier par jeu,
        `oracle/fixtures/calibration/` et `oracle/fixtures/validation/`, qui
        contient, en little-endian, pour `P` positions :

        | Fichier | Type | Taille |
        | --- | --- | --- |
        | `features.f32` | f32 | `P × 972` |
        | `logits-normal.f32`, `logits-f32acc.f32` | f32 | `P × 82` |
        | `value-normal.f64`, `value-f32acc.f64` | f64 | `P` |
        | `move-count.u16` | u16 | `P` |
        | `legal.u8` | u8 | `P × 82` |
        | `chosen-move.u8` | u8 | `P` |
        | `index.json` | JSON | `{ positions: P, games: [{ seed, probe, start, count }] }`, où `seed` vaut `null` et `probe` vaut `"journal"` ou `"smoke"` pour une sonde, et `probe` vaut `null` pour une partie |

        Les tenseurs sont dans `oracle/fixtures/tensors/`, un fichier
        `<nom du tenseur>.f32` par tenseur, avec un `index.json` qui donne la
        forme de chacun.
    *   **Option `--check` (G2) :** le générateur produit les données une
        seconde fois dans un dossier temporaire et les compare octet pour
        octet aux fichiers existants. Il vérifie aussi les sha256 listés dans
        `vendor/moka/VENDORED.md`. Il sort avec le code 1 à la première
        différence.

###### 13. `./oracle/calibrate-tolerance.ts` et `./oracle/tolerance.json` (Create)

*   **Type / Scope** : calibration.
*   **Proposed Changes** : le script calcule l'écart absolu maximal entre le
    mode normal et le mode `f32Accumulate`, sur le jeu de calibration, pour
    les logits et pour la valeur.
    *   Il écrit `{ logits: 4 × max, value: 4 × max, measured: {...} }`.
    *   Il échoue si `logits > 2e-3` ou si `value > 1e-3`.

    `tolerance.json` est versionné, car c'est la trace de la décision.

###### 14. `./bench/index.html` et `./bench/bench.ts` (Create)

*   **Type / Scope** : page de mesure.
*   **Proposed Changes** : reprise de `web/benchmark.ts` de Moka, avec des
    corrections.
    *   **Entrées :** `bench/dist/positions.f32` et `bench/dist/expected.f32`
        (fichier 9), soit les 100 premières positions du jeu de validation.
    *   **Paramètres d'URL** de la page `/bench/index.html` :
        *   `runtime=moka|zig` choisit le moteur ;
        *   `worker=1` passe par le client de Moka, créé avec
            `new Worker(url, { type: "module" })`, où `url` vaut
            `/bench/dist/moka-worker.js` ou `/dist/worker.js` ;
        *   `worker=0` appelle le moteur sur le thread principal :
            `GoModelRuntime.create` pour Moka, ou `js/engine.ts` pour Zig.

        En S0, seul `runtime=moka` existe. `runtime=zig` s'ajoute en S3
        (fichier 27), avec `js/engine.ts`.
    *   **Ordre d'une mesure :** initialisation jusqu'à `ready`, puis chauffe
        sur les positions 0 à 9 (`firstTenMs`), puis contrôle, puis les 100
        inférences mesurées, puis les rafales. Chaque mesure se fait après une
        nouvelle navigation vers la page, pour repartir d'un Worker neuf.
    *   **Contrôle avant mesure :** les 100 sorties sont comparées à
        `expected.f32` avec la tolérance de G3. Un écart fait échouer la
        mesure. La page échoue aussi si `crossOriginIsolated` est faux, car
        `performance.now()` serait alors arrondi.
    *   **Copie des entrées :** le client de Moka transfère
        `features.buffer`, ce qui le détache. Chaque appel reçoit donc un
        `Float32Array` neuf, copié depuis `positions.f32` hors de la zone
        chronométrée.
    *   **Rafales :** avec `worker=1`, la page mesure aussi 5 rafales de 35
        appels `infer` successifs, chacun attendu avant le suivant.
    *   **Échec :** une sortie `NaN` fait échouer la mesure.
    *   **Résultat :** la page publie dans `window.__benchResult` un objet
        `{ initializationMs, firstTenMs, p50Ms, p95Ms, meanMs, burst35Ms,
        iterations }` :
        *   `initializationMs` : du début du `fetch` des poids à la réponse
            `ready` (ou à la fin de `create` avec `worker=0`) ;
        *   `firstTenMs` : les 10 inférences de chauffe, une durée chacune ;
        *   `p50Ms`, `p95Ms`, `meanMs` : sur les `iterations` = 100
            inférences mesurées, chacune de l'appel `infer` à la résolution de
            sa promesse. Les percentiles reprennent `getPercentile` de
            `web/benchmark.ts:12-19` ;
        *   `burst35Ms` : la durée totale de chacune des 5 rafales (`null`
            avec `worker=0`). Les rafales réutilisent les positions 0 à 34.

        En cas d'échec (contrôle, `NaN`, fichier manquant, isolation
        absente), la page publie `{ error: string }` à la place.
        `initializationMs` ne compte pas la compilation d'`infer.wasm`, qui
        se fait au chargement du Worker.

###### 15. `./bench/serve.mjs` (Create)

*   **Type / Scope** : serveur statique.
*   **Proposed Changes** : un serveur `node:http` qui sert `bench/` sous
    `/bench/` et `dist/` sous `/dist/`, sur le port donné par `--port`
    (4173 par défaut). Types MIME : `application/wasm` pour `.wasm`,
    `text/javascript` pour `.js` (Chromium refuse un Worker module servi
    sans type JavaScript), `application/json` pour `.json`,
    `application/octet-stream` pour `.bin` et `.f32`.
    Il envoie le type `application/wasm` pour les `.wasm`, ainsi que les
    en-têtes `Cross-Origin-Opener-Policy: same-origin` et
    `Cross-Origin-Embedder-Policy: require-corp`, qui donnent à
    `performance.now()` une résolution fine. Le mode `serve` d'esbuild ne
    permet pas d'ajouter ces en-têtes.

###### 16. `./bench/run.mjs` (Create)

*   **Type / Scope** : pilote Playwright.
*   **Proposed Changes** :
    *   démarre `serve.mjs`, puis lance le Chromium de Playwright en headless ;
    *   enregistre `browser.version()` (le binaire qui mesure vraiment : en
        headless, Playwright lance `chromium-headless-shell`) et le modèle de
        CPU (`os.cpus()`) ;
    *   fait 5 tours pour chaque valeur de `worker`, en alternant l'ordre des
        moteurs d'un tour à l'autre ;
    *   écrit `bench/results/<horodatage>.json` et affiche un tableau ;
    *   sort avec le code 2 si une page publie `{ error }`, sans écrire de
        résultat.

    Schéma des fichiers de résultats, y compris `baseline-moka.json` :

    ```text
    { date, machine: { cpu, browserVersion, node },
      runs: [ { round: 1..5, runtime: "moka" | "zig", worker: 0 | 1,
                initializationMs, firstTenMs: number[10], p50Ms, p95Ms,
                meanMs, burst35Ms: number[5] | null, iterations } ],
      summary: { [runtime]: { [worker]: { medianP50Ms } } },
      verdict?: { ratio, passed, baselineDriftPercent } }
    ```

    Options :

    *   `--baseline` : ne mesure que Moka et écrit
        `bench/results/baseline-moka.json`, versionné en S0 ;
    *   `--verdict` : mesure les deux moteurs dans la même exécution, puis
        calcule G4 (rapport des médianes des p50 de Moka et de Zig,
        `worker=1`). Il sort avec le code 1 si ce rapport est inférieur à 2.
        La référence G1 sert seulement à vérifier que la mesure de Moka n'a
        pas dérivé depuis S0 : si la médiane des p50 de Moka s'en écarte de
        plus de 20 %, `run.mjs` affiche un avertissement et l'enregistre dans
        `baselineDriftPercent`. Le verdict reste celui de la mesure du jour,
        mais il ne vaut pas comparaison avec S0 et doit être noté comme tel
        au fichier 29.

##### S1 : squelette Zig

###### 17. `./build.zig`, `./build.zig.zon`, `./src/infer/exports.zig` (Create) et `./mise.toml` (Modify)

*   **Type / Scope** : build Zig.
*   **Proposed Changes** :
    *   `build.zig.zon` contient le nom, la version, `fingerprint`, `paths`
        et `minimum_zig_version` (les deux premiers champs sont obligatoires
        depuis Zig 0.14).
    *   La version exacte de Zig est ajoutée à `mise.toml` (fichier 1).
    *   `src/infer/exports.zig` est créé ici sous une forme minimale, qui
        n'exporte que `max_batch()` et `weights_capacity()`, cette dernière
        lue dans `model_layout`. L'étape `wasm` produit ainsi dès S1 un
        binaire construit à partir du manifeste, ce qui est le critère de
        sortie de S1. Il est complété au fichier 26. Dès ce premier build, on
        relève les exports que l'éditeur de liens ajoute avec
        `rdynamic = true`. Avec Zig 0.17.0, la seule exception relevée est
        `__stack_pointer` (pas d'`__indirect_function_table`). Elle est
        écrite dans le test du fichier 35.
    *   **Étape `gen`** :
        *   compile `tools/gen_model_layout.zig` pour l'hôte ;
        *   l'exécute avec `vendor/moka/model/go-model.json`, le `.bin` et le
            chemin du fichier de sortie, dans le cache
            (`addOutputFileArg("model_layout.zig")`) ;
        *   copie aussi ce fichier dans `zig-out/gen/model_layout.zig`, pour
            pouvoir le relire ;
        *   ajoute le module produit (`model_layout`) aux imports de
            `infer.wasm` et des tests.
    *   **Étape `test`** : un exécutable de test natif, de racine
        `src/test_root.zig`, plus les tests de `tools/gen_model_layout.zig`.
    *   **Étape `wasm`** :
        *   racine `src/infer/exports.zig` ;
        *   cible `wasm32-freestanding` avec `simd128`, mode `ReleaseFast` ;
        *   options : `entry = .disabled`, `rdynamic = true`,
            `initial_memory = max_memory = 16 × 65536`,
            `stack_size = 65536`, `import_memory = false`,
            `export_memory = true`. Le nombre de pages (16) est une
            estimation, confirmée en S2 quand `buffers.zig` existe ;
        *   `strip = true` : sans lui, `ReleaseFast` garde environ 5,5 Kio
            d'informations de débogage DWARF, qui fausseraient G6.
    *   **Option `-Dfixtures`** : le chemin des données de référence, par
        défaut `oracle/fixtures`. Il est passé aux tests par le module
        `build_options` (`fixtures_dir`), avec `addOptionPathUntracked`, pour
        que le build ne hache pas les données. `b.pathFromRoot` n'existe plus
        en Zig 0.17 :
        *   un chemin relatif reste relatif à la racine du build, et
            `zig build` lance les tests depuis cette racine, même quand on
            l'appelle depuis un sous-dossier ;
        *   un chemin absolu est gardé tel quel (`b.graph.cwdRelativePath`).

###### 18. `./src/test_root.zig` (Create)

*   **Type / Scope** : racine des tests natifs.
*   **Proposed Changes** : importe tous les modules de `src/` pour exécuter
    leurs blocs `test`.

###### 19. `./src/geometry.zig` (Create)

*   **Type / Scope** : bibliothèque Zig.
*   **Proposed Changes** :
    *   `pub fn Geometry(comptime N: u16) type` expose `AREA`,
        `SENTINEL = AREA` et `CONV_NEIGHBORS: [AREA][9]u16` ;
    *   les voisins sont rangés dans l'ordre ligne par ligne, de (-1,-1) à
        (+1,+1) ;
    *   la table est calculée au `comptime` ;
    *   les tests à N = 9 vérifient les coins, les bords et le centre.

###### 20. `./tools/gen_model_layout.zig` (Create)

*   **Type / Scope** : outil de build, côté hôte. Il peut allouer.
*   **Proposed Changes** :
    *   **Lecture du manifeste** avec `std.json`. L'outil refuse tout manifeste
        qui ne remplit pas toutes ces conditions :
        *   `format = million-go-int8` ;
        *   `version = 1` ;
        *   `residualBlockKind = nested-bottleneck` ;
        *   aucun tenseur avec `quantizationGroupSize` ;
        *   uniquement des tenseurs `int8` ou `float32`.
    *   **Contrôle du `.bin`** : il calcule le sha256 avec `std.crypto`, le
        compare au manifeste, et vérifie aussi la taille.
    *   **Sortie** : il écrit `model_layout.zig` *(généré)*, qui contient :

        ```zig
        pub const ModelConfig = struct {
            board_size: u16, input_planes: u16, trunk_channels: u16,
            bottleneck_channels: u16, residual_blocks: u16,
            global_block_interval: u16, global_hidden_channels: u16,
            policy_channels: u16, policy_moves: u16,
            value_channels: u16, value_hidden_channels: u16,
        };
        pub const Kind = enum { int8, float32 };
        pub const Layout = enum { conv3x3, conv1x1, linear, bias };
        pub const Tensor = struct {
            name: []const u8, kind: Kind, layout: Layout, shape: []const u32,
            data_offset: u32,    // octets dans le .bin
            scale_offset: ?u32,  // octets dans le .bin, int8 seulement
            f32_offset: u32,     // index dans weights_f32 (f32)
            len: u32,            // nombre de valeurs
        };
        pub const CONFIG: ModelConfig = .{ ... };
        pub const TENSORS = [_]Tensor{ ... };
        pub const WEIGHTS_BYTES: u32 = 113_648;
        pub const WEIGHTS_F32_LEN: u32 = ...; // somme des len
        pub const SHA256: [32]u8 = .{ ... };
        pub fn tensor(comptime name: []const u8) Tensor { ... }
        ```

        `tensor` cherche le nom au `comptime` et produit un `@compileError`
        si le tenseur manque.
    *   **Correspondance avec le manifeste** (`architecture` de
        `go-model.json`) :

        | Champ Zig | Champ JSON |
        | --- | --- |
        | `board_size` | `boardSize` |
        | `input_planes` | `inputPlaneCount` |
        | `trunk_channels` | `trunkChannelCount` |
        | `bottleneck_channels` | `bottleneckChannelCount` |
        | `residual_blocks` | `residualBlockCount` |
        | `global_block_interval` | `globalResidualBlockInterval` |
        | `global_hidden_channels` | `globalResidualHiddenChannelCount` |
        | `policy_channels` | `policyChannelCount` |
        | `policy_moves` | `policyMoveCount` |
        | `value_channels` | `valueChannelCount` |
        | `value_hidden_channels` | `scoreHiddenChannelCount` |

    *   **Tenseurs requis.** L'outil contient la liste des tenseurs dont le
        réseau a besoin, et la forme de chacun en fonction de `ModelConfig` :
        `stem.*`, `residual.<i>.{reduce,first,second,expand}.*`,
        `residual.<i>.global.{hidden,output}.*` pour les blocs 3, 7 et 11,
        `policy.{convolution,linear}.*`, `value.{convolution,hidden,output}.*`
        (noms de `src/runtime.ts:284-430`). Il refuse le manifeste si un
        tenseur manque ou n'a pas la forme attendue.
    *   **`layout` selon la forme** : `[o,3,3,i]` donne `conv3x3`,
        `[o,1,1,i]` donne `conv1x1`, une forme à 2 dimensions donne
        `linear`, et une forme à 1 dimension donne `bias`.
    *   **Autres refus**, sur le modèle de `isSupportedManifest`
        (`src/runtime.ts:20-42`) : un tenseur `int8` sans échelle (Moka lève
        « Missing quantization scale for … », `src/runtime.ts:214-216`), et
        tout décalage dont `offset + taille` dépasse `weightsBytes`. En
        `ReleaseFast`, une lecture hors bornes serait un comportement
        indéfini.
    *   **Tests Zig** : un tenseur `int4` et un tenseur avec
        `quantizationGroupSize` sont refusés. Ce champ appartient à chaque
        tenseur, et non au manifeste.

##### S2 : poids

###### 21. `./src/infer/buffers.zig` (Create)

*   **Type / Scope** : mémoire statique.
*   **Proposed Changes** :
    *   déclare les variables globales du tableau mémoire du Technical Plan ;
    *   leurs tailles sont dérivées de `ModelConfig` et de `Geometry(9)` ;
    *   les lignes sentinelles sont mises à zéro une fois, puis ne sont plus
        jamais écrites. Le test qui le vérifie après une inférence arrive au
        fichier 25, car l'inférence n'existe pas encore en S2 ;
    *   `reorder_tmp` a la taille du plus gros tenseur de convolution,
        calculée au `comptime` à partir de `TENSORS` ;
    *   la somme des tailles, pile comprise, fixe le nombre de pages mémoire
        de `build.zig` (fichier 17) : le plus petit nombre de pages qui
        laisse au moins 25 % de marge (voir Open Questions). Un test
        `comptime` fait échouer le build si la marge n'est plus tenue.

###### 22. `./src/infer/weights.zig` (Create)

*   **Type / Scope** : chargement des poids.
*   **Proposed Changes** : trois fonctions publiques.
    *   `dequantize(bin: []const u8, out: []f32) void` : pour chaque entrée
        de `TENSORS`, écrit à `f32_offset` :
        *   `int8` donne `f32(q) × scale[out]` ;
        *   `float32` est recopié tel quel.

        La lecture se fait en little-endian, sans hypothèse d'alignement.
    *   `reorder(weights: []f32, tmp: []f32) void` : ré-agence en place
        chaque tenseur de convolution selon `layout`. Il copie le tenseur
        dans `tmp`, puis le réécrit à sa place dans la nouvelle disposition.
        Les autres tenseurs ne bougent pas.
    *   `load() void` : appelle `dequantize(weights_bin, weights_f32)`, puis
        `reorder(weights_f32, reorder_tmp)`, et marque les poids comme
        chargés. Il lit `WEIGHTS_BYTES`
        octets ; la longueur a été contrôlée par le Worker avant la copie.

    Ces fonctions prennent des tranches, ce qui permet au test de travailler
    sur des tampons à lui. **Test natif** : `build.zig` ajoute le `.bin` comme
    import anonyme (`addAnonymousImport("go_model_bin", ...)`, lu avec
    `@embedFile`, dans les tests seulement). Le test lit les tenseurs de
    Moka dans `<fixtures>/tensors/`, où `<fixtures>` est l'option
    `-Dfixtures` (fichier 17). Le test appelle `dequantize`, puis compare
    chaque tenseur aux fichiers de Moka, bit à bit. Il échoue avec un message
    explicite si les données de référence manquent : `zig build test`
    demande d'avoir lancé `pnpm oracle` avant.

##### S3 : inférence et verdict

###### 23. `./src/infer/kernels.zig` (Create)

*   **Type / Scope** : noyaux SIMD.
*   **Proposed Changes** :
    *   les noyaux, avec l'ordre d'opérations du Technical Plan :
        *   convolution 3×3 avec ReLU au stockage (stem, `first`, `second`) ;
        *   convolution 1×1 avec ReLU au stockage (`reduce`, convs des deux
            têtes) ;
        *   convolution 1×1 avec ajout au tronc : `acc = biais + Σ x·w`, puis
            `trunk = relu(trunk + acc)` (`expand`) ;
    *   ils sont génériques sur la géométrie et le nombre de canaux ;
    *   ils utilisent `@Vector(4, f32)` sur les canaux de sortie. Quand le
        nombre de sorties n'est pas multiple de 4 (conv de la tête de valeur,
        2 sorties), une boucle scalaire f32 fait les mêmes opérations dans le
        même ordre ;
    *   la ReLU est une fonction `inline` à comparaisons explicites, sans
        `@max` (voir « ReLU » dans le Technical Plan).

###### 24. `./src/infer/dense.zig` (Create)

*   **Type / Scope** : couches scalaires f64.
*   **Proposed Changes** :
    *   pooling global (moyenne et maximum). Le maximum s'écrit
        `if (x > m or x != x) m = x`, pour propager `NaN` comme `Math.max` ;
    *   couche linéaire, avec un accumulateur f64 et un stockage f32 ;
    *   `tanh` en f64 ;
    *   arrondis conformes à l'annexe A3.

###### 25. `./src/infer/network.zig` (Create)

*   **Type / Scope** : passe avant.
*   **Proposed Changes** : `forward(k)` traite la position `k` du lot selon le
    schéma de « Architecture du réseau » :
    *   stem ;
    *   12 blocs ;
    *   tête de politique ;
    *   tête de valeur ;
    *   contrôle que les sorties sont finies.

    Un test natif lance `forward` sur une position des données de référence,
    puis vérifie que les lignes sentinelles de tous les tampons sont encore
    nulles.

###### 26. `./src/infer/exports.zig` (Modify)

*   **Type / Scope** : ABI de `infer.wasm`.
*   **Proposed Changes** :
    *   expose les exports de la liste du Technical Plan ;
    *   `expected_sha256_ptr` pointe sur `SHA256` ;
    *   `infer(count)` vérifie `1 ≤ count ≤ 8` et que les poids sont chargés,
        puis appelle `forward` pour chaque position ;
    *   le gestionnaire de `panic` fait `@trap()`. Depuis Zig 0.14,
        `pub const panic = std.debug.no_panic;` dans la racine le fait déjà ;
        sinon, la forme suit l'API de la version de Zig fixée en S1.

###### 27. `./js/engine.ts`, `./js/worker.ts` (Create) et `./bench/bench.ts` (Modify)

*   **Type / Scope** : moteur JS autour de `infer.wasm`, et Web Worker publié
    sous le nom `dist/worker.js`.
*   **Proposed Changes** :
    *   **`js/engine.ts`** exporte deux fonctions, sans état global :
        *   `createEngine(module: WebAssembly.Module, weights: ArrayBuffer):
            Promise<Engine>` : instanciation, puis contrôles de longueur et
            de hash, copie dans `weights_ptr()` et `load_weights()`, dans
            l'ordre de « `infer.wasm` : chargement des poids ». Le hash se
            calcule avec `getBufferSha256` de Moka
            (`src/utils/get-buffer-sha256.ts`), comparé aux 32 octets de
            `expected_sha256_ptr()` convertis en hexadécimal. Elle lève
            l'erreur du tableau des erreurs au premier contrôle qui échoue.
            Le message de hash cite les 8 premiers caractères du hash
            attendu, suivis de « … », comme dans le tableau ;
        *   `Engine.infer(features: Float32Array): GoModelInference` :
            1.  vérifie que `features.length === 972` ;
            2.  copie les plans à `input_ptr()` ;
            3.  appelle `infer(1)` ;
            4.  traduit un `Status` différent de `ok` en erreur, selon le
                tableau des erreurs ;
            5.  copie les 82 logits dans un nouveau `Float32Array` et lit la
                valeur f64.
    *   **`js/worker.ts`** garde la structure de `vendor/moka/src/worker.ts`
        et les types de messages de Moka :
        *   **au chargement du module** :
            `fetch(new URL("./infer.wasm", import.meta.url))`, puis
            `arrayBuffer()`, puis `WebAssembly.compile`. La promesse est
            gardée, et son erreur est conservée pour `initialize` ;
        *   **file d'attente** : les messages sont traités un par un, dans
            l'ordre d'arrivée ;
        *   **`initialize`** : met l'instance courante à `null`, attend la
            compilation, puis appelle `createEngine`. Il ne garde la nouvelle
            instance qu'en cas de succès, puis répond `ready`. En phase 1,
            tout échec de compilation donne « Failed to load infer.wasm. » ;
            la recherche de la cause vient au fichier 34 ;
        *   **`infer`** : sans instance, l'erreur est « Go model worker is
            not initialized. ». Sinon, il appelle `Engine.infer` et répond
            `result` en transférant les logits ;
        *   **erreurs** : comme Moka, un message `error` avec le texte de
            l'erreur, ou « Go model inference failed. » si ce n'est pas une
            `Error`.
    *   **`bench/bench.ts`** reçoit `runtime=zig` : avec `worker=1`, il crée
        le client avec `/dist/worker.js` ; avec `worker=0`, il compile
        `/dist/infer.wasm` et appelle `createEngine`.

###### 28. `./tests/infer-parity.test.mjs` (Create)

*   **Type / Scope** : test Node sur le `.wasm`, exécuté par V8 comme dans
    Chromium.
*   **Proposed Changes** : instancie `zig-out/bin/infer.wasm` et charge le
    `.bin`. Le test lit les fichiers de `oracle/fixtures/` décrits au
    fichier 12.
    *   **Sur les deux jeux** : chaque `infer` renvoie `Status == ok`, ce qui
        exclut toute sortie `NaN` ou infinie (G3). Ce test échoue
        normalement.
    *   **Diagnostic bit à bit, qui ne fait jamais échouer la suite** : les
        logits sont comparés bit à bit à ceux du mode `f32Accumulate`, et la
        valeur à 2 ulp près. Le test passe toujours ; il rapporte par
        `t.diagnostic()` le nombre de positions en écart et la première
        d'entre elles. Le code de sortie de `node --test` ne dépend donc que
        de G3 et des autres vérifications.
    *   **Sur le jeu de validation (G3)**, ces tests échouent normalement :
        *   l'écart avec le mode normal reste sous `tolerance.json` ;
        *   le meilleur coup légal est le même que `chosen-move.u8`. Le test
            reproduit la règle de `selectHighestLegalMove`
            (`src/game.ts:175-193`) à partir de `legal.u8` : départ sur la
            passe si elle est autorisée, sinon sur le premier coup légal,
            puis un coup ne remplace le meilleur que s'il est strictement
            supérieur. Une différence est acceptée seulement en cas de
            quasi-égalité, avec la règle de G3 :
            `|logit_Moka(coup Moka) − logit_Moka(coup Zig)| ≤ 2 ×
            tolerance.logits`.
    *   **Autres vérifications** :
        *   `memory.buffer.byteLength` reste constant sur 1 000 inférences
            (G5) ;
        *   `infer(8)` donne les mêmes résultats que 8 appels à `infer(1)` ;
        *   `infer(0)` et `infer(9)` renvoient `bad_batch_count` ;
        *   `infer(1)` avant `load_weights()`, sur une instance neuve, renvoie
            `weights_not_loaded` ;
        *   `weights_capacity()` vaut 113 648, `max_batch()` vaut 8, et les
            32 octets de `expected_sha256_ptr()` sont le sha256 du `.bin`.

###### 29. `./docs/designs/zig-wasm-moka.md` (Modify)

*   **Type / Scope** : ce document de design.
*   **Proposed Changes** : passer le statut à *Approved* après la revue. À la
    fin de S0 et de S3, enregistrer les valeurs mesurées :
    *   la référence G1 et la tolérance (S0) ;
    *   le verdict G4 et le nombre d'écarts du diagnostic bit à bit (S3) ;
    *   la taille d'`infer.wasm` avant et après gzip (zlib, niveau 9). Le
        plafond contrôlé au fichier 35 vaut 32 Kio, ou la taille mesurée
        plus 25 % si elle dépasse déjà 32 Kio ; il est écrit comme une
        constante dans le test ;
    *   le nombre de pages mémoire, la taille de pile et la version de Zig.

    En S4, enregistrer quelle CSP bloque la compilation dans Chromium
    (fichier 36), puis corriger le CUJ 2 si besoin.

#### Phase 2 : consolidation (seulement si G4 passe)

##### S4 : contrôles, préparation INT16, CI, documentation

###### 30. `./src/geometry.zig` (Modify)

*   **Type / Scope** : test de géométrie à N = 19.
*   **Proposed Changes** : un test vérifie `Geometry(19)` aux coins, sur les
    bords et au centre. C'est le seul test à N = 19 de la v1 (voir
    Non-Goals).

###### 31. `./tools/gen_model_layout.zig` et `./src/infer/weights.zig` (Modify)

*   **Type / Scope** : préparation INT16 (G7).
*   **Proposed Changes** : les tenseurs `int16` sont acceptés et déquantifiés
    comme les `int8`. Un test Zig vérifie un manifeste synthétique qui contient
    un tenseur `int16`, et un test unitaire de `dequantize` vérifie la
    lecture little-endian d'un `int16` négatif.

###### 32. `./build.zig` et `./package.json` (Modify)

*   **Type / Scope** : build sûr.
*   **Proposed Changes** : nouvelle étape `wasm-safe` (`ReleaseSafe`), qui
    produit `zig-out/bin/infer-safe.wasm`. Le script `build` de
    `package.json` l'inclut.

###### 33. `./tests/infer-parity.test.mjs` (Modify)

*   **Type / Scope** : parité sur le build sûr.
*   **Proposed Changes** : la suite tourne aussi sur
    `zig-out/bin/infer-safe.wasm`.

###### 34. `./js/load-error.ts` et `./tests/load-error.test.ts` (Create), `./js/worker.ts` (Modify)

*   **Type / Scope** : gestion des erreurs.
*   **Proposed Changes** :
    *   `load-error.ts` exporte une fonction pure
        `describeLoadError({ fetchFailed, simdValid })`. Elle renvoie le
        message du tableau des erreurs selon l'étape 2.2 de « `infer.wasm` :
        chargement des poids » : téléchargement, puis SIMD, sinon CSP.
        `fetchFailed` est vrai si `fetch` a levé une erreur ou si
        `response.ok` est faux.
    *   En cas d'échec de compilation, le Worker appelle
        `WebAssembly.validate` sur les octets d'`infer.wasm` gardés depuis le
        chargement, puis `describeLoadError`. Le Worker publié n'a aucun
        crochet de test.
    *   `load-error.test.ts` teste les trois cas sous Node (`node --test`,
        avec la suppression des types de Node).
    *   Une `WebAssembly.RuntimeError` met le moteur hors service jusqu'au
        prochain `initialize`. Ce cas n'est pas provoqué par un test, car
        aucun export ne permet de déclencher un `trap`.

###### 35. `./tests/wasm-checks.test.mjs` (Create)

*   **Type / Scope** : test Node.
*   **Proposed Changes** : contrôles sur `infer.wasm`, avec
    `WebAssembly.Module.exports` et une lecture directe de la section mémoire
    du binaire :
    *   présence d'instructions `v128`. Le test cherche l'octet préfixe
        `0xfd` dans la section code. C'est une heuristique : l'octet peut
        aussi apparaître dans un immédiat. Elle suffit ici, car elle ne sert
        qu'à détecter un build où `simd128` aurait été oublié, qui n'a alors
        presque aucune instruction vectorielle ;
    *   mémoire déclarée avec `initial == maximum` ;
    *   aucune instruction `memory.grow` (opcode `0x40` suivi de `0x00`) dans
        la section code (G5), avec la même réserve d'heuristique ;
    *   taille après gzip (`zlib.gzipSync`, niveau 9) sous le plafond, écrit
        comme une constante dans le test d'après le fichier 29 ;
    *   exports exactement conformes à la liste du Technical Plan, `memory`
        compris, plus les exceptions relevées en S1 (fichier 17), sans export
        de débogage.

    Aucune dépendance n'est ajoutée.

###### 36. `./tests/e2e/worker.spec.ts`, `./tests/e2e/index.html`, `./tests/e2e/page.ts`, `./playwright.config.ts` (Create), `./bench/serve.mjs` et `./scripts/build-js.mjs` (Modify)

*   **Type / Scope** : tests Playwright dans Chromium, avec le client de Moka.
*   **Proposed Changes** :
    *   **`playwright.config.ts`** : projet Chromium seul ; `webServer`
        lance `node bench/serve.mjs --port 4174` ; `testDir: tests/e2e`.
    *   **Page de test** `tests/e2e/index.html`, servie sous `/e2e/`. Tous
        les cas e2e passent par cette page. Elle charge un script externe,
        `tests/e2e/page.ts`, empaqueté par `build-js` en
        `tests/e2e/dist/page.js`, hors de `dist/`, qui est le dossier
        publié. Le script n'est pas inline, car une CSP de page bloquerait un
        script inline. Il expose à la page :
        *   le client de Moka, sur un premier Worker ;
        *   un accès direct à un second Worker, séparé, pour les
            `postMessage` directs. Le client de Moka occupe
            `worker.onmessage` et numérote ses requêtes à partir de 0
            (`src/client.ts:6`, `:12`) : le second Worker évite tout
            mélange de réponses.
    *   **`serve.mjs`** choisit la CSP à chaque requête, selon un paramètre
        de requête, pour qu'un seul `webServer` suffise :
        *   toute réponse demandée avec `?csp=strict` reçoit l'en-tête
            `Content-Security-Policy: script-src 'self'` ;
        *   la page `/e2e/index.html?worker-csp=strict` crée ses Workers avec
            l'URL `/dist/worker.js?csp=strict`.

        Il sert aussi `tests/e2e/` sous `/e2e/`.
    *   **Cas nominal :** avec les vrais fichiers, `initialize` réussit, puis
        `infer` renvoie un `Float32Array(82)` et une valeur finie, dans la
        tolérance de G3 sur 10 positions de validation.
    *   **Poids modifiés :** un `postMessage` direct au Worker envoie un `.bin`
        modifié d'un octet avec le manifeste d'origine, sans passer par le
        client. L'erreur est « do not match this engine build ».
    *   **`.bin` tronqué :** également par `postMessage` direct, car le client
        de Moka refuse un `.bin` tronqué avant le Worker (« Go model response
        was incomplete. », `src/utils/read-response-buffer.ts:32-34`).
        L'erreur est l'erreur de longueur.
    *   **`initialize` qui échoue après un `initialize` réussi :** l'`infer`
        suivant renvoie « Go model worker is not initialized. » (CUJ 3).
    *   **Entrée de 810 valeurs :** l'erreur de longueur de l'entrée.
    *   **Entrée qui contient un `NaN` :** l'erreur de sortie non finie.
    *   **CSP stricte :** deux cas envoient `script-src 'self'`, sans
        `'wasm-unsafe-eval'` :
        *   sur la réponse de `worker.js` (`/e2e/index.html?worker-csp=strict`) :
            l'erreur CSP est attendue ;
        *   sur la page seulement (`/e2e/index.html?csp=strict`) : le test
            enregistre si la compilation est bloquée. Ce résultat répond à la
            question ouverte et fixe le texte du README.

        Le cas « sans SIMD » n'est pas testé en e2e, car Chromium a toujours
        `simd128`. Il est couvert par le fichier 34.

###### 37. `./.github/workflows/ci.yml` et `./package.json` (Create, Modify)

*   **Type / Scope** : CI.
*   **Proposed Changes** :
    *   `package.json` reçoit les scripts `test:e2e` (`playwright test`) et
        `check:all`, qui enchaîne `gen`, `build`, `oracle`, `oracle:check`,
        `test` et `test:e2e`. Le nom `ci` n'est pas utilisé, car `pnpm ci` est
        une commande de pnpm et ne lancerait pas le script.
    *   Le workflow est une tâche Ubuntu qui exécute dans l'ordre :
        1.  `jdx/mise-action` (qui installe les outils de `mise.toml`), puis
            `pnpm install --frozen-lockfile`, puis
            `pnpm exec playwright install --with-deps chromium` ;
        2.  `zig fmt --check src tools build.zig`, puis
            `pnpm exec tsc --noEmit` ;
        3.  `pnpm check:all`.
    *   Le banc de vitesse n'est pas lancé en CI, car la vitesse des machines
        y varie trop.
    *   Ce fichier ne sert que si un dépôt GitHub existe. Aucune étape du
        plan ne le crée : le critère de sortie de S4 est `pnpm check:all` au
        vert sur la machine de développement, après les étapes 1 et 2 ci-dessus
        lancées à la main.

###### 38. `./README.md` (Create)

*   **Type / Scope** : documentation.
*   **Proposed Changes** : le README couvre :
    *   l'objectif et le statut, avec le verdict G4 ;
    *   l'installation et les commandes ;
    *   **l'intégration** :
        *   l'exemple du README de Moka, avec la seule URL du Worker changée ;
        *   `worker.js` et `infer.wasm` servis depuis l'origine de la page ;
        *   la CSP qui doit autoriser `'wasm-unsafe-eval'`, selon le résultat
            noté en S4 (fichiers 29 et 36) ;
        *   le code de repli sur le Worker de Moka (CUJ 4) ;
    *   la marche à suivre quand Moka publie de nouveaux poids (nouveau build) ;
    *   le support navigateur ;
    *   la garantie « zéro allocation » et son périmètre ;
    *   l'attribution à Moka.

#### Outils de diagnostic (seulement en cas de divergence)

Ces fichiers ne sont créés que si le diagnostic bit à bit du fichier 28
signale des écarts et que la cause n'est pas évidente. Ils servent à trouver la première couche qui
diverge. Ils ne font pas partie de l'artefact publié.

###### D1. `./src/infer/reference.zig` (Create) et `./src/test_root.zig` (Modify)

*   **Type / Scope** : référence scalaire, tests seulement.
*   **Proposed Changes** : les mêmes noyaux en scalaire, importés uniquement
    par `src/test_root.zig`. Des tests vérifient l'égalité bit à bit entre les
    noyaux SIMD et scalaires sur des entrées aléatoires, à N = 9.

###### D2. `./build.zig`, `./src/infer/network.zig` et `./src/infer/exports.zig` (Modify)

*   **Type / Scope** : build de débogage.
*   **Proposed Changes** :
    *   nouvelle étape `wasm-debug` (`-Dlayer-dump=true`), qui produit
        `zig-out/bin/infer-debug.wasm` ;
    *   dans ce build, `network.zig` copie chaque couche de la première
        position dans un tampon de vidage, et `exports.zig` ajoute
        `layer_dump_ptr()`. Les points de vidage sont exactement ceux de D3,
        dans le même ordre.

###### D3. `./oracle/moka-runtime-instrumented.ts` et `./oracle/generate-fixtures.ts` (Modify)

*   **Type / Scope** : vidages de l'oracle.
*   **Proposed Changes** : option `dumpLayers`, qui renvoie les activations
    après le stem, après chaque sous-couche de chaque bloc et après chaque
    couche des têtes. Le générateur écrit les vidages des 20 premières
    positions de la partie 1 de calibration.
    *   **Format**, commun à D2 et D3 : pour chaque point de vidage, dans
        l'ordre de la passe avant, un fichier
        `oracle/fixtures/layers/<position>/<NN>-<nom>.f32`, où `<nom>` est
        par exemple `stem`, `residual.3.global` ou `policy.linear`. Les
        couches convolutives ont 81 lignes, sans la ligne sentinelle, en
        disposition `[intersection][canal]` ; les couches denses ont un
        vecteur.

###### D4. `./tests/infer-layers.test.mjs` (Create)

*   **Type / Scope** : test Node de diagnostic.
*   **Proposed Changes** : avec `infer-debug.wasm`, compare chaque couche aux
    vidages de l'oracle en mode `f32Accumulate`, bit à bit. En cas d'échec, il
    nomme la première couche qui diverge.

### Verification Plan

*   **Compilation/Build** :
    *   S0, sans Zig : `mise install && pnpm install`, puis
        `pnpm setup:browser` (Chromium de Playwright, nécessaire à
        `pnpm bench`), puis `pnpm build:js`,
        qui produit `dist/` (sans `infer.wasm`), `bench/dist/` et
        `oracle/dist/`.
    *   À partir de S1 : `pnpm gen`, puis `pnpm build`, qui ajoute
        `zig-out/bin/infer.wasm` et sa copie dans `dist/`. La phase 2 ajoute
        `infer-safe.wasm`.
*   **Unit Tests** : `zig build test`, après `pnpm oracle`.
    *   En phase 1 : géométrie à N = 9, poids, sentinelles et outil de build.
    *   En phase 2 : géométrie à N = 19 et INT16.
*   **Données de référence** : `pnpm oracle`, puis `pnpm oracle:check` (G2),
    dès S0.
*   **Parité et intégration** :
    `node --test "tests/*.test.mjs" "tests/*.test.ts"`.
    *   En phase 1 : fichier 28. Les écarts bit à bit sont rapportés par
        `t.diagnostic()` et ne font jamais échouer la suite.
    *   En phase 2 : fichiers 33, 34 et 35, puis `pnpm test:e2e`
        (fichier 36).
*   **Static Analysis & Linting** : `zig fmt --check src tools build.zig`,
    puis `tsc --noEmit`.
*   **Test Coverage Requirement** : pas de pourcentage. La couverture est
    définie par les données de référence : toutes les positions des deux jeux,
    et tous les cas d'erreur du fichier 36, doivent passer. Chaque export de
    `infer.wasm` est appelé au moins une fois par un test Node.
*   **Mesures** :
    *   S0, dans cet ordre : `pnpm oracle`, puis `pnpm calibrate`, qui produit
        `oracle/tolerance.json`, puis `pnpm bench --baseline`, qui produit
        la référence G1. Les deux fichiers sont versionnés.
    *   Fin de S3 : `pnpm bench --verdict`. Un code de sortie 0 signifie que
        G4 est atteint et ouvre la phase 2. Un code 1 arrête le projet.
*   **Critères de sortie** : voir le tableau des étapes dans Goals.

--------------------------------------------------------------------------------

*Drafted with EGM*
