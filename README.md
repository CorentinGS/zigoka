# go-network

Moteur d'inférence Zig/WASM, sans allocation, pour le réseau de Go 9×9 de
[Moka](https://github.com/millionco/moka). Il remplace le Web Worker de Moka
sans changer son API : la page garde le client, les règles et l'encodage de
Moka, et seul le calcul de la passe avant change.

## Statut

La phase 1 (prototype et verdict) est terminée ; la phase 2 (consolidation)
est en cours. Le design complet est dans
[`docs/designs/zig-wasm-moka.md`](docs/designs/zig-wasm-moka.md). Le passage
en N×N et la mesure en 19×19 sont décrits dans
[`docs/designs/zig-wasm-nxn.md`](docs/designs/zig-wasm-nxn.md) (voir
[19×19](#1919)).

*   **Vitesse (G4)** : médiane des p50 d'un aller-retour `infer` via le Worker,
    en 9×9 (`moka-9`), dans Chromium 156 sur un AMD Ryzen 7 8845HS : 5,900 ms
    pour Moka et 0,365 ms pour ce moteur, soit un rapport de 16,16 (le seuil
    était 2). Ce résultat ne vaut que pour cette machine.
*   **Exactitude (G3)** : sur 1 481 positions de validation, l'écart maximal
    avec Moka est de 1,5e-5 sur les logits et de 1,9e-6 sur la valeur, et le
    meilleur coup légal est toujours le même. Moka accumule en f64 et ce
    moteur en f32 ; les logits sont identiques bit à bit à ceux d'un Moka
    instrumenté pour accumuler en f32.
*   **Taille** : l'`infer.wasm` de `moka-9` fait environ 26 Ko, et 8,8 Ko
    après gzip.

## 19×19

Le moteur est écrit pour N quelconque : N est lu dans le manifeste au build,
et toutes les tailles du moteur en dérivent. Faute de poids 19×19 publiés, il
est mesuré en 19×19 sur `synthetic-19` (`models/synthetic-19/`), dont le
tronc vient de `moka-9` et dont les têtes sont générées (voir
[`PROVENANCE.md`](models/synthetic-19/PROVENANCE.md)). Ses sorties sont
identiques bit à bit à celles d'un Moka instrumenté pour accumuler en f32 et
compilé pour 19×19.

Médiane des p50 d'un aller-retour `infer` via le Worker, dans Chromium 156 sur
un AMD Ryzen 7 8845HS, `synthetic-19` et `moka-9` mesurés dans la même
session :

| Modèle | Plateau | Moka | Ce moteur | Rapport | Tronc seul |
| --- | --- | --- | --- | --- | --- |
| `synthetic-19` | 19×19 | 26,215 ms | 1,130 ms | 23,20 | 0,985 ms |
| `moka-9` | 9×9 | 5,655 ms | 0,265 ms | 21,34 | 0,225 ms |

Le rapport est du même ordre sur les deux tailles. Le tronc suit la surface du
plateau (×4,46). Les têtes pèsent plus en 19×19 : `policy.linear` y a 20 fois
plus de poids qu'en 9×9.

Limites :

*   les têtes de `synthetic-19` sont générées : la mesure porte sur le coût
    du calcul, pas sur la force de jeu, et aucun accord sur le meilleur coup
    n'est vérifié en 19×19 ;
*   la comparaison porte sur le runtime TypeScript de Moka, et non sur tout
    moteur JavaScript ;
*   ces temps ne valent que pour cette machine et ce navigateur ;
*   `synthetic-19` sert au banc et aux tests ; `dist/` ne contient que le
    9×9.

Le détail, dont la part des têtes, est dans « Mesures enregistrées » de
[`docs/designs/zig-wasm-nxn.md`](docs/designs/zig-wasm-nxn.md).

## Installation et commandes

Les versions de Zig, Node et pnpm sont fixées dans `mise.toml`.

```sh
mise install
pnpm install
pnpm setup:browser   # installe Chromium pour Playwright
pnpm oracle          # données de référence, générées avec le code de Moka
pnpm build           # infer.wasm, infer-safe.wasm et les bundles JS
```

| Commande | Rôle |
| --- | --- |
| `pnpm build` | construit `dist/worker.js` et `dist/infer.wasm` |
| `pnpm oracle` | génère les données de référence avec le code de Moka |
| `pnpm oracle:check` | vérifie que les données de référence se régénèrent à l'identique |
| `pnpm calibrate` | recalcule `oracle/tolerance.json` (`moka-9` seulement) |
| `pnpm models` | régénère `models/synthetic-19/` à partir des poids de `moka-9` |
| `pnpm models:check` | vérifie que `models/synthetic-19/` se régénère à l'identique |
| `pnpm test` | tests Zig, puis tests Node sur les deux builds WASM |
| `pnpm test:e2e` | tests Playwright du Worker dans Chromium |
| `pnpm demo` | construit puis sert la page de démo sur <http://127.0.0.1:4173/demo/index.html> |
| `pnpm bench` | mesure Moka et ce moteur dans Chromium ; `--model <id>` choisit le modèle (`moka-9` par défaut), `--verdict` calcule G4 |
| `pnpm check:all` | enchaîne `typecheck`, `gen`, `build`, `models:check`, `oracle`, `oracle:check`, `test` et `test:e2e` |

`pnpm test` et `pnpm test:e2e` ont besoin des données de référence. L'ordre
compte : `pnpm build` copie dans `bench/dist/` les positions et les sorties
attendues que lisent le banc et les tests e2e, et ne le fait que si
`pnpm oracle` a déjà tourné.

## Démo

`pnpm demo` sert une page où l'on joue noir en 9×9 contre ce moteur
(`demo/`). Blanc joue le coup légal le mieux noté par le moteur Zig, chargé
depuis `dist/` comme dans une intégration réelle. La page utilise les règles,
l'encodage et le comptage de Moka (`vendor/moka/src/game.ts`). À chaque coup
de blanc, elle envoie aussi la position au Worker de Moka, et affiche les deux
temps d'aller-retour et si les deux moteurs choisissent le même coup. La case
« Zig plays both sides » fait jouer le moteur des deux côtés jusqu'au
comptage.

Les temps de la page sont plus élevés que ceux du banc : chaque appel arrive
après une attente, sur un Worker qui doit se réveiller, et les premiers appels
comprennent la compilation JIT. Les chiffres de référence sont ceux de
`pnpm bench`. La force de jeu est celle de Moka sans recherche : le moteur
choisit le même coup que Moka, plus vite.

## Match avec recherche

Sans recherche, les deux moteurs jouent la même partie, puisqu'ils choisissent
le même coup. La vitesse ne compte qu'avec une recherche. `match/search.ts` est
une recherche PUCT en TypeScript, sur les règles de Moka, qui évalue les
positions avec le moteur qu'on lui donne et s'arrête après un temps fixe par
coup. `match/games.ts` joue les parties : chaque ouverture de 4 coups est tirée
de la politique du réseau, puis jouée deux fois en échangeant les couleurs.

- `pnpm match` lance le match en Node, sur plusieurs threads
  (`--first`, `--second`, `--first-model`, `--second-model`, `--games`,
  `--time`, `--first-time`, `--second-time`, `--max-visits`, `--batch`,
  `--parallel`). Les joueurs sont `zig-reuse`, `zig-search`, `zig-puct`,
  `moka-puct`, `zig-policy` et `moka-policy` (`policy` : le coup le mieux
  noté, sans recherche). Chaque joueur prend son modèle 9×9 dans `zig-out/`
  (`moka-9` par défaut, ou `moka-9-v2`). `--first-time` et `--second-time`
  donnent à un joueur un autre temps par coup que `--time`, pour mesurer un
  écart de force en handicap de temps. Le résultat va dans
  `bench/results/match-*.json`.
- Le joueur `katago` (Node seulement) lance KataGo en GTP avec un nombre fixe
  de visites par coup (`--katago-visits`, `--katago-binary`,
  `--katago-model`, ou `KATAGO_BINARY` et `KATAGO_MODEL`), sous les règles de
  Moka (ko simple, score par zone, suicide interdit, komi 7), avec
  `match/katago-gtp.cfg`. Avec `--katago-human-model` (le réseau human SL
  `b18c384nbt-humanv0`) et `--katago-profile rank_5k` (de `rank_20k` à
  `rank_9d`), KataGo imite un joueur humain de ce rang avec
  `match/katago-human.cfg` : le coup vient de la politique humaine brute, et
  16 visites du réseau normal décident seulement quand passer. KataGo n'est pas fourni ; le processus hérite de
  l'environnement, donc avec le build OpenCL sur un GPU AMD intégré via Mesa
  il faut `RUSTICL_ENABLE=radeonsi` (paquet `mesa-opencl-icd`). En fin de partie, le banc
  compte les parties où le score de KataGo donne un autre vainqueur que celui
  de Moka.
- `demo/match.html`, servie par `pnpm demo`, lance le même match dans le
  navigateur : une partie par Web Worker, un petit goban par partie, et les
  statistiques mises à jour à chaque coup et à chaque résultat.

Résultats en Node, 100 parties, 100 ms par coup (AMD Ryzen 7 8845HS) :

| Match | Résultat du premier | Score [IC 95 %] | Elo [IC 95 %] | Visites par coup |
|---|---|---|---|---|
| Zig + recherche contre Moka + recherche | 93 V, 5 D, 2 N | 94,0 % [89,3 ; 98,7] | +478 [+369 ; +746] | 381 contre 15 |
| Zig + recherche contre Moka, coup le mieux noté | 99 V, 1 D | 99,0 % [97,0 ; 100] | +798 [+607 ; ∞] | 381 contre 1 |
| Moka + recherche contre Moka, coup le mieux noté | 71 V, 21 D, 8 N | 75,0 % [66,5 ; 83,5] | +191 [+119 ; +282] | 18 contre 1 |
| Contrôle : 16 visites de chaque côté | 47 V, 47 D, 6 N | 50,0 % [40,2 ; 59,8] | +0 [−69 ; +69] | 16 contre 16 |

Le contrôle montre que le banc ne favorise aucun côté. Ces Elo décrivent cette
recherche simple (sans réutilisation de l'arbre ni lots) sur 50 ouvertures ;
avec plus de temps par coup, l'écart devrait se réduire.

### Recherche dans `search.wasm`

`zig-search` fait la même recherche dans `zig-out/<id>/search.wasm`, qui
contient le réseau, les règles de Moka portées en Zig (`src/rules/rules.zig`)
et un arbre de taille fixe (`src/search/tree.zig`) ; `js/search-engine.ts` le
pilote. Pour un même nombre de visites, il visite exactement comme
`match/search.ts` (`tests/search-parity.test.ts`). Le design est dans
[`docs/designs/zig-search.md`](docs/designs/zig-search.md).

| | `zig-puct` | `zig-search` |
|---|---|---|
| Visites en 100 ms, 9×9, un thread | 331 | 477 (+44 %) |
| Visites en 300 ms, 19×19 (`synthetic-19`), un thread | 205 | 263 (+28 %) |
| Match de 200 parties à 100 ms, score de `zig-search` | | 54,3 % [47,3 ; 61,2], Elo +30 [−18 ; +79] |
| Match de 200 parties à 10 ms, score de `zig-search` | | 57,8 % [50,9 ; 64,6], Elo +54 [+6 ; +104] |

Évaluer plusieurs feuilles par appel (`--batch`) ne fait gagner aucune visite :
le réseau traite les positions d'un lot l'une après l'autre.

`zig-reuse` garde en plus, d'un coup à l'autre, le sous-arbre de la nouvelle
position. Contre `zig-search`, 200 parties à 100 ms par coup : 60,0 %
[53,2 ; 66,8], Elo +70 [+22 ; +121] avec `moka-9-v2`, et 60,3 %
[53,5 ; 67,0], Elo +72 [+24 ; +123] avec `moka-9`.

### Moka v2

`moka-9-v2` est le modèle de la PR
[millionco/moka#3](https://github.com/millionco/moka/pull/3), non fusionnée,
copié dans `vendor/moka-v2/` : tronc de 48 canaux, 16 blocs, environ 3 fois
le calcul de v1. Il passe les mêmes tests de parité que `moka-9`, mais
`dist/` reste en v1. Avec des tuiles de 3 positions × 12 sorties dans les
noyaux, une inférence via le Worker passe de 0,815 à 0,660 ms dans Chromium
(Moka : 15,0 ms). Matchs de 200 parties à 100 ms par coup :

| Match | Score du premier [IC 95 %] | Elo [IC 95 %] |
|---|---|---|
| `zig-search` v2 contre `zig-search` v1 | 81,0 % [75,6 ; 86,4] | +252 [+196 ; +322] |
| `zig-search` v1 contre `moka-puct` v2 | 68,3 % [61,8 ; 74,7] | +133 [+84 ; +188] |
| `zig-search` v2 contre `moka-puct` v2 | 92,3 % [88,5 ; 96,0] | +430 [+355 ; +550] |
| `moka-puct` v2 contre `moka-puct` v1 | 65,5 % [58,9 ; 72,1] | +111 [+63 ; +165] |

Le détail est dans [`docs/designs/moka-v2.md`](docs/designs/moka-v2.md).

### Force de Zig v2

`zig-reuse` avec `moka-9-v2` gagne 92,5 % de ses parties contre `moka-puct`
v1 à 100 ms par coup, et vaut encore Moka v1 avec 2 ms par coup contre 100
(54,8 % [47,9 ; 61,6]). Contre KataGo, Zig v2 à 100 ms ne gagne aucune
partie contre le réseau b10 sans recherche (8 % de score, avec les nulles),
et joue à égalité avec l'imitation human SL `rank_3d` en 9×9 (51 %, arbitre
KataGo), un rang qui surestime sans doute sa force. Le détail et les limites
sont dans [`docs/designs/force-zig-v2.md`](docs/designs/force-zig-v2.md).

### Vidéo

`video/` est un projet [Remotion](https://www.remotion.dev/), avec ses propres
dépendances, qui rejoue un match enregistré : 10 parties en parallèle, un
goban par partie, et les statistiques qui évoluent à chaque coup et à chaque
résultat. Remotion rend la vidéo image par image, sans tenir le temps réel :
la vidéo rejoue donc les coups et les temps d'un vrai match au lieu de lancer
la recherche pendant le rendu.

```sh
pnpm match --games 100 --parallel 10 --time 100
cd video
pnpm install
pnpm data      # dernier enregistrement de bench/results vers src/match-data.json
pnpm render    # out/zig-match.mp4
```

Avec 10 parties à la fois sur 8 cœurs, chaque recherche a moins de CPU : Zig
fait environ 264 visites par coup et Moka 9,4 dans l'enregistrement fourni
(94 V, 1 D, 5 N pour Zig).

Remotion n'est pas sous licence MIT : son usage est gratuit pour les
particuliers et les entreprises de 3 personnes au plus, et demande une licence
entreprise au-delà.

## Intégration

Le point de départ est une page qui suit le
[README de Moka](vendor/moka/README.md#embed-moka-on-a-site). Servez
`dist/worker.js` et `dist/infer.wasm` dans un même dossier, sur l'origine de la
page, puis changez seulement l'URL du Worker. Voici l'exemple de Moka, où
seule cette URL change :

```js
import {
  GoModelWorkerClient,
  createGameState,
  encodeStudentFeatures,
  getAutomaticallyDeadMoves,
  getAreaScore,
  playMove,
  removeDeadStones,
  selectHighestLegalMove,
} from "/moka/index.js";

const mokaWorker = new Worker("/go-network/worker.js", {
  type: "module",
});
const moka = new GoModelWorkerClient(mokaWorker);

await moka.initialize({
  manifestUrl: "/moka/go-model.json",
  weightsUrl: "/moka/go-model.bin",
});

let gameState = createGameState();
```

La suite (`moka.infer`, choix du coup, fin de partie) est celle du README de
Moka, sans changement.

`worker.js` charge `infer.wasm` depuis son propre dossier
(`new URL("./infer.wasm", import.meta.url)`). Aucun type MIME particulier n'est
exigé pour le `.wasm`.

### Content Security Policy

Chromium applique au Worker la CSP envoyée avec la réponse de `worker.js`, et
non celle de la page. Les tests e2e le vérifient : avec
`script-src 'self'` sur la page seule, la page ne peut pas compiler de WASM,
mais le Worker le peut. Si le serveur envoie une CSP avec `worker.js`, elle
doit autoriser la compilation :

```text
Content-Security-Policy: script-src 'self' 'wasm-unsafe-eval'
```

La CSP de la page doit par ailleurs autoriser l'URL du Worker (`worker-src`,
ou à défaut `script-src`), comme pour Moka.

### Repli sur le Worker de Moka

Sans `simd128`, ou si la CSP de `worker.js` bloque la compilation, `initialize`
est rejeté avec un message qui nomme la cause. L'API étant la même, la page
peut alors créer le Worker de Moka à la place :

```js
const MODEL = { manifestUrl: "/moka/go-model.json", weightsUrl: "/moka/go-model.bin" };

const createClient = async () => {
  const zig = new GoModelWorkerClient(new Worker("/go-network/worker.js", { type: "module" }));

  try {
    await zig.initialize(MODEL);
    return zig;
  } catch (error) {
    console.warn(`Zig engine unavailable, falling back to Moka: ${error.message}`);
    zig.dispose();
  }

  const moka = new GoModelWorkerClient(new Worker("/moka/worker.js", { type: "module" }));
  await moka.initialize(MODEL);
  return moka;
};
```

### Erreurs

| Cause | Message |
| --- | --- |
| pas de `simd128` | `WebAssembly SIMD is not supported in this browser.` |
| compilation bloquée par la CSP | `WebAssembly compilation was blocked; the Content Security Policy of worker.js must allow 'wasm-unsafe-eval'.` |
| `infer.wasm` introuvable | `Failed to load infer.wasm.` |
| `.bin` de mauvaise longueur | `Model weights have the wrong length (expected 113648 bytes).` |
| `.bin` d'un autre build | `Model weights do not match this engine build (expected sha256 35c1e51c…).` |
| entrée de mauvaise longueur | `Input features must have 972 values.` |
| sortie `NaN` ou infinie | `Model produced a non-finite output.` |
| `infer` sans `initialize` réussi | `Go model worker is not initialized.` |

Un `initialize` qui échoue abandonne les poids précédents : les `infer`
suivants échouent jusqu'au prochain `initialize` réussi. Moka, lui, garde
l'ancien modèle.

## Quand Moka publie de nouveaux poids

Le sha256 et la disposition des poids sont figés dans `infer.wasm` au build :
un `worker.js` existant refuse un autre `.bin`. Pour suivre de nouveaux poids :

1.  copier la nouvelle version de Moka dans `vendor/moka/`, et mettre à jour
    le commit et les empreintes de `vendor/moka/VENDORED.md` ;
2.  `pnpm models`, puisque le tronc de `synthetic-19` vient des poids de
    `moka-9` ;
3.  `pnpm gen`, qui contrôle le manifeste et refuse un format non pris en
    charge (`int4`, échelles groupées, blocs standard) ;
4.  `pnpm oracle`, `pnpm calibrate`, puis `pnpm build`, qui copie les
    nouvelles données de référence dans `bench/dist/` ;
5.  `pnpm test` et `pnpm test:e2e` ;
6.  déployer ensemble `worker.js`, `infer.wasm` et les nouveaux
    `go-model.json` et `go-model.bin`.

## Support navigateur

Le moteur demande WebAssembly SIMD (`simd128`), présent dans Chrome et Edge 91,
Firefox 89 et Safari 16.4. Il n'est mesuré et testé que dans Chromium sur
ordinateur ; les autres navigateurs et le mobile ne sont pas vérifiés.

## Zéro allocation : ce qui est garanti

*   `infer.wasm` ne contient aucun allocateur. Tous ses tampons sont
    statiques, et sa mémoire linéaire est fixée, avec une taille initiale
    égale à la taille maximale : 13 pages (832 Kio, dont 64 Kio de pile) pour
    `moka-9`, 66 pages (4 224 Kio) pour `synthetic-19`.
*   Le binaire ne contient aucune instruction `memory.grow`, et la taille de
    la mémoire ne change pas sur 1 000 inférences. Les deux points sont
    vérifiés par `tests/wasm-checks.test.mjs` et
    `tests/infer-parity.test.mjs`.
*   La garantie ne couvre que la mémoire WASM. Côté JavaScript, chaque
    `infer` crée un `Float32Array` de logits (82 pour `moka-9`, 362 pour
    `synthetic-19`), et les messages du Worker
    sont alloués par le navigateur, comme dans Moka.

## Attribution

Ce projet réutilise le modèle, le client et le code de référence de
[Moka](https://github.com/millionco/moka), publié par Million Software, Inc.
sous licence MIT. Une copie en lecture seule se trouve dans `vendor/moka/`, et
le modèle Moka v2 dans `vendor/moka-v2/` ;
voir [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Ce projet est
lui-même sous licence MIT ([`LICENSE`](LICENSE)).
