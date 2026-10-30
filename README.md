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
*   **Taille** : l'`infer.wasm` de `moka-9` fait environ 19 Ko, et 6,7 Ko
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
| `synthetic-19` | 19×19 | 26,245 ms | 1,415 ms | 18,55 | 1,260 ms |
| `moka-9` | 9×9 | 5,665 ms | 0,335 ms | 16,91 | 0,285 ms |

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
| `pnpm bench` | mesure Moka et ce moteur dans Chromium ; `--model <id>` choisit le modèle (`moka-9` par défaut), `--verdict` calcule G4 |
| `pnpm check:all` | enchaîne `typecheck`, `gen`, `build`, `models:check`, `oracle`, `oracle:check`, `test` et `test:e2e` |

`pnpm test` et `pnpm test:e2e` ont besoin des données de référence. L'ordre
compte : `pnpm build` copie dans `bench/dist/` les positions et les sorties
attendues que lisent le banc et les tests e2e, et ne le fait que si
`pnpm oracle` a déjà tourné.

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
sous licence MIT. Une copie en lecture seule se trouve dans `vendor/moka/` ;
voir [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Ce projet est
lui-même sous licence MIT ([`LICENSE`](LICENSE)).
