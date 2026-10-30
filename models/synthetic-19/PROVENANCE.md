# Provenance de `synthetic-19`

`synthetic-19` est un modèle 19×19 construit pour mesurer la vitesse du
moteur Zig contre le runtime de Moka. Il ne joue pas au Go : ses têtes sont
générées, pas entraînées.

*   **Tronc :** les poids de Moka (`vendor/moka/model/`, commit
    `a418039e9d823779e3e6d0ea17a244230abc8379` de
    <https://github.com/millionco/moka>), copiés octet pour octet. Couche
    d'entrée, 12 blocs résiduels, pooling global, convolutions des têtes,
    `value.hidden.bias` et `value.output.*` ne dépendent pas de la taille du
    plateau.
*   **Têtes générées :** `policy.linear.weight`, `policy.linear.bias` et
    `value.hidden.weight`, tirées des lignes de Moka selon la recette de
    `docs/designs/zig-wasm-nxn.md` (« Modèle synthétique »).
*   **Graine :** 19 (`oracle/mulberry32.ts`).
*   **Constantes d'échelle :** `POLICY_SCALE_FACTOR = 1`,
    `VALUE_SCALE_FACTOR = 1`.
*   **Licence :** les poids du tronc viennent de Moka, sous licence MIT
    (Copyright (c) 2026 Million Software, Inc.), reproduite dans
    `vendor/moka/LICENSE`. Voir aussi `THIRD_PARTY_NOTICES.md`.

## Régénération

```sh
pnpm models        # réécrit go-model.json et go-model.bin
pnpm models:check  # régénère dans un dossier temporaire et compare octet pour octet
```

Ce fichier est écrit à la main ; `pnpm models:check` ne le compare pas. Toute
modification d'une constante d'échelle y est notée, avec sa raison.
