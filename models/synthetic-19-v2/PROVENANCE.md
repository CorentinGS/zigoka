# Provenance de `synthetic-19-v2`

`synthetic-19-v2` est un modèle 19×19 construit pour mesurer ce que coûte
le tronc de Moka v2 en 19×19 dans le moteur Zig, avant d'en entraîner un
(`docs/designs/moka-19.md`). Il ne joue pas au Go : ses têtes sont
générées, pas entraînées. Il a la forme exacte d'un Moka v2 19×19 entraîné
par `pipeline/` : mêmes tenseurs, même fichier de 776 368 octets.

*   **Tronc :** les poids de Moka v2 (`vendor/moka-v2/model/`, commit
    `1ad518a3300cf14885293f1c8cfe4d840b975136` de la branche `moka-v2` de
    <https://github.com/millionco/moka>), copiés octet pour octet : 48
    canaux, goulot de 24, 16 blocs résiduels.
*   **Têtes générées :** `policy.linear.weight`, `policy.linear.bias` et
    `value.hidden.weight`, par la recette de `synthetic-19`
    (`docs/designs/zig-wasm-nxn.md`, « Modèle synthétique »).
*   **Graine :** 19 (`oracle/mulberry32.ts`).
*   **Constantes d'échelle :** `POLICY_SCALE_FACTOR = 1`,
    `VALUE_SCALE_FACTOR = 1`.
*   **Licence :** les poids du tronc viennent de Moka v2, sous licence MIT
    (Copyright (c) 2026 Million Software, Inc.), reproduite dans
    `vendor/moka-v2/LICENSE`. Voir aussi `THIRD_PARTY_NOTICES.md`.

## Régénération

```sh
pnpm models        # réécrit synthetic-19 et synthetic-19-v2
pnpm models:check  # régénère dans un dossier temporaire et compare octet pour octet
```

Ce fichier est écrit à la main ; `pnpm models:check` ne le compare pas.
