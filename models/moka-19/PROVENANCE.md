# Provenance de `moka-19`

`moka-19` est l'élève 19×19 de `docs/designs/moka-19.md` : un Moka v2
(tronc de 48 canaux, goulot de 24, 16 blocs résiduels) distillé de KataGo,
exporté en INT8 au format `million-go-int8`. C'est la graine 2 de la ronde
`moka-19-r1` de `pipeline/scripts/round-19.sh`.

*   **Fichier :** `go-model.bin`, 776 368 octets, sha256
    `92654e1cd63fb42ca977a70f5e911537e6280d0bc315f78353d0ab2bb6220c47`.
*   **Professeur :** le réseau KataGo
    `kata1-b18c384nbt-s9996604416-d4316597426`, interrogé par le moteur
    d'analyse de KataGo 1.18.1 (`pipeline/`, `go-selfplay`).
*   **Données :** 2 779 289 positions d'auto-jeu de KataGo sous les règles de
    Moka : le corpus du pilote (253 468 positions, graine 2) et celui de la
    ronde (2 525 821 positions, 12 221 parties, graines 11 à 20). La
    validation est `pilot-19-val` (graine 4).
*   **Entraînement :** départ à chaud du tronc de Moka v2
    (`vendor/moka-v2/model/`), têtes 19×19 neuves ; lots de 256, bf16, AdamW
    à 1e-3 avec cosinus, weight decay 0,1, valeur du professeur seule
    (`--outcome-mix 0`), 4 époques, graine 2, sur la Radeon (non
    déterministe). Le snapshot INT8 retenu est celui de la meilleure perte de
    validation (2,3962, quatrième époque), meilleure des trois graines.
*   **Mesures** sur `pilot-19-test` (25 129 positions, graine 3), INT8 :
    KL au professeur 0,981, accord top-1 50,6 %, valeur contre celle du
    professeur : MSE 0,250, corrélation 0,630, signe 68,8 %. Sa force en
    partie n'est pas encore mesurée.
*   **Licences :** le tronc de départ vient de Moka v2, sous licence MIT
    (Copyright (c) 2026 Million Software, Inc.), reproduite dans
    `vendor/moka-v2/LICENSE`. Le professeur est un réseau KataGo sous la
    « KataGo Neural Network License », de type MIT, reproduite dans
    `KATAGO-NETWORK-LICENSE.txt`. Elle ne dit pas si un réseau distillé de
    ses sorties en est une copie ; sa notice est jointe quand même. Voir
    aussi `THIRD_PARTY_NOTICES.md`.

## Régénération

L'entraînement sur GPU n'est pas déterministe : une nouvelle ronde donne
un modèle proche, pas le même fichier.

```sh
pipeline/scripts/round-19.sh <id>   # environ 10 h sur une Radeon de 16 Go
```

Ce fichier est écrit à la main.
