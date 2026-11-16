# Ablation des bandes Sentinel-2 (vigueur et stress hydrique), 09/11/2026

Question du sujet : qu'apporte chaque bande ? Je refais les cartes de `run.py` en changeant une seule chose à la fois (l'indice, ou la bande), avec les mêmes règles et les mêmes masques (parcelles, nuages). Script : `ablation.py`. Chiffres : `ablation.csv` et `saisons.csv`. Cartes : `planche_cameroun.png`, `planche_equateur.png` (rouge = pixel signalé).

Il n'y a pas de vérité terrain. Je mesure ce que change chaque bande, pas laquelle a raison. Pas de règle de choix : les chiffres sont côte à côte.

Mesures : part de la plantation signalée, parcelles en alerte (note sous 70, comme le bulletin), et accord avec la carte actuelle (Jaccard sur les pixels signalés : 1 = mêmes pixels, 0 = aucun en commun).

## Vigueur (règle : indice sous 80 % du 75e centile des parcelles)

| Variante | Cameroun 16/12/2025 | Cameroun 20/04/2026 | Équateur 31/03 et 22/09/2026 |
|---|---|---|---|
| NDVI B08/B04 (actuel) | 33,7 %, parcelles 3 4 5 6 13 | 12,2 %, parcelles 2 11 | 0 % et 0 % |
| NDRE B8A/B05 (bord du rouge) | 42,6 %, accord 0,79, parcelles 3 4 5 6 10 13 14 | 11,4 %, accord 0,63, parcelle 2 | 0 % et 0 % |
| GNDVI B08/B03 (vert au lieu du rouge) | 31,1 %, accord 0,92, mêmes parcelles | 2,1 %, accord 0,17, aucune | 0 % et 0 % |
| VARI B02/B03/B04 (sans infrarouge) | 65,3 %, accord 0,52, 11 parcelles | 63,8 %, accord 0,19, 10 parcelles | 26,8 % et 17,7 % |
| ExG B02/B03/B04 (sans infrarouge) | 51,0 %, accord 0,65, 10 parcelles | 57,0 %, accord 0,21, 9 parcelles | 13,0 % et 11,3 % |

Ce que je lis :
- Le **proche infrarouge (B08)** porte l'information. Sans lui (VARI, ExG), la moitié du Cameroun est signalée, et 11 à 27 % de l'Équateur où les trois indices infrarouges ne voient rien. Les cartes visibles réagissent à autre chose que le feuillage (en avril, tout le bloc du milieu du Cameroun est rouge). Sans terrain, je ne peux pas dire à quoi.
- Le **bord du rouge (B05)** à la place du rouge (B04) retrouve en gros les mêmes zones (accord 0,79 puis 0,63). Il en signale un peu plus en décembre (parcelles 10 et 14 en plus).
- Le **vert (B03)** à la place du rouge colle très bien en décembre (0,92) mais presque plus en avril (0,17, 2,1 % signalé). C'est instable d'une date à l'autre.

## Stress hydrique (règle : NDMI plus bas que prévu par le feuillage, écart sous -0,05)

| Variante | Cameroun 16/12/2025 | Cameroun 20/04/2026 | Équateur 31/03 et 22/09/2026 |
|---|---|---|---|
| NDMI B8A/B11 corrigé par le NDVI (actuel) | 0,8 %, aucune parcelle | 7,6 %, parcelle 14 | 0,1 % et 0 % |
| B12 au lieu de B11 | 1,0 %, accord 0,39 | 3,1 %, accord 0,31, aucune | 0,3 % et 0,2 % |
| Correction par le NDRE au lieu du NDVI | 0,6 %, accord 0,29 | 1,3 %, accord 0,15, aucune | 0 % et 0 % |
| NDMI sans correction (règle des 80 %) | 36,8 %, accord 0,02, 7 parcelles | 29,5 %, accord 0,16, 6 parcelles | 0 % et 0,5 % |

Ce que je lis :
- **B11 ou B12** : la carte garde la même taille (petite), mais les pixels changent (accord 0,31 à 0,39). Au pixel près, la carte de stress est fragile. Pour la comparaison des saisons, B11 et B12 signalent exactement les mêmes parcelles (Équateur : 9 ; Cameroun : 10 et 14).
- **La correction par le feuillage** compte beaucoup. Sans elle, le NDMI signale 30 à 37 % du Cameroun, par blocs entiers. Ça ressemble à une différence de couvert entre blocs plutôt qu'à un manque d'eau. Corriger avec le NDRE au lieu du NDVI réduit encore la carte (accord 0,15 à 0,29).
- La seule parcelle en alerte « eau » avec la règle actuelle (14, en avril) disparaît avec toutes les variantes. Je la présenterais comme un signal faible.

## Limites

- Deux sites, deux dates chacun, pas de mesure au sol : c'est un accord entre cartes, pas une précision.
- Quand la carte actuelle signale presque rien (Équateur), l'accord de Jaccard ne veut pas dire grand-chose.
- La règle de vigueur est relative à la plantation (75e centile) : une plantation homogène, comme l'Équateur, ne montre presque rien, quelle que soit la bande.
