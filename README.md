# Field Scan

Analyse d'une bananeraie vue du ciel : à partir d'images aériennes, de photos de feuilles et d'un fichier de parcelles, produire un bulletin de santé par parcelle (comptage des pieds, vigueur, stress hydrique, maladies des feuilles).

Projet piscine Intelligence Lab, ECE.

## Installation

Python 3.10 ou plus récent.

```
pip install -r requirements.txt
```

Tout tourne sans GPU. Les poids des modèles sont dans `modeles/`.

## Utilisation

```
python run.py --images DOSSIER --parcelles parcelles.geojson --out resultats/
```

`DOSSIER` peut contenir, dans n'importe quels sous-dossiers :

- **des images drone** géoréférencées (GeoTIFF RGB, pixel de moins de 1 m) : comptage des pieds, et vigueur (ExG, VARI) s'il n'y a pas d'image Sentinel-2 ;
- **des bandes Sentinel-2** exportées du Copernicus Browser (un fichier par bande, avec la date et le nom de la bande dans le nom du fichier, par exemple `2025-12-16-00_00_..._B04_(Raw).tiff`, plus la carte `Scene_classification_map`), pour une ou deux dates : vigueur (NDVI, NDRE) et stress hydrique (NDMI, et comparaison des deux saisons) ;
- **un dossier `feuilles`** avec des photos de feuilles de bananier : maladies (saine, cercosporiose (Sigatoka), fusariose à confirmer par une coupe du pseudo-tronc, maladie non identifiée, ou « je ne sais pas »). Une photo qui a une position GPS est rattachée à sa parcelle.

Chaque partie ne tourne que si ses images sont présentes. `parcelles.geojson` contient les polygones des parcelles (une colonne `id`, créée si elle manque).

Sorties dans `resultats/` :

| Fichier | Contenu |
|---|---|
| `bulletin.html` | carte des parcelles colorées par score de santé, pieds détectés, alertes avec leur image source, tableau ; s'ouvre dans un navigateur, sans internet |
| `bulletin.csv` | une ligne par parcelle : priorité, mesures, notes, score de santé, alertes |
| `pieds.geojson` | un point par pied détecté (parcelle, score) |
| `alertes.geojson` | zones et parcelles signalées, avec le message et l'image source |
| `vigueur.tif`, `stress.tif` | NDVI de la date la plus récente ; écart du NDMI à ce qu'on attend pour la quantité de feuillage (négatif = manque d'eau) |
| `ndvi_<date>.tif`, `ndre_<date>.tif`, `ndmi_<date>.tif` | indices Sentinel-2, pour chaque date |
| `exg_<image>.tif`, `vari_<image>.tif` | indices de l'image drone (cases de 50 cm) |
| `feuilles.csv` | réponse du modèle pour chaque photo de feuille |
| `apercus/` | images sources des alertes |

Toutes les cartes s'ouvrent dans QGIS.

## Comment le bulletin est calculé

- **Comptage** : détecteur de centres (ResNet-18 + décodeur), moyenne de 2 modèles sur 4 vues, image remise à 5 cm par pixel, un point par pied. Le seuil (0,25) équilibre fausses alertes et oublis, pour que le nombre de pieds par parcelle soit juste. Détail et résultats : `docs/experiences.md`, section 1.
- **Vigueur** : une zone est faible quand son indice (NDVI, ou part de végétation pour une image drone RGB) est sous 80 % du niveau normal de la plantation (75e centile).
- **Eau** : une zone manque probablement d'eau quand son NDMI est plus bas de plus de 0,05 que ce qu'on attend pour sa quantité de feuillage. Avec deux dates, une parcelle est signalée si son NDMI baisse de plus de 0,05 de plus que les autres entre la saison humide et la saison sèche.
- **Maladies** : moyenne de 3 ConvNeXt-Tiny aux poids DINOv3, photo lue en 384 px, sur 4 vues, avec 4 classes (saine, cercosporiose, fusariose, autre). Malade si 1 − P(saine) ≥ 0,725. Une feuille n'est dite saine que si P(saine) atteint 0,519 ; sinon c'est « je ne sais pas », qui donne une alerte. Un « je ne sais pas » avec 1 − P(saine) ≥ 0,675 devient « maladie non identifiée ». Détail : `docs/experiences.md`, section 2.
- **Score de santé** : moyenne des notes disponibles (vigueur, eau, maladies), chacune sur 100 = part de la parcelle (ou des feuilles) sans problème. Une note sous 70 déclenche une alerte.

## Ce que le système sait faire, et ses limites

- **Comptage** : bananeraies et fermes mélangées vues du dessus, à n'importe quelle résolution entre 2 et 20 cm par pixel. F1 d'environ 0,80 sur des plantations du type appris, d'environ 0,76 sur un type jamais vu (`outils/evaluer_comptage.py`). Sur de jeunes plants, il en oublie à peu près un sur trois. Un rejet collé à sa mère n'est pas compté à part.
- **Maladies** : photos de feuilles prises au sol, de près (pas les images drone). Sur deux jeux jamais vus, peu de feuilles malades passent : 0,9 % des feuilles tachées sont dites saines sur BanSpot (Inde), et 3,6 % au test neuf (Bangladesh). En échange, environ 10 % des feuilles sans Sigatoka de BanSpot reçoivent une alerte, le plus souvent des feuilles tachées, trouées ou déchirées. Le modèle nomme moins souvent la maladie qu'avant et répond plus souvent « maladie non identifiée » : au test neuf, il nomme la cercosporiose sur 30 % des feuilles atteintes et la fusariose sur 47 %. Une Sigatoka légère sur une feuille entière lui échappe encore souvent.
- **Vigueur et stress** : jugés par rapport au reste de la plantation analysée, pas avec des seuils fixes. Sur une ferme mélangée vue par un drone RGB, la vigueur n'est pas calibrée.
- **Temps de calcul** (processeur à 2 cœurs, sans GPU) : environ 1 min 40 par hectare d'image drone pour le comptage, environ 3 s par photo de feuille.

## Contenu du dépôt

- `run.py` : la chaîne d'analyse (commande ci-dessus).
- `entrees.py` : trouve les images drone, les bandes Sentinel-2 et les photos de feuilles ; lit les parcelles.
- `comptage/` : `detecteur.py` (modèle de comptage), `pieds.py` (comptage sur une image géoréférencée), `evaluation.py` (notation à 1 m), `baseline_exg.py` (méthode classique ExG + maxima locaux).
- `maladies/classifieur.py` : modèle maladies (préparation des photos, décision).
- `indices/` : `sentinel2.py` (lecture des bandes, nuages), `vigueur.py`, `stress.py`.
- `bulletin/` : `bulletin.py` (notes, alertes, page HTML), `apercus.py` (images des alertes et fond de carte).
- `modeles/` : poids des modèles (enregistrés en 16 bits) et seuils.
- `docs/carnet_donnees.md` : carnet de données (sources, licences, découpage, risques de fuite, biais).
- `docs/experiences.md` : journal de tous les essais et résultats.
- `outils/` : scripts de préparation et de vérification des données.
- `entrainement/` : notebooks d'entraînement (Kaggle, Colab).

Les données ne sont pas dans le dépôt : leurs sources sont décrites dans le carnet de données.
