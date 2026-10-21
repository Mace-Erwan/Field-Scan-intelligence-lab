# Field Scan

Analyse d'une bananeraie vue du ciel : à partir d'images aériennes et d'un fichier de parcelles, produire un bulletin par parcelle (vigueur, stress hydrique, maladies des feuilles).

Projet piscine Intelligence Lab, ECE.

## Installation

```
pip install -r requirements.txt
```

## Utilisation

```
python run.py --images DOSSIER --parcelles parcelles.geojson --out resultats/
```

- `DOSSIER` : une image GeoTIFF par bande Sentinel-2, avec le nom de la bande dans le nom du fichier (ex. `..._B04_(Raw).tiff`), comme les exports du Copernicus Browser.
- `parcelles.geojson` : les polygones des parcelles.

Sorties dans `resultats/` :

- `ndvi.tif`, `ndmi.tif` : cartes de vigueur et d'humidité des feuilles, pixel par pixel ;
- `parcelles.csv` : une ligne par parcelle (nombre de pixels, NDVI et NDMI moyens) ;
- `parcelles.geojson` : les parcelles avec leurs valeurs.

Version actuelle : vigueur et humidité seulement. Le comptage des plants, les maladies et le bulletin sont en cours.

## Contenu du dépôt

- `run.py` : la chaîne d'analyse (commande ci-dessus).
- `docs/carnet_donnees.md` : carnet de données (sources, licences, découpage, risques de fuite, biais).
- `outils/verifier_sentinel2.py` : contrôle des images Sentinel-2 téléchargées (bandes, nuages, NDVI et NDMI médians).

Les données ne sont pas dans le dépôt : leurs sources sont décrites dans le carnet de données.
