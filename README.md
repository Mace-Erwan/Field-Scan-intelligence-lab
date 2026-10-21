# Field Scan

Analyse d'une bananeraie vue du ciel : à partir d'images aériennes et d'un fichier de parcelles, produire un bulletin par parcelle (vigueur, stress hydrique, maladies des feuilles).

Projet piscine Intelligence Lab, ECE.

## Installation

```
pip install -r requirements.txt
```

## Utilisation

*En cours d'écriture.* Commande prévue :

```
python run.py --images DOSSIER --parcelles parcelles.geojson --out resultats/
```

## Contenu du dépôt

- `docs/carnet_donnees.md` : carnet de données (sources, licences, découpage, risques de fuite, biais).
- `outils/verifier_sentinel2.py` : contrôle des images Sentinel-2 téléchargées (bandes, nuages, NDVI et NDMI médians).

Les données ne sont pas dans le dépôt : leurs sources sont décrites dans le carnet de données.
