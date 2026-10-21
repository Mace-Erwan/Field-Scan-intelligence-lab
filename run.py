"""
Field Scan : analyse d'une bananeraie vue du ciel, parcelle par parcelle.

Version actuelle : vigueur (NDVI) et humidité des feuilles (NDMI) à partir d'images
Sentinel-2. Pour chaque parcelle, le script calcule la moyenne de ces indices.

Utilisation :
    python run.py --images DOSSIER --parcelles parcelles.geojson --out resultats/

DOSSIER contient une image GeoTIFF par bande, avec le nom de la bande dans le nom
du fichier (ex. ..._B04_(Raw).tiff), comme les exports du Copernicus Browser.

Sorties dans le dossier --out :
    ndvi.tif, ndmi.tif     cartes pixel par pixel (GeoTIFF)
    parcelles.csv          tableau : une ligne par parcelle
    parcelles.geojson      parcelles avec leurs valeurs, à ouvrir dans QGIS
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask

# Indices calculés : nom -> (bande a, bande b), indice = (a - b) / (a + b)
INDICES = {
    "ndvi": ("B08", "B04"),   # vigueur : quantité de feuillage vert
    "ndmi": ("B8A", "B11"),   # humidité : eau contenue dans les feuilles
}

# Couleurs de la carte SCL (export coloré du Copernicus Browser) à exclure :
# nuages, ombres de nuages, cirrus et pixels sans donnée.
COULEURS_A_EXCLURE = {
    (0.467, 0.298, 0.043),   # ombre de nuage
    (0.753, 0.753, 0.753),   # nuage (probabilité moyenne)
    (1.0, 1.0, 1.0),         # nuage (probabilité forte)
    (0.392, 0.784, 1.0),     # cirrus
    (0.0, 0.0, 0.0),         # sans donnée
}


def trouver_fichier(dossier, motif):
    """Renvoie le fichier GeoTIFF dont le nom contient le motif, ou None."""
    fichiers = [f for f in dossier.rglob("*")
                if f.suffix.lower() in (".tif", ".tiff") and motif in f.name]
    if len(fichiers) > 1:
        raise ValueError(f"Plusieurs fichiers contiennent '{motif}' dans {dossier}")
    return fichiers[0] if fichiers else None


def lire_bande(dossier, nom):
    fichier = trouver_fichier(dossier, f"_{nom}_")
    if fichier is None:
        return None
    with rasterio.open(fichier) as src:
        return src.read(1).astype("float32")


def masque_nuages(dossier):
    """Renvoie un tableau vrai/faux : vrai = pixel à exclure. None si pas de SCL."""
    fichier = trouver_fichier(dossier, "Scene_classification")
    if fichier is None:
        return None
    with rasterio.open(fichier) as src:
        scl = np.round(src.read().astype("float32"), 3)   # (3 couleurs, hauteur, largeur)
    masque = np.zeros(scl.shape[1:], dtype=bool)
    for r, v, b in COULEURS_A_EXCLURE:
        masque |= (scl[0] == r) & (scl[1] == v) & (scl[2] == b)
    return masque


def indice_normalise(a, b):
    """(a - b) / (a + b), avec NaN là où a + b vaut 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        resultat = (a - b) / (a + b)
    resultat[~np.isfinite(resultat)] = np.nan
    return resultat


def main():
    parser = argparse.ArgumentParser(description="Field Scan : analyse par parcelle")
    parser.add_argument("--images", required=True, type=Path, help="dossier des images")
    parser.add_argument("--parcelles", required=True, type=Path, help="fichier GeoJSON des parcelles")
    parser.add_argument("--out", required=True, type=Path, help="dossier des résultats")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Grille de référence : position, taille des pixels et projection de l'image
    reference = trouver_fichier(args.images, "_B04_")
    if reference is None:
        raise FileNotFoundError(f"Aucune bande B04 dans {args.images}")
    with rasterio.open(reference) as src:
        profil = src.profile
        transform, crs, forme = src.transform, src.crs, src.shape
    print(f"Images : {forme[1]} x {forme[0]} pixels, projection {crs}")

    # 2. Pixels à exclure (nuages, ombres, sans donnée)
    exclus = masque_nuages(args.images)
    if exclus is None:
        print("Pas de carte SCL : aucun pixel exclu.")
        exclus = np.zeros(forme, dtype=bool)
    else:
        print(f"Pixels exclus (nuages, ombres, sans donnée) : {exclus.sum()}")

    # 3. Calcul des indices pixel par pixel, et écriture des cartes
    cartes = {}
    for nom, (a, b) in INDICES.items():
        bande_a, bande_b = lire_bande(args.images, a), lire_bande(args.images, b)
        if bande_a is None or bande_b is None:
            print(f"{nom.upper()} non calculé : bande {a} ou {b} absente.")
            continue
        if bande_a.shape != forme or bande_b.shape != forme:
            raise ValueError(f"Les bandes {a} et {b} n'ont pas la taille de B04.")
        carte = indice_normalise(bande_a, bande_b)
        carte[exclus] = np.nan
        cartes[nom] = carte
        profil.update(count=1, dtype="float32", nodata=np.nan)
        with rasterio.open(args.out / f"{nom}.tif", "w", **profil) as dst:
            dst.write(carte, 1)

    # 4. Moyenne par parcelle
    parcelles = gpd.read_file(args.parcelles).to_crs(crs)
    if "id" not in parcelles.columns:
        parcelles["id"] = range(1, len(parcelles) + 1)
    for i, geometrie in zip(parcelles.index, parcelles.geometry):
        # Un pixel appartient à la parcelle si son centre est dans le polygone
        dedans = geometry_mask([geometrie], out_shape=forme, transform=transform, invert=True)
        valides = dedans & ~exclus
        parcelles.loc[i, "nb_pixels"] = int(valides.sum())
        for nom, carte in cartes.items():
            parcelles.loc[i, f"{nom}_moyen"] = (
                round(float(np.nanmean(carte[valides])), 3) if valides.any() else np.nan
            )
        if not valides.any():
            print(f"Attention : parcelle {parcelles.loc[i, 'id']} sans pixel valide.")
    parcelles["nb_pixels"] = parcelles["nb_pixels"].astype(int)

    # 5. Écriture du tableau et des parcelles avec leurs valeurs
    colonnes = ["id", "nb_pixels"] + [f"{nom}_moyen" for nom in cartes]
    parcelles[colonnes].to_csv(args.out / "parcelles.csv", index=False)
    parcelles[colonnes + ["geometry"]].to_crs("EPSG:4326").to_file(
        args.out / "parcelles.geojson", driver="GeoJSON"
    )

    print(parcelles[colonnes].to_string(index=False))
    print(f"Résultats écrits dans {args.out}")


if __name__ == "__main__":
    main()
