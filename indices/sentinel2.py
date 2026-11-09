"""Lecture des bandes Sentinel-2 d'une date et calcul des indices (une valeur par pixel de 10 m).

Les fichiers sont ceux exportés du Copernicus Browser (réflectance, 32 bits). Les pixels de nuage,
d'ombre de nuage, de cirrus ou sans donnée sont repérés grâce à la carte de classification de scène
(SCL), et exclus. La SCL peut être exportée en couleurs (3 bandes) ou brute (1 bande, numéro de classe) ;
elle est lue sur la grille de la bande B04 (une SCL brute est à 20 m).
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

# Indices normalisés : nom -> (bande a, bande b), indice = (a - b) / (a + b)
INDICES = {
    "ndvi": ("B08", "B04"),   # vigueur : quantité de feuillage vert (proche infrarouge contre rouge)
    "ndre": ("B8A", "B05"),   # vigueur : chlorophylle, plus sensible que le NDVI quand le feuillage est dense (red-edge)
    "ndmi": ("B8A", "B11"),   # eau contenue dans les feuilles (proche infrarouge contre infrarouge moyen)
}

# Couleurs de la carte SCL colorée à exclure : ombre de nuage, nuage moyen, nuage fort, cirrus, sans donnée
COULEURS_A_EXCLURE = {
    (0.467, 0.298, 0.043), (0.753, 0.753, 0.753), (1.0, 1.0, 1.0), (0.392, 0.784, 1.0), (0.0, 0.0, 0.0),
}
# Les mêmes classes dans une SCL brute : 0 sans donnée, 3 ombre de nuage, 8 nuage moyen, 9 nuage fort, 10 cirrus
CLASSES_A_EXCLURE = {0, 3, 8, 9, 10}


def indice_normalise(a, b):
    """(a - b) / (a + b), avec NaN là où a + b vaut 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (a - b) / (a + b)
    r[~np.isfinite(r)] = np.nan
    return r


def lire_date(bandes):
    """bandes : {"B04": chemin, ..., "SCL": chemin}. Renvoie (indices {nom: carte}, profil GeoTIFF, pixels exclus)."""
    if "B04" not in bandes:
        raise FileNotFoundError("bande B04 absente")
    with rasterio.open(bandes["B04"]) as src:
        profil, forme, grille = src.profile.copy(), src.shape, (src.crs, src.transform)
    lues = {}
    for nom, chemin in bandes.items():
        if nom == "SCL":
            continue
        with rasterio.open(chemin) as src:
            if src.shape != forme:
                raise ValueError(f"La bande {nom} n'a pas la taille de B04")
            lues[nom] = src.read(1).astype("float32")
    exclus = np.zeros(forme, bool)
    if "SCL" in bandes:
        with rasterio.open(bandes["SCL"]) as src, WarpedVRT(src, crs=grille[0], transform=grille[1], width=forme[1],
                                                            height=forme[0], resampling=Resampling.nearest) as vrt:
            scl = vrt.read().astype("float32")
        if scl.shape[0] >= 3:                                   # SCL en couleurs
            scl = np.round(scl, 3)
            for r, v, b in COULEURS_A_EXCLURE:
                exclus |= (scl[0] == r) & (scl[1] == v) & (scl[2] == b)
        else:                                                   # SCL brute : un numéro de classe par pixel
            exclus |= np.isin(scl[0], list(CLASSES_A_EXCLURE))
    indices = {}
    for nom, (a, b) in INDICES.items():
        if a in lues and b in lues:
            carte = indice_normalise(lues[a], lues[b])
            carte[exclus] = np.nan
            indices[nom] = carte
    profil.update(count=1, dtype="float32", nodata=np.nan)
    return indices, profil, exclus
