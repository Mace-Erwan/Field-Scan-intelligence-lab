"""Vigueur : indices de végétation et zones anormalement faibles.

Sentinel-2 (multispectral) : NDVI (quantité de feuillage vert) et NDRE (chlorophylle).
Image drone RGB : ExG et VARI, calculés seulement à partir du rouge, du vert et du bleu. Pour juger
les zones, l'ExG sépare végétation et sol (seuil d'Otsu, pixels de 50 cm), puis je calcule la part
de végétation de chaque case de 10 m : c'est elle qui joue le rôle du NDVI (un VARI ou un ExG moyen
mélange sol et feuilles, et ses valeurs proches de 0 rendent la règle des 80 % instable).

Règle des zones faibles, la même pour tous les indices : une zone est anormalement faible quand son
indice est sous 80 % du niveau normal de la plantation, c'est-à-dire du 75e centile des pixels de
toutes les parcelles (le niveau atteint par le quart le plus vert). Elle est relative à la plantation
analysée, et non un seuil fixe, parce que le niveau normal change d'un site, d'une saison et d'un
capteur à l'autre (voir docs/experiences.md, section 3).
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from shapely.geometry import box

from comptage.pieds import _bandes_rgb

FRACTION_NORMALE = 0.80
CENTILE_REFERENCE = 75
TAILLE_CASE_DRONE_M = 0.5        # les indices drone sont calculés sur des cases de 50 cm...
TAILLE_ZONE_M = 10.0             # ... puis moyennés sur des cases de 10 m pour juger les zones, comme un pixel
                                 # Sentinel-2 : à 50 cm, le sol entre deux plants paraîtrait « faible »


def zones_faibles(indice, dans_parcelles):
    """Pixels anormalement faibles (dans les parcelles) et niveau de référence de la plantation."""
    valeurs = indice[dans_parcelles & np.isfinite(indice)]
    if valeurs.size == 0:
        return np.zeros(indice.shape, bool), np.nan
    reference = float(np.percentile(valeurs, CENTILE_REFERENCE))
    with np.errstate(invalid="ignore"):
        faible = dans_parcelles & (indice < FRACTION_NORMALE * reference)
    return faible, reference


def indices_rgb(rgb):
    """ExG et VARI à partir d'un tableau (3, h, w) rouge, vert, bleu."""
    r, v, b = (rgb[k].astype("float32") for k in range(3))
    with np.errstate(divide="ignore", invalid="ignore"):
        somme = r + v + b
        exg = (2 * v - r - b) / somme                  # excès de vert, en couleurs normalisées
        vari = (v - r) / (v + r - b)                    # indice de végétation résistant à l'atmosphère
    exg[~np.isfinite(exg)] = np.nan
    vari[~np.isfinite(vari) | (np.abs(vari) > 1)] = np.nan   # VARI explose quand v + r - b est proche de 0
    return {"exg": exg, "vari": vari}


def cartes_drone(chemin, parcelles):
    """Indices ExG et VARI d'une image drone, sur une grille de 50 cm en mètres (UTM) couvrant les parcelles.
    Renvoie (indices, profil GeoTIFF, pixels valides) ou None si l'image ne couvre aucune parcelle."""
    utm = parcelles.estimate_utm_crs()
    zone = parcelles.to_crs(utm).union_all()
    with rasterio.open(chemin) as src:
        emprise = box(*transform_bounds(src.crs, utm, *src.bounds))
        zone = zone.intersection(emprise)
        if zone.is_empty:
            return None
        x0, y0, x1, y1 = zone.bounds
        c = TAILLE_CASE_DRONE_M
        largeur, hauteur = int(np.ceil((x1 - x0) / c)), int(np.ceil((y1 - y0) / c))
        t = from_origin(x0, y1, c, c)
        with WarpedVRT(src, crs=utm, transform=t, width=largeur, height=hauteur, resampling=Resampling.average) as vrt:
            rgb = vrt.read(_bandes_rgb(src)).astype("float32")
            valide = vrt.read_masks(1) > 0
    indices = indices_rgb(rgb)
    for carte in indices.values():
        carte[~valide] = np.nan
    profil = {"driver": "GTiff", "width": largeur, "height": hauteur, "count": 1, "dtype": "float32",
              "crs": utm, "transform": t, "nodata": np.nan, "compress": "deflate"}
    return indices, profil, valide


def moyenne_par_case(carte, transform, facteur):
    """Moyenne d'une carte sur des cases de facteur x facteur pixels (les NaN sont ignorés).
    Renvoie la carte agrégée et sa transformation géographique."""
    h, w = carte.shape
    H, W = int(np.ceil(h / facteur)), int(np.ceil(w / facteur))
    grand = np.full((H * facteur, W * facteur), np.nan, "float32")
    grand[:h, :w] = carte
    blocs = grand.reshape(H, facteur, W, facteur)
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        moyenne = np.nanmean(blocs, axis=(1, 3))
    from affine import Affine
    return moyenne, transform * Affine.scale(facteur)


def seuil_otsu(valeurs, n=256):
    """Seuil d'Otsu : la valeur qui sépare le mieux deux groupes (ici sol et végétation)."""
    hist, bords = np.histogram(valeurs, bins=n)
    centres = (bords[:-1] + bords[1:]) / 2
    poids1, poids2 = np.cumsum(hist), np.cumsum(hist[::-1])[::-1]
    moy1 = np.cumsum(hist * centres) / np.maximum(poids1, 1)
    moy2 = (np.cumsum((hist * centres)[::-1]) / np.maximum(poids2[::-1], 1))[::-1]
    variance = poids1[:-1] * poids2[1:] * (moy1[:-1] - moy2[1:]) ** 2
    return float(centres[:-1][np.argmax(variance)])


def couverture_vegetale(exg, dans):
    """Carte 0/1 (NaN hors image) : 1 = végétation (ExG au-dessus du seuil d'Otsu calculé dans les parcelles)."""
    valeurs = exg[dans & np.isfinite(exg)]
    seuil = seuil_otsu(valeurs) if valeurs.size else np.nan
    veg = np.where(np.isfinite(exg), (exg > seuil).astype("float32"), np.nan)
    return veg, seuil
