"""Comptage des pieds sur une image drone, parcelle par parcelle.

Pour chaque parcelle couverte par l'image : la zone de la parcelle (plus 2 m de marge) est reprojetée
en mètres (UTM) à 5 cm par pixel, ce qu'attend le modèle ; tout ce qui est hors de la parcelle est mis
en noir, comme pour les fermes ReforesTree à l'apprentissage ; puis le détecteur trouve les pieds,
qui sont replacés sur la carte.
"""

import time

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.features import geometry_mask, shapes
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from scipy.spatial import cKDTree
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union

from comptage import detecteur

MARGE_M = 2.0
CASE_SURFACE = 20          # surface couverte calculée par cases de 20 x 20 pixels (1 m)
RAYON_DOUBLON_M = 1.0      # deux détections d'images différentes à moins de 1 m = le même pied vu deux fois
ECARTEMENT_NORMAL_M = (1.3, 4.5)   # écartement médian attendu entre pieds voisins ; en dehors, l'échelle est douteuse


def _surface_couverte(valide, t):
    """Contour (en mètres) des pixels qui ont une donnée, par cases de 1 m : sert à réunir les surfaces de
    plusieurs images qui se recouvrent, au lieu de les additionner."""
    h, w = valide.shape
    H, W = -(-h // CASE_SURFACE) * CASE_SURFACE, -(-w // CASE_SURFACE) * CASE_SURFACE
    v = np.zeros((H, W), bool)
    v[:h, :w] = valide
    cases = v.reshape(H // CASE_SURFACE, CASE_SURFACE, W // CASE_SURFACE, CASE_SURFACE).mean((1, 3)) >= 0.5
    if not cases.any():
        return None
    t_cases = t * t.scale(CASE_SURFACE, CASE_SURFACE)
    return unary_union([shape(g) for g, val in shapes(cases.astype(np.uint8), mask=cases, transform=t_cases) if val])


def fusionner_doublons(pieds_gdf, crs_metrique, rayon=RAYON_DOUBLON_M):
    """Quand deux images se recouvrent, un pied de la zone commune est détecté deux fois. Deux détections venues
    d'images différentes et à moins de `rayon` mètres sont fusionnées : on garde la plus sûre. Les détections d'une
    même image ne sont jamais fusionnées (le détecteur sépare déjà ses pics de 1,4 m)."""
    if len(pieds_gdf) < 2 or pieds_gdf["image"].nunique() < 2:
        return pieds_gdf
    g = pieds_gdf.to_crs(crs_metrique)
    xy = np.c_[g.geometry.x.to_numpy(), g.geometry.y.to_numpy()]
    images, arbre = g["image"].to_numpy(), cKDTree(xy)
    pris, garde = np.zeros(len(g), bool), np.zeros(len(g), bool)
    for k in np.argsort(-g["score"].to_numpy(), kind="stable"):
        if pris[k]:
            continue
        garde[k] = pris[k] = True
        for j in arbre.query_ball_point(xy[k], rayon):
            if images[j] != images[k]:
                pris[j] = True
    return pieds_gdf[garde]


def ecartement_median(pieds_gdf, crs_metrique, minimum=20):
    """Distance médiane (m) entre chaque pied et son plus proche voisin, image par image (au moins `minimum`
    pieds). Une valeur loin de l'écartement d'une bananeraie (2 à 3 m) signale une échelle d'image douteuse."""
    resultat = {}
    g = pieds_gdf.to_crs(crs_metrique)
    for image, d in g.groupby("image"):
        if len(d) >= minimum:
            xy = np.c_[d.geometry.x.to_numpy(), d.geometry.y.to_numpy()]
            resultat[image] = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
    return resultat


def _bandes_rgb(src):
    """Numéros des bandes rouge, verte, bleue (d'après l'image, sinon 1, 2, 3)."""
    ordre = {c: i + 1 for i, c in enumerate(src.colorinterp)}
    if all(c in ordre for c in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)):
        return [ordre[ColorInterp.red], ordre[ColorInterp.green], ordre[ColorInterp.blue]]
    return [1, 2, 3]


def _en_8_bits(rgb, valide):
    """Image en 0-255. Une image qui n'est pas en 8 bits est étirée entre ses centiles 0,5 et 99,5 %."""
    if rgb.dtype == np.uint8:
        return rgb
    sortie = np.zeros(rgb.shape, np.uint8)
    for k in range(3):
        v = rgb[k][valide]
        if v.size:
            bas, haut = np.percentile(v, [0.5, 99.5])
            sortie[k] = np.clip((rgb[k] - bas) / max(haut - bas, 1e-9) * 255, 0, 255).astype(np.uint8)
    return sortie


def compter(chemin, parcelles, reseaux):
    """Pieds trouvés dans les parcelles couvertes par l'image : GeoDataFrame en EPSG:4326
    (colonnes id_parcelle, score, image), et {id de parcelle: surface couverte}, en contour (mètres, UTM de
    parcelles.estimate_utm_crs()). La surface couverte ne compte que les pixels qui ont une donnée : une
    orthomosaïque a souvent des bords sans image (transparents ou noirs) à l'intérieur de son rectangle."""
    utm = parcelles.estimate_utm_crs()
    p_utm = parcelles.to_crs(utm)
    pieds, surfaces, debut = [], {}, time.time()
    with rasterio.open(chemin) as src:
        bandes = _bandes_rgb(src)
        emprise = box(*transform_bounds(src.crs, utm, *src.bounds))
        if src.dtypes[0] != "uint8":
            print(f"    {chemin.name} n'est pas en 8 bits ({src.dtypes[0]}) : couleurs étirées pour le comptage")
        for _, parcelle in p_utm.iterrows():
            zone = parcelle.geometry.intersection(emprise)
            if zone.is_empty:
                continue
            x0, y0, x1, y1 = zone.buffer(MARGE_M).bounds
            largeur = int(np.ceil((x1 - x0) / detecteur.PIXEL_M))
            hauteur = int(np.ceil((y1 - y0) / detecteur.PIXEL_M))
            t = from_origin(x0, y1, detecteur.PIXEL_M, detecteur.PIXEL_M)
            with WarpedVRT(src, crs=utm, transform=t, width=largeur, height=hauteur,
                           resampling=Resampling.lanczos) as vrt:
                rgb = vrt.read(bandes)
                valide = vrt.read_masks(1) > 0
            dans = geometry_mask([zone], out_shape=(hauteur, largeur), transform=t, invert=True)
            valide &= dans & (rgb.sum(0) > 0)                  # un pixel tout noir est un pixel sans image
            image = _en_8_bits(rgb, valide).transpose(1, 2, 0) * valide[..., None]
            xy, scores = detecteur.detecter(reseaux, np.ascontiguousarray(image))
            for (x, y), s in zip(xy, scores):
                X, Y = t * (float(x), float(y))
                if zone.contains(Point(X, Y)):
                    pieds.append({"id_parcelle": parcelle["id"], "score": round(float(s), 3),
                                  "image": chemin.name, "geometry": Point(X, Y)})
            couverte = _surface_couverte(valide, t)
            if couverte is not None:
                couverte = couverte.intersection(zone)
                if not couverte.is_empty:
                    surfaces[parcelle["id"]] = couverte if parcelle["id"] not in surfaces else surfaces[parcelle["id"]].union(couverte)
    duree = time.time() - debut
    surface_ha = sum(s.area for s in surfaces.values()) / 1e4
    if surface_ha:
        print(f"    {chemin.name} : {len(pieds)} pieds sur {surface_ha:.2f} ha, "
              f"{duree:.0f} s ({duree / surface_ha:.0f} s par hectare)")
    gdf = gpd.GeoDataFrame(pieds, columns=["id_parcelle", "score", "image", "geometry"], geometry="geometry", crs=utm)
    return gdf.to_crs("EPSG:4326"), surfaces
