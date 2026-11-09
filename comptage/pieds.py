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
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from shapely.geometry import Point, box

from comptage import detecteur

MARGE_M = 2.0


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
    (colonnes id_parcelle, score, image), et {id de parcelle: surface couverte en hectares}.
    La surface couverte ne compte que les pixels qui ont une donnée : une orthomosaïque a souvent des
    bords sans image (transparents ou noirs) à l'intérieur de son rectangle."""
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
            ha = float(valide.sum()) * detecteur.PIXEL_M ** 2 / 1e4
            if ha > 0:
                surfaces[parcelle["id"]] = surfaces.get(parcelle["id"], 0.0) + ha
    duree = time.time() - debut
    surface_ha = sum(surfaces.values())
    if surface_ha:
        print(f"    {chemin.name} : {len(pieds)} pieds sur {surface_ha:.2f} ha, "
              f"{duree:.0f} s ({duree / surface_ha:.0f} s par hectare)")
    gdf = gpd.GeoDataFrame(pieds, columns=["id_parcelle", "score", "image", "geometry"], geometry="geometry", crs=utm)
    return gdf.to_crs("EPSG:4326"), surfaces
