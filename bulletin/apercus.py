"""Images du bulletin : fond de carte et aperçus des images sources des alertes (PNG).

Tout est dessiné dans une grille en mètres (UTM) : l'image source (bandes Sentinel-2 ou image drone)
y est reprojetée, puis on trace par-dessus le contour de la parcelle et les zones signalées.
"""

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageOps
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

from comptage.pieds import _bandes_rgb


def grille(bornes, taille_max_px, marge_m=20.0):
    """Grille (transform, largeur, hauteur) couvrant des bornes UTM avec une marge, au plus taille_max_px de côté."""
    x0, y0, x1, y1 = bornes
    x0, y0, x1, y1 = x0 - marge_m, y0 - marge_m, x1 + marge_m, y1 + marge_m
    pas = max(x1 - x0, y1 - y0) / taille_max_px
    largeur, hauteur = int(np.ceil((x1 - x0) / pas)), int(np.ceil((y1 - y0) / pas))
    return from_origin(x0, y1, pas, pas), largeur, hauteur


def _etirer(rgb, valide):
    """Couleurs étirées entre les centiles 1 et 99 des trois bandes ensemble (garde l'équilibre des couleurs)."""
    v = rgb[:, valide]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros(rgb.shape, np.uint8)
    bas, haut = np.percentile(v, [1, 99])
    return np.clip((np.nan_to_num(rgb) - bas) / max(haut - bas, 1e-9) * 255, 0, 255).astype(np.uint8)


def rendre(source, crs, t, largeur, hauteur):
    """Image RGB (PIL) de la source dans la grille. source : chemin d'une image drone, ou
    {"B04": chemin, "B03": chemin, "B02": chemin} pour Sentinel-2 (couleurs naturelles)."""
    if isinstance(source, dict):
        couches, valide = [], np.ones((hauteur, largeur), bool)
        for b in ("B04", "B03", "B02"):
            with rasterio.open(source[b]) as src, WarpedVRT(src, crs=crs, transform=t, width=largeur, height=hauteur,
                                                           resampling=Resampling.nearest) as vrt:
                couches.append(vrt.read(1).astype("float32"))
                valide &= vrt.read_masks(1) > 0
        rgb = np.stack(couches)
    else:
        with rasterio.open(source) as src, WarpedVRT(src, crs=crs, transform=t, width=largeur, height=hauteur,
                                                    resampling=Resampling.average) as vrt:
            rgb = vrt.read(_bandes_rgb(src)).astype("float32")
            valide = vrt.read_masks(1) > 0
    rgb = _etirer(rgb, valide) * valide
    return Image.fromarray(rgb.transpose(1, 2, 0).astype(np.uint8))


def anneaux_px(geometrie, t):
    """Contours extérieurs d'un (multi)polygone en pixels de la grille."""
    inverse = ~t
    polys = getattr(geometrie, "geoms", [geometrie])
    return [[inverse * (x, y) for x, y in p.exterior.coords] for p in polys if p.geom_type == "Polygon"]


def apercu(chemin_png, source, crs, parcelle, zones, couleur_zone, legende):
    """Aperçu d'une alerte : image source autour de la parcelle, parcelle en blanc, zones signalées en couleur."""
    t, largeur, hauteur = grille(parcelle.bounds, 700, marge_m=15)
    image = rendre(source, crs, t, largeur, hauteur)
    dessin = ImageDraw.Draw(image)
    if zones is not None and not zones.is_empty:
        for anneau in anneaux_px(zones, t):
            if len(anneau) >= 3:
                dessin.polygon(anneau, outline=couleur_zone, width=2)
    for anneau in anneaux_px(parcelle, t):
        dessin.line(anneau + anneau[:1], fill=(255, 255, 255), width=3)
    dessin.rectangle([0, 0, largeur, 22], fill=(0, 0, 0))
    dessin.text((6, 5), legende, fill=(255, 255, 255))
    image.save(chemin_png)


def miniature_photo(chemin_photo, chemin_png, legende):
    with Image.open(chemin_photo) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((600, 600))
    dessin = ImageDraw.Draw(im)
    dessin.rectangle([0, 0, im.width, 22], fill=(0, 0, 0))
    dessin.text((6, 5), legende, fill=(255, 255, 255))
    im.save(chemin_png)
