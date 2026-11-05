"""Ce que contient le dossier d'images donné à run.py, et lecture des parcelles.

Le dossier peut contenir, dans n'importe quels sous-dossiers :
  - des images drone géoréférencées (GeoTIFF, une seule image avec ses bandes, pixel de moins de 1 m) ;
  - des bandes Sentinel-2 exportées du Copernicus Browser (un fichier par bande, date et nom de bande
    dans le nom du fichier, par exemple 2025-12-16-00_00_..._B04_(Raw).tiff), une ou plusieurs dates ;
  - des photos de feuilles, dans un dossier nommé « feuilles ».
"""

import re
from pathlib import Path

import geopandas as gpd
import rasterio

EXTENSIONS_RASTER = (".tif", ".tiff")
EXTENSIONS_PHOTO = (".jpg", ".jpeg", ".png")
BANDES_S2 = ("B02", "B03", "B04", "B05", "B08", "B8A", "B11", "B12")
MOTIF_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")
MOTIF_BANDE = re.compile(r"_(B\d[\dA])_")


def _rasters(dossier):
    return sorted(f for f in Path(dossier).rglob("*") if f.suffix.lower() in EXTENSIONS_RASTER)


def _est_sentinel2(fichier):
    return bool(MOTIF_BANDE.search(fichier.name)) or "Scene_classification" in fichier.name


def sentinel2(dossier):
    """Bandes Sentinel-2 rangées par date : {"2025-12-16": {"B04": chemin, ..., "SCL": chemin}}."""
    dates = {}
    for f in _rasters(dossier):
        if not _est_sentinel2(f):
            continue
        date = MOTIF_DATE.search(f.name)
        date = date.group(1) if date else "sans_date"
        bande = "SCL" if "Scene_classification" in f.name else MOTIF_BANDE.search(f.name).group(1)
        if bande in dates.setdefault(date, {}):
            raise ValueError(f"Deux fichiers pour la bande {bande} du {date} : {dates[date][bande]} et {f}")
        dates[date][bande] = f
    return dict(sorted(dates.items()))


def taille_pixel_m(src):
    """Taille d'un pixel en mètres (approximative si l'image est en degrés)."""
    if src.crs is not None and src.crs.is_geographic:
        import math
        lat = (src.bounds.top + src.bounds.bottom) / 2
        return abs(src.res[0]) * 111320 * math.cos(math.radians(lat))
    return abs(src.res[0])


def images_drone(dossier):
    """Images drone géoréférencées : GeoTIFF d'au moins 3 bandes, pixel de moins de 1 m."""
    images = []
    for f in _rasters(dossier):
        if _est_sentinel2(f):
            continue
        with rasterio.open(f) as src:
            if src.crs is None or src.count < 3:
                print(f"  Image ignorée (pas géoréférencée ou moins de 3 bandes) : {f.name}")
                continue
            pixel = taille_pixel_m(src)
            if pixel >= 1:
                print(f"  Image ignorée (pixel de {pixel:.1f} m, trop grand pour une image drone) : {f.name}")
                continue
            images.append({"chemin": f, "bandes": src.count, "type": src.dtypes[0], "pixel_m": pixel})
    return images


def photos_feuilles(dossier):
    """Photos rangées dans un dossier nommé « feuilles » (à n'importe quelle profondeur)."""
    photos = []
    for d in Path(dossier).rglob("*"):
        if d.is_dir() and d.name.lower() == "feuilles":
            photos += [f for f in d.rglob("*") if f.suffix.lower() in EXTENSIONS_PHOTO]
    return sorted(set(photos))


def parcelles(chemin):
    """Parcelles du fichier GeoJSON, avec une colonne id (créée si absente) et leur surface en hectares."""
    p = gpd.read_file(chemin)
    if "id" not in p.columns:
        p["id"] = range(1, len(p) + 1)
    p = p[["id", "geometry"]].copy()
    p["geometry"] = p.geometry.force_2d().make_valid()      # répare les contours qui se recoupent
    utm = p.estimate_utm_crs()
    p["surface_ha"] = p.to_crs(utm).area / 1e4
    return p
