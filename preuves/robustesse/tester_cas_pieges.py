"""Batterie de cas « piégés » pour run.py (16/11/2026).

Chaque cas fabrique un dossier d'entrée à partir des données de démo (Cameroun : Sentinel-2 à 2 dates et
9 photos de feuilles ; Manuel : une ortho drone), puis lance run.py dessus. Un cas réussit si run.py se
termine sans erreur et écrit bulletin.html, ou s'arrête avec un message clair quand il n'y a rien à
analyser. Usage : python lancer_cas_pieges.py [CHEMIN_DU_DEPOT] [PYTHON]
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import geopandas as gpd
import rasterio
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.affinity import translate
from shapely.geometry import Polygon

DEPOT = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Field_scan\Field-Scan-intelligence-lab")
PYTHON = sys.argv[2] if len(sys.argv) > 2 else r"C:\Field_scan\.venv\Scripts\python.exe"
ICI = Path(__file__).parent
CAS = ICI / "cas"
S2 = Path(r"C:\Field_scan\demo_complete_2026-11-14\images\sentinel2")
FEUILLES = Path(r"C:\Field_scan\demo_complete_2026-11-14\images\feuilles")
PARCELLES = Path(r"C:\Field_scan\demo_complete_2026-11-14\parcelles.geojson")
DRONE = Path(r"C:\Field_scan\data\demo\manuel\drone\manuel.tif")
PARCELLE_DRONE = Path(r"C:\Field_scan\data\demo\manuel\parcelle_manuel.geojson")


def nouveau(nom):
    d = CAS / nom
    if d.exists():
        shutil.rmtree(d)
    (d / "images").mkdir(parents=True)
    return d


def copier_s2(dest, dates=None, sans=(), renommer=None):
    for dd in sorted(S2.iterdir()):
        if dates and dd.name not in dates:
            continue
        cible = dest / (renommer(dd.name) if renommer else dd.name)
        cible.mkdir(parents=True, exist_ok=True)
        for f in dd.glob("*.tiff"):
            if any(b in f.name for b in sans):
                continue
            shutil.copy2(f, cible / f.name)


def parcelles_cameroun():
    return gpd.read_file(PARCELLES)


def cas_reference():
    d = nouveau("01_reference")
    copier_s2(d / "images" / "sentinel2")
    shutil.copytree(FEUILLES, d / "images" / "feuilles")
    shutil.copy2(PARCELLES, d / "parcelles.geojson")
    return d, d / "parcelles.geojson", "bulletin"


def cas_parcelles_utm():
    d = nouveau("02_parcelles_utm")
    copier_s2(d / "images" / "sentinel2")
    parcelles_cameroun().to_crs(32632).to_file(d / "parcelles.geojson", driver="GeoJSON")
    return d, d / "parcelles.geojson", "bulletin"


def cas_parcelles_sans_id():
    d = nouveau("03_parcelles_sans_id")
    copier_s2(d / "images" / "sentinel2")
    g = parcelles_cameroun()
    g["nom"] = [f"Bloc {i}" for i in g["id"]]
    g.drop(columns=["id"]).to_file(d / "parcelles.geojson", driver="GeoJSON")
    return d, d / "parcelles.geojson", "bulletin"


def cas_parcelles_gpkg():
    d = nouveau("04_parcelles_geopackage")
    copier_s2(d / "images" / "sentinel2")
    parcelles_cameroun().to_file(d / "parcelles.gpkg", driver="GPKG")
    return d, d / "parcelles.gpkg", "bulletin"


def cas_geometries_bizarres():
    d = nouveau("05_multipolygone_et_contour_croise")
    copier_s2(d / "images" / "sentinel2")
    g = parcelles_cameroun()
    fusion = g.iloc[[0, 1]].union_all()                      # deux parcelles en une seule (multipolygone)
    x0, y0, x1, y1 = g.iloc[2].geometry.bounds
    noeud = Polygon([(x0, y0), (x1, y1), (x1, y0), (x0, y1)])   # contour qui se recoupe (papillon)
    g2 = gpd.GeoDataFrame({"id": ["fusion", "papillon"] + list(g.id.iloc[3:].astype(str))},
                          geometry=[fusion, noeud] + list(g.geometry.iloc[3:]), crs=g.crs)
    g2.to_file(d / "parcelles.geojson", driver="GeoJSON")
    return d, d / "parcelles.geojson", "bulletin"


def cas_une_date():
    d = nouveau("06_une_seule_date")
    copier_s2(d / "images" / "sentinel2", dates=["2026-04-20"])
    shutil.copy2(PARCELLES, d / "parcelles.geojson")
    return d, d / "parcelles.geojson", "bulletin"


def cas_bandes_manquantes():
    d = nouveau("07_bandes_manquantes")
    copier_s2(d / "images" / "sentinel2", sans=("B11", "B12", "B05", "Scene_classification"))
    shutil.copy2(PARCELLES, d / "parcelles.geojson")
    return d, d / "parcelles.geojson", "bulletin ou message clair"


def cas_noms_courts():
    d = nouveau("08_noms_courts")
    for dd in sorted(S2.iterdir()):
        cible = d / "images" / dd.name
        cible.mkdir(parents=True)
        for f in dd.glob("*.tiff"):
            for b in ("B02", "B03", "B04", "B05", "B08", "B8A", "B11", "B12"):
                if f"_{b}_" in f.name:
                    shutil.copy2(f, cible / f"{b}.tiff")
            if "Scene_classification" in f.name:
                shutil.copy2(f, cible / "Scene_classification_map.tiff")
    shutil.copy2(PARCELLES, d / "parcelles.geojson")
    return d, d / "parcelles.geojson", "bulletin"


def cas_date_compacte():
    d = nouveau("09_date_compacte")
    for dd in sorted(S2.iterdir()):
        cible = d / "images" / dd.name.replace("-", "")
        cible.mkdir(parents=True)
        for f in dd.glob("*.tiff"):
            for b in ("B02", "B03", "B04", "B05", "B08", "B8A", "B11", "B12"):
                if f"_{b}_" in f.name:
                    shutil.copy2(f, cible / f"{b}.tif")
            if "Scene_classification" in f.name:
                shutil.copy2(f, cible / "SCL.tif")
    shutil.copy2(PARCELLES, d / "parcelles.geojson")
    return d, d / "parcelles.geojson", "bulletin"


def cas_feuilles_difficiles():
    d = nouveau("10_feuilles_png_sans_gps_tournees")
    (d / "images" / "Feuilles").mkdir()
    for k, f in enumerate(sorted(FEUILLES.glob("*.jpg"))):
        im = Image.open(f)
        if k % 3 == 0:                                       # PNG, sans EXIF (donc sans GPS)
            im.convert("RGB").save(d / "images" / "Feuilles" / (f.stem + ".PNG"))
        elif k % 3 == 1:                                     # JPEG tourné par l'étiquette EXIF d'orientation
            exif = im.getexif()
            exif[0x0112] = 6
            im.save(d / "images" / "Feuilles" / (f.stem + ".JPG"), exif=exif)
        else:
            shutil.copy2(f, d / "images" / "Feuilles" / f.name)
    shutil.copy2(PARCELLES, d / "parcelles.geojson")
    return d, d / "parcelles.geojson", "bulletin"


def cas_dossier_vide():
    d = nouveau("11_dossier_vide")
    shutil.copy2(PARCELLES, d / "parcelles.geojson")
    return d, d / "parcelles.geojson", "message clair"


def cas_drone_utm():
    d = nouveau("12_drone_en_utm")
    with rasterio.open(DRONE) as src:
        t, w, h = calculate_default_transform(src.crs, "EPSG:32717", src.width, src.height, *src.bounds)
        prof = src.profile.copy()
        prof.update(crs="EPSG:32717", transform=t, width=w, height=h)
        with rasterio.open(d / "images" / "manuel_utm.TIF", "w", **prof) as dst:
            for b in range(1, src.count + 1):
                reproject(rasterio.band(src, b), rasterio.band(dst, b), resampling=Resampling.bilinear)
    shutil.copy2(PARCELLE_DRONE, d / "parcelles.geojson")
    return d, d / "parcelles.geojson", "bulletin"


def cas_parcelles_hors_images():
    d = nouveau("13_parcelles_hors_images")
    copier_s2(d / "images" / "sentinel2")
    g = parcelles_cameroun()
    g["geometry"] = g.geometry.apply(lambda p: translate(p, xoff=2.0))   # 2 degrés plus à l'est
    g.to_file(d / "parcelles.geojson", driver="GeoJSON")
    return d, d / "parcelles.geojson", "bulletin ou message clair"


def cas_chemin_accents():
    d = nouveau("14_chemin avec espaces et accents é")
    copier_s2(d / "images" / "données satellite")
    shutil.copytree(FEUILLES, d / "images" / "feuilles")
    shutil.copy2(PARCELLES, d / "parcelles été.geojson")
    return d, d / "parcelles été.geojson", "bulletin"


def cas_drone_et_s2():
    d = nouveau("15_drone_et_feuilles_meme_dossier")
    shutil.copy2(DRONE, d / "images" / "manuel.tif")
    shutil.copytree(FEUILLES, d / "images" / "feuilles")
    shutil.copy2(PARCELLE_DRONE, d / "parcelles.geojson")
    return d, d / "parcelles.geojson", "bulletin"


CASES = [cas_reference, cas_parcelles_utm, cas_parcelles_sans_id, cas_parcelles_gpkg, cas_geometries_bizarres,
         cas_une_date, cas_bandes_manquantes, cas_noms_courts, cas_date_compacte, cas_feuilles_difficiles,
         cas_dossier_vide, cas_drone_utm, cas_parcelles_hors_images, cas_chemin_accents, cas_drone_et_s2]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    CAS.mkdir(exist_ok=True)
    bilan = []
    for fabrique in CASES:
        try:
            d, parcelles, attendu = fabrique()
        except Exception as e:                                   # le cas lui-même n'a pas pu être fabriqué
            bilan.append({"cas": fabrique.__name__, "resultat": "cas non fabriqué", "detail": repr(e)})
            continue
        sortie = d / "resultats"
        debut = time.time()
        p = subprocess.run([PYTHON, "-B", "run.py", "--images", str(d / "images"), "--parcelles", str(parcelles),
                            "--out", str(sortie)], cwd=DEPOT, capture_output=True, text=True, encoding="utf-8",
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                           errors="replace")
        duree = time.time() - debut
        texte = (p.stdout + p.stderr).strip().splitlines()
        a_bulletin = (sortie / "bulletin.html").exists()
        trace = any("Traceback" in l for l in texte)
        if p.returncode == 0 and a_bulletin:
            res = "OK : bulletin écrit"
        elif not trace and p.returncode != 0:
            res = "OK : arrêt avec message" if "message" in attendu else "ÉCHEC : arrêt sans bulletin"
        else:
            res = "ÉCHEC : plantage"
        bilan.append({"cas": d.name, "attendu": attendu, "resultat": res, "code": p.returncode,
                      "duree_s": round(duree), "fin": " | ".join(texte[-3:])[:400]})
        print(f"{d.name:42s} {res:28s} {duree:5.0f} s  {texte[-1][:110] if texte else ''}", flush=True)
    (ICI / "bilan_cas_pieges.json").write_text(json.dumps(bilan, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
