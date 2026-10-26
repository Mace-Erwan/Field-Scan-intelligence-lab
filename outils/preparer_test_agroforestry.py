"""
Prépare une ferme ReforesTree pour tester le modèle « Agroforestry inventory » (banane),
cité par le sujet : https://huggingface.co/JefersonPMS/agroforestry-inventory

Ce modèle a été entraîné sur une ferme du Brésil, avec des images à 1,66 cm par pixel ;
ses auteurs préviennent que des résolutions très différentes dégradent le résultat.
Le script :
  1. découpe l'orthomosaïque d'origine (environ 1 cm par pixel) autour de la parcelle, avec une marge ;
  2. la ramène à 1,66 cm par pixel ;
  3. écrit à côté les vrais bananiers et le contour de la parcelle, pour la notation à 1 m.

Sorties dans <dossier sortie> :
    <ferme>_1p66cm.tif        image à donner au modèle
    verite_<ferme>.csv         vrais arbres de la ferme (longitude, latitude, banane ou non)
    parcelle_<ferme>.geojson   contour de la parcelle

Utilisation :
    python outils/preparer_test_agroforestry.py --brut C:/Field_scan/data/brut/reforestree/reforesTree --traite C:/Field_scan/data/traite/reforestree --site "Manuel Macias RGB" --out C:/Field_scan/data/traite/test_agroforestry
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

RESOLUTION_CIBLE_M = 0.0166   # résolution d'entraînement du modèle banane
MARGE_M = 10                  # marge autour de la parcelle


def main():
    parser = argparse.ArgumentParser(description="Préparation du test du modèle Agroforestry inventory")
    parser.add_argument("--brut", required=True, type=Path, help="dossier reforesTree décompressé")
    parser.add_argument("--traite", required=True, type=Path, help="dossier data/traite/reforestree")
    parser.add_argument("--site", required=True, help='ferme, ex. "Manuel Macias RGB"')
    parser.add_argument("--out", required=True, type=Path, help="dossier de sortie")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    nom = args.site.replace(" RGB", "").replace(" ", "_")

    parcelle = gpd.read_file(args.traite / "parcelles_reforestree.geojson")
    parcelle = parcelle[parcelle.site == args.site]
    parcelle.to_file(args.out / f"parcelle_{nom}.geojson", driver="GeoJSON")
    verite = pd.read_csv(args.traite / "verite_terrain.csv")
    verite[verite.site == args.site].to_csv(args.out / f"verite_{nom}.csv", index=False)

    with rasterio.open(args.brut / "wwf_ecuador" / "RGB Orthomosaics" / f"{args.site}.tif") as src:
        xmin, ymin, xmax, ymax = parcelle.total_bounds
        lat = (ymin + ymax) / 2
        metres_par_degre = 111320 * np.cos(np.radians(lat))
        marge = MARGE_M / metres_par_degre
        fenetre = from_bounds(xmin - marge, ymin - marge, xmax + marge, ymax + marge, src.transform).round_offsets().round_lengths()
        taille_origine = abs(src.transform.a) * metres_par_degre
        facteur = RESOLUTION_CIBLE_M / taille_origine
        forme = (3, int(fenetre.height / facteur), int(fenetre.width / facteur))
        image = src.read([1, 2, 3], window=fenetre, out_shape=forme, resampling=Resampling.average)
        transform = src.window_transform(fenetre) * src.transform.scale(fenetre.width / forme[2], fenetre.height / forme[1])
        profil = src.profile | {"count": 3, "height": forme[1], "width": forme[2], "transform": transform,
                                "compress": "deflate", "tiled": True, "photometric": "RGB"}
        profil.pop("nodata", None)
    with rasterio.open(args.out / f"{nom}_1p66cm.tif", "w", **profil) as dst:
        dst.write(image)
    print(f"{args.site} : pixel d'origine {taille_origine * 100:.2f} cm -> {RESOLUTION_CIBLE_M * 100:.2f} cm, "
          f"image de {forme[2]} x {forme[1]} pixels")
    print(f"Résultats écrits dans {args.out}")


if __name__ == "__main__":
    main()
