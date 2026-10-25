"""
Découpe les orthomosaïques réduites de ReforesTree en tuiles pour entraîner un détecteur de bananiers.

Pour chaque ferme :
  1. découpe l'image en tuiles de TAILLE x TAILLE pixels (environ 18 m de côté) ;
  2. garde les tuiles qui recouvrent la parcelle sur au moins 10 % de leur surface, et met en noir
     tout ce qui est hors de la parcelle : hors parcelle, les arbres ne sont pas tous annotés,
     et le détecteur apprendrait à tort que ces bananiers sont « du fond » ;
  3. recopie les rectangles d'arbres des auteurs (après fusion des doublons) qui tombent dans la tuile.

Les fermes d'entraînement sont découpées avec un pas de TAILLE / 2 (tuiles qui se chevauchent :
plus d'exemples) ; les autres sans chevauchement.

Sorties dans <dossier>/tuiles/ :
    images/<ferme>_<ligne>_<colonne>.png
    tuiles.csv        une ligne par tuile : image, ferme, rôle, nombre de bananiers
    annotations.csv   une ligne par rectangle : image, xmin, ymin, xmax, ymax, label (banane / autre)

Utilisation :
    python outils/preparer_tuiles_comptage.py --donnees C:/Field_scan/data/traite/reforestree
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.features import geometry_mask
from shapely import force_2d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluer_baseline_comptage import DECOUPAGE           # noqa: E402  (même découpage partout)
from preparer_reforestree import REDUCTION                # noqa: E402

TAILLE = 400            # pixels, soit environ 18 m à 4,5 cm par pixel
PART_MINIMALE = 0.10    # part minimale de la tuile située dans la parcelle


def main():
    parser = argparse.ArgumentParser(description="Découpage en tuiles pour le détecteur")
    parser.add_argument("--donnees", required=True, type=Path, help="dossier data/traite/reforestree")
    args = parser.parse_args()
    sortie = args.donnees / "tuiles"
    (sortie / "images").mkdir(parents=True, exist_ok=True)

    verite = pd.read_csv(args.donnees / "verite_terrain.csv")
    parcelles = gpd.read_file(args.donnees / "parcelles_reforestree.geojson")
    tuiles, annotations = [], []

    for site, role in DECOUPAGE.items():
        with rasterio.open(args.donnees / "orthos" / f"{site}.tif") as src:
            image = src.read([1, 2, 3])
            transform = src.transform
        zone = force_2d(parcelles[parcelles.site == site].geometry.union_all())
        dans_parcelle = geometry_mask([zone], out_shape=image.shape[1:], transform=transform, invert=True)
        image = image * dans_parcelle            # noir hors de la parcelle

        # Arbres de la ferme, en pixels de l'image réduite
        arbres = verite[verite.site == site].copy()
        lignes, colonnes = rasterio.transform.rowcol(transform, arbres.lon, arbres.lat)
        arbres["ligne"], arbres["colonne"] = np.array(lignes), np.array(colonnes)
        arbres["demi"] = arbres.largeur_px / REDUCTION / 2
        arbres = arbres[dans_parcelle[arbres.ligne.clip(0, image.shape[1] - 1),
                                      arbres.colonne.clip(0, image.shape[2] - 1)]]

        pas = TAILLE // 2 if role == "entrainement" else TAILLE
        nombre = 0
        for y0 in range(0, image.shape[1] - TAILLE + 1, pas):
            for x0 in range(0, image.shape[2] - TAILLE + 1, pas):
                if dans_parcelle[y0:y0 + TAILLE, x0:x0 + TAILLE].mean() < PART_MINIMALE:
                    continue
                nom = f"{site.replace(' RGB', '').replace(' ', '_')}_{y0}_{x0}.png"
                Image.fromarray(np.moveaxis(image[:, y0:y0 + TAILLE, x0:x0 + TAILLE], 0, -1)).save(sortie / "images" / nom)
                # Rectangles dont le centre est dans la tuile, coupés aux bords de la tuile
                dedans = arbres[arbres.ligne.between(y0, y0 + TAILLE - 1) & arbres.colonne.between(x0, x0 + TAILLE - 1)]
                for _, a in dedans.iterrows():
                    annotations.append({
                        "image_path": nom,
                        "xmin": max(0, round(a.colonne - a.demi - x0)), "ymin": max(0, round(a.ligne - a.demi - y0)),
                        "xmax": min(TAILLE, round(a.colonne + a.demi - x0)), "ymax": min(TAILLE, round(a.ligne + a.demi - y0)),
                        "label": "banane" if a.est_banane else "autre",
                    })
                tuiles.append({"image_path": nom, "site": site, "role": role, "bananes": int(dedans.est_banane.sum())})
                nombre += 1
        print(f"{site:25s} {role:13s} {nombre:4d} tuiles")

    tuiles, annotations = pd.DataFrame(tuiles), pd.DataFrame(annotations)
    tuiles.to_csv(sortie / "tuiles.csv", index=False)
    annotations.to_csv(sortie / "annotations.csv", index=False)
    resume = tuiles.groupby("role").agg(tuiles=("image_path", "size"), bananes=("bananes", "sum"))
    print("\n" + resume.to_string())
    print(f"\nRésultats écrits dans {sortie}")


if __name__ == "__main__":
    main()
