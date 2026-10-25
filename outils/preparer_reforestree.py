"""
Prépare le jeu ReforesTree (Équateur) pour le comptage des bananiers.

Le script :
  1. nettoie les annotations : le fichier annotations/cleaned/clean_annotations.csv contient
     le même arbre plusieurs fois (boîtes proposées par un détecteur sur des tuiles qui se
     chevauchent, et fichier entièrement dupliqué pour un site). Deux boîtes d'un même site
     qui se recouvrent à plus de 50 % (IoU > 0,5) sont fusionnées en un seul arbre ;
  2. réduit chaque orthomosaïque d'un facteur REDUCTION (environ 1 cm -> 4 cm par pixel),
     en GeoTIFF géoréférencé, pour que les calculs tiennent sur un PC sans GPU ;
  3. exporte le contour des 6 parcelles en GeoJSON.

Sorties dans <dossier sortie> :
    verite_terrain.csv          un arbre par ligne : site, longitude, latitude, banane ou non
    orthos/<site>.tif           orthomosaïques réduites
    parcelles_reforestree.geojson

Utilisation :
    python outils/preparer_reforestree.py --brut C:/Field_scan/data/brut/reforestree/reforesTree --out C:/Field_scan/data/traite/reforestree
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling

REDUCTION = 4        # 1 pixel réduit = 4 x 4 pixels d'origine
SEUIL_IOU = 0.5      # recouvrement au-delà duquel deux boîtes sont le même arbre

SITES = {  # nom dans les annotations -> mot qui le désigne dans le shapefile des parcelles
    "Carlos Vera Arteaga RGB": "Arteaga",
    "Carlos Vera Guevara RGB": "Gevara",
    "Flora Pluas RGB": "Flora",
    "Leonor Aspiazu RGB": "leonor",
    "Manuel Macias RGB": "Manuel",
    "Nestor Macias RGB": "Nestor",
}


def recouvrements(boites):
    """Matrice des IoU (intersection / union) entre toutes les boîtes [xmin, ymin, xmax, ymax]."""
    x1 = np.maximum(boites[:, None, 0], boites[None, :, 0])
    y1 = np.maximum(boites[:, None, 1], boites[None, :, 1])
    x2 = np.minimum(boites[:, None, 2], boites[None, :, 2])
    y2 = np.minimum(boites[:, None, 3], boites[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aire = (boites[:, 2] - boites[:, 0]) * (boites[:, 3] - boites[:, 1])
    return inter / (aire[:, None] + aire[None, :] - inter)


def fusionner(site, groupe):
    """Regroupe les boîtes qui se recouvrent (de proche en proche) et renvoie un arbre par groupe."""
    boites = groupe[["Xmin", "Ymin", "Xmax", "Ymax"]].to_numpy()
    iou = recouvrements(boites)
    parent = list(range(len(boites)))

    def racine(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(iou > SEUIL_IOU, 1))):
        parent[racine(i)] = racine(j)

    arbres = []
    for membres in pd.Series(range(len(boites))).groupby([racine(i) for i in range(len(boites))]):
        g = groupe.iloc[membres[1].to_numpy()]
        arbres.append({
            "site": site,
            "lon": g.lon.mean(), "lat": g.lat.mean(),
            "x_px": g.X.mean(), "y_px": g.Y.mean(),            # pixels de l'orthomosaïque d'origine
            "largeur_px": (g.Xmax - g.Xmin).mean(),
            "est_banane": bool(g.is_banana.mean() >= 0.5),     # vote si les boîtes ne sont pas d'accord
            "boites_fusionnees": len(g),
        })
    return arbres


def main():
    parser = argparse.ArgumentParser(description="Préparation de ReforesTree pour le comptage")
    parser.add_argument("--brut", required=True, type=Path, help="dossier reforesTree décompressé")
    parser.add_argument("--out", required=True, type=Path, help="dossier de sortie")
    args = parser.parse_args()
    (args.out / "orthos").mkdir(parents=True, exist_ok=True)

    # 1. Vérité terrain sans doublons
    annotations = pd.read_csv(args.brut / "annotations" / "cleaned" / "clean_annotations.csv")
    arbres = []
    for site, groupe in annotations.groupby("img_name"):
        arbres += fusionner(site, groupe)
    arbres = pd.DataFrame(arbres)
    arbres.to_csv(args.out / "verite_terrain.csv", index=False)
    resume = pd.DataFrame({
        "boites": annotations.groupby("img_name").size(),
        "bananes_avant": annotations.groupby("img_name").is_banana.sum(),
        "arbres": arbres.groupby("site").size(),
        "bananiers": arbres.groupby("site").est_banane.sum(),
    })
    print(resume.to_string())
    print(f"Total : {len(annotations)} boîtes -> {len(arbres)} arbres, dont {arbres.est_banane.sum()} bananiers")

    # 2. Orthomosaïques réduites (lecture directe dans les aperçus internes du GeoTIFF)
    for site in SITES:
        with rasterio.open(args.brut / "wwf_ecuador" / "RGB Orthomosaics" / f"{site}.tif") as src:
            forme = (src.count, src.height // REDUCTION, src.width // REDUCTION)
            image = src.read(out_shape=forme, resampling=Resampling.average)
            transform = src.transform * src.transform.scale(src.width / forme[2], src.height / forme[1])
            profil = src.profile | {"height": forme[1], "width": forme[2], "transform": transform,
                                    "compress": "deflate", "tiled": True}
        with rasterio.open(args.out / "orthos" / f"{site}.tif", "w", **profil) as dst:
            dst.write(image)
        print(f"{site:25s} {forme[2]} x {forme[1]} pixels")

    # 3. Parcelles
    parcelles = gpd.read_file(args.brut / "wwf_ecuador" / "Merged_final_plots" / "Merged_final_plots.shp")
    parcelles["site"] = [next(s for s, mot in SITES.items() if mot in nom) for nom in parcelles.Name]
    parcelles["id"] = range(1, len(parcelles) + 1)
    parcelles[["id", "site", "geometry"]].to_file(args.out / "parcelles_reforestree.geojson", driver="GeoJSON")
    print(f"Résultats écrits dans {args.out}")


if __name__ == "__main__":
    main()
