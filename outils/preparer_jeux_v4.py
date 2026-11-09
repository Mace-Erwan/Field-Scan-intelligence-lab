"""Prépare les 2 jeux ajoutés pour l'essai v4 du comptage et assemble les données comptage_v4.

Jeux (Roboflow Universe, CC BY 4.0, apprentissage seulement, carnet section 14) :
  - banavista (« Banana Detection ») : photos drone DJI de 4000 × 3000 px et orthomosaïques de bananeraies
    aux Philippines ; je garde la classe « Bananas » (« Others » = bâtiments, arbres, objets) ;
  - jorges (« banano ») : 71 tuiles de 1024 px d'une orthomosaïque de jeune bananeraie, un masque par plant.
Version téléchargée : copie (fork) sans redimensionnement ni augmentation.

Règles :
  - une image par original. Dans banavista, 50 noms de photos (DJI_0572.JPG…) apparaissent deux fois, mais ce
    sont des photos différentes (autres vols, numérotation DJI recommencée : pixels et annotations différents),
    gardées toutes les deux. Les vrais doublons ont été cherchés par empreinte puis regardés à l'œil
    (outils/verifier_recouvrements.py) : 2 orthomosaïques en double, écartées (liste DOUBLONS_VUS) ;
  - images sans aucune annotation écartées (une tuile de jorges pleine de plants n'est pas annotée) ;
  - jorges : masques dont la boîte dépasse 300 px écartés (2,5 fois le plant médian de 121 px) : ils couvrent
    toute la tuile ou plusieurs plants ;
  - un point par plant au centre de sa boîte, diamètre = racine de la surface de la boîte ;
  - échelle : la distance médiane d'un plant à son plus proche voisin est ramenée à 2 m, comme pour les jeux
    de la v3 (outils/preparer_nouveaux_roboflow.py) ; par image pour banavista (altitudes différentes),
    une seule fois pour jorges (une seule orthomosaïque) ; puis image remise à 5 cm par pixel.

Usage :
    python outils/preparer_jeux_v4.py --dezippe C:/Field_scan/data/traite/roboflow_v4/dezippe
        --v3 C:/Field_scan/data/traite/comptage_v3 --sortie C:/Field_scan/data/traite/comptage_v4
"""
import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

Image.MAX_IMAGE_PIXELS = None          # une orthomosaïque de banavista fait 13 000 × 13 000 px
PIXEL_M, ECART_M = 0.05, 2.0
BOITE_MAX_JORGES = 300
# doublons confirmés à l'œil (image écartée : image gardée)
DOUBLONS_VUS = {"15m.png": "clear-sky-15m.png (même image)",
                "Task-of-2024-05-10T162037110Z-orthophoto.png": "815-am.png (même orthomosaïque)"}


def lire(dossier):
    """Images du jeu (parties valid, test puis train) : liste de (chemin, nom d'origine, boîtes de plants)."""
    lots = []
    for partie in ("valid", "test", "train"):
        f = dossier / partie / "_annotations.coco.json"
        j = json.loads(f.read_text(encoding="utf-8"))
        plants = {c["id"] for c in j["categories"] if c["name"].lower().startswith("banan")}
        boites = defaultdict(list)
        for a in j["annotations"]:
            if a["category_id"] in plants:
                boites[a["image_id"]].append(a["bbox"])
        for im in j["images"]:
            nom = im.get("extra", {}).get("name", im["file_name"])
            lots.append((dossier / partie / im["file_name"], nom, np.array(boites[im["id"]], float).reshape(-1, 4)))
    return lots


def une_par_original(lots):
    gardes, ecartes = [], []
    for chemin, nom, b in lots:
        if nom in DOUBLONS_VUS:
            ecartes.append((nom, DOUBLONS_VUS[nom]))
        else:
            gardes.append((chemin, nom, b))
    return gardes, ecartes


def voisin_median(b):
    c = b[:, :2] + b[:, 2:] / 2
    return float(np.median(cKDTree(c).query(c, k=2)[0][:, 1])) if len(c) > 2 else np.nan


def ecrire(jeu, chemin, nom, b, f, sortie, origine):
    with Image.open(chemin) as im:
        im = im.convert("RGB")
        W, H = max(32, round(im.width * f)), max(32, round(im.height * f))
        petit = im.resize((W, H), Image.LANCZOS)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(nom).stem)
    nom_sortie, k = f"{jeu}__{stem}.jpg", 2
    while (sortie / "images" / nom_sortie).exists():          # deux photos différentes au même nom DJI
        nom_sortie, k = f"{jeu}__{stem}_{k}.jpg", k + 1
    petit.save(sortie / "images" / nom_sortie, quality=95)
    image = {"image": nom_sortie, "jeu": jeu, "partie": "apprentissage", "origine": origine,
             "largeur": W, "hauteur": H, "pixel_m": PIXEL_M}
    points = [{"image": nom_sortie, "x": round(float((x + w / 2) * f), 1), "y": round(float((y + h / 2) * f), 1),
               "diametre": round(float(np.sqrt(w * h) * f), 1)} for x, y, w, h in b]
    return image, points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dezippe", required=True, type=Path)
    ap.add_argument("--v3", required=True, type=Path)
    ap.add_argument("--sortie", required=True, type=Path)
    a = ap.parse_args()
    if a.sortie.exists():
        shutil.rmtree(a.sortie)
    (a.sortie / "images").mkdir(parents=True)

    images, points, bilan = [], [], []
    # banavista : échelle par image
    gardes, ecartes = une_par_original(lire(a.dezippe / "banavista"))
    gardes = [g for g in gardes if len(g[2])]
    for chemin, nom, b in gardes:
        nn = voisin_median(b)
        f = ECART_M / (nn * PIXEL_M)
        im, pts = ecrire("banavista", chemin, nom, b, f, a.sortie, "photo" if nom.startswith("DJI_") else Path(nom).stem)
        images.append(im); points += pts
        bilan.append({"jeu": "banavista", "image": nom, "plants": len(b), "pixel_origine_cm": round(100 * ECART_M / nn, 2), "facteur": round(f, 3)})
    print(f"banavista : {len(gardes)} images gardées ; écartées : {ecartes}")

    # jorges : une échelle pour toute l'orthomosaïque
    gardes, ecartes = une_par_original(lire(a.dezippe / "jorges"))
    vides = [nom for _, nom, b in gardes if len(b) == 0]
    n_avant = sum(len(b) for _, _, b in gardes)
    gardes = [(c, n, b[np.maximum(b[:, 2], b[:, 3]) <= BOITE_MAX_JORGES]) for c, n, b in gardes if len(b)]
    nn = float(np.median([voisin_median(b) for _, _, b in gardes if len(b) > 2]))
    f = ECART_M / (nn * PIXEL_M)
    for chemin, nom, b in gardes:
        im, pts = ecrire("jorges", chemin, nom, b, f, a.sortie, "orthomosaique")
        images.append(im); points += pts
        bilan.append({"jeu": "jorges", "image": nom, "plants": len(b), "pixel_origine_cm": round(100 * ECART_M / nn, 2), "facteur": round(f, 3)})
    print(f"jorges : {len(gardes)} images gardées ; sans annotation, écartées : {vides} ; doublons : {ecartes} ; "
          f"plants {sum(len(b) for _, _, b in gardes)} sur {n_avant} (boîtes > {BOITE_MAX_JORGES} px retirées)")

    # assemblage avec les données v3
    im3 = pd.read_csv(a.v3 / "images.csv")
    pt3 = pd.read_csv(a.v3 / "points.csv")
    for n in im3.image:
        shutil.copy2(a.v3 / "images" / n, a.sortie / "images" / n)
    im_new, pt_new = pd.DataFrame(images), pd.DataFrame(points)
    assert not set(im_new.image) & set(im3.image), "noms d'images en double"
    pd.concat([im3, im_new]).to_csv(a.sortie / "images.csv", index=False)
    pd.concat([pt3, pt_new]).to_csv(a.sortie / "points.csv", index=False)
    b = pd.DataFrame(bilan)
    b.to_csv(a.sortie.parent / "bilan_jeux_v4.csv", index=False)
    print(b.groupby("jeu").agg(images=("image", "size"), plants=("plants", "sum"),
                               pixel_median_cm=("pixel_origine_cm", "median"),
                               pixel_min_cm=("pixel_origine_cm", "min"), pixel_max_cm=("pixel_origine_cm", "max")).to_string())
    shutil.make_archive(str(a.sortie), "zip", a.sortie)
    print(f"Écrit dans {a.sortie} et {a.sortie}.zip")


if __name__ == "__main__":
    main()
