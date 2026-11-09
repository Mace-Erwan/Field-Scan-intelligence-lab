"""Prépare les données de l'essai v5 du comptage : celles de la v3, plus des tuiles de palmiers à huile sans aucun
bananier, données au modèle comme exemples « rien à trouver » (aucun point).

Jeu : Roboflow Universe « Multiple tree detection » (espace « Cuboid », CC BY 4.0, carnet section 15), copié et
exporté sans redimensionnement ni modification. Seules les tuiles de palmiers à huile servent : 500 tuiles de
1024 px, tirées de 3 zones de 4 000 × 4 000 px (nom : <zone x>_<zone y>_<x>_<y>), qui se recouvrent à l'intérieur
d'une zone. Les zones 44000_4000 et 44000_16000 vont à l'apprentissage ; la zone 52000_20000, à au moins 350 m,
est gardée de côté pour mesurer les faux bananiers (outils/evaluer_comptage.py, option --palmiers).
Les images de cocotiers sont écartées : ce sont de petites fermes mélangées où des plants non annotés
ressemblent à de jeunes bananiers.

Échelle : les palmiers à huile sont plantés à environ 9 m les uns des autres ; la distance médiane d'un palmier
annoté à son plus proche voisin est de 102,6 px, d'où un pixel d'environ 8,8 cm ; les tuiles sont remises à 5 cm.

Usage :
    python outils/preparer_jeux_v5.py --cuboid C:/Field_scan/data/traite/cuboid
        --v3 C:/Field_scan/data/traite/comptage_v3 --sortie C:/Field_scan/data/traite/comptage_v5
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

PIXEL_M = 0.05
PIXEL_PALMIERS_M = 9.0 / 102.6
ZONES_APPRENTISSAGE = ("44000_4000", "44000_16000")
ZONE_GARDEE = "52000_20000"


def tuiles_palmiers(cuboid):
    """Tuiles de palmiers seuls (aucune boîte de bananier ni de cocotier), avec leur zone."""
    t = pd.read_csv(cuboid / "inventaire.csv")
    t = t[(t.palmier > 0) & (t.cocotier == 0) & (t.banane == 0)].copy()
    t["zone"] = t.nom.str.extract(r"^(\d+_\d+)_")[0]
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuboid", required=True, type=Path)
    ap.add_argument("--v3", required=True, type=Path)
    ap.add_argument("--sortie", required=True, type=Path)
    a = ap.parse_args()
    if a.sortie.exists():
        shutil.rmtree(a.sortie)
    (a.sortie / "images").mkdir(parents=True)

    t = tuiles_palmiers(a.cuboid)
    f = PIXEL_PALMIERS_M / PIXEL_M
    images = []
    for r in t[t.zone.isin(ZONES_APPRENTISSAGE)].itertuples():
        with Image.open(a.cuboid / "dezippe" / r.partie / r.fichier) as im:
            im = im.convert("RGB")
            petit = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
        nom = f"palmiers__{Path(r.nom).stem}.jpg"
        petit.save(a.sortie / "images" / nom, quality=95)
        images.append({"image": nom, "jeu": "palmiers", "partie": "apprentissage", "origine": r.zone,
                       "largeur": petit.width, "hauteur": petit.height, "pixel_m": PIXEL_M})

    im3 = pd.read_csv(a.v3 / "images.csv")
    for n in im3.image:
        shutil.copy2(a.v3 / "images" / n, a.sortie / "images" / n)
    im_new = pd.DataFrame(images)
    assert not set(im_new.image) & set(im3.image)
    pd.concat([im3, im_new]).to_csv(a.sortie / "images.csv", index=False)
    shutil.copy2(a.v3 / "points.csv", a.sortie / "points.csv")      # aucun point pour les palmiers
    print(f"palmiers : {len(im_new)} tuiles d'apprentissage ({', '.join(ZONES_APPRENTISSAGE)}), "
          f"{im_new.largeur.iloc[0]} px de côté à 5 cm ; zone gardée de côté : {ZONE_GARDEE} "
          f"({(t.zone == ZONE_GARDEE).sum()} tuiles)")
    shutil.make_archive(str(a.sortie), "zip", a.sortie)
    print(f"Écrit dans {a.sortie} et {a.sortie}.zip")


if __name__ == "__main__":
    main()
