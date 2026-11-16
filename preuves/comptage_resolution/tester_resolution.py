"""palm3 installé face à une image moins fine (14/11/2026).

Question : le modèle s'effondre-t-il quand le drone vole plus haut, comme un détecteur entraîné à 5 cm
peut le faire sur un jeu à 11 cm par pixel ? Ce n'est pas un choix de modèle : aucune règle, aucun réglage, seuil installé 0,25.

Méthode : chaque image préparée à 5 cm par pixel est réduite à g cm par pixel (moyenne des pixels, comme un
capteur plus grossier), puis remise à 5 cm (Lanczos, comme run.py). Les points annotés ne bougent pas.
g = 5 (référence), 8, 11, 14, 20 cm.
Jeux : validation (Manuel, Count banana, segmento : appris par le modèle final, donc flattés ; seule la baisse
compte), juges (Detecting Plantain Crops, Banano Finca : jamais appris), jorges (jamais appris, bananeraie dense).
Limite : une vraie image à 11 cm a aussi son flou, son bruit et sa compression ; cette simulation ne les a pas.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

DEPOT = Path(r"C:\Field_scan\Field-Scan-intelligence-lab")
sys.path.insert(0, str(DEPOT))
sys.path.insert(0, str(DEPOT / "outils"))
from comptage import detecteur                     # noqa: E402
from comptage.evaluation import apparier           # noqa: E402
import evaluer_comptage as ev                       # noqa: E402

torch.set_num_threads(4)
SORTIE = Path(__file__).parent
GSD_CM = [5, 8, 11, 14, 20]
SEUIL = 0.25


def charger_jorges():
    d = Path(r"C:\Field_scan\data\traite\comptage_v4")
    images, points = pd.read_csv(d / "images.csv"), pd.read_csv(d / "points.csv")
    liste = images[images.jeu == "jorges"].image
    return [(n, np.asarray(Image.open(d / "images" / n).convert("RGB")),
             points[points.image == n][["x", "y"]].to_numpy(np.float32)) for n in sorted(liste)]


def degrader(image, g):
    if g == 5:
        return image
    im = Image.fromarray(image)
    k = 5 / g
    petit = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.Resampling.BOX)
    return np.asarray(petit.resize(im.size, Image.Resampling.LANCZOS))


def main():
    validation, cible = ev.charger_validation(Path(r"C:\Field_scan\data\traite\comptage_v5"))
    juges = ev.charger_juges(Path(r"C:\Field_scan\data\traite\roboflow_juges"), cible)
    jeux = {**validation, **juges, "jorges": charger_jorges()}
    reseaux = detecteur.charger(sorted((DEPOT / "modeles" / "comptage").glob("centres_final_palm3_*.pt")))
    lignes = []
    for g in GSD_CM:
        debut = time.time()
        for jeu, liste in jeux.items():
            vp = nd = nv = 0
            for nom, image, vrai in liste:
                xy, sc = detecteur.detecter(reseaux, degrader(image, g), seuil=SEUIL)
                xy = xy[sc >= SEUIL]
                vp += apparier(xy * detecteur.PIXEL_M, vrai * detecteur.PIXEL_M)[0]
                nd += len(xy)
                nv += len(vrai)
            f1 = 2 * vp / max(nd + nv, 1)
            lignes.append({"gsd_cm": g, "jeu": jeu, "plants": nv, "detections": nd, "vp": vp, "f1": round(f1, 3),
                           "precision": round(vp / max(nd, 1), 3), "rappel": round(vp / max(nv, 1), 3),
                           "erreur_compte": round((nd - nv) / max(nv, 1), 3)})
            print(f"{g:>3} cm  {jeu:14s} F1 {f1:.3f}  compte {(nd - nv) / max(nv, 1):+.1%}", flush=True)
        print(f"  ({time.time() - debut:.0f} s)", flush=True)
        pd.DataFrame(lignes).to_csv(SORTIE / "resultats.csv", index=False)
    t = pd.DataFrame(lignes)
    print("\nF1 par jeu et résolution :")
    print(t.pivot(index="jeu", columns="gsd_cm", values="f1").to_string())
    print("\nErreur de compte :")
    print(t.pivot(index="jeu", columns="gsd_cm", values="erreur_compte").to_string())


if __name__ == "__main__":
    main()
