"""Seuil de détection selon la résolution (règle : regle.md, écrite avant toute mesure).

    python seuil_resolution.py regler   -> seuils_figes.json (validation seulement, essai palm3)
    python seuil_resolution.py noter    -> notation.csv et verdict.txt (juges, jorges, palmiers ; une seule fois)
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

DEPOT = Path(r"C:\Field_scan\Field-Scan-intelligence-lab")
sys.path.insert(0, str(DEPOT))
sys.path.insert(0, str(DEPOT / "outils"))
from comptage import detecteur                     # noqa: E402
import evaluer_comptage as ev                       # noqa: E402

torch.set_num_threads(4)
ICI = Path(__file__).parent
GSD = [5, 8, 11, 14, 20]
ESSAI = [Path(r"C:\Field_scan\modeles\comptage_v6_essai\comptage_v6") / f"centres_palm3_{k}.pt" for k in (0, 1)]
FINAL = sorted((DEPOT / "modeles" / "comptage").glob("centres_final_palm3_*.pt"))
JUGES = ["Plantain", "Banano Finca"]


def degrader(image, g):
    if g == 5:
        return image
    im = Image.fromarray(image)
    petit = im.resize((max(1, round(im.width * 5 / g)), max(1, round(im.height * 5 / g))), Image.Resampling.BOX)
    return np.asarray(petit.resize(im.size, Image.Resampling.LANCZOS))


def detecter_tout(reseaux, jeux, g):
    return {n: detecteur.detecter(reseaux, degrader(im, g), seuil=0.05) for liste in jeux.values() for n, im, _ in liste}


def charger_jorges():
    d = Path(r"C:\Field_scan\data\traite\comptage_v4")
    images, points = pd.read_csv(d / "images.csv"), pd.read_csv(d / "points.csv")
    return [(n, np.asarray(Image.open(d / "images" / n).convert("RGB")),
             points[points.image == n][["x", "y"]].to_numpy(np.float32)) for n in sorted(images[images.jeu == "jorges"].image)]


def regler():
    validation, _ = ev.charger_validation(Path(r"C:\Field_scan\data\traite\comptage_v5"))
    reseaux = detecteur.charger(ESSAI)
    seuils, tables = {}, []
    for g in GSD:
        debut = time.time()
        t = ev.noter(detecter_tout(reseaux, validation, g), validation)
        s = ev.seuil_equilibre(t)
        seuils[g] = s
        t.insert(0, "gsd_cm", g)
        tables.append(t)
        ligne = t[t.seuil == s]
        print(f"{g:>3} cm : seuil d'équilibre {s:.2f} ; F1 {ligne.f1.mean():.3f} ; "
              f"compte {ligne.erreur_compte.abs().mean():.3f} ({time.time() - debut:.0f} s)", flush=True)
    pd.concat(tables).round(4).to_csv(ICI / "reglage_validation.csv", index=False)
    note = "à 5 cm, la règle redonne 0,25" if abs(seuils[5] - 0.25) < 1e-9 else \
        f"à 5 cm, la règle donne {seuils[5]:.2f} : le seuil à 5 cm reste 0,25 (regle.md)"
    seuils[5] = 0.25
    (ICI / "seuils_figes.json").write_text(json.dumps(
        {"fige_le": datetime.now().isoformat(timespec="seconds"), "modele_reglage": "essai palm3 v6",
         "seuils_par_gsd_cm": seuils, "note": note}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("gelé :", seuils, "|", note)


def noter():
    fige = json.loads((ICI / "seuils_figes.json").read_text(encoding="utf-8"))
    seuils = {int(k): v for k, v in fige["seuils_par_gsd_cm"].items()}
    print("seuils gelés le", fige["fige_le"], ":", seuils, flush=True)
    _, cible = ev.charger_validation(Path(r"C:\Field_scan\data\traite\comptage_v5"))
    jeux = {**ev.charger_juges(Path(r"C:\Field_scan\data\traite\roboflow_juges"), cible), "jorges": charger_jorges()}
    palmiers = ev.charger_palmiers(Path(r"C:\Field_scan\data\traite\cuboid"))
    reseaux = detecteur.charger(FINAL)
    lignes = []
    for g in GSD:
        debut = time.time()
        det = detecter_tout(reseaux, {**jeux, "palmiers": palmiers}, g)
        t = ev.noter(det, jeux)
        for nom, s in (("fixe", 0.25), ("variable", seuils[g])):
            l = t[np.isclose(t.seuil, s)]
            for _, r in l.iterrows():
                lignes.append({"gsd_cm": g, "version": nom, "seuil": s, "jeu": r.jeu, "f1": round(r.f1, 4),
                               "erreur_compte": round(r.erreur_compte, 4), "plants": r.plants,
                               "detections": r.detections})
            lignes.append({"gsd_cm": g, "version": nom, "seuil": s, "jeu": "palmiers",
                           "faux_par_ha": round(ev.faux_par_ha(det, palmiers, s), 2)})
        print(f"{g:>3} cm noté ({time.time() - debut:.0f} s)", flush=True)
    r = pd.DataFrame(lignes)
    r.to_csv(ICI / "notation.csv", index=False)

    j = r[r.jeu.isin(JUGES)]
    deg = j[j.gsd_cm > 5]
    compte = deg.assign(a=deg.erreur_compte.abs()).groupby("version").a.mean()
    f1 = j.groupby(["gsd_cm", "version"]).f1.mean().unstack()
    palm = r[r.jeu == "palmiers"].set_index(["gsd_cm", "version"]).faux_par_ha.unstack()
    c1 = compte["fixe"] - compte["variable"] >= 0.10
    c2 = bool((f1.loc[[8, 11, 14, 20], "variable"] - f1.loc[[8, 11, 14, 20], "fixe"] >= -0.02).all())
    c3 = bool((palm["variable"] < 5).all())
    texte = [f"Seuils gelés le {fige['fige_le']} : {seuils}",
             f"Erreur de compte moyenne des juges (8 à 20 cm) : fixe {compte['fixe']:.3f}, variable {compte['variable']:.3f} "
             f"-> baisse {compte['fixe'] - compte['variable']:.3f} (il faut 0,10) : {'oui' if c1 else 'non'}",
             "F1 moyenne des juges par résolution :\n" + f1.round(3).to_string(),
             f"Aucune baisse de plus de 0,02 : {'oui' if c2 else 'non'}",
             "Faux bananiers par ha sur les palmiers :\n" + palm.to_string(),
             f"Sous 5 par ha partout : {'oui' if c3 else 'non'}",
             "jorges (information) :\n" + r[r.jeu == "jorges"].pivot(index="gsd_cm", columns="version",
                                                                     values=["f1", "erreur_compte"]).round(3).to_string(),
             f"VERDICT : {'ADOPTÉ (non confirmé : dégradation simulée, pas de jeu neuf à 11 cm)' if c1 and c2 and c3 else 'NON ADOPTÉ'}"]
    (ICI / "verdict.txt").write_text("\n\n".join(texte), encoding="utf-8")
    print("\n\n".join(texte))


if __name__ == "__main__":
    {"regler": regler, "noter": noter}[sys.argv[1]]()
