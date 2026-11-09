"""Note des modèles de comptage sur la validation et sur les deux plantations « juges », et choisit
le seuil d'équilibre de chacun (docs/experiences.md, section 1). Tout tourne sur CPU.

Les juges (carnet de données, section 12) servent seulement à décider, jamais à l'apprentissage :
  - Detecting Plantain Crops : les 19 images non modifiées (parties valid et test), 208 plantains.
    Taille du pixel inconnue : le plant médian (côté le plus long de sa boîte) est ramené à la taille du
    bananier médian des fermes d'apprentissage ReforesTree, comme pour segmento (outils/preparer_comptage.py) ;
  - Banano Finca : une copie par tuile d'origine (25 tuiles, 409 plants), environ 1,75 cm par pixel.

Notation (comme dans les notebooks) : pour chaque jeu, les vrais positifs, détections et vrais plants de
toutes ses images sont additionnés ; une détection est juste à moins de 1 m d'un plant annoté, chaque plant
n'étant trouvé qu'une fois ; une détection sur un pixel noir est ignorée.

Seuil d'équilibre (règle du 05/11) : parmi les seuils dont la F1 moyenne des 3 validations est à moins de
0,005 de la meilleure, celui dont l'erreur de comptage moyenne des 3 validations (|détections - plants| / plants)
est la plus petite ; à égalité, celui qui a la meilleure F1.

Usage :
    python outils/evaluer_comptage.py --comptage C:/Field_scan/data/traite/comptage --juges DOSSIER_JUGES
        --modele v2=a_0.pt,a_1.pt --modele v3=b_0.pt,b_1.pt --sortie DOSSIER
    DOSSIER_JUGES contient banano/ et plantain/ : les zips Roboflow (data/brut/roboflow_juges) décompressés.
    Le premier modèle est la référence : chaque autre est comparé à lui avec la règle de l'essai v3.
    --facteur-plantain 0.8 : plants de Plantain 0,8 fois la taille supposée (l'échelle de ce jeu est une
    hypothèse ; relancer avec 0.7 à 1.4 montre si la décision en dépend).
    --palmiers C:/Field_scan/data/traite/cuboid : mesure aussi les faux bananiers par hectare sur 40 tuiles
    (tirées au hasard, graine 0) de la zone de palmiers à huile jamais apprise (52000_20000), et applique la
    règle de l'essai v5 : faux bananiers -50 % au moins, F1 des juges pas plus de 0,01 en dessous, aucune
    validation qui perd plus de 0,02.
    --modele v3=@C:/.../detections_v3.csv : reprend les détections enregistrées par un lancement précédent
    (mêmes images), au lieu de refaire tourner le modèle.

Sorties dans --sortie : detections_<modèle>.csv (toutes les détections de score >= 0,05, pour refaire les
calculs sans les modèles), scores.csv (une ligne par modèle, jeu et seuil), bilan.json.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comptage import detecteur                      # noqa: E402
from comptage.evaluation import apparier            # noqa: E402

SEUILS = np.round(np.arange(0.05, 0.96, 0.05), 2)
ECART_F1 = 0.005             # règle d'équilibre : F1 à moins de 0,005 de la meilleure
GAIN_MIN = 0.01              # règle de l'essai v3 : gain minimal sur la moyenne des juges...
PERTE_MAX = 0.02             # ... sans perte de plus de 0,02 sur un jeu de validation
PIXEL_BANANO_M = 0.0175      # tuiles de 640 px espacées de 11,2 m (carnet, section 12)
VALIDATIONS = {"Manuel": "reforestree", "Count banana": "count_banana", "segmento": "segmento"}
JUGES = ["Plantain", "Banano Finca"]
N_TIRAGES = 5000             # tirages pour l'intervalle de l'écart sur les juges (par image)
ZONE_PALMIERS = "52000_20000"            # zone de palmiers jamais apprise (outils/preparer_jeux_v5.py)
N_TUILES_PALMIERS = 40
PIXEL_PALMIERS_M = 9.0 / 102.6           # palmiers plantés à environ 9 m : 102,6 px entre voisins
BAISSE_PALMIERS_MIN = 0.5                # règle de l'essai v5
PERTE_JUGES_MAX = 0.01


def lire_coco(fichier):
    d = json.loads(Path(fichier).read_text(encoding="utf-8"))
    boites = defaultdict(list)
    for a in d["annotations"]:
        boites[a["image_id"]].append(a["bbox"])
    return [(Path(fichier).parent / i["file_name"], i.get("extra", {}).get("name", i["file_name"]), boites[i["id"]])
            for i in d["images"]]


def a_5cm(chemin, boites, facteur):
    """Image remise à 5 cm par pixel (uint8, h x w x 3) et centres des boîtes en pixels."""
    im = Image.open(chemin).convert("RGB")
    im = im.resize((round(im.width * facteur), round(im.height * facteur)), Image.Resampling.LANCZOS)
    pts = np.array([[(x + w / 2) * facteur, (y + h / 2) * facteur] for x, y, w, h in boites], np.float32).reshape(-1, 2)
    return np.asarray(im), pts


def charger_juges(dossier, cible_px, facteur_plantain=1.0):
    jeux = {}
    lots = [l for p in ("valid", "test") for l in lire_coco(dossier / "plantain" / p / "_annotations.coco.json")]
    mediane = float(np.median([max(b[2], b[3]) for _, _, bb in lots for b in bb]))
    f = cible_px / mediane * facteur_plantain
    jeux["Plantain"] = [(nom, *a_5cm(ch, bb, f)) for ch, nom, bb in lots]
    print(f"  Plantain : {len(lots)} images, plant médian {mediane:.1f} px -> x {f:.3f}")
    vus, lots = set(), []
    for p in ("valid", "test", "train"):                      # de préférence les copies non modifiées
        for ch, nom, bb in lire_coco(dossier / "banano" / p / "_annotations.coco.json"):
            if nom not in vus:
                vus.add(nom)
                lots.append((ch, nom, bb))
    f = PIXEL_BANANO_M / detecteur.PIXEL_M
    jeux["Banano Finca"] = [(nom, *a_5cm(ch, bb, f)) for ch, nom, bb in lots]
    print(f"  Banano Finca : {len(lots)} images -> x {f:.3f}")
    return jeux


def charger_palmiers(dossier):
    """40 tuiles de palmiers à huile de la zone jamais apprise, remises à 5 cm : (nom, image, aucun plant)."""
    t = pd.read_csv(dossier / "inventaire.csv")
    t = t[(t.palmier > 0) & (t.cocotier == 0) & (t.banane == 0) & t.nom.str.startswith(ZONE_PALMIERS + "_")]
    t = t.sample(N_TUILES_PALMIERS, random_state=0)
    f = PIXEL_PALMIERS_M / detecteur.PIXEL_M
    tuiles = []
    for r in t.itertuples():
        im = Image.open(dossier / "dezippe" / r.partie / r.fichier).convert("RGB")
        im = im.resize((round(im.width * f), round(im.height * f)), Image.Resampling.LANCZOS)
        tuiles.append((r.nom, np.asarray(im), np.zeros((0, 2), np.float32)))
    return tuiles


def faux_par_ha(detections, tuiles, seuil):
    n = sum(int((detections[nom][1] >= seuil).sum()) for nom, _, _ in tuiles)
    ha = sum(im.shape[0] * im.shape[1] for _, im, _ in tuiles) * detecteur.PIXEL_M ** 2 / 1e4
    return n / ha


def relire_detections(fichier, noms):
    d = pd.read_csv(fichier)
    det = {n: (np.zeros((0, 2), np.float32), np.zeros(0, np.float32)) for n in noms}
    for n, g in d.groupby("image"):
        det[n] = (g[["x", "y"]].to_numpy(np.float32), g.score.to_numpy(np.float32))
    return det


def charger_validation(dossier):
    images = pd.read_csv(dossier / "images.csv")
    points = pd.read_csv(dossier / "points.csv")
    jeux = {}
    for nom, jeu in VALIDATIONS.items():
        liste = images[(images.jeu == jeu) & (images.partie == "validation")].image
        jeux[nom] = [(n, np.asarray(Image.open(dossier / "images" / n).convert("RGB")),
                      points[points.image == n][["x", "y"]].to_numpy(np.float32)) for n in sorted(liste)]
    app = points.merge(images[["image", "jeu", "partie"]], on="image")
    cible = float(np.median(app[(app.jeu == "reforestree") & (app.partie == "apprentissage")].diametre))
    return jeux, cible


def noter(detections, jeux):
    """detections : {image: (xy, scores)}. Une ligne par jeu et par seuil."""
    lignes = []
    for jeu, liste in jeux.items():
        for s in SEUILS:
            vp = nd = nv = 0
            for nom, _, vrai in liste:
                xy, sc = detections[nom]
                xy = xy[sc >= s]
                vp += apparier(xy * detecteur.PIXEL_M, vrai * detecteur.PIXEL_M)[0]
                nd += len(xy)
                nv += len(vrai)
            lignes.append({"jeu": jeu, "seuil": s, "vp": vp, "detections": nd, "plants": nv})
    t = pd.DataFrame(lignes)
    t["f1"] = 2 * t.vp / (t.detections + t.plants).clip(lower=1)
    t["precision"] = t.vp / t.detections.clip(lower=1)
    t["rappel"] = t.vp / t.plants.clip(lower=1)
    t["erreur_compte"] = (t.detections - t.plants) / t.plants.clip(lower=1)
    return t


def seuil_equilibre(t):
    v = t[t.jeu.isin(VALIDATIONS)].groupby("seuil")
    f1, erreur = v.f1.mean(), v.erreur_compte.apply(lambda e: e.abs().mean())
    candidats = f1[f1 >= f1.max() - ECART_F1].index
    return float(min(candidats, key=lambda s: (round(erreur[s], 6), -f1[s])))


def f1_par_image(detections, liste, seuil):
    """(vp, détections, plants) de chaque image : pour tirer des images au hasard."""
    r = []
    for nom, _, vrai in liste:
        xy, sc = detections[nom]
        xy = xy[sc >= seuil]
        r.append((apparier(xy * detecteur.PIXEL_M, vrai * detecteur.PIXEL_M)[0], len(xy), len(vrai)))
    return np.array(r, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comptage", required=True, type=Path)
    ap.add_argument("--juges", required=True, type=Path)
    ap.add_argument("--modele", required=True, action="append", help="nom=poids_0.pt,poids_1.pt")
    ap.add_argument("--sortie", required=True, type=Path)
    ap.add_argument("--facteur-plantain", type=float, default=1.0)
    ap.add_argument("--palmiers", type=Path, help="dossier du jeu cuboid (inventaire.csv et dezippe/)")
    a = ap.parse_args()
    a.sortie.mkdir(parents=True, exist_ok=True)

    print("Données")
    validation, cible = charger_validation(a.comptage)
    print(f"  bananier médian ReforesTree (apprentissage) : {cible:.1f} px à 5 cm")
    juges = charger_juges(a.juges, cible, a.facteur_plantain)
    jeux = {**validation, **juges}
    for jeu, liste in jeux.items():
        print(f"  {jeu:13s} {len(liste):3d} images, {sum(len(v) for _, _, v in liste):5d} plants")
    palmiers = charger_palmiers(a.palmiers) if a.palmiers else []
    if palmiers:
        ha = sum(im.shape[0] * im.shape[1] for _, im, _ in palmiers) * detecteur.PIXEL_M ** 2 / 1e4
        print(f"  palmiers      {len(palmiers):3d} tuiles de la zone {ZONE_PALMIERS}, {ha:.1f} ha, aucun bananier")
    tout = [x for liste in jeux.values() for x in liste] + palmiers

    tableaux, detections, bilan = {}, {}, {}
    for texte in a.modele:
        nom, chemins = texte.split("=", 1)
        debut = time.time()
        if chemins.startswith("@"):            # détections d'un lancement précédent
            det = relire_detections(chemins[1:], [n for n, _, _ in tout])
            print(f"\n{nom} : détections reprises de {chemins[1:]}")
        else:
            reseaux = detecteur.charger([Path(c) for c in chemins.split(",")])
            det = {n: detecteur.detecter(reseaux, image, seuil=float(SEUILS.min())) for n, image, _ in tout}
            print(f"\n{nom} : {len(reseaux)} réseaux x 4 vues, {time.time() - debut:.0f} s")
        pd.DataFrame([{"image": n, "x": round(float(x), 1), "y": round(float(y), 1), "score": round(float(s), 4)}
                      for n, (xy, sc) in det.items() for (x, y), s in zip(xy, sc)]
                     ).to_csv(a.sortie / f"detections_{nom}.csv", index=False)
        t = noter(det, jeux)
        t.insert(0, "modele", nom)
        if palmiers:
            t["faux_par_ha_palmiers"] = t.seuil.map(lambda s: round(faux_par_ha(det, palmiers, s), 2))
        s = seuil_equilibre(t)
        tableaux[nom], detections[nom] = t, det
        ligne = t[t.seuil == s].set_index("jeu")
        bilan[nom] = {"seuil_equilibre": s,
                      "f1": {j: round(float(ligne.f1[j]), 3) for j in jeux},
                      "erreur_compte": {j: round(float(ligne.erreur_compte[j]), 3) for j in jeux},
                      "f1_validation_moyenne": round(float(ligne.f1[list(VALIDATIONS)].mean()), 3),
                      "f1_juges_moyenne": round(float(ligne.f1[JUGES].mean()), 3)}
        if palmiers:
            bilan[nom]["faux_bananiers_par_ha_palmiers"] = round(faux_par_ha(det, palmiers, s), 2)
        print(f"  seuil d'équilibre {s:.2f}" + (f" ; faux bananiers sur les palmiers : "
              f"{bilan[nom]['faux_bananiers_par_ha_palmiers']:.1f} par ha" if palmiers else ""))
        print(t[t.seuil.isin(sorted({0.25, s}))].round(3).to_string(index=False))

    pd.concat(tableaux.values()).round(4).to_csv(a.sortie / "scores.csv", index=False)

    noms = list(bilan)
    ref = noms[0]
    for nom in noms[1:]:
        b, r = bilan[nom], bilan[ref]
        gain = b["f1_juges_moyenne"] - r["f1_juges_moyenne"]
        pertes = {j: round(b["f1"][j] - r["f1"][j], 3) for j in VALIDATIONS}
        if palmiers:
            # règle de l'essai v5 : moins de faux bananiers sur les palmiers, sans perdre sur les bananiers
            baisse = 1 - b["faux_bananiers_par_ha_palmiers"] / max(r["faux_bananiers_par_ha_palmiers"], 1e-9)
            garde = (baisse >= BAISSE_PALMIERS_MIN - 1e-9 and gain >= -PERTE_JUGES_MAX - 1e-9
                     and min(pertes.values()) >= -PERTE_MAX - 1e-9)
            print(f"\n{nom} contre {ref} : faux bananiers sur les palmiers {r['faux_bananiers_par_ha_palmiers']:.1f} -> "
                  f"{b['faux_bananiers_par_ha_palmiers']:.1f} par ha (baisse {100 * baisse:.0f} %, il faut au moins "
                  f"{100 * BAISSE_PALMIERS_MIN:.0f} %) ; juges {gain:+.3f} (perte max {PERTE_JUGES_MAX})")
        else:
            garde = gain >= GAIN_MIN - 1e-9 and min(pertes.values()) >= -PERTE_MAX - 1e-9
        # Intervalle à 95 % de l'écart sur chaque juge et sur leur moyenne, en tirant des images au hasard
        # (même tirage pour les deux modèles). Pour information : la décision suit la règle, pas l'intervalle.
        rng = np.random.default_rng(0)
        ecarts = {jeu: np.zeros(N_TIRAGES) for jeu in JUGES}
        for jeu in JUGES:
            liste = jeux[jeu]
            pb = f1_par_image(detections[nom], liste, b["seuil_equilibre"])
            pr = f1_par_image(detections[ref], liste, r["seuil_equilibre"])
            for k in range(N_TIRAGES):
                i = rng.integers(0, len(liste), len(liste))
                f = lambda p: 2 * p[i, 0].sum() / max(p[i, 1].sum() + p[i, 2].sum(), 1)
                ecarts[jeu][k] = f(pb) - f(pr)
        ecarts["moyenne des juges"] = np.mean([ecarts[j] for j in JUGES], 0)
        intervalles = {j: {"intervalle_95": [round(float(q), 3) for q in np.percentile(e, [2.5, 97.5])],
                           "part_des_tirages_positifs": round(float((e > 0).mean()), 3)} for j, e in ecarts.items()}
        bilan[nom]["comparaison"] = {"reference": ref, "gain_juges": round(gain, 3), "ecart_validation": pertes,
                                     "tirages": N_TIRAGES, "intervalles": intervalles,
                                     "garde_selon_la_regle": bool(garde)}
        if palmiers:
            bilan[nom]["comparaison"]["baisse_faux_bananiers_palmiers"] = round(baisse, 3)
        regle = "" if palmiers else f" (il faut au moins +{GAIN_MIN})"
        print(f"{'' if palmiers else chr(10)}{nom} contre {ref} : juges {gain:+.3f}{regle}, "
              f"validation {pertes} (perte max {PERTE_MAX}) -> {'GARDÉ' if garde else 'NON GARDÉ'}")
        for j, v in intervalles.items():
            print(f"  {j} : écart dans {v['intervalle_95']} (95 % des tirages), positif dans "
                  f"{100 * v['part_des_tirages_positifs']:.0f} % des tirages")

    (a.sortie / "bilan.json").write_text(json.dumps(bilan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRésultats dans {a.sortie}")


if __name__ == "__main__":
    main()
