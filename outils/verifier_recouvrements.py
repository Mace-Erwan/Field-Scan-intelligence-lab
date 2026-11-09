"""Vérifie qu'un nouveau jeu de comptage ne recouvre ni le test, ni la validation, ni les juges.

Deux contrôles :
  1. images identiques ou presque : empreinte dHash (64 bits) de l'image et de ses 8 orientations (quarts de
     tour, miroirs), au plus 6 bits d'écart, comme dans `verifier_doublons_comptage.py`. Contre toutes les
     images de comptage déjà utilisées (apprentissage compris), toutes les photos brutes de Plants-bana et de
     Count banana, les juges et les photos DJI du test ; et entre images du nouveau jeu ;
  2. même champ vu sur deux images (recadrage, autre photo du même vol) : points SIFT appariés puis
     transformation géométrique cohérente (homographie, RANSAC). Une paire est signalée à partir de
     15 appariements cohérents, puis regardée à l'œil (image enregistrée). Contre le test, la validation,
     les juges et les photos DJI de Plants-bana (dont les 3 photos du test).
Les empreintes proches sont nombreuses entre images de feuillage uniforme ou presque noires (bords
d'orthomosaïque) : chaque paire est à confirmer en la regardant ; seuls les recouvrements avec le test, la
validation ou les juges seraient une fuite.

Usage :
    python outils/verifier_recouvrements.py --donnees C:/Field_scan/data --sortie DOSSIER
        --nouveau banavista=C:/Field_scan/data/traite/roboflow_v4/dezippe/banavista
        --nouveau jorges=C:/Field_scan/data/traite/roboflow_v4/dezippe/jorges
Sorties : empreintes_proches.csv, recouvrements.csv, paires/*.jpg (paires signalées, côte à côte).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

SEUIL_BITS = 6
# Réglage vérifié sur des paires connues : DJI_0544 / 0548 (même champ) et DJI_0544 / 0546, 0546 / 0548
# (recouvrement partiel) donnent 60, 25 et 23 appariements cohérents ; 12 paires sans lien (photos banavista
# contre photos DJI du test) en donnent au plus 6. Avec 1 200 px et 1 500 points, la paire 0544 / 0548
# n'en donnait que 19 : ce réglage plus faible ratait des recouvrements.
COTE_SIFT = 2000          # images réduites à 2 000 px de côté pour les points SIFT
POINTS_SIFT = 5000
RATIO = 0.75              # test du rapport de Lowe (fait dans chaque image de référence)
VOISINS = 10              # voisins gardés pour chaque point avant le test du rapport
REFERENCES_TESTEES = 15   # références testées par nouvelle image (celles qui ont le plus d'appariements)
VOTES_MIN = 10            # appariements bruts minimum pour tester une paire
COHERENTS_MIN = 15        # appariements cohérents (homographie) pour signaler une paire
EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG")


def images_de(dossier):
    return sorted(p for p in Path(dossier).rglob("*") if p.suffix in EXTENSIONS and p.is_file())


def references(donnees):
    """(sift, empreinte) : listes de (groupe, chemin). sift = test, validation, juges, DJI ; empreinte = tout."""
    c = donnees / "traite/comptage_v3"
    im = pd.read_csv(c / "images.csv")
    sift, empreinte = [], []
    for _, r in im.iterrows():
        g = f"{r.jeu}/{r.partie}"
        empreinte.append((g, c / "images" / r.image))
        if r.partie in ("validation", "test"):
            sift.append((g, c / "images" / r.image))
    for jeu in ("plantain", "banano"):
        for p in images_de(donnees / "traite/roboflow_juges" / jeu):
            empreinte.append((f"juge_{jeu}", p))
            if jeu == "banano" or p.parent.name in ("valid", "test"):
                sift.append((f"juge_{jeu}", p))
    for p in images_de(donnees / "annotations_manuelles/verif_jeunes_bananiers"):
        empreinte.append(("dji_test", p))
        sift.append(("dji_test", p))
    for p in images_de(donnees / "brut/plants_banana"):
        g = "plants_bana_DJI" if p.name.startswith("DJI_") else "plants_bana"
        empreinte.append((g, p))
        if g == "plants_bana_DJI":
            sift.append((g, p))
    for p in images_de(donnees / "brut/count_banana_plants"):
        empreinte.append(("count_banana_brut", p))
    return sift, empreinte


def lire_gris(chemin):
    with Image.open(chemin) as im:
        return np.asarray(im.convert("L"))


def dhash(gris):
    petit = np.asarray(Image.fromarray(gris).resize((9, 8), Image.BILINEAR), np.int16)
    return np.packbits(petit[:, 1:] > petit[:, :-1]).tobytes()


def orientations(gris):
    for k in range(4):
        r = np.rot90(gris, k)
        yield r
        yield r[:, ::-1]


def distance(a, b):
    return int(np.unpackbits(np.frombuffer(a, np.uint8) ^ np.frombuffer(b, np.uint8)).sum())


def points_sift(gris, sift):
    f = COTE_SIFT / max(gris.shape)
    if f < 1:
        gris = cv2.resize(gris, (round(gris.shape[1] * f), round(gris.shape[0] * f)), interpolation=cv2.INTER_AREA)
    kp, desc = sift.detectAndCompute(gris, None)
    return np.float32([k.pt for k in kp]).reshape(-1, 2), desc, gris


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donnees", required=True, type=Path)
    ap.add_argument("--nouveau", required=True, action="append", help="nom=dossier")
    ap.add_argument("--sortie", required=True, type=Path)
    ap.add_argument("--sans-empreintes", action="store_true", help="ne refait que le contrôle 2 (SIFT)")
    a = ap.parse_args()
    (a.sortie / "paires").mkdir(parents=True, exist_ok=True)

    nouveaux = [(n, p) for t in a.nouveau for n, d in [t.split("=", 1)] for p in images_de(d)]
    ref_sift, ref_empreinte = references(a.donnees)
    print(f"{len(nouveaux)} nouvelles images ; références : {len(ref_empreinte)} pour les empreintes, "
          f"{len(ref_sift)} pour SIFT")

    if not a.sans_empreintes:
        empreintes(nouveaux, ref_empreinte, a.sortie)
    recouvrements(nouveaux, ref_sift, a.sortie)


def empreintes(nouveaux, ref_empreinte, sortie):
    emp_ref = []
    for g, p in ref_empreinte:
        try:
            emp_ref.append((g, p, dhash(lire_gris(p))))
        except Exception as e:
            print("  illisible :", p, e)
    emp_new = [(n, p, [dhash(o) for o in orientations(lire_gris(p))]) for n, p in nouveaux]
    lignes = []
    for i, (n, p, hs) in enumerate(emp_new):
        for g, q, h in emp_ref:
            d = min(distance(x, h) for x in hs)
            if d <= SEUIL_BITS:
                lignes.append({"nouveau": n, "image": p.name, "groupe": g, "reference": q.name, "bits": d})
        for n2, p2, hs2 in emp_new[i + 1:]:
            d = min(distance(x, hs2[0]) for x in hs)
            if d <= SEUIL_BITS:
                lignes.append({"nouveau": n, "image": p.name, "groupe": f"{n2} (même lot)", "reference": p2.name, "bits": d})
    pd.DataFrame(lignes, columns=["nouveau", "image", "groupe", "reference", "bits"]).to_csv(sortie / "empreintes_proches.csv", index=False)
    print(f"1. empreintes proches : {len(lignes)} paires")


def recouvrements(nouveaux, ref_sift, sortie):
    sift = cv2.SIFT_create(nfeatures=POINTS_SIFT)
    refs = []
    for g, p in ref_sift:
        pts, desc, petit = points_sift(lire_gris(p), sift)
        if desc is not None and len(desc) >= 10:
            refs.append((g, p, pts, desc, petit))
    proprietaire = np.concatenate([np.full(len(r[3]), k) for k, r in enumerate(refs)])
    index_local = np.concatenate([np.arange(len(r[3])) for r in refs])
    apparieur = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
    apparieur.add([np.concatenate([r[3] for r in refs]).astype(np.float32)])
    apparieur.train()
    lignes = []
    for k, (n, p) in enumerate(nouveaux):
        pts, desc, petit = points_sift(lire_gris(p), sift)
        if desc is None or len(desc) < 10:
            continue
        # Test du rapport fait image de référence par image de référence : un même champ peut apparaître sur
        # plusieurs références (photos d'un même vol, copies), et le 2e meilleur voisin global serait alors le
        # même point vu sur une autre photo, ce qui rejetterait à tort l'appariement. Pour chaque point, je
        # garde ses VOISINS plus proches voisins ; dans chaque référence, le meilleur doit battre le 2e de la
        # même référence (ou, s'il n'y en a pas, le plus lointain des voisins gardés).
        votes = {}
        for voisins in apparieur.knnMatch(desc.astype(np.float32), k=VOISINS):
            if not voisins:
                continue
            plafond = voisins[-1].distance
            par_ref = {}
            for m in voisins:
                par_ref.setdefault(proprietaire[m.trainIdx], []).append(m)
            for r, ms in par_ref.items():
                d2 = ms[1].distance if len(ms) > 1 else plafond
                if ms[0].distance < RATIO * d2:
                    votes.setdefault(r, []).append((ms[0].queryIdx, index_local[ms[0].trainIdx]))
        for r, paires in sorted(votes.items(), key=lambda kv: -len(kv[1]))[:REFERENCES_TESTEES]:
            if len(paires) < VOTES_MIN:
                break
            g, q, pts_r, _, petit_r = refs[r]
            src = pts[[i for i, _ in paires]]
            dst = pts_r[[j for _, j in paires]]
            H, masque = cv2.findHomography(src, dst, cv2.RANSAC, 8.0)
            coherents = int(masque.sum()) if masque is not None else 0
            if coherents >= COHERENTS_MIN:
                lignes.append({"nouveau": n, "image": p.name, "groupe": g, "reference": q.name,
                               "appariements": len(paires), "coherents": coherents})
                h = max(petit.shape[0], petit_r.shape[0])
                cote = np.zeros((h, petit.shape[1] + petit_r.shape[1]), np.uint8)
                cote[:petit.shape[0], :petit.shape[1]] = petit
                cote[:petit_r.shape[0], petit.shape[1]:] = petit_r
                cote = cv2.cvtColor(cote, cv2.COLOR_GRAY2BGR)
                for (i, j), ok in zip(paires, masque.ravel()):
                    if ok:
                        x1, y1 = pts[i]; x2, y2 = pts_r[j]
                        cv2.line(cote, (int(x1), int(y1)), (int(x2) + petit.shape[1], int(y2)), (0, 255, 0), 1)
                cv2.imwrite(str(sortie / "paires" / f"{n}__{p.stem}__{g.replace('/', '-')}__{q.stem}.jpg"), cote)
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(nouveaux)} images comparées, {len(lignes)} paires signalées", flush=True)
    pd.DataFrame(lignes, columns=["nouveau", "image", "groupe", "reference", "appariements", "coherents"]).to_csv(
        sortie / "recouvrements.csv", index=False)
    print(f"2. recouvrements (SIFT) : {len(lignes)} paires signalées ; images dans {sortie / 'paires'}")


if __name__ == "__main__":
    main()
