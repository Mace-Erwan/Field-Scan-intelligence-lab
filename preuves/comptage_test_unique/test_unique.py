"""Test unique du comptage, lancé une seule fois le 10/11/2026 avec le modèle installé (palm3 final).

Protocole fixé le 05/11 (docs/experiences.md, « Protocole du test ») ; détails d'application écrits ici AVANT le
lancement, rien n'est réglé après :
  - modèle : les poids de modeles/comptage/ (comme run.py), 2 réseaux, 4 vues, 16 bits, seuil 0,25 ;
  - notation : une détection est juste à moins de 1 m d'un plant annoté, un plant trouvé une seule fois ;
    F1, précision, rappel, erreur de comptage (détections - plants) / plants ;
  - Nestor (ReforesTree, ferme de test) : image à 5 cm de data/traite/comptage_v5 ; intervalle à 95 % en tirant au
    hasard des blocs de 20 m (400 px), 5 000 tirages, graine 0 ; une paire juste compte dans le bloc du plant ;
  - segmento : les 26 morceaux de test moins les 4 qui bordent la validation (colonne x = 29 440, 303 bananiers) :
    22 morceaux, 2 498 bananiers attendus ; intervalle en tirant des morceaux entiers (5 000 tirages, graine 0) ;
    erreur de comptage totale et moyenne par morceau (en valeur absolue) ;
  - jeunes bananiers (dji_0544, 0546, 0548) : positions des points prises dans la géométrie corrigée sous QGIS
    (pixel = 1 unité, décalage de chaque photo lu dans son .tif) ; photo remise à 5 cm avec 1,1 cm par pixel
    (facteur 0,22, LANCZOS) ; seuls les plants ET les détections à moins de 2 000 px (photo d'origine) du centre
    comptent ; notation « mères seules » (vérité = mères, toutes les détections) et « tous les plants » ;
    rejets comptés comme un pied de plus = rejets appariés à une détection dans la notation « tous les plants » ;
    pas d'intervalle (3 photos).

    C:/Field_scan/.venv/Scripts/python.exe -I -B test_unique.py
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from scipy.optimize import linear_sum_assignment

sys.stdout.reconfigure(encoding="utf-8")
DEPOT = Path(r"C:\Field_scan\Field-Scan-intelligence-lab")
sys.path.insert(0, str(DEPOT))
from comptage import detecteur  # noqa: E402

ICI = Path(__file__).parent
C = Path(r"C:\Field_scan\data\traite\comptage_v5")
DJI = Path(r"C:\Field_scan\data\annotations_manuelles\verif_jeunes_bananiers")
SEUIL = 0.25
PX = detecteur.PIXEL_M
RAYON_M = 1.0
N_TIRAGES, GRAINE = 5000, 0
BLOC_PX = 400                      # 20 m à 5 cm
X_BORD = 29440                     # morceaux de segmento qui touchent la validation
DJI_CM = 0.011
DJI_RAYON_PX = 2000


def paires(det, vrai):
    """Indices (détection, plant) des paires justes à moins de 1 m (positions en pixels à 5 cm)."""
    if len(det) == 0 or len(vrai) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    d = np.linalg.norm(det[:, None] * PX - vrai[None] * PX, axis=2)
    cout = np.where(d <= RAYON_M, d, 1e6)
    i, j = linear_sum_assignment(cout)
    ok = cout[i, j] <= RAYON_M
    return i[ok], j[ok]


def mesures(vp, nd, nv):
    return {"vp": int(vp), "detections": int(nd), "plants": int(nv), "f1": 2 * vp / max(nd + nv, 1),
            "precision": vp / max(nd, 1), "rappel": vp / max(nv, 1), "erreur_compte": (nd - nv) / max(nv, 1)}


def intervalle(unites):
    """unites : (n, 3) vp, détections, plants par unité tirée. Intervalle à 95 % de la F1 et de l'erreur de compte."""
    rng = np.random.default_rng(GRAINE)
    t = unites[rng.integers(0, len(unites), (N_TIRAGES, len(unites)))].sum(1)
    f1 = 2 * t[:, 0] / np.maximum(t[:, 1] + t[:, 2], 1)
    err = (t[:, 1] - t[:, 2]) / np.maximum(t[:, 2], 1)
    return np.percentile(f1, [2.5, 97.5]), np.percentile(err, [2.5, 97.5])


reseaux = detecteur.charger(sorted((DEPOT / "modeles" / "comptage").glob("*.pt")))
print("modèle :", [p.name for p in sorted((DEPOT / "modeles" / "comptage").glob("*.pt"))], "seuil", SEUIL)
images = pd.read_csv(C / "images.csv")
points = pd.read_csv(C / "points.csv")
test = images[images.partie == "test"]
lignes, toutes_det = [], []


def lire(nom):
    return np.asarray(Image.open(C / "images" / nom).convert("RGB")), points[points.image == nom][["x", "y"]].to_numpy(np.float32)


# ---- Nestor
nom = test[test.jeu == "reforestree"].image.item()
img, vrai = lire(nom)
xy, sc = detecteur.detecter(reseaux, img, seuil=SEUIL)
toutes_det += [{"jeu": "nestor", "image": nom, "x": x, "y": y, "score": s} for (x, y), s in zip(xy, sc)]
i, j = paires(xy, vrai)
r = mesures(len(i), len(xy), len(vrai))
bloc = lambda p: (p[:, 0] // BLOC_PX).astype(int) * 1000 + (p[:, 1] // BLOC_PX).astype(int)   # noqa: E731
bv, bd = bloc(vrai), bloc(xy) if len(xy) else np.zeros(0, int)
cles = sorted(set(bv) | set(bd))
u = np.array([[np.isin(j, np.where(bv == k)[0]).sum(), (bd == k).sum(), (bv == k).sum()] for k in cles], float)
(f1b, f1h), (eb, eh) = intervalle(u)
r.update(jeu="Nestor", unites=f"{len(cles)} blocs de 20 m", f1_ic=f"[{f1b:.3f} ; {f1h:.3f}]",
         erreur_ic=f"[{eb:+.3f} ; {eh:+.3f}]", erreur_abs_par_image=abs(r["erreur_compte"]))
lignes.append(r)
print("Nestor fait")

# ---- segmento
seg = test[(test.jeu == "segmento")].copy()
seg["x0"] = seg.image.str.extract(r"segmento_(\d+)_\d+\.jpg$")[0].astype(int)
retires = seg[seg.x0 == X_BORD]
seg = seg[seg.x0 != X_BORD]
n_retires = int(points.image.isin(retires.image).sum())
u, par_image = [], []
for nom in sorted(seg.image):
    img, vrai = lire(nom)
    xy, sc = detecteur.detecter(reseaux, img, seuil=SEUIL)
    toutes_det += [{"jeu": "segmento", "image": nom, "x": x, "y": y, "score": s} for (x, y), s in zip(xy, sc)]
    i, _ = paires(xy, vrai)
    u.append([len(i), len(xy), len(vrai)])
    par_image.append(abs(len(xy) - len(vrai)) / max(len(vrai), 1))
u = np.array(u, float)
r = mesures(*u.sum(0))
(f1b, f1h), (eb, eh) = intervalle(u)
r.update(jeu="segmento", unites=f"{len(u)} morceaux (retirés : {len(retires)}, {n_retires} bananiers)",
         f1_ic=f"[{f1b:.3f} ; {f1h:.3f}]", erreur_ic=f"[{eb:+.3f} ; {eh:+.3f}]",
         erreur_abs_par_image=float(np.mean(par_image)))
lignes.append(r)
print("segmento fait")

# ---- jeunes bananiers (DJI)
pts = gpd.read_file(DJI / "points_jeunes_bananiers.geojson")
facteur = DJI_CM / PX
lignes_dji = []
for photo in sorted(pts.image.unique()):
    with rasterio.open(DJI / f"{photo}.tif") as src:
        inv = ~src.transform
        w, h = src.width, src.height
    p = pts[pts.image == photo]
    col_lig = np.array([inv * (g.x, g.y) for g in p.geometry], np.float32)
    centre = np.array([w / 2, h / 2], np.float32)
    dedans = np.linalg.norm(col_lig - centre, axis=1) < DJI_RAYON_PX
    p, col_lig = p[dedans], col_lig[dedans]
    im = Image.open(DJI / f"{photo}.tif").convert("RGB")
    im = im.resize((round(w * facteur), round(h * facteur)), Image.Resampling.LANCZOS)
    xy, sc = detecteur.detecter(reseaux, np.asarray(im), seuil=SEUIL)
    garde = np.linalg.norm(xy / facteur - centre, axis=1) < DJI_RAYON_PX if len(xy) else np.zeros(0, bool)
    xy, sc = xy[garde], sc[garde]
    toutes_det += [{"jeu": "dji", "image": photo, "x": x, "y": y, "score": s} for (x, y), s in zip(xy, sc)]
    vrai = col_lig * facteur
    mere = (p["type"] == "mere").to_numpy()
    i, _ = paires(xy, vrai[mere])
    rm = mesures(len(i), len(xy), mere.sum())
    i, j = paires(xy, vrai)
    rt = mesures(len(i), len(xy), len(vrai))
    rejets_comptes = int((~mere[j]).sum())
    lignes_dji.append({"photo": photo, "notation": "mères seules", **rm, "rejets_annotes": int((~mere).sum()),
                       "rejets_avec_detection": rejets_comptes})
    lignes_dji.append({"photo": photo, "notation": "tous les plants", **rt, "rejets_annotes": int((~mere).sum()),
                       "rejets_avec_detection": rejets_comptes})
d = pd.DataFrame(lignes_dji)
for notation in ("mères seules", "tous les plants"):
    s = d[d.notation == notation][["vp", "detections", "plants", "rejets_annotes", "rejets_avec_detection"]].sum()
    lignes_dji.append({"photo": "les 3", "notation": notation, **mesures(s.vp, s.detections, s.plants),
                       "rejets_annotes": int(s.rejets_annotes), "rejets_avec_detection": int(s.rejets_avec_detection)})
print("DJI fait")

pd.DataFrame(toutes_det).to_csv(ICI / "detections.csv", index=False)
t1 = pd.DataFrame(lignes)[["jeu", "unites", "plants", "detections", "vp", "f1", "f1_ic", "precision", "rappel",
                           "erreur_compte", "erreur_ic", "erreur_abs_par_image"]]
t2 = pd.DataFrame(lignes_dji)[["photo", "notation", "plants", "detections", "vp", "f1", "precision", "rappel",
                               "erreur_compte", "rejets_annotes", "rejets_avec_detection"]]
texte = ("Test unique du comptage (10/11/2026), modèle installé, seuil 0,25\n\n"
         + t1.round(3).to_string(index=False) + "\n\nJeunes bananiers (photos DJI, 2 000 px autour du centre)\n"
         + t2.round(3).to_string(index=False) + "\n")
(ICI / "resultats.txt").write_text(texte, encoding="utf-8")
print(texte)
