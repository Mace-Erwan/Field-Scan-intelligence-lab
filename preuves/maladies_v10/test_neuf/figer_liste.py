"""Test neuf de la v10 : liste des photos retenues, figée AVANT toute notation (10/11/2026).

Jeux : Mendeley wfzpdmc5vx (424 photos) et Mendeley 5nfjzntwd8 v3, RAW, dossiers de feuilles seulement.
Copies cherchées contre TOUS nos jeux (apprentissage, validation, juges, tests) avec la méthode du 09/11 :
dHash à 4 bits au plus sur les 8 orientations de chaque photo du test, puis corrélation >= 0,9 en 32 x 32 couleur
et >= 0,9 en 112 x 112 gris dans l'orientation trouvée.
- photo du test copiée d'un de nos jeux : retirée ;
- copies internes au test (dans un jeu ou entre les deux) : un groupe, une seule photo gardée (la plus grande en pixels,
  puis le premier nom) ; un groupe à deux classes est retiré.

    C:/Field_scan/.venv/Scripts/python.exe -I -B figer_liste.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

sys.stdout.reconfigure(encoding="utf-8")
ICI = Path(__file__).parent
BRUT = Path("C:/Field_scan/data/brut")
T = Path("C:/Field_scan/data/traite")
COMPTE = {  # (jeu, dossier) -> (compte comme, réponse juste)
    ("wfzpdmc5vx", "fresh"): "saine",
    ("wfzpdmc5vx", "sigatoka"): "sigatoka",
    ("wfzpdmc5vx", "cordana"): "autre",
    ("wfzpdmc5vx", "pestalotiopsis"): "autre",
    ("5nfjzntwd8", "Healthy Banana  leaf"): "saine",
    ("5nfjzntwd8", "Black and Yellow Sigatoka"): "sigatoka",
    ("5nfjzntwd8", "Panama Wilt Disease"): "fusariose",
    ("5nfjzntwd8", "Chewing insect damage on banana leaf"): "autre",
    ("5nfjzntwd8", "Banana Skipper Damage"): "autre",
}
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ouvrir(p):
    with Image.open(p) as im:
        im.draft("RGB", (256, 256))
        return ImageOps.exif_transpose(im).convert("RGB")


def dhash(im):
    petit = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    return np.packbits((petit[:, 1:] > petit[:, :-1]).flatten())


def v32(im):
    v = np.asarray(im.resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def v112(im):
    v = np.asarray(im.convert("L").resize((112, 112), Image.BILINEAR), dtype=np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


ORIENT = [lambda i: i, lambda i: i.transpose(Image.FLIP_LEFT_RIGHT), lambda i: i.transpose(Image.FLIP_TOP_BOTTOM),
          lambda i: i.transpose(Image.ROTATE_90), lambda i: i.transpose(Image.ROTATE_180), lambda i: i.transpose(Image.ROTATE_270),
          lambda i: i.transpose(Image.TRANSPOSE), lambda i: i.transpose(Image.TRANSVERSE)]
BITS = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

# 1. photos du test
lignes = []
for jeu, racine in (("wfzpdmc5vx", BRUT / "mendeley_wfzpdmc5vx"),
                    ("5nfjzntwd8", BRUT / "mendeley_5nfjzntwd8/dezippe/SHOWROV AZAM BANANA DATASET RAW/RAW IMAGES/RAWWWW")):
    for (j, dossier), compte in COMPTE.items():
        if j != jeu:
            continue
        for f in sorted((racine / dossier).iterdir()):
            if f.suffix.lower() in EXT:
                lignes.append({"jeu": jeu, "classe_origine": dossier, "compte_comme": compte, "fichier": str(f)})
L = pd.DataFrame(lignes)
print("photos téléchargées :", L.groupby(["jeu", "classe_origine"]).size().to_dict())
ims = [ouvrir(f) for f in L.fichier]
L["pixels"] = [Image.open(f).size[0] * Image.open(f).size[1] for f in L.fichier]
H8 = np.stack([[dhash(o(im)) for o in ORIENT] for im in ims])           # (n, 8, 8)
V32 = [[v32(o(im)) for o in ORIENT] for im in ims]
V112 = [[v112(o(im)) for o in ORIENT] for im in ims]

# 2. nos jeux (toutes les images déjà réduites que l'on a)
ref = []
f448 = pd.read_csv(T / "feuilles_448/images.csv")
ref += [(j, str(T / "feuilles_448" / p)) for j, p in zip(f448.jeu, f448.image)]
for csv in (T / "audit_feuilles/images_ajoutees.csv", T / "audit_nouveaux/images_nouvelles.csv"):
    d = pd.read_csv(csv)
    ref += list(zip(d.jeu, d.petite))
d = pd.read_csv(T / "audit_nouveaux/cm8zf62kzh_groupes.csv")
ref += list(zip(d.jeu, d.petite))
for p in (BRUT / "bangladesh_bananalsd").rglob("*"):
    if p.suffix.lower() in EXT and "AugmentedSet" in p.parts:
        ref.append(("bananalsd_augmentee", str(p)))
print("images de nos jeux :", len(ref))
R = pd.DataFrame(ref, columns=["jeu", "chemin"])
RH, RV32, RV112 = [], [], []
for k, p in enumerate(R.chemin):
    im = ouvrir(p)
    RH.append(dhash(im))
    if k % 5000 == 0:
        print("  ", k, flush=True)
RH = np.stack(RH)

copies = []
for i in range(len(L)):
    d = BITS[H8[i][:, None, :] ^ RH[None, :, :]].sum(2)                # (8, n_ref)
    for o, j in zip(*np.nonzero(d <= 4)):
        im = ouvrir(R.chemin[j])
        c32, c112 = float(V32[i][o] @ v32(im)), float(V112[i][o] @ v112(im))
        if c32 >= 0.9 and c112 >= 0.9:
            copies.append({"i": i, "fichier": L.fichier[i], "jeu_ref": R.jeu[j], "ref": R.chemin[j], "orientation": int(o),
                           "bits": int(d[o, j]), "corr32": c32, "corr112": c112})
C = pd.DataFrame(copies)
C.to_csv(ICI / "copies_nos_jeux.csv", index=False)
print("photos du test copiées d'un de nos jeux :", C.i.nunique() if len(C) else 0,
      C.drop_duplicates("i").jeu_ref.value_counts().to_dict() if len(C) else {})

# 3. copies internes au test
parent = list(range(len(L)))


def racine(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


internes = 0
for i in range(len(L)):
    d = BITS[H8[i][:, None, :] ^ H8[:, 0][None, :, :]].sum(2)           # i dans 8 orientations contre j tel quel
    for o, j in zip(*np.nonzero(d <= 4)):
        if j <= i:
            continue
        if float(V32[i][o] @ V32[j][0]) >= 0.9 and float(V112[i][o] @ V112[j][0]) >= 0.9:
            parent[racine(i)] = racine(j)
            internes += 1
L["groupe"] = [racine(i) for i in range(len(L))]
print("paires de copies internes :", internes)
L["copie_nos_jeux"] = L.index.isin(set(C.i) if len(C) else set())
contra = L.groupby("groupe").compte_comme.nunique()
L["groupe_contradictoire"] = L.groupe.isin(set(contra[contra > 1].index))
garde = L[~L.copie_nos_jeux & ~L.groupe_contradictoire].sort_values(["groupe", "pixels", "fichier"], ascending=[True, False, True])
garde = garde.drop_duplicates("groupe")
L.to_csv(ICI / "toutes_les_photos.csv", index=False)
garde[["jeu", "classe_origine", "compte_comme", "fichier", "groupe"]].to_csv(ICI / "liste.csv", index=False)
print("retenues :", len(garde))
print(garde.groupby(["jeu", "classe_origine", "compte_comme"]).size().to_string())
