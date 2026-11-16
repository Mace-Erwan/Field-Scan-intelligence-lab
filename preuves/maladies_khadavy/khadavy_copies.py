"""khadavy : préparation 224 px et recherche de copies (dans le jeu, et contre nos jeux de feuilles + BDR).

Même méthode que outils/preparer_feuilles.py (empreinte dHash <= 4 bits, puis corrélation >= 0,9 en 32 x 32 couleur
et >= 0,9 en 112 x 112 gris), avec en plus les 8 orientations de chaque photo khadavy (copies tournées ou retournées).
Aucune photo n'est passée dans le modèle ici.
"""
import sys

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

sys.path.insert(0, "C:/Field_scan/Field-Scan-intelligence-lab/outils")
import preparer_feuilles as pf  # noqa: E402

BRUT = Path("C:/Field_scan/data/brut/khadavy/Klasifikasi Daun Pisang.v1i.folder")
BDR = Path("C:/Field_scan/data/brut/bangladesh_bdr/dezippe/Original Images")
FEUILLES = Path("C:/Field_scan/data/traite/feuilles")
SORTIE = Path("C:/Field_scan/data/traite/khadavy")
RES = Path("C:/Field_scan/modeles/maladies_khadavy")
(SORTIE / "images").mkdir(parents=True, exist_ok=True)


def petite_image(f):
    with Image.open(f) as im:
        im.draft("RGB", (pf.TAILLE * 2, pf.TAILLE * 2))
        im = ImageOps.exif_transpose(im).convert("RGB")
        return pf.preparer(im)


def orientations(im):
    out = []
    for flip in (False, True):
        base = ImageOps.mirror(im) if flip else im
        for k in range(4):
            out.append(base.rotate(90 * k, expand=True))
    return out


# 1. khadavy : préparation, empreintes des 8 orientations
lignes, H8, P = [], [], []
for f in sorted(BRUT.rglob("*.jpg")):
    partie, classe = f.parts[-3], f.parts[-2]
    nom = re.sub(r"_(jpe?g|png)\.rf\.[0-9a-f]+\.jpg$", "", f.name, flags=re.I)
    sortie = SORTIE / "images" / classe / f"{partie}__{f.stem}.jpg"
    sortie.parent.mkdir(parents=True, exist_ok=True)
    im = petite_image(f)
    im.save(sortie, quality=95)
    P.append(im)
    H8.append(np.stack([pf.empreinte(o) for o in orientations(im)]))
    lignes.append({"fichier": str(f), "partie": partie, "classe": classe, "nom": nom, "petite": str(sortie)})
k = pd.DataFrame(lignes)
H8 = np.stack(H8)                       # (n, 8, 8 octets)
n = len(k)
print("khadavy :", n, "photos préparées")
TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def confirme(a, b):
    """a, b : images 224 (b déjà orientée). Renvoie (corr32, corr112)."""
    return float(pf.vecteur(a) @ pf.vecteur(b)), float(pf.vecteur_fin(a) @ pf.vecteur_fin(b))


# 2. copies à l'intérieur de khadavy
paires = []
d = TABLE[H8[:, 0, :][:, None, None, :] ^ H8[None, :, :, :]].sum(-1)      # (n, n, 8) : i à l'endroit contre j orienté
for i in range(n):
    for j in range(i + 1, n):
        o = int(d[i, j].argmin())
        if d[i, j, o] <= pf.SEUIL_DOUBLON:
            c32, c112 = confirme(P[i], orientations(P[j])[o])
            paires.append((i, j, int(d[i, j, o]), o, c32, c112))
pp = pd.DataFrame(paires, columns=["i", "j", "bits", "orientation", "corr32", "corr112"])
conf = pp[(pp.corr32 >= pf.SEUIL_CORRELATION) & (pp.corr112 >= pf.SEUIL_CORRELATION_FINE)]
print(f"dans khadavy : {len(pp)} paires candidates, {len(conf)} confirmées")
for c in ("classe", "nom", "petite"):
    pp[c + "_i"], pp[c + "_j"] = k[c].values[pp.i], k[c].values[pp.j]
pp.to_csv(RES / "copies_internes_candidates.csv", index=False)

# groupes de copies confirmées (union-find)
parent = list(range(n))


def racine(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for i, j in zip(conf.i, conf.j):
    parent[racine(i)] = racine(j)
k["groupe"] = [racine(i) for i in range(n)]
g = k.groupby("groupe").classe.agg(["size", "nunique"])
print("groupes :", len(g), "| groupes avec 2 classes :", int((g["nunique"] > 1).sum()))
print(k.drop_duplicates("groupe").classe.value_counts().to_dict(), "(une photo par groupe)")

# 3. contre nos jeux (Tanzanie 2023 / 2025, Pérou, Bangladesh BananaLSD) et BDR
ref = pd.read_csv(FEUILLES / "images.csv")
ref["chemin"] = [str(FEUILLES / p) for p in ref.image]
bdr = []
for f in sorted(BDR.rglob("*")):
    if f.suffix.lower() in pf.EXTENSIONS:
        im = petite_image(f)
        bdr.append({"jeu": "bdr", "classe": f.parent.name, "source": str(f), "chemin": str(f),
                    "empreinte": pf.empreinte(im).tobytes().hex()})
ref = pd.concat([ref, pd.DataFrame(bdr)], ignore_index=True)
HR = np.stack([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in ref.empreinte])
print("références :", ref.jeu.value_counts().to_dict())
croisees = []
for i in range(n):
    dd = TABLE[H8[i][:, None, :] ^ HR[None, :, :]].sum(-1)   # (8, m)
    m = dd.min(0)
    for r in np.nonzero(m <= pf.SEUIL_DOUBLON)[0]:
        o = int(dd[:, r].argmin())
        b = petite_image(ref.chemin[r]) if ref.jeu[r] == "bdr" else Image.open(ref.chemin[r]).convert("RGB")
        c32, c112 = confirme(orientations(P[i])[o], b)
        croisees.append({"khadavy": k.petite[i], "classe": k.classe[i], "jeu": ref.jeu[r], "ref": ref.chemin[r],
                         "bits": int(m[r]), "corr32": c32, "corr112": c112})
cr = pd.DataFrame(croisees, columns=["khadavy", "classe", "jeu", "ref", "bits", "corr32", "corr112"])
cr.to_csv(RES / "copies_croisees_candidates.csv", index=False)
cc = cr[(cr.corr32 >= pf.SEUIL_CORRELATION) & (cr.corr112 >= pf.SEUIL_CORRELATION_FINE)]
print(f"contre nos jeux : {len(cr)} paires candidates, {len(cc)} confirmées", cc.jeu.value_counts().to_dict())
k.to_csv(RES / "khadavy_images.csv", index=False)
