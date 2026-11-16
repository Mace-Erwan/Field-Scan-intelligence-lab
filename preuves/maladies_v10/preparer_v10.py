"""Données en plus pour la v10 maladies (10/11/2026) : cm8zf62kzh, BLADE et godliver, en 448 px comme feuilles_448.

Classes : saine, black_sigatoka, fusarium, autre (toute autre atteinte : insectes, Sigatoka jaune, Moko ou feuille
morte, flétrissement bactérien, Cordana, Pestalotiopsis). La Cordana vient de BLADE (Inde) : la Cordana du Pérou
n'est jamais lue.

Copies (audit du 09/11, `data/traite/audit_nouveaux/paires.csv`, paires confirmées seulement) :
- une nouvelle photo copiée d'un jeu qui sert à juger ou à régler (BDR, khadavy, twakihasan, Tanzanie 2025,
  BananaLSD) est retirée ;
- les copies internes forment des groupes ; un groupe à deux classes est retiré ; le découpage
  apprentissage / validation se fait par groupe (75 / 25, graine 0).

    C:/Field_scan/.venv/Scripts/python.exe -I -B preparer_v10.py
"""
import random
import sys
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

sys.stdout.reconfigure(encoding="utf-8")
A = Path("C:/Field_scan/data/traite/audit_nouveaux")
SORTIE = Path("C:/Field_scan/data/traite/feuilles_v10")
TAILLE = 448
JUGES = {"bdr", "khadavy", "twakihasan", "tanzanie_2025", "bangladesh"}
CLASSES = {
    ("cm8zf62kzh", "Banana healthy leaf"): "saine",
    ("cm8zf62kzh", "Banana Black Sigatoka Disease"): "black_sigatoka",
    ("cm8zf62kzh", "Fusarium wilt panama Disease"): "fusarium",
    ("cm8zf62kzh", "Banana Yellow Sigatoka Disease"): "autre",
    ("cm8zf62kzh", "Banana Insect Pest Disease"): "autre",
    ("cm8zf62kzh", "Banana Moko Disease or dead"): "autre",
    ("godliver", "healthy"): "saine",
    ("godliver", "bbs"): "black_sigatoka",
    ("godliver", "bbw"): "autre",
    ("blade", "Healthy"): "saine",
    ("blade", "Sigatoka"): "black_sigatoka",
    ("blade", "Cordana"): "autre",
    ("blade", "Pestalotiopsis"): "autre",
}

cm8 = pd.read_csv(A / "cm8zf62kzh_groupes.csv")[["jeu", "classe", "source", "petite"]]
autres = pd.read_csv(A / "images_nouvelles.csv")
autres = autres[autres.jeu.isin(["godliver", "blade"])]
N = pd.concat([cm8, autres], ignore_index=True)
N["classe_origine"] = N.classe
N["classe"] = [CLASSES.get((j, c)) for j, c in zip(N.jeu, N.classe_origine)]
print("hors classes (retirées) :", N[N.classe.isna()].groupby(["jeu", "classe_origine"]).size().to_dict())
N = N[N.classe.notna()].reset_index(drop=True)

P = pd.read_csv(A / "paires.csv")
P = P[P.confirmee == True]  # noqa: E712
idx = {p: i for i, p in enumerate(N.petite)}

# 1. copies d'un jeu qui juge ou règle : retirées
copie_juge = set()
for a, b, ja, jb in zip(P.petite_i, P.petite_j, P.jeu_i, P.jeu_j):
    if a in idx and jb in JUGES:
        copie_juge.add(idx[a])
    if b in idx and ja in JUGES:
        copie_juge.add(idx[b])
print("copies d'un jeu qui juge ou règle :", len(copie_juge), N.loc[sorted(copie_juge)].jeu.value_counts().to_dict())

# 2. groupes de copies internes (union-find)
parent = list(range(len(N)))


def racine(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for a, b in zip(P.petite_i, P.petite_j):
    if a in idx and b in idx:
        ra, rb = racine(idx[a]), racine(idx[b])
        if ra != rb:
            parent[ra] = rb
N["groupe"] = [racine(i) for i in range(len(N))]
N = N.drop(index=sorted(copie_juge))
contra = N.groupby("groupe").classe.nunique()
contra = set(contra[contra > 1].index)
print("groupes à deux classes retirés :", len(contra), "photos :", int(N.groupe.isin(contra).sum()))
N = N[~N.groupe.isin(contra)]
# une seule photo par groupe (la plus grande en octets, puis le premier nom)
N["octets"] = [Path(s).stat().st_size for s in N.source]
N = N.sort_values(["groupe", "octets", "source"], ascending=[True, False, True]).drop_duplicates("groupe")

# 3. apprentissage / validation par groupe, 75 / 25, graine 0, dans chaque (jeu, classe)
rng = random.Random(0)
parties = {}
for (_, _), d in N.groupby(["jeu", "classe_origine"]):
    g = sorted(d.groupe)
    rng.shuffle(g)
    coupe = round(len(g) * 0.25)
    for k, x in enumerate(g):
        parties[x] = "validation" if k < coupe else "apprentissage"
N["partie"] = N.groupe.map(parties)

SORTIE.mkdir(parents=True, exist_ok=True)
lignes = []
for k, r in enumerate(N.itertuples()):
    rel = f"images/{r.jeu}/{r.classe}/{r.jeu}_{k:05d}.jpg"
    (SORTIE / rel).parent.mkdir(parents=True, exist_ok=True)
    with Image.open(r.source) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        ImageOps.fit(im, (TAILLE, TAILLE), method=Image.BILINEAR).save(SORTIE / rel, quality=92)
    lignes.append({"source": r.source.split("data\\brut\\")[-1].replace("\\", "/"), "jeu": r.jeu, "classe": r.classe,
                   "classe_origine": r.classe_origine, "image": rel, "groupe": int(r.groupe), "partie": r.partie})
L = pd.DataFrame(lignes)
L.to_csv(SORTIE / "images_v10.csv", index=False)
print(L.groupby(["jeu", "classe", "partie"]).size().unstack(fill_value=0).to_string())

with zipfile.ZipFile(SORTIE.parent / "feuilles_v10.zip", "w", zipfile.ZIP_STORED) as z:
    z.write(SORTIE / "images_v10.csv", "images_v10.csv")
    for rel in L.image:
        z.write(SORTIE / rel, rel)
print("zip :", SORTIE.parent / "feuilles_v10.zip", f"{(SORTIE.parent / 'feuilles_v10.zip').stat().st_size / 1e6:.0f} Mo")
