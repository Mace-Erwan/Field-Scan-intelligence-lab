"""Jeu Kaggle « juges-v10 » : khadavy (552) et BDR (408) réduits à 896 px de petit côté, plus les poids de la v2.
Sert seulement à NOTER les candidats (aucun apprentissage, aucun réglage). 10/11/2026.

    C:/Field_scan/.venv/Scripts/python.exe -I -B preparer_juges.py
"""
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

sys.stdout.reconfigure(encoding="utf-8")
S = Path("C:/Field_scan/data/traite/kaggle_juges_v10")
TMP = Path("C:/Field_scan/data/traite/juges_v10")
k = pd.read_csv("C:/Field_scan/modeles/maladies_khadavy/tri_distance.csv")
k = pd.DataFrame({"jeu": "khadavy", "classe": k.classe, "fichier": k.fichier, "serie": k.serie, "distance": k.distance})
j = pd.read_csv("C:/Field_scan/modeles/maladies_zoom/reponses_juges.csv")
j = j[j.jeu == "bdr"]
b = pd.DataFrame({"jeu": "bdr", "classe": j.classe, "fichier": j.fichier, "serie": "", "distance": ""})
L = pd.concat([k, b], ignore_index=True)
lignes = []
for n, r in enumerate(L.itertuples()):
    rel = f"images/{r.jeu}/{n:04d}.jpg"
    (TMP / rel).parent.mkdir(parents=True, exist_ok=True)
    with Image.open(r.fichier) as im:
        im.draft("RGB", (896, 896))
        im = ImageOps.exif_transpose(im).convert("RGB")
        w, h = im.size
        if min(w, h) > 896:
            f = 896 / min(w, h)
            im = im.resize((round(w * f), round(h * f)), Image.BILINEAR)
        im.save(TMP / rel, quality=95)
    lignes.append({"jeu": r.jeu, "classe": r.classe, "image": rel, "serie": r.serie, "distance": r.distance, "source": r.fichier})
pd.DataFrame(lignes).to_csv(TMP / "juges.csv", index=False)
S.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(S / "juges_v10.zip", "w", zipfile.ZIP_STORED) as z:
    z.write(TMP / "juges.csv", "juges.csv")
    for rel in pd.DataFrame(lignes).image:
        z.write(TMP / rel, rel)
    v2 = Path("C:/Field_scan/Field-Scan-intelligence-lab/modeles/maladies")
    for f in sorted(v2.glob("*.pt")) + [v2 / "seuils.json"]:
        z.write(f, f"v2/{f.name}")
(S / "dataset-metadata.json").write_text('{"title": "juges-v10", "id": "wawannn1/juges-v10", "licenses": [{"name": "other"}]}')
print(pd.DataFrame(lignes).groupby(["jeu", "classe"]).size().to_string())
print(f"{(S / 'juges_v10.zip').stat().st_size / 1e6:.0f} Mo")
