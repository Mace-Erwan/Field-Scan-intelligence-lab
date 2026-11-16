"""Liste figée du test twakihasan (protocole.md, règles « Photos retenues »). Aucun modèle n'est utilisé ici."""
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "C:/Field_scan/Field-Scan-intelligence-lab/outils")
import preparer_feuilles as pf  # noqa: E402

R = Path("C:/Field_scan/modeles/maladies_twakihasan")
t = pd.read_csv("C:/Field_scan/data/traite/audit_nouveaux/twakihasan_groupes.csv")

# 1. une photo par groupe de copies : la plus grande en pixels, puis la première par nom
t["nom"] = [Path(s).name for s in t.source]
t["pixels"] = [np.prod(Image.open(s).size) for s in t.source]
classes = t.groupby("groupe").classe.agg(lambda s: set(s))
t["contradictoire"] = t.groupe.map(lambda g: len(classes[g]) > 1)
t["n_copies"] = t.groupe.map(t.groupe.value_counts())
rep = (t[~t.contradictoire].sort_values(["groupe", "pixels", "nom"], ascending=[True, False, True])
       .drop_duplicates("groupe").copy())
print("groupes contradictoires écartés :", t[t.contradictoire].groupe.nunique())
print("photos retenues :", len(rep), rep.classe.value_counts().to_dict())


# 2. heure de prise de vue lue dans le nom
def heure(nom):
    s = nom.replace("IMG_", "").replace("P", "0")
    m = re.match(r"(\d{8})_(\d{6})", s)
    return datetime.strptime(m[1] + m[2], "%Y%m%d%H%M%S").timestamp() if m else np.nan


rep["heure"] = rep.nom.map(heure)
print("photos sans heure lisible :", int(rep.heure.isna().sum()))

# 3. séries : même classe, <= 60 s d'écart et corrélation >= 0,8 entre photos successives
serie, sid = {}, 0
for c, d in rep.groupby("classe"):
    d = d.sort_values(["heure", "nom"])
    prev_t, prev_v = None, None
    for i, h, p in zip(d.index, d.heure, d.petite):
        v = pf.vecteur(Image.open(p).convert("RGB"))
        lien = (prev_t is not None and not np.isnan(h) and not np.isnan(prev_t) and h - prev_t <= 60
                and float(v @ prev_v) >= 0.8)
        if not lien:
            sid += 1
        serie[i] = sid
        prev_t, prev_v = h, v
rep["serie"] = rep.index.map(serie)
print("séries :", rep.groupby("classe").serie.nunique().to_dict(), "| plus grandes :", rep.serie.value_counts().head(5).to_dict())

out = rep[["source", "classe", "groupe", "n_copies", "serie", "nom", "heure"]].rename(columns={"source": "fichier"})
out.to_csv(R / "liste.csv", index=False)
print("écrit :", R / "liste.csv", len(out), "lignes")
