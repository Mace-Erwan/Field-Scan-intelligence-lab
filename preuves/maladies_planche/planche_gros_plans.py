"""Planche du README : gros plans du jeu de test AgroVisionBD, recadrés AVANT de lancer le modèle,
puis le modèle installé lit exactement l'image recadrée qui est affichée."""
import sys
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont
DEPOT = Path(r"C:\Field_scan\Field-Scan-intelligence-lab"); sys.path.insert(0, str(DEPOT))
from maladies import classifieur
R = Path(r"C:\Field_scan\data\brut\agrovisionbd")
G = Path(r"C:\Field_scan\brouillons_run\gros_plans"); OUT = G / "recadrees"; OUT.mkdir(exist_ok=True)
def nom(c, k):
    return {int(l.split("\t")[0]): l.split("\t")[2] for l in (G / f"{c}.txt").read_text(encoding="utf-8").splitlines()}[k]
S, F, H = "banana_black_sigatoka_leaf", "banana_panama_leaf", "banana_healthy_leaf"
CHOIX = [  # (classe, rang, centre x, centre y) ; carré de côté = petit côté de la photo
    (S, 16, .5, .5, 1., "cercosporiose"), (S, 22, .5, .5, 1., "cercosporiose"), (S, 39, .5, .5, 1., "cercosporiose"),
    (F, 22, .55, .3, .7, "fusariose"), (F, 5, .35, .62, .65, "fusariose"), (F, 21, .45, .28, .75, "fusariose"),
    (H, 0, .5, .5, 1., "saine"), (H, 10, .6, .6, .7, "saine"), (H, 19, .5, 0., 1., "saine")]
chemins, vrais = [], []
for c, k, cx, cy, fs, v in CHOIX:
    im = ImageOps.exif_transpose(Image.open(R / c / nom(c, k))).convert("RGB")
    w, h = im.size; s = int(min(w, h) * fs)
    x0 = min(max(int(cx * w - s / 2), 0), w - s); y0 = min(max(int(cy * h - s / 2), 0), h - s)
    p = OUT / f"{len(chemins):02d}_{nom(c, k)}"
    im.crop((x0, y0, x0 + s, y0 + s)).resize((768, 768), Image.LANCZOS).save(p, quality=95)
    chemins.append(p); vrais.append(v)
membres, seuils = classifieur.charger(DEPOT / "modeles" / "maladies")
probas = classifieur.probabilites(membres, chemins)
reps = classifieur.decider(probas, seuils)
REP = {"black_sigatoka": ("cercosporiose", "#cf222e"), "fusarium": ("fusariose", "#cf222e"),
       "malade_non_identifiee": ("malade, non identifiée", "#cf222e"), "saine": ("saine", "#1a7f37"),
       "inconnu": ("à vérifier", "#b45309")}
C, M, HT = 360, 12, 92
f1 = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 21); f2 = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 19)
out = Image.new("RGB", (3 * C + 4 * M, 3 * (C + HT + M) + M), "white"); d = ImageDraw.Draw(out)
for k, (p, v, pr, r) in enumerate(zip(chemins, vrais, probas, reps)):
    i, j = divmod(k, 3); x, y = M + j * (C + M), M + i * (C + HT + M)
    out.paste(Image.open(p).resize((C, C), Image.LANCZOS), (x, y))
    n, col = REP.get(r, (r, "#6e7781"))
    d.text((x + 2, y + C + 8), f"vraie classe : {v}", font=f2, fill="#57606a")
    d.text((x + 2, y + C + 36), f"modèle : {n}", font=f1, fill=col)
    d.text((x + 2, y + C + 64), f"P(saine) = {pr[0]:.2f}".replace(".", ","), font=f2, fill="#57606a")
    print(p.name, v, r, [round(float(q), 3) for q in pr])
out.save(DEPOT / "docs" / "images" / "maladies_demo_reponses.jpg", quality=88)
