"""Ablation des bandes Sentinel-2 pour la vigueur et le stress hydrique (09/11/2026).

Question du sujet : qu'apporte chaque bande ? Je refais les cartes de run.py en changeant UNE chose à la fois
(l'indice ou la bande), avec exactement les mêmes règles et les mêmes masques (parcelles, nuages), et je compare :
part de la plantation signalée, parcelles en alerte (note < 70, comme le bulletin), et accord avec la carte actuelle
(indice de Jaccard sur les pixels signalés). Pas d'apprentissage, pas de vérité terrain : on mesure ce que change
chaque bande, pas laquelle a « raison ».
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "C:/Field_scan/Field-Scan-intelligence-lab")
import entrees  # noqa: E402
from indices import sentinel2, stress, vigueur  # noqa: E402

OUT = Path("C:/Field_scan/analyses/ablation_bandes")
SITES = {"equateur": "C:/Field_scan/data/parcelles/parcelles_equateur.geojson",
         "cameroun": "C:/Field_scan/data/parcelles/parcelles_cameroun.geojson"}
SEUIL_NOTE = 70                       # bulletin : alerte si la note (100 x part sans problème) est sous 70


def nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (a - b) / (a + b)
    x[~np.isfinite(x)] = np.nan
    return x


def jaccard(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def seche_brut(ndmi, dans):
    """Variante sans correction par le feuillage : même règle relative que la vigueur (80 % du 75e centile)."""
    return vigueur.zones_faibles(ndmi, dans)[0]


COURTS = {"NDVI (B08/B04, actuel)": "NDVI (actuel)", "NDRE (B8A/B05, bord du rouge)": "NDRE",
          "GNDVI (B08/B03, vert au lieu du rouge)": "GNDVI", "VARI (B02/B03/B04, sans infrarouge)": "VARI (visible)",
          "ExG (B02/B03/B04, sans infrarouge)": "ExG (visible)",
          "NDMI B8A/B11 corrigé par le NDVI (actuel)": "eau B11 (actuel)",
          "NDMI B8A/B12 (infrarouge moyen 2) corrigé par le NDVI": "eau B12",
          "NDMI B8A/B11 corrigé par le NDRE au lieu du NDVI": "eau, corr. NDRE",
          "NDMI B8A/B11 sans correction (règle des 80 %)": "eau sans corr."}


def planche(site, par_date):
    """Une ligne par date, une colonne par variante. Blanc : hors parcelles ; vert pâle : parcelle sans alerte ;
    rouge : pixel signalé ; bleu pâle : nuage ou ombre (exclu)."""
    from PIL import Image, ImageDraw, ImageFont
    police = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
    dates = sorted(par_date)
    h, w = next(iter(par_date.values()))[1].shape
    k = max(1, min(round(260 / h), round(200 / w)))
    noms = list(next(iter(par_date.values()))[0])
    marge_g, marge_h, ecart = 95, 44, 8
    col = max(w * k, 120)
    W = marge_g + len(noms) * (col + ecart)
    H = marge_h + len(dates) * (h * k + ecart)
    out = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(out)
    for j, (theme, nom) in enumerate(noms):
        dr.multiline_text((marge_g + j * (col + ecart), 4), COURTS[nom].replace(" (", chr(10) + "("), fill="black", font=police)
    for i, d in enumerate(dates):
        cartes, dans, exclus, interieur = par_date[d]
        dr.text((5, marge_h + i * (h * k + ecart) + h * k // 2 - 8), d, fill="black", font=police)
        for j, cle in enumerate(noms):
            rgb = np.full((h, w, 3), 255, np.uint8)
            rgb[interieur & exclus] = (190, 215, 240)
            rgb[dans] = (205, 230, 195)
            rgb[cartes[cle] & dans] = (215, 40, 40)
            tuile = Image.fromarray(rgb).resize((w * k, h * k), Image.NEAREST)
            out.paste(tuile, (marge_g + j * (col + ecart), marge_h + i * (h * k + ecart)))
    out.save(OUT / f"planche_{site}.png")


lignes, parcelles_alerte, saisons = [], [], []
for site, fichier_parcelles in SITES.items():
    dates = entrees.sentinel2(Path("C:/Field_scan/data/brut/sentinel2") / site)
    parcelles = entrees.parcelles(Path(fichier_parcelles))
    ndmi_med = {"B11": {}, "B12": {}}
    faibles_par_date, pour_planche = {}, {}
    for d, bandes in sorted(dates.items()):
        _, profil, exclus = sentinel2.lire_date(bandes)
        B = {}
        for nom in ("B02", "B03", "B04", "B05", "B08", "B8A", "B11", "B12"):
            with rasterio.open(bandes[nom]) as src:
                B[nom] = src.read(1).astype("float32")
        p = parcelles.to_crs(profil["crs"])
        masques, dans = {}, np.zeros(exclus.shape, bool)
        interieur = ~geometry_mask(list(p.geometry), out_shape=exclus.shape, transform=profil["transform"])
        for _, r in p.iterrows():
            m = geometry_mask([r.geometry], out_shape=exclus.shape, transform=profil["transform"], invert=True) & ~exclus
            if m.any():
                masques[r["id"]] = m
                dans |= m
        ndvi = nd(B["B08"], B["B04"])
        idx = {"NDVI (B08/B04, actuel)": ndvi,
               "NDRE (B8A/B05, bord du rouge)": nd(B["B8A"], B["B05"]),
               "GNDVI (B08/B03, vert au lieu du rouge)": nd(B["B08"], B["B03"])}
        with np.errstate(divide="ignore", invalid="ignore"):
            vari = (B["B03"] - B["B04"]) / (B["B03"] + B["B04"] - B["B02"])
            exg = (2 * B["B03"] - B["B04"] - B["B02"]) / (B["B03"] + B["B04"] + B["B02"])
        vari[~np.isfinite(vari) | (np.abs(vari) > 1)] = np.nan
        exg[~np.isfinite(exg)] = np.nan
        idx["VARI (B02/B03/B04, sans infrarouge)"] = vari
        idx["ExG (B02/B03/B04, sans infrarouge)"] = exg

        cartes = {}
        for nom, x in idx.items():
            cartes[("vigueur", nom)] = vigueur.zones_faibles(x, dans)[0]
        faible = cartes[("vigueur", "NDVI (B08/B04, actuel)")]
        faibles_par_date[d] = {i: float(faible[m].mean()) for i, m in masques.items()}
        ndmi11, ndmi12 = nd(B["B8A"], B["B11"]), nd(B["B8A"], B["B12"])
        cartes[("eau", "NDMI B8A/B11 corrigé par le NDVI (actuel)")] = stress.zones_seches(ndvi, ndmi11, dans & ~faible)[0]
        cartes[("eau", "NDMI B8A/B12 (infrarouge moyen 2) corrigé par le NDVI")] = stress.zones_seches(ndvi, ndmi12, dans & ~faible)[0]
        cartes[("eau", "NDMI B8A/B11 corrigé par le NDRE au lieu du NDVI")] = stress.zones_seches(idx["NDRE (B8A/B05, bord du rouge)"], ndmi11, dans & ~faible)[0]
        cartes[("eau", "NDMI B8A/B11 sans correction (règle des 80 %)")] = seche_brut(ndmi11, dans & ~faible)
        for i, m in masques.items():
            ndmi_med["B11"].setdefault(d, {})[i] = float(np.nanmedian(ndmi11[m]))
            ndmi_med["B12"].setdefault(d, {})[i] = float(np.nanmedian(ndmi12[m]))

        pour_planche[d] = (cartes, dans, exclus, interieur)
        ref = {"vigueur": cartes[("vigueur", "NDVI (B08/B04, actuel)")],
               "eau": cartes[("eau", "NDMI B8A/B11 corrigé par le NDVI (actuel)")]}
        for (theme, nom), c in cartes.items():
            alerte = sorted(int(i) for i, m in masques.items() if 100 * (1 - c[m].mean()) < SEUIL_NOTE)
            lignes.append({"site": site, "date": d, "theme": theme, "variante": nom,
                           "part_signalee_%": round(100 * c[dans].mean(), 1),
                           "jaccard_avec_actuel": round(jaccard(c & dans, ref[theme] & dans), 2),
                           "parcelles_en_alerte": " ".join(map(str, alerte)) or "aucune"})

    planche(site, pour_planche)
    # saisons : baisse du NDMI entre la date humide et la date sèche, avec B11 (actuel) ou B12
    actuelle = max(dates)
    comparables = [i for i in faibles_par_date[actuelle] if all(faibles_par_date[d].get(i, 1) < 0.3 for d in dates)]
    for bande, med in ndmi_med.items():
        humide, sec, baisse, signales = stress.comparer_saisons(med, comparables)
        saisons.append({"site": site, "bande_infrarouge_moyen": bande, "date_humide": humide, "date_seche": sec,
                        "parcelles_signalees": " ".join(str(int(i)) for i in signales) or "aucune"})

T = pd.DataFrame(lignes)
S = pd.DataFrame(saisons)
T.to_csv(OUT / "ablation.csv", index=False)
S.to_csv(OUT / "saisons.csv", index=False)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 60)
print(T.to_string(index=False))
print()
print(S.to_string(index=False))
