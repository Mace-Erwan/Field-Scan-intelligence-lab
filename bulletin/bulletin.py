"""Bulletin par parcelle : notes, score de santé, alertes, et les fichiers qui les montrent.

Notes sur 100 (seulement celles que les images permettent de calculer) :
  - vigueur : part de la parcelle qui n'est pas en zone de vigueur faible
    (Sentinel-2 s'il y en a, sinon l'image drone) ;
  - eau : part de la parcelle qui ne manque pas d'eau (Sentinel-2) ;
  - maladies : part des photos de feuilles de la parcelle dites saines (une feuille malade ou
    « je ne sais pas » fait baisser la note).
Score de santé = moyenne des notes disponibles. Une note sous 70 déclenche une alerte, comme une
feuille malade ou une parcelle dont l'eau baisse plus que les autres entre les saisons. Chaque
alerte renvoie à l'image qui la justifie (aperçu dans apercus/, nom du fichier d'origine).
"""

import html
import json
from datetime import datetime

import numpy as np
import pandas as pd

SEUIL_ALERTE = 70


def notes_et_alertes(parcelles, mesures):
    """parcelles : GeoDataFrame (id, surface_ha) ; mesures : {id: dict de valeurs mesurées}.
    Renvoie le tableau du bulletin (une ligne par parcelle) et la liste des alertes."""
    lignes, alertes = [], []
    for _, p in parcelles.iterrows():
        m = mesures.get(p["id"], {})
        ligne = {"parcelle": p["id"], "surface_ha": round(float(p["surface_ha"]), 2)}
        ligne.update({k: v for k, v in m.items() if not k.startswith("_")})
        notes = {k: m[k] for k in ("note_vigueur", "note_eau", "note_maladies") if m.get(k) is not None}
        ligne["score_sante"] = round(float(np.mean(list(notes.values())))) if notes else None
        textes = []
        for nom, note in notes.items():
            if note < SEUIL_ALERTE:
                theme = {"note_vigueur": "vigueur", "note_eau": "eau", "note_maladies": "maladies"}[nom]
                detail = m.get(f"_alerte_{theme}", {})
                textes.append(detail.get("message", f"{theme} : note {note}"))
                alertes.append({"parcelle": p["id"], "theme": theme, "note": note, **detail})
        for detail in m.get("_alertes_supplementaires", []):
            textes.append(detail["message"])
            alertes.append({"parcelle": p["id"], **detail})
        ligne["alertes"] = " ; ".join(textes)
        lignes.append(ligne)
    tableau = pd.DataFrame(lignes)
    if "score_sante" in tableau and tableau.score_sante.notna().any():
        tableau = tableau.sort_values("score_sante", na_position="last")
    tableau.insert(1, "priorite", range(1, len(tableau) + 1))
    return tableau, alertes


def _couleur(score):
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return "#9aa0a6"
    return "#2e7d32" if score >= 85 else "#f9a825" if score >= SEUIL_ALERTE else "#c62828"


def ecrire_html(chemin, tableau, alertes, fond, parcelles_px, pieds_px, sources, avertissements):
    """Page autonome (sans connexion internet) : carte des parcelles sur l'image de fond, tableau,
    alertes avec leur image source. fond : (nom du PNG, largeur, hauteur) ; parcelles_px : {id: [anneaux
    de points en pixels du fond]} ; pieds_px : liste de (x, y)."""
    l, h = (fond[1], fond[2]) if fond else (800, 600)
    formes = []
    for _, r in tableau.iterrows():
        anneaux = parcelles_px.get(r["parcelle"], [])
        d = " ".join("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in a) + " Z" for a in anneaux)
        score = r.get("score_sante")
        formes.append(f'<path class="parcelle" data-id="{r["parcelle"]}" d="{d}" fill="{_couleur(score)}" '
                      f'fill-opacity="0.35" stroke="{_couleur(score)}" stroke-width="2"><title>Parcelle {r["parcelle"]}'
                      f' : score {score if score is not None else "—"}</title></path>')
        if anneaux:
            xs, ys = zip(*anneaux[0])
            formes.append(f'<text x="{np.mean(xs):.0f}" y="{np.mean(ys):.0f}" class="etiquette">{r["parcelle"]}</text>')
    points = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" class="pied"/>' for x, y in pieds_px)
    image = f'<image href="{fond[0]}" x="0" y="0" width="{l}" height="{h}"/>' if fond else ""
    carte = (f'<svg viewBox="0 0 {l} {h}" class="carte" xmlns="http://www.w3.org/2000/svg">{image}'
             f'{"".join(formes)}<g>{points}</g></svg>')

    colonnes = [c for c in tableau.columns if c != "alertes"] + ["alertes"]
    entete = "".join(f"<th>{html.escape(str(c))}</th>" for c in colonnes)
    corps = ""
    for _, r in tableau.iterrows():
        cellules = "".join(f"<td>{'' if pd.isna(r[c]) else html.escape(str(r[c]))}</td>" for c in colonnes)
        corps += f'<tr data-id="{r["parcelle"]}">{cellules}</tr>'

    cartes_alertes = ""
    for a in alertes:
        apercu = f'<a href="{a["apercu"]}" target="_blank"><img src="{a["apercu"]}" alt="image source"></a>' if a.get("apercu") else ""
        cartes_alertes += (f'<div class="alerte" data-id="{a["parcelle"]}"><b>Parcelle {a["parcelle"]}</b> — '
                           f'{html.escape(a.get("message", ""))}<br><span class="source">Image source : '
                           f'{html.escape(str(a.get("source", "—")))}</span>{apercu}</div>')
    if not alertes:
        cartes_alertes = "<p>Aucune alerte.</p>"

    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Field Scan — bulletin</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px; color: #1f2328; background: #fff; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }} .sous-titre {{ color: #57606a; margin-bottom: 16px; }}
.haut {{ display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }}
.carte {{ width: 100%; max-width: 760px; border: 1px solid #d0d7de; border-radius: 8px; background: #f6f8fa; }}
.parcelle {{ cursor: pointer; }} .parcelle.choisie {{ stroke: #0969da; stroke-width: 4; fill-opacity: 0.55; }}
.etiquette {{ font-size: 14px; font-weight: 600; fill: #fff; paint-order: stroke; stroke: #000; stroke-width: 3px; text-anchor: middle; pointer-events: none; }}
.pied {{ fill: #00e5ff; stroke: #003a40; stroke-width: 0.6; }}
#alertes {{ flex: 1; min-width: 280px; max-height: 640px; overflow: auto; }}
.alerte {{ border: 1px solid #d0d7de; border-left: 4px solid #c62828; border-radius: 6px; padding: 8px; margin-bottom: 8px; }}
.alerte img {{ display: block; max-width: 100%; max-height: 280px; margin-top: 6px; border-radius: 4px; }}
.alerte.cachee {{ display: none; }} .source {{ color: #57606a; font-size: 12px; }}
table {{ border-collapse: collapse; margin-top: 16px; font-size: 13px; }}
th, td {{ border: 1px solid #d0d7de; padding: 4px 6px; text-align: left; }} th {{ background: #f6f8fa; }}
tr.choisie td {{ background: #ddf4ff; }} .legende span {{ display: inline-block; width: 12px; height: 12px; margin: 0 4px 0 12px; }}
.avert {{ color: #9a6700; font-size: 13px; }}
</style></head><body>
<h1>Field Scan — bulletin par parcelle</h1>
<div class="sous-titre">Produit le {datetime.now():%d/%m/%Y à %H:%M}. Sources : {html.escape(", ".join(sources)) or "aucune"}.</div>
{"".join(f'<p class="avert">{html.escape(a)}</p>' for a in avertissements)}
<div class="legende">Score de santé :<span style="background:#2e7d32"></span>85 et plus
<span style="background:#f9a825"></span>70 à 84<span style="background:#c62828"></span>sous 70
<span style="background:#9aa0a6"></span>non calculé{" · points bleus : pieds détectés" if pieds_px else ""}.
Cliquer sur une parcelle n'affiche que ses alertes ; cliquer à nouveau les affiche toutes.</div>
<div class="haut">{carte}<div id="alertes"><h2>Alertes</h2>{cartes_alertes}</div></div>
<table><thead><tr>{entete}</tr></thead><tbody>{corps}</tbody></table>
<script>
let choisie = null;
document.querySelectorAll('.parcelle, tbody tr').forEach(el => el.addEventListener('click', () => {{
  const id = el.dataset.id; choisie = (choisie === id) ? null : id;
  document.querySelectorAll('.parcelle').forEach(p => p.classList.toggle('choisie', p.dataset.id === choisie));
  document.querySelectorAll('tbody tr').forEach(t => t.classList.toggle('choisie', t.dataset.id === choisie));
  document.querySelectorAll('.alerte').forEach(a => a.classList.toggle('cachee', choisie !== null && a.dataset.id !== choisie));
}}));
</script></body></html>"""
    chemin.write_text(page, encoding="utf-8")


def ecrire_tableau(chemin, tableau):
    tableau.to_csv(chemin, index=False, encoding="utf-8")


def ecrire_json(chemin, objet):
    chemin.write_text(json.dumps(objet, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
