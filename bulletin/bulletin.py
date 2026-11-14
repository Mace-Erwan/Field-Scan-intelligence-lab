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


VERT, ORANGE, ROUGE, GRIS = "#1a7f37", "#d4a106", "#cf222e", "#8c959f"

NOMS_COLONNES = {
    "parcelle": "Parcelle", "priorite": "Priorité", "surface_ha": "Surface (ha)", "pieds": "Pieds",
    "pieds_par_ha": "Pieds par ha", "vari": "VARI", "couverture_vegetale_%": "Couverture végétale (%)",
    "vigueur_drone_faible_%": "Vigueur faible, drone (%)", "vigueur_faible_%": "Vigueur faible (%)",
    "eau_faible_%": "Manque d'eau (%)", "baisse_ndmi_saison": "Baisse du NDMI entre saisons",
    "photos_feuilles": "Photos de feuilles", "feuilles_malades": "Feuilles malades",
    "feuilles_inconnu": "« Je ne sais pas »", "note_vigueur": "Note vigueur", "note_eau": "Note eau",
    "note_maladies": "Note maladies", "score_sante": "Score de santé",
}
THEMES = {"vigueur": "Vigueur", "eau": "Eau", "maladies": "Maladies", "saisons": "Saisons"}


def _vide(v):
    if v is None or isinstance(v, str):
        return v is None
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _couleur(score):
    if _vide(score):
        return GRIS
    return VERT if score >= 85 else ORANGE if score >= SEUIL_ALERTE else ROUGE


def _nom_colonne(c):
    if c in NOMS_COLONNES:
        return NOMS_COLONNES[c]
    for prefixe in ("ndvi", "ndre", "ndmi"):
        if c.startswith(prefixe + "_"):
            return f"{prefixe.upper()} {c[len(prefixe) + 1:]}"
    return c.replace("_", " ")


def _nombre(v):
    """Nombre à la française : entier sans virgule, sinon 3 décimales sous 1, 1 au-dessus."""
    if _vide(v):
        return ""
    if isinstance(v, (int, np.integer)) or (isinstance(v, (float, np.floating)) and float(v).is_integer()):
        return f"{int(v):,}".replace(",", " ")
    if isinstance(v, (float, np.floating)):
        return (f"{v:.3f}" if abs(v) < 1 else f"{v:.1f}").replace(".", ",")
    return str(v)


def _pastille(score):
    texte = "?" if _vide(score) else int(score)
    return f'<span class="pastille" style="background:{_couleur(score)}">{texte}</span>'


def ecrire_html(chemin, tableau, alertes, fond, parcelles_px, pieds_px, sources, avertissements):
    """Page autonome (sans connexion internet) : chiffres clés, carte des parcelles sur l'image de fond,
    alertes regroupées par parcelle avec leurs images, tableau. fond : (nom du PNG, largeur, hauteur) ;
    parcelles_px : {id: [anneaux de points en pixels du fond]} ; pieds_px : liste de (x, y)."""
    l, h = (fond[1], fond[2]) if fond else (800, 600)
    e = max(l, h) / 900                                   # épaisseurs à l'échelle de la carte
    scores = {r["parcelle"]: r.get("score_sante") for _, r in tableau.iterrows()}

    # carte
    formes = []
    for _, r in tableau.iterrows():
        anneaux = parcelles_px.get(r["parcelle"], [])
        d = " ".join("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in a) + " Z" for a in anneaux)
        c = _couleur(r.get("score_sante"))
        formes.append(f'<path class="parcelle" data-id="{r["parcelle"]}" d="{d}" fill="{c}" fill-opacity="0.22" '
                      f'stroke="{c}" stroke-width="{2.5 * e:.1f}"><title>Parcelle {r["parcelle"]}</title></path>')
        if anneaux:
            xs, ys = zip(*anneaux[0])
            formes.append(f'<text x="{np.mean(xs):.0f}" y="{np.mean(ys):.0f}" class="etiquette" '
                          f'style="font-size:{15 * e:.0f}px;stroke-width:{4 * e:.1f}px">{r["parcelle"]}</text>')
    points = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{2.6 * e:.1f}" class="pied"/>' for x, y in pieds_px)
    image = f'<image href="{fond[0]}" x="0" y="0" width="{l}" height="{h}"/>' if fond else ""
    carte = (f'<svg viewBox="0 0 {l} {h}" class="carte" xmlns="http://www.w3.org/2000/svg">{image}'
             f'{"".join(formes)}<g>{points}</g></svg>')

    # chiffres clés
    cles = [(str(len(tableau)), "parcelles"), (str(len(alertes)), "alertes")]
    if "score_sante" in tableau and tableau.score_sante.notna().any():
        cles.append((str(int(round(tableau.score_sante.mean()))), "score moyen sur 100"))
        cles.append((str(int((tableau.score_sante < SEUIL_ALERTE).sum())), "parcelles sous 70"))
    if "pieds" in tableau and tableau.pieds.notna().any():
        cles.append((_nombre(int(tableau.pieds.sum())), "pieds détectés"))
    if "photos_feuilles" in tableau and tableau.photos_feuilles.notna().any():
        cles.append((f"{int(tableau.feuilles_malades.sum())} / {int(tableau.photos_feuilles.sum())}",
                     "feuilles en alerte"))
    bloc_cles = "".join(f'<div class="cle"><b>{v}</b><span>{html.escape(t)}</span></div>' for v, t in cles)

    # alertes regroupées par parcelle, dans l'ordre de priorité du tableau
    par_parcelle = {}
    for a in alertes:
        par_parcelle.setdefault(a["parcelle"], []).append(a)
    blocs = []
    for _, r in tableau.iterrows():
        liste = par_parcelle.get(r["parcelle"])
        if not liste:
            continue
        lignes, images = [], []
        for a in liste:
            theme = THEMES.get(a.get("theme"), str(a.get("theme", "")).capitalize())
            photo = a.get("theme") == "maladies" and a.get("apercu")
            lien = (f' <a class="lien" href="{a["apercu"]}" target="_blank">voir l’image</a>'
                    if a.get("apercu") and not photo else "")
            lignes.append(f'<li><span class="theme">{html.escape(theme)}</span>{html.escape(a.get("message", ""))}{lien}</li>')
            if photo:                                  # vignettes pour les photos de feuilles seulement
                src = html.escape(str(a.get("source", "")))
                images.append(f'<a href="{a["apercu"]}" target="_blank"><img src="{a["apercu"]}" alt="{src}" '
                              f'title="{html.escape(a.get("message", ""))} ({src})" loading="lazy"></a>')
        galerie = f'<div class="images">{"".join(images)}</div>' if images else ""
        blocs.append(f'<article class="bloc" data-id="{r["parcelle"]}"><header>{_pastille(scores.get(r["parcelle"]))}'
                     f'<h3>Parcelle {r["parcelle"]}</h3><span class="nb">{len(liste)} alerte(s)</span></header>'
                     f'<ul>{"".join(lignes)}</ul>{galerie}</article>')
    bloc_alertes = "".join(blocs) or '<p class="rien">Aucune alerte : toutes les parcelles vont bien.</p>'

    # tableau (les alertes sont déjà à droite de la carte)
    colonnes = [c for c in tableau.columns if c != "alertes"]
    entete = "".join(f"<th>{html.escape(_nom_colonne(c))}</th>" for c in colonnes)
    corps = ""
    for _, r in tableau.iterrows():
        cellules = "".join(f'<td>{_pastille(r[c]) if c == "score_sante" else html.escape(_nombre(r[c]))}</td>'
                           for c in colonnes)
        corps += f'<tr data-id="{r["parcelle"]}">{cellules}</tr>'

    remarques = "".join(f"<li>{html.escape(a)}</li>" for a in avertissements)
    bloc_remarques = f'<div class="remarques"><b>Remarques</b><ul>{remarques}</ul></div>' if remarques else ""
    legende_pieds = '<span><i class="rond"></i>pied détecté</span>' if pieds_px else ""
    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Field Scan : bulletin</title>
<style>
:root {{ --fond: #f5f6f8; --carte: #fff; --texte: #1f2328; --doux: #59636e; --bord: #e1e4e8; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: "Segoe UI", system-ui, -apple-system, sans-serif; color: var(--texte); background: var(--fond); font-size: 14px; line-height: 1.45; }}
.page {{ max-width: 1280px; margin: 0 auto; padding: 24px 20px 40px; }}
.titre {{ display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 8px; margin-bottom: 18px; }}
.titre h1 {{ margin: 0; font-size: 26px; letter-spacing: -0.3px; }}
.titre h1 small {{ font-weight: 400; color: var(--doux); font-size: 16px; margin-left: 10px; }}
.titre .info {{ color: var(--doux); font-size: 13px; text-align: right; }}
.cles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 18px; }}
.cle {{ background: var(--carte); border: 1px solid var(--bord); border-radius: 12px; padding: 14px 16px; }}
.cle b {{ display: block; font-size: 24px; }} .cle span {{ color: var(--doux); font-size: 13px; }}
.corps {{ display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 18px; align-items: start; }}
@media (max-width: 960px) {{ .corps {{ grid-template-columns: 1fr; }} }}
.panneau {{ background: var(--carte); border: 1px solid var(--bord); border-radius: 12px; padding: 14px; }}
.panneau h2 {{ font-size: 16px; margin: 0 0 10px; }}
.carte {{ display: block; width: 100%; height: auto; border-radius: 8px; background: #e9ecef; }}
.parcelle {{ cursor: pointer; transition: fill-opacity .15s; }} .parcelle:hover {{ fill-opacity: 0.4; }}
.parcelle.choisie {{ stroke: #0969da; fill-opacity: 0.45; }}
.etiquette {{ font-weight: 700; fill: #fff; paint-order: stroke; stroke: rgba(0,0,0,.75); text-anchor: middle; dominant-baseline: middle; pointer-events: none; }}
.pied {{ fill: #00d2ff; stroke: #00323d; stroke-width: 0.8; }}
.legende {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; color: var(--doux); font-size: 13px; }}
.legende i {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }}
.legende i.rond {{ border-radius: 50%; background: #00d2ff; border: 1px solid #00323d; }}
.aide {{ color: var(--doux); font-size: 12px; margin: 8px 0 0; }}
#alertes {{ max-height: 820px; overflow: auto; }}
.bloc {{ border: 1px solid var(--bord); border-radius: 10px; padding: 12px; margin-bottom: 12px; }}
.bloc.cachee {{ display: none; }}
.bloc header {{ display: flex; align-items: center; gap: 10px; }} .bloc h3 {{ margin: 0; font-size: 15px; flex: 1; }}
.nb {{ color: var(--doux); font-size: 12px; }}
.pastille {{ display: inline-flex; align-items: center; justify-content: center; min-width: 34px; height: 24px; padding: 0 8px; border-radius: 12px; color: #fff; font-weight: 700; font-size: 12px; }}
.bloc ul {{ margin: 10px 0 0; padding: 0; list-style: none; }} .bloc li {{ margin: 4px 0; }}
.theme {{ display: inline-block; min-width: 76px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .4px; color: var(--doux); }}
.images {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 8px; margin-top: 10px; }}
.images img {{ display: block; width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 8px; border: 1px solid var(--bord); }}
.lien {{ color: #0969da; font-size: 12px; text-decoration: none; margin-left: 4px; }} .lien:hover {{ text-decoration: underline; }}
.rien {{ color: var(--doux); }}
.remarques {{ margin: 18px 0 0; color: var(--doux); font-size: 13px; }} .remarques ul {{ margin: 6px 0 0; padding-left: 18px; }}
.defile {{ overflow-x: auto; margin-top: 18px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ padding: 8px 10px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--bord); }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f0f2f4; font-weight: 600; color: #3b434b; }}
tbody tr {{ cursor: pointer; }} tbody tr:hover td {{ background: #f6f8fa; }} tr.choisie td {{ background: #ddf4ff; }}
</style></head><body><div class="page">
<div class="titre"><h1>Field Scan<small>bulletin de santé par parcelle</small></h1>
<div class="info">Produit le {datetime.now():%d/%m/%Y à %H:%M}<br>{html.escape(", ".join(sources)) or "aucune source"}</div></div>
<div class="cles">{bloc_cles}</div>
<div class="corps">
<section class="panneau"><h2>Carte</h2>{carte}
<div class="legende"><span><i style="background:{VERT}"></i>85 et plus</span><span><i style="background:{ORANGE}"></i>70 à 84</span>
<span><i style="background:{ROUGE}"></i>sous 70</span><span><i style="background:{GRIS}"></i>non calculé</span>{legende_pieds}</div>
<p class="aide">Cliquer sur une parcelle (ou une ligne du tableau) n'affiche que ses alertes. Cliquer à nouveau les affiche toutes.</p></section>
<section class="panneau" id="alertes"><h2>À voir en priorité</h2>{bloc_alertes}</section>
</div>
<section class="panneau defile"><h2>Toutes les parcelles</h2><table><thead><tr>{entete}</tr></thead><tbody>{corps}</tbody></table></section>
{bloc_remarques}
</div>
<script>
let choisie = null;
document.querySelectorAll('.parcelle, tbody tr').forEach(el => el.addEventListener('click', () => {{
  const id = el.dataset.id; choisie = (choisie === id) ? null : id;
  document.querySelectorAll('.parcelle').forEach(p => p.classList.toggle('choisie', p.dataset.id === choisie));
  document.querySelectorAll('tbody tr').forEach(t => t.classList.toggle('choisie', t.dataset.id === choisie));
  document.querySelectorAll('.bloc').forEach(a => a.classList.toggle('cachee', choisie !== null && a.dataset.id !== choisie));
}}));
</script></body></html>"""
    chemin.write_text(page, encoding="utf-8")


def ecrire_tableau(chemin, tableau):
    tableau.to_csv(chemin, index=False, encoding="utf-8")


def ecrire_json(chemin, objet):
    chemin.write_text(json.dumps(objet, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
