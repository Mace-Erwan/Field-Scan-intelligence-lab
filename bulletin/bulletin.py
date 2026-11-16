"""Bulletin par parcelle : notes, score de santé, alertes, et les fichiers qui les montrent.

Notes sur 100 (seulement celles que les images permettent de calculer) :
  - vigueur : part de la parcelle qui n'est pas en zone de vigueur faible, face à la plantation ou face au
    reste de la parcelle (Sentinel-2 s'il y en a, sinon l'image drone) ;
  - eau : part de la parcelle qui ne manque pas d'eau (Sentinel-2) ;
  - maladies, par gravité : 100 si toutes les feuilles sont saines, 75 s'il n'y a que des « je ne sais pas »
    (à vérifier), sous 70 dès qu'une feuille est malade.
Score de santé = la pire des notes : le plus gros problème de la parcelle décide de sa couleur. Une note sous
70 déclenche une alerte, comme une feuille malade ou une parcelle dont l'eau baisse plus que les autres entre
les saisons. Chaque alerte renvoie à l'image qui la justifie (aperçu dans apercus/, nom du fichier d'origine).
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
        ligne["score_sante"] = min(notes.values()) if notes else None      # la pire note : le plus gros problème
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


VERT, ORANGE, ROUGE, GRIS = "#1a7f37", "#b45309", "#cf222e", "#6e7781"

NOMS_COLONNES = {
    "parcelle": "Parcelle", "priorite": "Priorité", "surface_ha": "Surface (ha)", "pieds": "Pieds",
    "pieds_par_ha": "Pieds par ha", "vari": "VARI", "couverture_vegetale_%": "Couverture végétale (%)",
    "vigueur_drone_faible_%": "Faible face à la plantation, drone (%)",
    "vigueur_drone_taches_%": "Taches faibles dans la parcelle, drone (%)",
    "vigueur_faible_%": "Faible face à la plantation (%)", "vigueur_taches_%": "Taches faibles dans la parcelle (%)",
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


def _nombre(v, pourcent=False):
    """Nombre à la française : entier sans virgule, sinon 3 décimales sous 1, 1 au-dessus."""
    if _vide(v):
        return ""
    if isinstance(v, (int, np.integer)) or (isinstance(v, (float, np.floating)) and float(v).is_integer()):
        return f"{int(v):,}".replace(",", " ")
    if isinstance(v, (float, np.floating)):
        return (f"{v:.3f}" if abs(v) < 1 and not pourcent else f"{v:.1f}").replace(".", ",")
    return str(v)


def _pastille(score):
    texte = "?" if _vide(score) else int(score)
    return f'<span class="pastille" style="background:{_couleur(score)}">{texte}</span>'


COLONNES_RESUME = ["parcelle", "surface_ha", "pieds", "pieds_par_ha", "note_vigueur", "note_eau", "note_maladies",
                   "score_sante"]
ICONES = {
    "parcelles": '<path d="M3 3h8v8H3zM13 3h8v8h-8zM3 13h8v8H3zM13 13h8v8h-8z"/>',
    "alertes": '<path d="M12 3a6 6 0 0 0-6 6v4l-2 3h16l-2-3V9a6 6 0 0 0-6-6zM10 19a2 2 0 0 0 4 0"/>',
    "score": '<path d="M12 21s-7-4.5-9-9a5 5 0 0 1 9-3 5 5 0 0 1 9 3c-2 4.5-9 9-9 9z"/>',
    "rouge": '<path d="M12 3 2 20h20L12 3zM12 10v4M12 17v.5"/>',
    "pieds": '<circle cx="6" cy="7" r="2"/><circle cx="12" cy="7" r="2"/><circle cx="18" cy="7" r="2"/>'
             '<circle cx="6" cy="15" r="2"/><circle cx="12" cy="15" r="2"/><circle cx="18" cy="15" r="2"/>',
    "feuilles": '<path d="M5 19c0-8 6-14 15-14 0 9-6 15-14 15M5 19l7-7"/>',
}


def _icone(nom):
    return (f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
            f'stroke-linejoin="round" aria-hidden="true" focusable="false">{ICONES[nom]}</svg>')


def ecrire_html(chemin, tableau, alertes, fond, parcelles_px, pieds_px, sources, avertissements, calques=None):
    """Page autonome (sans connexion internet) : chiffres clés, carte à calques (image, vigueur, eau), fiche de
    chaque parcelle (notes, alertes, photos), tableau. fond : (nom du PNG, largeur, hauteur) ; parcelles_px :
    {id: [anneaux de points en pixels du fond]} ; pieds_px : liste de (x, y) ; calques : [(id, nom, PNG)]."""
    l, h = (fond[1], fond[2]) if fond else (800, 600)
    e = max(l, h) / 900
    calques = calques or ([("image", "Image", fond[0])] if fond else [])

    # données de chaque parcelle, pour la fiche (lue par le script de la page)
    par_parcelle = {}
    for a in alertes:
        par_parcelle.setdefault(a["parcelle"], []).append(a)
    fiches = []
    for _, r in tableau.iterrows():
        notes = {k: (None if _vide(r.get(f"note_{k}")) else int(r[f"note_{k}"])) for k in ("vigueur", "eau", "maladies")
                 if f"note_{k}" in tableau}
        liste = []
        for a in par_parcelle.get(r["parcelle"], []):
            liste.append({"theme": THEMES.get(a.get("theme"), str(a.get("theme", "")).capitalize()),
                          "message": a.get("message", ""), "apercu": a.get("apercu", ""),
                          "photo": bool(a.get("theme") == "maladies" and a.get("apercu")),
                          "source": str(a.get("source", ""))})
        infos = []
        if "surface_ha" in tableau:
            infos.append(["Surface", f"{_nombre(r['surface_ha'])} ha"])
        if "pieds" in tableau and not _vide(r.get("pieds")):
            infos.append(["Pieds détectés", f"{_nombre(r['pieds'])} ({_nombre(r['pieds_par_ha'])} par ha)"])
        if "photos_feuilles" in tableau and not _vide(r.get("photos_feuilles")):
            infos.append(["Photos de feuilles", f"{_nombre(r['photos_feuilles'])}, dont {_nombre(r['feuilles_malades'])} "
                                                f"malade(s)"])
        score = r.get("score_sante")
        fiches.append({"id": str(r["parcelle"]), "priorite": int(r["priorite"]),
                       "score": None if _vide(score) else int(score), "couleur": _couleur(score),
                       "notes": notes, "infos": infos, "alertes": liste})
    donnees = json.dumps(fiches, ensure_ascii=False).replace("</", "<\\/")

    # carte
    images = "".join(f'<image class="calque" data-calque="{c[0]}" href="{c[2]}" x="0" y="0" width="{l}" height="{h}"'
                     f'{"" if k == 0 else " style=&quot;display:none&quot;"}/>'.replace("&quot;", '"')
                     for k, c in enumerate(calques))
    formes = []
    for _, r in tableau.iterrows():
        anneaux = parcelles_px.get(r["parcelle"], [])
        d = " ".join("M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in a) + " Z" for a in anneaux)
        c = _couleur(r.get("score_sante"))
        s = r.get("score_sante")
        nom = f"Parcelle {r['parcelle']}, score {'non calculé' if _vide(s) else int(s)}"
        formes.append(f'<path class="parcelle" data-id="{r["parcelle"]}" d="{d}" style="--c:{c}" '
                      f'stroke-width="{2.4 * e:.1f}" tabindex="0" role="button" aria-label="{html.escape(nom)}">'
                      f'<title>{html.escape(nom)}</title></path>')
        if anneaux:
            xs, ys = zip(*anneaux[0])
            formes.append(f'<text x="{np.mean(xs):.0f}" y="{np.mean(ys):.0f}" class="etiquette" '
                          f'style="font-size:{15 * e:.0f}px;stroke-width:{4 * e:.1f}px">{r["parcelle"]}</text>')
    points = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{2.6 * e:.1f}"/>' for x, y in pieds_px)
    carte = (f'<svg viewBox="0 0 {l} {h}" class="carte" xmlns="http://www.w3.org/2000/svg" role="group" '
             f'aria-label="Carte des parcelles">{images}'
             f'<g class="parcelles">{"".join(formes)}</g><g class="pieds">{points}</g></svg>')
    boutons = "".join(f'<button type="button" data-calque="{c[0]}" aria-pressed="{"true" if k == 0 else "false"}"'
                      f'{" class=actif" if k == 0 else ""}>{html.escape(c[1])}</button>'
                      for k, c in enumerate(calques)) if len(calques) > 1 else ""
    legendes_calques = (
        '<div class="echelle" data-pour="vigueur" hidden><span>peu de feuillage</span><i class="degrade vig"></i>'
        '<span>feuillage dense</span></div>'
        '<div class="echelle" data-pour="eau" hidden><span>plus sec que prévu</span><i class="degrade eau"></i>'
        '<span>plus humide que prévu</span></div>')

    # chiffres clés
    cles = [("parcelles", str(len(tableau)), "parcelles"), ("alertes", str(len(alertes)), "alertes")]
    if "score_sante" in tableau and tableau.score_sante.notna().any():
        n_rouge = int((tableau.score_sante < SEUIL_ALERTE).sum())
        cles.append(("rouge", str(n_rouge), "parcelles à visiter (sous 70)"))
        cles.append(("score", str(int(round(tableau.score_sante.median()))), "score médian sur 100"))
    if "pieds" in tableau and tableau.pieds.notna().any():
        cles.append(("pieds", _nombre(int(tableau.pieds.sum())), "pieds détectés"))
    if "photos_feuilles" in tableau and tableau.photos_feuilles.notna().any():
        cles.append(("feuilles", f"{int(tableau.feuilles_malades.sum())} sur {int(tableau.photos_feuilles.sum())}",
                     "photos de feuilles malades"))
    bloc_cles = "".join(f'<div class="cle"><div class="ico">{_icone(i)}</div><div><b>{v}</b>'
                        f'<span>{html.escape(t)}</span></div></div>' for i, v, t in cles)

    # tableau : le résumé, puis toutes les mesures dans un volet repliable
    def table(colonnes):
        entete = "".join(f"<th>{html.escape(_nom_colonne(c))}</th>" for c in colonnes)
        corps = ""
        for _, r in tableau.iterrows():
            cel = []
            for c in colonnes:
                if c == "score_sante":
                    cel.append(f"<td>{_pastille(r[c])}</td>")
                elif c.startswith("note_"):
                    v = r[c]
                    cel.append("<td></td>" if _vide(v) else
                               f'<td><span class="mini"><i style="width:{int(v)}%;background:{_couleur(v)}"></i></span>'
                               f'{int(v)}</td>')
                else:
                    cel.append(f'<td>{html.escape(_nombre(r[c], "%" in c))}</td>')
            corps += (f'<tr data-id="{r["parcelle"]}" tabindex="0" aria-label="Voir la fiche de la parcelle '
                      f'{r["parcelle"]}">{"".join(cel)}</tr>')
        return f"<table><thead><tr>{entete}</tr></thead><tbody>{corps}</tbody></table>"
    resume = [c for c in COLONNES_RESUME if c in tableau]
    toutes = [c for c in tableau.columns if c not in ("alertes", "priorite")]

    remarques = "".join(f"<li>{html.escape(a)}</li>" for a in avertissements)
    bloc_remarques = f'<details class="remarques"><summary>Remarques ({len(avertissements)})</summary><ul>{remarques}</ul></details>' if remarques else ""

    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#f3f4f1">
<title>Field Scan : bulletin</title>
<style>
:root {{ --fond:#f3f4f1; --carte:#fff; --texte:#1c2321; --doux:#5f6b66; --bord:#e3e6e1; --marque:#1f6f43; --vert:{VERT}; --orange:{ORANGE}; --rouge:{ROUGE}; --gris:{GRIS}; }}
* {{ box-sizing:border-box; }}
html {{ color-scheme:light; }}
button, a, [tabindex] {{ touch-action:manipulation; -webkit-tap-highlight-color:transparent; }}
:focus-visible {{ outline:2px solid var(--marque); outline-offset:2px; border-radius:6px; }}
.parcelle:focus-visible {{ outline:none; stroke:#0b57d0; fill-opacity:.4; }}
tbody tr:focus-visible {{ outline:2px solid var(--marque); outline-offset:-2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}
.cle b, .note b, .pastille, .grand-score, td {{ font-variant-numeric:tabular-nums; }}
h1, h2, h3 {{ text-wrap:balance; }}
body {{ margin:0; font-family:"Segoe UI", system-ui, -apple-system, sans-serif; color:var(--texte); background:var(--fond); font-size:14px; line-height:1.45; }}
.page {{ max-width:1320px; margin:0 auto; padding:22px 20px 48px; }}
.entete {{ display:flex; align-items:center; gap:14px; margin-bottom:20px; }}
.logo {{ width:42px; height:42px; border-radius:12px; background:var(--marque); color:#fff; display:grid; place-items:center; }}
.logo svg {{ width:24px; height:24px; }}
.entete h1 {{ margin:0; font-size:22px; letter-spacing:-.3px; }} .entete p {{ margin:2px 0 0; color:var(--doux); font-size:13px; }}
.entete .info {{ margin-left:auto; text-align:right; color:var(--doux); font-size:12.5px; }}
.cles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:18px; }}
.cle {{ display:flex; gap:12px; align-items:center; background:var(--carte); border:1px solid var(--bord); border-radius:14px; padding:14px 16px; }}
.cle .ico {{ width:38px; height:38px; border-radius:10px; background:#e8f2ec; color:var(--marque); display:grid; place-items:center; flex:none; }}
.cle .ico svg {{ width:20px; height:20px; }}
.cle b {{ display:block; font-size:22px; line-height:1.1; }} .cle span {{ color:var(--doux); font-size:12.5px; }}
.corps {{ display:grid; grid-template-columns:minmax(0,1.3fr) minmax(340px,1fr); gap:18px; align-items:start; }}
@media (max-width:980px) {{ .corps {{ grid-template-columns:1fr; }} }}
.panneau {{ background:var(--carte); border:1px solid var(--bord); border-radius:16px; padding:16px; box-shadow:0 1px 2px rgba(0,0,0,.03); }}
.titre-panneau {{ display:flex; align-items:center; justify-content:space-between; gap:10px; margin-bottom:12px; }}
.titre-panneau h2 {{ margin:0; font-size:15px; }}
.calques {{ display:inline-flex; background:#eef0ec; border-radius:10px; padding:3px; gap:2px; }}
.calques button {{ border:0; background:transparent; padding:6px 12px; border-radius:8px; font:inherit; font-size:12.5px; color:var(--doux); cursor:pointer; }}
.calques button:hover {{ color:var(--texte); }}
.calques button.actif {{ background:#fff; color:var(--texte); font-weight:600; box-shadow:0 1px 2px rgba(0,0,0,.08); }}
.carte {{ display:block; width:100%; height:auto; border-radius:12px; background:#e7eae4; }}
.parcelle {{ fill:var(--c); fill-opacity:.16; stroke:var(--c); cursor:pointer; transition:fill-opacity .15s; }}
.parcelle:hover {{ fill-opacity:.32; }} .parcelle.choisie {{ fill-opacity:.38; stroke:#fff; }}
.etiquette {{ font-weight:700; fill:#fff; paint-order:stroke; stroke:rgba(0,0,0,.7); text-anchor:middle; dominant-baseline:middle; pointer-events:none; }}
.pieds circle {{ fill:#00d2ff; stroke:#00323d; stroke-width:.8; }}
.bas-carte {{ display:flex; flex-wrap:wrap; justify-content:space-between; gap:10px; margin-top:12px; color:var(--doux); font-size:12.5px; }}
.legende {{ display:flex; flex-wrap:wrap; gap:14px; }}
.legende i {{ display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:6px; vertical-align:-1px; }}
.legende i.rond {{ border-radius:50%; background:#00d2ff; border:1px solid #00323d; }}
.echelle {{ display:flex; align-items:center; gap:8px; }} .echelle[hidden] {{ display:none; }}
.degrade {{ display:inline-block; width:140px; height:9px; border-radius:5px; }}
.degrade.vig {{ background:linear-gradient(90deg,#8c510a,#d8b365,#f6e8c3,#c7eaaa,#5aae61,#006837); }}
.degrade.eau {{ background:linear-gradient(90deg,#b2182b,#ef8a62,#f7f7f7,#67a9cf,#2166ac); }}
#fiche {{ position:sticky; top:16px; max-height:calc(100vh - 32px); overflow:auto; }}
.ligne {{ display:flex; align-items:center; gap:10px; padding:9px 10px; border-radius:10px; cursor:pointer; width:100%;
  border:0; background:none; font:inherit; color:inherit; text-align:left; }}
.ligne:hover {{ background:#f4f6f3; }}
.ligne .nom {{ font-weight:600; min-width:84px; }}
.ligne .resume {{ color:var(--doux); font-size:12.5px; flex:1; min-width:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.pastille {{ display:inline-flex; align-items:center; justify-content:center; min-width:36px; height:24px; padding:0 8px; border-radius:12px; color:#fff; font-weight:700; font-size:12px; }}
.retour {{ border:0; background:none; color:var(--marque); font:inherit; font-size:13px; cursor:pointer; padding:2px 0; margin-bottom:10px; }}
.retour:hover {{ text-decoration:underline; }}
.tete-fiche {{ display:flex; align-items:center; gap:14px; margin-bottom:14px; }}
.grand-score {{ width:64px; height:64px; border-radius:18px; color:#fff; display:grid; place-items:center; font-size:24px; font-weight:800; }}
.tete-fiche h3 {{ margin:0; font-size:19px; }} .tete-fiche p {{ margin:2px 0 0; color:var(--doux); font-size:12.5px; }}
.note {{ display:grid; grid-template-columns:80px 1fr 34px; align-items:center; gap:10px; margin:7px 0; font-size:13px; }}
.barre {{ height:8px; background:#edf0eb; border-radius:4px; overflow:hidden; }} .barre i {{ display:block; height:100%; border-radius:4px; }}
.note b {{ text-align:right; }}
.infos {{ display:grid; grid-template-columns:auto 1fr; gap:4px 14px; margin:14px 0; font-size:13px; }} .infos span {{ color:var(--doux); }}
.alerte {{ display:flex; gap:10px; padding:10px 0; border-top:1px solid var(--bord); font-size:13px; }}
.theme {{ flex:none; font-size:10.5px; font-weight:700; text-transform:uppercase; letter-spacing:.5px; color:var(--doux); padding:3px 7px; background:#eef0ec; border-radius:6px; height:fit-content; }}
.alerte a {{ color:var(--marque); text-decoration:none; font-size:12px; margin-left:6px; }} .alerte a:hover {{ text-decoration:underline; }}
.alerte > div {{ min-width:0; overflow-wrap:anywhere; }}
.galerie {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(96px,1fr)); gap:8px; margin-top:12px; }}
.vignette {{ padding:0; border:0; background:none; border-radius:10px; cursor:zoom-in; }}
.vignette img {{ display:block; width:100%; height:auto; aspect-ratio:1; object-fit:cover; border-radius:10px; border:1px solid var(--bord); }}
.vignette:hover img {{ border-color:var(--marque); }}
.vide {{ color:var(--doux); }}
.mesures {{ margin-top:18px; }}
.mesures .defile {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
th, td {{ padding:9px 10px; text-align:right; white-space:nowrap; border-bottom:1px solid var(--bord); }}
th:first-child, td:first-child {{ text-align:left; font-weight:600; }}
th {{ color:var(--doux); font-weight:600; font-size:12px; background:#f7f8f6; }}
tbody tr {{ cursor:pointer; }} tbody tr:hover td {{ background:#f6f8f5; }} tr.choisie td {{ background:#e8f2ec; }}
.mini {{ display:inline-block; width:60px; height:6px; background:#edf0eb; border-radius:3px; margin-right:8px; vertical-align:middle; overflow:hidden; }}
.mini i {{ display:block; height:100%; }}
details summary {{ cursor:pointer; color:var(--marque); font-weight:600; margin:14px 0 6px; }}
.remarques {{ margin-top:18px; color:var(--doux); font-size:13px; }} .remarques ul {{ margin:6px 0 0; padding-left:18px; }}
#visionneuse {{ position:fixed; inset:0; background:rgba(10,14,12,.86); display:none; place-items:center; z-index:9; cursor:zoom-out; overscroll-behavior:contain; }}
#visionneuse .fermer {{ position:fixed; top:16px; right:16px; width:40px; height:40px; border-radius:20px; border:0; background:#fff; color:var(--texte); font-size:22px; line-height:1; cursor:pointer; }}
#visionneuse .fermer:focus-visible {{ outline:2px solid #fff; outline-offset:3px; }}
#visionneuse:not([hidden]) {{ display:grid; }} #visionneuse img {{ max-width:92vw; max-height:84vh; border-radius:12px; }}
#visionneuse p {{ color:#e8ece9; text-align:center; margin:12px 0 0; font-size:13px; }}
</style></head><body><div class="page">
<div class="entete"><div class="logo">{_icone("feuilles")}</div>
<div><h1>Field Scan</h1><p>Bulletin de santé par parcelle</p></div>
<div class="info">Produit le <time datetime="{datetime.now():%Y-%m-%dT%H:%M}">{datetime.now():%d/%m/%Y à %H:%M}</time><br>{html.escape(", ".join(sources)) or "aucune source"}</div></div>
<div class="cles">{bloc_cles}</div>
<div class="corps">
<section class="panneau"><div class="titre-panneau"><h2>Carte des parcelles</h2><div class="calques">{boutons}</div></div>{carte}
<div class="bas-carte"><div class="legende"><span><i style="background:{VERT}"></i>85 et plus</span>
<span><i style="background:{ORANGE}"></i>70 à 84</span><span><i style="background:{ROUGE}"></i>sous 70</span><span><i style="background:{GRIS}"></i>non calculé (nuages ou sans image)</span>
{'<span><i class="rond"></i>pied détecté</span>' if pieds_px else ''}</div>{legendes_calques}</div></section>
<section class="panneau" id="fiche" aria-live="polite"></section>
</div>
<section class="panneau mesures"><div class="titre-panneau"><h2>Toutes les parcelles</h2></div>
<div class="defile">{table(resume)}</div>
<details><summary>Toutes les mesures</summary><div class="defile">{table(toutes)}</div></details></section>
{bloc_remarques}
</div>
<div id="visionneuse" role="dialog" aria-modal="true" aria-label="Image agrandie" hidden>
<button type="button" class="fermer" aria-label="Fermer l’image">×</button><div><img alt=""><p></p></div></div>
<script>
const FICHES = {donnees};
const parId = Object.fromEntries(FICHES.map(f => [f.id, f]));
const fiche = document.getElementById('fiche');
const esc = s => String(s).replace(/[&<>"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}})[c]);
const pastille = f => `<span class="pastille" style="background:${{f.couleur}}">${{f.score ?? '?'}}</span>`;
const coul = v => v == null ? '{GRIS}' : v >= 85 ? '{VERT}' : v >= {SEUIL_ALERTE} ? '{ORANGE}' : '{ROUGE}';
function liste() {{
  const avec = FICHES.filter(f => f.alertes.length);
  fiche.innerHTML = `<div class="titre-panneau"><h2>À voir en priorité</h2><span class="vide">${{avec.length}} parcelle(s) avec alerte</span></div>` +
    (avec.length ? avec.map(f => `<button type="button" class="ligne" data-id="${{f.id}}">${{pastille(f)}}<span class="nom">Parcelle ${{esc(f.id)}}</span>` +
      `<span class="resume">${{esc([...new Set(f.alertes.map(a => a.theme))].join(' · '))}}</span></button>`).join('')
      : '<p class="vide">Aucune alerte : toutes les parcelles vont bien.</p>') +
    '<p class="vide" style="font-size:12px;margin-top:12px">Cliquer sur une parcelle, sur la carte ou dans la liste, pour voir sa fiche.</p>';
  fiche.querySelectorAll('.ligne').forEach(el => el.onclick = () => choisir(el.dataset.id));
}}
function montrer(f) {{
  const notes = Object.entries(f.notes).filter(([k, v]) => v != null).map(([k, v]) =>
    `<div class="note"><span>${{k[0].toUpperCase() + k.slice(1)}}</span><div class="barre"><i style="width:${{v}}%;background:${{coul(v)}}"></i></div><b>${{v}}</b></div>`).join('');
  const infos = f.infos.map(([k, v]) => `<span>${{esc(k)}}</span><b>${{esc(v)}}</b>`).join('');
  const alertes = f.alertes.map(a => `<div class="alerte"><span class="theme">${{esc(a.theme)}}</span><div>${{esc(a.message)}}` +
    (a.apercu && !a.photo ? `<a href="${{a.apercu}}" data-zoom="${{a.apercu}}" data-legende="${{esc(a.message)}}">voir l’image</a>` : '') + `</div></div>`).join('');
  const photos = f.alertes.filter(a => a.photo).map(a => `<button type="button" class="vignette" data-zoom="${{a.apercu}}" data-legende="${{esc(a.message + ' (' + a.source + ')')}}" aria-label="Agrandir la photo : ${{esc(a.message)}}">` +
    `<img src="${{a.apercu}}" width="96" height="96" loading="lazy" alt="${{esc(a.message)}}"></button>`).join('');
  fiche.innerHTML = `<button type="button" class="retour"><span aria-hidden="true">← </span>toutes les parcelles</button>` +
    `<div class="tete-fiche"><div class="grand-score" style="background:${{f.couleur}}">${{f.score ?? '?'}}</div>` +
    `<div><h3>Parcelle ${{esc(f.id)}}</h3><p>Priorité ${{f.priorite}} sur ${{FICHES.length}} · score de santé = la pire note</p></div></div>` +
    notes + `<div class="infos">${{infos}}</div>` +
    (alertes || '<p class="vide">Aucune alerte sur cette parcelle.</p>') + (photos ? `<div class="galerie">${{photos}}</div>` : '');
  fiche.querySelector('.retour').onclick = () => choisir(null);
}}
let choisie = null;
function choisir(id) {{
  choisie = (id === null || choisie === id) ? null : id;
  document.querySelectorAll('.parcelle').forEach(p => p.classList.toggle('choisie', p.dataset.id === choisie));
  document.querySelectorAll('tbody tr').forEach(t => t.classList.toggle('choisie', t.dataset.id === choisie));
  choisie ? montrer(parId[choisie]) : liste();
}}
document.querySelectorAll('.parcelle, tbody tr').forEach(el => {{
  el.addEventListener('click', () => choisir(el.dataset.id));
  el.addEventListener('keydown', ev => {{ if (ev.key === 'Enter' || ev.key === ' ') {{ ev.preventDefault(); choisir(el.dataset.id); }} }});
}});
document.querySelectorAll('.calques button').forEach(b => b.onclick = () => {{
  document.querySelectorAll('.calques button').forEach(x => {{ x.classList.toggle('actif', x === b); x.setAttribute('aria-pressed', x === b); }});
  document.querySelectorAll('.calque').forEach(c => c.style.display = (c.dataset.calque === b.dataset.calque || c.dataset.calque === 'image') ? '' : 'none');
  document.querySelectorAll('.echelle').forEach(x => x.hidden = x.dataset.pour !== b.dataset.calque);
}});
const vis = document.getElementById('visionneuse');
let ouvreur = null;
function fermer() {{ vis.hidden = true; ouvreur?.focus(); }}
document.addEventListener('click', ev => {{
  const z = ev.target.closest('[data-zoom]');
  if (!z) return;
  ev.preventDefault(); ouvreur = z;
  const img = vis.querySelector('img');
  img.src = z.dataset.zoom; img.alt = z.dataset.legende || 'Image source de l’alerte';
  vis.querySelector('p').textContent = z.dataset.legende || '';
  vis.hidden = false; vis.querySelector('.fermer').focus();
}});
vis.addEventListener('click', ev => {{ if (ev.target.tagName !== 'IMG') fermer(); }});
document.addEventListener('keydown', ev => {{ if (ev.key === 'Escape' && !vis.hidden) fermer(); }});
const quand = document.querySelector('time');
try {{ quand.textContent = new Intl.DateTimeFormat('fr-FR', {{ dateStyle: 'long', timeStyle: 'short' }}).format(new Date(quand.dateTime)); }} catch (e) {{}}
liste();
// lien direct : bulletin.html#parcelle=11&calque=vigueur
const lien = new URLSearchParams(location.hash.slice(1));
if (lien.get('calque')) document.querySelector(`.calques button[data-calque="${{lien.get('calque')}}"]`)?.click();
if (lien.get('parcelle') && parId[lien.get('parcelle')]) choisir(lien.get('parcelle'));
</script></body></html>"""
    chemin.write_text(page, encoding="utf-8")


def ecrire_tableau(chemin, tableau):
    tableau.to_csv(chemin, index=False, encoding="utf-8")


def ecrire_json(chemin, objet):
    chemin.write_text(json.dumps(objet, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
