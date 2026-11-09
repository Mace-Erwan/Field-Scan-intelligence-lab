"""
Field Scan : analyse d'une bananeraie vue du ciel, parcelle par parcelle.

Utilisation :
    python run.py --images DOSSIER --parcelles parcelles.geojson --out resultats/

DOSSIER peut contenir (dans n'importe quels sous-dossiers) :
  - des images drone géoréférencées (GeoTIFF RGB)          -> comptage des pieds, vigueur (ExG, VARI) ;
  - des bandes Sentinel-2 du Copernicus Browser, 1 ou 2 dates -> vigueur (NDVI, NDRE), stress hydrique (NDMI) ;
  - un dossier « feuilles » avec des photos de feuilles      -> maladies.
Chaque partie ne tourne que si ses images sont présentes. Tout tourne sans GPU.

Sorties dans --out :
  bulletin.html      carte, alertes avec leur image source, tableau (s'ouvre sans internet)
  bulletin.csv       une ligne par parcelle : mesures, notes, score de santé, alertes
  pieds.geojson      un point par pied détecté
  alertes.geojson    zones et parcelles signalées
  vigueur.tif, stress.tif, ndvi_<date>.tif, ndre_<date>.tif, ndmi_<date>.tif, vari_<image>.tif, exg_<image>.tif
  feuilles.csv       réponse du modèle pour chaque photo de feuille
  apercus/           images sources des alertes
"""

import argparse
import shutil
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask, shapes
from shapely.geometry import Point, shape
from shapely.ops import unary_union

import entrees
from bulletin import apercus, bulletin
from indices import sentinel2, stress, vigueur

MODELES = Path(__file__).parent / "modeles"


def ecrire_tif(chemin, carte, profil):
    p = dict(profil)
    p.update(count=1, dtype="float32", nodata=np.nan)
    with rasterio.open(chemin, "w", **p) as dst:
        dst.write(carte.astype("float32"), 1)


def zones_en_polygones(masque, transform, crs, surface_min_m2=0.0):
    """Masque vrai/faux -> polygones (dans le CRS de la grille)."""
    if not masque.any():
        return None
    polys = [shape(g) for g, v in shapes(masque.astype("uint8"), mask=masque, transform=transform) if v == 1]
    polys = [p for p in polys if p.area >= surface_min_m2]
    return unary_union(polys) if polys else None


def note(part_signalee):
    return int(round(100 * (1 - part_signalee)))


def partie_comptage(images_drone, parcelles, mesures, sortie):
    from comptage import detecteur, pieds
    poids = sorted((MODELES / "comptage").glob("*.pt"))
    reseaux = detecteur.charger(poids)
    print(f"Comptage : {len(reseaux)} modèles, {len(images_drone)} image(s) drone")
    tous, couvert = [], {}
    for im in images_drone:
        gdf, surfaces = pieds.compter(im["chemin"], parcelles, reseaux)
        tous.append(gdf)
        for i, ha in surfaces.items():                 # surface réellement couverte par des pixels d'image
            couvert[i] = couvert.get(i, 0) + ha
    pieds_gdf = pd.concat(tous, ignore_index=True) if tous else gpd.GeoDataFrame(columns=["id_parcelle", "score", "image", "geometry"])
    pieds_gdf = gpd.GeoDataFrame(pieds_gdf, geometry="geometry", crs="EPSG:4326")
    pieds_gdf.to_file(sortie / "pieds.geojson", driver="GeoJSON")
    for _, p in parcelles.iterrows():
        ha = couvert.get(p["id"], 0)
        if ha <= 0:
            continue
        n = int((pieds_gdf.id_parcelle == p["id"]).sum())
        mesures[p["id"]]["pieds"] = n
        mesures[p["id"]]["pieds_par_ha"] = round(n / ha)
        if ha < 0.95 * p["surface_ha"]:
            mesures[p["id"]]["couverture_drone_%"] = round(100 * ha / p["surface_ha"])
    return pieds_gdf


def partie_vigueur_drone(images_drone, parcelles, mesures, sortie, alertes_zones, utiliser_pour_note):
    print("Vigueur drone (ExG, VARI)")
    for im in images_drone:
        res = vigueur.cartes_drone(im["chemin"], parcelles)
        if res is None:
            continue
        indices, profil, valide = res
        nom = im["chemin"].stem
        for k, carte in indices.items():
            ecrire_tif(sortie / f"{k}_{nom}.tif", carte, profil)
        p_utm = parcelles.to_crs(profil["crs"])
        # végétation / sol à 50 cm (ExG, seuil d'Otsu), puis part de végétation par case de 10 m
        dans_fin = np.zeros(valide.shape, bool)
        for _, p in p_utm.iterrows():
            dans_fin |= geometry_mask([p.geometry], out_shape=valide.shape, transform=profil["transform"], invert=True)
        veg, seuil_exg = vigueur.couverture_vegetale(indices["exg"], dans_fin & valide)
        facteur = int(round(vigueur.TAILLE_ZONE_M / vigueur.TAILLE_CASE_DRONE_M))
        couv, t_case = vigueur.moyenne_par_case(veg, profil["transform"], facteur)
        vari_case, _ = vigueur.moyenne_par_case(indices["vari"], profil["transform"], facteur)
        dans = np.zeros(couv.shape, bool)
        masques = {}
        for _, p in p_utm.iterrows():
            m = geometry_mask([p.geometry], out_shape=couv.shape, transform=t_case, invert=True) & np.isfinite(couv)
            if m.any():
                masques[p["id"]] = m
                dans |= m
        faible, ref = vigueur.zones_faibles(couv, dans)
        print(f"  {nom} : végétation si ExG > {seuil_exg:.3f} ; couverture de référence {100 * ref:.0f} % "
              f"(cases de 10 m), zones faibles sous {100 * vigueur.FRACTION_NORMALE * ref:.0f} %")
        for i, m in masques.items():
            part = float(faible[m].mean())
            mesures[i]["vari"] = round(float(np.nanmedian(vari_case[m])), 3)
            mesures[i]["couverture_vegetale_%"] = round(100 * float(np.nanmean(couv[m])), 1)
            mesures[i]["vigueur_drone_faible_%"] = round(100 * part, 1)
            if utiliser_pour_note:
                mesures[i]["note_vigueur"] = note(part)
                zones = zones_en_polygones(faible & m, t_case, profil["crs"])
                geom = p_utm[p_utm.id == i].geometry.iloc[0]
                png = f"apercus/vigueur_drone_parcelle_{i}.png"
                apercus.apercu(sortie / png, im["chemin"], profil["crs"], geom, zones, (255, 140, 0),
                               f"Parcelle {i} - vigueur faible (ExG) en orange - {im['chemin'].name}")
                mesures[i]["_alerte_vigueur"] = {
                    "message": f"vigueur faible (peu de végétation, ExG) sur {100 * part:.0f} % de la parcelle (image drone)",
                    "source": im["chemin"].name, "apercu": png}
                if zones is not None:
                    alertes_zones.append({"parcelle": i, "theme": "vigueur", "geometry": zones, "crs": profil["crs"]})


def partie_sentinel2(dates, parcelles, mesures, sortie, alertes_zones):
    print(f"Sentinel-2 : {len(dates)} date(s) : {', '.join(dates)}")
    par_date, faibles_par_date, ndmi_par_date = {}, {}, {}
    for d, bandes in dates.items():
        indices, profil, exclus = sentinel2.lire_date(bandes)
        for k, carte in indices.items():
            ecrire_tif(sortie / f"{k}_{d}.tif", carte, profil)
        p = parcelles.to_crs(profil["crs"])
        masques = {}
        dans = np.zeros(exclus.shape, bool)
        for _, r in p.iterrows():
            m = geometry_mask([r.geometry], out_shape=exclus.shape, transform=profil["transform"], invert=True) & ~exclus
            if m.any():
                masques[r["id"]] = m
                dans |= m
        faible, ref = vigueur.zones_faibles(indices["ndvi"], dans)
        seche, droite = stress.zones_seches(indices["ndvi"], indices["ndmi"], dans & ~faible)
        par_date[d] = dict(indices=indices, profil=profil, masques=masques, faible=faible, seche=seche,
                           ref=ref, droite=droite, bandes=bandes, crs=p.crs, parcelles=p)
        faibles_par_date[d] = {i: float(faible[m].mean()) for i, m in masques.items()}
        ndmi_par_date[d] = {i: float(np.nanmedian(indices["ndmi"][m])) for i, m in masques.items()}
        print(f"  {d} : NDVI de référence {ref:.3f} (zones faibles sous {vigueur.FRACTION_NORMALE * ref:.3f}) ; "
              f"NDMI attendu = {droite[0]:.3f} + {droite[1]:.3f} x NDVI")

    actuelle = max(par_date)
    a = par_date[actuelle]
    ecrire_tif(sortie / "vigueur.tif", a["indices"]["ndvi"], a["profil"])
    residu = a["indices"]["ndmi"] - (a["droite"][0] + a["droite"][1] * a["indices"]["ndvi"])
    ecrire_tif(sortie / "stress.tif", residu, a["profil"])
    for i, m in a["masques"].items():
        mi = mesures[i]
        for k in ("ndvi", "ndre", "ndmi"):
            if k in a["indices"]:
                mi[f"{k}_{actuelle}"] = round(float(np.nanmedian(a["indices"][k][m])), 3)
        part_f, part_s = float(a["faible"][m].mean()), float(a["seche"][m].mean())
        mi["vigueur_faible_%"], mi["eau_faible_%"] = round(100 * part_f, 1), round(100 * part_s, 1)
        mi["note_vigueur"], mi["note_eau"] = note(part_f), note(part_s)
        geom = a["parcelles"][a["parcelles"].id == i].geometry.iloc[0]
        for theme, masque, couleur, texte in (("vigueur", a["faible"], (255, 140, 0), "vigueur faible (NDVI)"),
                                              ("eau", a["seche"], (0, 160, 255), "manque d'eau probable (NDMI)")):
            part = float(masque[m].mean())
            if note(part) >= bulletin.SEUIL_ALERTE:
                continue
            zones = zones_en_polygones(masque & m, a["profil"]["transform"], a["crs"])
            png = f"apercus/{theme}_parcelle_{i}_{actuelle}.png"
            apercus.apercu(sortie / png, a["bandes"], a["crs"], geom, zones, couleur,
                           f"Parcelle {i} - {texte} - Sentinel-2 du {actuelle}")
            mi[f"_alerte_{theme}"] = {"message": f"{texte} sur {100 * part:.0f} % de la parcelle (Sentinel-2 du {actuelle})",
                                      "source": f"Sentinel-2 {actuelle} ({a['bandes']['B04'].name})", "apercu": png}
            if zones is not None:
                alertes_zones.append({"parcelle": i, "theme": theme, "geometry": zones, "crs": a["crs"]})

    if len(par_date) >= 2:
        comparables = [i for i in ndmi_par_date[actuelle]
                       if all(faibles_par_date[d].get(i, 1) < 0.3 for d in par_date)]
        humide, seche_d, baisse, signales = stress.comparer_saisons(ndmi_par_date, comparables)
        print(f"  saisons : humide {humide}, sèche {seche_d} ; parcelles signalées : {signales or 'aucune'}")
        for i, b in baisse.items():
            mesures[i]["baisse_ndmi_saison"] = round(b, 3)
        for i in signales:
            s = par_date[seche_d]
            geom = s["parcelles"][s["parcelles"].id == i].geometry.iloc[0]
            png = f"apercus/saison_parcelle_{i}.png"
            apercus.apercu(sortie / png, s["bandes"], s["crs"], geom, None, (0, 160, 255),
                           f"Parcelle {i} - saison sèche ({seche_d})")
            mesures[i].setdefault("_alertes_supplementaires", []).append({
                "theme": "saisons", "message": f"l'eau des feuilles baisse plus que dans les autres parcelles entre "
                f"{humide} et {seche_d} (NDMI -{baisse[i]:.2f})", "source": f"Sentinel-2 {humide} et {seche_d}", "apercu": png})
    return a


# Ni le Pérou ni le Bangladesh ne disent si leur Sigatoka est noire ou jaune : le bulletin dit « cercosporiose ».
# Sur une feuille, la fusariose se voit surtout comme un jaunissement, qui n'est pas spécifique : le vrai signe est
# dans le pseudo-tronc.
LIBELLES = {"black_sigatoka": "cercosporiose (Sigatoka)",
            "fusarium": "jaunissement compatible avec la fusariose (Fusarium), à confirmer par une coupe du pseudo-tronc",
            "malade_non_identifiee": "maladie non identifiée"}


def partie_feuilles(photos, parcelles, mesures, sortie):
    from maladies import classifieur
    reseaux, seuils = classifieur.charger(MODELES / "maladies")
    print(f"Maladies : {len(photos)} photo(s) de feuilles, {len(reseaux)} modèles")
    debut = time.time()
    probas = classifieur.probabilites(reseaux, photos)
    reponses = classifieur.decider(probas, seuils)
    print(f"  {time.time() - debut:.0f} s ({(time.time() - debut) / max(len(photos), 1):.1f} s par photo)")
    lignes = []
    for chemin, p, r in zip(photos, probas, reponses):
        pos = classifieur.position_gps(chemin)
        id_p = None
        if pos:
            pt = Point(pos)
            dedans = parcelles[parcelles.contains(pt)]
            id_p = dedans.id.iloc[0] if len(dedans) else None
        lignes.append({"photo": str(chemin), "reponse": r, "confiance": round(float(p.max()), 3),
                       "p_saine": round(float(p[0]), 3), "p_black_sigatoka": round(float(p[1]), 3),
                       "p_fusarium": round(float(p[2]), 3),
                       "longitude": pos[0] if pos else None, "latitude": pos[1] if pos else None,
                       "parcelle": id_p if id_p is not None else "non localisée"})
    feuilles = pd.DataFrame(lignes)
    feuilles.to_csv(sortie / "feuilles.csv", index=False, encoding="utf-8")
    for i, g in feuilles[feuilles.parcelle != "non localisée"].groupby("parcelle"):
        mi = mesures[i]
        mi["photos_feuilles"] = len(g)
        mi["feuilles_malades"] = int(g.reponse.isin(["black_sigatoka", "fusarium", "malade_non_identifiee"]).sum())
        mi["feuilles_inconnu"] = int((g.reponse == "inconnu").sum())
        mi["note_maladies"] = note(1 - (g.reponse == "saine").mean())
        for k, (_, f) in enumerate(g[g.reponse != "saine"].iterrows()):
            png = f"apercus/feuille_parcelle_{i}_{k + 1}.png"
            apercus.miniature_photo(f.photo, sortie / png, f"{f.reponse} (confiance {f.confiance})")
            texte = "maladie inconnue ou photo douteuse (« je ne sais pas »)" if f.reponse == "inconnu" else f"feuille malade : {LIBELLES[f.reponse]}"
            mi.setdefault("_alertes_supplementaires", []).append(
                {"theme": "maladies", "message": texte, "source": Path(f.photo).name, "apercu": png})
    return feuilles


def main():
    parser = argparse.ArgumentParser(description="Field Scan : bulletin de santé par parcelle")
    parser.add_argument("--images", required=True, type=Path, help="dossier des images")
    parser.add_argument("--parcelles", required=True, type=Path, help="fichier GeoJSON des parcelles")
    parser.add_argument("--out", required=True, type=Path, help="dossier des résultats")
    args = parser.parse_args()
    sortie = args.out
    sortie.mkdir(parents=True, exist_ok=True)
    if (sortie / "apercus").exists():
        shutil.rmtree(sortie / "apercus")
    (sortie / "apercus").mkdir()

    parcelles = entrees.parcelles(args.parcelles).to_crs("EPSG:4326")
    drone = entrees.images_drone(args.images)
    dates = entrees.sentinel2(args.images)
    photos = entrees.photos_feuilles(args.images)
    print(f"{len(parcelles)} parcelles ; trouvé : {len(drone)} image(s) drone, {len(dates)} date(s) Sentinel-2, "
          f"{len(photos)} photo(s) de feuilles")
    if not (drone or dates or photos):
        raise SystemExit("Aucune image utilisable dans le dossier.")

    mesures = {i: {} for i in parcelles.id}
    alertes_zones, sources, avertissements = [], [], []
    pieds_gdf, s2 = None, None
    if drone:
        pieds_gdf = partie_comptage(drone, parcelles, mesures, sortie)
        partie_vigueur_drone(drone, parcelles, mesures, sortie, alertes_zones, utiliser_pour_note=not dates)
        sources += [f"drone : {d['chemin'].name}" for d in drone]
        avertissements.append("Le modèle de comptage attend des bananiers vus du dessus ; sur un type de plantation "
                              "jamais vu à l'apprentissage, il peut perdre jusqu'à 0,24 de F1 (docs/experiences.md).")
    if dates:
        s2 = partie_sentinel2(dates, parcelles, mesures, sortie, alertes_zones)
        sources += [f"Sentinel-2 du {d}" for d in dates]
    if photos:
        feuilles = partie_feuilles(photos, parcelles, mesures, sortie)
        sources.append(f"{len(photos)} photos de feuilles")
        n_hors = int((feuilles.parcelle == "non localisée").sum())
        if n_hors:
            avertissements.append(f"{n_hors} photo(s) de feuilles sans position GPS : résultats dans feuilles.csv, "
                                  "non rattachés à une parcelle.")

    tableau, alertes = bulletin.notes_et_alertes(parcelles, mesures)
    bulletin.ecrire_tableau(sortie / "bulletin.csv", tableau)

    # alertes.geojson : zones signalées, ou la parcelle entière quand l'alerte n'a pas de zone
    elements = []
    for z in alertes_zones:
        g = gpd.GeoSeries([z["geometry"]], crs=z["crs"]).to_crs("EPSG:4326").iloc[0]
        elements.append({"parcelle": z["parcelle"], "theme": z["theme"], "geometry": g})
    for a in alertes:
        if not any(e["parcelle"] == a["parcelle"] and e["theme"] == a["theme"] for e in elements):
            elements.append({"parcelle": a["parcelle"], "theme": a["theme"],
                             "geometry": parcelles[parcelles.id == a["parcelle"]].geometry.iloc[0]})
    infos = {(a["parcelle"], a["theme"]): a for a in alertes}
    for e in elements:
        a = infos.get((e["parcelle"], e["theme"]), {})
        e.update({"message": a.get("message", ""), "source": a.get("source", ""), "apercu": a.get("apercu", "")})
    gpd.GeoDataFrame(elements, columns=["parcelle", "theme", "message", "source", "apercu", "geometry"],
                     geometry="geometry", crs="EPSG:4326").to_file(sortie / "alertes.geojson", driver="GeoJSON")

    # fond de carte du bulletin : Sentinel-2 (date la plus récente) sinon la 1re image drone
    utm = parcelles.estimate_utm_crs()
    p_utm = parcelles.to_crs(utm)
    t, l, h = apercus.grille(p_utm.total_bounds, 900)
    fond = None
    source_fond = s2["bandes"] if s2 else (drone[0]["chemin"] if drone else None)
    if source_fond is not None:
        apercus.rendre(source_fond, utm, t, l, h).save(sortie / "fond.png")
        fond = ("fond.png", l, h)
    parcelles_px = {r["id"]: apercus.anneaux_px(r.geometry, t) for _, r in p_utm.iterrows()}
    pieds_px = []
    if pieds_gdf is not None and len(pieds_gdf):
        inverse = ~t
        pieds_px = [inverse * (pt.x, pt.y) for pt in pieds_gdf.to_crs(utm).geometry]
    bulletin.ecrire_html(sortie / "bulletin.html", tableau, alertes, fond, parcelles_px, pieds_px, sources, avertissements)

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(tableau.drop(columns=["alertes"]).to_string(index=False))
    print(f"{len(alertes)} alerte(s). Résultats dans {sortie} (ouvrir bulletin.html)")


if __name__ == "__main__":
    main()
