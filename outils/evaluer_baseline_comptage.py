"""
Évalue la baseline classique de comptage (ExG + maxima locaux) sur ReforesTree.

Découpage par site (jamais aléatoire) :
    entraînement : réglage des paramètres de la baseline
    validation   : vérification des réglages
    test         : site jamais utilisé pour régler quoi que ce soit

Les réglages (lissage, distance minimale entre deux pieds, seuil de verdure) sont choisis
par une recherche sur grille, uniquement sur les sites d'entraînement.
L'évaluation se fait à l'intérieur des parcelles, seule zone où les arbres sont annotés
de façon complète.

Utilisation :
    python outils/evaluer_baseline_comptage.py --donnees C:/Field_scan/data/traite/reforestree
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely import contains_xy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comptage.baseline_exg import detecter, exg          # noqa: E402
from comptage.evaluation import apparier, scores          # noqa: E402

DECOUPAGE = {
    "Flora Pluas RGB": "entrainement",
    "Leonor Aspiazu RGB": "entrainement",
    "Carlos Vera Guevara RGB": "entrainement",
    "Carlos Vera Arteaga RGB": "entrainement",
    "Manuel Macias RGB": "validation",
    "Nestor Macias RGB": "test",
}
GRILLE = {
    "lissage_m": [0.3, 0.5, 0.8],
    "distance_min_m": [1.5, 2.0, 3.0],
    "seuil": [0.0, 0.05, 0.1],
}
METRIQUE = "EPSG:32717"   # projection UTM en mètres pour mesurer les distances


def charger_site(dossier, site, verite, parcelles):
    with rasterio.open(dossier / "orthos" / f"{site}.tif") as src:
        image = src.read()
        transform = src.transform
    lat = verite[verite.site == site].lat.mean()
    taille_pixel_m = abs(transform.a) * 111320 * np.cos(np.radians(lat))
    valide = image[3] > 0 if image.shape[0] == 4 else np.ones(image.shape[1:], bool)
    zone = parcelles[parcelles.site == site].to_crs(METRIQUE).geometry.union_all()
    vt = verite[verite.site == site]
    vt = gpd.GeoSeries(gpd.points_from_xy(vt.lon, vt.lat), crs=4326).to_crs(METRIQUE)
    vt_xy = np.c_[vt.x, vt.y]
    est_banane = verite[verite.site == site].est_banane.to_numpy()
    dedans = contains_xy(zone, vt_xy[:, 0], vt_xy[:, 1])
    return {"image": image, "transform": transform, "taille_pixel_m": taille_pixel_m, "valide": valide,
            "zone": zone, "carte": exg(image), "bananes": vt_xy[dedans & est_banane], "arbres": vt_xy[dedans]}


def evaluer(donnees, reglages):
    pics = detecter(donnees["image"], donnees["valide"], donnees["taille_pixel_m"], carte=donnees["carte"], **reglages)
    lon, lat = rasterio.transform.xy(donnees["transform"], pics[:, 0], pics[:, 1])
    pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=4326).to_crs(METRIQUE)
    det = np.c_[pts.x, pts.y]
    det = det[contains_xy(donnees["zone"], det[:, 0], det[:, 1])]
    resultat = {"detections": len(det), "bananes_vraies": len(donnees["bananes"])}
    vp, fp, fn = apparier(det, donnees["bananes"])
    resultat |= {"vp": vp, "fp": fp, "fn": fn} | scores(vp, fp, fn)
    vp_a, fp_a, fn_a = apparier(det, donnees["arbres"])
    resultat["f1_tous_arbres"] = scores(vp_a, fp_a, fn_a)["f1"]
    resultat["erreur_comptage"] = (len(det) - len(donnees["bananes"])) / max(1, len(donnees["bananes"]))
    return resultat


def main():
    parser = argparse.ArgumentParser(description="Évaluation de la baseline de comptage")
    parser.add_argument("--donnees", required=True, type=Path, help="dossier data/traite/reforestree")
    args = parser.parse_args()

    verite = pd.read_csv(args.donnees / "verite_terrain.csv")
    parcelles = gpd.read_file(args.donnees / "parcelles_reforestree.geojson")
    sites = {s: charger_site(args.donnees, s, verite, parcelles) for s in DECOUPAGE}

    # Recherche des réglages sur les sites d'entraînement uniquement
    meilleur, meilleure_f1 = None, -1
    for valeurs in itertools.product(*GRILLE.values()):
        reglages = dict(zip(GRILLE, valeurs))
        total = np.zeros(3)
        for s, role in DECOUPAGE.items():
            if role == "entrainement":
                r = evaluer(sites[s], reglages)
                total += [r["vp"], r["fp"], r["fn"]]
        f1 = scores(*total)["f1"]
        if f1 > meilleure_f1:
            meilleur, meilleure_f1 = reglages, f1
    print(f"Réglages retenus (sur l'entraînement) : {meilleur}  F1 entraînement = {meilleure_f1:.3f}\n")

    lignes = [{"site": s, "role": role} | evaluer(sites[s], meilleur) for s, role in DECOUPAGE.items()]
    tableau = pd.DataFrame(lignes)
    pd.set_option("display.width", 200)
    print(tableau.round(3).to_string(index=False))

    sortie = args.donnees / "resultats_baseline"
    sortie.mkdir(exist_ok=True)
    tableau.to_csv(sortie / "resultats.csv", index=False)
    json.dump({"reglages": meilleur, "grille": GRILLE, "decoupage": DECOUPAGE},
              open(sortie / "reglages.json", "w"), indent=2, ensure_ascii=False)
    print(f"\nRésultats écrits dans {sortie}")


if __name__ == "__main__":
    main()
