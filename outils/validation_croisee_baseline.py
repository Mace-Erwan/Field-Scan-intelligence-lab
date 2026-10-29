"""
Validation croisée par ferme de la baseline classique de comptage (ExG + maxima locaux).

Les 5 fermes hors test servent chacune une fois de validation ; à chaque tour, les réglages
sont choisis uniquement sur les 4 autres fermes. La ferme de test (Nestor Macias) n'est pas utilisée.

Pour aller vite, chaque combinaison de réglages est évaluée une seule fois par ferme ;
les résultats sont gardés dans grille_par_ferme.csv (relancer le script reprend là où il s'était arrêté).

Utilisation :
    python outils/validation_croisee_baseline.py --donnees C:/Field_scan/data/traite/reforestree
"""

import argparse
import itertools
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from comptage.evaluation import scores                              # noqa: E402
from evaluer_baseline_comptage import charger_site, evaluer          # noqa: E402

FERMES = ["Flora Pluas RGB", "Leonor Aspiazu RGB", "Carlos Vera Guevara RGB",
          "Carlos Vera Arteaga RGB", "Manuel Macias RGB"]          # Nestor Macias = test, exclu
GRILLE = {
    "lissage_m": [0.2, 0.3, 0.5],
    "distance_min_m": [1.0, 1.5, 2.0, 3.0],
    "seuil": [0.05, 0.1, 0.15, 0.2],
}


def main():
    parser = argparse.ArgumentParser(description="Validation croisée par ferme de la baseline")
    parser.add_argument("--donnees", required=True, type=Path, help="dossier data/traite/reforestree")
    args = parser.parse_args()
    sortie = args.donnees / "validation_croisee_baseline"
    sortie.mkdir(exist_ok=True)
    cache = sortie / "grille_par_ferme.csv"
    deja = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["site"])

    # 1. Chaque combinaison de réglages, sur chaque ferme (une seule fois)
    verite = pd.read_csv(args.donnees / "verite_terrain.csv")
    parcelles = gpd.read_file(args.donnees / "parcelles_reforestree.geojson")
    for site in FERMES:
        if site in set(deja.site):
            continue
        donnees = charger_site(args.donnees, site, verite, parcelles)
        lignes = []
        for valeurs in itertools.product(*GRILLE.values()):
            reglages = dict(zip(GRILLE, valeurs))
            lignes.append({"site": site} | reglages | evaluer(donnees, reglages))
        pd.DataFrame(lignes).to_csv(cache, mode="a", header=not cache.exists(), index=False)
        print(f"{site} : {len(lignes)} réglages évalués")
    grille = pd.read_csv(cache)

    # 2. Les 5 tours : réglages choisis sur 4 fermes, note sur la 5e
    tours = []
    for validation in FERMES:
        apprentissage = grille[grille.site != validation]
        totaux = apprentissage.groupby(list(GRILLE))[["vp", "fp", "fn"]].sum()
        f1 = totaux.apply(lambda r: scores(r.vp, r.fp, r.fn)["f1"], axis=1)
        meilleur = dict(zip(GRILLE, f1.idxmax()))
        ligne = grille[(grille.site == validation) & (grille[list(GRILLE)] == pd.Series(meilleur)).all(axis=1)].iloc[0]
        tours.append({"ferme_validation": validation.replace(" RGB", ""), **meilleur,
                      "bananes_vraies": ligne.bananes_vraies, "detections": ligne.detections, "vp": ligne.vp,
                      "fp": ligne.fp, "fn": ligne.fn, "precision": ligne.precision, "rappel": ligne.rappel,
                      "f1": ligne.f1, "erreur_comptage": ligne.erreur_comptage})
    tours = pd.DataFrame(tours)
    pd.set_option("display.width", 200)
    print("\n" + tours.round(3).to_string(index=False))
    total = scores(tours.vp.sum(), tours.fp.sum(), tours.fn.sum())
    print(f"\nF1 globale sur les 5 fermes ({int(tours.bananes_vraies.sum())} bananiers) : {total['f1']:.3f} "
          f"(précision {total['precision']:.3f}, rappel {total['rappel']:.3f})")
    print(f"F1 par ferme : moyenne {tours.f1.mean():.3f}, de {tours.f1.min():.3f} à {tours.f1.max():.3f}")
    tours.to_csv(sortie / "resultats_par_ferme.csv", index=False)
    print(f"Résultats écrits dans {sortie}")


if __name__ == "__main__":
    main()
