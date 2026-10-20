"""
Vérifie les images Sentinel-2 téléchargées depuis le Copernicus Browser.

Pour chaque site et chaque date, le script :
  1. vérifie que les 8 bandes et la carte SCL sont présentes ;
  2. vérifie que tous les fichiers ont la même taille (même grille de pixels) ;
  3. compte les pixels de chaque classe de la carte SCL (végétation, sol nu, nuages...) ;
  4. calcule le NDVI et le NDMI médians de la zone.

Organisation attendue des dossiers :
    <dossier sentinel2>/<site>/<date>/Browser_images/*.tiff

Utilisation (dans l'invite de commandes) :
    python outils/verifier_sentinel2.py C:/Field_scan/data/brut/sentinel2

Bibliothèques nécessaires : numpy, tifffile  (pip install numpy tifffile)
"""

import sys
from pathlib import Path

import numpy as np
import tifffile

# Bandes brutes attendues dans chaque dossier de date
BANDES = ["B02", "B03", "B04", "B05", "B08", "B8A", "B11", "B12"]

# Couleurs de la carte SCL telle que le Copernicus Browser l'exporte :
# chaque classe a une couleur (rouge, vert, bleu), avec des valeurs entre 0 et 1.
# Les deux premières ont été vérifiées sur nos fichiers ; les autres viennent de la
# palette standard de Sentinel Hub et seront signalées si elles apparaissent.
COULEURS_SCL = {
    (0.0, 0.627, 0.0): "végétation",
    (1.0, 0.902, 0.353): "sol nu",
    (0.0, 0.0, 1.0): "eau",
    (0.467, 0.298, 0.043): "ombre de nuage",
    (0.753, 0.753, 0.753): "nuage (probabilité moyenne)",
    (1.0, 1.0, 1.0): "nuage (probabilité forte)",
    (0.392, 0.784, 1.0): "cirrus",
    (0.0, 0.0, 0.0): "sans donnée",
}
CLASSES_NUAGEUSES = {"ombre de nuage", "nuage (probabilité moyenne)",
                     "nuage (probabilité forte)", "cirrus"}


def trouver_fichier(dossier, motif):
    """Renvoie le fichier .tiff dont le nom contient le motif (ex. '_B04_')."""
    fichiers = [f for f in dossier.rglob("*.tiff") if motif in f.name]
    if len(fichiers) != 1:
        raise FileNotFoundError(f"{len(fichiers)} fichier(s) '{motif}' dans {dossier}")
    return fichiers[0]


def indice_normalise(a, b):
    """Indice normalisé (a - b) / (a + b), comme le NDVI ou le NDMI."""
    return (a - b) / (a + b)


def verifier_date(dossier):
    print(f"\n--- {dossier.parent.name} / {dossier.name} ---")

    # 1. Lecture des 8 bandes
    bandes = {}
    for nom in BANDES:
        bandes[nom] = tifffile.imread(trouver_fichier(dossier, f"_{nom}_")).astype("float32")

    # 2. Toutes les bandes doivent avoir la même taille
    tailles = {nom: b.shape for nom, b in bandes.items()}
    taille = tailles["B02"]
    alignees = all(t == taille for t in tailles.values())
    print(f"Taille : {taille[1]} x {taille[0]} pixels | bandes alignées : {alignees}")

    # 3. Classes de la carte SCL (image en 3 couleurs)
    scl = tifffile.imread(trouver_fichier(dossier, "Scene_classification")).astype("float32")
    if scl.shape[0] == 3:                      # (couleurs, hauteur, largeur) -> (hauteur, largeur, couleurs)
        scl = np.moveaxis(scl, 0, -1)
    pixels = np.round(scl.reshape(-1, 3), 3)   # une ligne par pixel : (rouge, vert, bleu)
    couleurs, nombres = np.unique(pixels, axis=0, return_counts=True)
    total = pixels.shape[0]
    pixels_nuageux = 0
    pixels_inconnus = 0
    for couleur, nombre in zip(couleurs, nombres):
        cle = tuple(round(float(c), 3) for c in couleur)   # arrondi pour comparer avec le tableau
        classe = COULEURS_SCL.get(cle, f"couleur non reconnue {cle}")
        if classe in CLASSES_NUAGEUSES:
            pixels_nuageux += nombre
        if classe.startswith("couleur non reconnue"):
            pixels_inconnus += nombre
        print(f"  {classe:28s} {nombre:7d} pixels ({100 * nombre / total:5.1f} %)")
    print(f"Pixels nuageux ou ombragés : {pixels_nuageux}")
    if pixels_inconnus:
        print(f"ATTENTION : {pixels_inconnus} pixels de couleur non reconnue, à vérifier dans QGIS")

    # 4. Indices médians sur toute la zone
    ndvi = indice_normalise(bandes["B08"], bandes["B04"])   # vigueur
    ndmi = indice_normalise(bandes["B8A"], bandes["B11"])   # humidité des feuilles
    print(f"NDVI médian : {np.nanmedian(ndvi):.2f} | NDMI médian : {np.nanmedian(ndmi):.2f}")


def main():
    racine = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("C:/Field_scan/data/brut/sentinel2")
    for site in sorted(p for p in racine.iterdir() if p.is_dir()):
        for date in sorted(p for p in site.iterdir() if p.is_dir()):
            verifier_date(date)


if __name__ == "__main__":
    main()
