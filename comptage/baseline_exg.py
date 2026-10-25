"""Baseline classique de détection des pieds, sans apprentissage :
  1. indice de verdure ExG = 2g - r - b (r, g, b = parts de rouge, vert, bleu dans le pixel) ;
  2. lissage, pour qu'une couronne forme une seule « bosse » ;
  3. maxima locaux : un pixel plus vert que tous ses voisins dans un rayon donné = un pied."""

import numpy as np
from scipy import ndimage


def exg(image):
    """image : tableau (3, hauteur, largeur) en RVB. Renvoie l'ExG de chaque pixel."""
    r, v, b = image[:3].astype(np.float32)
    somme = r + v + b + 1e-6
    return 2 * v / somme - r / somme - b / somme


def detecter(image, valide, taille_pixel_m, lissage_m=0.5, distance_min_m=2.0, seuil=0.05, carte=None):
    """Renvoie les positions (ligne, colonne) des pieds détectés.
    valide : pixels qui font partie de l'image (hors bords noirs).
    carte : ExG déjà calculée (pour éviter de la recalculer)."""
    if carte is None:
        carte = exg(image)
    lisse = ndimage.gaussian_filter(carte, sigma=lissage_m / taille_pixel_m)
    taille = max(3, int(round(distance_min_m / taille_pixel_m)) | 1)   # fenêtre impaire
    maximum = ndimage.maximum_filter(lisse, size=taille)
    pics = (lisse == maximum) & (lisse > seuil) & valide
    return np.argwhere(pics)
