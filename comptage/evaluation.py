"""Évaluation du comptage, selon la règle du sujet : une détection est juste si elle tombe
à moins de 1 m d'un vrai pied, chaque vrai pied ne pouvant être trouvé qu'une fois."""

import numpy as np
from scipy.optimize import linear_sum_assignment

RAYON = 1.0  # mètres


def apparier(detections, verite, rayon=RAYON):
    """detections, verite : tableaux (n, 2) de positions en mètres.
    Renvoie (vrais positifs, fausses détections, pieds manqués)."""
    if len(detections) == 0 or len(verite) == 0:
        return 0, len(detections), len(verite)
    distances = np.linalg.norm(detections[:, None, :] - verite[None, :, :], axis=2)
    cout = np.where(distances <= rayon, distances, 1e6)          # interdit les paires à plus de 1 m
    lignes, colonnes = linear_sum_assignment(cout)               # meilleure association un pour un
    vp = int((cout[lignes, colonnes] <= rayon).sum())
    return vp, len(detections) - vp, len(verite) - vp


def scores(vp, fp, fn):
    precision = vp / (vp + fp) if vp + fp else 0.0
    rappel = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precision * rappel / (precision + rappel) if precision + rappel else 0.0
    return {"precision": precision, "rappel": rappel, "f1": f1}
