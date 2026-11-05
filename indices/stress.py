"""Stress hydrique : zones qui manquent probablement d'eau (Sentinel-2, NDMI).

Le NDMI mesure l'eau contenue dans le feuillage, mais il monte aussi avec la quantité de feuilles :
une parcelle clairsemée a un NDMI bas sans manquer d'eau. Je compare donc chaque pixel à ce qu'on
attend pour sa quantité de feuillage (droite NDMI = a + b x NDVI ajustée sur les pixels de végétation
de la plantation) : un pixel dont le NDMI est plus bas que prévu de plus de 0,05 manque probablement
d'eau. Les zones de vigueur faible ne sont pas jugées ici (leur NDMI bas vient du manque de feuilles).

Avec deux dates, je compare aussi les saisons : la date où le NDMI médian de la plantation est le plus
haut est la saison humide, l'autre la saison sèche. Une parcelle dont le NDMI baisse de plus de 0,05
de plus que la baisse habituelle des parcelles est signalée.
"""

import numpy as np

ECART_EAU = 0.05
ECART_SAISON = 0.05


def zones_seches(ndvi, ndmi, vegetation):
    """Pixels qui manquent probablement d'eau, et la droite (a, b) attendue NDMI = a + b x NDVI."""
    ok = vegetation & np.isfinite(ndvi) & np.isfinite(ndmi)
    if ok.sum() < 10:
        return np.zeros(ndvi.shape, bool), (np.nan, np.nan)
    b, a = np.polyfit(ndvi[ok], ndmi[ok], 1)
    with np.errstate(invalid="ignore"):
        seche = ok & (ndmi - (a + b * ndvi) < -ECART_EAU)
    return seche, (float(a), float(b))


def comparer_saisons(ndmi_par_date, comparables):
    """ndmi_par_date : {date: {id_parcelle: NDMI médian}} pour 2 dates ou plus ; comparables : parcelles
    sans zone de vigueur faible aux deux dates. Renvoie (date humide, date sèche, {id: baisse}, ids signalés)."""
    medianes = {d: np.nanmedian(list(v.values())) for d, v in ndmi_par_date.items()}
    humide, seche = max(medianes, key=medianes.get), min(medianes, key=medianes.get)
    baisse = {i: ndmi_par_date[humide][i] - ndmi_par_date[seche][i] for i in ndmi_par_date[humide]}
    ref = [baisse[i] for i in comparables if np.isfinite(baisse[i])]
    if not ref:
        return humide, seche, baisse, []
    habituelle = float(np.median(ref))
    signales = [i for i in comparables if baisse[i] > habituelle + ECART_SAISON]
    return humide, seche, baisse, signales
