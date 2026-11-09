"""Détecteur de centres de bananiers (modèle retenu, voir docs/experiences.md, section 1).

Le réseau produit une carte : pour chaque case de 20 cm (4 x 4 pixels à 5 cm), la probabilité
qu'un centre de bananier s'y trouve. Un bananier = une case plus forte que toutes ses voisines
dans un carré de 1,4 m, au-dessus du seuil. Le modèle final fait la moyenne de 2 réseaux,
chacun sur 4 vues de l'image (l'image, ses deux miroirs, un quart de tour).

Les images doivent être à 5 cm par pixel (run.py s'en charge).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

PIXEL_M = 0.05          # taille d'un pixel attendue par le modèle (m)
SEUIL = 0.25            # équilibre fausses alertes / oublis, choisi sur la validation de l'essai v3 (docs/experiences.md)
FENETRE_PIC = 7         # 7 x 7 cases de 20 cm = 1,4 m
TUILE = 2048            # les grandes images sont traitées par morceaux de 2048 px...
MARGE = 128             # ... qui se recouvrent de 2 x 128 px, pour éviter les effets de bord

MOY = np.array([0.485, 0.456, 0.406], np.float32)   # normalisation ImageNet
ECT = np.array([0.229, 0.224, 0.225], np.float32)


class DetecteurCentres(nn.Module):
    """ResNet-18 + décodeur qui additionne ses 4 étages ; sortie au quart de la résolution."""

    def __init__(self):
        super().__init__()
        r = models.resnet18(weights=None)            # les poids viennent du fichier entraîné
        self.souche = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.etages = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])
        self.lateral = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in (64, 128, 256, 512)])
        self.tete = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 1, 1))

    def forward(self, x):
        x = self.souche(x)
        sorties = []
        for etage in self.etages:
            x = etage(x)
            sorties.append(x)
        y = self.lateral[3](sorties[3])
        for k in (2, 1, 0):
            y = F.interpolate(y, size=sorties[k].shape[-2:], mode="nearest") + self.lateral[k](sorties[k])
        return self.tete(y)


def charger(chemins):
    """Charge les réseaux ; les poids enregistrés en 16 bits sont remis en 32 bits pour le calcul sur CPU."""
    if not chemins:
        raise FileNotFoundError("Aucun fichier de poids du comptage (.pt) : ils doivent être dans modeles/comptage/")
    reseaux = []
    for chemin in chemins:
        poids = torch.load(chemin, map_location="cpu", weights_only=True)
        reseau = DetecteurCentres()
        reseau.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in poids.items()})
        reseaux.append(reseau.eval())
    return reseaux


def _vue(x, k):
    return [x, x.flip(-1), x.flip(-2), x.rot90(1, (-2, -1))][k]


def _vue_inverse(c, k):
    return [c, c.flip(-1), c.flip(-2), c.rot90(-1, (-2, -1))][k]


@torch.no_grad()
def _carte_morceau(reseaux, morceau):
    """Carte de probabilité d'un morceau d'image (moyenne des réseaux et des 4 vues)."""
    h, w = morceau.shape[:2]
    H, W = math.ceil(h / 32) * 32, math.ceil(w / 32) * 32
    a = np.zeros((H, W, 3), np.float32)
    a[:h, :w] = morceau / 255
    x = torch.from_numpy(((a - MOY) / ECT).transpose(2, 0, 1).copy())[None]
    somme = 0
    for reseau in reseaux:
        for k in range(4):
            somme = somme + _vue_inverse(torch.sigmoid(reseau(_vue(x, k))), k)
    return (somme / (4 * len(reseaux)))[0, 0].numpy()


def _positions(n):
    """Débuts des morceaux le long d'un côté de n pixels (n multiple de 4)."""
    if n <= TUILE:
        return [0]
    return list(range(0, n - TUILE, TUILE - 2 * MARGE)) + [n - TUILE]


def carte(reseaux, image):
    """Carte de probabilité de toute l'image (une case = 4 x 4 pixels).
    Une image plus grande que TUILE est découpée en morceaux qui se recouvrent ;
    de chaque morceau, on ne garde que l'intérieur (sans la marge, sauf au bord de l'image)."""
    h, w = image.shape[:2]
    H, W = math.ceil(h / 4) * 4, math.ceil(w / 4) * 4          # complété par du noir jusqu'à un multiple de 4
    if (H, W) != (h, w):
        complete = np.zeros((H, W, 3), image.dtype)
        complete[:h, :w] = image
        image = complete
    if max(H, W) <= TUILE:
        return _carte_morceau(reseaux, image)[:H // 4, :W // 4]
    somme = np.zeros((H // 4, W // 4), np.float32)
    nombre = np.zeros((H // 4, W // 4), np.float32)
    for y0 in _positions(H):
        for x0 in _positions(W):
            morceau = image[y0:y0 + TUILE, x0:x0 + TUILE]
            if not morceau.any():
                continue                                       # morceau tout noir : rien à chercher
            c = _carte_morceau(reseaux, morceau)
            hm, wm = morceau.shape[:2]
            haut, gauche = (0 if y0 == 0 else MARGE), (0 if x0 == 0 else MARGE)
            bas = hm if y0 + hm >= H else hm - MARGE
            droite = wm if x0 + wm >= W else wm - MARGE
            zone = (slice((y0 + haut) // 4, (y0 + bas) // 4), slice((x0 + gauche) // 4, (x0 + droite) // 4))
            somme[zone] += c[haut // 4:bas // 4, gauche // 4:droite // 4]
            nombre[zone] += 1
    return somme / np.maximum(nombre, 1)


def pics(c, seuil=SEUIL):
    """Positions (x, y en pixels de l'image) et scores des sommets de la carte au-dessus du seuil."""
    t = torch.from_numpy(c)[None, None]
    maxi = F.max_pool2d(t, FENETRE_PIC, stride=1, padding=FENETRE_PIC // 2)
    yy, xx = torch.nonzero((t == maxi)[0, 0] & (t[0, 0] >= seuil), as_tuple=True)
    xy = np.c_[(xx.numpy() + 0.5) * 4, (yy.numpy() + 0.5) * 4].astype(np.float32)
    return xy, c[yy.numpy(), xx.numpy()]


def detecter(reseaux, image, seuil=SEUIL):
    """Bananiers d'une image RGB (uint8, h x w x 3) à 5 cm par pixel.
    Les détections tombant sur un pixel noir (hors image, sans donnée) sont ignorées."""
    xy, scores = pics(carte(reseaux, image), seuil)
    h, w = image.shape[:2]
    dedans = (xy[:, 0] < w) & (xy[:, 1] < h)
    xy, scores = xy[dedans], scores[dedans]
    if len(xy):
        noir = image[xy[:, 1].astype(int), xy[:, 0].astype(int)].sum(1) == 0
        xy, scores = xy[~noir], scores[~noir]
    return xy, scores
