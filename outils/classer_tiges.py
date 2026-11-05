"""
Classe toutes les photos préparées, de la plus « tige » à la plus « feuille », pour repérer les tiges
(coupes de pseudo-tronc) dans tous les jeux de feuilles.

Méthode : résumer chaque image avec un réseau pré-entraîné, puis la comparer à des exemples de tiges
et de feuilles, sur tous les jeux à la fois.

Principe :
  1. chaque image (224 x 224, sortie de preparer_feuilles.py) passe dans EfficientNet-B0 pré-entraîné
     sur ImageNet (torchvision), sans sa dernière couche : on obtient un vecteur de 1 280 nombres qui
     résume ce que montre l'image (formes, textures), pas seulement ses couleurs ;
  2. exemples : les tiges déjà listées dans outils/listes/tiges_tanzanie_2023.txt (relues dans les photos
     d'origine, avec la même préparation), et autant de feuilles de Tanzanie 2023 tirées au hasard ;
  3. score d'une image = ressemblance moyenne (cosinus) à ses 5 tiges les plus proches
     - ressemblance moyenne à ses 5 feuilles les plus proches ;
  4. chaque jeu est classé par score décroissant, et ses premières images sont mises en planches.
Le classement ne décide rien : il évite d'oublier des images. La décision se fait à l'œil sur les planches.

Sorties dans <dossier sortie> :
    vecteurs.npz                  les vecteurs calculés (repris tels quels si on relance)
    classement_tiges.csv          toutes les images : jeu, classe, rang dans le jeu, score, image, source
    planche_<jeu>_<n>.jpg          les premières images de chaque jeu, 96 par planche, numérotées par rang

Utilisation (PyTorch et torchvision nécessaires : pip install torch torchvision) :
    python outils/classer_tiges.py --brut C:/Field_scan/data/brut --feuilles C:/Field_scan/data/traite/feuilles_v2 --out C:/Field_scan/exploration/classement_tiges
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from preparer_feuilles import TAILLE, preparer   # même préparation (224 x 224) que les images classées

VOISINS = 5
PAR_PLANCHE = 96          # 12 x 8 vignettes
VIGNETTE = 120
LOT = 32


def creer_resumeur():
    """Renvoie une fonction : liste d'images PIL -> vecteurs normés (EfficientNet-B0, ImageNet)."""
    import torch
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
    poids = EfficientNet_B0_Weights.DEFAULT
    reseau = efficientnet_b0(weights=poids)
    reseau.classifier = torch.nn.Identity()
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    reseau = reseau.eval().to(appareil)
    transformation = poids.transforms()
    print(f"Réseau : EfficientNet-B0 sur {appareil}")

    def resumer(images):
        with torch.no_grad():
            x = torch.stack([transformation(im) for im in images]).to(appareil)
            v = reseau(x).cpu().numpy()
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    return resumer


def main():
    parser = argparse.ArgumentParser(description="Classement tige / feuille des photos préparées")
    parser.add_argument("--brut", required=True, type=Path, help="dossier data/brut (pour relire les tiges connues)")
    parser.add_argument("--feuilles", required=True, type=Path, help="dossier de sortie de preparer_feuilles.py")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--planches", type=int, default=480, help="nombre d'images mises en planches par jeu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    resumer = creer_resumeur()

    with open(args.feuilles / "images.csv", encoding="utf-8") as f:
        lignes = list(csv.DictReader(f))
    cache = args.out / "vecteurs.npz"
    cles = np.array([l["image"] for l in lignes])
    if cache.exists() and np.array_equal(np.load(cache)["cles"], cles):
        vecteurs = np.load(cache)["vecteurs"]
        print("Vecteurs repris de", cache)
    else:
        morceaux = []
        for i in range(0, len(lignes), LOT):
            lot = [Image.open(args.feuilles / l["image"]).convert("RGB") for l in lignes[i:i + LOT]]
            morceaux.append(resumer(lot))
            print(f"\r  {min(i + LOT, len(lignes))}/{len(lignes)} images résumées", end="", flush=True)
        print()
        vecteurs = np.concatenate(morceaux)
        np.savez(cache, cles=cles, vecteurs=vecteurs)

    # Exemples de tiges : celles déjà listées, relues dans les photos d'origine (preparer_feuilles.py les écarte)
    with open(Path(__file__).parent / "listes" / "tiges_tanzanie_2023.txt", encoding="utf-8") as f:
        tiges_connues = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    images_tiges = []
    for t in tiges_connues:
        dossier, fichier = t.split("/")
        chemin = args.brut / "tanzanie_2023_dataverse" / dossier / dossier / fichier
        if chemin.exists():
            with Image.open(chemin) as im:
                im.draft("RGB", (TAILLE * 2, TAILLE * 2))
                images_tiges.append(preparer(ImageOps.exif_transpose(im).convert("RGB")))
    if len(images_tiges) < 50:
        raise SystemExit(f"Seulement {len(images_tiges)} exemples de tiges retrouvés dans {args.brut}.")
    exemples_tiges = np.concatenate([resumer(images_tiges[i:i + LOT]) for i in range(0, len(images_tiges), LOT)])
    feuilles_2023 = [i for i, l in enumerate(lignes) if l["jeu"] == "tanzanie_2023"]
    exemples_feuilles = vecteurs[random.Random(0).sample(feuilles_2023, len(exemples_tiges))]
    print(f"Exemples : {len(exemples_tiges)} tiges, {len(exemples_feuilles)} feuilles")

    def proches(exemples):
        return np.sort(vecteurs @ exemples.T, axis=1)[:, -VOISINS:].mean(axis=1)
    scores = proches(exemples_tiges) - proches(exemples_feuilles)

    for jeu in sorted({l["jeu"] for l in lignes}):
        indices = sorted((i for i, l in enumerate(lignes) if l["jeu"] == jeu), key=lambda i: -scores[i])
        for rang, i in enumerate(indices, 1):
            lignes[i]["rang_dans_le_jeu"] = rang
            lignes[i]["score"] = round(float(scores[i]), 4)
        haut = indices[:args.planches]
        for p in range(0, len(haut), PAR_PLANCHE):
            bloc = haut[p:p + PAR_PLANCHE]
            planche = Image.new("RGB", (12 * VIGNETTE, 8 * (VIGNETTE + 14)), "white")
            dessin = ImageDraw.Draw(planche)
            for k, i in enumerate(bloc):
                x, y = (k % 12) * VIGNETTE, (k // 12) * (VIGNETTE + 14)
                planche.paste(Image.open(args.feuilles / lignes[i]["image"]).convert("RGB").resize((VIGNETTE, VIGNETTE)), (x, y + 14))
                dessin.text((x + 2, y + 1), f"{p + k + 1} {lignes[i]['classe'][:8]}", fill="red")
            planche.save(args.out / f"planche_{jeu}_{p // PAR_PLANCHE + 1:02d}.jpg", quality=85)
        print(f"{jeu:15s} {len(indices):6d} images classées, {len(haut)} en planches, "
              f"{int((scores[indices] > 0).sum())} plus proches d'une tige que d'une feuille")

    with open(args.out / "classement_tiges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["jeu", "classe", "rang_dans_le_jeu", "score", "image", "source"])
        w.writeheader()
        for l in sorted(lignes, key=lambda l: (l["jeu"], l["rang_dans_le_jeu"])):
            w.writerow({k: l[k] for k in w.fieldnames})
    print(f"Résultats écrits dans {args.out}")


if __name__ == "__main__":
    main()
