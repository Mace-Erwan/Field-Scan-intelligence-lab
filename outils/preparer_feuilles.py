"""
Prépare les photos de feuilles pour le modèle maladies.

Pour chaque jeu (Tanzanie 2023, Tanzanie 2025, Pérou, Bangladesh), le script :
  1. écarte les photos de tiges (liste dans outils/listes/tiges_tanzanie_2023.txt) ;
  2. ramène chaque image à 224 x 224 pixels (côté le plus court à 224, puis recadrage au centre) ;
  3. repère les quasi-doublons, à l'intérieur de chaque jeu et entre chaque paire de jeux
     (dont le Bangladesh, le jeu de test),
     en trois temps :
       - une empreinte rapide (dHash, 64 bits) donne des paires candidates ;
       - 1re confirmation : les deux images, réduites à 32 x 32 pixels en couleur, sont très
         corrélées (corrélation >= SEUIL_CORRELATION) ;
       - 2e confirmation : elles le sont encore en 112 x 112 pixels en gris
         (corrélation >= SEUIL_CORRELATION_FINE).
     L'empreinte et la 1re confirmation ne suffisent pas : sur des gros plans de feuilles à nervures
     parallèles, elles associent des milliers de feuilles différentes, souvent de classes différentes.
     En 112 x 112, les détails (nervures, taches) ne tombent au même endroit que pour deux copies de
     la même photo, ou deux photos de la même feuille prises d'affilée. Les paires restantes sont
     à vérifier à l'œil.

Les images d'origine ne sont jamais modifiées. Si le script est interrompu, il suffit de le relancer :
les images déjà préparées sont reprises telles quelles.

Sorties dans <dossier sortie> :
    images/<jeu>/<classe>/<fichier>.jpg   images prêtes, en 224 x 224
    images.csv                             une ligne par image (source, jeu, classe, sortie, empreinte)
    doublons.csv                           paires d'images quasi identiques

Utilisation :
    python outils/preparer_feuilles.py --brut C:/Field_scan/data/brut --out C:/Field_scan/data/traite/feuilles
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

TAILLE = 224
SEUIL_DOUBLON = 4         # empreinte : nombre maximal de bits différents (sur 64) pour une paire candidate
SEUIL_CORRELATION = 0.9   # 1re confirmation : corrélation minimale en 32 x 32 couleur
SEUIL_CORRELATION_FINE = 0.9   # 2e confirmation : corrélation minimale en 112 x 112 gris

DOSSIER_BD = ("bangladesh_bananalsd/Banana Leaf Spot Diseases (BananaLSD) Dataset for "
              "Classification of Banana Leaf Diseases Using Machine Learning/BananaLSD/OriginalSet")

DOSSIER_PEROU = "perou_mendeley/Banana Leaf Image Dataset for Classification of Si"

# (jeu, dossier relatif à --brut, classe)
SOURCES = [
    ("tanzanie_2023", "tanzanie_2023_dataverse/HEALTHY_1/HEALTHY_1", "saine"),
    ("tanzanie_2023", "tanzanie_2023_dataverse/HEALTHY_2/HEALTHY_2", "saine"),
    ("tanzanie_2023", "tanzanie_2023_dataverse/BLACK SIGATOKA_1/BLACK SIGATOKA_1", "black_sigatoka"),
    ("tanzanie_2023", "tanzanie_2023_dataverse/BLACK SIGATOKA_2/BLACK SIGATOKA_2", "black_sigatoka"),
    ("tanzanie_2023", "tanzanie_2023_dataverse/FUSARIUM WILT_1/FUSARIUM WILT_1", "fusarium"),
    ("tanzanie_2023", "tanzanie_2023_dataverse/FUSARIUM WILT_2/FUSARIUM WILT_2", "fusarium"),
    ("tanzanie_2025", "tanzanie_2025_zenodo/HEALTHY-1/HEALTHY-1", "saine"),
    ("tanzanie_2025", "tanzanie_2025_zenodo/BLACK SIGATOKA-1/BLACK SIGATOKA-1", "black_sigatoka"),
    ("tanzanie_2025", "tanzanie_2025_zenodo/FUSARIUM WILT-1/FUSARIUM WILT-1", "fusarium"),
    ("tanzanie_2025", "tanzanie_2025_zenodo/HEALTHY-2/HEALTHY-2", "saine"),
    ("tanzanie_2025", "tanzanie_2025_zenodo/HEALTHY-3/HEALTHY-3", "saine"),
    ("tanzanie_2025", "tanzanie_2025_zenodo/BLACK SIGATOKA-2/BLACK SIGATOKA-2", "black_sigatoka"),
    ("tanzanie_2025", "tanzanie_2025_zenodo/BLACK SIGATOKA-3/BLACK SIGATOKA-3", "black_sigatoka"),
    ("tanzanie_2025", "tanzanie_2025_zenodo/FUSARIUM WILT-2/FUSARIUM WILT-2", "fusarium"),
    ("tanzanie_2025", "tanzanie_2025_zenodo/FUSARIUM WILT-3/FUSARIUM WILT-3", "fusarium"),
    # Pérou : la Cordana n'est volontairement pas déclarée (maladie inconnue du test, jamais lue)
    ("perou", f"{DOSSIER_PEROU}/sanas", "saine"),
    ("perou", f"{DOSSIER_PEROU}/sigatoka", "black_sigatoka"),
    ("bangladesh", f"{DOSSIER_BD}/healthy", "saine"),
    ("bangladesh", f"{DOSSIER_BD}/sigatoka", "sigatoka"),
    ("bangladesh", f"{DOSSIER_BD}/cordana", "cordana"),
    ("bangladesh", f"{DOSSIER_BD}/pestalotiopsis", "pestalotiopsis"),
]

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def lire_tiges(fichier):
    """Renvoie l'ensemble des chemins 'DOSSIER/Image_n.jpg' à écarter."""
    with open(fichier, encoding="utf-8") as f:
        return {l.strip() for l in f if l.strip() and not l.startswith("#")}


def preparer(image):
    """Côté le plus court à TAILLE, puis recadrage carré au centre."""
    return ImageOps.fit(image, (TAILLE, TAILLE), method=Image.BILINEAR)


def empreinte(image):
    """dHash sur 64 bits : compare chaque pixel à son voisin de droite, sur une image 9 x 8 en gris."""
    petit = np.asarray(image.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (petit[:, 1:] > petit[:, :-1]).flatten()
    return np.packbits(bits)  # 8 octets


def vecteur(image):
    """Image réduite à 32 x 32 en couleur, centrée et normée : sert à comparer deux images."""
    v = np.asarray(image.convert("RGB").resize((32, 32), Image.BILINEAR), dtype=np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def vecteur_fin(image):
    """Image réduite à 112 x 112 en gris, centrée et normée : comparaison plus fine, sur les détails."""
    v = np.asarray(image.convert("L").resize((112, 112), Image.BILINEAR), dtype=np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def distances(a, b):
    """Nombre de bits différents entre chaque empreinte de a et chaque empreinte de b."""
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return table[a[:, None, :] ^ b[None, :, :]].sum(axis=2)


def chercher_doublons(hashes, groupe_a, groupe_b, meme_groupe):
    """Renvoie les paires (i, j, distance) avec distance <= SEUIL_DOUBLON."""
    paires = []
    a, b = np.array(groupe_a), np.array(groupe_b)
    for debut in range(0, len(a), 500):
        bloc = a[debut:debut + 500]
        d = distances(hashes[bloc], hashes[b])
        for i, j in zip(*np.nonzero(d <= SEUIL_DOUBLON)):
            ia, jb = int(bloc[i]), int(b[j])
            if meme_groupe and jb <= ia:
                continue
            paires.append((ia, jb, int(d[i, j])))
    return paires


def main():
    parser = argparse.ArgumentParser(description="Préparation des photos de feuilles")
    parser.add_argument("--brut", required=True, type=Path, help="dossier data/brut")
    parser.add_argument("--out", required=True, type=Path, help="dossier de sortie")
    args = parser.parse_args()

    tiges = lire_tiges(Path(__file__).parent / "listes" / "tiges_tanzanie_2023.txt")
    lignes, hashes, vecteurs = [], [], []
    tiges_ecartees = 0

    for jeu, dossier, classe in SOURCES:
        source = args.brut / dossier
        sortie = args.out / "images" / jeu / classe
        sortie.mkdir(parents=True, exist_ok=True)
        fichiers = sorted(f for f in source.iterdir()
                          if f.suffix.lower() in EXTENSIONS and not f.name.startswith("."))
        gardees = 0
        for f in fichiers:
            if f"{source.name}/{f.name}" in tiges:
                tiges_ecartees += 1
                continue
            nom = f"{source.name.replace(' ', '_')}__{f.stem}.jpg"
            if (sortie / nom).exists():
                # Déjà préparée lors d'un lancement précédent : on la relit seulement
                petite = Image.open(sortie / nom).convert("RGB")
            else:
                with Image.open(f) as im:
                    im.draft("RGB", (TAILLE * 2, TAILLE * 2))   # décodage JPEG réduit : beaucoup plus rapide
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    petite = preparer(im)
                petite.save(sortie / nom, quality=95)
            h = empreinte(petite)
            hashes.append(h)
            vecteurs.append(vecteur(petite))
            lignes.append({
                "source": f"{dossier}/{f.name}", "jeu": jeu, "classe": classe,
                "image": f"images/{jeu}/{classe}/{nom}", "empreinte": h.tobytes().hex(),
            })
            gardees += 1
        print(f"{jeu:14s} {classe:15s} {gardees:6d} images  ({source.name})")
    print(f"Tiges écartées : {tiges_ecartees}")

    with open(args.out / "images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(lignes[0]))
        w.writeheader()
        w.writerows(lignes)

    # Quasi-doublons : dans chaque jeu, et entre chaque paire de jeux
    hashes = np.stack(hashes)
    par_jeu = {}
    for i, l in enumerate(lignes):
        par_jeu.setdefault(l["jeu"], []).append(i)
    paires = []
    for jeu, indices in par_jeu.items():
        paires += chercher_doublons(hashes, indices, indices, meme_groupe=True)
    # Entre deux jeux différents, quels qu'ils soient : le test (Bangladesh) ne doit avoir aucune image
    # en commun avec l'entraînement ou la validation
    jeux = list(par_jeu)
    for k, jeu_a in enumerate(jeux):
        for jeu_b in jeux[k + 1:]:
            paires += chercher_doublons(hashes, par_jeu[jeu_a], par_jeu[jeu_b], meme_groupe=False)

    print(f"Paires candidates (empreinte, au plus {SEUIL_DOUBLON} bits différents sur 64) : {len(paires)}")
    fins = {}   # vecteurs fins calculés seulement pour les images qui en ont besoin

    def fin(i):
        if i not in fins:
            fins[i] = vecteur_fin(Image.open(args.out / lignes[i]["image"]))
        return fins[i]

    confirmees, premiere = [], 0
    for i, j, d in paires:
        correlation = float(vecteurs[i] @ vecteurs[j])
        if correlation < SEUIL_CORRELATION:
            continue
        premiere += 1
        correlation_fine = float(fin(i) @ fin(j))
        if correlation_fine >= SEUIL_CORRELATION_FINE:
            confirmees.append((i, j, d, correlation, correlation_fine))
    confirmees.sort(key=lambda p: -p[4])
    print(f"Après la 1re confirmation (32 x 32, corrélation >= {SEUIL_CORRELATION}) : {premiere}")

    with open(args.out / "doublons.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_1", "classe_1", "image_2", "classe_2", "bits_differents", "correlation",
                    "correlation_fine"])
        for i, j, d, c, cf in confirmees:
            w.writerow([lignes[i]["image"], lignes[i]["classe"], lignes[j]["image"], lignes[j]["classe"],
                        d, round(c, 3), round(cf, 3)])

    print(f"Paires confirmées (2e confirmation, 112 x 112, corrélation >= {SEUIL_CORRELATION_FINE}) : "
          f"{len(confirmees)}")
    compte = {}
    for i, j, *_ in confirmees:
        cle = f"{lignes[i]['jeu']} / {lignes[j]['jeu']}"
        compte[cle] = compte.get(cle, 0) + 1
    for cle, n in sorted(compte.items()):
        print(f"    {cle:32s} {n:6d}")
    print(f"Résultats écrits dans {args.out}")


if __name__ == "__main__":
    main()
