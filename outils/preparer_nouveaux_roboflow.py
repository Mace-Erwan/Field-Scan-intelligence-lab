"""Prépare les 3 nouveaux jeux Roboflow pour l'apprentissage du comptage (format de outils/preparer_comptage.py).

- une seule image par original (les copies modifiées par Roboflow sont écartées) ;
- un point par plant (centre de la boîte) et son diamètre (racine de la surface de la boîte) ;
  plantas_platano n'a que de petites boîtes sur le cœur du plant : diamètre fixé à 2,0 m ;
- échelle : taille du pixel inconnue. La distance médiane d'un plant à son plus proche voisin est ramenée à
  2,0 m (écartement 2 x 2 à 3 x 3 m recommandé par l'IITA ; 1,9 à 2,0 m dans Count banana et corte),
  calculée séparément pour chaque orthomosaïque ; puis l'image est remise à 5 cm par pixel ;
- tout va à l'apprentissage (aucune image de ces jeux ne sert à noter).
Usage : python preparer_nouveaux.py DOSSIER_ZIPS_DEZIPPES SORTIE
"""
import sys, os, re, json, collections, numpy as np, pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

SRC, SORTIE = sys.argv[1:3]
os.makedirs(f'{SORTIE}/images', exist_ok=True)
PIXEL_M, ECART_M, DIAM_DEFAUT_M = 0.05, 2.0, 2.0
JEUX = {'tree': 'jeunes_rangs', 'platano': 'plantain', 'lasuiza': 'jeunes_lasuiza'}


def originaux(dossier):
    """{clé: (chemin, boîtes)} : une image par original, de préférence non modifiée (valid, test)."""
    choix = {}
    for partie in ['valid', 'test', 'train']:
        f = f'{dossier}/{partie}/_annotations.coco.json'
        if not os.path.exists(f):
            continue
        j = json.load(open(f))
        boites = collections.defaultdict(list)
        for a in j['annotations']:
            boites[a['image_id']].append(a['bbox'])
        for im in j['images']:
            base = re.sub(r'_(jpg|jpeg|png)\.rf\..*$', '', im['file_name'])
            cle = (base, im['width'], im['height'])            # tree : deux découpages (1024 et 1280 px) aux mêmes noms
            if cle not in choix:
                choix[cle] = (f'{dossier}/{partie}/{im["file_name"]}', np.array(boites[im['id']], np.float32).reshape(-1, 4))
    return choix


def origine(jeu, base, w):
    if jeu == 'platano':
        return re.sub(r'_chunk_.*$', '', base)                 # 100_metros, lote_2, lote_2_parte_2
    if jeu == 'tree':
        return f'uav20221018_{w}px'
    return 'lasuiza'


lignes_im, lignes_pt, bilan = [], [], []
for d, jeu in JEUX.items():
    imgs = originaux(f'{SRC}/{d}')
    groupes = collections.defaultdict(list)
    for (base, w, h), v in imgs.items():
        groupes[origine(d, base, w)].append(((base, w, h), v))
    for org, liste in groupes.items():
        dist = []
        for _, (_, b) in liste:
            if len(b) > 2:
                p = b[:, :2] + b[:, 2:] / 2
                dist += list(cKDTree(p).query(p, k=2)[0][:, 1])
        nn = float(np.median(dist))
        f = ECART_M / (nn * PIXEL_M)                          # facteur pour passer à 5 cm par pixel
        n_pl = 0
        for (base, w, h), (chemin, b) in liste:
            nom = f'{jeu}__{base}_{w}.jpg'
            im = Image.open(chemin).convert('RGB')
            W, H = max(32, round(im.width * f)), max(32, round(im.height * f))
            im.resize((W, H), Image.LANCZOS).save(f'{SORTIE}/images/{nom}', quality=95)
            lignes_im.append({'image': nom, 'jeu': jeu, 'partie': 'apprentissage', 'origine': org,
                              'largeur': W, 'hauteur': H, 'pixel_m': PIXEL_M})
            for x, y, bw, bh in b:
                diam = DIAM_DEFAUT_M / PIXEL_M if d == 'platano' else np.sqrt(bw * bh) * f
                lignes_pt.append({'image': nom, 'x': round(float((x + bw / 2) * f), 1),
                                  'y': round(float((y + bh / 2) * f), 1), 'diametre': round(float(diam), 1)})
                n_pl += 1
        bilan.append({'jeu': jeu, 'origine': org, 'images': len(liste), 'plants': n_pl,
                      'voisin_median_px': round(nn, 1), 'pixel_d_origine_cm': round(100 * ECART_M / nn, 2),
                      'facteur': round(f, 3)})
pd.DataFrame(lignes_im).to_csv(f'{SORTIE}/images_nouveaux.csv', index=False)
pd.DataFrame(lignes_pt).to_csv(f'{SORTIE}/points_nouveaux.csv', index=False)
b = pd.DataFrame(bilan); b.to_csv(f'{SORTIE}/bilan_nouveaux.csv', index=False)
print(b.to_string(index=False))
