"""Prépare les données du comptage des bananiers : trois jeux mis au même format et à la même échelle.

Pour chaque jeu, les images sont remises à 5 cm par pixel (ReforesTree, dont la taille du pixel est connue)
ou à une échelle où le bananier médian a la même taille que dans ReforesTree (jeux Roboflow, taille du
pixel inconnue). Chaque bananier devient un point (centre de sa boîte) avec son diamètre en pixels.

Jeux et découpage (vérifiés par outils/verifier_doublons_comptage.py) :
  - ReforesTree : une image par ferme, recadrée sur la parcelle, noir hors de la parcelle (hors parcelle,
    les bananiers ne sont pas tous annotés). Apprentissage : Flora, Leonor, Guevara, Arteaga ;
    validation : Manuel ; test : Nestor ;
  - Count banana plants (v15) : apprentissage = vols 100_0221 et 100_0222, validation = 100_0223 et 100_0224 ;
  - Plants-bana : seulement les sources `segmento` (grande image drone, découpée en 3 zones éloignées :
    apprentissage, validation, test) et `corte` (apprentissage), une seule image par nom d'origine.
    Les autres sources sont écartées (copies de Count banana, vues en oblique, bananiers non annotés).

Sorties dans <sortie>/ :
    images/<jeu>__<nom>.jpg
    images.csv   image, jeu, partie, origine, largeur, hauteur, pixel_m (taille du pixel en mètres, estimée
                 pour les jeux Roboflow)
    points.csv   image, x, y, diametre (en pixels) : un bananier par ligne
    puis <sortie>.zip à envoyer sur Kaggle

Usage :
    python outils/preparer_comptage.py --donnees C:/Field_scan/data --sortie C:/Field_scan/data/traite/comptage
"""
import argparse, json, re, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from PIL import Image
from rasterio.features import geometry_mask
from shapely import force_2d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preparer_reforestree import REDUCTION            # noqa: E402

PIXEL_M = 0.05            # taille de pixel commune
MARGE_M = 2.0             # marge autour de la parcelle (m)
TAILLE_MIN_BOITE = 5      # boîtes plus petites (px d'origine) écartées : erreurs d'annotation
QUALITE_JPEG = 92
FERMES = {'Flora Pluas RGB': 'apprentissage', 'Leonor Aspiazu RGB': 'apprentissage',
          'Carlos Vera Guevara RGB': 'apprentissage', 'Carlos Vera Arteaga RGB': 'apprentissage',
          'Manuel Macias RGB': 'validation', 'Nestor Macias RGB': 'test'}
VOLS_COUNT = {'100_0221': 'apprentissage', '100_0222': 'apprentissage', '100_0223': 'validation', '100_0224': 'validation'}
COUNT = 'brut/count_banana_plants/Count banana plants.v15-dataset-final-sin-aug.coco'
PLANTS = 'brut/plants_banana/Plants-bana.v1i.coco'


def zone_segmento(x, y):
    # Coordonnées du morceau dans la grande image (en pixels de 640). Trois groupes éloignés l'un de l'autre :
    # à gauche (47 morceaux) apprentissage ; en bas à droite, colonnes 22 à 27 validation, 28 à 35 test ;
    # en haut (8 morceaux isolés) test.
    c, r = (x - 11520) // 640, (y - 11520) // 640
    if r < 10:
        return 'test', 'segmento_haut'
    if c < 12:
        return 'apprentissage', 'segmento_gauche'
    return ('validation', 'segmento_bas_droite_ouest') if c <= 27 else ('test', 'segmento_bas_droite_est')


def enregistrer(img, nom, sortie):
    img.convert('RGB').save(sortie / 'images' / nom, quality=QUALITE_JPEG)


def reforestree(donnees, sortie):
    rt = donnees / 'traite/reforestree'
    verite = pd.read_csv(rt / 'verite_terrain.csv')
    verite = verite[verite.est_banane]
    parcelles = gpd.read_file(rt / 'parcelles_reforestree.geojson')
    images, points, diametres_m = [], [], []
    for site, partie in FERMES.items():
        with rasterio.open(rt / 'orthos' / f'{site}.tif') as src:
            img = np.moveaxis(src.read([1, 2, 3]), 0, -1); t = src.transform
        arbres = verite[verite.site == site].copy()
        pixel = abs(t.a) * 111320 * np.cos(np.radians(arbres.lat.mean()))      # m par pixel réduit
        zone = force_2d(parcelles[parcelles.site == site].geometry.union_all())
        masque = geometry_mask([zone], out_shape=img.shape[:2], transform=t, invert=True)
        img = img * masque[..., None]
        # Position = centre de la boîte dessinée sur l'image (x_px, y_px : pixels de l'orthomosaïque d'origine).
        # Les colonnes lon/lat des annotations sont à environ 20 m des boîtes sur 5 fermes sur 6 et ne tombent
        # pas sur les couronnes : elles ne servent pas ici.
        arbres['l'], arbres['c'] = arbres.y_px / REDUCTION, arbres.x_px / REDUCTION      # en pixels réduits
        dedans = masque[arbres.l.astype(int).clip(0, img.shape[0] - 1), arbres.c.astype(int).clip(0, img.shape[1] - 1)]
        arbres = arbres[dedans]
        arbres['d_m'] = arbres.largeur_px / REDUCTION * pixel
        if partie == 'apprentissage':
            diametres_m += list(arbres.d_m)
        # recadrage sur la parcelle (avec marge), puis passage à 5 cm
        ys, xs = np.nonzero(masque); m = int(MARGE_M / pixel)
        y0, y1 = max(0, ys.min() - m), min(img.shape[0], ys.max() + m)
        x0, x1 = max(0, xs.min() - m), min(img.shape[1], xs.max() + m)
        f = pixel / PIXEL_M
        petit = Image.fromarray(img[y0:y1, x0:x1].astype(np.uint8))
        petit = petit.resize((round(petit.width * f), round(petit.height * f)), Image.Resampling.LANCZOS)
        nom = 'reforestree__' + site.replace(' RGB', '').replace(' ', '_') + '.jpg'
        enregistrer(petit, nom, sortie)
        images.append({'image': nom, 'jeu': 'reforestree', 'partie': partie, 'origine': site.replace(' RGB', ''),
                       'largeur': petit.width, 'hauteur': petit.height, 'pixel_m': PIXEL_M})
        for _, a in arbres.iterrows():
            points.append({'image': nom, 'x': (a.c - x0) * f, 'y': (a.l - y0) * f, 'diametre': a.d_m / PIXEL_M})
        print(f'  ReforesTree {site:25s} {partie:13s} {len(arbres):5d} bananiers   {petit.width} x {petit.height} px')
    return images, points, float(np.median(diametres_m)) / PIXEL_M


def coco(fichiers, filtre_image, cible_px, jeu, sortie, partie_de):
    # Lit les annotations COCO, ne garde qu'une image par nom d'origine, met le bananier médian à cible_px
    vus, lots = set(), []
    for f in fichiers:
        d = json.load(open(f)); cats = {c['id']: c['name'] for c in d['categories']}
        boites = {}
        for a in d['annotations']:
            if cats[a['category_id']].lower().startswith('banan') and min(a['bbox'][2:]) >= TAILLE_MIN_BOITE:
                boites.setdefault(a['image_id'], []).append(a['bbox'])
        for i in d['images']:
            nom = i.get('extra', {}).get('name', i['file_name'])
            if not filtre_image(nom) or nom in vus:
                continue
            vus.add(nom); lots.append((f.parent / i['file_name'], nom, boites.get(i['id'], [])))
    mediane = np.median([max(b[2:]) for _, _, bb in lots for b in bb])
    facteur = cible_px / mediane
    images, points = [], []
    for chemin, nom, bb in lots:
        partie, origine = partie_de(nom)
        img = Image.open(chemin)
        img = img.resize((round(img.width * facteur), round(img.height * facteur)), Image.Resampling.LANCZOS)
        nom_sortie = f"{jeu}__{Path(nom).stem}.jpg"
        enregistrer(img, nom_sortie, sortie)
        images.append({'image': nom_sortie, 'jeu': jeu, 'partie': partie, 'origine': origine,
                       'largeur': img.width, 'hauteur': img.height, 'pixel_m': PIXEL_M})
        for x, y, w, h in bb:
            points.append({'image': nom_sortie, 'x': (x + w / 2) * facteur, 'y': (y + h / 2) * facteur,
                           'diametre': max(w, h) * facteur})
    print(f'  {jeu:16s} {len(lots):4d} images, bananier médian {mediane:.0f} px -> x {facteur:.2f} '
          f'(pixel d\'origine estimé {PIXEL_M * facteur * 100:.1f} cm)')
    return images, points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--donnees', required=True, type=Path)
    ap.add_argument('--sortie', required=True, type=Path)
    a = ap.parse_args()
    if a.sortie.exists():
        shutil.rmtree(a.sortie)
    (a.sortie / 'images').mkdir(parents=True)

    images, points, cible = reforestree(a.donnees, a.sortie)
    print(f'  bananier médian ReforesTree (fermes d\'apprentissage) : {cible:.0f} px à 5 cm')

    cb = sorted((a.donnees / COUNT).glob('*/_annotations.coco.json'))
    i, p = coco(cb, lambda n: True, cible, 'count_banana', a.sortie,
                lambda n: (VOLS_COUNT[n[:8]], n[:8]))
    images += i; points += p

    pb = sorted((a.donnees / PLANTS).glob('*/_annotations.coco.json'))
    def partie_seg(n):
        x, y = map(int, re.match(r'segmento_(\d+)_(\d+)', n).groups())
        return zone_segmento(x, y)
    i, p = coco(pb, lambda n: n.startswith('segmento'), cible, 'segmento', a.sortie, partie_seg)
    images += i; points += p
    i, p = coco(pb, lambda n: n.startswith('corte'), cible, 'corte', a.sortie, lambda n: ('apprentissage', 'corte'))
    images += i; points += p

    images, points = pd.DataFrame(images), pd.DataFrame(points)
    points[['x', 'y', 'diametre']] = points[['x', 'y', 'diametre']].round(1)
    images.to_csv(a.sortie / 'images.csv', index=False)
    points.to_csv(a.sortie / 'points.csv', index=False)
    bilan = images.merge(points.groupby('image').size().rename('bananiers').reset_index(), how='left').fillna({'bananiers': 0})
    print('\n' + bilan.groupby(['jeu', 'partie']).agg(images=('image', 'size'), bananiers=('bananiers', 'sum')).astype(int).to_string())
    shutil.make_archive(str(a.sortie), 'zip', a.sortie)
    print(f'\nÉcrit dans {a.sortie} et {a.sortie}.zip')


if __name__ == '__main__':
    main()
