"""Vérifie qu'aucune image ne se retrouve deux fois dans les données du comptage, ni entre deux parties
du découpage (apprentissage / validation / test).

Jeux vérifiés : ReforesTree (tuiles préparées), Count banana plants (v15), Plants-bana (copie sans modification).
Trois contrôles :
  1. copies déclarées : dans Plants-bana, plusieurs images portant le même nom d'origine ;
  2. même photo d'origine : les images Plants-bana `100_022…` sont-elles des photos de Count banana plants,
     et de quel vol (apprentissage ou validation) ;
  3. images identiques ou presque, quel que soit leur nom : empreinte d'image (dHash 64 bits) calculée sur
     l'image et ses 7 versions tournées ou retournées, puis confirmation en comparant les images réduites
     (corrélation, comme pour les feuilles). Les images presque noires (hors parcelle) sont ignorées.

Usage :
    python outils/verifier_doublons_comptage.py --donnees C:/Field_scan/data --sortie C:/Field_scan/exploration/doublons_comptage
"""
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

SEUIL_BITS = 6            # empreintes à au plus 6 bits d'écart (sur 64) : candidates
SEUIL_CORRELATION = 0.9   # confirmation sur les images réduites en 32 × 32 puis 112 × 112

COUNT = 'brut/count_banana_plants/Count banana plants.v15-dataset-final-sin-aug.coco'
PLANTS = 'brut/plants_banana/Plants-bana.v1i.coco'
FERMES = {'Flora_Pluas': 'apprentissage', 'Leonor_Aspiazu': 'apprentissage', 'Carlos_Vera_Guevara': 'apprentissage',
          'Carlos_Vera_Arteaga': 'apprentissage', 'Manuel_Macias': 'validation', 'Nestor_Macias': 'test'}
VOLS_COUNT = {'100_0221': 'apprentissage', '100_0222': 'apprentissage', '100_0223': 'validation', '100_0224': 'validation'}


def source_plants(nom):
    for g, motif in [('count_banana', r'^100_022'), ('cacau', r'^(CACAU|cacao)'), ('segmento', r'^segmento'),
                     ('corte', r'^corte'), ('video', r'(?i)^(vi-deo|whatsapp|dji_fly)'), ('dji', r'^DJI_')]:
        if re.match(motif, nom):
            return g
    return 'autre'


def inventaire(racine):
    lignes = []
    for p in sorted((racine / 'traite/reforestree/tuiles/images').glob('*.png')):
        ferme = '_'.join(p.stem.split('_')[:-2])
        lignes.append({'jeu': 'reforestree', 'image': p, 'origine': ferme, 'partie': FERMES.get(ferme, '?')})
    for p in sorted((racine / COUNT).glob('*/*.jpg')):
        photo = '_'.join(p.name.split('_')[:3])
        lignes.append({'jeu': 'count_banana', 'image': p, 'origine': photo, 'partie': VOLS_COUNT[photo[:8]]})
    for partie in ['train', 'valid', 'test']:
        d = json.load(open(racine / PLANTS / partie / '_annotations.coco.json'))
        for i in d['images']:
            nom = i['extra']['name']
            lignes.append({'jeu': 'plants_bana', 'image': racine / PLANTS / partie / i['file_name'],
                           'origine': nom, 'source': source_plants(nom), 'partie': 'non découpé'})
    return pd.DataFrame(lignes)


def charger(p):
    im = Image.open(p).convert('L')
    noir = (np.asarray(im.resize((64, 64))) < 10).mean()
    return im, noir


def dhash(im):
    a = np.asarray(im.resize((9, 8)), dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).ravel())


def vecteur(im, cote):
    v = np.asarray(im.resize((cote, cote)), dtype=np.float32).ravel(); v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


VARIANTES = [lambda im: im, lambda im: im.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
             lambda im: im.transpose(Image.Transpose.FLIP_TOP_BOTTOM), lambda im: im.transpose(Image.Transpose.ROTATE_90),
             lambda im: im.transpose(Image.Transpose.ROTATE_180), lambda im: im.transpose(Image.Transpose.ROTATE_270),
             lambda im: im.transpose(Image.Transpose.TRANSPOSE), lambda im: im.transpose(Image.Transpose.TRANSVERSE)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--donnees', required=True, type=Path)
    ap.add_argument('--sortie', required=True, type=Path)
    a = ap.parse_args()
    a.sortie.mkdir(parents=True, exist_ok=True)

    df = inventaire(a.donnees)
    print('Images :', df.groupby('jeu').size().to_dict())

    # 1. copies déclarées dans Plants-bana
    pb = df[df.jeu == 'plants_bana']
    copies = pb.groupby('origine').size()
    print(f"\n1. Plants-bana : {len(pb)} images, {copies.size} noms d'origine distincts, "
          f"{(copies > 1).sum()} noms présents plusieurs fois ({int((copies - 1).sum())} copies)")
    print(pb.drop_duplicates('origine').groupby('source').size().rename('images distinctes').to_string())

    # 2. photos de Count banana plants présentes dans Plants-bana
    cb_pb = pb[pb.source == 'count_banana'].drop_duplicates('origine')
    photos = cb_pb.origine.str.split('_').str[:3].str.join('_')
    vols = photos.str[:8].map(VOLS_COUNT).value_counts().to_dict()
    print(f"\n2. Plants-bana contient {len(cb_pb)} images de Count banana plants (vols : {vols})")

    # 3. images identiques ou presque, toutes sources confondues
    hashes, base, noir, petits = [], [], [], {}
    for k, p in enumerate(df.image):
        im, n = charger(p); noir.append(n)
        hs = [dhash(f(im)) for f in VARIANTES]
        base.append(hs[0]); hashes.append(np.stack(hs))
        petits[k] = im.resize((112, 112))
    df['part_noire'] = noir
    garde = np.array(noir) < 0.5
    print(f"\n3. Empreintes : {garde.sum()} images comparées ({(~garde).sum()} presque noires ignorées)")
    B = np.unpackbits(np.stack(base), axis=1).astype(bool)          # (n, 64)
    idx = np.flatnonzero(garde)
    candidats = set()
    for i in idx:
        V = np.unpackbits(hashes[i], axis=1).astype(bool)             # (8, 64)
        d = (V[:, None, :] != B[None, idx, :]).sum(2).min(0)          # meilleure variante
        for j in idx[(d <= SEUIL_BITS)]:
            if j != i:
                candidats.add((min(i, j), max(i, j)))
    paires = []
    for i, j in candidats:
        ri, rj = df.iloc[i], df.iloc[j]
        if ri.jeu == rj.jeu == 'plants_bana' and ri.origine == rj.origine:
            continue                                                  # copies déjà comptées au contrôle 1
        meilleure = 0
        for f in VARIANTES:
            c32 = float(vecteur(f(petits[i]), 32) @ vecteur(petits[j], 32))
            c112 = float(vecteur(f(petits[i]), 112) @ vecteur(petits[j], 112))
            meilleure = max(meilleure, min(c32, c112))
        if meilleure >= SEUIL_CORRELATION:
            paires.append({'image_1': str(ri.image), 'jeu_1': ri.jeu, 'partie_1': ri.partie, 'origine_1': ri.origine,
                           'image_2': str(rj.image), 'jeu_2': rj.jeu, 'partie_2': rj.partie, 'origine_2': rj.origine,
                           'correlation': round(meilleure, 3)})
    paires = pd.DataFrame(paires, columns=['image_1', 'jeu_1', 'partie_1', 'origine_1', 'image_2', 'jeu_2', 'partie_2',
                                           'origine_2', 'correlation'])
    paires.to_csv(a.sortie / 'paires_confirmees.csv', index=False)
    print(f"   {len(candidats)} paires candidates, {len(paires)} confirmées")
    if len(paires):
        print(paires.groupby(['jeu_1', 'jeu_2', 'partie_1', 'partie_2']).size().rename('paires').to_string())
        entre_parties = paires[(paires.partie_1 != paires.partie_2) & ~paires.partie_1.eq('non découpé') & ~paires.partie_2.eq('non découpé')]
        print(f"   dont entre deux parties différentes du découpage : {len(entre_parties)}")
    df.drop(columns='image').to_csv(a.sortie / 'inventaire.csv', index=False)
    print('\nÉcrit dans', a.sortie)


if __name__ == '__main__':
    main()
