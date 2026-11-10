# Carnet de données de Field Scan

Mis à jour le 10/11/2026.

## Vue d'ensemble

### Photos de feuilles (maladies)

| Jeu | Section | v2 (installée) | v10 (prête, pas installée) |
|---|---|---|---|
| Tanzanie 2023 | 1 | apprentissage (15 691) | apprentissage (15 691) |
| Pérou | 9 | apprentissage (1 268) | apprentissage (1 268) |
| Tanzanie 2025 | 2 | validation (11 766) | validation (11 766) |
| BananaLSD, Bangladesh | 3 | test du 03/11 (893) | apprentissage (75 %) et validation (25 %) |
| cm8zf62kzh, Bangladesh | 22 | test indicatif du 09/11 (1 800) | apprentissage (3 750) et validation (1 251) |
| BLADE, Inde | 23 | test indicatif du 09/11 (1 215) | apprentissage (911) et validation (304) |
| godliver | 24 | test indicatif du 09/11 (620) | apprentissage (465) et validation (156) |
| BDR, Bangladesh | 16 | juge (408) | juge (408) |
| khadavy, Indonésie probable | 20 | mesure du 08/11 (552) | juge, photos de près (487) |
| twakihasan | 21 | test neuf du 09/11 (1 869) | jamais noté |
| Mendeley wfzpdmc5vx et 5nfjzntwd8, Bangladesh | 25, 26 | test neuf du 10/11 (1 602) | test neuf du 10/11 (1 602) |
| PSFD-Musa, Inde | 19 | noté pour information le 06/11 | aucun |
| banana-buddy | 18 | noté pour information le 06/11 | aucun |

Un juge sert à comparer des modèles. Un test ne sert qu'une fois. Un test indicatif donne une tendance, sans servir à choisir.

![Exemples de feuilles par jeu](images/feuilles_par_jeu.jpg)
*Photos tirées au hasard dans chaque jeu, après préparation. Les cadrages et les couleurs changent d'un jeu à l'autre : gros plans en 2023, feuilles plus entières avec du fond en 2025, couleurs plus ternes au Bangladesh.*

### Images drone (comptage)

| Jeu | Section | Rôle pour palm3 (installé) |
|---|---|---|
| ReforesTree | 4 | apprentissage : Flora, Leonor, Guevara, Arteaga ; validation : Manuel ; test du 10/11 : Nestor |
| Count banana plants | 8 | apprentissage : vols 0221 et 0222 ; validation : vols 0223 et 0224 |
| Plants-bana, `segmento` | 11 | apprentissage : zone de gauche ; validation : bas droite, côté ouest ; test du 10/11 : 22 morceaux |
| Plants-bana, `corte` | 11 | apprentissage |
| Photos DJI de jeune plantation | 11 | test du 10/11 (3 photos) |
| `jeunes_rangs`, `plantain`, `jeunes_lasuiza` | 13 | apprentissage |
| Detecting Plantain Crops, Banano Finca | 12 | juges |
| Palmiers à huile (Cuboid) | 15 | apprentissage (2 zones) ; juge : zone 52000_20000 |
| `banavista`, `jorges` | 14 | test neuf gardé depuis le 10/11, jamais vu par palm3 |
| Fusariose en Chine (multispectral) | 17 | aucun |

Le modèle final apprend aussi la validation. Les scores de validation des essais viennent des modèles d'essai, qui ne l'ont pas apprise.

## 1. Feuilles de bananier : Tanzanie 2023 (Harvard Dataverse)

- **Provenance** : Mduma & Leo, *Dataset of banana leaves and stem images for object detection, classification and segmentation: A case of Tanzania*, Data in Brief 49 (2023) 109322, [DOI 10.1016/j.dib.2023.109322](https://doi.org/10.1016/j.dib.2023.109322). Données sur [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LQUWXW), téléchargées le 18/10/2026. Photos prises dans les régions de Kagera, Arusha, Dar es Salaam, Kilimandjaro et Mbeya.
- **Licence** : CC0 1.0 (domaine public).
- **Résolution** : smartphones Samsung 13 Mpx, photos prises au sol. Les images fournies sont réduites : 1024 × 768 px pour la plupart (en paysage ou en portrait), avec une partie en 640 × 480 px dans FUSARIUM WILT_1 et en 640 × 288 px dans HEALTHY_1 (sondage de 200 images par dossier).
- **Date d'acquisition** : février à juillet 2021 d'après l'article. Les métadonnées des photos (EXIF) donnent une date pour 12 456 des 16 092 images : du 18/03 au 12/04/2021. Les 6 175 photos qui ont une position GPS ont toutes été prises autour d'Arusha, et toutes par le même téléphone (Samsung SM-A015F).
- **Composition** : 16 092 images après suppression des doublons : 5 628 saines, 5 767 Black Sigatoka, 4 697 Fusarium Wilt Race 1. Ce sont les chiffres de l'article, que j'ai retrouvés exactement en comptant les fichiers. La fiche descriptive fournie avec les données (17 068 images) décrit l'état avant suppression des doublons. Les dossiers « sain » ne contiennent que des feuilles ; d'après l'article, les dossiers malades contiennent des feuilles et des tiges. Les images sont nommées `Image_<numéro>.jpg`, sans indication de feuille ou de tige. Je ne garde que les feuilles, puisque le sujet porte sur la classification de feuilles : le tri se fait donc sur le contenu des images. Les archives contiennent aussi des fichiers système macOS (`__MACOSX`, `.DS_Store`), que j'ignore.
- **Utilisation** : apprentissage du modèle maladies (v1, v2, v10).
- **Copies et fuites** : ce jeu vient de la même équipe que Tanzanie 2025, avec des régions en commun (Arusha, Mbeya). Les mêmes fermes, voire les mêmes plants, peuvent apparaître dans les deux. Je n'ai trouvé aucune copie entre les deux jeux (méthode en section 2). La fiche descriptive précise que quelques doublons et des images très proches peuvent subsister à l'intérieur du jeu. Ma recherche en trouve 31 paires : surtout des rafales (même feuille, cadrage à peine différent), que je garde puisqu'elles restent dans le même jeu, et 3 copies exactes, que j'écarte (`outils/listes/doublons_tanzanie.txt`).
- **Biais et limites** :
    - les tiges n'apparaissent que dans les classes malades : sans tri, un modèle pourrait apprendre « tige = malade ». Sur 40 images tirées au hasard dans chacun des 6 dossiers, je n'ai trouvé des tiges que dans Fusarium, et aucune dans les 80 images Black Sigatoka. J'ai ensuite passé en revue à l'œil, sur des planches de vignettes, toutes les images Fusarium et Black Sigatoka : 318 tiges (317 Fusarium et une seule Black Sigatoka, `BLACK SIGATOKA_2/Image_6450.jpg`), toutes des coupes du pseudo-tronc ou du bulbe, faciles à reconnaître (forme ronde ou fibres, anneaux bruns sur fond blanc). Pour vérifier ce tri, j'ai classé automatiquement les 30 064 photos de tous les jeux de feuilles selon leur ressemblance avec 6 photos de tiges et 42 photos de feuilles (`outils/reperer_tiges.py`, réseau EfficientNet-B0 pré-entraîné), et j'ai regardé les 411 premières. Ce classement a trouvé 80 tiges que j'avais manquées, sur une planche sautée (`FUSARIUM WILT_1/Image_1949` à `Image_2032`), et aucune tige dans les autres jeux (Tanzanie 2025, Pérou, Bangladesh). J'écarte donc 398 images de tiges (`outils/listes/tiges_tanzanie_2023.txt`). Avec les 3 copies exactes, il reste 15 691 images de feuilles : 5 627 saines, 5 766 Black Sigatoka et 4 298 Fusarium. Le déséquilibre « tige = Fusarium » était donc le vrai risque de ce jeu ;
    - certains formats d'image n'existent que dans une classe (640 × 288 px chez les saines, 640 × 480 px chez Fusarium). Toutes les images sont donc ramenées à 224 × 224 px (`outils/preparer_feuilles.py`), pour que le modèle ne puisse pas s'en servir. Pour les modèles qui lisent les photos en plus grand (v6, v10), elles sont aussi préparées en 448 × 448 px ;
    - cadrage : gros plans qui remplissent presque toute l'image, avec très peu de fond. Les feuilles saines montrent souvent la nervure centrale ;
    - lumière : photos de jour. Une partie des images Black Sigatoka a une teinte violette ou bleutée, sans doute un réglage des couleurs du téléphone ;
    - chaque classe a été photographiée à des jours différents (EXIF) : saines du 18 au 25 mars, Black Sigatoka surtout du 26 au 30 mars, Fusarium du 2 au 12 avril 2021. Un modèle pourrait apprendre la lumière ou le champ d'un jour plutôt que la maladie : c'est une raison de plus pour valider et tester sur d'autres jeux ;
    - un seul pays, une seule région (Arusha) pour les photos géolocalisées.

![Tiges écartées](images/tiges_ecartees.jpg)
*8 des 398 photos de tiges écartées : coupes de pseudo-tronc ou de bulbe, rangées avec les feuilles dans les dossiers Fusarium.*

## 2. Feuilles de bananier : Tanzanie 2025 (Zenodo)

- **Provenance** : Mduma & Elinisa, *Banana Leaves Imagery Dataset*, Scientific Data (2025), [DOI 10.1038/s41597-025-04456-4](https://doi.org/10.1038/s41597-025-04456-4). Données sur [Zenodo](https://doi.org/10.5281/zenodo.7670326), téléchargées le 18/10/2026. Photos prises dans les districts de Rungwe, Mbeya, Arumeru et Arusha.
- **Licence** : CC BY 4.0.
- **Résolution** : smartphone, 4624 × 3468 px pour la grande majorité des images (sondage de 200 images par dossier), quelques images recadrées.
- **Date d'acquisition** : novembre 2022 à janvier 2023 (article) ; les métadonnées des photos vont du 22/11/2022 au 20/01/2023. Autre téléphone que pour 2023 (Samsung SM-A715F), pas de position GPS.
- **Composition** : 11 767 images de feuilles (3 339 saines, 3 496 Black Sigatoka, 4 932 Fusarium Wilt Race 1), réparties en 9 archives, 3 par classe. J'avais d'abord pris une archive par classe (`HEALTHY-1`, `BLACK SIGATOKA-1`, `FUSARIUM WILT-1`, 3 001 images). Je les utilise maintenant toutes : avec plus de photos, les seuils sont plus fiables. Après retrait d'une copie exacte (`HEALTHY-2/HEALTHY_1926.jpg`), il reste 11 766 images.
- **Utilisation** : validation du modèle maladies (v2 et v10). Les seuils sont réglés dessus. Le 08/11, 600 de ses photos ont aussi servi de juge pour comparer des lectures de la v2.
- **Copies et fuites** : voir Tanzanie 2023. La séparation entre les deux jeux est temporelle (2021 contre 2022-2023), pas géographique. Je recherche les doublons sur tous les jeux de feuilles à la fois (`outils/preparer_feuilles.py`) : empreinte d'image (au plus 4 bits de différence), puis deux confirmations en comparant les images réduites (32 × 32 px en couleur, puis 112 × 112 px en gris, corrélation d'au moins 0,9), puis vérification à l'œil. La 2e confirmation est nécessaire : sans elle, la méthode signalait 4 356 paires, presque toutes fausses, parce que deux feuilles différentes ont souvent la même texture uniforme. Avec elle, il reste 81 paires : 42 dans le Bangladesh, 31 dans Tanzanie 2023, 5 dans le Pérou, 3 dans Tanzanie 2025, et **aucune entre deux jeux différents**. Je n'écarte que les copies exactes (mêmes détails aux mêmes endroits) ; les rafales restent, puisqu'elles sont dans le même jeu. Cette méthode repère les copies d'une même photo. Elle ne garantit pas qu'une même feuille n'a pas été photographiée à nouveau sous un autre angle.
- **Biais et limites** :
    - même pays et même équipe que le jeu d'apprentissage, donc des scores de validation probablement optimistes. C'est pour cette raison que je teste sur d'autres pays ;
    - cadrage différent du jeu 2023 : feuilles souvent entières, avec du fond (sol, autres plants, ciel), et non des gros plans ;
    - comme en 2023, chaque classe a surtout été photographiée à des jours différents (EXIF, toutes les photos sont datées) : saines les 22 et 23/11/2022 (2 224 photos) puis du 16 au 18/01/2023 (1 115), Fusarium du 6 au 22/12/2022 (4 344) puis du 18 au 20/01/2023 (588), Black Sigatoka les 12 et 13/01/2023 (3 422) puis les 18 et 20/01 (74). Seul le 18/01/2023 réunit les trois classes. Le biais « classe = jour » existe donc aussi dans la validation ;
    - dans un échantillon de 24 images par classe, des tuyaux d'irrigation bleus apparaissent derrière 3 feuilles saines, et jamais dans les autres classes : un modèle pourrait associer ce décor à la classe saine ;
    - dans ce même échantillon, 2 feuilles étiquetées saines portent des taches brunes (`HEALTHY_453.jpg`, `HEALTHY_486.jpg`) : il peut y avoir des erreurs d'étiquetage.

![Exemples de doublons](images/doublons_exemples.jpg)
*À gauche, une copie exacte trouvée dans Tanzanie 2023 : une des deux images est écartée. À droite, une rafale (même feuille, cadrage à peine différent) : les deux restent, puisqu'elles sont dans le même jeu.*

## 3. Feuilles de bananier : Bangladesh, BananaLSD (Mendeley Data)

- **Provenance** : Arman et al., *BananaLSD: A banana leaf images dataset for classification of banana leaf diseases using machine learning*, Data in Brief (2023) 109608, [DOI 10.1016/j.dib.2023.109608](https://doi.org/10.1016/j.dib.2023.109608). Données sur [Mendeley Data](https://data.mendeley.com/datasets/9tb7k297ff/1), téléchargées le 18/10/2026. Photos prises dans les champs de la Bangabandhu Sheikh Mujibur Rahman Agricultural University et les champs voisins, à Gazipur.
- **Licence** : CC BY 4.0.
- **Résolution** : 3 smartphones, lumière naturelle ; images fournies en 224 × 224 px.
- **Date d'acquisition** : juin 2021.
- **Composition** : 937 photos originales (473 sigatoka, 173 pestalotiopsis, 162 cordana, 129 saines), étiquetées par un phytopathologiste. Vérifié après décompression. Le dépôt contient aussi 1 600 images augmentées, copies modifiées des originales : je ne les utilise pas.
- **Utilisation** :
    - test de la v1 et de la v2, sur un autre pays. Les classes cordana et pestalotiopsis, jamais apprises par la v2, servent à tester la réponse « je ne sais pas » ;
    - pour la v10 : 75 % de chaque classe en apprentissage et 25 % en validation (graine 0). La Cordana et la Pestalotiopsis vont dans la classe « autre ». Pour la v10, ce jeu n'est donc plus un test.
- **Copies et fuites** : aucune source commune avec les jeux tanzaniens (autre pays, autre équipe). Les images augmentées sont exclues, car elles feraient compter plusieurs fois la même photo. Le jeu contient des doublons, recherchés automatiquement (`outils/preparer_feuilles.py`) puis vérifiés à l'œil : 42 groupes d'images identiques ou presque. 38 concernent la classe saine, ce qui ramène les 129 photos saines à environ 90 photos distinctes. Une même photo apparaît aussi deux fois avec deux étiquettes différentes (`healthy/122.jpg` et `pestalotiopsis/78.jpg`). Je retire 44 images (`outils/listes/doublons_bangladesh.txt`) : une seule image est gardée par groupe, et les deux images à l'étiquette contradictoire sont écartées. Il reste 893 images : 90 saines, 472 sigatoka, 159 cordana, 172 pestalotiopsis.
- **Biais et limites** : petit jeu (937 images), déséquilibré (473 sigatoka contre 129 saines), un seul site, un seul mois. Pas de Fusarium. Le type de Sigatoka (noire ou jaune) n'est pas précisé. Cadrage proche de Tanzanie 2023 (gros plans), avec des couleurs plus ternes. Les images sont déjà en 224 px alors que les jeux tanzaniens sont en haute résolution : toutes les images des trois jeux sont ramenées à 224 × 224 px, pour que le modèle ne puisse pas reconnaître un jeu à sa résolution.

## 4. Images drone : ReforesTree, Équateur (Zenodo)

- **Provenance** : Reiersen et al., *ReforesTree: A Dataset for Estimating Tropical Forest Carbon Stock with Deep Learning and Aerial Imagery*, AAAI 2022, [arXiv:2201.11192](https://arxiv.org/abs/2201.11192). Données sur [Zenodo](https://zenodo.org/records/6813783), téléchargées le 18/10/2026. Six fermes de cacao en agroforesterie : Carlos Vera Arteaga, Carlos Vera Guevara, Flora Pluas, Leonor Aspiazu, Manuel Macias, Nestor Macias, en deux groupes distants d'environ 110 km (Flora et Nestor au sud, près de Milagro ; les quatre autres au nord, près de Mocache).
- **Licence** : CC BY 4.0.
- **Résolution** : drone Mavic 2 Pro, caméra RGB, vol à 40 m. Une orthomosaïque par ferme, de 0,97 à 1,19 cm par pixel selon la ferme (fichiers `.tfw`), erreur de positionnement de 0,25 à 0,69 m. Je réduis les images 4 fois pour travailler plus vite : 3,9 à 4,8 cm par pixel selon la ferme.
- **Date d'acquisition** : 27 et 28 octobre, probablement 2020 (rapport de vol daté du 1.11.2020).
- **Composition** : chaque arbre est annoté par une boîte autour de sa couronne, étiquetée bananier ou non (`annotations/cleaned/clean_annotations.csv`, 7 969 boîtes). Les auteurs ont annoté sur des tuiles qui se chevauchent, sans fusionner les doublons : un même arbre a souvent deux boîtes, et toutes celles d'Arteaga sont en double. Je fusionne les boîtes d'une même ferme qui se recouvrent à plus de 50 % (`outils/preparer_reforestree.py`) : il reste 5 328 arbres, dont 2 731 bananiers. Les boîtes couvrent toute l'orthomosaïque, mais certaines zones hors des parcelles n'en ont aucune (par exemple le bord droit de Manuel, image ci-dessous). Chaque ferme a une **parcelle** inventoriée sur le terrain (shapefile `Merged_final_plots`) : je ne garde que l'intérieur. Dans les parcelles :

| Ferme | Surface | Arbres | Bananiers | Part de bananiers |
|---|---|---|---|---|
| Flora Pluas | 0,40 ha | 335 | 291 | 87 % |
| Leonor Aspiazu | 0,48 ha | 331 | 220 | 66 % |
| Carlos Vera Guevara | 0,43 ha | 319 | 52 | 16 % |
| Carlos Vera Arteaga | 0,53 ha | 137 | 58 | 42 % |
| Manuel Macias | 0,41 ha | 190 | 133 | 70 % |
| Nestor Macias | 0,45 ha | 389 | 235 | 60 % |

![ReforesTree : boîtes et parcelles](images/reforestree_parcelles_boites.jpg)
*Quatre fermes en entier. En jaune la parcelle inventoriée, en cyan les bananiers annotés, en rose les autres arbres annotés (un point = le centre d'une boîte).*

- **Position des arbres** : le fichier donne pour chaque arbre sa boîte (`xmin`, `ymin`, `xmax`, `ymax` dans la tuile, `X`, `Y` pour son centre dans l'orthomosaïque) et une longitude-latitude (`lon`, `lat`). **Les colonnes `lon`, `lat` sont fausses sur 5 fermes sur 6.** Elles sont calculées à partir de `X`, `Y` avec une mauvaise taille de pixel (par exemple 0,74 cm au lieu de 1,13 cm pour Manuel, 0,52 cm au lieu de 0,97 cm pour Flora). Les points sont donc tassés vers le coin de l'image, à environ 20 m de leur boîte en moyenne (17 à 21 m selon la ferme). Seule Arteaga est juste (1,03 cm au lieu de 1,04 cm, 1,3 m d'écart). Vérifications : les tuiles sont identiques à l'orthomosaïque au même endroit ; 79 à 98 % des centres de boîtes de bananiers tombent sur de la végétation (plus verte que la médiane de l'image), contre 32 à 56 % des positions `lon`, `lat`, autant qu'au hasard. **J'utilise donc le centre des boîtes** (`X`, `Y`), jamais `lon`, `lat`.

![ReforesTree : boîtes et longitude-latitude](images/reforestree_boites_vs_lonlat.jpg)
*Une tuile brute de Manuel (environ 1 cm par pixel). En bleu les boîtes de bananiers du fichier, en rouge la longitude-latitude des mêmes bananiers : le même dessin, rétréci et décalé.*

- **Utilisation** : par ferme entière. Apprentissage : Flora, Leonor, Guevara, Arteaga. Validation : Manuel. Test : Nestor, noté le 10/11. Le comptage est noté uniquement à l'intérieur des parcelles. Chaque ferme devient une seule image à 5 cm par pixel, recadrée sur sa parcelle avec 2 m de marge, noire hors de la parcelle (`outils/preparer_comptage.py`).
- **Copies et fuites** : les tuiles d'une même ferme se chevauchent. Un découpage au hasard mettrait les mêmes arbres en apprentissage et en test, d'où la séparation par ferme. Le modèle DeepForest fourni avec le jeu a été entraîné par les auteurs sur ces mêmes fermes : je ne l'utilise pas.
- **Biais et limites** :
    - fermes de cacao où le bananier est une culture secondaire, mêlée à d'autres arbres, et non une bananeraie ;
    - les boîtes ne distinguent pas pied mère et rejets ;
    - hors des parcelles, certaines zones n'ont aucune boîte ; à l'intérieur des parcelles, je n'ai pas vérifié à l'œil que chaque bananier a sa boîte ;
    - Arteaga est très peu dense : 137 arbres dans 0,53 ha, environ 260 par hectare contre environ 840 à Flora. Sa parcelle (contour officiel des auteurs, vérifié) est une zone sèche avec beaucoup de petites plantes non annotées ;
    - Guevara n'a que 52 bananiers dans sa parcelle (16 % des arbres) ;
    - éclairage différent selon les fermes : ciel couvert à Flora, Nestor, Manuel et Leonor, soleil à Arteaga et Guevara ;
    - l'erreur de positionnement (jusqu'à 0,69 m) est proche de la tolérance de 1 m utilisée pour noter.

## 5. Images satellite : Sentinel-2 (Copernicus)

- **Provenance** : [Copernicus Browser](https://browser.dataspace.copernicus.eu/), produits de niveau L2A (corrigés de l'atmosphère), téléchargés le 20/10/2026. Deux bananeraies industrielles dans deux pays, repérées au préalable sur une imagerie satellite haute résolution :
    - **Équateur**, El Guabo (province d'El Oro) : longitude -79.8463 à -79.8272, latitude -3.1623 à -3.1497 ;
    - **Cameroun**, Njombe-Penja (région du Littoral) : longitude 9.6426 à 9.6502, latitude 4.6213 à 4.6399.
- **Licence** : accès libre et gratuit. Mention obligatoire sur les produits dérivés : « Contains modified Copernicus Sentinel data [année] ».
- **Résolution** : bandes rééchantillonnées à 10 m par pixel au téléchargement (résolution native : 10 m pour B02, B03, B04, B08 ; 20 m pour B05, B8A, B11, B12). Projection UTM 17S (EPSG:32717) pour l'Équateur, UTM 32N (EPSG:32632) pour le Cameroun.
- **Date d'acquisition** : une date par saison et par site.

| Site | Saison des pluies | Saison sèche |
| --- | --- | --- |
| Équateur | 31/03/2026 | 22/09/2026 |
| Cameroun | 20/04/2026 | 16/12/2025 |

- **Composition** : pour chaque date, 8 bandes en réflectance (B02, B03, B04, B05, B08, B8A, B11, B12) et la carte de classification de scène (SCL), qui signale nuages et ombres (téléchargée dans sa version colorée, une couleur par classe). Équateur : 2,95 km² (211 × 138 pixels). Cameroun : 1,73 km² (84 × 205 pixels). Pour chaque site, les deux dates couvrent exactement la même grille de pixels. D'après la SCL, aucun pixel de nuage ni d'ombre dans les quatre images. Ces vérifications sont reproductibles avec `outils/verifier_sentinel2.py`.
- **Utilisation** : pas d'apprentissage. Les seuils des cartes de vigueur et de stress hydrique sont réglés sur l'Équateur et évalués sur le Cameroun.
- **Copies et fuites** : aucune entre les sites, situés dans deux pays différents. Au sein d'un site, les deux dates sont comparées entre elles, pas utilisées l'une pour régler l'autre.
- **Biais et limites** :
    - un pixel couvre 10 m, donc plusieurs bananiers à la fois : aucune lecture à l'échelle du plant ;
    - zones très nuageuses : les rectangles ont été placés sur les seuls blocs dégagés aux deux dates, ce qui limite leur taille (en particulier au Cameroun) et peut les rendre peu représentatifs de toute la plantation ;
    - au Cameroun, 9,5 % des pixels sont classés « sol nu » en décembre contre 0,2 % en avril : ces parcelles étaient probablement en cours de replantation et ont été couvertes par la végétation entre les deux dates. Une différence d'indice sur ces parcelles reflète ce changement, pas un manque d'eau : elles sont exclues de la comparaison du stress hydrique ;
    - les deux sites réagissent très différemment aux saisons. En Équateur, l'écart d'indice d'humidité (NDMI) médian entre les deux dates est faible (0,43 contre 0,41), probablement grâce à l'irrigation. Au Cameroun, il est plus marqué (0,34 contre 0,30). Des seuils absolus ne seraient donc pas transposables d'un site à l'autre ;
    - les dates de saison des pluies et de saison sèche sont séparées de 4 à 6 mois : l'état de la plantation (récoltes, replantations) a pu changer entre les deux.

## 6. Parcelles : tracées à la main (QGIS)

- **Provenance** : polygones que j'ai tracés à la main dans QGIS 3.44, sur la bande rouge (B04) des images Sentinel-2 de la section 5 : image du 16/12/2025 pour le Cameroun, du 22/09/2026 pour l'Équateur. Tracé le 20/10/2026 (Cameroun) et le 21/10/2026 (Équateur).
- **Licence** : travail personnel, livré avec le projet. Les polygones sont tracés sur des données Copernicus (mention : « Contains modified Copernicus Sentinel data 2025, 2026 »).
- **Résolution** : précision d'environ un pixel, soit 10 m. Tracé dans la projection de l'image (EPSG:32632 pour le Cameroun, EPSG:32717 pour l'Équateur), puis exporté en GeoJSON en latitude-longitude (EPSG:4326).
- **Date d'acquisition** : celle des images utilisées pour le tracé (voir Provenance).
- **Composition** : deux fichiers, `parcelles_cameroun.geojson` et `parcelles_equateur.geojson`, de 15 polygones chacun, avec un champ `id` de 1 à 15. Surfaces : 2,8 à 4,4 ha par parcelle au Cameroun (50 ha au total), 1,4 à 3,0 ha en Équateur (32 ha au total).
- **Utilisation** : comme les images de la section 5 : réglage sur l'Équateur, évaluation sur le Cameroun.
- **Copies et fuites** : aucune, car aucun modèle n'apprend sur ces parcelles.
- **Biais et limites** :
    - au Cameroun, les limites des parcelles se voient bien sur l'image : les chemins qui les séparent apparaissent comme des lignes claires. J'ai tracé chaque parcelle à l'intérieur de ces chemins ;
    - en Équateur, les limites des lots de la plantation ne se voient pas à 10 m. Les seules lignes visibles sont des couloirs de câbles parallèles, en diagonale, plus fins qu'un pixel : ils apparaissent comme des lignes floues. J'ai donc placé les parcelles au centre des bandes de végétation entre deux couloirs, en coupant chaque bande en plusieurs morceaux. Ces parcelles sont un découpage de ma part, pas les vrais lots de l'exploitation ;
    - en Équateur, l'image paraissait presque noire à l'ouverture, car le toit très clair de l'hacienda écrasait le contraste. J'ai réglé l'affichage (coupe à 2 % et 98 % de l'histogramme) pour voir les couloirs ; ce réglage ne modifie pas les valeurs des pixels ;
    - j'ai évité les routes, l'hacienda et les zones hors bananeraie. Les parcelles ne contiennent donc que l'intérieur de la plantation, sans ses bords. Malgré ces précautions, les pixels en bordure de parcelle peuvent mélanger bananiers et chemins ;
    - avec un pixel de 10 m, une parcelle de 1,4 ha ne contient qu'environ 140 pixels.

## 7. Modèles pré-entraînés

**MobileNetV3-Large (maladies)**

- **Modèle** : poids `IMAGENET1K_V2` de torchvision, pré-entraînés sur ImageNet-1k (1,2 million de photos, 1 000 classes d'objets).
- **Usage** : modèle maladies v1, puis comparaison pour la v2. Je remplace la dernière couche (3 classes : saine, Black Sigatoka, Fusarium), puis j'entraîne tout le réseau sur Tanzanie 2023 et le Pérou (`entrainement/maladies_v2_kaggle.ipynb`).
- **Pourquoi** : réseau léger, conçu pour tourner sans GPU, comme le demande le sujet. Le pré-entraînement évite d'apprendre à partir de zéro avec seulement 16 959 photos.
- **Fuite** : ImageNet ne contient pas les photos de Tanzanie, du Pérou ni du Bangladesh.

**EfficientNet-B0 (maladies, comparaison ; tri des tiges)**

- **Modèle** : poids `IMAGENET1K_V1` de torchvision, pré-entraînés sur ImageNet-1k.
- **Usage** : comparé à MobileNetV3 pour le modèle maladies, avec les mêmes réglages (résultats dans [experiences.md](experiences.md)). Sert aussi, sans réentraînement, à classer les photos selon leur ressemblance à des tiges (`outils/reperer_tiges.py`).

**EfficientNet-B2 et ConvNeXt-Tiny (maladies)**

- **Modèles** : poids `IMAGENET1K_V1` de torchvision, pré-entraînés sur ImageNet-1k. EfficientNet-B2 : 7,7 millions de paramètres ; ConvNeXt-Tiny : 28 millions.
- **Usage** : comparés à MobileNetV3 et EfficientNet-B0, avec la même recette et des photos en 224 × 224 px pour tous. ConvNeXt-Tiny est le réseau de la v2 installée (moyenne de 3 modèles), de la v6 et de la v10a. Les deux restent utilisables sans GPU.
- **Fuite** : ImageNet ne contient pas les photos de Tanzanie, du Pérou ni du Bangladesh.

**ConvNeXt DINOv3 (maladies et comptage)**

- **Modèles** : ConvNeXt-Tiny aux poids DINOv3 (timm `convnext_tiny.dinov3_lvd1689m`), appris sans étiquette sur 1,7 milliard d'images (LVD-1689M). Même architecture que le ConvNeXt-Tiny ImageNet, 28 millions de paramètres. Aussi ConvNeXt-Base (timm `convnext_base.dinov3_lvd1689m`, 89 millions de paramètres).
- **Licence** : licence DINOv3, qui demande la mention « Built with DINOv3 ».
- **Usage** : réseau de la v10b (maladies). Le 10/11, ConvNeXt-Base pour l'essai maladies v11a, et ConvNeXt-Tiny comme base du détecteur de centres dans l'essai comptage v7, à la place du ResNet-18.
- **Fuite** : je ne connais pas le détail des images de LVD-1689M. Je ne peux pas vérifier qu'elles ne contiennent aucune de nos photos publiques.

**ViT-B/14 Pl@ntNet (maladies)**

- **Modèle** : ViT-B/14 DINOv2 affiné par Pl@ntNet sur PlantCLEF 2024 (7 806 espèces de plantes), timm `hf-hub:PlantNet/vit_base_patch14_reg4_dinov2_lvd142m.pc24_all`.
- **Usage** : candidat v10c (1 réseau).
- **Licence** : non vérifiée.

**DeepForest (comptage)**

- **Modèle** : détecteur de couronnes d'arbres `weecology/deepforest-tree` (bibliothèque DeepForest 2, réseau RetinaNet), pré-entraîné par ses auteurs sur des images aériennes de forêts, à environ 10 cm par pixel.
- **Usage** : point de départ d'un des deux modèles de comptage de la v2. Je le réentraîne sur mes jeux de comptage (ReforesTree, Count banana plants, Plants-bana `segmento` et `corte`) et je le compare au détecteur de centres (`entrainement/comptage_v2_kaggle.ipynb`).
- **Limite** : mes images sont à 5 cm par pixel, deux fois plus fines que celles de son pré-entraînement : les arbres lui paraissent deux fois plus gros.
- **Fuite** : ses données de pré-entraînement ne contiennent pas les fermes ReforesTree.

**ResNet-18 (comptage, détecteur de centres)**

- **Modèle** : poids `IMAGENET1K_V1` de torchvision, pré-entraînés sur ImageNet-1k (11 millions de paramètres).
- **Usage** : base du détecteur de centres, de la v2 à palm3 (installé). Je garde ses 4 étages et j'ajoute un petit décodeur qui produit une carte : pour chaque carré de 20 cm, la probabilité qu'un centre de bananier s'y trouve (`entrainement/comptage_v2_kaggle.ipynb`).
- **Fuite** : ImageNet est fait de photos prises au sol ; il ne contient pas mes images de drone.

## 8. Images drone : Count banana plants (Roboflow Universe)

- **Provenance** : projet [« Count banana plants »](https://universe.roboflow.com/count-banana-plants/count-banana-plants) publié sur Roboflow Universe par « COUNT BANANA PLANTS ». Version **v15 « DATASET FINAL SIN AUG »** (générée le 27/07/2022), téléchargée au format COCO le 28/10/2026. Pays, date et auteurs réels : non indiqués.
- **Licence** : CC BY 4.0.
- **Choix de la version** : le projet a 10 versions des mêmes images. Toutes sauf deux contiennent des copies modifiées des images d'entraînement (retournées, pivotées, plus ou moins claires : ×2 pour la v38, ×3 pour les v25 et v32) ou des images redimensionnées en 456 × 456 (v6 à v9). Ces copies seraient des quasi-doublons et créeraient des fuites. La v15 est la seule grande version sans copie et sans redimensionnement.
- **Résolution** : 400 images de 480 × 480 pixels, découpées dans 77 photos de drone brutes (le nom du fichier donne le dossier de l'appareil, le numéro de photo et la position du morceau, par exemple `100_0221_0204_0_2880`). Ce ne sont pas des orthomosaïques : pas de coordonnées, et la taille du pixel n'est pas donnée. Je l'estime à environ 3 cm : la boîte médiane d'un bananier fait 105 pixels, pour une couronne adulte d'environ 3 m. C'est une hypothèse, pas une mesure.
- **Composition** : 4 672 boîtes d'une seule classe, `Banana-plant` (j'en écarte une de 1,3 pixel de large), environ 10 par image (au plus 32), aucune image vide. Par dossier : `100_0221` 217 morceaux / 15 photos / 3 083 bananiers, `100_0222` 85 / 25 / 977, `100_0223` 93 / 35 / 591, `100_0224` 5 / 2 / 21.

![Count banana plants : annotations](images/count_banana_annotations.jpg)
*12 images tirées au hasard, boîtes des auteurs en rouge.*

- **Utilisation** : le découpage train / valid / test de Roboflow est fait au hasard, et des morceaux d'une même photo se retrouvent des deux côtés. Je ne l'utilise pas. Je sépare par dossier, que je considère comme un vol : apprentissage `100_0221` + `100_0222` (302 images, 4 060 bananiers), validation `100_0223` + `100_0224` (98 images, 611). Images remises à 5 cm par pixel (`outils/preparer_comptage.py`).
- **Copies et fuites** : les photos successives d'un même vol se recouvrent et montrent les mêmes plants ; c'est pourquoi un vol entier va d'un seul côté. Je ne sais pas si les dossiers correspondent à des parcelles différentes ou à la même parcelle à des dates différentes.
- **Biais et limites** : bananeraie presque pure, plants bien séparés, très différente des fermes mélangées de ReforesTree. Sur les 12 images regardées, presque chaque bananier a sa boîte (quelques oublis au bord des images). Photos brutes, donc vue un peu en perspective sur les bords. Taille du pixel estimée. La validation repose presque sur un seul vol (93 morceaux sur 98 viennent de `100_0223`), et aucun test ne contient ce type de bananeraie : son score est peut-être un peu flatteur.

## 9. Feuilles de bananier : Pérou (Mendeley Data)

- **Provenance** : Bazan, Rojas-Córdova, Sánchez-Flores & Santa-María (Universidad Nacional de San Martín), *Banana Leaf Image Dataset for Classification of Sigatoka, Cordana, and Healthy Leaves*, Mendeley Data, version 1, [DOI 10.17632/hzttvh774g.1](https://doi.org/10.17632/hzttvh774g.1), téléchargé le 29/10/2026. Photos prises dans les districts de Calzada et Habana (province de Moyobamba, région San Martín, Amazonie péruvienne).
- **Licence** : CC BY 4.0.
- **Résolution** : smartphones, d'après les métadonnées des photos : Xiaomi 14T Pro pour la plupart, aussi Xiaomi M2003J15SC et Samsung Galaxy A06. Surtout 3072 × 4096 px. Pas de position GPS.
- **Date d'acquisition** : non indiquée sur la page du jeu. Les métadonnées des photos vont du 08/01/2025 au 22/06/2025, et saines et Sigatoka ont été photographiées sur les mêmes mois (janvier à juin) : pas de biais « classe = jour » visible, contrairement aux jeux tanzaniens.
- **Composition** : 1 773 images : 544 saines (dossier `sanas`), 724 Sigatoka, 505 Cordana. J'utilise les saines et les Sigatoka, soit 1 268 images.
- **Utilisation** : apprentissage du modèle maladies (v2, v10), avec Tanzanie 2023. Le jeu étant petit, chaque photo du Pérou est tirée 3 fois plus souvent, ce qui fait environ 20 % de chaque tour d'apprentissage de la v2.
- **Copies et fuites** : autre pays et autre équipe que les autres jeux. Aucun doublon avec les autres jeux ; 5 paires très proches à l'intérieur du jeu, aucune copie exacte. Je ne lis jamais la Cordana de ce jeu (`outils/preparer_feuilles.py` ne la déclare pas), pour que la Cordana reste une maladie inconnue au test du Bangladesh.
- **Biais et limites** : le type de Sigatoka (noire ou jaune) n'est pas précisé : je la compte comme Black Sigatoka. Pas de Fusarium. Plusieurs téléphones.

## 10. Jeu écarté : Éthiopie (Mendeley Data)

- **Provenance** : Hailu (Addis Ababa Science and Technology University), *Banana Leaf Disease Images*, Mendeley Data, version 1, [DOI 10.17632/rjykr62kdh.1](https://doi.org/10.17632/rjykr62kdh.1). Photos prises chez quatre agriculteurs d'Arba Minch Zuria et de Mierab Abaya. Licence CC BY 4.0.
- **Composition** : 155 saines, 320 Sigatoka, 814 Xanthomonas (dossier `resized`).
- **Pourquoi je l'écarte** : sur les planches de vignettes, beaucoup de photos ne sont pas nettes. Je ne l'utilise ni pour l'apprentissage, ni pour la validation, ni pour le test.

## 11. Images drone : Plants-bana (Roboflow Universe)

- **Provenance** : projet [« Plants-bana »](https://universe.roboflow.com/work-sd4ee/plants-bana) publié sur Roboflow Universe par « Work » (juillet 2024), sans description. Pays, dates de vol, appareils et auteurs réels : non indiqués.
- **Licence** : CC BY 4.0.
- **Version récupérée** : la version 3 publiée contient 7 706 images, parce que chaque image d'entraînement y est copiée 3 fois avec des modifications et réduite en 640 × 640 px. J'ai copié le projet dans mon espace Roboflow et créé une version sans modification ni réduction : 2 840 images, téléchargées au format COCO le 02/11/2026 (`data/brut/plants_banana/`). Mais ces 2 840 images contiennent **déjà des copies** : beaucoup d'images portent le même nom d'origine et sont des versions tournées ou modifiées de la même photo, ajoutées par les auteurs eux-mêmes. En ne gardant qu'une image par nom d'origine, il reste **802 images distinctes**.
- **Composition** : ce n'est pas un seul jeu, mais **une compilation de plusieurs sources**, reconnaissables au début des noms de fichiers. J'ai compté les images et les boîtes de chacune (une seule image par nom d'origine), et regardé des exemples :

| Source (nom des fichiers) | Images distinctes (avec copies) | Bananiers annotés | Cacaoyers annotés | Ce que c'est |
|---|---|---|---|---|
| `segmento_…` | 90 (230), 640 × 640 px | **16 711** | 0 | morceaux sans recouvrement d'une même grande image drone vue du dessus : bananeraie dense, forêt autour. Bananier médian : 29 px de côté |
| `100_022…` | 401 (2 227), 256 × 256 px | 4 703 | 0 | **le jeu Count banana plants lui-même** (section 8 : 400 images, 4 672 boîtes), réduit en 256 px ; ses copies sont réparties entre les parties train, valid et test de Roboflow |
| `corte__…` | 49 (117), 256 × 256 px | 809 | 0 | gros plans de bananeraie, vue du dessus |
| vidéos (`Vi-deo…`, `WhatsApp…`, `dji_fly…`) | 66 (70) | 2 504 | 0 | images de vidéo filmées **en oblique** : les boîtes entourent les troncs, pas les couronnes |
| `DJI_…` | 12, 5 472 × 3 648 px | 276 | 762 | photos drone d'une parcelle mélangée, petites boîtes |
| `CACAU_…`, `cacao…` | 184 | 0 | 5 364 (contours) | parcelles de cacao : **seuls les cacaoyers sont annotés**, les bananiers visibles ne le sont pas |

- **Copies et fuites** : j'ai comparé les empreintes d'image (au plus 6 bits de différence) des images `segmento` et `corte` à celles de Count banana plants : aucune paire. Les images `100_022…` sont celles de Count banana plants, y compris ses vols de validation (`100_0223`, `100_0224`) : je ne les utilise pas. Les copies d'une même image doivent rester du même côté du découpage : je n'en garde qu'une. Le découpage fourni par Roboflow est fait au hasard et mélange les sources : je ne l'utilise pas.
- **Utilisation** : seulement `segmento` et `corte` (`outils/preparer_comptage.py`). `segmento` est coupé en 3 zones de la grande image (leurs coordonnées sont dans les noms) : apprentissage = zone de gauche (47 images, 12 039 bananiers), loin des deux autres ; validation = bas droite, côté ouest (17 images, 1 870) ; test = bas droite côté est et haut (26 images, 2 801). La validation (colonnes 22 à 27) et le test du bas (colonnes 28 à 35) **se touchent** : 4 morceaux de test (303 bananiers) bordent la validation. Le modèle final apprend aussi la validation : je retire donc ces 4 morceaux du test, qui garde 22 morceaux et 2 498 bananiers. `corte` sert à l'apprentissage (49 images, 809 bananiers). Les autres sources sont écartées (copies de Count banana plants, vues en oblique, bananiers non annotés).
- **Biais et limites** : sources très différentes (vue du dessus ou oblique, tailles d'image, échelles) ; images obliques et parcelles `CACAU` inutilisables pour compter les bananiers. Taille du pixel inconnue : je remets chaque source à 5 cm par pixel en supposant que son bananier médian a la même taille que dans ReforesTree. Pour `segmento`, cela donne un pixel d'origine d'environ 10 cm, pour `corte` environ 4,7 cm. C'est une hypothèse, pas une mesure. Le facteur d'échelle de `segmento` (× 2,04) a été calculé avec la taille médiane des boîtes de toutes ses zones, test compris. Calculé sur la seule zone d'apprentissage, il serait plus grand de 7,5 % (× 1,075). Un seul facteur s'applique à toute la grande image, et l'écart reste dans le zoom de 0,67 à 1,5 appris par le modèle. Pour Count banana, l'écart est nul (× 0,999).
- **Photos DJI de jeune plantation** : j'ai annoté 3 photos (`DJI_0544`, `DJI_0546`, `DJI_0548`, 5 472 × 3 648 px, environ 1,1 cm par pixel, sans position ni date), en points au centre de la rosette : 545 plants, 512 mères (plant avec son propre pseudo-tronc) et 33 rejets (pousse collée au pied d'un plant). `DJI_0544` est annotée entièrement à la main ; pour `DJI_0546` et `DJI_0548`, les points de départ viennent du modèle Roboflow `plants-bana/3`, corrigés à la main. Ensuite, les trois photos ont été vérifiées case par case dans QGIS (8 plants ajoutés). Ce sont mes seules images où mères et rejets sont séparés. Elles servent de test (jeune plantation, autre drone, jamais vue à l'apprentissage), noté le 10/11, et à mesurer combien de rejets sont comptés comme un pied de plus. Seuls les points à moins de 2 000 px du centre de chaque photo sont notés (plants vus de biais vers les bords).
    - **Biais possible** : en corrigeant une pré-annotation, on garde surtout ce que le modèle avait vu, et ce modèle a appris sur le même projet Plants-bana que le mien. Des plants difficiles pourraient donc manquer dans la vérité de 0546 et 0548, ce qui flatterait le rappel. Je compare le score de 0544 à celui des deux autres ;
    - **recouvrement** : `DJI_0544` et `DJI_0548` montrent en partie le même champ (environ la moitié de chaque photo, mêmes objets au sol retrouvés sur les deux) ; `DJI_0546` en recouvre un peu les deux. Ce n'est pas une fuite, mais certains plants comptent deux fois ;
    - les autres photos `DJI_…` ne sont pas annotées : 0537 à 0543 sont des parcelles de cacao, et 0545, 0547 et 0549 montrent le même champ que les trois photos annotées.

## 12. Images drone : Detecting Plantain Crops et Banano Finca (Roboflow Universe)

- **Sources** : [Detecting Plantain Crops](https://universe.roboflow.com/michael-armah/detecting-plantain-crops) (michael-armah, publié en mai 2023, version 3) et [Banano Finca](https://universe.roboflow.com/edison-zapata/banano-finca) (Edison Finca La Esmeralda, version 1 exportée le 22/10/2026). Téléchargés le 05/11/2026 au format COCO, dans `data/brut/roboflow_juges/`.
- **Licence** : CC BY 4.0 (page de chaque jeu, vérifiée le 05/11/2026). Aucune description de la provenance des images.
- **Contenu** :
  - Detecting Plantain Crops : 62 morceaux de 1 049 × 626 px d'une seule grande image (noms `sub_image_<ligne>_<colonne>`), jeunes plantains épars dans la brousse. La version 3 compte 148 images, dont des copies modifiées par Roboflow (miroirs, et un flou ajouté à l'intérieur des boîtes). Je ne garde que les **19 images non modifiées** (parties `valid` et `test`), soit 208 plantains annotés en boîtes ;
  - Banano Finca : 25 tuiles de jeunes bananiers dans l'herbe, vues de très bas, redimensionnées à 640 × 640 px par Roboflow. Les copies modifiées (miroirs, quarts de tour, luminosité ± 15 %) ne changent pas la taille des plants : je garde une copie par original, soit **409 plants** annotés en boîtes. La copie gardée est celle de `valid`, puis `test`, puis la première de `train`.
- **Résolution** : Detecting Plantain Crops, inconnue. Je ramène le bananier médian à la taille de ReforesTree, comme pour `segmento` : le côté le plus long de la boîte médiane (126 px) devient 59,1 px, soit un facteur de 0,469 (pixel d'origine estimé à environ 10,7 cm). C'est une hypothèse : dans `experiences.md`, je mesure l'effet de plants 0,7 à 1,4 fois plus grands que supposé. Banano Finca : les noms des tuiles portent des coordonnées en mètres (`Ban19oct_E451042_N625796`) ; les tuiles sont espacées de 11,2 m pour 640 px, soit environ **1,75 cm par pixel** si elles se touchent sans se recouvrir.
- **Pays, date, drone** : inconnus.
- **Utilisation** : juges, jamais à l'apprentissage. Ils notent les essais du comptage sur deux types de plantation que le modèle n'a jamais vus (`experiences.md`, section 1). Préparation et notation : `outils/evaluer_comptage.py`, qui lit les zips décompressés (`data/traite/roboflow_juges/`).
- **Limites** : petits, et chacun vient d'une seule scène, donc leur F1 est imprécise à quelques points près. Ils ont servi à comparer beaucoup d'essais : leur score est un peu flatté pour le modèle installé. Qualité des annotations inconnue ; mères et rejets non séparés.

## 13. Images drone : trois jeux de jeunes plantations et de plantain (Roboflow Universe)

- **Pourquoi** : sur un type de plantation jamais vu, mon modèle oubliait beaucoup de jeunes plants (`experiences.md`, section 1). Ces trois jeux lui en montrent.
- **Sources** (version 1 de chaque jeu, téléchargée le 05/11/2026 au format COCO, dans `data/brut/roboflow/`) :
  - [Banana tree detection](https://universe.roboflow.com/elliot-su-mv7fl/banana-tree-detection) : une orthomosaïque du 18/10/2022 découpée en tuiles, jeunes bananiers en doubles rangs sur sol nu. Mon nom : `jeunes_rangs` (400 images, 38 932 plants) ;
  - [plantas_platano](https://universe.roboflow.com/agromatica2025/plantas_platano) : 3 orthomosaïques de plantain, avec une petite boîte sur le cœur de chaque plant. Mon nom : `plantain` (304 images, 5 824 plants) ;
  - [platano-lasuiza](https://universe.roboflow.com/daniel-f-tovar-r-s-workspace/platano-lasuiza) : une orthomosaïque de jeunes plants sur sol nu. Mon nom : `jeunes_lasuiza` (153 images, 735 plants).
- **Licence** : CC BY 4.0 (page de chaque jeu, vérifiée le 05/11/2026). Pays, drone et provenance ne sont pas décrits.
- **Préparation** : Roboflow a parfois ajouté des copies modifiées des images (rotations, luminosité) et étiré les images en 640 × 640 px. Je garde une seule image par original. Chaque plant devient un point, au centre de sa boîte. Son diamètre est celui de la boîte, sauf pour `plantain`, dont les boîtes ne couvrent que le cœur (diamètre fixé à 2 m). Script : `outils/preparer_nouveaux_roboflow.py`.
- **Échelle** : la taille du pixel n'est pas donnée, et pour de jeunes plants, supposer qu'ils ont la taille des bananiers adultes de ReforesTree serait faux. Je ramène donc la distance entre plants voisins à **2 m** : l'IITA recommande un écartement de 2 × 2 m à 3 × 3 m ([fiche](https://propas.iita.org/en/solutions/spacing-and-stand-management-in-banana-and-plantain/153/details)), et cette distance vaut environ 2 m dans Count banana et `corte`. Contrôle : pour le vol `100_metros` du plantain, j'obtiens environ 2,4 cm par pixel, ce qu'un drone courant donne à 100 m. C'est une hypothèse, juste à environ 30 % près, ce que le modèle supporte.
- **Utilisation** : apprentissage seulement (depuis la v3). Aucune image de ces jeux ne sert à noter, donc pas de découpage par zone. `jeunes_rangs` pèse 38 932 plants à lui seul : je limite sa part à 12 % des morceaux, pour qu'il n'écrase pas les autres jeux.
- **Copies et fuites** : j'ai comparé une empreinte de chaque image à celles du test, de l'apprentissage, de la validation et des deux juges : aucune image proche. J'ai aussi cherché un recouvrement partiel (même champ sur deux images, points SIFT, méthode de la section 14) entre les 857 images de ces trois jeux et le test, la validation, les juges et les photos DJI du test : aucun.
- **Limites** : échelle supposée ; qualité des annotations inconnue (17 images de plantain sans aucun plant annoté) ; mères et rejets non séparés.

## 14. Images drone : banavista et jorges (Roboflow Universe)

- **Sources** :
  - [Banana Detection](https://universe.roboflow.com/banavista/banana-detection-4hbck) (espace « banavista », mis à jour en mai 2024). Mon nom : `banavista`. Toutes ses versions publiées sont étirées en carré (4000 × 3000 px écrasés en 640 × 640 px, ce qui déforme les plants et l'écartement entre eux) et la plupart contiennent des copies modifiées (2 102 images pour 364 photos). J'ai donc copié le projet dans mon espace Roboflow et créé une version sans redimensionnement ni modification, téléchargée le 05/11/2026 au format COCO ;
  - [banano](https://universe.roboflow.com/jorges-workspace-4uit3/banano-qpl6j) (espace « Jorges Workspace »). Mon nom : `jorges`. Aucune version publiée : même méthode (copie, version sans modification), le 05/11/2026 ;
  - zips dans `data/brut/roboflow/`, décompressés dans `data/traite/roboflow_v4/dezippe/`.
- **Licence** : CC BY 4.0 (page de chaque jeu, vérifiée le 05/11/2026). Ni auteur, ni drone, ni description de provenance.
- **Contenu** :
  - `banavista` : 363 images, dont 348 photos drone DJI de 4000 × 3000 px et 15 orthomosaïques. D'après leurs noms (« Tagum-City-Bypass-Road », « University-of-Southeastern-Philippines-Tagum-Mabini », dates de décembre 2023 à mai 2024), ce sont des bananeraies de la région de Tagum, aux Philippines : parcelles d'essai de jeunes plants en pots, et plantations adultes denses. Boîtes de deux classes : « Bananas » (52 087) et « Others » (870 : bâtiments, arbres, objets), que j'écarte ;
  - `jorges` : 71 tuiles de 1024 px d'une seule orthomosaïque de jeune bananeraie dense (noire hors de l'image), un masque par plant (5 078). Pays inconnu.
- **Préparation** (`outils/preparer_jeux_v4.py`) :
  - doublons : 50 noms de photos (`DJI_0572.JPG`…) apparaissent deux fois dans `banavista`, mais ce sont des photos différentes (autres vols, numérotation DJI recommencée : pixels et annotations différents) : je garde les deux. Les vrais doublons, cherchés par empreinte puis regardés à l'œil, sont 2 orthomosaïques : `15m.png` (identique à `clear-sky-15m.png`) et `Task-of-2024-05-10T162037110Z-orthophoto.png` (même orthomosaïque que `815-am.png`), écartées ;
  - `jorges` : 2 tuiles sans aucune annotation écartées (l'une presque noire, l'autre pleine de plants non annotés) ; 61 masques dont la boîte dépasse 300 px écartés (2,5 fois le plant médian) : ils couvrent toute une tuile ou plusieurs plants, et les plants qu'ils recouvrent ont aussi leur propre masque ;
  - un point par plant au centre de sa boîte, diamètre = racine de la surface de la boîte.
- **Échelle** : inconnue. Comme pour la section 13, la distance médiane entre plants voisins est ramenée à 2 m, image par image pour `banavista` (altitudes différentes : pixel d'origine estimé de 0,4 à 5 cm, médiane 1 cm) et une fois pour l'orthomosaïque de `jorges` (2,9 cm). Contrôle : une fois les images remises à 5 cm, la couronne médiane mesure 2,6 m dans `banavista` et 3,1 m dans `jorges`, comme dans Count banana (3,0 m) et ReforesTree (2,9 m).
- **Résultat** : `banavista` 361 images, 50 094 plants ; `jorges` 69 images, 5 017 plants (`data/traite/comptage_v4`).
- **Utilisation** : apprentissage de l'essai v4 du 05/11. Le modèle installé (palm3) ne les a jamais vus. Depuis le 10/11, ils sont gardés comme test neuf du comptage : des bananeraies denses, le type que vise le tuteur.
- **Copies et fuites** (`outils/verifier_recouvrements.py`) :
  - empreinte de chaque image (et de ses 8 orientations) comparée à toutes les images de comptage déjà utilisées, aux photos brutes de Plants-bana et de Count banana, aux juges et aux photos DJI du test : aucune image proche d'une image de test, de validation ou d'un juge. Les paires proches trouvées (entre images de feuillage uniforme ou presque noires) ont été regardées : ce ne sont pas des doublons, sauf les 2 orthomosaïques ci-dessus ;
  - recouvrement partiel (même champ sur deux images) : points SIFT appariés puis transformation géométrique cohérente, contre le test, la validation, les juges et les photos DJI de Plants-bana. Réglage vérifié avant sur des paires connues : les photos DJI du test qui montrent le même champ donnent 23 à 60 appariements cohérents, des paires sans lien au plus 6 ; seuil à 15. Contrôle de bout en bout : une photo DJI du test et un morceau recadré, tourné et agrandi d'une tuile de test de `segmento`, donnés comme « nouvelles images », sont bien retrouvés avec toutes les photos du même champ. Résultat sur les 434 images : **aucun recouvrement** ;
  - piège : les noms DJI se répètent d'un vol à l'autre. Il ne faut jamais dédoublonner par nom.
- **Limites** : échelle supposée ; qualité des annotations inconnue (boîtes parfois imbriquées dans les plantations denses) ; une partie des photos de `banavista` montre les mêmes parcelles que ses orthomosaïques ou que d'autres vols, donc certains plants comptent plusieurs fois ; vues de biais au bord des photos brutes ; mères et rejets non séparés.

## 15. Images drone : palmiers à huile, « Multiple tree detection » (Roboflow Universe)

- **Provenance** : projet « Multiple tree detection » (espace « Cuboid ») sur Roboflow Universe, copié dans mon espace et exporté le 05/11/2026 au format COCO, sans redimensionnement ni modification. Pays, drone et date : non indiqués.
- **Licence** : CC BY 4.0.
- **Composition** (`data/traite/cuboid/inventaire.csv`) : 1 790 images. 500 tuiles de 1024 px ne contiennent que des palmiers à huile (34 890 palmiers annotés). Elles viennent de 3 zones de 4 000 × 4 000 px (nom : `<zone x>_<zone y>_<x>_<y>`) : 176 tuiles pour la zone 44000_4000, 144 pour 44000_16000, 180 pour 52000_20000. Les autres images : 596 de cocotiers, 4 avec palmiers et cocotiers, et 690 avec des bananiers.
- **Résolution** : les palmiers à huile sont plantés à environ 9 m les uns des autres. La distance médiane d'un palmier annoté à son plus proche voisin est de 102,6 px, d'où un pixel d'environ 8,8 cm. Les tuiles sont remises à 5 cm (`outils/preparer_jeux_v5.py`).
- **Utilisation** : les tuiles de palmiers seuls des zones 44000_4000 et 44000_16000 (320 tuiles) servent d'exemples « rien à trouver » à l'apprentissage, sans aucun point : 10 % des morceaux dans l'essai v5, 3 % dans palm3 (installé). La zone 52000_20000, à au moins 350 m des deux autres, est gardée de côté : 40 de ses tuiles, tirées au hasard (graine 0), soit 32,3 ha, mesurent les faux bananiers par hectare (`outils/evaluer_comptage.py`, option `--palmiers`).
- **Copies et fuites** : les tuiles se recouvrent à l'intérieur d'une zone, d'où la séparation par zone. Les 690 images de bananiers sont des copies de Count banana (`100_0221` à `100_0224`, validation comprise) et de `corte`, d'après leurs noms : je ne les utilise pas. Les images de cocotiers sont écartées : ce sont de petites fermes mélangées où des plants non annotés ressemblent à de jeunes bananiers.
- **Limites** : échelle supposée à partir de l'écartement ; une seule plantation de palmiers ; la zone gardée vient de la même plantation que les zones apprises.

## 16. Feuilles de bananier : BDR, Bangladesh (Mendeley Data)

- **Provenance** : *Banana Disease Recognition Dataset* (BDR), [Mendeley Data 79w2n6b4kf, version 1](https://data.mendeley.com/datasets/79w2n6b4kf/1), téléchargé le 05/11/2026 (archive `Original Images.zip`). Bangladesh, lieu non indiqué. Photos prises au champ.
- **Licence** : CC BY 4.0.
- **Résolution** : 512 × 512 px.
- **Date d'acquisition** : non indiquée.
- **Composition** : 408 photos originales : 86 saines, 67 Black Sigatoka, 23 Yellow Sigatoka, 41 Panama (fusariose), 86 dégâts d'insectes, 55 Moko, 50 virus de la mosaïque des bractées. Le dépôt contient aussi 2 856 images augmentées, que je n'utilise pas.
- **Utilisation** : jamais appris. Juge des maladies : comparaison des essais v3 et v4 avec la v2 (06/11), des lectures de la v2 (08/11), puis des candidats v10 (10/11). La Black Sigatoka doit être nommée, la fusariose aussi ; toutes les autres maladies et les insectes doivent déclencher une alerte. Pour la v10, les photos sont réduites à 896 px de petit côté (`modeles/maladies_v10/preparer_juges.py`).
- **Copies et fuites** : aucune copie avec nos autres jeux (audit du 08/11, 8 orientations). Mais 123 de ses photos se retrouvent dans cm8zf62kzh, avec les mêmes noms de fichiers (section 22) : je les retire de cm8zf62kzh.
- **Biais et limites** : petit jeu (de 23 à 86 photos par classe) ; images carrées en 512 px ; pas de date ni de lieu ; étiquettes sans expert annoncé. Il a servi à comparer plusieurs modèles, donc son score est un peu flatté.

## 17. Images drone multispectrales : fusariose en Chine (Science Data Bank)

- **Provenance** : *A dataset of UAV multispectral images for banana Fusarium wilt survey*, [Science Data Bank](https://www.scidb.cn/en/detail?dataSetId=8d77781a1d754db4b8842708a69c2c22), Long'an (Guangxi, Chine). Trois archives téléchargées le 05/11/2026 : réflectance multispectrale du drone, relevés de fusariose au sol, contour de la bananeraie.
- **Licence** : non indiquée sur ce que j'ai lu.
- **Résolution** : orthomosaïque de 5 bandes de réflectance, 8 cm par pixel, 14 420 × 7 885 px, 20,5 ha valides.
- **Composition** : une bananeraie dense et industrielle (canopée fermée). Aucun plant n'est annoté. Les 80 points du relevé de fusariose ne sont pas des comptages.
- **Utilisation** : aucune. Ce n'est pas une image RGB : il faudrait fabriquer des couleurs à partir des bandes, et l'ordre des bandes n'est pas documenté.

## 18. Feuilles de bananier : banana-buddy (Roboflow Universe)

- **Provenance** : projet [« Banana Disease Identification »](https://universe.roboflow.com/banana-buddy-research-sgury/banana-disease-identification) (espace banana-buddy-research), version 3 du 05/11/2023, téléchargée le 06/11/2026 au format dossiers. Pays et auteurs non indiqués.
- **Licence** : CC BY 4.0 (fichier README du jeu).
- **Composition** : 5 360 images en 6 dossiers : Healthy 855, Healthy Leaf 1 004, Leafspot 751, Leafspot Leaf 999, Panama 1 003, Panama Leaf 748.
- **Utilisation** : le 06/11, apprentissage de l'essai maladies v4 (562 photos saines de ce jeu) et notes pour information de la v2 sur ses photos à taches. Ce jeu ne sert plus depuis l'audit du 08/11.
- **Copies et fuites** : le premier contrôle du 06/11 ne testait pas les rotations et n'avait trouvé aucune copie avec nos jeux. L'audit du 08/11, sur 8 orientations, montre que les dossiers « … Leaf » sont surtout des **copies tournées de Tanzanie 2023** : 705 des 1 004 photos de Healthy Leaf, 817 des 999 de Leafspot Leaf, 624 des 748 de Panama Leaf. Les notes du 06/11 sur ces photos étaient donc faussées. Une photo de ce jeu est aussi copiée dans le test neuf du 10/11 (sections 25 et 26) : elle en est retirée.
- **Biais et limites** : les photos « Panama » montrent des plantes de loin ; les « taches » mélangent plusieurs causes (Cordana, bords brûlés).

## 19. Feuilles de bananier : PSFD-Musa, Inde (Mendeley Data)

- **Provenance** : PSFD-Musa, [Mendeley Data 4wyymrcpyz, version 1](https://data.mendeley.com/datasets/4wyymrcpyz/1), article Data in Brief (DOI 10.1016/j.dib.2022.108427). Assam (Inde). Trois archives téléchargées le 06/11/2026 : maladies, carence, variétés.
- **Licence** : CC BY 4.0 d'après ma recherche du 06/11. Le 10/11, la page Mendeley ne répond plus : je n'ai pas pu la revérifier.
- **Résolution** : 256 × 256 px (vérifié sur les fichiers).
- **Composition** des photos de feuilles que j'ai préparées : 4 311, dont 1 605 saines de la variété Malbhog, 336 saines de la variété Jahaji, 1 530 carences en potassium, 474 Black Sigatoka, 264 Yellow Sigatoka et 102 fusarioses (Panama).
- **Utilisation** : le 06/11, la v2 y a été notée pour information. Ce jeu ne sert plus depuis l'audit du 08/11.
- **Copies et fuites** : aucune copie avec nos jeux. Mais **3 860 photos sur 4 311** font partie de paires de copies internes, souvent avec des étiquettes contradictoires (par exemple la même photo en carence en potassium et en Black Sigatoka). Le jeu est inutilisable pour noter un modèle.
- **Biais et limites** : basse résolution ; beaucoup de copies d'une même photo.

## 20. Feuilles de bananier : khadavy, Indonésie probable (Roboflow Universe)

- **Provenance** : projet « Klasifikasi Daun Pisang » (espace khadavy) sur Roboflow Universe, exporté le 08/11/2026 sans augmentation, au format dossiers. Pays probable : Indonésie (noms de classes en indonésien).
- **Licence** : CC BY 4.0.
- **Composition** : 755 images, en 3 classes : `Bercak-Hitam` (taches noires) 252, `Sehat` (saine) 251, `Bercak-Kuning` (taches jaunes) 252. Ces traductions sont une interprétation.
- **Préparation** (`modeles/maladies_khadavy/`) : une seule photo par groupe de copies internes, et les photos présentes avec deux étiquettes sont écartées. Il reste 552 photos : 251 taches noires, 160 saines, 141 taches jaunes. Beaucoup sont des **rafales** (la même feuille photographiée plusieurs fois d'affilée) : je les regroupe en 276 séries, d'après le nom (heure ou numéro qui se suit) et la ressemblance (corrélation d'au moins 0,8 en 32 × 32 couleur). J'ai aussi trié à l'œil, avant toute mesure, les photos « de près » (une feuille occupe la plus grande partie de la photo) et « de loin » (`tri_distance.csv`) : 487 photos de près, en 238 séries.
- **Utilisation** : jamais appris. Mesure de la v2 le 08/11 (552 photos), notes des finalistes maladies le 09/11, puis juge des candidats v10 (photos de près). Taches noires = Black Sigatoka à nommer ; taches jaunes = maladie jamais apprise, qui doit déclencher une alerte.
- **Copies et fuites** : aucune copie avec nos jeux (Tanzanie 2023 et 2025, Pérou, BananaLSD, BDR, puis audit du 08/11 sur 8 orientations).
- **Biais et limites** : beaucoup de feuilles à quelques petites taches, des symptômes légers ; les rafales font peser certaines scènes plus que d'autres ; étiquettes sans expert annoncé. C'est ce jeu qui m'a donné l'idée de lire les photos en 384 px : il n'est donc pas neutre pour juger cette idée.

## 21. Feuilles de bananier : twakihasan (Kaggle)

- **Provenance** : [BananaLeafDisease](https://www.kaggle.com/datasets/twakihasan/bananaleafdisease), jeu Kaggle de l'utilisateur twakihasan (il en a publié 3 versions, je prends `bananaleafdisease`), téléchargé le 09/11/2026. Pays et auteur inconnus.
- **Licence** : MIT.
- **Résolution** : 8 160 × 6 120 px (quelques-unes en 4 000 × 3 000), photos de téléphone au champ.
- **Date d'acquisition** : presque toutes du 05/12/2025, d'après les noms de fichiers.
- **Composition** : 2 638 photos, en 3 classes : `healthy`, `panama` (fusariose), `sigatoka`. Beaucoup de Sigatoka à quelques petites taches ; les fusarioses sont des feuilles jaunies et sèches.
- **Préparation** (`modeles/maladies_twakihasan/`) : les parties `train`, `val` et `test` du jeu partagent des copies (489 groupes à cheval sur deux parties) : je les ignore. Je garde une photo par groupe de copies (1 883 photos distinctes) et j'écarte les 14 groupes qui mélangent `panama` et `sigatoka`. Il reste **1 869 photos** : 1 183 Sigatoka, 407 saines, 279 fusarioses. Les rafales (photos de la même classe à 60 s au plus d'intervalle, et très ressemblantes) forment des séries : 1 768 séries.
- **Utilisation** : test neuf, noté une seule fois le 09/11 avec les finalistes maladies de ce jour, dont la v2 et la v6. La v10 n'y est pas notée.
- **Copies et fuites** : aucune copie avec nos jeux (Tanzanie 2023 et 2025, Pérou, BananaLSD, BDR, PSFD, banana-buddy, khadavy ; 8 orientations).
- **Biais et limites** : presque tout est pris le même jour, sans doute au même endroit ; étiquettes sans expert annoncé ; une feuille jaunie et sèche peut aussi venir de la vieillesse ; la méthode des copies ne voit pas une autre photo de la même feuille.

## 22. Feuilles de bananier : cm8zf62kzh, Bangladesh 2026 (Mendeley Data)

- **Provenance** : *Banana Leaf Disease Image Dataset 2026*, [Mendeley Data cm8zf62kzh](https://data.mendeley.com/datasets/cm8zf62kzh), téléchargé le 09/11/2026. Rangunia et Hathazari (Chittagong, Bangladesh), smartphone Realme 9 5G Speed Edition, feuilles et plantes au sol, à plusieurs angles et distances.
- **Licence** : CC BY 4.0.
- **Résolution** : 224 × 224 px.
- **Composition** : 6 005 photos en 6 dossiers : feuilles saines, Black Sigatoka, Yellow Sigatoka, fusariose (Panama), dégâts d'insectes, « Moko or dead ».
- **Utilisation** :
    - test indicatif de la v2 le 09/11 (1 800 photos, 300 par classe) ;
    - pour la v10 : 5 001 photos (640 saines, 875 Black Sigatoka, 699 fusarioses, 2 787 « autre » : Sigatoka jaune, insectes, Moko ou feuille morte), 75 % en apprentissage et 25 % en validation, par groupes de copies (`modeles/maladies_v10/preparer_v10.py`).
- **Copies et fuites** : **123 copies de BDR**, avec les mêmes noms de fichiers (35 Black Sigatoka, 26 Panama, 24 saines, 23 insectes, 15 Yellow Sigatoka) : retirées, pour que BDR reste un juge. 1 237 paires de copies internes (1 297 photos), et 5 paires à deux étiquettes (Sigatoka jaune et fusariose), retirées. La page de recherche signale un dépôt voisin de la même autrice, Mendeley 6yy9522rdy, probablement les mêmes photos : je ne l'ai pas téléchargé.
- **Biais et limites** : petites images ; une partie des « Black Sigatoka » ressemble à des feuilles mortes ; « Moko or dead » montre surtout des plantes mortes au sol ; aucun expert n'est cité pour les étiquettes ; même pays que BananaLSD et BDR.

## 23. Feuilles de bananier : BLADE, Inde (Mendeley Data)

- **Provenance** : *BLADE: Banana Leaf Agricultural Disease Evaluation*, [Mendeley Data w7ytz3twjv](https://data.mendeley.com/datasets/w7ytz3twjv), téléchargé le 09/11/2026. Frontière entre l'Uttar Pradesh et le Bihar (Inde). Le dépôt donne aussi, pour chaque image, des mesures de gravité (part de feuille touchée, nombre et surface des taches).
- **Licence** : CC BY 4.0.
- **Résolution** : morceaux de 512 × 512 px.
- **Composition** : 1 218 morceaux : 430 saines, 249 Sigatoka, 203 Cordana, 336 Pestalotiopsis.
- **Utilisation** : test indicatif de la v2 le 09/11 (1 215 photos). Pour la v10 : apprentissage (911) et validation (304), la Cordana et la Pestalotiopsis dans la classe « autre ».
- **Copies et fuites** : aucune copie avec nos jeux, ni avec les images originales ou augmentées de BananaLSD (vérifié aussi à l'œil d'après son fichier de provenance). 3 paires de copies internes.
- **Biais et limites** : morceaux très zoomés et souvent flous, qualité faible.

## 24. Feuilles de bananier : godliver (GitHub)

- **Provenance** : dépôt GitHub [godliver/source-code-BBW-BBS](https://github.com/godliver/source-code-BBW-BBS) (code d'un diagnostic du flétrissement bactérien et de la Black Sigatoka), téléchargé le 09/11/2026. Pays non indiqué dans le dépôt.
- **Licence** : aucune licence indiquée. Je l'utilise seulement en interne, pour ce projet.
- **Résolution** : environ 450 × 300 px.
- **Composition** : 624 images de feuilles **découpées sur fond blanc** : saines, flétrissement bactérien (BBW), Black Sigatoka (BBS). J'en garde 621 : 358 saines, 220 flétrissements bactériens, 43 Black Sigatoka.
- **Utilisation** : test indicatif de la v2 le 09/11 (620 photos). Pour la v10 : apprentissage (465) et validation (156), le flétrissement bactérien dans la classe « autre ».
- **Copies et fuites** : aucune copie avec nos jeux. 3 paires de copies internes, dont 1 à deux étiquettes.
- **Biais et limites** : aucune de nos autres photos n'est sur fond blanc : un modèle pourrait reconnaître ce jeu au fond. Très peu de Sigatoka.

## 25. Feuilles de bananier : Mendeley wfzpdmc5vx, Bangladesh

- **Provenance** : *Banana_Leaf_Disease_Dataset_Bangladesh*, [Mendeley Data wfzpdmc5vx, version 1](https://data.mendeley.com/datasets/wfzpdmc5vx/1), téléchargé le 10/11/2026. Noakhali, Rangamati, Bandarban et Chittagong (Bangladesh), smartphone Xiaomi M2006C3MG, lumière naturelle.
- **Licence** : CC BY 4.0.
- **Résolution** : petites photos, surtout 320 × 427 px (ou 427 × 320).
- **Composition** : 424 photos : 107 saines (`fresh`), 100 Sigatoka, 100 Cordana, 117 Pestalotiopsis.
- **Utilisation** : moitié du test neuf du 10/11 (420 photos retenues, section 26). Saine = « saine » ; Sigatoka = alerte, et « Sigatoka nommée » si le modèle dit black_sigatoka ; Cordana et Pestalotiopsis = classe « autre », toute alerte est juste.
- **Copies et fuites** : voir section 26.
- **Biais et limites** : mêmes classes et même pays que BananaLSD ; petites photos.

## 26. Feuilles de bananier : Mendeley 5nfjzntwd8, dossier RAW (Bangladesh)

- **Provenance** : *Banana and Banana Leaf Dataset for Classification and Disease Detection*, [Mendeley Data 5nfjzntwd8](https://data.mendeley.com/datasets/5nfjzntwd8), version 3, téléchargé le 10/11/2026 (archive `raw.zip` du dossier RAW). Bangladesh.
- **Licence** : CC BY 4.0.
- **Résolution** : surtout 750 × 1000 px.
- **Composition** : je ne prends que les dossiers de feuilles brutes : 1 057 saines, 126 Black and Yellow Sigatoka, 54 fusarioses (Panama Wilt), 53 dégâts d'insectes, 17 dégâts de skipper. Les dossiers de fruits et les images augmentées ne sont pas utilisés.
- **Utilisation** : autre moitié du test neuf du 10/11, noté une seule fois. Les deux jeux ensemble font **1 602 photos** : 1 067 saines, 207 Sigatoka, 45 fusarioses et 283 « autre » (Cordana, Pestalotiopsis, insectes, skipper). Liste figée avant toute note (`modeles/maladies_v10/test_neuf/figer_liste.py`).
- **Copies et fuites** : avant de noter, j'ai cherché les copies contre les 58 590 images de tous nos jeux (apprentissage, validation, juges, tests), avec 8 orientations. Une seule photo du test était copiée d'un de nos jeux (banana-buddy) : je l'ai retirée. J'ai aussi trouvé 165 paires de copies à l'intérieur du test : une seule photo est gardée par groupe, et les groupes à deux classes sont retirés.
- **Biais et limites** : « Black and Yellow Sigatoka » mélange deux maladies ; étiquettes des auteurs, sans expert annoncé ; ce test vient du Bangladesh, comme deux sources d'apprentissage de la v10 (BananaLSD, cm8zf62kzh) et le juge BDR : il favorise la v10.

## Recherche des copies entre tous les jeux de feuilles

Le 08/11, j'ai cherché les copies entre tous nos jeux de feuilles à la fois, en testant **les 8 orientations** de chaque photo (quarts de tour et miroirs) : empreinte à 4 bits au plus, puis corrélation d'au moins 0,9 en 32 × 32 couleur et en 112 × 112 gris (`data/traite/audit_feuilles/`). Le 09/11, même méthode pour les nouveaux jeux (`data/traite/audit_nouveaux/`).

| Jeu | Copies dans nos autres jeux | Copies internes |
|---|---|---|
| banana-buddy | dossiers « … Leaf » : 705 / 1 004, 817 / 999, 624 / 748 photos copiées de Tanzanie 2023 | 1 331 paires |
| PSFD-Musa | aucune | 3 860 photos sur 4 311, souvent à étiquettes contradictoires |
| BDR | aucune | aucune |
| khadavy | aucune | rafales et copies, regroupées (section 20) |
| twakihasan | aucune | 1 113 paires (1 290 photos), 14 groupes à deux étiquettes |
| cm8zf62kzh | 123 copies de BDR | 1 237 paires (1 297 photos), 5 paires à deux étiquettes |
| BLADE | aucune | 3 paires |
| godliver | aucune | 3 paires, dont 1 à deux étiquettes |

La méthode trouve les copies d'une même photo, même tournée, retournée, réduite ou recompressée. Elle ne trouve pas une autre photo de la même feuille, ni un petit morceau découpé dans une photo.

## Recherche web de jeux de maladies (08/11)

J'ai cherché des jeux publics de photos de maladies du bananier : dépôts scientifiques, Kaggle, Roboflow, Hugging Face, GitHub, articles en plusieurs langues. J'ai trouvé **198 jeux** : 53 déjà connus, 107 sans intérêt et 38 candidats vérifiés. Liste complète : `C:\Field_scan\fiches\jeux_maladies_recherche_2026-11-08.csv`.

- Presque aucun jeu public n'est à la fois propre, téléchargeable, pris de près, dans un pays nouveau, avec de la fusariose.
- Les plus précieux sont **sur demande** : la bibliothèque CGIAR / Bioversity (Afrique, Inde), PlantVillage, les jeux DOST-ASTI des Philippines, et un jeu de Sigatoka noire par stades en Colombie (sans lien publié).
- J'ai téléchargé ensuite twakihasan, godliver, cm8zf62kzh et BLADE (09/11), puis wfzpdmc5vx et 5nfjzntwd8 (10/11). PlantSeg (Hugging Face, licence CC BY-NC-ND) n'est pas téléchargé.
- Écartés : jeux de carences sur fond noir artificiel, compilations sans source, jeux sans étiquette de maladie, abaca (autre plante), et plusieurs copies de jeux déjà connus. Le jeu Mendeley wzshm7ky49 (Bangladesh, 497 Panama) a été retiré par son auteur.

## Écart entre photos de feuilles et images aériennes

Toutes les photos de feuilles sont prises au sol, de près, au smartphone. Le modèle maladies ne sait donc pas lire une image drone ou satellite. En conditions réelles, il sert d'outil de terrain : les cartes aériennes indiquent où envoyer une équipe, qui photographie les feuilles pour confirmer la maladie. Détecter les maladies directement depuis le drone demanderait des images aériennes de feuilles malades annotées, que je n'ai pas trouvées en données publiques.
