# Carnet de données — Field Scan

Mis à jour le 28/10/2026.

## 1. Feuilles de bananier — Tanzanie 2023 (Harvard Dataverse)

- **Provenance** : Mduma & Leo, *Dataset of banana leaves and stem images for object detection, classification and segmentation: A case of Tanzania*, Data in Brief 49 (2023) 109322, [DOI 10.1016/j.dib.2023.109322](https://doi.org/10.1016/j.dib.2023.109322). Données sur [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LQUWXW), téléchargées le 18/10/2026. Photos prises dans les régions de Kagera, Arusha, Dar es Salaam, Kilimandjaro et Mbeya.
- **Licence** : CC0 1.0 (domaine public).
- **Résolution** : smartphones Samsung 13 Mpx, photos prises au sol. Les images fournies sont réduites : 1024 × 768 px pour la plupart (en paysage ou en portrait), avec une partie en 640 × 480 px dans FUSARIUM WILT_1 et en 640 × 288 px dans HEALTHY_1 (sondage de 200 images par dossier).
- **Date d'acquisition** : février à juillet 2021 d'après l'article. Les métadonnées des photos (EXIF) donnent une date pour 12 456 des 16 092 images : du 18/03 au 12/04/2021. Les 6 175 photos qui ont une position GPS ont toutes été prises autour d'Arusha, et toutes par le même téléphone (Samsung SM-A015F).
- **Composition** : 16 092 images après suppression des doublons : 5 628 saines, 5 767 Black Sigatoka, 4 697 Fusarium Wilt Race 1. Ce sont les chiffres de l'article, que j'ai retrouvés exactement en comptant les fichiers ; la fiche descriptive fournie avec les données (17 068 images) décrit l'état avant suppression des doublons. Les dossiers « sain » ne contiennent que des feuilles ; d'après l'article, les dossiers malades contiennent des feuilles et des tiges. Les images sont nommées `Image_<numéro>.jpg`, sans indication de feuille ou de tige. Je ne garde que les feuilles, puisque le sujet porte sur la classification de feuilles : le tri se fera donc sur le contenu des images. Les archives contiennent aussi des fichiers système macOS (`__MACOSX`, `.DS_Store`), que j'ignore.
- **Découpage** : entraînement du modèle maladies.
- **Risques de fuite** : ce jeu vient de la même équipe que le jeu Tanzanie 2025, avec des régions en commun (Arusha, Mbeya). Les mêmes fermes, voire les mêmes plants, peuvent apparaître dans les deux. J'ai recherché les doublons entre les deux jeux : je n'en ai trouvé aucun (détail en section 2). La fiche descriptive précise aussi que quelques doublons et des images très proches peuvent subsister à l'intérieur du jeu.
- **Biais et limites** :
    - les tiges n'apparaissent que dans les classes malades ; sans tri, un modèle pourrait apprendre « tige = malade ». J'ai regardé 40 images tirées au hasard dans chacun des 6 dossiers : je n'ai trouvé des tiges que dans Fusarium, et aucune dans les 80 images Black Sigatoka. J'ai ensuite passé en revue, sur des planches de vignettes, les 4 697 images Fusarium : 317 sont des tiges (273 dans FUSARIUM WILT_1, 44 dans FUSARIUM WILT_2), toutes des coupes du pseudo-tronc ou du bulbe, faciles à reconnaître (forme ronde ou fibres, anneaux bruns sur fond blanc). J'ai fait de même pour les 5 767 images Black Sigatoka : une seule est une tige (`BLACK SIGATOKA_2/Image_6450.jpg`). J'écarte ces 318 images de l'entraînement, ce qui laisse 4 380 images de feuilles Fusarium et 5 766 images de feuilles Black Sigatoka. Le déséquilibre « tige = Fusarium » est donc le vrai risque de ce jeu ;
    - certains formats d'image n'existent que dans une classe (640 × 288 px chez les saines, 640 × 480 px chez Fusarium) : toutes les images sont ramenées à 224 × 224 px (`outils/preparer_feuilles.py`), pour que le modèle ne puisse pas s'en servir ;
    - cadrage : gros plans qui remplissent presque toute l'image, avec très peu de fond. Les feuilles saines montrent souvent la nervure centrale ;
    - lumière : photos de jour. Une partie des images Black Sigatoka a une teinte violette ou bleutée, sans doute un réglage des couleurs du téléphone ;
    - chaque classe a été photographiée à des jours différents (métadonnées EXIF) : saines du 18 au 25 mars, Black Sigatoka surtout du 26 au 30 mars, Fusarium du 2 au 12 avril 2021. Un modèle pourrait apprendre la lumière ou le champ d'un jour plutôt que la maladie : c'est une raison de plus pour valider et tester sur d'autres jeux ;
    - un seul pays, une seule région (Arusha) pour les photos géolocalisées.

## 2. Feuilles de bananier — Tanzanie 2025 (Zenodo)

- **Provenance** : Mduma & Elinisa, *Banana Leaves Imagery Dataset*, Scientific Data (2025), [DOI 10.1038/s41597-025-04456-4](https://doi.org/10.1038/s41597-025-04456-4). Données sur [Zenodo](https://doi.org/10.5281/zenodo.7670326), téléchargées le 18/10/2026. Photos prises dans les districts de Rungwe, Mbeya, Arumeru et Arusha.
- **Licence** : CC BY 4.0.
- **Résolution** : smartphone, 4624 × 3468 px pour la grande majorité des images (sondage de 200 images par dossier), quelques images recadrées.
- **Date d'acquisition** : novembre 2022 à janvier 2023 (article) ; les métadonnées des photos vont du 22/11/2022 au 20/01/2023. Autre téléphone que pour 2023 (Samsung SM-A715F), pas de position GPS.
- **Composition** : le jeu complet compte 11 767 images de feuilles (3 339 saines, 3 496 Black Sigatoka, 4 932 Fusarium Wilt Race 1), réparties en 9 archives. J'en ai téléchargé une par classe (BLACK SIGATOKA-1, FUSARIUM WILT-1, HEALTHY-1), soit 3 001 images : 1 000 saines, 1 000 Black Sigatoka, 1 001 Fusarium Wilt Race 1. Ce sous-ensemble équilibré suffit pour la validation.
- **Découpage** : validation du modèle maladies.
- **Risques de fuite** : voir Tanzanie 2023. La séparation entre les deux jeux est temporelle (2021 contre 2022-2023), pas géographique. J'ai recherché les doublons entre les deux jeux (empreinte d'image, puis comparaison des images réduites, puis vérification à l’œil des 29 paires les plus proches) : je n'en ai trouvé aucun. Cette méthode repère les copies d'une même photo ; elle ne garantit pas qu'une même feuille n'a pas été photographiée à nouveau sous un autre angle.
- **Biais et limites** :
    - même pays et même équipe que le jeu d'entraînement, donc des scores de validation probablement optimistes. C'est pour cette raison que je teste sur un autre pays ;
    - cadrage différent du jeu 2023 : feuilles souvent entières, avec du fond (sol, autres plants, ciel), et non des gros plans. Le modèle sera donc validé sur des images qui ne ressemblent pas tout à fait à son entraînement ;
    - dans un échantillon de 24 images par classe, des tuyaux d'irrigation bleus apparaissent derrière 3 feuilles saines, et jamais dans les autres classes : un modèle pourrait associer ce décor à la classe saine ;
    - dans ce même échantillon, 2 feuilles étiquetées saines portent des taches brunes (`HEALTHY_453.jpg`, `HEALTHY_486.jpg`) : il peut y avoir des erreurs d'étiquetage.

## 3. Feuilles de bananier — Bangladesh, BananaLSD (Mendeley Data)

- **Provenance** : Arman et al., *BananaLSD: A banana leaf images dataset for classification of banana leaf diseases using machine learning*, Data in Brief (2023) 109608, [DOI 10.1016/j.dib.2023.109608](https://doi.org/10.1016/j.dib.2023.109608). Données sur [Mendeley Data](https://data.mendeley.com/datasets/9tb7k297ff/1), téléchargées le 18/10/2026. Photos prises dans les champs de la Bangabandhu Sheikh Mujibur Rahman Agricultural University et les champs voisins, à Gazipur.
- **Licence** : CC BY 4.0.
- **Résolution** : 3 smartphones, lumière naturelle ; images fournies en 224 × 224 px.
- **Date d'acquisition** : juin 2021.
- **Composition** : 937 photos originales (473 sigatoka, 173 pestalotiopsis, 162 cordana, 129 saines), étiquetées par un phytopathologiste. Vérifié après décompression. Le dépôt contient aussi 1 600 images augmentées, copies modifiées des originales : je ne les utilise pas.
- **Découpage** : test sur un autre pays. Les classes cordana et pestalotiopsis, jamais vues à l'entraînement, servent à tester la réponse « je ne sais pas ».
- **Risques de fuite** : aucune source commune avec les jeux tanzaniens (autre pays, autre équipe). Les images augmentées sont exclues, car elles feraient compter plusieurs fois la même photo.
- **Biais et limites** : petit jeu (937 images), déséquilibré (473 sigatoka contre 129 saines), un seul site, un seul mois. Pas de Fusarium. Le type de Sigatoka (noire ou jaune) n'est pas précisé. Cadrage proche de Tanzanie 2023 (gros plans), avec des couleurs plus ternes. Le jeu contient des doublons, que j'ai recherchés automatiquement (`outils/preparer_feuilles.py`) puis vérifiés à l'œil : 42 groupes d'images identiques ou presque. 38 concernent la classe saine, ce qui ramène les 129 photos saines à environ 90 photos distinctes. Une même photo apparaît aussi deux fois avec deux étiquettes différentes (`healthy/122.jpg` et `pestalotiopsis/78.jpg`). Je retire 44 images du test (liste dans `outils/listes/doublons_bangladesh.txt`) : une seule image est gardée par groupe, et les deux images à l'étiquette contradictoire sont écartées. Il reste 893 images de test : 90 saines, 472 sigatoka, 159 cordana, 172 pestalotiopsis. Les images sont déjà en 224 px alors que les jeux tanzaniens sont en haute résolution : toutes les images des trois jeux sont ramenées à 224 × 224 px, pour que le modèle ne puisse pas reconnaître un jeu à sa résolution.

## 4. Images drone — ReforesTree, Équateur (Zenodo)

- **Provenance** : Reiersen et al., *ReforesTree: A Dataset for Estimating Tropical Forest Carbon Stock with Deep Learning and Aerial Imagery*, AAAI 2022, [arXiv:2201.11192](https://arxiv.org/abs/2201.11192). Données sur [Zenodo](https://zenodo.org/records/6813783), téléchargées le 18/10/2026. Six fermes de cacao en agroforesterie : Carlos Vera Arteaga, Carlos Vera Guevara, Flora Pluas, Leonor Aspiazu, Manuel Macias, Nestor Macias, en deux groupes distants d'environ 110 km (Flora et Nestor au sud, près de Milagro ; les quatre autres au nord, près de Mocache).
- **Licence** : CC BY 4.0.
- **Résolution** : drone Mavic 2 Pro, caméra RGB, vol à 40 m. Une orthomosaïque par ferme, de 0,97 à 1,19 cm par pixel selon la ferme (fichiers `.tfw`), erreur de positionnement de 0,25 à 0,69 m. Je réduis les images 4 fois pour travailler plus vite : 3,9 à 4,8 cm par pixel selon la ferme.
- **Date d'acquisition** : 27 et 28 octobre, probablement 2020 (rapport de vol daté du 1.11.2020).
- **Composition** : chaque arbre est annoté par une boîte autour de sa couronne, étiquetée bananier ou non (`annotations/cleaned/clean_annotations.csv`, 7 969 boîtes). Les auteurs ont annoté sur des tuiles qui se chevauchent, sans fusionner les doublons : un même arbre a souvent deux boîtes, et toutes celles d'Arteaga sont en double. Je fusionne les boîtes d'une même ferme qui se recouvrent à plus de 50 % (`outils/preparer_reforestree.py`) : il reste 5 328 arbres, dont 2 731 bananiers. Chaque ferme a une **parcelle** inventoriée sur le terrain (shapefile `Merged_final_plots`) ; les annotations ne sont complètes qu'à l'intérieur. Dans les parcelles :

| Ferme | Surface | Arbres | Bananiers | Part de bananiers |
|---|---|---|---|---|
| Flora Pluas | 0,40 ha | 1 051 | 701 | 67 % |
| Leonor Aspiazu | 0,48 ha | 631 | 292 | 46 % |
| Carlos Vera Guevara | 0,43 ha | 769 | 202 | 26 % |
| Carlos Vera Arteaga | 0,53 ha | 142 | 61 | 43 % |
| Manuel Macias | 0,41 ha | 439 | 226 | 51 % |
| Nestor Macias | 0,45 ha | 965 | 667 | 69 % |

- **Découpage** : par ferme entière. Entraînement : Flora, Leonor, Guevara, Arteaga. Validation : Manuel. Test : Nestor, que je ne regarde qu'une fois, à la fin. Pour des scores plus fiables, je fais aussi une validation croisée par ferme sur les 5 fermes hors Nestor : chaque ferme sert une fois de validation. Le comptage est noté uniquement à l'intérieur des parcelles. Pour DeepForest, les images sont découpées en tuiles de 400 × 400 pixels (environ 18 m), et tout ce qui est hors de la parcelle est peint en noir (`outils/preparer_tuiles_comptage.py`).
- **Risques de fuite** : les tuiles d'une même ferme se chevauchent ; un découpage au hasard mettrait les mêmes arbres en entraînement et en test, d'où la séparation par ferme. Le modèle DeepForest fourni avec le jeu a été entraîné par les auteurs sur ces mêmes fermes : je ne l'utilise pas.
- **Biais et limites** :
    - fermes de cacao où le bananier est une culture secondaire, mêlée à d'autres arbres, et non une bananeraie ;
    - les boîtes ne distinguent pas pied mère et rejets ;
    - hors des parcelles, beaucoup de bananiers visibles n'ont pas de boîte. Dans les parcelles aussi, il manque des bananiers et des boîtes sont mal placées (vérification ci-dessous) ;
    - Arteaga est très peu dense : 142 arbres dans 0,53 ha, environ 270 par hectare contre environ 2 600 à Flora. Sa parcelle (contour officiel des auteurs, vérifié) est une zone sèche avec beaucoup de petites plantes non annotées ;
    - éclairage différent selon les fermes : ciel couvert à Flora, Nestor, Manuel et Leonor, soleil à Arteaga et Guevara ;
    - l'erreur de positionnement (jusqu'à 0,69 m) est proche de la tolérance de 1 m utilisée pour noter.

### Qualité des annotations : ce que j'ai vérifié à l'œil

![Manuel : parcelle et arbres annotés](images/reforestree_manuel_annotations.jpg)
*Manuel Macias. En jaune la parcelle inventoriée, en rouge les bananiers annotés, en bleu les autres arbres annotés (un rond = le centre d'une boîte).*

Les bananiers annotés de Manuel sont regroupés dans le sud de la parcelle. Au nord, des bananiers sont visibles sur l'image sans être annotés.

![Flora : carré de 20 m](images/reforestree_flora_zoom_annotations.jpg)
*Flora Pluas, carré de 20 m au centre de la parcelle, image d'origine (environ 1 cm par pixel). Annotations après fusion des doublons : 90 bananiers (rouge) et 10 autres arbres (bleu). Le point est le centre de la boîte, c'est la position de l'arbre que j'utilise pour noter. Le carré a la largeur de la boîte d'origine (après fusion, je ne garde que le centre et la largeur, la forme est donc approchée).*

Dans les zones denses, les boîtes ne suivent pas toujours les couronnes : certaines n'en couvrent qu'une partie, plusieurs boîtes se partagent parfois une même couronne, et quelques grandes couronnes n'ont aucune boîte.

**Vérification sur 50 fausses alertes.** Sur Manuel, ma méthode simple fait 511 détections, dont 436 sans bananier annoté à moins de 1 m. J'en ai tiré 50 au hasard et je les ai classées à l'œil :

| Ce qu'il y a sous la détection | Nombre |
|---|---|
| une autre plante | 22 |
| **un bananier sans annotation** | **15** |
| un bananier annoté, mais à plus de 1 m de la détection | 8 |
| du sol, rien de clair | 5 |

![Bananiers non annotés](images/reforestree_bananiers_non_annotes.jpg)
*6 des 15 bananiers non annotés (Manuel). Croix jaune = détection de la méthode simple, cercle = 1 m, rond bleu = autre arbre annoté. Aucun bananier n'est annoté dans le cercle.*

Environ 30 % de mes fausses alertes sur Manuel sont donc des bananiers absents des annotations (estimation sur 50 cas, donc imprécise). Le score mesuré sur ReforesTree sous-estime la qualité réelle d'une méthode, et un modèle entraîné sur ces annotations reçoit des exemples contradictoires : un bananier annoté, un bananier identique non annoté. Je ne corrige pas les annotations de validation et de test : la note doit rester comparable et ne pas être ajustée à ma méthode.

## 5. Images satellite — Sentinel-2 (Copernicus)

- **Provenance** : [Copernicus Browser](https://browser.dataspace.copernicus.eu/), produits de niveau L2A (corrigés de l'atmosphère), téléchargés le 20/10/2026. Deux bananeraies industrielles dans deux pays, identifiées au préalable sur une imagerie satellite haute résolution :
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
- **Découpage** : pas d'apprentissage ; les seuils des cartes de vigueur et de stress hydrique sont réglés sur l'Équateur et évalués sur le Cameroun.
- **Risques de fuite** : aucun entre les sites, situés dans deux pays différents. Au sein d'un site, les deux dates sont comparées entre elles, pas utilisées l'une pour régler l'autre.
- **Biais et limites** :
    - un pixel couvre 10 m, donc plusieurs bananiers à la fois : aucune lecture à l'échelle du plant ;
    - zones très nuageuses : les rectangles ont été placés sur les seuls blocs dégagés aux deux dates, ce qui limite leur taille (en particulier au Cameroun) et peut les rendre peu représentatifs de toute la plantation ;
    - au Cameroun, 9,5 % des pixels sont classés « sol nu » en décembre contre 0,2 % en avril : ces parcelles étaient probablement en cours de replantation et ont été couvertes par la végétation entre les deux dates. Une différence d'indice sur ces parcelles reflète ce changement, pas un manque d'eau : elles seront exclues de la comparaison du stress hydrique ;
    - les deux sites réagissent très différemment aux saisons : en Équateur, l'écart d'indice d'humidité (NDMI) médian entre les deux dates est faible (0,43 contre 0,41), probablement grâce à l'irrigation ; au Cameroun, il est plus marqué (0,34 contre 0,30). Des seuils absolus ne seraient donc pas transposables d'un site à l'autre ;
    - les dates de saison des pluies et de saison sèche sont séparées de 4 à 6 mois : l'état de la plantation (récoltes, replantations) a pu changer entre les deux.

## 6. Parcelles — tracées à la main (QGIS)

- **Provenance** : polygones que j'ai tracés à la main dans QGIS 3.44, sur la bande rouge (B04) des images Sentinel-2 de la section 5 : image du 16/12/2025 pour le Cameroun, du 22/09/2026 pour l'Équateur. Tracé le 20/10/2026 (Cameroun) et le 21/10/2026 (Équateur).
- **Licence** : travail personnel, livré avec le projet. Les polygones sont tracés sur des données Copernicus (mention : « Contains modified Copernicus Sentinel data 2025, 2026 »).
- **Résolution** : précision d'environ un pixel, soit 10 m. Tracé dans la projection de l'image (EPSG:32632 pour le Cameroun, EPSG:32717 pour l'Équateur), puis exporté en GeoJSON en latitude-longitude (EPSG:4326).
- **Date d'acquisition** : celle des images utilisées pour le tracé (voir Provenance).
- **Composition** : deux fichiers, `parcelles_cameroun.geojson` et `parcelles_equateur.geojson`, de 15 polygones chacun, avec un champ `id` de 1 à 15. Surfaces : 2,8 à 4,4 ha par parcelle au Cameroun (50 ha au total), 1,4 à 3,0 ha en Équateur (32 ha au total).
- **Découpage** : comme les images de la section 5 : réglage sur l'Équateur, évaluation sur le Cameroun.
- **Risques de fuite** : aucun car aucun modèle n'est entraîné sur ces parcelles.
- **Biais et limites** :
    - au Cameroun, les limites des parcelles se voient bien sur l'image : les chemins qui les séparent apparaissent comme des lignes claires. J'ai tracé chaque parcelle à l'intérieur de ces chemins ;
    - en Équateur, les limites des lots de la plantation ne se voient pas à 10 m. Les seules lignes visibles sont des couloirs de câbles parallèles, en diagonale, plus fins qu'un pixel : ils apparaissent comme des lignes floues. J'ai donc placé les parcelles au centre des bandes de végétation entre deux couloirs, en coupant chaque bande en plusieurs morceaux. Ces parcelles sont un découpage de ma part, pas les vrais lots de l'exploitation ;
    - en Équateur, l'image paraissait presque noire à l'ouverture, car le toit très clair de l'hacienda écrasait le contraste. J'ai réglé l'affichage (coupe à 2 % et 98 % de l'histogramme) pour voir les couloirs ; ce réglage ne modifie pas les valeurs des pixels ;
    - j'ai évité les routes, l'hacienda et les zones hors bananeraie. Les parcelles ne contiennent donc que l'intérieur de la plantation, sans ses bords. Malgré ces précautions, les pixels en bordure de parcelle peuvent mélanger bananiers et chemins ;
    - avec un pixel de 10 m, une parcelle de 1,4 ha ne contient qu'environ 140 pixels.

## 7. Modèles pré-entraînés

**MobileNetV3-Large (maladies)**

- **Modèle** : poids `IMAGENET1K_V2` de torchvision, pré-entraînés sur ImageNet-1k (1,2 million de photos, 1 000 classes d'objets).
- **Usage** : point de départ du modèle maladies. Je remplace la dernière couche (3 classes : saine, Black Sigatoka, Fusarium), puis j'entraîne tout le réseau sur Tanzanie 2023 (`entrainement/maladies.ipynb`).
- **Pourquoi** : réseau léger, conçu pour tourner sans GPU, comme le demande le sujet ; le pré-entraînement évite d'apprendre à partir de zéro avec seulement 15 774 photos.
- **Risque de fuite** : ImageNet ne contient pas les photos de Tanzanie ni du Bangladesh.

**DeepForest (comptage)**

- **Modèle** : détecteur de couronnes d'arbres `weecology/deepforest-tree` (bibliothèque DeepForest 2, réseau RetinaNet), pré-entraîné par ses auteurs sur des images aériennes de forêts, à environ 10 cm par pixel.
- **Usage** : point de départ du détecteur de bananiers. Je le réentraîne sur les bananiers de ReforesTree et de Count banana plants (`entrainement/comptage_deepforest_*.ipynb`).
- **Limite** : mes images sont à environ 4,5 cm par pixel, deux fois plus fines que celles de son pré-entraînement : les arbres lui paraissent deux fois plus gros.
- **Risque de fuite** : ses données de pré-entraînement ne contiennent pas les fermes ReforesTree.

## 8. Images drone — Count banana plants (Roboflow Universe)

- **Provenance** : projet [« Count banana plants »](https://universe.roboflow.com/count-banana-plants/count-banana-plants) publié sur Roboflow Universe par « COUNT BANANA PLANTS ». Version **v15 « DATASET FINAL SIN AUG »** (générée le 27/07/2022), téléchargée au format COCO le 28/10/2026. Je l'ai trouvé en cherchant d'autres jeux de bananiers vus par drone, parce que les annotations de ReforesTree sont incomplètes (section 4). Pays, date et auteurs réels : non indiqués.
- **Licence** : CC BY 4.0.
- **Choix de la version** : le projet a 10 versions des mêmes images. Toutes sauf deux contiennent des copies modifiées des images d'entraînement (retournées, pivotées, plus ou moins claires : ×2 pour la v38, ×3 pour les v25 et v32) ou des images redimensionnées en 456 × 456 (v6 à v9). Ces copies seraient des quasi-doublons et créeraient des fuites. La v15 est la seule grande version sans copie et sans redimensionnement.
- **Résolution** : 400 images de 480 × 480 pixels, découpées dans 77 photos de drone brutes (le nom du fichier donne le dossier de l'appareil, le numéro de photo et la position du morceau, par exemple `100_0221_0204_0_2880`). Ce ne sont pas des orthomosaïques : pas de coordonnées, et la taille du pixel n'est pas donnée. Je l'estime à environ 3 cm : la boîte médiane d'un bananier fait 105 pixels, pour une couronne adulte d'environ 3 m. C'est une hypothèse, pas une mesure.
- **Composition** : 4 672 boîtes d'une seule classe, `Banana-plant` (j'en écarte une de 1,3 pixel de large), environ 10 par image (au plus 32), aucune image vide. Par dossier : `100_0221` 217 morceaux / 15 photos / 3 083 bananiers, `100_0222` 85 / 25 / 977, `100_0223` 93 / 35 / 591, `100_0224` 5 / 2 / 21.

![Count banana plants : annotations](images/count_banana_annotations.jpg)
*12 images tirées au hasard, boîtes des auteurs en rouge.*

- **Découpage** : le découpage train / valid / test fourni par Roboflow est fait au hasard, et des morceaux d'une même photo se retrouvent des deux côtés. Je ne l'utilise pas. Je sépare par dossier, que je considère comme un vol : apprentissage `100_0221` + `100_0222` (302 images), validation `100_0223` + `100_0224` (98 images).
- **Risques de fuite** : les photos successives d'un même vol se recouvrent et montrent les mêmes plants ; c'est pourquoi un vol entier va d'un seul côté. Je ne sais pas si les dossiers correspondent à des parcelles différentes ou à la même parcelle à des dates différentes.
- **Biais et limites** : bananeraie presque pure, plants bien séparés, très différente des fermes mélangées de ReforesTree. Sur les 12 images regardées, presque chaque bananier a sa boîte (quelques oublis au bord des images) : les annotations semblent beaucoup plus complètes que celles de ReforesTree. Photos brutes, donc vue un peu en perspective sur les bords. Taille du pixel estimée.
- **Utilisation** : test de la méthode simple sur une zone jamais vue, et apprentissage de DeepForest (seul, puis mélangé à ReforesTree). Résultats : voir [experiences.md](experiences.md).

## Écart entre photos de feuilles et images aériennes

Toutes les photos de feuilles sont prises au sol, de près, au smartphone. Le modèle maladies ne sait donc pas lire une image drone ou satellite. En conditions réelles, il sert d'outil de terrain : les cartes aériennes indiquent où envoyer une équipe, qui photographie les feuilles pour confirmer la maladie. Détecter les maladies directement depuis le drone demanderait des images aériennes de feuilles malades annotées, que je n'ai pas trouvées en données publiques.
