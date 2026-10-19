# Carnet de données — Field Scan

Mis à jour le 19/10/2026. Les mentions *à compléter* signalent ce que je n'ai pas encore vérifié.

## 1. Feuilles de bananier — Tanzanie 2023 (Harvard Dataverse)

- **Provenance** : Mduma & Leo, *Dataset of banana leaves and stem images for object detection, classification and segmentation: A case of Tanzania*, Data in Brief 49 (2023) 109322, [DOI 10.1016/j.dib.2023.109322](https://doi.org/10.1016/j.dib.2023.109322). Données sur [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/LQUWXW), téléchargées le 18/10/2026. Photos prises dans les régions de Kagera, Arusha, Dar es Salaam, Kilimandjaro et Mbeya.
- **Licence** : CC0 1.0 (domaine public).
- **Résolution** : smartphones Samsung 13 Mpx, photos prises au sol. Les images fournies sont réduites : 1024 × 768 px pour la plupart (en paysage ou en portrait), avec une partie en 640 × 480 px dans FUSARIUM WILT_1 et en 640 × 288 px dans HEALTHY_1 (sondage de 200 images par dossier).
- **Date d'acquisition** : février à juillet 2021.
- **Composition** : 16 092 images après suppression des doublons : 5 628 saines, 5 767 Black Sigatoka, 4 697 Fusarium Wilt Race 1. Ce sont les chiffres de l'article, que j'ai retrouvés exactement en comptant les fichiers ; la fiche descriptive fournie avec les données (17 068 images) décrit l'état avant suppression des doublons. Les dossiers « sain » ne contiennent que des feuilles ; les dossiers malades contiennent des feuilles et des tiges. Les images sont nommées `Image_<numéro>.jpg`, sans indication de feuille ou de tige. Je ne garde que les feuilles, puisque le sujet porte sur la classification de feuilles : le tri se fera donc sur le contenu des images. Les archives contiennent aussi des fichiers système macOS (`__MACOSX`, `.DS_Store`), que j'ignore.
- **Découpage** : entraînement du modèle maladies.
- **Risques de fuite** : ce jeu vient de la même équipe que le jeu Tanzanie 2025, avec des régions en commun (Arusha, Mbeya). Les mêmes fermes, voire les mêmes plants, peuvent apparaître dans les deux. Je rechercherai les quasi-doublons entre les deux jeux avant l'entraînement. La fiche descriptive précise aussi que quelques doublons et des images très proches peuvent subsister à l'intérieur du jeu.
- **Biais et limites** : les tiges n'apparaissent que dans les classes malades ; sans tri, un modèle pourrait apprendre « tige = malade ». De même, certains formats d'image n'existent que dans une classe (640 × 288 px chez les saines, 640 × 480 px chez Fusarium) : je ramènerai toutes les images à la même taille pour que le modèle ne puisse pas s'en servir. Un seul pays. Fonds, lumière et cadrage : *à compléter après observation des images*.

## 2. Feuilles de bananier — Tanzanie 2025 (Zenodo)

- **Provenance** : Mduma & Elinisa, *Banana Leaves Imagery Dataset*, Scientific Data (2025), [DOI 10.1038/s41597-025-04456-4](https://doi.org/10.1038/s41597-025-04456-4). Données sur [Zenodo](https://doi.org/10.5281/zenodo.7670326), téléchargées le 18/10/2026. Photos prises dans les districts de Rungwe, Mbeya, Arumeru et Arusha.
- **Licence** : CC BY 4.0.
- **Résolution** : smartphone, 4624 × 3468 px pour la grande majorité des images (sondage de 200 images par dossier), quelques images recadrées.
- **Date d'acquisition** : du 22/11/2022 au 13/01/2023.
- **Composition** : le jeu complet compte 11 767 images de feuilles (3 339 saines, 3 496 Black Sigatoka, 4 932 Fusarium Wilt Race 1), réparties en 9 archives. J'en ai téléchargé une par classe (BLACK SIGATOKA-1, FUSARIUM WILT-1, HEALTHY-1), soit 3 001 images : 1 000 saines, 1 000 Black Sigatoka, 1 001 Fusarium Wilt Race 1. Ce sous-ensemble équilibré suffit pour la validation.
- **Découpage** : validation du modèle maladies.
- **Risques de fuite** : voir Tanzanie 2023. La séparation entre les deux jeux est temporelle (2021 contre 2022-2023), pas géographique.
- **Biais et limites** : même pays, même équipe et même protocole que le jeu d'entraînement, donc des scores de validation probablement optimistes. C'est pour cette raison que je teste sur un autre pays.

## 3. Feuilles de bananier — Bangladesh, BananaLSD (Mendeley Data)

- **Provenance** : Arman et al., *BananaLSD: A banana leaf images dataset for classification of banana leaf diseases using machine learning*, Data in Brief (2023) 109608, [DOI 10.1016/j.dib.2023.109608](https://doi.org/10.1016/j.dib.2023.109608). Données sur [Mendeley Data](https://data.mendeley.com/datasets/9tb7k297ff/1), téléchargées le 18/10/2026. Photos prises dans les champs de la Bangabandhu Sheikh Mujibur Rahman Agricultural University et les champs voisins, à Gazipur.
- **Licence** : CC BY 4.0.
- **Résolution** : 3 smartphones, lumière naturelle ; images fournies en 224 × 224 px.
- **Date d'acquisition** : juin 2021.
- **Composition** : 937 photos originales (473 sigatoka, 173 pestalotiopsis, 162 cordana, 129 saines), étiquetées par un phytopathologiste. Vérifié après décompression. Le dépôt contient aussi 1 600 images augmentées, copies modifiées des originales : je ne les utilise pas.
- **Découpage** : test sur un autre pays. Les classes cordana et pestalotiopsis, jamais vues à l'entraînement, servent à tester la réponse « je ne sais pas ».
- **Risques de fuite** : aucune source commune avec les jeux tanzaniens (autre pays, autre équipe). Les images augmentées sont exclues, car elles feraient compter plusieurs fois la même photo.
- **Biais et limites** : petit jeu (937 images), déséquilibré (473 sigatoka contre 129 saines), un seul site, un seul mois. Pas de Fusarium. Le type de Sigatoka (noire ou jaune) n'est pas précisé. Les images sont déjà en 224 px alors que les jeux tanzaniens sont en haute résolution : je ramènerai toutes les images à la même taille, pour que le modèle ne puisse pas reconnaître un jeu à sa résolution.

## 4. Images drone — ReforesTree, Équateur (Zenodo)

- **Provenance** : Reiersen et al., *ReforesTree: A Dataset for Estimating Tropical Forest Carbon Stock with Deep Learning and Aerial Imagery*, AAAI 2022, [arXiv:2201.11192](https://arxiv.org/abs/2201.11192). Données sur [Zenodo](https://zenodo.org/records/6813783), téléchargées le 18/10/2026. Vols réalisés par OpenForests (rapport *Drone Mapping of Cacao Agroforestry Parcels in Ecuador*, fourni dans le jeu). Six parcelles de cacao en agroforesterie d'environ 0,5 ha chacune : Carlos Vera Arteaga, Carlos Vera Guevara, Flora Pluas, Leonor Aspiazu, Manuel Macias, Nestor Macias. D'après leurs coordonnées, elles forment deux groupes distants d'environ 110 km : Flora Pluas et Nestor Macias au sud (près de Milagro), les quatre autres au nord (près de Mocache).
- **Licence** : CC BY 4.0.
- **Résolution** : drone Mavic 2 Pro, caméra RGB, vol à 40 m avec 80 % de recouvrement. Une orthomosaïque par site (par exemple 12 425 × 12 305 px pour Carlos Vera Guevara). Les orthomosaïques sont géoréférencées en WGS 84 (latitude, longitude) par un fichier `.tfw` placé à côté de chaque image. La taille de pixel qu'il indique va de 0,97 à 1,19 cm selon le site, comme dans le rapport de vol (l'article annonce environ 2 cm). Erreur de positionnement : 0,25 à 0,69 m.
- **Date d'acquisition** : 27 et 28 octobre, entre 9 h et 12 h. L'année n'est pas écrite dans le rapport, daté du 1.11.2020 : probablement octobre 2020.
- **Composition** : le fichier `annotations/cleaned/clean_annotations.csv` contient 7 969 couronnes annotées par des boîtes, dont 3 436 bananiers. Par site (bananiers / total) : Carlos Vera Arteaga 358 / 1 122, Carlos Vera Guevara 226 / 1 636, Flora Pluas 1 076 / 1 655, Leonor Aspiazu 382 / 983, Manuel Macias 349 / 728, Nestor Macias 1 045 / 1 845. Le jeu contient aussi 4 663 arbres mesurés au sol, dont 1 504 bananiers (`field_data.csv`), le contour des 6 parcelles (0,40 à 0,53 ha, shapefile `Merged_final_plots`), des tuiles de 4 000 × 4 000 px qui se chevauchent de 200 px, et un modèle DeepForest entraîné par les auteurs. D'après la documentation du jeu, les boîtes ont été proposées par un détecteur, puis nettoyées à la main et étiquetées banane ou non banane : j'utilise uniquement ce fichier nettoyé comme vérité terrain.
- **Découpage** : comptage des pieds, par site entier : 4 sites pour l'entraînement, 1 pour la validation, 1 pour le test. Répartition à fixer en tenant compte du nombre de bananiers (de 226 à 1 076 selon le site), de l'éclairage et des deux zones géographiques.
- **Risques de fuite** : les tuiles d'un même site se chevauchent. Un découpage aléatoire des tuiles placerait les mêmes arbres en entraînement et en test ; c'est pourquoi je sépare par site. Le modèle DeepForest fourni a été entraîné par les auteurs sur ces mêmes sites : je ne l'évalue pas sur ReforesTree.
- **Biais et limites** : parcelles de cacao où le bananier est une culture secondaire, mêlée à d'autres arbres, et non une plantation industrielle : la densité et la disposition des plants diffèrent. Les boîtes entourent des couronnes, sans distinguer pied mère et rejets. Comme elles ont été proposées par un détecteur, des couronnes qu'il a manquées peuvent être absentes des annotations. Un seul pays, une seule campagne de vol. Éclairage différent selon les sites : ciel couvert pour Flora Pluas, Nestor Macias, Manuel Macias et Leonor Aspiazu, soleil pour Carlos Vera Arteaga et Carlos Vera Guevara. L'erreur de positionnement (jusqu'à 0,69 m) est proche de la tolérance d'appariement de 1 m utilisée pour l'évaluation.

## 5. Images satellite — Sentinel-2 (Copernicus)

- **Provenance** : [Copernicus Browser](https://browser.dataspace.copernicus.eu/), produits de niveau L2A (corrigés de l'atmosphère). Sites : *à compléter* (2 bananeraies dans 2 pays).
- **Licence** : accès libre et gratuit. Mention obligatoire sur les produits dérivés : « Contains modified Copernicus Sentinel data [année] ».
- **Résolution** : 10 m (bleu, vert, rouge, proche infrarouge) et 20 m (red-edge, infrarouge moyen).
- **Date d'acquisition** : *à compléter* (une date en saison sèche et une en saison des pluies par site).
- **Composition** : *à compléter*.
- **Découpage** : pas d'apprentissage ; les seuils des cartes de vigueur et de stress hydrique sont réglés sur un site et évalués sur l'autre.
- **Risques de fuite** : aucun, les deux sites étant dans des pays différents.
- **Biais et limites** : un pixel couvre 10 à 20 m, donc plusieurs bananiers à la fois : aucune lecture à l'échelle du plant. Les nuages sont fréquents en zone tropicale.

## Écart entre photos de feuilles et images aériennes

Toutes les photos de feuilles sont prises au sol, de près, au smartphone. Le modèle maladies ne sait donc pas lire une image drone ou satellite. En conditions réelles, il sert d'outil de terrain : les cartes aériennes indiquent où envoyer une équipe, qui photographie les feuilles pour confirmer la maladie. Détecter les maladies directement depuis le drone demanderait des images aériennes de feuilles malades annotées, que je n'ai pas trouvées en données publiques.
