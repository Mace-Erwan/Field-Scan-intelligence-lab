# Journal des essais — Field Scan

Mis à jour le 02/11/2026. Les données utilisées sont décrites dans le [carnet de données](carnet_donnees.md).

## Comment je note

- **Précision** : parmi mes détections, combien sont justes (« quand je dis oui, ai-je raison ? »). Basse = beaucoup de fausses alertes.
- **Rappel** : parmi les vrais cas, combien j'en trouve (« est-ce que je n'en oublie pas ? »). Bas = beaucoup de ratés.
- **F1** : combine les deux ; il n'est bon que si les deux le sont. F1 = 2 × précision × rappel / (précision + rappel).
- **Comptage** : une détection est juste si son centre est à moins de 1 m du centre d'un vrai bananier, chaque bananier ne pouvant être trouvé qu'une fois (règle du sujet).
- **Score honnête ou optimiste** : un score est honnête si les réglages (seuil, époque…) ont été choisis sur d'autres données que celles qui sont notées. Sinon, il est optimiste.

## 1. Comptage des bananiers

Validation : réglages choisis sur Leonor ou par validation croisée par ferme ; Manuel sert à comparer les méthodes. Nestor (test) a été noté une fois par erreur, avec la méthode simple (F1 0,40) ; il ne servira plus qu'une fois, à la fin.

| Essai | Données d'apprentissage | F1 sur Manuel, honnête | F1 sur Manuel, optimiste |
|---|---|---|---|
| Méthode simple (indice de vert ExG, sommets) | aucune (3 réglages) | 0,204 | — |
| DeepForest pré-entraîné, sans réapprentissage | — | — | 0,208 |
| Agroforestry inventory (modèle bananier publié, RF-DETR) | — | — | 0,218 |
| DeepForest, 10 époques (essai 0) | 4 fermes ReforesTree | — | 0,291 |
| DeepForest, 30 époques (essai 1) | 4 fermes ReforesTree | — | 0,262 |
| DeepForest | Count banana plants | 0,197 | 0,292 |
| **DeepForest mixte** | **Count banana + Flora, Guevara, Arteaga** | **0,254** | 0,262 |

**Validation croisée par ferme** (chaque ferme notée avec un modèle qui ne l'a pas vue) :
- méthode simple : F1 global 0,247 (précision 0,18, rappel 0,41), de 0,12 (Arteaga) à 0,41 (Flora) ;
- DeepForest réentraîné sur ReforesTree : 0,175 sur 4 fermes (réglage commun, optimiste), 0,12 avec un réglage choisi sur les autres fermes, contre 0,255 pour la méthode simple sur les mêmes fermes. Le tour où Manuel est la ferme notée a échoué deux fois : la loss n'est plus un nombre (NaN) dès la première époque, le modèle ne détecte plus rien. Cause non trouvée ; j'ai ajouté une vérification de la loss dans les notebooks ;
- DeepForest mixte (Count banana + les fermes d'entraînement) : 0,288 sur 4 fermes contre 0,255 pour la méthode simple sur les mêmes fermes (Flora 0,440 contre 0,410, Leonor 0,267 contre 0,232, Guevara 0,227 contre 0,165, Arteaga 0,107 contre 0,119). Le tour Manuel a encore donné une loss NaN, pour la 3e fois.

**Count banana plants, zone jamais vue :**
- méthode simple, réglages de ReforesTree non retouchés : rappel 86 %, précision 31 %, F1 environ 0,46 (0,31 à 0,54 selon la taille de pixel supposée) ;
- DeepForest entraîné sur Count banana plants : F1 0,876 sur les vols de validation (précision 0,86, rappel 0,90), score optimiste car le seuil a été choisi sur ces mêmes vols ;
- DeepForest mixte : 0,69 sur ces mêmes vols avec le réglage choisi sur Leonor (0,85 au mieux).

**Ce que j'en retiens :**
- avec des annotations propres, DeepForest apprend très bien à reconnaître un bananier (0,876, optimiste, sur une bananeraie jamais vue) ;
- sur Manuel, toutes les méthodes plafonnent entre 0,26 et 0,29, même avec le meilleur réglage. La limite semble venir des annotations de Manuel (bananiers non annotés, boîtes mal placées : voir le carnet, section 4) plutôt que des modèles ;
- le bon seuil dépend du type de plantation (bananeraie pure ou ferme mélangée). Un réglage choisi sur une ferme mélangée (Leonor) se transfère bien à une autre ferme mélangée (Manuel) ;
- mélanger les deux jeux donne le meilleur score honnête sur Manuel (0,254), au prix d'une baisse sur la bananeraie pure.

**À faire :** noter les méthodes sur un échantillon que j'ai pointé moi-même ; choisir le réglage final par validation croisée ; tester l'effet de la résolution ; test unique sur Nestor.

## 2. Maladies

### v1 (MobileNetV3)

Apprentissage sur Tanzanie 2023, validation sur Tanzanie 2025 (3 001 images), test sur le Bangladesh (autre pays).

- **Validation** : F1 0,926, meilleure époque : la 3e sur 8. Seuil « je ne sais pas » : 0,679 (confiance en dessous de laquelle le modèle refuse de répondre).
- **Test Bangladesh** : 69 % de bonnes réponses sur les classes connues, rappel de la Sigatoka 66 %, 3 % de fausses alertes sur les feuilles saines. Maladies inconnues (Cordana, Pestalotiopsis) : seulement 20 % rejetées, le modèle les prend le plus souvent pour une maladie connue, avec confiance.

### v2 (`entrainement/maladies_v2_kaggle.ipynb`)

Données nettoyées à nouveau (voir le carnet) : apprentissage sur Tanzanie 2023 + Pérou (16 959 images), validation sur toute la Tanzanie 2025 (11 766), test sur le Bangladesh (893). Tous les réglages sont choisis sur la validation.

**Ce qui change par rapport à la v1 :**
- le Pérou rejoint l'apprentissage : un 2e pays, sans le biais « une classe = un jour de prise de vue » ;
- augmentations plus fortes : cadrage, rotation, couleurs, flou, et une zone masquée au hasard, pour que le modèle ne se fie pas à un seul détail ;
- **décision en deux étapes** :
    1. *malade ou saine ?* La feuille est malade si P(Black Sigatoka) + P(Fusarium) dépasse un seuil d'alerte. Une feuille malade ratée est l'erreur la plus grave : je choisis le seuil qui rend le plus petit possible *2 × malades ratées + fausses alertes*. Une malade ratée compte donc comme deux fausses alertes ;
    2. *quelle maladie ?* La maladie n'est nommée que si le modèle est assez sûr entre les deux. Le seuil est choisi pour que le nom donné soit juste 95 % du temps. Sinon, la réponse est « malade, non identifiée » : l'alerte est donnée, sans nom faux ;
- « je ne sais pas » quand la confiance du modèle est trop basse (seuil qui garde 95 % des photos de validation) ;
- chaque photo est regardée sous 4 vues (la photo, ses deux miroirs, sa rotation) et le modèle fait la moyenne de ses réponses.

**Choix testés sur la validation :**

| Question | Résultat | Choix |
|---|---|---|
| Comment choisir le seuil d'alerte ? | 0,5 fixe : 167 malades ratées et 105 fausses alertes, contre 96 et 189 au seuil de coût minimal (0,30, même entraînement). F2 : choisit 0,05, le plus bas testé (427 fausses alertes), car avec 2,5 fois plus de malades que de saines elle réagit à peine aux fausses alertes | coût (malade ratée = 2 fausses alertes) |
| Quelle époque garder ? | garder l'époque au coût le plus bas a donné une fois l'époque 1 : un modèle qui repère bien les malades mais ne nomme que 49 % des Fusarium (35 % en « malade, non identifiée »). Le coût saute aussi beaucoup d'une époque à l'autre (par exemple 408, 817, 583) | la dernière époque, pour tous les essais |
| 1 vue ou 4 vues ? | 4 vues ont fait mieux dans tous les essais (par exemple coût 392 avec 1 vue, 337 avec 4 vues pour EfficientNet-B2) | 4 vues |
| Comment repérer une maladie inconnue ? | voir « Inconnues simulées » ci-dessous | confiance |
| Quel réseau, et quel poids pour le Pérou ? | voir le tableau des essais ci-dessous | en cours |

**Essais comparés** (6 époques, modèle de la dernière époque, mêmes réglages ; « Pérou 3 » : chaque photo du Pérou est tirée 3 fois plus souvent, soit 20 % de chaque tour, « Pérou 0 » : sans le Pérou). Les deux dernières colonnes reprennent les archives `HEALTHY-1`, `BLACK SIGATOKA-1` et `FUSARIUM WILT-1` de la validation (1 000, 1 000 et 1 001 photos), avec la règle simple « malade si P(malade) ≥ 0,5 », sans « je ne sais pas ».

| Réseau | Pérou | Coût | Malades dites saines (sur 8 428) | dont Fusarium (sur 4 932) | Fausses alertes (saines) | Nom juste | Archives -1 bien classées : saine / Sigatoka / Fusarium | Fusarium dites saines (archives -1) |
|---|---|---|---|---|---|---|---|---|
| MobileNetV3 | 3 | 427 | 88 | 81 | 4,4 % | 93,5 % | 987 / 967 / 875 | 39 |
| MobileNetV3 | 1 | 564 | 131 | 114 | 6,1 % | 92,8 % | 969 / 951 / 865 | 49 |
| MobileNetV3 | 0 | 642 | 132 | 91 | 7,1 % | 92,2 % | 948 / 927 / 912 | 29 |
| EfficientNet-B0 | 0 | 419 | 78 | 56 | 4,2 % | 93,5 % | 964 / 964 / 912 | 11 |
| EfficientNet-B0 | 3 | 367 | 108 | 82 | 2,7 % | 94,1 % | 987 / 985 / 870 | 36 |
| **EfficientNet-B2** | **3** | **337** | **77** | **43** | **1,9 %** | **94,4 %** | 987 / 981 / 888 | 13 |

- **Le réseau** : les EfficientNet font mieux que MobileNetV3, et B2 est le meilleur sur presque tous les critères à la fois. Restent à essayer : B2 sans le Pérou, et ConvNeXt-Tiny ;
- **le Pérou** : avec MobileNetV3, plus il y a de Pérou, meilleur est le résultat sur tous les critères (coût 642, puis 564, puis 427). Avec EfficientNet-B0, c'est un échange : moins de fausses alertes avec le Pérou, mais plus de Fusarium ratées. Le Pérou n'a pas de Fusarium, ce qui peut rendre le modèle plus sûr de lui sur les feuilles saines ;
- chaque réglage n'a été entraîné qu'une fois. Pour des réglages identiques, le coût a varié de 324 à 427 d'un entraînement à l'autre : je ne tire de conclusion que des écarts nets ou réguliers.

**Détail du meilleur essai à ce jour** (EfficientNet-B2, Pérou 3 ; seuil d'alerte 0,40, seuil pour nommer la maladie 0,69) :

| Classe réelle | Saine | Black Sigatoka | Fusarium | Malade, non identifiée | Je ne sais pas |
|---|---|---|---|---|---|
| Saine (3 338) | **3 154 (94 %)** | 2 | 63 | 0 | 119 |
| Black Sigatoka (3 496) | 34 | **3 209 (92 %)** | 63 | 29 | 161 |
| Fusarium (4 932) | 43 | 315 | **4 196 (85 %)** | 69 | 309 |

- 77 feuilles malades sur 8 428 (0,9 %) sont dites saines ; toutes les autres reçoivent une alerte ou un « je ne sais pas » ;
- 1,9 % des feuilles saines reçoivent une fausse alerte ;
- le Fusarium reste le plus difficile : 6 % sont prises pour de la Black Sigatoka.

**Inconnues simulées.** Pour choisir comment repérer une maladie jamais vue sans toucher au Bangladesh, j'entraîne un modèle sans l'une des deux maladies, puis je la lui montre. Chaque méthode est réglée pour garder 95 % des photos connues de la validation.

| Maladie retirée | Rejetée par la confiance | Rejetée par la distance aux photos d'apprentissage | Les deux |
|---|---|---|---|
| Fusarium | **50 %** | 1 % | 32 % |
| Black Sigatoka | 8 % | 14 % | 14 % |

La distance (Mahalanobis, sur le résumé de 1 280 nombres que le réseau calcule pour chaque photo) fait moins bien que le hasard pour séparer connu et inconnu (AUROC 0,40 et 0,43, 0,5 = hasard). Mon explication probable : elle repère surtout que les photos de 2025 diffèrent de celles de 2023, pas qu'une maladie est nouvelle. Je garde donc la confiance seule (simulation faite avec MobileNetV3). Une maladie à taches comme la Sigatoka est très difficile à reconnaître comme inconnue (8 %) : la Cordana et la Pestalotiopsis du Bangladesh étant aussi des maladies à taches, je m'attends à peu de « je ne sais pas » sur elles au test. Ce qui comptera surtout, c'est qu'elles ne soient pas dites saines.

**Cartes d'attention.** Pour vérifier où le modèle regarde, j'utilise Grad-CAM : en rouge, les zones qui ont le plus pesé dans sa réponse (modèle MobileNetV3 du 01/11).

![Cartes d'attention du modèle maladies v2](images/maladies_cartes_attention.jpg)
*Photos de validation tirées au hasard : 4 saines, 4 Black Sigatoka, 4 Fusarium, et 4 Fusarium prises pour de la Black Sigatoka.*

- Black Sigatoka : le modèle regarde les taches et les stries noires ;
- Fusarium : zones jaunies ou desséchées, de façon plus diffuse ;
- Fusarium pris pour de la Black Sigatoka : le modèle s'arrête sur une petite marque locale (déchirure, tache sèche) qu'il prend pour une lésion ;
- saines : sur 2 photos sur 4, le rouge est en bas de la nervure centrale. C'est peut-être un raccourci (« nervure visible = saine »), à surveiller : 4 photos ne suffisent pas pour conclure ;
- aucun raccourci de décor visible (fond, mains, tuyaux).

**Test Bangladesh** : pas encore lancé. Il sera fait une seule fois, avec le modèle final.

## 3. Vigueur et stress hydrique (Sentinel-2, `run.py`)

NDVI (vigueur) et NDMI (eau dans le feuillage) moyens par parcelle, pixels nuageux exclus.

- **Cameroun** : en décembre, 10 parcelles sont denses et hydratées (NDVI 0,61 à 0,81), 5 sont clairsemées (NDVI 0,34 à 0,51, NDMI négatif) ; en avril, ces 5 parcelles remontent à 0,82–0,85 : c'était une replantation, pas un stress.
- **Équateur** : plantation homogène, NDVI de 0,79 à 0,86.
- **À faire** : règle de stress qui combine NDVI et NDMI, pour ne pas confondre parcelle sèche et parcelle clairsemée.
