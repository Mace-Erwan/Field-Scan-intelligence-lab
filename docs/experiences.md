# Journal des essais — Field Scan

Mis à jour le 28/10/2026. Les données utilisées sont décrites dans le [carnet de données](carnet_donnees.md).

## Comment je note

- **Précision** : parmi mes détections, combien sont justes (« quand je dis oui, ai-je raison ? »). Basse = beaucoup de fausses alertes.
- **Rappel** : parmi les vrais cas, combien j'en trouve (« est-ce que je n'en oublie pas ? »). Bas = beaucoup de ratés.
- **F1** : combine les deux ; il n'est bon que si les deux le sont. F1 = 2 × précision × rappel / (précision + rappel).
- **Comptage** : une détection est juste si son centre est à moins de 1 m du centre d'un vrai bananier, chaque bananier ne pouvant être trouvé qu'une fois (règle du sujet).
- **Score honnête ou optimiste** : un score est honnête si les réglages (seuil, époque…) ont été choisis sur d'autres données que celles qui sont notées. Sinon, il est optimiste.

## 1. Comptage des bananiers

Validation : réglages choisis sur Leonor ou par validation croisée par ferme ; Manuel sert à comparer les méthodes. Nestor (test) n'a pas encore été utilisé.

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
- DeepForest réentraîné sur ReforesTree : 0,175 sur 4 fermes (réglage commun, optimiste), 0,12 avec un réglage choisi sur les autres fermes, contre 0,255 pour la méthode simple sur les mêmes fermes. Le tour où Manuel est la ferme notée a échoué deux fois : la loss n'est plus un nombre (NaN) dès la première époque, le modèle ne détecte plus rien. Cause non trouvée ; j'ai ajouté une vérification de la loss dans les notebooks.

**Count banana plants, zone jamais vue :**
- méthode simple, réglages de ReforesTree non retouchés : rappel 86 %, précision 31 %, F1 environ 0,46 (0,31 à 0,54 selon la taille de pixel supposée) ;
- DeepForest entraîné sur Count banana plants : F1 0,876 sur les vols de validation (précision 0,86, rappel 0,90) ;
- DeepForest mixte : 0,69 sur ces mêmes vols avec le réglage choisi sur Leonor (0,85 au mieux).

**Ce que j'en retiens :**
- avec des annotations propres, DeepForest apprend très bien à reconnaître un bananier (0,876 sur une bananeraie jamais vue) ;
- sur Manuel, toutes les méthodes plafonnent entre 0,26 et 0,29, même avec le meilleur réglage. La limite semble venir des annotations de Manuel (bananiers non annotés, boîtes mal placées : voir le carnet, section 4) plutôt que des modèles ;
- le bon seuil dépend du type de plantation (bananeraie pure ou ferme mélangée). Un réglage choisi sur une ferme mélangée (Leonor) se transfère bien à une autre ferme mélangée (Manuel) ;
- mélanger les deux jeux donne le meilleur score honnête sur Manuel (0,254), au prix d'une baisse sur la bananeraie pure.

**À faire :** noter les méthodes sur un échantillon que j'ai pointé moi-même ; choisir le réglage final par validation croisée ; tester l'effet de la résolution ; test unique sur Nestor.

## 2. Maladies (MobileNetV3)

Apprentissage sur Tanzanie 2023, validation sur Tanzanie 2025, test sur le Bangladesh (autre pays).

- **Validation** : F1 0,926, meilleure époque : la 3e sur 8. Seuil « je ne sais pas » : 0,679 (confiance en dessous de laquelle le modèle refuse de répondre).
- **Test Bangladesh** : 69 % de bonnes réponses sur les classes connues, rappel de la Sigatoka 66 %, 3 % de fausses alertes sur les feuilles saines. Maladies inconnues (Cordana, Pestalotiopsis) : seulement 20 % rejetées, le modèle les prend le plus souvent pour une maladie connue, avec confiance.
- **À faire** : améliorer le rejet des maladies inconnues.

## 3. Vigueur et stress hydrique (Sentinel-2, `run.py`)

NDVI (vigueur) et NDMI (eau dans le feuillage) moyens par parcelle, pixels nuageux exclus.

- **Cameroun** : en décembre, 10 parcelles sont denses et hydratées (NDVI 0,61 à 0,81), 5 sont clairsemées (NDVI 0,34 à 0,51, NDMI négatif) ; en avril, ces 5 parcelles remontent à 0,82–0,85 : c'était une replantation, pas un stress.
- **Équateur** : plantation homogène, NDVI de 0,79 à 0,86.
- **À faire** : règle de stress qui combine NDVI et NDMI, pour ne pas confondre parcelle sèche et parcelle clairsemée.
