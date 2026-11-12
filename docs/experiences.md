# Journal des essais : Field Scan

Mis à jour le 10/11/2026. Les données sont décrites dans le [carnet de données](carnet_donnees.md).

## Où j'en suis

- **Comptage** : palm3, installé le 10/11. C'est un détecteur de centres (ResNet-18) : 2 réseaux, 4 vues, seuil 0,25. Au test du 10/11, sa F1 va de 0,767 à 0,778 sur trois types de plantation.
- **Maladies** : la v10b (ConvNeXt-Tiny aux poids DINOv3, 4 classes, lue en 384 px), avec la règle B, est installée depuis le 11/11 à la place de la v2. Au test neuf, elle laisse 12,0 % des feuilles malades sans alerte, contre 21,3 % pour la v2, mais elle nomme moins souvent la maladie. Sur BanSpot (Inde), elle rate 0,93 % des feuilles tachées, contre 3,56 % pour la v2, pour 10,2 % de fausses alertes contre 5,1 %. La règle écrite avant BanSpot gardait la v2, parce qu'elle n'acceptait que 3 points de fausses alertes en plus. Je l'ai installée quand même, en connaissant ce verdict : rater une feuille malade est plus grave qu'une fausse alerte.
- **Vigueur et stress hydrique** : indices Sentinel-2 et règles simples, sans apprentissage.

## Comment je note

- **Précision** : parmi mes détections, combien sont justes (« quand je dis oui, ai-je raison ? »). Basse = beaucoup de fausses alertes.
- **Rappel** : parmi les vrais cas, combien j'en trouve (« est-ce que je n'en oublie pas ? »). Bas = beaucoup de ratés.
- **F1** : combine les deux ; il n'est bon que si les deux le sont. F1 = 2 × précision × rappel / (précision + rappel).
- **Comptage** : une détection est juste si son centre est à moins de 1 m du centre d'un vrai bananier, chaque bananier ne pouvant être trouvé qu'une fois (règle du sujet). L'erreur de compte vaut (détections − plants) / plants.
- **Maladies** : « malades dites saines » = part des feuilles malades auxquelles le modèle répond « saine ». « Fausses alertes » = part des feuilles saines signalées malades. « Malades sans alerte » ajoute aux malades dites saines celles qui reçoivent un « je ne sais pas ».
- **Score honnête ou optimiste** : un score est honnête si les réglages (seuil, époque…) ont été choisis sur d'autres données que celles qui sont notées. Sinon, il est optimiste.
- **Test** : chaque test n'est noté qu'une fois. Sinon, ses chiffres ne voudraient plus rien dire.

## 1. Comptage des bananiers

### Premier essai

DeepForest entraîné sur Count banana plants seul : F1 0,876 sur ses vols de validation (précision 0,86, rappel 0,90). Score optimiste : le seuil a été choisi sur ces mêmes vols.

### Les données

Toutes les images sont remises à 5 cm par pixel (`outils/preparer_comptage.py` ; détail dans le carnet : ReforesTree, Count banana plants, Plants-bana). Chaque bananier devient un point (le centre de sa boîte) avec le diamètre de sa couronne. Le découpage se fait par ferme, par vol ou par zone, jamais au hasard. Il n'y a aucun doublon entre les parties (`outils/verifier_doublons_comptage.py`).

| Jeu | Apprentissage | Validation | Test |
|---|---|---|---|
| ReforesTree (fermes mélangées) | Flora, Leonor, Guevara, Arteaga : 621 bananiers | Manuel : 133 | Nestor : 235 |
| Count banana plants (bananeraie) | vols 0221 et 0222 : 302 images, 4 060 bananiers | vols 0223 et 0224 : 98 images, 611 | aucun |
| Plants-bana `segmento` (bananeraie dense) | zone de gauche : 47 images, 12 039 | bas droite, ouest : 17 images, 1 870 | 22 images, 2 498 (26 images et 2 801 bananiers avant le retrait des 4 morceaux qui touchent la validation, voir « Le test du 10/11 ») |
| Plants-bana `corte` | 49 images, 809 | aucune | aucun |

![Comptage : la référence](images/comptage_references.jpg)
*La référence, sans aucun modèle : un morceau de chaque jeu. Point rouge : bananier annoté par les auteurs du jeu ; cercle blanc : rayon de 1 m autour de lui, dans lequel une prédiction doit tomber pour compter comme juste.*

### v2 : détecteur de centres contre DeepForest (04/11/2026)

Deux modèles, mêmes données, même notation (`entrainement/comptage_v2_kaggle.ipynb`) :
- **détecteur de centres** : ResNet-18 pré-entraîné + un petit décodeur. Il produit une carte : pour chaque carré de 20 cm, la probabilité qu'un centre de bananier s'y trouve. Pour apprendre, la carte idéale a une petite bosse sur chaque bananier, d'autant plus large que sa couronne est grande ; la perte focale empêche le modèle de tout prédire à zéro (presque tous les carrés sont vides). Un bananier = un sommet de la carte (le plus fort dans un carré de 1,4 m) au-dessus du seuil. 3 détecteurs entraînés avec des graines différentes (0, 1, 2), chacun pendant 30 époques de 2 000 morceaux de 448 px tirés au hasard, et chacun garde sa **dernière époque**. Part de chaque jeu : ReforesTree 30 %, `segmento` 30 %, Count banana 25 %, `corte` 15 %. Miroirs, quarts de tour, changement d'échelle (0,8 à 1,25), luminosité, contraste et saturation ;
- **DeepForest** : détecteur de boîtes, pré-entraîné sur des couronnes d'arbres. Tuiles de 400 px (les images plus petites sont complétées par du noir pour garder la même échelle), mêmes parts de jeux (tuiles répétées), 10 époques, miroirs et luminosité, écrêtage des gradients (sans lui, la perte devenait NaN dès la 1re époque et le modèle ne détectait plus rien). Les boîtes en double sont fusionnées au-delà d'un recouvrement testé entre 0,2 et 0,5 (par défaut, DeepForest fusionne dès 0,05, ce qui efface des bananiers voisins dans une bananeraie dense).

![Comptage : apprentissage du détecteur de centres](images/comptage_apprentissage_centres.jpg)
*Ce que voit le détecteur de centres pendant l'apprentissage : un morceau tiré au hasard (en haut) et la carte qu'il doit reproduire (en bas).*

Je compare tout sur la moyenne des F1 des 3 validations (Manuel, Count banana, `segmento`) :
- détecteur de centres : je note le modèle 0 seul et la moyenne des 3 modèles, avec **1 vue** (l'image) ou **4 vues** (l'image, ses deux miroirs et un quart de tour ; les 4 cartes sont remises à l'endroit puis moyennées). Les modèles 1 et 2 seuls montrent l'écart dû au hasard ;
- DeepForest : l'époque, le recouvrement de fusion et le seuil sont choisis sur la validation.

#### Résultats (validation)

**Détecteur de centres** (F1 moyenne des 3 validations, chaque modèle à sa dernière époque) :

| Variante | 1 vue | 4 vues |
|---|---|---|
| modèle 0 | 0,781 | **0,800** |
| modèle 1 | 0,762 | 0,784 |
| modèle 2 | 0,769 | 0,785 |
| moyenne des 3 | 0,786 | 0,795 |

- les 4 vues améliorent chaque modèle, de 0,016 à 0,022 ;
- le hasard seul fait varier un modèle entre 0,762 et 0,781 ;
- la moyenne des 3 fait mieux qu'un modèle pris au hasard, mais pas mieux que le modèle 0 avec 4 vues. L'écart (0,005) est plus petit que celui dû au hasard, et un modèle seul est 3 fois plus rapide. Pour la suite, je prends le **modèle 0, 4 vues, seuil 0,30**.

**Comparaison des deux modèles :**

| Validation | Centres : F1 | précision | rappel | compte | DeepForest : F1 | précision | rappel | compte |
|---|---|---|---|---|---|---|---|---|
| Manuel (ferme mélangée) | 0,781 | 0,81 | 0,75 | 123 / 133 (−8 %) | **0,784** | 0,84 | 0,74 | 117 / 133 (−12 %) |
| Count banana (bananeraie) | **0,873** | 0,82 | 0,94 | 697 / 611 (+14 %) | 0,807 | 0,91 | 0,73 | 490 / 611 (−20 %) |
| `segmento` (bananeraie dense) | **0,747** | 0,75 | 0,74 | 1 839 / 1 870 (−2 %) | 0,704 | 0,63 | 0,81 | 2 411 / 1 870 (+29 %) |
| **F1 moyenne** | **0,800** | | | | 0,765 | | | |

DeepForest : meilleure époque 4 (sur 10), fusion 0,3, seuil 0,5. Temps sur une T4 : 12 min par détecteur de centres, 86 min pour DeepForest.

![Comptage : courbes](images/comptage_courbes.jpg)
*À gauche : F1 moyenne à chaque époque (1 vue). DeepForest plafonne dès l'époque 4 ; les détecteurs de centres montent jusqu'à l'époque 10 environ puis oscillent. À droite : F1 moyenne selon le seuil pour les deux modèles.*

![Comptage : prédictions sur la validation](images/comptage_predictions_validation.jpg)
*Un morceau de chaque validation. Vert : détection juste ; croix orange : fausse alerte ; cercle rouge : bananier annoté raté.*

![Comptage : carte du détecteur de centres](images/comptage_carte_centres.jpg)
*Ce que produit le détecteur de centres : l'image, sa carte de probabilité (4 vues), puis les sommets retenus comparés aux bananiers annotés.*

**Ce que j'en retiens :**
- **le détecteur de centres fait mieux** (0,800 contre 0,765), et c'est lui que je garde. Il est nettement meilleur dans les bananeraies (+0,07 sur Count banana, +0,04 sur `segmento`) et à égalité sur la ferme mélangée ;
- il compte aussi mieux : −8 %, +14 % et −2 %, contre −12 %, −20 % et +29 % pour DeepForest, qui met trop de boîtes dans la bananeraie dense et en oublie dans Count banana ;
- la comparaison est plutôt favorable à DeepForest : son époque est choisie sur la validation (score un peu optimiste), alors que le détecteur de centres garde sa dernière époque, et je n'ai pas testé les 4 vues sur DeepForest ;
- **limite** : chaque validation vient d'un lieu dont une autre partie sert à l'apprentissage (une autre ferme de la même région pour Manuel, d'autres vols du même jeu pour Count banana, une autre zone de la même photo pour `segmento`). Sur une plantation entièrement nouvelle (autre pays, autre drone, autre lumière), le score sera probablement plus bas ; c'est surtout vrai pour `segmento`. Le test (Nestor et la zone de test de `segmento`) a la même limite.

#### Analyse des erreurs (validation)

Pour savoir quoi améliorer, j'ai classé toutes les erreurs du détecteur de centres selon la distance au bananier annoté le plus proche (fausses alertes) ou au sommet le plus proche (bananiers ratés), puis regardé des exemples.

| Fausses alertes | Count banana | Manuel | `segmento` |
|---|---|---|---|
| à 1-2 m d'un bananier annoté | 76 | 16 | 70 |
| aucun bananier annoté à moins de 2 m | 46 | 7 | 383 |
| doublon (bananier déjà trouvé à moins de 1 m) | 4 | 0 | 1 |
| **total** | **126** | **23** | **454** |

| Bananiers ratés | Count banana | Manuel | `segmento` |
|---|---|---|---|
| un sommet existe à moins de 1 m, mais sous le seuil de 0,30 | 22 | 5 | 347 |
| sommet placé à 1-2 m du centre annoté | 14 | 16 | 121 |
| aucun sommet à moins de 2 m | 0 | 10 | 16 |
| sommet proche, mais déjà attribué à un voisin | 4 | 2 | 1 |
| **total** | **40** | **33** | **485** |

- **Bords des images de Count banana** : 30 % de leur surface est à moins de 1 m du bord, mais seulement 14 % des bananiers annotés s'y trouvent (les plants coupés par le bord sont souvent non annotés). Or 48 % des fausses alertes sont près du bord. Une bonne partie du +14 % de Count banana vient donc du découpage des photos en morceaux de 12 m ; sur une orthomosaïque entière, ce cas n'existe qu'au bord des parcelles.
- **Grandes couronnes (Manuel)** : le modèle trouve le plant, mais place son sommet à 1-2 m du centre annoté, ou en met deux sur la même couronne.
- **Image floue et couronnes imbriquées (`segmento`)** : la plupart des ratés ont un sommet trop faible juste au bon endroit.
- **Petits plants et touffes (Count banana)** : les ratés sont plus petits que la moyenne (couronne médiane de 2,2 m contre 2,9 m), et 10 % d'entre eux ont un autre bananier annoté à moins de 1 m, contre 2 % pour l'ensemble.

![Comptage : erreurs typiques](images/comptage_erreurs.jpg)
*Exemples tirés au hasard dans les trois cas principaux.*

**Réglages après le modèle** (modèle 0, sans réapprentissage ; F1 moyenne, seuil rechoisi à chaque fois) :

| Vues | fenêtre de 1 m | 1,4 m (actuel) | 1,8 m |
|---|---|---|---|
| 1 vue | 0,779 | 0,782 | 0,785 |
| **4 vues (actuel)** | 0,799 | **0,800** | 0,800 |
| 8 vues (4 quarts de tour, avec ou sans miroir) | 0,791 | 0,791 | 0,792 |

8 vues n'apportent rien de plus que 4, et la fenêtre des sommets ne change presque rien : **je garde 4 vues et 1,4 m**. La fenêtre est un carré de cases de 20 cm (5, 7 ou 9 cases) : un sommet doit être le plus fort de ce carré centré sur lui. Deux sommets peuvent donc être à 0,6 m, 0,8 m ou 1 m l'un de l'autre selon la fenêtre ; avec 1,4 m, un rejet à plus de 0,8 m de sa mère peut former son propre sommet et être compté. Les gains éventuels sont donc à chercher dans l'apprentissage.

**Surapprentissage ?** Même modèle (modèle 0, 4 vues, seuil 0,30), noté sur ses propres images d'apprentissage :

| Jeu | F1 sur l'apprentissage | F1 sur la validation | Couronne médiane (apprentissage / validation) |
|---|---|---|---|
| ReforesTree | 0,974 | 0,781 (Manuel) | 2,96 m / 3,14 m |
| Count banana | 0,908 | 0,873 | 2,96 m / 2,90 m |
| `segmento` | 0,898 | 0,747 | 2,75 m / **5,00 m** |
| `corte` | 0,943 | pas de validation | 2,96 m |

- Count banana : petit écart, le modèle généralise bien d'un vol à l'autre ;
- ReforesTree : grand écart. Les 4 fermes d'apprentissage ne sont que 4 images (621 bananiers), revues très souvent (30 % des morceaux) : le modèle les connaît presque par cœur ;
- `segmento` : une partie de l'écart vient de la zone de validation elle-même, où les couronnes annotées sont presque deux fois plus grandes que dans la zone d'apprentissage (même photo, même échelle) ;
- la F1 de validation ne baisse pas au fil des époques (courbes plus haut) : cet écart ne dégrade pas la validation, mais il montre que le modèle a appris des détails propres aux images d'apprentissage ;
- Manuel ne compte que 133 bananiers : sa F1 est précise à environ ± 0,04 près.

**Robustesse à l'échelle.** Le modèle attend des images à 5 cm par pixel ; pour les jeux Roboflow, cette échelle est estimée, et une vraie plantation peut avoir des plants plus grands ou plus petits. J'ai réduit ou agrandi les images de validation (et les positions des bananiers) avant de les donner au modèle (4 vues, seuil 0,30 inchangé) :

| Facteur | Les bananiers paraissent | F1 moyenne | Manuel | Count banana | `segmento` | Compte (Manuel / Count / `segmento`) |
|---|---|---|---|---|---|---|
| × 0,67 | 1,5 fois plus petits | 0,797 | 0,768 | 0,833 | 0,790 | −32 % / −15 % / +13 % |
| × 0,8 | 1,25 fois plus petits | 0,812 | 0,790 | 0,869 | 0,776 | −21 % / −5 % / +3 % |
| **× 1** | **comme prévu** | **0,800** | 0,781 | 0,873 | 0,747 | −8 % / +14 % / −2 % |
| × 1,25 | 1,25 fois plus grands | 0,767 | 0,731 | 0,884 | 0,686 | −13 % / 0 % / −2 % |
| × 1,5 | 1,5 fois plus grands | 0,693 | 0,651 | 0,851 | 0,575 | −4 % / −5 % / −19 % |

- le modèle supporte bien des bananiers plus petits que prévu, mais perd beaucoup quand ils sont plus grands (−0,11 à × 1,5) ;
- `segmento` gagne même à être réduit : sa zone de validation a des couronnes de 5 m, contre 2,75 m dans sa zone d'apprentissage. C'est cohérent avec l'écart apprentissage / validation plus haut ;
- le compte est plus sensible que la F1 à l'échelle (−32 % sur Manuel à × 0,67) ;
- conséquence pour `run.py` : il faut remettre chaque orthomosaïque à 5 cm à partir de sa vraie taille de pixel, et les plantations à très grandes couronnes seront le cas le plus difficile. La variante « conditions réelles » (zoom de 0,67 à 1,5 pendant l'apprentissage) vise ce point.

#### Variantes d'apprentissage (`entrainement/comptage_v2_variantes_kaggle.ipynb`, 04/11/2026)

Un seul lancement (environ 3 h sur une T4) pour vérifier le détecteur de centres de A à Z :
- **5 variantes, 2 graines chacune** : `reference` (les réglages actuels, réentraînés dans le même lancement pour comparer à égalité) ; **conditions réelles** (flou de rayon 0,5 à 2 px et recompression JPEG de qualité 30 à 90 au hasard, zoom de 0,67 à 1,5 au lieu de 0,8 à 1,25), visée : images floues, petits plants, autre résolution ; **bosse étroite** (bosse cible = diamètre / 10 au lieu de / 6), visée : sommets décalés sur les grandes couronnes ; **les deux** ; **ReforesTree à 15 %** des morceaux au lieu de 30 %, visée : la mémorisation des 4 fermes ;
- **3 modèles entraînés sans un jeu entier** (1 graine) : sans ReforesTree, sans Count banana, sans Plants-bana. La validation du jeu retiré devient un type de plantation jamais vu ;
- **surapprentissage** : toutes les 5 époques, F1 sur une partie des images d'apprentissage (les 4 fermes ReforesTree, 60 images Count banana, 10 `segmento`, 20 `corte`), au même seuil que la validation.

Le hasard seul fait varier un modèle d'environ 0,02 : un écart de moins de 0,01 sur la moyenne des 2 graines ne veut rien dire.

**Variantes** (F1 sur la validation, 4 vues) :

| Variante | Graine 0 | Graine 1 | Moyenne | Manuel | Count banana | `segmento` |
|---|---|---|---|---|---|---|
| référence | 0,786 | 0,784 | 0,785 | 0,752 | 0,864 | 0,740 |
| **conditions réelles** | 0,795 | 0,800 | **0,798** | 0,760 | 0,879 | 0,753 |
| bosse étroite | 0,783 | 0,773 | 0,778 | 0,742 | 0,863 | 0,728 |
| les deux | 0,789 | 0,789 | 0,789 | 0,756 | 0,866 | 0,746 |
| ReforesTree à 15 % | 0,793 | 0,788 | 0,790 | 0,770 | 0,859 | 0,742 |

- **conditions réelles** progresse sur les trois validations à la fois, avec ses deux graines (+0,013 sur la moyenne) : c'est la seule variante qui se détache ;
- **bosse étroite** n'aide pas : les sommets décalés ne viennent pas de la largeur de la cible ;
- **ReforesTree à 15 %** fait monter Manuel (0,770 contre 0,752) et baisser un peu la mémorisation des fermes, mais pas assez au total ;
- le modèle 0 du lancement précédent (0,800) avait eu un peu de chance : la même recette donne ici 0,786 et 0,784.

**Surapprentissage** (F1 sur l'apprentissage et sur la validation, 4 vues, moyenne des 2 graines) :

| Variante | Apprentissage | Validation | Écart | ReforesTree apprentissage / Manuel |
|---|---|---|---|---|
| référence | 0,933 | 0,785 | 0,148 | 0,975 / 0,752 |
| conditions réelles | 0,921 | 0,798 | **0,124** | 0,965 / 0,760 |
| les deux | 0,900 | 0,789 | 0,111 | 0,960 / 0,756 |
| ReforesTree à 15 % | 0,930 | 0,790 | 0,139 | **0,950 / 0,770** |

- la F1 sur l'apprentissage monte jusqu'à la dernière époque, alors que la validation plafonne vers l'époque 10 sans jamais redescendre : le modèle continue d'apprendre ses images sans mieux généraliser, mais ce n'est pas un surapprentissage qui dégrade la validation ;
- les augmentations « conditions réelles » réduisent l'écart (0,124 contre 0,148) tout en améliorant la validation ;
- l'écart vient surtout de ReforesTree (4 fermes, presque apprises par cœur) et de `segmento` (zone de validation aux couronnes deux fois plus grandes) ; il est faible sur Count banana (0,02 à 0,04).

**Sur un type de plantation jamais vu** (1 graine, 4 vues) :

| Jeu retiré de l'apprentissage | F1 sur ce jeu, jamais vu | F1 quand il est appris (référence) | Perte |
|---|---|---|---|
| Count banana (bananeraie, plants séparés) | 0,831 | 0,864 | −0,03 |
| ReforesTree (fermes mélangées) | 0,577 (Manuel) | 0,752 | −0,18 |
| Plants-bana (bananeraie dense et floue) | 0,504 (`segmento`) | 0,740 | −0,24 |

![Comptage : vérification](images/comptage_verification.jpg)
*À gauche : F1 sur l'apprentissage (tirets) et sur la validation (traits pleins) au fil des époques. À droite : F1 sur un type de plantation vu ou jamais vu à l'apprentissage.*

- une bananeraie aux plants bien séparés se reconnaît même sans exemple de ce jeu (−0,03) ;
- une ferme mélangée ou une bananeraie dense et floue jamais vues font chuter le score de 0,18 à 0,24 : **chaque type de plantation doit être représenté à l'apprentissage**. Sur une plantation d'un type nouveau, il faut s'attendre à un score nettement plus bas que celui de la validation, et le dire dans le bulletin ;
- sans Plants-bana, le modèle n'a plus que 4 fermes et des plants séparés : sa perte d'apprentissage descend très bas (0,19 contre 0,45) et il apprend ses images par cœur (0,98 sur les fermes d'apprentissage).

**La suite part de la variante conditions réelles**, en moyenne des 2 modèles, 4 vues : 0,803 au seuil 0,25 (Manuel 0,769, Count banana 0,879, `segmento` 0,761 ; compte +2 %, +14 %, +11 %). Le modèle 0 seul fait 0,795. Le seuil passe ensuite à 0,30 (voir « Le seuil »). Les poids sont `centres_conditions_reelles_0.pt` et `centres_conditions_reelles_1.pt`. Je les ai rechargés hors de Kaggle et j'obtiens exactement les mêmes scores (0,803 au seuil 0,25).

**Les autres arbres sont-ils pris pour des bananiers ?** Dans la parcelle de Manuel, 57 arbres annotés ne sont pas des bananiers (cacaoyers et autres arbres de la ferme). **Aucun** n'a de détection à moins de 1 m, et aucune des 30 fausses alertes de Manuel ne tombe sur l'un d'eux. Le modèle ne confond donc pas les autres arbres de ces fermes avec des bananiers. Les palmiers, eux, sont un vrai problème (essais v5 et v6, plus bas).

**Plusieurs échelles à la prédiction ?** Prédire aussi sur l'image réduite (× 0,8, × 0,67) et faire la moyenne des cartes n'apporte rien (0,797 à 0,805 contre 0,803) : le zoom de 0,67 à 1,5 pendant l'apprentissage rend déjà le modèle robuste à la taille des plants. Je garde une seule échelle.

#### Modèle final v2 et contrôles (05/11/2026)

**Fichiers plus légers.** Les poids sont enregistrés en 16 bits au lieu de 32 : 23 Mo par fichier au lieu de 45. Sur la validation, une seule détection change sur 2 906 (0,8029 contre 0,8030). Le code de run.py (`comptage/detecteur.py`) redonne exactement les scores du notebook, y compris sur Manuel, qu'il découpe en morceaux de 2 048 px.

**Réentraînement final** (`entrainement/comptage_v2_final_kaggle.ipynb`). Je réentraîne les 2 modèles avec la même recette sur apprentissage + validation : une 5e ferme ReforesTree, les 4 vols de Count banana, deux zones de `segmento`. Le manque de types de plantation est la principale limite : je préfère montrer au modèle livré le plus de données possible.

**Garde-fou.** Noté sur ses 518 images apprises, le modèle réentraîné fait 0,901, contre 0,903 pour le modèle de validation. L'écart est dix fois plus petit que la variation due au hasard d'un entraînement, et sur la validation, qu'il a apprise, il passe de 0,803 à 0,864 : l'entraînement n'a pas raté. En apprenant plus d'images avec le même nombre de pas, le modèle revoit simplement moins souvent chacune. Sur deux plantations qu'aucun des deux n'a vues (plus bas), il fait 0,761 contre 0,737 (0,725 contre 0,727 avec la première préparation de Plantain, voir « Plantations jamais vues »). Je garde le modèle réentraîné, en 16 bits (0,8643 contre 0,8642 en 32 bits).

#### Le seuil : équilibrer fausses alertes et oublis

L'organisation note la F1, mais aussi l'erreur de comptage par parcelle. Le mentor le dit simplement : sur 1 000 bananiers, en compter 50 de trop et en rater 60 donne un bon total. Le seuil de 0,25 maximisait la F1 seule, et le modèle comptait trop (+2 %, +14 %, +11 %). Je choisis donc le seuil autrement, sur la validation : parmi les seuils dont la F1 moyenne est à moins de 0,005 de la meilleure, je prends celui où fausses alertes et oublis sont le plus proches. C'est le « seuil d'équilibre » de tous les essais suivants.

![Comptage : choix du seuil](images/comptage_seuil.jpg)
*Fausses alertes et oublis sur les 3 jeux de validation, selon le seuil (modèle de validation, 4 vues).*

| Seuil | F1 moyenne | Compte (Manuel / Count banana / `segmento`) |
|---|---|---|
| 0,25 | **0,803** | +2 % / +14 % / +11 % |
| **0,30** | 0,800 | −2 % / +9 % / −5 % |
| 0,35 | 0,789 | −10 % / 0 % / −18 % |

**Seuil retenu : 0,30.** La F1 ne perd que 0,003, et l'erreur de compte moyenne passe de 8,8 % à 5,0 %. Pour les maladies, je fais l'inverse : une feuille malade ratée est plus grave qu'une fausse alerte.

#### Ce que le modèle supporte : taille des plants et résolution

![Comptage : taille et résolution](images/comptage_taille_resolution.jpg)
*À gauche : images de validation agrandies ou réduites avant de les donner au modèle. À droite : images réduites à 10 ou 20 cm par pixel, puis remises à 5 cm, comme un vol plus haut.*

- **La taille des plants n'est plus un gros risque.** Le zoom de 0,67 à 1,5 pendant l'apprentissage a rendu le modèle robuste : avec des bananiers 1,5 fois plus grands que prévu, il perd 1,6 point, contre 10,7 pour l'ancien modèle. Il ne décroche vraiment que si les plants paraissent deux fois plus petits (× 0,5).
- **Une image moins fine coûte peu** : moins de 3 points à 20 cm par pixel, soit un vol 4 fois plus haut.

#### Plantations jamais vues

Pour savoir ce que vaut le modèle sur un type de plantation qu'il n'a jamais appris, j'utilise deux petits jeux décrits dans le carnet : Detecting Plantain Crops (jeunes plantains dans la brousse) et Banano Finca (jeunes plants vus de très bas). Je ne les utilise jamais à l'apprentissage. Je m'en sers pour comparer les modèles entre eux, plusieurs fois : leurs scores ne valent donc pas un test. Je les appelle « les juges ».

| Jeu jamais vu (seuil 0,25) | Modèle réentraîné | Modèle de validation |
|---|---|---|
| Detecting Plantain Crops (19 images, 208 plantains) | 0,741 (compte +4 %) | 0,692 (−8 %) |
| Banano Finca (25 images, 409 plants) | 0,780 (compte −24 %) | 0,781 (−27 %) |
| **Moyenne** | **0,761** | **0,737** |

Ces chiffres sont calculés par `outils/evaluer_comptage.py`, qui prépare les deux jeux comme le décrit le carnet et enregistre toutes les détections. Une première notation, faite avec une préparation de Plantain que je n'ai pas conservée, donnait 0,670 et 0,673 sur Plantain (moyennes 0,725 et 0,727) ; Banano Finca était identique. Je garde les chiffres du script, que l'on peut relancer.

![Comptage : plantations jamais vues](images/comptage_plantations_inconnues.jpg)
*Modèle réentraîné sur une image de chaque jeu. Cercle vert : détection juste ; croix orange : fausse alerte ; cercle rouge : plant annoté raté.*

Sur un type de plantation jamais vu, la F1 tombe vers 0,74 à 0,76, contre 0,80 sur la validation. Sur les jeunes plants, le modèle en oublie environ un sur trois. Ces deux jeux sont petits et viennent chacun d'une seule scène : leur F1 est imprécise à quelques points près.

#### Deux idées écartées : choix automatique du zoom et AdaBN

Deux idées qui ne changent que la façon de prédire, sans réentraîner :
- **choix automatique du zoom** : sur une zone, le modèle prédit avec plusieurs zooms (× 0,5 à × 2) et garde celui où ses détections sont les plus sûres d'elles. Une moyenne de plusieurs zooms n'apportait rien ; ici, on en choisit un ;
- **AdaBN** : les moyennes que le ResNet-18 a retenues des images d'apprentissage sont recalculées sur les images de la nouvelle zone, sans étiquette.

| | Plantain | Banano Finca | Moyenne | Validation |
|---|---|---|---|---|
| sans idée | 0,670 | 0,779 | 0,725 | |
| zoom automatique | 0,698 | 0,765 | 0,732 | `segmento` −0,027 |
| AdaBN | 0,593 | 0,589 | 0,591 | −0,07 à −0,10 sur chaque jeu |

**Les deux sont écartées.** La confiance du modèle ne désigne pas le bon zoom, et recalculer la normalisation sur quelques images d'une seule scène abîme nettement le modèle. Ces chiffres viennent de la première préparation de Plantain (voir plus haut) ; les deux idées font de toute façon perdre plus de 0,02 sur un jeu de validation.

### v3 : trois jeux de jeunes plantations et de plantain (05/11/2026)

Le modèle rate surtout les plantations qu'il n'a jamais vues : il faut lui en montrer. J'ajoute trois jeux Roboflow, à l'apprentissage seulement (carnet : jeunes_rangs, plantain, jeunes_lasuiza) : des jeunes plants en rangs, du plantain et une autre jeune plantation. Même recette, 2 graines ; les nouveaux jeux font 30 % des morceaux, les anciens gardent leurs proportions entre eux. Chaque modèle est noté à son seuil d'équilibre.

**Résultat** (essai : `entrainement/comptage_v3_essai_kaggle.ipynb` ; notation sur CPU : `outils/evaluer_comptage.py`, détections et scores enregistrés) :

| | Modèle v2 (seuil 0,30) | Avec les nouveaux jeux (seuil 0,25) | Écart |
|---|---|---|---|
| Detecting Plantain Crops | 0,684 | **0,771** | +0,087 |
| Banano Finca | 0,745 | 0,750 | +0,005 |
| **Moyenne des plantations jamais vues** | 0,715 | **0,761** | **+0,046** |
| Manuel | 0,765 | 0,780 | +0,015 |
| Count banana | 0,880 | 0,860 | −0,020 |
| `segmento` | 0,754 | 0,758 | +0,004 |
| F1 moyenne de validation | 0,800 | 0,800 | 0 |

- **Les nouveaux jeux font gagner +0,046 sur les plantations jamais vues**, sans qu'aucun jeu de validation ne perde plus de 0,02 (Count banana est juste à la limite, −0,0199) ;
- le gain vient du plantain, un type que le modèle a maintenant appris. Sur les jeunes plants de Banano Finca, presque rien ne change : il en oublie toujours plus d'un sur trois (rappel 0,63). Les jeunes plantations restent le point faible ;
- avec les nouveaux jeux, le seuil d'équilibre descend à 0,25 (à 0,30, la F1 moyenne de validation perd 0,0053, plus que les 0,005 permis) ;
- **le comptage n'évolue pas dans le même sens partout.** Sur la validation, le nouveau modèle compte plus mal : −1,5 %, +18,8 % et +13,9 %, soit 11,4 % d'erreur moyenne contre 5,0 % pour le modèle v2. Sur les plantations jamais vues, il compte mieux : −6,7 % et −32,5 %, soit 19,6 % contre 26,0 % ;
- **vérifications après coup**, pour voir si le gain tient au seuil, au hasard ou à l'échelle supposée du plantain :
    - à seuil égal (0,25 pour les deux, le seuil de meilleure F1 du modèle v2), l'ancien modèle fait un peu mieux sur la validation (0,803 contre 0,800), et le gain sur les plantations jamais vues tombe à +0,024 ;
    - en tirant les images au hasard (5 000 tirages avec remise, par image), le gain moyen reste positif dans plus de 99,9 % des tirages (intervalle à 95 % : +0,017 à +0,077). Il est porté par le plantain (+0,036 à +0,145) ; sur Banano Finca, l'écart ne se distingue pas de zéro (−0,021 à +0,031) ;
    - la taille du pixel du plantain est inconnue. Avec des plants supposés 0,7 à 1,4 fois plus grands qu'estimé (option `--facteur-plantain`), le nouveau modèle reste devant sur le plantain à chaque fois, de +0,05 à +0,09, et le gain moyen sur les deux juges va de +0,026 à +0,046 : le résultat ne dépend pas de cette hypothèse ;
- **une première notation** de cet essai, faite avec une préparation de Plantain que je n'ai pas conservée, donnait 0,686 contre 0,745 sur Plantain (+0,032 en moyenne). La validation et Banano Finca sont identiques ; seul Plantain change, parce que son échelle doit être supposée. Les deux notations vont dans le même sens ; je garde les chiffres du script, que l'on peut relancer ;
- en résumé : le nouveau modèle n'est pas meilleur sur les types qu'il connaissait déjà. Il est meilleur sur le plantain, qu'il ne connaissait pas.

Fichiers : poids de l'essai et tableaux du notebook dans `C:\Field_scan\modeles\comptage_v3_essai\`, notation (détections, scores, bilans, un dossier par facteur d'échelle) dans `evaluation_juges\`.

**Le seuil vu sur toutes les plantations.** Le seuil d'équilibre se choisit sur la validation, donc sur des plantations du type appris. Or le test de l'organisation mélangera des types de plantations, et ce qui compte au comptage, c'est que fausses alertes et oublis s'équilibrent sur l'ensemble. J'ai donc regardé l'erreur de comptage selon le seuil sur les 5 jeux à la fois (`evaluation_juges\facteur_1.0\scores.csv`) :

| Nouveau modèle, seuil | F1 validation | F1 plantations jamais vues | Compte : Manuel / Count banana / `segmento` | Compte : Plantain / Banano Finca | Erreur de compte moyenne sur les 5 jeux |
|---|---|---|---|---|---|
| 0,20 | 0,781 | 0,771 | +10 % / +30 % / +35 % | +18 % / −23 % | 23,2 % |
| **0,25** | **0,799** | **0,761** | −2 % / +19 % / +14 % | −7 % / −33 % | **14,7 %** |
| 0,30 | 0,794 | 0,724 | −8 % / +8 % / −5 % | −21 % / −42 % | 16,8 % |
| 0,35 | 0,779 | 0,665 | −17 % / −1 % / −22 % | −34 % / −51 % | 25,2 % |

- sur les plantations du type appris, 0,25 compte trop (+19 % sur Count banana) : sur la validation seule, 0,30 compterait mieux ;
- sur les plantations jamais vues, c'est l'inverse : le modèle oublie des plants, surtout les jeunes (−33 % sur Banano Finca), et monter le seuil aggrave ces oublis tout en faisant baisser la F1 ;
- sur les 5 jeux ensemble, 0,25 est le seuil le plus équilibré. C'est aussi vrai pour le modèle v2 : 12,2 % d'erreur moyenne à 0,25, contre 13,4 % à 0,30 et 15,6 % à 0,20 ;
- un coefficient de correction du nombre de pieds ne corrigerait pas les deux défauts à la fois, puisque le modèle compte trop dans les plantations adultes denses et pas assez dans les jeunes. Il faudrait d'abord reconnaître le type de plantation. Je ne le fais pas : je le garde comme perspective.

### Essai v4 : deux bananeraies de plus (05/11/2026)

Les nouveaux jeux de la v3 ont aidé sur le plantain, pas sur les jeunes bananiers de Banano Finca. J'ajoute deux jeux de bananeraies vues par drone (carnet : banavista et jorges) : `banavista` (photos et orthomosaïques de plantations des Philippines, jeunes et adultes) et `jorges` (orthomosaïque d'une jeune bananeraie). Même recette que l'essai v3, 2 graines (`entrainement/comptage_v4_essai_kaggle.ipynb`) ; `banavista` fait 15 % des morceaux et `jorges` 5 %, les parts des jeux de la v3 sont multipliées par 0,8. Je le compare à l'essai v3, chacun à son seuil d'équilibre.

**Résultat** (notation par `outils/evaluer_comptage.py` ; poids, tableaux du notebook et notation dans `C:\Field_scan\modeles\comptage_v4_essai\`) :

| | Essai v3 (seuil 0,25) | Essai v4 (seuil 0,30) | Écart |
|---|---|---|---|
| Detecting Plantain Crops | **0,771** | 0,636 | −0,135 |
| Banano Finca | 0,750 | 0,763 | +0,013 |
| **Moyenne des plantations jamais vues** | **0,761** | 0,699 | **−0,062** |
| Manuel | 0,780 | 0,797 | +0,017 |
| Count banana | 0,860 | 0,885 | +0,025 |
| `segmento` | 0,758 | 0,762 | +0,004 |
| F1 moyenne de validation | 0,799 | **0,815** | +0,016 |
| Erreur de compte moyenne sur les 5 jeux | **14,7 %** | 18,8 % | |

- **l'essai v4 perd 0,062 sur les plantations jamais vues.** En tirant les images au hasard (5 000 tirages), l'écart moyen est négatif dans tous les tirages (−0,096 à −0,022) ;
- v4 est meilleur sur la validation (+0,016), c'est-à-dire sur les types de plantation déjà appris, et un peu meilleur sur les jeunes bananiers de Banano Finca (rappel 0,66 contre 0,63). Mais il perd beaucoup sur le plantain (−0,135) : à son seuil d'équilibre (0,30), il en oublie près de la moitié (compte −46 %). Même au seuil de 0,25, il resterait sous v3 sur les plantations jamais vues (0,752 contre 0,761) ;
- explication probable, que je n'ai pas vérifiée : les deux jeux ajoutés sont des bananeraies, et le plantain ne fait plus que 9,6 % des morceaux (12 % en v3). Le modèle reconnaît mieux les couronnes de bananier, moins bien les jeunes plantains épars ;
- sur ses propres images d'apprentissage, l'essai v4 fait 0,83 sur `banavista` et 0,81 sur `jorges` : ces jeux sont bien appris. `jeunes_lasuiza` reste mal appris (0,67 à 0,71), comme en v3 ;
- je garde donc la v3. `banavista` et `jorges` ne sont dans aucun modèle installé.

### Final v3 (05/11/2026)

Le réentraînement final de la v3 (`entrainement/comptage_v3_final_kaggle.ipynb`) apprend sur apprentissage + validation, avec 2 graines. Sur des images apprises par l'essai et par le final, il fait aussi bien que l'essai (+0,001). Ses poids sont enregistrés en 16 bits. Il a été le modèle de comptage installé du 05/11 au 10/11, au seuil de 0,25, le seuil d'équilibre de l'essai v3.

### Essai v5 : des palmiers « rien à trouver » (05/11/2026)

Les palmiers à huile ressemblent à de grands bananiers vus du dessus. Sur une zone de palmiers jamais apprise (40 tuiles de la zone 52000_20000, 32,3 ha, aucun bananier), l'essai v3 voit **34,8 faux bananiers par hectare**. Pour l'essai v5, j'ajoute à l'apprentissage 320 tuiles de palmiers sans aucun bananier (carnet : palmiers à huile), dans 10 % des morceaux : le modèle y apprend qu'il n'y a rien à trouver. Même recette que la v3, 2 graines.

| Seuil 0,25 | Essai v3 | Essai v5 |
|---|---|---|
| Manuel | 0,780 | 0,765 |
| Count banana | 0,860 | 0,873 |
| `segmento` | 0,758 | 0,762 |
| Detecting Plantain Crops | 0,771 | 0,739 |
| Banano Finca | 0,750 | 0,720 |
| Moyenne des juges | 0,761 | 0,730 |
| Faux bananiers par hectare sur les palmiers | 34,8 | **1,3** |

- les faux bananiers sur les palmiers baissent de 96 % ;
- mais le modèle devient prudent : il perd 0,031 sur les juges (intervalle à 95 % : −0,047 à −0,016), sur Plantain (−0,060 à −0,007) comme sur Banano Finca (−0,045 à −0,014). Il oublie plus de jeunes plants (compte de Banano Finca : −36,4 % contre −32,5 %) ;
- la validation ne bouge presque pas (0,800 contre 0,799).

### Essai v6 : témoin, zoom élargi, palmiers dosés (07/11/2026)

Quatre variantes de 2 graines dans un seul lancement (`entrainement/comptage_v6_essai_kaggle.ipynb`) :
- `temoin` : la recette de l'essai v3 à l'identique, avec deux autres graines (2 et 3). Il mesure le hasard de l'entraînement ;
- `zoom2` : zoom des morceaux de 0,67 à 2,0 au lieu de 1,5. Le réseau voit des couronnes de 5 m ou plus dans environ 15 % des morceaux, au lieu de 2,8 % : c'est la taille des plants de `segmento` en validation ;
- `palm3` : les palmiers de l'essai v5, mais dans 3 % des morceaux au lieu de 10 %. Les parts des jeux de la v3 sont multipliées par 0,97 ;
- `zoom2_palm3` : les deux changements ensemble.

Notation par `outils/evaluer_comptage.py`, chaque variante à son seuil d'équilibre (0,25 pour toutes) :

| | Essai v3 | temoin | zoom2 | palm3 | zoom2_palm3 |
|---|---|---|---|---|---|
| Manuel | 0,780 | 0,789 | 0,794 | 0,773 | 0,783 |
| Count banana | 0,860 | 0,868 | 0,870 | 0,870 | 0,864 |
| `segmento` | 0,758 | 0,753 | 0,758 | 0,757 | 0,761 |
| **Moyenne de validation** | 0,799 | 0,804 | 0,807 | 0,800 | 0,803 |
| Detecting Plantain Crops | 0,771 | 0,741 | 0,754 | 0,770 | 0,749 |
| Banano Finca | 0,750 | 0,736 | 0,723 | 0,747 | 0,726 |
| **Moyenne des juges** | 0,761 | 0,738 | 0,739 | 0,758 | 0,738 |
| Erreur de compte moyenne (5 jeux, en valeur absolue) | 14,7 % | 15,2 % | 14,6 % | 12,8 % | 13,3 % |
| **Faux bananiers par hectare sur les palmiers** | 34,8 | 32,2 | 39,3 | **1,1** | 3,4 |

Écarts avec l'essai v3, mesurés sur les mêmes images tirées au hasard (bootstrap apparié, 2 000 tirages ; `C:\Field_scan\modeles\comptage_v6_essai\intervalles\`) :
- **le hasard d'un entraînement pèse environ 0,02** : le témoin, même recette que la v3 avec d'autres graines, perd 0,022 sur les juges (−0,050 à +0,004) ;
- **aucune variante ne monte la F1 au-delà de ce hasard**, ni sur la validation ni sur les juges ;
- `zoom2` ajoute des faux bananiers sur les palmiers (+4,5 par hectare, de +2,6 à +6,6) et perd sur les juges (−0,022, de −0,039 à −0,005) ;
- `palm3` garde la F1 de la v3 (validation +0,000, de −0,009 à +0,010 ; juges −0,002, de −0,019 à +0,014) et fait presque disparaître les faux bananiers sur les palmiers (−33,7 par hectare, de −40,8 à −26,4). C'est aussi celle qui compte le plus juste sur les 5 jeux ;
- `zoom2_palm3` perd sur les juges (−0,023, de −0,043 à −0,002).

### Sans apprentissage : une méthode mathématique (07/11/2026)

Une piste du tuteur : trouver les centres des bananiers avec des formules, sans apprentissage. Toutes mes méthodes partent de l'axe de chaque feuille (tenseur de structure, sur les pixels verts), puis en tirent des centres de façons différentes : votes le long des axes (A), modèle de plante en étoile (B), séparation des couronnes (C), modèle objet avec une énergie à minimiser (D), flot vers le cœur (E), feuilles orientées vers leur cœur (F). La méthode finale, M, reprend A avec les votes orientés de F. J'ai aussi essayé de combiner les maths avec le réseau v3, sans le réentraîner (familles H1, H2 et V3P). Tous les paramètres sont des longueurs en mètres. Détail : `C:\Field_scan\recherche_maths\final\SYNTHESE.md`.

Notation par `outils/evaluer_comptage.py`, chaque méthode à son seuil d'équilibre :

| Méthode (seuil) | Moyenne de validation | Moyenne des juges | Faux bananiers par ha (palmiers) | Erreur de compte, validation | Erreur de compte, juges |
|---|---|---|---|---|---|
| **v3 (0,25)** | **0,799** | **0,761** | 34,8 | 11,4 % | 19,6 % |
| M (0,60) | 0,679 | 0,541 | 22,3 | 11,1 % | 30,8 % |
| F (0,55) | 0,707 | 0,554 | 82,8 | 22,8 % | 25,1 % |
| H2_comb_seuil (0,60) | 0,807 | 0,760 | 24,6 | 13,9 % | 19,1 % |
| V3P_image (0,25) | 0,799 | 0,761 | 21,9 | 11,4 % | 19,6 % |

- **les maths seules restent loin du réseau** : les 13 méthodes font de 0,602 à 0,707 en validation et de 0,509 à 0,609 sur les juges. Elles sont nettement sous le v3 sur 63 cases sur 65 (13 méthodes × 5 jeux) ;
- **aucun hybride n'améliore la F1 au-delà du hasard** : de −0,004 à +0,007 en validation. Les deux gains sur Count banana ne tiennent plus après la correction de Holm, qui tient compte des 312 comparaisons faites ;
- **ce que les maths apportent vraiment** : moins de faux bananiers sur les palmiers. V3P_image retire tous les pics d'une image qui « préfère » des couronnes de 8,9 m : −13,0 faux bananiers par hectare (de −18,4 à −8,2, soit −37 %), avec une F1 identique au v3 sur les 5 jeux de bananiers. Un palmier se distingue d'un bananier par sa taille : une feuille de bananier s'arrête vers 2,9 m du cœur, une palme fait 4 à 5,5 m ;
- M compte plus juste que le v3 sur Count banana (+1,3 % contre +18,8 %) et sur `segmento` (−1,2 % contre +13,9 %), mais sous-compte Manuel (−30,8 %) et surcompte Plantain (+34,6 %) ;
- ce travail explique aussi des erreurs du réseau : la validation de `segmento` a des couronnes de 5,0 m, contre 2,75 m dans sa zone d'apprentissage, et le réseau ne voyait une couronne de 5 m ou plus que dans 2,8 % des morceaux. C'est l'origine de la variante `zoom2` ;
- contre les palmiers, l'apprentissage fait finalement mieux : 1,3 faux bananier par hectare pour l'essai v5, 1,1 pour palm3 ;
- un incident est écrit dans la synthèse : pendant le réglage d'une méthode, une commande a affiché une fois, sans filtre, les noms et diamètres médians d'images de `segmento` hors réglage, dont des images de test. Aucune de ces images n'a été ouverte ni notée.

### palm3 final (09/11/2026), installé le 10/11

Recette de la variante `palm3` (parts des jeux de la v3 × 0,97, palmiers dans 3 % des morceaux, zoom de 0,67 à 1,5, flou et JPEG), apprise sur apprentissage + validation, 2 graines (`entrainement/comptage_palm3_final_kaggle.ipynb`).

- **Garde-fou** : sur des images apprises par l'essai et par le final, le final fait 0,860 et l'essai 0,863 (écart −0,003).
- **16 bits** : 1 détection de différence sur 4 113.
- **Face au final v3** (les deux ont appris la validation, donc je ne compare que les juges et les palmiers) :

| | Final v3 | Final palm3 |
|---|---|---|
| Detecting Plantain Crops | 0,781 (compte −0,5 %) | 0,771 (compte +3,4 %) |
| Banano Finca | 0,737 (compte −34,2 %) | 0,737 (compte −34,2 %) |
| Moyenne des juges | 0,759 | 0,754 (écart −0,005, de −0,025 à +0,014) |
| Faux bananiers par hectare sur les palmiers | 23,0 | **1,7** (−93 %) |

palm3 est installé depuis le 10/11 dans `modeles/comptage/`, au seuil de 0,25. Sur Manuel, la chaîne installée redonne exactement les détections du final palm3 : 131 détections, dont 116 justes.

### Le test du 10/11/2026

Le modèle installé (palm3 : 2 réseaux, 4 vues, 16 bits, seuil 0,25), noté une seule fois. Script : `C:\Field_scan\modeles\comptage_test_unique\test_unique.py`.
- **Nestor** : la ferme de test de ReforesTree, une image à 5 cm par pixel, 235 bananiers. L'intervalle vient de blocs de 20 m tirés au hasard (5 000 tirages), parce que les bananiers d'une même image ne sont pas indépendants.
- **`segmento`** : la zone de validation et la zone de test du bas se touchent. Les 4 morceaux de test qui bordent la validation (303 bananiers) sont retirés : un bananier coupé par ce bord aurait été vu à moitié pendant l'apprentissage. Il reste 22 morceaux et 2 498 bananiers. L'intervalle vient de morceaux entiers tirés au hasard.
- **Photos DJI de jeune plantation** (carnet : Plants-bana, photos `DJI_…`) : seuls les plants et les détections à moins de 2 000 px du centre de chaque photo comptent ; les photos sont remises à 5 cm à partir de leur échelle estimée (1,1 cm par pixel). Je note les mères seules, puis tous les plants. Un rejet est « compté » s'il reçoit sa propre détection, c'est-à-dire un pied de plus.

| Jeu | Plants | Détections | Justes | F1 [intervalle à 95 %] | Précision | Rappel | Erreur de compte [intervalle] |
|---|---|---|---|---|---|---|---|
| Nestor (ferme mélangée) | 235 | 282 | 199 | **0,770** [0,723 ; 0,810] | 0,706 | 0,847 | **+20,0 %** [+9,9 ; +29,8] |
| `segmento` (bananeraie dense) | 2 498 | 2 361 | 1 891 | **0,778** [0,758 ; 0,798] | 0,801 | 0,757 | −5,5 % [−10,8 ; −0,2] |

Sur `segmento`, l'erreur de compte moyenne par morceau, en valeur absolue, est de 10,2 %.

| Photo DJI | Notation | Plants | Détections | Justes | F1 | Précision | Rappel | Erreur de compte | Rejets comptés |
|---|---|---|---|---|---|---|---|---|---|
| 0544 (annotée à la main) | mères seules | 119 | 110 | 89 | 0,777 | 0,809 | 0,748 | −7,6 % | 8 sur 16 |
| 0546 (pré-annotée, corrigée) | mères seules | 119 | 104 | 86 | 0,771 | 0,827 | 0,723 | −12,6 % | 2 sur 5 |
| 0548 (pré-annotée, corrigée) | mères seules | 138 | 122 | 98 | 0,754 | 0,803 | 0,710 | −11,6 % | 3 sur 5 |
| **les 3** | **mères seules** | 376 | 336 | 273 | **0,767** | 0,812 | 0,726 | **−10,6 %** | 13 sur 26 |
| les 3 | tous les plants | 402 | 336 | 286 | 0,775 | 0,851 | 0,711 | −16,4 % | 13 sur 26 |

- **La F1 tient sur trois types de plantation très différents** : 0,770, 0,778 et 0,767. C'est proche des juges (0,754) et un peu sous la validation (0,80).
- **Nestor surcompte** : 282 détections pour 235 bananiers. Le rappel est bon (85 %) : l'erreur vient de fausses détections dans cette ferme mélangée. Je n'ai pas vérifié sur quoi elles tombent.
- **`segmento` et les photos DJI sous-comptent un peu** : le modèle y oublie environ un plant sur quatre.
- **Les annotations des photos DJI ne flattent pas le score** : la photo annotée entièrement à la main (0544) fait 0,777, les deux pré-annotées 0,771 et 0,754.
- **Les rejets** : 13 sur 26 reçoivent leur propre détection, et comptent donc comme un pied de plus.
- Même limite que la validation : la zone de test de `segmento` vient de la même grande image que sa zone d'apprentissage, et Nestor est dans la même région que Flora.

## 2. Maladies

### v1 (MobileNetV3)

Apprentissage sur Tanzanie 2023, validation sur Tanzanie 2025 (3 001 images), test sur le Bangladesh (autre pays).

- **Validation** : F1 0,926, meilleure époque : la 3e sur 8. Seuil « je ne sais pas » : 0,679 (confiance en dessous de laquelle le modèle refuse de répondre).
- **Test Bangladesh** : 69 % de bonnes réponses sur les classes connues, rappel de la Sigatoka 66 %, 3 % de fausses alertes sur les feuilles saines. Maladies inconnues (Cordana, Pestalotiopsis) : seulement 20 % rejetées, le modèle les prend le plus souvent pour une maladie connue, avec confiance.

### v2, installée (`entrainement/maladies_v2_kaggle.ipynb`)

Données nettoyées à nouveau (voir le carnet) : apprentissage sur Tanzanie 2023 + Pérou (16 959 images), validation sur toute la Tanzanie 2025 (11 766), test sur le Bangladesh (893). Tous les réglages viennent de la validation ; le Bangladesh n'a servi qu'au test.

**Le modèle :** ConvNeXt-Tiny pré-entraîné sur ImageNet, Pérou tiré 3 fois plus souvent, recompression JPEG et lissage des étiquettes, **moyenne de 3 modèles** entraînés avec des graines différentes, 4 vues.

**Résultat au test (Bangladesh, autre pays, 03/11) :** 95 % des Sigatoka nommées, aucune dite saine ; 1 feuille saine sur 90 signalée malade ; aucune des 331 maladies inconnues dite saine (21 % en « je ne sais pas », les autres signalées malades). Détail en fin de section.

#### Comment le modèle apprend

Pendant l'entraînement, chaque photo est modifiée au hasard à chaque passage : cadrage et zoom, miroirs, rotation (jusqu'à 30°), couleurs et lumière, flou, une zone masquée, et une recompression JPEG plus ou moins forte (pour imiter d'autres téléphones). Le modèle ne voit jamais deux fois exactement la même image, et ne peut pas se fier à un seul détail.

![Modifications aléatoires pendant l'entraînement](images/maladies_augmentations.jpg)
*Une photo d'entraînement et 9 versions tirées au hasard. Les coins noirs viennent de la rotation, le rectangle gris est la zone masquée.*

Autres réglages : 6 époques, modèle de la dernière époque gardé ; lissage des étiquettes 0,1 (la bonne réponse vaut 0,9 au lieu de 1, pour que le modèle soit moins sûr de lui à tort). Chacun des 3 modèles part des mêmes poids ImageNet ; seule la graine du hasard change (départ de la dernière couche, ordre des photos, modifications tirées). Leurs probabilités sont moyennées.

#### Comment le modèle décide

Chaque photo est regardée sous 4 vues ; les probabilités des 3 modèles et des 4 vues sont moyennées.

![Les 4 vues](images/maladies_quatre_vues.jpg)

La décision se fait ensuite en deux étapes :

1. **malade ou saine ?** La feuille est malade si P(Black Sigatoka) + P(Fusarium) dépasse le seuil d'alerte. Une feuille malade ratée est l'erreur la plus grave : le seuil est celui qui rend le plus petit possible *2 × malades ratées + fausses alertes* (une malade ratée compte comme deux fausses alertes) ;
2. **quelle maladie ?** La maladie n'est nommée que si le modèle est assez sûr entre les deux (seuil choisi pour que le nom soit juste 95 % du temps). Sinon : « malade, non identifiée ».

Si la confiance du modèle est trop basse (seuil qui garde 95 % des photos de validation), la réponse est « je ne sais pas ».

| | Modèle sûr de lui | Modèle hésitant |
|---|---|---|
| Malade (étape 1), maladie claire | Black Sigatoka ou Fusarium | je ne sais pas |
| Malade, hésite entre les deux maladies | malade, non identifiée | je ne sais pas |
| Saine (étape 1) | saine | je ne sais pas |

#### Ce que j'ai comparé sur la validation

| Question | Résultat | Dans la v2 |
|---|---|---|
| Comment choisir le seuil d'alerte ? | 0,5 fixe : 167 malades ratées et 105 fausses alertes, contre 96 et 189 au seuil de coût minimal (0,30, même entraînement). F2 : choisit 0,05, le plus bas testé (427 fausses alertes), car avec 2,5 fois plus de malades que de saines elle réagit à peine aux fausses alertes | coût (malade ratée = 2 fausses alertes) |
| Quelle époque garder ? | garder l'époque au coût le plus bas a donné une fois l'époque 1 : un modèle qui repère bien les malades mais ne nomme que 49 % des Fusarium. Le coût saute aussi beaucoup d'une époque à l'autre | la dernière époque |
| 1 vue ou 4 vues ? | 4 vues ont fait mieux dans tous les essais (par exemple coût 392 contre 337 pour EfficientNet-B2) | 4 vues |
| Comment repérer une maladie inconnue ? | voir « Inconnues simulées » | confiance |
| Quel réseau ? | ConvNeXt-Tiny gagne sur tous les critères, avec ou sans le Pérou | ConvNeXt-Tiny |
| Le Pérou aide-t-il ? | le coût baisse avec le Pérou pour les 4 réseaux | Pérou, poids 3 |
| Recompression JPEG et lissage ? | coût égal (213 et 213) ; à coût égal, je garde celui qui rate le moins de malades (36 contre 48) | oui |
| Un modèle ou la moyenne de 3 ? | coût 191 pour le 1er modèle seul, 167 pour la moyenne des 3 | moyenne de 3 |

#### Tous les essais

6 époques, modèle de la dernière époque, mêmes réglages sauf ce qui est indiqué. « Pérou 3 » : chaque photo du Pérou est tirée 3 fois plus souvent (20 % de chaque tour) ; « Pérou 0 » : sans le Pérou. Les deux dernières colonnes reprennent les archives `HEALTHY-1`, `BLACK SIGATOKA-1` et `FUSARIUM WILT-1` de la validation (1 000, 1 000 et 1 001 photos), avec la règle simple « malade si P(malade) ≥ 0,5 », sans « je ne sais pas ».

| Réseau (taille) | Réglage | Coût | Malades dites saines (sur 8 428) | dont Fusarium (sur 4 932) | Fausses alertes (saines) | Nom juste | Archives -1 bien classées : saine / Sigatoka / Fusarium | Fusarium dites saines (archives -1) |
|---|---|---|---|---|---|---|---|---|
| MobileNetV3 (4,2 M) | Pérou 3 | 427 | 88 | 81 | 4,4 % | 93,5 % | 987 / 967 / 875 | 39 |
| MobileNetV3 | Pérou 1 | 564 | 131 | 114 | 6,1 % | 92,8 % | 969 / 951 / 865 | 49 |
| MobileNetV3 | Pérou 0 | 642 | 132 | 91 | 7,1 % | 92,2 % | 948 / 927 / 912 | 29 |
| EfficientNet-B0 (4,0 M) | Pérou 0 | 419 | 78 | 56 | 4,2 % | 93,5 % | 964 / 964 / 912 | 11 |
| EfficientNet-B0 | Pérou 3 | 367 | 108 | 82 | 2,7 % | 94,1 % | 987 / 985 / 870 | 36 |
| EfficientNet-B2 (7,7 M) | Pérou 0 | 493 | 70 | 43 | 6,6 % | 92,4 % | 952 / 932 / 949 | 6 |
| EfficientNet-B2 | Pérou 3 | 337 | 77 | 43 | 1,9 % | 94,4 % | 987 / 981 / 888 | 13 |
| ConvNeXt-Tiny (27,8 M) | Pérou 0 | 242 | 23 | 21 | 2,8 % | 95,0 % | 941 / 970 / 938 | 6 |
| ConvNeXt-Tiny | Pérou 3 | 213 | 48 | 40 | 1,0 % | 96,0 % | 992 / 983 / 911 | 9 |
| ConvNeXt-Tiny | Pérou 3, JPEG, lissage | 213 | 36 | 29 | 1,5 % | 95,5 % | 963 / 977 / 913 | 4 |
| ConvNeXt-Tiny | Pérou 3, JPEG, lissage, moyenne de 3 | 167 | 30 | 26 | 1,8 % | 96,5 % | 993 / 981 / 922 | 10 |
| **ConvNeXt-Tiny** | **mêmes réglages, entraînement final (celui testé)** | **154** | **30** | **25** | **1,0 %** | **96,9 %** | **988 / 985 / 923** | **7** |

- **Le réseau** : ConvNeXt-Tiny fait nettement mieux que les autres (coût 213 contre 337 pour B2, avec ou sans le Pérou). Il est 6 à 7 fois plus gros que B0, mais reste utilisable sans GPU ;
- **le Pérou** : avec MobileNetV3, plus il y a de Pérou, meilleur est le résultat sur tous les critères (642, 564, 427). Avec les autres réseaux, il réduit surtout les fausses alertes, parfois au prix d'un peu plus de Fusarium ratées (le Pérou n'a pas de Fusarium). Le coût baisse avec le Pérou pour les 4 réseaux ;
- **la moyenne de 3 modèles** baisse le coût de 191 à 167 (puis de 196 à 154 lors de l'entraînement final) : les erreurs dues au hasard d'un entraînement se compensent ;
- **le hasard d'un entraînement** : deux entraînements identiques de MobileNetV3 ont donné 427 et 419 ; les 3 ConvNeXt de la moyenne, seuls et sur une vue, 225, 243 et 201. Je ne tire donc de conclusion que des écarts nets ou réguliers.

Le modèle testé est celui de l'entraînement final (mêmes réglages, réentraîné avant le test). Ses seuils, réglés sur la validation : alerte 0,30, nom de la maladie 0,50, confiance 0,646. Le seuil pour nommer la maladie tombe à 0,50 : le nom est assez fiable pour être toujours donné, la réponse « malade, non identifiée » n'est donc jamais utilisée.

**Détail par classe sur la validation** (entraînement précédent avec les mêmes réglages ; l'entraînement final donne des chiffres très proches : 30 malades dites saines, 1,0 % de fausses alertes, nom juste 96,9 %) :

| Classe réelle | Saine | Black Sigatoka | Fusarium | Malade, non identifiée | Je ne sais pas |
|---|---|---|---|---|---|
| Saine (3 338) | **3 231 (96,8 %)** | 10 | 51 | 0 | 46 |
| Black Sigatoka (3 496) | 4 | **3 267 (93,5 %)** | 41 | 0 | 184 |
| Fusarium (4 932) | 26 | 174 | **4 373 (88,7 %)** | 0 | 359 |

- 30 feuilles malades sur 8 428 (0,4 %) sont dites saines ; toutes les autres reçoivent une alerte ou un « je ne sais pas » ;
- 1,8 % des feuilles saines reçoivent une fausse alerte ;
- quand la maladie est nommée, elle est juste 96,5 % du temps ;
- le Fusarium reste le plus difficile : 3,5 % sont prises pour de la Black Sigatoka.

#### Inconnues simulées

Pour choisir comment repérer une maladie jamais vue sans toucher au Bangladesh, j'entraîne un modèle sans l'une des deux maladies, puis je la lui montre. Chaque méthode est réglée pour garder 95 % des photos connues de la validation.

| Maladie retirée | Rejetée par la confiance | Rejetée par la distance aux photos d'apprentissage | Les deux |
|---|---|---|---|
| Fusarium | **50 %** | 1 % | 32 % |
| Black Sigatoka | 8 % | 14 % | 14 % |

La distance (Mahalanobis, sur le résumé de 1 280 nombres que le réseau calcule pour chaque photo) fait moins bien que le hasard pour séparer connu et inconnu (AUROC 0,40 et 0,43, 0,5 = hasard). Mon explication probable : elle repère surtout que les photos de 2025 diffèrent de celles de 2023, pas qu'une maladie est nouvelle. Je garde donc la confiance seule (simulation faite avec MobileNetV3). Une maladie à taches comme la Sigatoka est très difficile à reconnaître comme inconnue (8 %) : la Cordana et la Pestalotiopsis du Bangladesh étant aussi des maladies à taches, je m'attendais à peu de « je ne sais pas » sur elles au test. Ce qui compte surtout, c'est qu'elles ne soient pas dites saines.

#### Cartes d'attention

Pour vérifier où le modèle regarde, j'utilise Grad-CAM : en rouge, les zones qui ont le plus pesé dans sa réponse (premier des 3 modèles finaux).

![Cartes d'attention du modèle maladies final](images/maladies_cartes_attention.jpg)
*Photos de validation tirées au hasard : 4 saines, 4 Black Sigatoka, 4 Fusarium, et 4 Fusarium prises pour de la Black Sigatoka.*

- saines : l'attention est répartie sur le limbe et les bords, et non plus concentrée sur la nervure centrale comme avec le premier MobileNetV3 (zone masquée et recadrages) ;
- Black Sigatoka : le modèle regarde les lésions (4e photo), mais aussi parfois des bords d'image ;
- Fusarium : zones jaunies ou desséchées ;
- Fusarium pris pour de la Black Sigatoka : le modèle s'arrête sur une marque locale (tache sèche, bord déchiré) qu'il prend pour une lésion ;
- aucun raccourci de décor net (fond, mains, tuyaux), mais les cartes de ConvNeXt sont plus diffuses que celles de MobileNetV3 : elles montrent une région, pas une tache précise. 16 photos ne suffisent pas pour conclure.

#### Test Bangladesh (03/11/2026)

Modèle final, avec les seuils réglés sur la validation, noté une seule fois. La Sigatoka du Bangladesh (type non précisé) compte comme Black Sigatoka.

| Classe réelle (Bangladesh) | Saine | Black Sigatoka | Fusarium | Je ne sais pas |
|---|---|---|---|---|
| Saine (90) | **86 (95,6 %)** | 0 | 1 | 3 |
| Sigatoka (472) | 0 | **449 (95,1 %)** | 10 | 13 |
| Cordana, inconnue (159) | 0 | 93 | 13 | **53 (33 %)** |
| Pestalotiopsis, inconnue (172) | 0 | 138 | 17 | **17 (10 %)** |

| | v1 | v2 |
|---|---|---|
| Bonnes réponses sur les classes connues (saine et Sigatoka) | 69 % | **95,2 %** |
| Sigatoka nommée | 66 % | **95,1 %** |
| Sigatoka dite saine | non mesuré | **0 %** |
| Feuilles saines signalées malades | 3 % | **1,1 %** (1 sur 90) |
| Maladies inconnues en « je ne sais pas » | 20 % | 21 % |
| Maladies inconnues dites saines | non mesuré | **0 %** |

- **Le changement de pays tient** : la v1 retrouvait 66 % des Sigatoka au Bangladesh, la v2 en nomme 95 % et n'en déclare aucune saine, pour des scores de validation proches ;
- **aucune feuille malade n'est dite saine**, ni Sigatoka, ni maladie inconnue : toutes reçoivent une alerte ou un « je ne sais pas » ;
- **le « je ne sais pas » reste faible** (21 %, comme en v1), comme la simulation l'annonçait pour des maladies à taches : 70 % des maladies inconnues sont prises pour de la Black Sigatoka. La confiance sépare mal connu et inconnu (AUROC 0,73 ; 0,5 = hasard). Pour un agriculteur, la feuille est quand même signalée malade, mais avec un nom qui peut être faux ;
- le test ne contient pas de Fusarium : la reconnaissance du Fusarium dans un autre pays n'est pas mesurée.

#### Fichiers plus légers (16 bits, 05/11/2026)

Les poids des 3 modèles sont enregistrés en 16 bits au lieu de 32 : chaque fichier passe de 111 à 56 Mo (168 Mo pour les 3). Au moment de prédire, ils sont remis en 32 bits : seul l'arrondi des poids peut changer une réponse. Sur 1 200 photos de validation tirées au hasard (400 par classe), **aucune réponse ne change** ; le plus grand écart de probabilité est de 0,0006. Ce sont ces fichiers que run.py utilise.

### v3 et v4 : plus de données pour la même recette (06/11/2026)

- **v3** : la recette de la v2, avec la Tanzanie 2025 ajoutée à l'apprentissage. La validation n'est donc plus séparée : je garde les seuils de la v2.
- **v4** : la recette de la v2, avec 562 feuilles saines de banana-buddy en plus (tirées 3 fois plus souvent). Seuils réglés sur la Tanzanie 2025 : alerte 0,20 (0,30 pour la v2), confiance 0,650. Coût sur la validation : 153 (154 pour la v2).

Je les ai notées sur BDR (408 photos, jamais apprises, carnet : BDR) :

| BDR | v2 | v3 | v4 |
|---|---|---|---|
| F1 moyenne (saine, Black Sigatoka, Fusarium) | 0,796 | 0,806 | 0,775 |
| Black Sigatoka nommée (sur 67) | 57 | 56 | 59 |
| Fusarium nommé (sur 41) | 18 | 20 | 15 |
| Dégâts d'insectes dits sains (sur 86) | 9 | 17 | 13 |
| Feuilles saines bien classées (sur 86) | 86 | 86 | 86 |

Et sur BananaLSD. **Attention** : BananaLSD avait déjà servi de test le 03/11. Ce n'est plus un test neuf pour ces deux modèles.

| BananaLSD | v2 | v3 | v4 |
|---|---|---|---|
| Saines bien classées (sur 90) | 95,6 % | **76,7 %** | 96,7 % |
| Sigatoka nommée (sur 472) | 95,1 % | 93,2 % | 96,0 % |

- la v3 dit « fusariose » à 9 feuilles saines de BananaLSD, des feuilles vert-jaune ;
- la v4 fait à peu près comme la v2 : un peu mieux sur BananaLSD (F1 0,981 contre 0,976), moins bien sur BDR (0,775 contre 0,796, fusariose nommée 15 fois sur 41 contre 18) ;
- j'ai aussi noté la v4 sur deux autres jeux, avec des chiffres qui ne valent pas grand-chose. Sur PSFD (Inde), elle dit saines 9 fusarioses sur 102, contre aucune pour la v2. La v2 y nomme la fusariose une fois sur trois et appelle « fusariose » 69 % des carences en potassium. Mais PSFD est plein de copies aux étiquettes contradictoires (carnet : PSFD). Sur les « taches » de banana-buddy, la v4 nomme plus souvent la Black Sigatoka (72,8 % contre 67,3 %). Mais ces dossiers sont à 70 à 83 % des copies tournées de la Tanzanie 2023, que la v2 a apprises : la comparaison était faussée ;
- la v2 reste installée.

### Lire la photo autrement : v2_384 et v2_zoom5 (08/11/2026)

Même v2 (mêmes poids, mêmes seuils), avec trois façons de lui donner la photo :
- **v2** : la photo réduite à 224 px, comme à l'apprentissage ;
- **v2_384** : la photo réduite à 384 px ;
- **v2_zoom5** : la photo entière et 4 zooms.

Notation sur 600 photos de la Tanzanie 2025 (haute résolution) et sur BDR (408 photos, 512 px) :

| Mesure | v2 (224 px) | v2_384 | v2_zoom5 |
|---|---|---|---|
| Tanzanie : saines bien classées (200) | 96,0 % | **99,0 %** | 78,5 % |
| Tanzanie : fausses alertes sur les saines | 0,5 % | 0,5 % | **8,5 %** |
| Tanzanie : Sigatoka noire nommée (200) | 93,0 % | 91,0 % | 86,0 % |
| Tanzanie : fusariose nommée (200) | 90,5 % | 89,0 % | 87,5 % |
| Tanzanie : malades dites saines (400) | 0,8 % | 1,0 % | **0,0 %** |
| BDR : saines bien classées (86) | 100 % | 100 % | 100 % |
| BDR : Sigatoka noire nommée (67) | 85,1 % | **94,0 %** | 92,5 % |
| BDR : fusariose nommée (41) | 43,9 % | 41,5 % | 46,3 % |
| BDR : malades dites saines (322) | 3,1 % | 2,5 % | **0,6 %** |

- **v2_384** n'ajoute aucune fausse alerte et donne moins de « je ne sais pas » (saines de Tanzanie : 1 contre 7 ; Sigatoka de BDR : 2 contre 7). La Sigatoka de BDR passe de 85 à 94 %. En Tanzanie, Sigatoka et fusariose nommées perdent 2 points (4 photos sur 200, dans le hasard). Le modèle n'a pourtant jamais appris en 384 px ;
- **v2_zoom5** rate presque aucune maladie (0 % en Tanzanie, 0,6 % sur BDR, surtout grâce aux dégâts d'insectes), mais déclenche 17 fois plus de fausses alertes sur les feuilles saines de Tanzanie (17 sur 200 contre 1), et confond plus souvent Sigatoka et fusariose ;
- **limite** : ces deux jeux contiennent surtout des Sigatoka bien visibles, et presque pas de feuilles « à quelques petites taches », le défaut vu sur khadavy (ci-dessous).

### khadavy : la v2 sur des feuilles à petites taches (08/11/2026)

khadavy (Roboflow « Klasifikasi Daun Pisang », Indonésie, carnet : khadavy) : 552 photos distinctes, regroupées en 276 séries de rafales. La v2 y a été notée une fois.

| Par photo | v2 |
|---|---|
| Saines bien classées (160) | 94,4 % |
| Fausses alertes sur les saines | 3,1 % |
| Taches noires (251) : Sigatoka nommée | 26,7 % |
| Taches noires : dites saines | 25,1 % |
| Taches noires : « je ne sais pas » | 33,9 % (85) |
| Taches jaunes (141) : dites malades | 92,2 % |

Les feuilles saines sont bien reconnues. Mais une feuille à taches noires sur quatre est dite saine : ce sont souvent des feuilles à quelques petites taches, des symptômes légers. C'est le défaut que les essais suivants visent.

### v6 : la v2 apprise en 384 px (09/11/2026)

La recette de la v2 à l'identique, avec un seul changement : les photos en 384 px (et la demi-précision sur le GPU, pour la vitesse).

| Validation (Tanzanie 2025) | v2 | v6 |
|---|---|---|
| Coût des erreurs (4 vues) | 154 | 141 |
| Malades dites saines | 30 | 26 |
| dont fusarioses dites saines | 25 | 25 |
| Fausses alertes | 1,0 % | 0,4 % |
| Nom juste quand le modèle nomme | 96,9 % | 97,6 % |

Seuils de la v6 : alerte 0,40, nom 0,50, confiance 0,685.

Sur les juges de la lecture en 384 px (600 photos de Tanzanie, BDR), la v6 fait 95 % de saines bien classées en Tanzanie (96 % pour la v2), nomme 88,1 % des Sigatoka de BDR (85,1 %) et 56,1 % de ses fusarioses (43,9 %). Mais elle dit saines 20 des 86 feuilles à dégâts d'insectes de BDR, contre 9 pour la v2 (détail : `C:\Field_scan\modeles\maladies_zoom\resultats_juges_finalistes.txt`).

### Trois lectures de la v2 sur des jeux neufs (09/11/2026)

Je compare la v2 (224 px), la v2_384 (mêmes poids, photo en 384 px) et la v6 (apprise en 384 px) sur des jeux qu'aucune n'a vus. Je compte les dégâts d'insectes comme des maladies : une feuille abîmée doit déclencher une alerte.

#### twakihasan (1 869 photos)

Jeu Kaggle twakihasan (carnet : twakihasan) : 1 183 Sigatoka, 407 saines, 279 fusarioses, photos de terrain en très haute résolution. Noté une seule fois.

| Par photo | v2 | v2_384 | v6 |
|---|---|---|---|
| **Malades dites saines** | 4,4 % | 3,6 % | 4,7 % |
| Sigatoka dite saine | 5,1 % | 4,3 % | 5,6 % |
| Fusariose dite saine | 1,4 % | 0,7 % | 0,7 % |
| Fausses alertes | 6,9 % | 5,2 % | 3,4 % |
| Saines bien classées | 84,3 % | 86,2 % | 84,5 % |
| Sigatoka nommée | 42,0 % | 55,8 % | 39,9 % |
| Fusariose nommée | 24,4 % | 18,3 % | 25,8 % |
| F1 moyenne (3 classes) | 52,2 % | 56,5 % | 51,8 % |

Écarts avec la v2, dont l'intervalle à 95 % ne contient pas 0 :
- **v2_384** : Sigatoka nommée +13,8 points (de +11,6 à +15,9), F1 +4,3 points (de +2,9 à +5,6), mais fusariose nommée −6,1 points (de −9,0 à −3,5). Pour les malades dites saines, l'écart (−0,8 point) n'est pas net ;
- **v6** : fausses alertes −3,4 points (de −5,5 à −1,5), mais Sigatoka nommée −2,1 points (de −3,9 à −0,4). Pour le reste, elle fait comme la v2.

Avec une seule photo par série (1 768 photos), on retrouve la même chose. Les trois modèles confondent beaucoup Sigatoka et fusariose : la v2 appelle « fusarium » 370 des 1 183 Sigatoka, et « black_sigatoka » 153 des 279 fusarioses.

Limites : presque toutes les photos sont du même jour, sans doute du même lieu. Les étiquettes ne sont pas vérifiées par un expert annoncé. Les feuilles « fusariose » sont jaunies et sèches, ce qui peut aussi venir de la vieillesse.

#### khadavy de près (487 photos, 238 séries)

Les photos de khadavy prises de près, notées une seule fois pour v2_384 et v6. Attention : khadavy n'est pas neutre pour ces deux-là, parce que l'idée de lire la photo en 384 px vient de ce jeu.

| Par photo | v2 | v2_384 | v6 |
|---|---|---|---|
| **Malades dites saines** | 18,0 % | 10,2 % | 20,1 % |
| Taches noires dites saines | 24,5 % | 13,9 % | 27,3 % |
| Sigatoka nommée (taches noires) | 26,9 % | 42,0 % | 12,2 % |
| Taches jaunes dites malades | 93,2 % | 100 % | 90,9 % |
| Saines bien classées | 94,2 % | 98,7 % | 98,7 % |
| Fausses alertes | 3,2 % | 0,0 % | 0,0 % |
| F1 moyenne (saine et Sigatoka) | 61,5 % | 74,3 % | 51,7 % |

Les intervalles sont larges : pour les malades dites saines, de 11,6 à 25,1 % pour la v2, de 5,1 à 16,4 % pour la v2_384. La v6 répond « je ne sais pas » à 119 des 245 feuilles à taches noires.

#### Trois jeux indicatifs

cm8zf62kzh (Bangladesh, 224 px), godliver (Ouganda probable, fond blanc) et BLADE (Inde, morceaux très zoomés et flous) : leurs défauts en font des indications, pas des tests (carnet : ces trois jeux).

| | v2 | v2_384 | v6 |
|---|---|---|---|
| **cm8zf62kzh (1 800 photos)** | | | |
| Malades dites saines | 10,5 % | 6,2 % | 11,9 % |
| dont dégâts d'insectes dits sains | 50,0 % | 28,0 % | 54,7 % |
| Sigatoka nommée | 51,3 % | 76,3 % | 37,0 % |
| Fusariose nommée | 75,3 % | 72,0 % | 78,7 % |
| Saines bien classées | 99,3 % | 99,3 % | 99,0 % |
| **godliver (620 photos)** | | | |
| Malades dites saines | 5,3 % | 7,3 % | 6,9 % |
| Sigatoka nommée | 88,1 % | 92,9 % | 85,7 % |
| Saines bien classées | 88,5 % | 93,3 % | 88,3 % |
| **BLADE (1 215 photos)** | | | |
| Malades dites saines | 9,8 % | 7,8 % | 10,6 % |
| dont Pestalotiopsis dite saine | 19,8 % | 13,5 % | 20,1 % |
| Sigatoka nommée | 72,3 % | 75,5 % | 63,1 % |
| Saines bien classées | 84,7 % | 80,0 % | 83,5 % |

#### Ce que j'en retiens

- **v2_384** nomme le mieux la Sigatoka sur tous les jeux neufs, sans réentraîner. Elle nomme moins bien la fusariose sur twakihasan ;
- **v6** fait mieux que la v2 sur la validation, mais pas sur les jeux neufs ;
- pour les trois, les dégâts d'insectes restent souvent dits « sains » (la moitié sur cm8zf62kzh pour la v2).

### v10 : plus de sources, une classe « autre » et DINOv3 (nuit du 09 au 10/11/2026)

Détail : `C:\Field_scan\modeles\maladies_v10\rapport.md`.

#### Ce qui change

1. **Plus de sources à l'apprentissage** : BananaLSD, cm8zf62kzh, BLADE et godliver, 75 % de chacun (le reste va dans la validation). Je retire les 123 copies de BDR (un de mes juges) trouvées dans cm8zf62kzh, et 5 groupes de copies aux étiquettes contradictoires (10 photos). BananaLSD, le test du 03/11, fait donc maintenant partie de l'apprentissage.
2. **Une 4e classe « autre »** : dégâts d'insectes, Sigatoka jaune, Moko ou feuille morte, flétrissement bactérien, Cordana (Inde et Bangladesh, jamais celle du Pérou) et Pestalotiopsis. L'alerte part sur 1 − P(saine) : une feuille « autre » reçoit une alerte, sans nom de maladie.
3. **Des seuils réglés sur une validation à 5 sources** (Tanzanie 2025 et 25 % de chaque nouvelle source), chaque source pesant autant.
4. **Cinq réseaux appris sur Kaggle** (environ 7 h de GPU), chacun lu à sa taille d'apprentissage et en 384 px (392 pour le ViT) :

| Candidat | Ce qui change |
|---|---|
| v10a | même réseau que la v2 (ConvNeXt-Tiny ImageNet) |
| v10b | ConvNeXt-Tiny aux poids DINOv3 (`convnext_tiny.dinov3_lvd1689m`) |
| v10c | ViT-B/14 affiné par Pl@ntNet sur PlantCLEF 2024 |
| v10d | v10a sans la classe « autre » |
| v10e | v10b appris en 320 px |

Sur la validation, en Tanzanie comme sur les nouvelles sources, tous les candidats sont presque parfaits : elle ne permet pas de les départager.

#### Sur les juges : khadavy de près et BDR

14 lectures candidates (chaque réseau à deux tailles, plus deux moyennes de réseaux, chacune à deux tailles), à côté de la v2 à deux tailles. Extrait, en part des photos :

| Lecture | Malades dites saines : BDR | Malades dites saines : khadavy | Fausses alertes (BDR / khadavy) | Sigatoka nommée (BDR / khadavy) | Insectes de BDR dits sains |
|---|---|---|---|---|---|
| v2 (224 px, installée) | 3,4 % | 17,7 % | 0 % / 3,9 % | 85,1 % / 26,9 % | 11,6 % |
| v2 en 384 px | 2,8 % | 10,2 % | 0 % / 0 % | 94,0 % / 42,0 % | 9,3 % |
| v10a en 384 px | 0,9 % | 10,8 % | 0 % / 0 % | 74,6 % / 21,2 % | 2,3 % |
| **v10b en 384 px** | **0,3 %** | **1,5 %** | 0 % / 0 % | 76,1 % / 19,6 % | **1,2 %** |
| v10c en 392 px | 1,6 % | 17,1 % | 0 % / 0 % | 70,1 % / 39,6 % | 4,7 % |
| v10e en 384 px | 1,9 % | 8,4 % | 0 % / 0 % | 91,0 % / 33,5 % | 7,0 % |

Les photos des juges sont ici réduites à 896 px avant la lecture, un peu autrement que le 09/11 : la v2 y fait 17,7 % de malades dites saines sur khadavy de près, au lieu de 18,0 %.

- **v10b en 384 px est la meilleure lecture**, et de loin : 0,3 % et 1,5 % de malades dites saines, contre 3,4 % et 17,7 % pour la v2 ;
- ce n'est pas un effet des « je ne sais pas » : sur les 245 feuilles à taches noires de khadavy, v10b en donne 5 « saines », 26 « je ne sais pas » et 214 alertes. La v2 en donnait 59 « saines » et 86 « je ne sais pas » ;
- le ViT de Pl@ntNet (v10c) ne fait pas mieux que la v2 sur khadavy.

Seuils de v10b lue en 384 px : alerte 0,725, nom 0,79, confiance 0,519.

#### Montrer d'autres maladies aide-t-il ? Oui

v10d est v10a sans la classe « autre » : même réseau, mêmes sources, sans les photos « autre ». Sur les juges :

| | v10a (avec « autre ») | v10d (sans) |
|---|---|---|
| Malades dites saines, moyenne des juges (224 / 384 px) | 8,1 % / 5,9 % | 11,0 % / 8,9 % |
| Insectes de BDR dits sains (224 / 384 px) | 7,0 % / 2,3 % | 15,1 % / 8,1 % |
| Sigatoka nommée sur BDR (224 / 384 px) | 77,6 % / 74,6 % | 92,5 % / 98,5 % |

La classe « autre » fait rater moins de malades, surtout les dégâts d'insectes. En échange, le modèle nomme moins souvent la Sigatoka.

#### Test neuf (1 602 photos, 10/11/2026)

Deux jeux Mendeley du Bangladesh (CC BY 4.0, carnet : wfzpdmc5vx et 5nfjzntwd8). J'ai cherché les copies contre toutes mes images (58 590) avant de noter, et j'en ai retiré une. Il reste 1 602 photos : 1 067 saines, 207 Sigatoka, 45 fusarioses et 283 « autre » (Cordana, Pestalotiopsis, insectes). Noté une seule fois, avec v10b en 384 px.

| | v10b_384 | v2 | v2_384 | Écart v10b − v2 [intervalle à 95 %] |
|---|---|---|---|---|
| **Malades dites saines** | **3,6 %** | 5,4 % | 5,0 % | −1,9 point [−3,4 ; −0,4] |
| Malades en « je ne sais pas » | 8,4 % | 15,9 % | 12,5 % | −7,5 points [−10,8 ; −3,7] |
| **Malades sans alerte** | **12,0 %** | 21,3 % | 17,5 % | |
| Fausses alertes | 0,9 % | 0,3 % | 0,1 % | +0,7 point [+0,2 ; +1,2] |
| Saines bien classées | 98,5 % | 99,3 % | 99,3 % | −0,7 point [−1,4 ; −0,2] |
| « Autre » dite saine | 6,4 % | 10,2 % | 9,5 % | −3,9 points [−6,7 ; −1,4] |
| Sigatoka nommée | 30,4 % | **59,4 %** | 61,4 % | −29,0 points [−35,7 ; −22,7] |
| Fusariose nommée | 46,7 % | **82,2 %** | 80,0 % | −35,6 points [−51,1 ; −22,2] |

Intervalles par bootstrap apparié (2 000 tirages, graine 0). Malades sans alerte : 64 photos sur 535 pour v10b, 114 pour la v2.

- **v10b rate moins de feuilles malades**, et l'écart est net : −1,9 point de malades dites saines, et presque deux fois moins de malades sans alerte ;
- en échange, elle donne un peu plus de fausses alertes (+0,7 point) ;
- le gain vient surtout de 5nfjzntwd8 : 8,6 % de malades dites saines contre 12,2 % pour la v2. Sur wfzpdmc5vx, les deux modèles sont presque à 0 (0,0 % contre 0,6 %).

#### Ce que ça coûte : le nom de la maladie

v10b donne plus souvent l'alerte, mais nomme moins souvent la maladie : elle répond souvent « malade, non identifiée ». Sur les 107 Sigatoka de 5nfjzntwd8, elle en nomme 6 et en met 80 en « malade, non identifiée ». La v2 en nommait 34, en appelait 42 « fusariose » et en mettait 31 en « je ne sais pas ».

En échange, v10b se trompe moins de nom. Sur wfzpdmc5vx, la v2 appelle « Black Sigatoka » 175 des 213 Cordana et Pestalotiopsis, contre 86 pour v10b.

Une piste, pas testée : l'alerte de v10b avec le nom de la v2 lue en 384 px. Sur les juges, elle garde 0,3 % et 1,5 % de malades dites saines, et la Sigatoka nommée remonte à 92,5 % sur BDR et 38,4 % sur khadavy. Mais elle donne plus souvent un faux nom aux autres maladies. Il faudrait un autre jeu neuf pour la juger.

#### Où en est v10b

Prête, pas installée.
- Le nouveau `classifieur.py` lit 3 ou 4 classes et plusieurs réseaux. Il redonne la v2 à l'identique (0 réponse différente sur 40, écart de probabilité nul).
- Sur les mêmes images, le PC redonne les réponses de Kaggle : 80 sur 80.
- Il faut ajouter `timm` aux dépendances. Les poids DINOv3 demandent la mention « Built with DINOv3 ».
- La lecture en 384 px rend le calcul environ 3 fois plus lent que la v2 sur PC.

#### Limites

- Le test neuf vient du Bangladesh, comme deux sources d'apprentissage (BananaLSD, cm8zf62kzh) et le juge BDR : il favorise la v10. khadavy (Indonésie) est le seul juge d'une autre région, et v10b y fait aussi beaucoup mieux (1,5 % contre 17,7 %).
- v10b est la meilleure de 14 lectures sur les deux juges : son score sur les juges est un peu flatté. Sur le test neuf, l'écart est plus petit (−1,9 point).
- La réponse dépend un peu de la façon dont la photo est réduite : sur les mêmes 80 photos réduites autrement, 8 réponses changent.
- cm8zf62kzh, BLADE et godliver sont maintenant à l'apprentissage, et twakihasan est réservé aux modèles d'avant : aucun de ces jeux ne peut comparer la v10 à la v2.

## 3. Vigueur et stress hydrique (Sentinel-2 et drone, `run.py`)

Ici, pas d'apprentissage : des indices calculés à partir des bandes, et des règles simples. Je n'ai pas de vérité terrain pour noter ces cartes. Je vérifie qu'elles sont cohérentes avec ce qu'on voit sur les images et avec les saisons.

**Les indices, et pourquoi ceux-là :**
- **NDVI** (proche infrarouge B08, rouge B04) : la chlorophylle absorbe le rouge, et les feuilles saines renvoient beaucoup de proche infrarouge. Il mesure la quantité de feuillage vert ;
- **NDRE** (B8A, bord du rouge B05) : plus sensible à la chlorophylle quand le feuillage est déjà dense et que le NDVI sature. Calculé et exporté, il ne sert pas aux alertes ;
- **NDMI** (B8A, infrarouge moyen B11) : l'eau des feuilles absorbe l'infrarouge moyen. Il baisse quand le feuillage s'assèche ;
- **drone RGB** (sans infrarouge) : ExG et VARI, comme le demande le sujet. Pour juger les zones, je sépare végétation et sol avec l'ExG (seuil d'Otsu), puis je calcule la part de végétation par case de 10 m.

**Les règles :**
- **vigueur faible** : un indice sous 80 % du niveau normal de la plantation (75e centile). La règle est relative, parce que le niveau normal change d'un site, d'une saison et d'un capteur à l'autre ;
- **manque d'eau probable** : un NDMI plus bas de 0,05 que ce qu'on attend pour sa quantité de feuillage. J'ajuste une droite NDMI = a + b × NDVI sur la plantation : une parcelle clairsemée a naturellement un NDMI bas, et ne doit pas passer pour sèche ;
- **saisons** : avec deux dates, une parcelle est signalée si son NDMI baisse de plus de 0,05 de plus que les autres parcelles comparables entre la saison humide et la saison sèche.

**Ce que donnent ces règles :**
- **Cameroun** (15 parcelles, 16/12/2025 et 20/04/2026) : en décembre, 5 parcelles sont clairsemées (NDVI 0,34 à 0,51). En avril, elles remontent à 0,82 à 0,85 : c'était une replantation, pas un stress. À la date la plus récente, le bulletin signale 4 parcelles : vigueur faible sur les parcelles 2 (83 % de la parcelle) et 11 (44 %), manque d'eau probable sur la 14 (74 %), et une baisse d'eau plus forte que les autres entre les deux dates sur les parcelles 10 et 14 ;
- **Équateur** (15 parcelles, 31/03/2026 et 22/09/2026) : plantation homogène, NDVI de 0,79 à 0,86. Aucune zone faible ni sèche ; seule la parcelle 9 perd plus d'eau que les autres entre les deux saisons ;
- **drone, ferme de Manuel** : 65 % de la parcelle passe en vigueur faible. C'est une ferme mélangée, avec des trous et d'autres arbres, et la règle n'est pas calibrée pour ce cas. Je le signale comme limite.

#### Ablation des bandes (09/11/2026)

Qu'apporte chaque bande ? Je refais les cartes de `run.py` en changeant une seule chose à la fois (l'indice, ou une bande), avec les mêmes règles et les mêmes masques. Sans vérité terrain, je mesure ce que change chaque bande, pas laquelle a raison. Mesures : part de la plantation signalée, parcelles en alerte (note sous 70, comme le bulletin), et accord avec la carte actuelle (Jaccard sur les pixels signalés : 1 = mêmes pixels, 0 = aucun en commun). Détail : `C:\Field_scan\analyses\ablation_bandes\resultats.md`.

| Vigueur | Cameroun 16/12/2025 | Cameroun 20/04/2026 | Équateur (31/03 et 22/09/2026) |
|---|---|---|---|
| NDVI B08/B04 (actuel) | 33,7 %, parcelles 3 4 5 6 13 | 12,2 %, parcelles 2 11 | 0 % et 0 % |
| NDRE B8A/B05 (bord du rouge) | 42,6 %, accord 0,79, parcelles 3 4 5 6 10 13 14 | 11,4 %, accord 0,63, parcelle 2 | 0 % et 0 % |
| GNDVI B08/B03 (vert au lieu du rouge) | 31,1 %, accord 0,92, mêmes parcelles | 2,1 %, accord 0,17, aucune | 0 % et 0 % |
| VARI B02/B03/B04 (sans infrarouge) | 65,3 %, accord 0,52, 11 parcelles | 63,8 %, accord 0,19, 10 parcelles | 26,8 % et 17,7 % |
| ExG B02/B03/B04 (sans infrarouge) | 51,0 %, accord 0,65, 10 parcelles | 57,0 %, accord 0,21, 9 parcelles | 13,0 % et 11,3 % |

| Stress hydrique | Cameroun 16/12/2025 | Cameroun 20/04/2026 | Équateur (31/03 et 22/09/2026) |
|---|---|---|---|
| NDMI B8A/B11 corrigé par le NDVI (actuel) | 0,8 %, aucune parcelle | 7,6 %, parcelle 14 | 0,1 % et 0 % |
| B12 au lieu de B11 | 1,0 %, accord 0,39 | 3,1 %, accord 0,31, aucune | 0,3 % et 0,2 % |
| Correction par le NDRE au lieu du NDVI | 0,6 %, accord 0,29 | 1,3 %, accord 0,15, aucune | 0 % et 0 % |
| NDMI sans correction (règle des 80 %) | 36,8 %, accord 0,02, 7 parcelles | 29,5 %, accord 0,16, 6 parcelles | 0 % et 0,5 % |

- **Le proche infrarouge (B08) porte l'information.** Sans lui (VARI, ExG), la moitié du Cameroun est signalée, et 11 à 27 % de l'Équateur, où les trois indices infrarouges ne voient rien. Les cartes visibles réagissent à autre chose que le feuillage ; sans terrain, je ne peux pas dire à quoi ;
- **le bord du rouge (B05)** à la place du rouge retrouve en gros les mêmes zones (accord 0,79 puis 0,63) ;
- **le vert (B03)** à la place du rouge colle très bien en décembre (0,92), mais presque plus en avril (0,17) : c'est instable d'une date à l'autre ;
- **B11 ou B12** pour le stress : la carte garde la même taille, mais les pixels changent (accord 0,31 à 0,39). Pour la comparaison des saisons, B11 et B12 signalent exactement les mêmes parcelles (Équateur : 9 ; Cameroun : 10 et 14) ;
- **la correction par le feuillage compte beaucoup** : sans elle, le NDMI signale 30 à 37 % du Cameroun, par blocs entiers, ce qui ressemble à une différence de couvert plutôt qu'à un manque d'eau ;
- la seule alerte « eau » de la règle actuelle (parcelle 14, en avril) disparaît avec toutes les variantes : c'est un signal faible.

Limites : deux sites, deux dates chacun, pas de mesure au sol. C'est un accord entre cartes, pas une précision. Quand la carte actuelle signale presque rien (Équateur), l'accord de Jaccard ne veut pas dire grand-chose.
