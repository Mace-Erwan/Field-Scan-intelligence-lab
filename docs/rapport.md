# Field Scan : rapport

## 1. Le problème et la chaîne

Un planteur veut savoir, parcelle par parcelle, combien il a de pieds, où la plante pousse mal, où l'eau manque et si une maladie arrive. Une seule commande y répond :

```
python run.py --images DOSSIER --parcelles parcelles.geojson --out resultats/
```

Le dossier peut contenir une orthomosaïque drone RGB (comptage), des bandes Sentinel-2 à une ou deux dates (vigueur, stress hydrique) et des photos de feuilles (maladies). Chaque brique tourne seule si ses images sont là. La chaîne sort un bulletin HTML (carte, note par parcelle, alertes avec leur image source), un CSV et des couches QGIS (`pieds.geojson`, `alertes.geojson`, `vigueur.tif`, `stress.tif`). J'ai fait le socle et deux modules : stress hydrique et maladies. Tout tourne sans GPU.

Je découpe par ferme, vol, zone ou pays, jamais au hasard. Je règle sur la validation, je compare sur des « juges » (jeux jamais appris), et chaque test n'est noté qu'une fois.

## 2. Choix techniques

### Comptage : un détecteur de centres

Je n'ai pas besoin d'une boîte autour de chaque bananier, seulement de son cœur. Un ResNet-18 pré-entraîné sur ImageNet (11 millions de paramètres), suivi d'un petit décodeur, donne pour chaque carré de 20 cm la probabilité qu'un centre de bananier s'y trouve. Un pied est un pic de cette carte (le plus fort dans un carré de 1,4 m) au-dessus du seuil.

Face à DeepForest (détecteur de couronnes d'arbres réentraîné sur les mêmes données), il gagne en F1 de validation (0,800 contre 0,765) et surtout en compte : −8 %, +14 % et −2 % sur les trois validations, contre −12 %, −20 % et +29 %. Dans une bananeraie dense, les couronnes se chevauchent et les boîtes se gênent.

Les autres choix :
- **5 cm par pixel.** Chaque image est remise à cette échelle : le réseau a appris une taille de plant, il faut la lui redonner. Si les pieds sont espacés de moins de 1,3 m ou de plus de 4,5 m en médiane, le bulletin avertit que la taille du pixel est sans doute fausse.
- **4 vues, 2 réseaux.** L'image, ses deux miroirs et un quart de tour ; 8 vues n'apportent rien de plus.
- **Seuil 0,25, choisi pour le compte.** L'erreur de compte par parcelle est notée aussi. Parmi les seuils dont la F1 est à moins de 0,005 de la meilleure, je prends celui où fausses détections et oublis s'équilibrent. Sur les 5 jeux réunis (essai v3), 0,25 donne 14,7 % d'erreur de compte moyenne, contre 16,8 % à 0,30.
- **Des palmiers « rien à trouver ».** Vus du dessus, les palmiers à huile ressemblent à de grands bananiers. J'ajoute des tuiles de palmiers sans bananier dans 3 % des morceaux d'apprentissage : les faux bananiers sur les palmiers passent de 34,8 par hectare (essai v3) à 1,1 (palm3), sans perte de F1.

### Vigueur et stress hydrique : des indices et des règles relatives

Pas d'apprentissage ici : je n'ai pas de vérité terrain. Le NDVI (B08, B04) mesure la vigueur : la chlorophylle absorbe le rouge, la feuille saine renvoie le proche infrarouge. Le NDMI (B8A, B11) mesure l'eau : l'eau des feuilles absorbe l'infrarouge moyen. Sur drone RGB, j'utilise ExG et VARI.

Les règles sont relatives : une zone est faible si son indice est sous 80 % du niveau normal de la plantation (75e centile), ou de sa propre parcelle. Ce niveau change avec le site et la saison (NDMI médian de 0,43 et 0,41 aux deux dates en Équateur, 0,34 et 0,30 au Cameroun) : un seuil fixe ne passerait pas d'un site à l'autre. Pour l'eau, je compare le NDMI à ce qu'on attend pour la quantité de feuillage (droite NDMI = a + b × NDVI) : une parcelle clairsemée a un NDMI bas sans manquer d'eau. Avec deux dates, je signale une parcelle dont le NDMI baisse de plus de 0,05 de plus que les autres entre saison humide et saison sèche.

### Maladies : ConvNeXt-Tiny DINOv3 et « je ne sais pas »

Le modèle installé, v10b, fait la moyenne de 3 ConvNeXt-Tiny aux poids DINOv3 sur 4 vues. Il a appris sur 6 sources de 5 pays et connaît 4 classes : saine, cercosporiose (Sigatoka), fusariose, et « autre » (insectes, Cordana, Pestalotiopsis…). ConvNeXt-Tiny bat les réseaux légers (tableau 2). DINOv3 rate beaucoup moins de malades qu'ImageNet (section 4). La lecture en 384 px garde les petites taches. La classe « autre » apprend qu'une feuille abîmée n'est pas saine.

Une maladie ratée coûte plus cher qu'une fausse alerte : la fausse alerte fait faire un détour, la Sigatoka ratée peut coûter la parcelle. D'où la règle B : une feuille n'est dite saine que si P(saine) atteint 0,519. Sinon, la réponse est « je ne sais pas », et le bulletin en fait une alerte « à vérifier ». Sur la validation, les malades dites saines passent de 85 à 61 (sur 9 982), pour 137 saines en alerte au lieu de 111 (sur 3 718). La maladie n'est nommée que si le modèle est assez sûr, sinon c'est « maladie non identifiée ».

### Bulletin : le pire problème décide

Chaque parcelle reçoit une note sur 100 par thème. Vigueur et eau : la part de la parcelle sans problème. Maladies, selon la gravité : 100 si tout est sain, 75 s'il n'y a que des « je ne sais pas », sous 70 dès qu'une feuille est malade. Le score de santé est la pire des notes : une bonne vigueur ne doit pas cacher une parcelle malade. Vert à 85 et plus, rouge sous 70.

## 3. Comparaison des approches

**Tableau 1. Comptage.** F1 à 1 m. Juges : Detecting Plantain Crops et Banano Finca, deux plantations jamais apprises.

| Approche | F1 validation | F1 juges | Faux bananiers/ha (palmiers) | Taille et vitesse |
|---|---|---|---|---|
| Méthode mathématique M (axes des feuilles, sans apprentissage) | 0,679 | 0,541 | 22,3 | aucun réseau |
| DeepForest (boîtes) | 0,765 | | | apprentissage 86 min (T4) |
| Centres v2 (seuil 0,30) | 0,800 | 0,715 | | 2 × 23 Mo ; apprentissage 12 min par réseau (T4) |
| Essai v3 (+ jeunes plantations, plantain) | 0,799 | 0,761 | 34,8 | 2 ResNet-18 |
| Essai v5 (palmiers dans 10 % des morceaux) | 0,800 | 0,730 | 1,3 | 2 ResNet-18 |
| **palm3 (palmiers dans 3 %), installé** | **0,800** | **0,758** | **1,1** | 2 × 23 Mo (16 bits) ; 1 min 40 par ha sur 2 cœurs, 54 s par ha sur mon PC |
| Ensemble de 12 réseaux (C2) | 0,811 | 0,753 | 2,73 | 6 fois le temps de palm3 |
| v7 : ConvNeXt-Tiny DINOv3 | 0,814 | palm3 − 0,048 | | 28 M paramètres |
| ConvNeXt DINOv3 gelé + décodeur | 0,767 | 0,704 | | |

L'ensemble de 12 réseaux gagne en validation, rien sur les juges, pour 6 fois plus de calcul. palm3 garde la F1 de la v3 et supprime presque toutes les fausses détections sur les palmiers.

**Tableau 2. Maladies, choix du réseau** (validation Tanzanie 2025, même recette, 224 px). Coût = 2 × malades ratées + fausses alertes.

| Réseau | Paramètres | Coût | Malades dites saines (sur 8 428) | Fausses alertes |
|---|---|---|---|---|
| MobileNetV3 | 4,2 M | 427 | 88 | 4,4 % |
| EfficientNet-B0 | 4,0 M | 367 | 108 | 2,7 % |
| EfficientNet-B2 | 7,7 M | 337 | 77 | 1,9 % |
| ConvNeXt-Tiny | 27,8 M | 213 | 48 | 1,0 % |
| ConvNeXt-Tiny, moyenne de 3 (v2) | 3 × 27,8 M | 167 | 30 | 1,8 % |

**Tableau 3. Maladies : v2 (ConvNeXt ImageNet, 224 px) contre v10b (ConvNeXt DINOv3, 384 px).**

| Mesure | v2 | v10b |
|---|---|---|
| Taille (16 bits) | 3 × 56 Mo | 3 × 56 Mo |
| Vitesse | référence | environ 3 fois plus lente ; environ 3 s par photo sur 2 cœurs, 1,7 à 1,8 s sur mon PC |
| Juge khadavy (Indonésie) : malades dites saines | 17,7 % | 1,5 % |
| Test neuf Bangladesh (1 602 photos) : malades sans alerte | 21,3 % | 12,0 % |
| Même test : fausses alertes | 0,3 % | 0,9 % |
| Même test : Sigatoka nommée | 59,4 % | 30,4 % |
| BanSpot, Inde : feuilles tachées ratées | 3,56 % | 0,93 % |
| BanSpot : feuilles sans taches en alerte | 5,11 % | 10,22 % |
| AgroVisionBD, Bangladesh : malades ratées / saines en alerte | | 1,26 % / 15,26 % |

Le test neuf du 10/11 est noté avant la règle B. v10b rate beaucoup moins de malades, mais donne plus de fausses alertes et nomme moins la maladie. Ma règle écrite avant BanSpot gardait la v2 (3 points de fausses alertes en plus au plus). J'ai installé v10b en connaissant ce verdict, pour la raison de la section 2.

## 4. Ablations

**Tableau 4. Ce que chaque élément apporte** (on retire ou on change un seul élément).

| Élément retiré ou changé | Mesure | Sans | Avec |
|---|---|---|---|
| Comptage : 4 vues | F1 validation, par modèle | référence | +0,016 à +0,022 |
| Comptage : flou, JPEG, zoom 0,67 à 1,5 | F1 validation (2 graines) | 0,785 | 0,798 |
| Même augmentation, bananiers 1,5 fois plus grands | perte de F1 | 10,7 points | 1,6 point |
| Comptage : jeunes plantations et plantain | F1 juges | 0,715 | 0,761 |
| Comptage : tuiles de palmiers | faux bananiers/ha | 34,8 | 1,1 |
| Comptage : un type de plantation à l'apprentissage | F1 sur ce type | perte de 0,03 à 0,24 | référence |
| Maladies : photos du Pérou (MobileNetV3) | coût validation | 642 | 427 |
| Maladies : JPEG et lissage des étiquettes | malades ratées (coût égal, 213) | 48 | 36 |
| Maladies : 4 vues (EfficientNet-B2) | coût validation | 392 | 337 |
| Maladies : classe « autre » (384 px) | malades dites saines, juges | 8,9 % | 5,9 % |
| Maladies : DINOv3 au lieu d'ImageNet | malades dites saines, khadavy / BDR | 10,8 % / 0,9 % | 1,5 % / 0,3 % |
| Maladies : lecture en 384 px (v2) | malades dites saines, khadavy | 18,0 % | 10,2 % |
| Vigueur : proche infrarouge (VARI contre NDVI) | part du Cameroun signalée, avril | 63,8 % | 12,2 % |
| Eau : correction du NDMI par le NDVI | part du Cameroun signalée | 29,5 à 36,8 % | 0,8 à 7,6 % |

**Bandes.** Sans vérité terrain, je mesure ce que change chaque bande, pas laquelle a raison. Le proche infrarouge porte l'information : sans lui, 11 à 27 % de l'Équateur est signalé, là où les indices infrarouges ne voient rien. Le bord du rouge retrouve à peu près les mêmes zones que le rouge (accord de Jaccard 0,79 puis 0,63) ; le vert est instable (0,92 en décembre, 0,17 en avril). B11 ou B12 signalent les mêmes parcelles entre les saisons. Sans correction par le feuillage, le NDMI signale des blocs entiers : une différence de couvert, pas un manque d'eau. La seule alerte « eau » actuelle (parcelle 14 du Cameroun) disparaît avec toutes les variantes : c'est un signal faible.

**Augmentations.** Au comptage, flou, JPEG et zoom aident sur les trois validations et réduisent l'écart entre apprentissage et validation (0,124 contre 0,148). Pousser plus loin fait perdre sur les juges : zoom jusqu'à 2 (−0,022), apparence très forte (−0,024).

**Pré-entraînement.** DINOv3 est le plus gros gain des maladies. Au comptage, c'est l'inverse : un ConvNeXt DINOv3 gagne en validation et perd sur les juges (−0,048), et gelé il perd partout. Pour des points vus du ciel, ResNet-18 ImageNet suffit.

**Données.** Un type de plantation absent de l'apprentissage fait perdre de 0,03 (plants séparés) à 0,24 (bananeraie dense et floue). Mais plus de données ne veut pas dire mieux : deux bananeraies des Philippines en plus (essai v4) gagnent 0,016 en validation et perdent 0,062 sur les juges. Le gain des jeunes plantations tombe à +0,024 si v2 et v3 sont comparées au même seuil.

**Résolution.** Mon premier test, avec la v2 sur la validation, disait qu'une image à 20 cm par pixel coûte moins de 3 points de F1. Refait avec palm3 sur des images jamais apprises, il est moins rassurant. À 11 cm par pixel, la F1 baisse de 0,771 à 0,685 sur Detecting Plantain Crops, de 0,737 à 0,632 sur Banano Finca, de 0,806 à 0,789 sur jorges. Surtout, le compte chute : −41 % et −51 % sur les deux juges, −19 % sur jorges. La validation, elle, ne bouge presque pas, parce que le modèle final l'a apprise. Un seuil abaissé à 0,20 pour les images de 11 cm et plus remonte la F1 des juges (0,658 à 0,726 à 11 cm) et réduit l'erreur de compte, mais il dépasse de peu la limite de faux bananiers sur les palmiers à 20 cm (5,12 par hectare pour 5) : je ne l'ai pas adopté. Aux maladies, apprendre en 384 px (v6) n'aide pas sur les jeux neufs : lire en 384 px suffit.

## 5. Analyse des cas d'échec

**Le compte casse dans les champs denses.** Au test, la F1 tient sur quatre types de plantation (0,767 à 0,784), mais le compte varie. Sur les 7 champs de banavista et jorges, il va de −38,4 % à +30,6 %, avec un rappel de 0,55 et 0,60 sur deux champs serrés. Sur le juge Banano Finca, il compte −33 %. Banano Finca a 1 600 à 2 500 plants par hectare, la validation 250 à 860. Sur 154 plants ratés, 104 ont un pic sous 0,25, et 28 sont collés à un voisin à moins de 1,4 m, la fenêtre des pics : deux plants, un seul pic. Sur les juges, l'erreur de compte suit la densité (Spearman −0,90).

**Jeunes plants et rejets.** Sur une jeune plantation (photos DJI), le compte est de −10,6 %. Sur Banano Finca, plus d'un jeune plant sur trois est oublié. 13 rejets sur 26 reçoivent leur propre détection ; un rejet collé à sa mère n'est pas compté à part.

**La ferme mélangée surcompte.** Sur Nestor (Équateur), 282 détections pour 235 bananiers : +20,0 %, avec un bon rappel (0,847) et une précision de 0,706. Je n'ai pas vérifié sur quoi tombent ces fausses détections. Sur ce type de ferme, la vigueur drone n'est pas calibrée : le sol nu fait passer 65 % de la parcelle de Manuel en vigueur faible.

**Ma validation ne prédit pas les juges.** Du 12 au 14/11, plus de 40 essais n'ont battu ni palm3 ni v10b. Sur 12 modèles de comptage, les classements en validation et sur les juges vont plutôt en sens inverse (Spearman −0,47 ; −0,78 avec Count banana seul). Le meilleur en validation est 9e sur 12 aux juges.

**Réapprendre avec plus de données fait perdre en sécurité.** Les modèles maladies réappris ou ajustés (v12a, v12b, v12e, v12g) sont meilleurs en validation, mais disent « saine » à 5,71 à 13,81 % des malades de khadavy, contre 0,90 % pour v10b. Je ne sais pas pourquoi.

**Fausses alertes sur les feuilles abîmées.** J'ai relu les 50 premières fausses alertes de v10b sur BanSpot : 30 feuilles trouées par des insectes ou tachées de brun, 7 déchirures, 1 jaunissement, 12 feuilles qui ont l'air propres. BanSpot appelle « sans taches » toute feuille sans Sigatoka, même trouée. Sur AgroVisionBD, les saines sont photographiées sur pied et les malades au sol, et deux séries portent 21 des 56 fausses alertes.

**Le nom de la maladie est faible.** v10b nomme la Sigatoka sur 26,8 % des feuilles atteintes d'AgroVisionBD : elle alerte, mais répond souvent « maladie non identifiée ». L'inconnu est mal reconnu comme tel : au test du Bangladesh, la v2 prenait 70 % de la Cordana et de la Pestalotiopsis pour de la Sigatoka, sans en dire aucune saine.

**Des étiquettes fausses dans la validation.** HEALTHY-1 (Tanzanie 2025) contient sans doute 140 à 260 feuilles malades rangées « saines », selon mon estimation. Mon détecteur d'étiquettes fausses ne les retrouve pas : il ne voit pas la Sigatoka légère. Ces erreurs tirent les seuils : retirer 34 saines douteuses sur 13 700 fait passer le seuil d'alerte de 0,725 à 0,625.

**Sentinel-2 voit des blocs de 10 m.** Un pixel couvre plusieurs bananiers, et une parcelle de 1,4 ha ne contient qu'environ 140 pixels. Au Cameroun, 5 parcelles clairsemées en décembre (NDVI 0,34 à 0,51) remontent à 0,82 à 0,85 en avril : une replantation, pas un stress, qu'une seule date aurait prise pour une faiblesse.

## 6. Limites et domaine de validité

**Feuilles au sol, pas images drone.** Toutes les photos de feuilles sont prises au sol, de près, au smartphone : le modèle maladies ne sait pas lire une image drone. En conditions réelles, c'est un outil de terrain. Les cartes aériennes disent où envoyer une équipe, qui photographie les feuilles pour confirmer. Détecter les maladies depuis le drone demanderait des images aériennes de feuilles malades annotées, que je n'ai pas trouvées en données publiques.

**Comptage.** Bananeraies et fermes mélangées vues du dessus, de préférence à 5 à 8 cm par pixel : au-delà, le compte baisse vite. L'agroforesterie n'a pas été testée, et les rejets ne sont pas séparés de leur mère. Les juges sont petits (une scène chacun) : leur F1 est imprécise à quelques points près.

**Vigueur et stress.** Deux sites, deux dates chacun, aucune mesure au sol : je mesure un accord entre cartes, pas une précision.

**Robustesse et vitesse.** 15 cas de fichiers « piégés » (projections, noms de bandes, dates, bandes manquantes, photos sans GPS, chemins accentués) passent tous, après 4 corrections. Une installation dans un environnement vide redonne exactement les mêmes résultats. Sans GPU, une orthomosaïque de 858 Mo (1,55 ha à 1 cm par pixel) passe en 4 min 29 s, dont 110 s par hectare pour le comptage.

**Maladies.** Le test neuf du 10/11 vient du même pays que deux sources d'apprentissage, ce qui favorise v10b. Plusieurs jeux ne citent aucun expert pour leurs étiquettes. Une Sigatoka légère sur feuille entière reste le point faible.
