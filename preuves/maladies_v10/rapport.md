# Maladies v10 : résultats de la nuit du 10/11/2026

**En bref.** La v10b, lue en 384 px, dit « saine » à moins de feuilles malades que la v2. Sur un test neuf de 1 602 photos, elle en dit saines 3,6 % contre 5,4 % (écart −1,9 point, intervalle de −3,4 à −0,4). La règle d'adoption, écrite avant la mesure, est remplie. En échange, elle nomme beaucoup moins souvent la maladie : elle répond plus souvent « malade, non identifiée ». Elle n'est pas installée : c'est à toi de décider.

Le protocole complet, écrit avant toute mesure, est dans `protocole.md`.

## Ce que j'ai changé

1. **Plus de sources en apprentissage.** J'ai ajouté BananaLSD, cm8zf62kzh, BLADE et godliver (75 % de chacun). Les copies de nos juges sont retirées (`preparer_v10.py`).
2. **Une 4e classe « autre ».** Elle regroupe les insectes, la Sigatoka jaune, le Moko ou la feuille morte, le flétrissement bactérien, la Cordana (Inde et Bangladesh, jamais celle du Pérou) et la Pestalotiopsis. L'alerte part sur 1 − P(saine).
3. **Des seuils réglés sur une validation à 5 sources**, chaque source pesant autant.
4. **Cinq réseaux appris sur Kaggle** (environ 7 h de GPU) :

| Candidat | Ce qui change |
|---|---|
| v10a | même réseau que la v2 (ConvNeXt-Tiny ImageNet) |
| v10b | ConvNeXt-Tiny aux poids DINOv3 |
| v10c | ViT-B/14 affiné par Pl@ntNet sur PlantCLEF 2024 |
| v10d | v10a sans la classe « autre » |
| v10e | v10b appris en 320 px |

## Choix sur les juges (khadavy de près, BDR)

14 lectures candidates (chaque réseau à deux tailles, plus des moyennes de réseaux), à côté de la v2 à deux tailles. La règle garde v10b en 384 px. Extrait, en part des photos :

| Lecture | Malades dites saines : BDR | Malades dites saines : khadavy | Fausses alertes (BDR / khadavy) | Insectes de BDR dits sains |
|---|---|---|---|---|
| v2 (224 px, installée) | 3,4 % | 17,7 % | 0 % / 3,9 % | 11,6 % |
| v2 en 384 px | 2,8 % | 10,2 % | 0 % / 0 % | 9,3 % |
| v10a en 384 px | 0,9 % | 10,8 % | 0 % / 0 % | 2,3 % |
| **v10b en 384 px (choisie)** | **0,3 %** | **1,5 %** | 0 % / 0 % | **1,2 %** |
| v10c en 392 px | 1,6 % | 17,1 % | 0 % / 0 % | 4,7 % |
| v10e en 384 px | 1,9 % | 8,4 % | 0 % / 0 % | 7,0 % |

Ce n'est pas un effet des « je ne sais pas ». Sur les 245 feuilles à taches noires de khadavy :
- v10b en donne 5 « saines », 26 « je ne sais pas » et 214 alertes ;
- la v2 en donnait 59 « saines » et 86 « je ne sais pas ».

Tableau complet : `juges_resultats.txt`.

## Montrer d'autres maladies aide-t-il ? Oui

v10d est v10a sans la classe « autre » : même réseau, mêmes sources, sans les photos « autre ».

| | v10a (avec « autre ») | v10d (sans) |
|---|---|---|
| Malades dites saines, moyenne des juges (224 / 384 px) | 8,1 % / 5,9 % | 11,0 % / 8,9 % |
| Insectes de BDR dits sains (224 / 384 px) | 7,0 % / 2,3 % | 15,1 % / 8,1 % |
| Sigatoka nommée sur BDR (224 / 384 px) | 77,6 % / 74,6 % | 92,5 % / 98,5 % |

La classe « autre » fait rater moins de malades, surtout les dégâts d'insectes. En échange, le modèle nomme moins souvent la Sigatoka.

## Test neuf, noté une seule fois

Le test réunit deux jeux Mendeley du Bangladesh (CC BY 4.0) : wfzpdmc5vx (424 photos) et 5nfjzntwd8 (feuilles brutes). J'ai cherché les copies contre tous nos jeux avant de noter, et j'en ai retiré une. Il reste **1 602 photos** : 1 067 saines, 207 Sigatoka, 45 fusarioses et 283 « autre » (Cordana, Pestalotiopsis, insectes).

| | v10b_384 | v2 | Écart [intervalle à 95 %] |
|---|---|---|---|
| **Malades dites saines** | **3,6 %** | 5,4 % | −1,9 pt [−3,4 ; −0,4] |
| Fausses alertes | 0,9 % | 0,3 % | +0,7 pt [+0,2 ; +1,2] |
| Saines bien classées | 98,5 % | 99,3 % | −0,7 pt |
| Malades en « je ne sais pas » | 8,4 % | 15,9 % | −7,5 pts |
| « Autre » dite saine | 6,4 % | 10,2 % | −3,9 pts |
| Sigatoka nommée | 30,4 % | 59,4 % | −29,0 pts |
| Fusariose nommée | 46,7 % | 82,2 % | −35,6 pts |

Les trois conditions de la règle sont remplies : (a) baisse nette des malades dites saines, (b) fausses alertes à +0,7 point (la limite était +3), (c) saines bien classées à −0,7 point (la limite était −5). La v2 en 384 px, donnée pour information, dit saines 5,0 % des malades.

Détail : `test_neuf/resultats_test.txt`.

## Ce que ça coûte : le nom de la maladie

v10b donne plus souvent l'alerte, mais elle nomme moins souvent la maladie. Exemple sur 5nfjzntwd8 : sur 107 Sigatoka, elle en nomme 6 et met 80 en « malade, non identifiée ».

Elle se trompe aussi moins de nom. Sur wfzpdmc5vx, la v2 appelle « black_sigatoka » 175 des 213 Cordana et Pestalotiopsis, contre 86 pour v10b.

Si le nom compte autant que l'alerte, une piste est d'utiliser l'alerte de v10b et le nom de la v2_384 (`info_alerte_v10b_nom_v2.py`, juges seulement) :
- sur les juges, elle garde 0,3 % et 1,5 % de malades dites saines, et la Sigatoka nommée remonte à 92,5 % sur BDR et 38,4 % sur khadavy ;
- mais elle donne plus souvent un faux nom aux autres maladies ;
- elle n'est pas testée : il faudrait un autre jeu neuf.

## Prêt à installer (pas fait)

Tout est dans `installation/`.
- **Poids et seuils** : `modele_v10b_384/`, poids en 16 bits.
- **Nouveau `classifieur.py`** : il lit 3 ou 4 classes et plusieurs réseaux. Il redonne la v2 à l'identique : 0 réponse différente sur 40, écart de probabilité 0 (`garde_fou_v2.py`).
- **Fidélité à Kaggle** : sur les mêmes images, le PC redonne les réponses de Kaggle, 80 sur 80 (`garde_fou_v10b.py`).
- **`installer_v10b.py`** : il sauvegarde la v2 hors du dépôt, installe v10b et ajoute `timm` à requirements.txt. Il ne commite rien.
- **Licence** : les poids DINOv3 demandent la mention « Built with DINOv3 » dans le README et le rapport.
- **Coût** : la lecture en 384 px rend le calcul environ 3 fois plus lent que la v2 sur PC.

## Limites

- Le test neuf vient du Bangladesh, comme deux sources d'apprentissage (BananaLSD, cm8zf62kzh) et le juge BDR. Il favorise la v10. khadavy (Indonésie) est le seul juge d'une autre région, et v10b y fait aussi beaucoup mieux (1,5 % contre 17,7 %).
- v10b a été choisie parmi 14 lectures sur deux juges : son score sur les juges est donc un peu flatté. Le test neuf corrige cela, et l'écart y est plus petit (−1,9 point).
- La réponse dépend un peu de la façon dont la photo est réduite. Sur les mêmes 80 photos, réduites autrement, 8 réponses changent.
- twakihasan ne note pas la v10, comme le prévoit la règle du 09/11.
