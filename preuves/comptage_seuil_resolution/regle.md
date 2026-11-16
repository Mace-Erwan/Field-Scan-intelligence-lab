# Seuil selon la résolution : règle écrite avant toute mesure (14/11/2026 vers 22 h 30)

## Le problème

Le test du 14/11 (`modeles/comptage_resolution_palm3/`) montre que palm3 sous-compte quand l'image est moins fine : à 11 cm par pixel, −41 % et −51 % sur les juges, −19 % sur jorges. Les scores des pics baissent, et le seuil fixe de 0,25 en laisse passer moins.

## L'idée

`run.py` connaît la taille du pixel d'origine de chaque image. Le seuil de détection devient une fonction de cette taille : 0,25 jusqu'à 5 cm par pixel (rien ne change pour les images fines), puis le seuil réglé pour 8, 11, 14 et 20 cm, avec une interpolation linéaire entre ces points, et le seuil de 20 cm au-delà.

## Réglage (validation seulement)

- **Modèle de réglage** : l'essai palm3 de la v6 (`modeles/comptage_v6_essai/comptage_v6/centres_palm3_0.pt` et `_1.pt`). Il n'a pas appris la validation, contrairement au final installé.
- **Images** : les 3 validations (Manuel, Count banana, `segmento`), dégradées comme dans le test du 14/11 : réduction à g cm (moyenne des pixels), puis retour à 5 cm (Lanczos).
- **Choix du seuil pour chaque g** : la règle d'équilibre de `outils/evaluer_comptage.py`, à l'identique. Parmi les seuils (0,05 à 0,95, pas de 0,05) dont la F1 moyenne des 3 validations est à moins de 0,005 de la meilleure, celui dont l'erreur de compte moyenne (en valeur absolue) est la plus petite.
- **Contrôle** : à g = 5, cette règle doit redonner 0,25 avec l'essai. Sinon, je le note, et le seuil à 5 cm reste 0,25.
- Les seuils retenus sont écrits dans `seuils_figes.json`, avec l'heure, avant toute lecture des juges.

## Notation (une seule fois, après le gel)

- **Modèle noté** : palm3 final installé, avec la fonction de seuil, contre le même modèle au seuil fixe de 0,25.
- **Jeux** : les 2 juges (Detecting Plantain Crops, Banano Finca) et jorges, dégradés à 5, 8, 11, 14 et 20 cm ; plus les 40 tuiles de palmiers à huile (zone jamais apprise), dégradées pareil, pour les fausses détections.
- **Mesures** : F1 à 1 m, erreur de compte, faux bananiers par hectare sur les palmiers.

## Règle d'adoption (sur les juges, à 8, 11, 14 et 20 cm ensemble)

Le seuil variable est adopté dans `run.py` si les trois conditions tiennent :
1. l'erreur de compte moyenne (valeur absolue, moyenne des 2 juges et des 4 résolutions) baisse d'au moins 0,10 ;
2. la F1 moyenne des 2 juges ne baisse de plus de 0,02 à aucune des 4 résolutions ;
3. les faux bananiers sur les palmiers restent sous 5 par hectare à chaque résolution (condition d du point 1).

jorges est noté pour information : il compte dans le rapport, pas dans la règle.

Même adopté, le résultat reste « non confirmé » : la dégradation est simulée, et il n'y a pas de vrai jeu neuf à 11 cm.
