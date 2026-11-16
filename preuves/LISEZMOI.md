# Preuves

Les scripts tels que je les ai lancés, et leurs résultats bruts. Chaque dossier correspond à une mesure du journal ([docs/experiences.md](../docs/experiences.md)).

| Dossier | Mesure | Fichier à lire en premier |
|---|---|---|
| `comptage_test_unique/` | test du comptage, lu une seule fois le 10/11 (Nestor, segmento, photos DJI) | `resultats.txt` |
| `comptage_resolution/` | palm3 face à une image moins fine (8 à 20 cm par pixel) | `resultats.csv` |
| `comptage_seuil_resolution/` | seuil qui dépend de la résolution : règle écrite avant, seuils gelés, verdict (non adopté) | `regle.md`, puis `verdict.txt` |
| `robustesse/` | 15 cas de fichiers « piégés » passés à `run.py` | `bilan_cas_pieges.json` |
| `maladies_v10/` | essais v10 des maladies, préparation des données et liste figée du test neuf | `rapport.md` |
| `maladies_khadavy/`, `maladies_twakihasan/` | préparation de deux jeux juges (copies écartées, liste figée) | le script du dossier |
| `maladies_zoom/` | juges de la lecture en 384 px | `resultats_juges_finalistes.txt` |
| `maladies_planche/` | planche du README : 9 gros plans d'AgroVisionBD recadrés, puis lus par le modèle installé (les `.txt` donnent le classement des photos par part de feuille dans l'image) | `planche_gros_plans.py` |
| `ablation_bandes/` | ce qu'apporte chaque bande Sentinel-2 | `resultats.md` |

Les données ne sont pas dans le dépôt. Les scripts gardent donc les chemins de mon PC (`C:/Field_scan/data/...`) : pour les relancer, il faut les remplacer par l'endroit où sont tes données.
