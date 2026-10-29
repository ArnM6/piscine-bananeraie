# Suivi de bananeraie par drone et satellite

Piscine Intelligence Lab ECE × Horama — chaîne qui transforme des images aériennes
d'une plantation en **carte des pieds**, **carte de vigueur**, **alertes** et
**bulletin par parcelle**.

> État : socle en place (lecture GeoTIFF, parcelles, vigueur RGB, comptage baseline,
> bulletin). Modules maladies et stress hydrique en cours.

## Installation

```bash
git clone https://github.com/<ton-compte>/piscine-bananeraie.git
cd piscine-bananeraie
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+. Aucun GPU nécessaire pour l'inférence.

## Utilisation

```bash
python run.py --images DOSSIER --parcelles parcelles.geojson --out resultats/
```

- `DOSSIER` : un ou plusieurs GeoTIFF RGB (orthomosaïques), **en projection métrique** (UTM).
- `parcelles.geojson` : polygones des parcelles (n'importe quel système, reprojeté
  automatiquement). Une colonne `parcelle_id` est utilisée si elle existe.

### Test rapide sans données réelles

```bash
python scripts/creer_donnees_test.py --out data/test_synthetique
python run.py --images data/test_synthetique/images --parcelles data/test_synthetique/parcelles.geojson --out resultats/
python scripts/evaluer_comptage.py --pred resultats/pieds.geojson \
    --verite data/test_synthetique/verite_terrain.geojson \
    --parcelles data/test_synthetique/parcelles.geojson
```

## Module maladies (feuilles)

Entraînement sur Colab : ouvre `notebooks/maladies_colab.ipynb` dans Google Colab (GPU T4), puis
*Exécution → Tout exécuter*. Le notebook télécharge les données (Tanzanie pour l'entraînement,
Bangladesh pour le test), entraîne MobileNetV3, calibre le seuil de rejet, mesure les performances et
exporte le modèle en ONNX sur ton Drive. Copie ensuite `maladies_*.onnx` et `maladies_*.json` dans `models/`.

Classer un dossier de photos de feuilles (CPU) :

```bash
python scripts/classer_feuilles.py --images DOSSIER_FEUILLES --out resultats/feuilles.csv
```

Chaque feuille reçoit `sain`, `sigatoka`, `fusarium` ou **`je_ne_sais_pas`** si le modèle n'est pas assez sûr.

## Sorties (`resultats/`), toutes ouvrables dans QGIS

| Fichier | Contenu |
|---|---|
| `pieds.geojson` | un point par pied détecté, avec `parcelle_id`, `score`, `image` |
| `<image>_exg.tif`, `<image>_vari.tif` | indices de végétation |
| `<image>_vigueur_anomalie.tif` | écart de vigueur à la médiane de la parcelle (−2 = nettement faible) |
| `alertes.geojson` | zones de vigueur faible, avec leur parcelle et l'image source |
| `parcelles_bulletin.geojson` | parcelles + indicateurs |
| `bulletin.csv`, `bulletin.html` | bulletin trié : parcelles à visiter en priorité en haut |

## Organisation du code

```
run.py                     point d'entrée
bananes/
  io.py                    lecture/écriture GeoTIFF, parcelles
  vigueur.py               ExG, VARI, anomalie relative par parcelle
  comptage.py              détection des pieds (baseline maxima locaux)
  bulletin.py              statistiques, alertes, bulletin
  maladies.py              classifieur de feuilles ONNX avec rejet de l'inconnu
scripts/
  creer_donnees_test.py    plantation synthétique (test de fumée)
  evaluer_comptage.py      F1 à 1 m + erreur de comptage par parcelle
  classer_feuilles.py      classe un dossier de photos de feuilles
notebooks/
  maladies_colab.ipynb     entraînement maladies (Colab)
  build_notebook.py        source lisible du notebook (le régénère)
docs/
  carnet_de_donnees.md     sources, licences, résolution, découpage, fuites
  rapport.md               rapport (4 pages max)
data/README.md             où télécharger chaque jeu de données
models/                    poids des modèles (ONNX)
```

## Données et modèles pré-entraînés

Voir [`docs/carnet_de_donnees.md`](docs/carnet_de_donnees.md). Les données ne sont
pas versionnées ; [`data/README.md`](data/README.md) indique comment les récupérer.
