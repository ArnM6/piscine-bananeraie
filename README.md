# Suivi de bananeraie par drone et satellite

Piscine Intelligence Lab ECE × Horama — chaîne qui transforme des images aériennes
d'une plantation en **carte des pieds**, **carte de vigueur**, **alertes** et
**bulletin par parcelle**.

> État : socle en place (lecture GeoTIFF RGB ou multispectral, parcelles, vigueur, comptage par
> détecteur appris, bulletin) et module maladies. Stress hydrique par Sentinel-2 en place (notebook Colab + fusion dans le bulletin).

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
- `--comptage` : méthode de comptage des pieds.
  - `affine` (défaut) : détecteur affiné sur Guangxi, le meilleur sur une **monoculture dense** (F1 0,52) ;
  - `robuste` : détecteur sans Guangxi, plus stable sur un **site très différent** (F1 0,37 à 0,41 en Équateur) ;
  - `baseline` : maxima locaux de verdure, sans réseau (F1 0,17 sur canopée fermée).

Le détecteur a besoin de `models/pieds_refo_multires_affine_O.pt` et/ou `models/pieds_refo_multires.pt`
(sortis de `notebooks/comptage_colab.ipynb`, dossier `MyDrive/piscine-bananeraie/models/`). S'ils manquent,
la chaîne repasse automatiquement en `baseline`. L'image est ramenée à **8 cm/pixel** (résolution
d'entraînement) ; au-delà de 10 cm/pixel, un avertissement signale que le comptage n'est plus fiable.

### Test rapide sans données réelles

```bash
python scripts/creer_donnees_test.py --out data/test_synthetique
python run.py --images data/test_synthetique/images --parcelles data/test_synthetique/parcelles.geojson --out resultats/
python scripts/evaluer_comptage.py --pred resultats/pieds.geojson \
    --verite data/test_synthetique/verite_terrain.geojson \
    --parcelles data/test_synthetique/parcelles.geojson
```

## Démo : carte cliquable

Après `run.py`, une seule commande produit `resultats/carte.html`, une page autonome (~5 Mo) à ouvrir dans un
navigateur, sans serveur :

```bash
python scripts/demo_carte.py --image data/guangxi_pret/images/guangxi_plantation.tif --resultats resultats/
```

- image du drone en fond (et carte de vigueur relative en option), fond satellite pour le contexte ;
- parcelles colorées selon le score de santé ; à gauche, la liste des **parcelles à visiter en premier** ;
- un clic sur une parcelle : surface, pieds, densité, part en vigueur faible, stress hydrique, vignette drone ;
- un clic sur une **zone de vigueur faible** : sa parcelle, sa surface et l'**image qui justifie l'alerte**
  (extrait drone à 8 cm, zone entourée en rouge, pieds détectés) ;
- les pieds détectés en couche activable.

Il faut Internet à l'ouverture (bibliothèque Leaflet et fond satellite) ; les données et images sont dans le fichier.

## Bonus : plan d'intervention de la semaine

```bash
python scripts/plan_intervention.py --resultats resultats/
python scripts/demo_carte.py --image data/guangxi_pret/images/guangxi_plantation.tif --resultats resultats/
```

Les zones de vigueur faible deviennent des **haltes** (regroupées à moins de 8 m). Priorité 1 : toutes celles des
parcelles en alerte ; priorité 2 : les taches d'au moins 50 m² ailleurs. La tournée à pied passe d'abord par la
priorité 1, dans un ordre optimisé. Sorties : `plan_intervention.csv` (ordre, parcelle, surface, heure d'arrivée,
action), `plan_intervention.geojson`, `plan_intervention_resume.json` ; la carte de démo affiche la tournée.
Hypothèses de temps (3 km/h, 10 min par halte…) en tête du script, à régler avec le chef de culture.

## Module stress hydrique (Sentinel-2)

L'eau des feuilles se voit dans l'infrarouge moyen (SWIR), que le drone n'a pas : on utilise Sentinel-2
(gratuit, tous les 5 jours) sur la même plantation. Depuis le dossier du repo :
`python scripts/stress_sentinel2.py` (~10-20 min, pas de GPU ni de compte ; résultats dans `docs/resultats_stress/`).
Même code en notebook Colab : `notebooks/stress_colab.ipynb`. Le notebook :
- lit B04, B8A, B11 et le masque de nuages (SCL) de 2021 à 2023 sur une petite fenêtre autour des parcelles ;
- calcule le **NDMI** = (B8A − B11) / (B8A + B11) par parcelle et retire la part due à la quantité de feuilles (NDVI) ;
- compare chaque parcelle aux autres **à la même date** (`z_stress`, −2 = nettement plus sèche) ;
- vérifie la cohérence avec la météo (pluie − évaporation, ERA5), saison sèche contre saison des pluies, et la stabilité.

Une parcelle est **en alerte** si elle est plus sèche que les autres (z < −2) sur **2 dates claires de suite**.
Sur Colab, copie ensuite `MyDrive/piscine-bananeraie/resultats_stress/` dans `docs/resultats_stress/`. Puis :

```bash
python run.py --images DOSSIER --parcelles parcelles.geojson --out resultats/ \
    --stress docs/resultats_stress/stress_par_parcelle.csv
```

Le bulletin gagne `date_stress`, `z_stress`, `alerte_stress` et `stress_hydrique` (« à vérifier sur le terrain »).
Cet indicateur n'a pas pu être validé sur Guangxi (pas de lien avec la pluie) : il est **affiché mais ne compte pas
dans le score de santé**, qui reste fondé sur la vigueur seule.

## Module maladies (feuilles)

Entraînement sur Colab : ouvre `notebooks/maladies_colab.ipynb` dans Google Colab (GPU T4), puis
*Exécution → Tout exécuter*. Le notebook télécharge les données (Tanzanie pour l'entraînement,
Bangladesh pour le test), entraîne MobileNetV3, calibre le seuil de rejet, mesure les performances et
exporte le modèle en ONNX sur ton Drive. Si Colab coupe, relancer reprend là où il s'était arrêté.
Copie ensuite `maladies_*.onnx`, `maladies_*.json` et `maladies_*.banque.npy` dans `models/`.

Classer un dossier de photos de feuilles (CPU) :

```bash
python scripts/classer_feuilles.py --images DOSSIER_FEUILLES --out resultats/feuilles.csv
```

Chaque feuille reçoit `sain`, `sigatoka`, `fusarium` ou **`je_ne_sais_pas`** si elle ne ressemble pas assez
aux feuilles vues à l'entraînement (distance aux 10 plus proches exemples connus, méthode kNN).

## Sorties (`resultats/`), toutes ouvrables dans QGIS

| Fichier | Contenu |
|---|---|
| `pieds.geojson` | un point par pied détecté, avec `parcelle_id`, `score`, `image` |
| `<image>_exg.tif`, `<image>_vari.tif` | indices de végétation |
| `<image>_vigueur_anomalie.tif` | écart de vigueur à la médiane de la parcelle (−2 = nettement faible) |
| `alertes.geojson` | zones de vigueur faible, avec leur parcelle et l'image source |
| `parcelles_bulletin.geojson` | parcelles + indicateurs |
| `bulletin.csv`, `bulletin.html` | bulletin trié : parcelles à visiter en priorité en haut |
| `carte.html` | démo cliquable (`scripts/demo_carte.py`) |

## Organisation du code

```
run.py                     point d'entrée
bananes/
  io.py                    lecture/écriture GeoTIFF, parcelles
  vigueur.py               ExG, VARI, anomalie relative par parcelle
  comptage.py              détection des pieds (baseline maxima locaux)
  detecteur.py             détection des pieds par RetinaNet (DeepForest affiné), CPU
  bulletin.py              statistiques, alertes, bulletin
  stress.py                NDMI Sentinel-2, contrôle canopée, écart relatif, alerte persistante
  maladies.py              classifieur de feuilles ONNX avec rejet de l'inconnu
scripts/
  creer_donnees_test.py    plantation synthétique (test de fumée)
  evaluer_comptage.py      F1 à 1 m + erreur de comptage par parcelle
  classer_feuilles.py      classe un dossier de photos de feuilles
  stress_sentinel2.py      stress hydrique Sentinel-2 sans Colab (généré depuis le notebook)
  demo_carte.py            démo : carte HTML cliquable (+ demo_carte_gabarit.html)
  plan_intervention.py     bonus : tournée de la semaine (haltes, ordre, temps)
notebooks/
  maladies_colab.ipynb     entraînement maladies (Colab)
  comptage_colab.ipynb     entraînement et évaluation du détecteur de pieds (Colab)
  stress_colab.ipynb       stress hydrique Sentinel-2 + vérifications (Colab, CPU)
  build_stress.py          source lisible du notebook stress (génère aussi scripts/stress_sentinel2.py)
  build_comptage.py        source lisible du notebook comptage
  build_notebook.py        source lisible du notebook (le régénère)
docs/
  carnet_de_donnees.md     sources, licences, résolution, découpage, fuites
  rapport.md               rapport (4 pages max)
data/README.md             où télécharger chaque jeu de données
models/                    poids des modèles (ONNX maladies, .pt comptage + pieds_modeles.json)
```

## Données et modèles pré-entraînés

Voir [`docs/carnet_de_donnees.md`](docs/carnet_de_donnees.md). Les données ne sont
pas versionnées ; [`data/README.md`](data/README.md) indique comment les récupérer.
