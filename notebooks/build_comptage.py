"""Génère notebooks/comptage_colab.ipynb (détecteur de pieds de bananier).

Comme pour le module maladies, le notebook est écrit en Python pour être relu et versionné proprement.
Les fonctions déjà utilisées par la chaîne (baseline de comptage, appariement à 1 m) sont COPIÉES
depuis le repo au moment de la génération : le notebook évalue exactement le même code que run.py.

Usage : python notebooks/build_comptage.py
"""
import inspect
import sys
from pathlib import Path

import nbformat as nbf

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))
from scripts.evaluer_comptage import apparier  # noqa: E402

SOURCE_COMPTAGE = (RACINE / "bananes" / "comptage.py").read_text(encoding="utf-8")
SOURCE_APPARIER = inspect.getsource(apparier)

CELLULES = []


def md(texte):
    CELLULES.append(nbf.v4.new_markdown_cell(texte.strip()))


def code(texte):
    CELLULES.append(nbf.v4.new_code_cell(texte.strip()))


md("""
# Comptage des pieds de bananier — détecteur appris

**Pourquoi** : le chef de culture a besoin du **nombre de pieds par parcelle** (pieds manquants à replanter,
prévision de récolte, doses d'engrais). Pour décider, l'erreur doit rester autour de **±10 % par parcelle**.
La baseline classique (sommets de NDVI) se trompe de **−78 %** sur canopée fermée : elle est inutilisable.

**Idée** : partir d'un détecteur d'arbres déjà entraîné (DeepForest, RetinaNet, images aériennes à ~10 cm),
lui apprendre ce qu'est un **bananier** avec ReforesTree (Équateur, bananiers annotés), puis l'ajuster
sur nos quelques pieds annotés en zone **ouest** de Guangxi. Le test se fait sur la zone **est**, jamais vue.

| Expérience | Point de départ | Appris sur | Question posée |
|---|---|---|---|
| `baseline_ndvi` | — | — | référence classique (celle de `run.py`) |
| `arbres_brut` | DeepForest arbres | rien | un détecteur d'arbres générique suffit-il ? |
| `refo_8cm` | DeepForest arbres | ReforesTree à 8 cm | apprendre « bananier » sur un autre pays aide-t-il ? |
| `refo_multires` | DeepForest arbres | ReforesTree à 4, 8 et 16 cm | entraîner à plusieurs altitudes de vol rend-il plus robuste ? |
| **`refo_multires_affine_O`** | `refo_multires` | + carrés O1, O2 (46 pieds) | **modèle retenu a priori** |
| `arbres_affine_O` | DeepForest arbres | carrés O1, O2 seulement | ReforesTree sert-il à quelque chose ? |

**Résolution** : le même modèle est testé sur ReforesTree à 4, 8 et 16 cm et sur Guangxi à 8 et 16 cm.
C'est la question terrain « **à quelle altitude faut-il voler ?** ».

**Avant de lancer** :
1. dans ton Drive, crée `MyDrive/piscine-bananeraie/comptage_entree/` et mets-y 4 fichiers :
   `guangxi_plantation.tif`, `parcelles.geojson` (dossier `data/guangxi_pret`), `carres_15m.geojson`,
   `verite_comptage.geojson` (dossier `annotations/guangxi`) ;
2. *Exécution → Modifier le type d'exécution → GPU T4* ;
3. *Exécution → Tout exécuter*.

**Reprise automatique** : ReforesTree réduit et chaque expérience terminée sont gardés sur le Drive.
Si Colab coupe, relance *Tout exécuter* : ce qui est fait est sauté.
""")

md("## 0. Réglages")
code("""
FAIRE_ABLATIONS = True      # False = seulement le modèle retenu (et ce dont il dépend)

RESOLUTION_SOURCE = 0.02    # ReforesTree : 2 cm/pixel
RESOLUTION_CIBLE = 0.08     # Guangxi : 8 cm/pixel
RESOLUTIONS = [0.04, 0.08, 0.16]   # résolutions fabriquées à partir de ReforesTree

TAILLE_PATCH = 400          # fenêtre vue par le réseau (comme DeepForest) : 32 m à 8 cm
PAS_PATCH = 300             # fenêtres qui se chevauchent de 100 px
NMS_IOU = 0.3               # deux boîtes qui se recouvrent à plus de 30 % = un seul pied
DISTANCE_APPARIEMENT_M = 1.0   # règle de l'évaluation officielle

EPOQUES_REFO = 10
EPOQUES_AFFINAGE = 30       # 2 carrés seulement : on les montre 8 fois par époque, retournés/tournés
LR_REFO = 1e-3
LR_AFFINAGE = 5e-4
TAILLE_LOT = 4
GRAINE = 42

MODELE_HF = "weecology/deepforest-tree"
MODE_TEST = False           # petites fausses données pour vérifier que le notebook tourne (ne pas toucher)
""")

md("## 1. Installation et Google Drive")
code("""
import os, sys, json, time, random, shutil, zipfile, tarfile, hashlib, subprocess, io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torchvision
from torchvision.models.detection import RetinaNet
from torchvision.ops import nms
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from PIL import Image
import geopandas as gpd
import rasterio
from rasterio.transform import Affine
from shapely.geometry import Point
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None
random.seed(GRAINE); np.random.seed(GRAINE); torch.manual_seed(GRAINE)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Calcul sur :", DEVICE)

TRAVAIL = Path("/content") if Path("/content").exists() and not MODE_TEST else Path.cwd() / "colab_comptage"
TEMP = TRAVAIL / "temp"
if Path("/content").exists() and not MODE_TEST:
    from google.colab import drive
    drive.mount("/content/drive")
    SORTIE = Path("/content/drive/MyDrive/piscine-bananeraie")
else:
    SORTIE = TRAVAIL / "sortie"
ENTREE = SORTIE / "comptage_entree"          # les 4 fichiers que tu as déposés
DRIVE_REFO = SORTIE / "reforestree_reduit"   # ReforesTree déjà réduit (une archive .tar par résolution)
MODELES = SORTIE / "models"
RES = SORTIE / "resultats_comptage"
for d in [TEMP, ENTREE, DRIVE_REFO, MODELES, RES]:
    d.mkdir(parents=True, exist_ok=True)
cm = lambda r: int(round(r * 100))           # 0.08 -> 8
print("Sauvegardes dans :", SORTIE)
""")

md("""
### Mode test uniquement : fausses données
(Sautée en vrai : sert à vérifier que tout le notebook tourne, sans GPU ni téléchargement.)
""")
code("""
if MODE_TEST:
    from rasterio.transform import from_origin
    from shapely.geometry import box as rect
    rng = np.random.default_rng(0)
    EPOQUES_REFO, EPOQUES_AFFINAGE = 1, 1

    def disques(h, w, centres, rayon_px):
        img = np.zeros((h, w), "float32")
        yy, xx = np.mgrid[0:h, 0:w]
        for x, y in centres:
            img = np.maximum(img, np.clip(1 - np.hypot(xx - x, yy - y) / rayon_px, 0, 1))
        return img

    # Faux ReforesTree : 3 sites x 2 tuiles de 16 m à 2 cm
    faux = TRAVAIL / "faux_refo" / "reforesTree"
    (faux / "tiles" / "processed").mkdir(parents=True, exist_ok=True)
    (faux / "mapping").mkdir(parents=True, exist_ok=True)
    lignes = []
    for site in ["Site A", "Site B", "Site C"]:
        for k in range(2):
            nom = f"{site} RGB_{k}_0_0_800_800.png"
            centres = rng.uniform(80, 720, size=(6, 2))
            v = disques(800, 800, centres, 70)
            rgb = np.stack([0.4 - 0.2 * v, 0.3 + 0.4 * v, 0.2 + 0.0 * v], -1)
            Image.fromarray((np.clip(rgb, 0, 1) * 255).astype("uint8")).save(faux / "tiles" / "processed" / nom)
            for i, (x, y) in enumerate(centres):
                lignes.append({"img_path": nom, "xmin": x - 70, "ymin": y - 70, "xmax": x + 70, "ymax": y + 70,
                               "group": "banana" if i % 3 else "cacao", "AGB": 1.0})
    pd.DataFrame(lignes).to_csv(faux / "mapping" / "final_dataset.csv", index=False)
    FAUX_ZIP = TRAVAIL / "faux_reforestree.zip"
    with zipfile.ZipFile(FAUX_ZIP, "w") as z:
        for p in faux.rglob("*"):
            z.write(p, p.relative_to(faux.parent))

    # Fausse plantation de Guangxi : 80 m x 40 m à 8 cm, 5 bandes, UTM 48N
    px, l, h = 0.08, 1000, 500
    x0, y0 = 780_000.0, 2_560_000.0
    gx, gy = np.meshgrid(np.arange(1.5, 80, 2.5), np.arange(1.5, 40, 2.5))
    plants = np.c_[gx.ravel(), gy.ravel()] + rng.normal(0, 0.3, (gx.size, 2))
    v = disques(h, l, plants / px, 1.2 / px)
    bandes = np.stack([0.04 + 0.0 * v, 0.06 + 0.04 * v, 0.08 - 0.04 * v, 0.2 + 0.3 * v, 0.15 + 0.15 * v])
    profil = dict(driver="GTiff", width=l, height=h, count=5, dtype="uint16", crs="EPSG:32648",
                  transform=from_origin(x0, y0, px, px))
    with rasterio.open(ENTREE / "guangxi_plantation.tif", "w", **profil) as dst:
        dst.write((bandes * 10000).astype("uint16"))
        dst.update_tags(echelle_reflectance="10000")
    noms = ["O1_15m", "O2_15m", "E1_15m", "E2_15m", "E3_15m", "E4_15m"]
    coins = [(2, 3), (20, 20), (42, 3), (60, 3), (42, 22), (62, 22)]
    carres = gpd.GeoDataFrame({"carre": noms, "zone": ["ouest_reglage"] * 2 + ["est_test"] * 4},
                              geometry=[rect(x0 + a, y0 - b - 15, x0 + a + 15, y0 - b) for a, b in coins],
                              crs="EPSG:32648")
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x0 + plants[:, 0], y0 - plants[:, 1]), crs="EPSG:32648")
    verite = gpd.sjoin(pts, carres, predicate="within").drop(columns="index_right")
    carres.to_crs(4326).to_file(ENTREE / "carres_15m.geojson")
    verite.to_crs(4326).to_file(ENTREE / "verite_comptage.geojson")
    gpd.GeoDataFrame({"parcelle_id": ["P01", "P02"]},
                     geometry=[rect(x0, y0 - 40, x0 + 40, y0), rect(x0 + 40, y0 - 40, x0 + 80, y0)],
                     crs="EPSG:32648").to_crs(4326).to_file(ENTREE / "parcelles.geojson")
    print("Fausses données prêtes")
""")

md("""
## 2. ReforesTree : téléchargé et réduit **une seule fois**

ReforesTree (Reiersen et al., 2022, Zenodo 6813783, CC BY 4.0) : 6 parcelles d'agroforesterie en Équateur,
drone RGB à **2 cm/pixel**, chaque arbre dans une boîte avec son espèce (banana, cacao, citrus, fruit, timber…).

On fabrique trois versions **plus grossières** (4, 8 et 16 cm) en moyennant les pixels par blocs : c'est ce
que verrait la même caméra en volant 2, 4 ou 8 fois plus haut. Les boîtes sont mises à l'échelle.
Seules les boîtes **banana** sont des cibles ; les autres arbres restent dans l'image comme **exemples
négatifs** : le modèle doit apprendre à ne pas compter un cacaoyer ou un agrume.

Cinq tuiles sont écartées, comme le recommandent les auteurs (annotations incomplètes).
Archive de 7,5 Go : téléchargée, réduite, puis effacée (≈ 1 Go gardé sur le Drive).
""")
code("""
URL_REFO = "https://zenodo.org/records/6813783/files/reforesTree.zip?download=1"
MD5_REFO = "f6a4a1d8207aeaa5fbab7b21b683a302"
TUILES_EXCLUES = {
    "Carlos Vera Guevara RGB_15_8425_8305_12425_12305.png",
    "Flora Pluas RGB_3_0_11400_4000_15400.png",
    "Flora Pluas RGB_4_0_11578_4000_15578.png",
    "Flora Pluas RGB_23_12782_11400_16782_15400.png",
    "Flora Pluas RGB_24_12782_11578_16782_15578.png",
}
if MODE_TEST:
    URL_REFO, MD5_REFO = str(FAUX_ZIP), None

def telecharger(url, destination):
    \"\"\"Télécharge en affichant la progression, et renvoie l'empreinte MD5 (contrôle d'intégrité).\"\"\"
    md5 = hashlib.md5()
    if url.startswith("/"):
        shutil.copy(url, destination)
        md5.update(Path(destination).read_bytes())
        return md5.hexdigest()
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total, palier = 0, 500e6
        with open(destination, "wb") as f:
            for morceau in r.iter_content(1 << 22):
                f.write(morceau); md5.update(morceau)
                total += len(morceau)
                if total >= palier:
                    print(f"     {total / 1e9:.1f} Go", flush=True); palier += 500e6
    return md5.hexdigest()

def preparer_reforestree():
    manquantes = [r for r in RESOLUTIONS if not (DRIVE_REFO / f"refo_{cm(r)}cm.tar").exists()]
    if not manquantes:
        print("✓ ReforesTree déjà réduit sur le Drive"); return
    print("↓ Téléchargement de ReforesTree (7,5 Go)…", flush=True)
    archive = TEMP / "reforesTree.zip"
    empreinte = telecharger(URL_REFO, archive)
    if MD5_REFO and empreinte != MD5_REFO:
        raise RuntimeError(f"Archive corrompue (md5 {empreinte}) : relance la cellule")
    extrait = TEMP / "refo_extrait"
    shutil.rmtree(extrait, ignore_errors=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(extrait)
    archive.unlink()

    tableau = pd.read_csv(next(extrait.rglob("final_dataset.csv")))
    fichiers = {p.name: p for p in extrait.rglob("*.png") if "__MACOSX" not in p.parts}
    tableau["img_path"] = tableau["img_path"].map(lambda s: Path(str(s)).name)
    tableau = tableau[~tableau.img_path.isin(TUILES_EXCLUES) & tableau.img_path.isin(fichiers)]
    tableau["site"] = tableau.img_path.str.split(" RGB").str[0]
    print("Arbres par espèce :", tableau.group.value_counts().to_dict())

    for r in manquantes:
        facteur = RESOLUTION_SOURCE / r
        dossier = TEMP / f"refo_{cm(r)}cm"
        dossier.mkdir(parents=True, exist_ok=True)
        boites, tuiles = [], []
        for nom, arbres in tableau.groupby("img_path"):
            img = Image.open(fichiers[nom])
            if img.mode == "RGBA":     # zones transparentes (hors vol) -> noir
                fond = Image.new("RGB", img.size); fond.paste(img, mask=img.split()[3]); img = fond
            img = img.convert("RGB")
            # BOX = moyenne des pixels par blocs : simule un capteur plus grossier (vol plus haut)
            petite = img.resize((round(img.width * facteur), round(img.height * facteur)), Image.BOX)
            petite.save(dossier / nom)
            bananes = arbres[arbres.group.str.lower() == "banana"]
            for b in bananes.itertuples():
                boites.append({"tuile": nom, "xmin": b.xmin * facteur, "ymin": b.ymin * facteur,
                               "xmax": b.xmax * facteur, "ymax": b.ymax * facteur})
            tuiles.append({"tuile": nom, "site": arbres.site.iloc[0], "n_bananiers": len(bananes),
                           "n_autres_arbres": len(arbres) - len(bananes)})
        pd.DataFrame(boites).to_csv(dossier / "boites.csv", index=False)
        pd.DataFrame(tuiles).to_csv(dossier / "tuiles.csv", index=False)
        with tarfile.open(DRIVE_REFO / f"refo_{cm(r)}cm.tar", "w") as t:
            t.add(dossier, arcname=dossier.name)
        print(f"  ✓ {cm(r)} cm : {len(tuiles)} tuiles, {len(boites)} bananiers")
    shutil.rmtree(extrait)

preparer_reforestree()

REFO = {}   # résolution -> (images en mémoire, boîtes par tuile, tableau des tuiles)
for r in RESOLUTIONS:
    dossier = TEMP / f"refo_{cm(r)}cm"
    if not (dossier / "tuiles.csv").exists():
        with tarfile.open(DRIVE_REFO / f"refo_{cm(r)}cm.tar") as t:
            t.extractall(TEMP)
    tuiles = pd.read_csv(dossier / "tuiles.csv")
    boites = pd.read_csv(dossier / "boites.csv")
    images = {n: np.array(Image.open(dossier / n).convert("RGB")) for n in tuiles.tuile}
    par_tuile = {n: boites[boites.tuile == n][["xmin", "ymin", "xmax", "ymax"]].to_numpy("float32")
                 for n in tuiles.tuile}
    REFO[r] = (images, par_tuile, tuiles)

tuiles = REFO[RESOLUTION_CIBLE][2]
par_site = tuiles.groupby("site")[["n_bananiers", "n_autres_arbres"]].sum()
print(par_site)

# Taille typique d'un bananier vu du dessus : sert à fabriquer les boîtes à partir de nos points de Guangxi
b = pd.concat([pd.DataFrame(v, columns=["xmin", "ymin", "xmax", "ymax"]) for v in REFO[RESOLUTION_CIBLE][1].values()])
COTE_BANANIER_M = float(np.clip(np.median(np.sqrt((b.xmax - b.xmin) * (b.ymax - b.ymin))) * RESOLUTION_CIBLE, 2, 5))
print(f"Côté médian d'une boîte de bananier ReforesTree : {COTE_BANANIER_M:.2f} m")
""")

md("""
### Découpage honnête : par **site**
Une parcelle entière d'Équateur est gardée de côté pour la validation (celle dont le nombre de bananiers est le
plus proche de la médiane). Des tuiles voisines d'un même site se ressemblent trop pour servir de test.
""")
code("""
ecart = (par_site.n_bananiers - par_site.n_bananiers.median()).abs()
SITE_VALIDATION = ecart.idxmin()
SITES_ENTRAINEMENT = [s for s in par_site.index if s != SITE_VALIDATION]
print("Validation :", SITE_VALIDATION, "| Entraînement :", SITES_ENTRAINEMENT)
pd.DataFrame({"site": par_site.index, "role": ["validation" if s == SITE_VALIDATION else "entrainement"
              for s in par_site.index], **{c: par_site[c].values for c in par_site.columns}}
             ).to_csv(RES / "decoupage_reforestree.csv", index=False)
""")

md("""
## 3. Guangxi : image, carrés annotés, vérité terrain

L'orthomosaïque est multispectrale (bleu, vert, rouge, proche infrarouge, red-edge, en réflectance).
Le réseau a été pré-entraîné sur des **photos couleur** : on recompose une image RGB.
- étirement de chaque bande entre ses centiles 2 et 98, calculés **une fois** sur toute l'image (pas carré par
  carré). Pas le maximum : quelques surfaces très claires (bâtiments, bâches) écraseraient tout le reste ;
- correction gamma (puissance 1/1,5) : une réflectance est linéaire, une photo d'appareil ne l'est pas ;
  sans elle, l'ombre entre les feuilles est trop noire.

On fabrique aussi une version à **16 cm** (moyenne de 2 × 2 pixels) pour simuler un vol deux fois plus haut.
""")
code("""
with rasterio.open(ENTREE / "guangxi_plantation.tif") as src:
    bandes = src.read().astype("float32")
    TRANSFORM_GX, CRS_GX = src.transform, src.crs
    echelle = float(src.tags().get("echelle_reflectance", 10000))
bandes /= echelle
valide = bandes.sum(0) > 0
rgb = np.stack([bandes[2], bandes[1], bandes[0]], -1)                # bleu, vert, ROUGE -> rouge, vert, bleu
bas, haut = np.percentile(rgb[valide][::10], [2, 98], axis=0)
RGB_GX = {RESOLUTION_CIBLE: (np.clip((rgb - bas) / (haut - bas), 0, 1) ** (1 / 1.5) * 255).astype("uint8")}
TRANSFORM = {RESOLUTION_CIBLE: TRANSFORM_GX}
PIR_ROUGE = {"pir": bandes[3], "rouge": bandes[2]}                     # pour la baseline NDVI
img16 = Image.fromarray(RGB_GX[RESOLUTION_CIBLE])
RGB_GX[0.16] = np.array(img16.resize((img16.width // 2, img16.height // 2), Image.BOX))
TRANSFORM[0.16] = TRANSFORM_GX * Affine.scale(2)
del rgb, img16

CARRES = gpd.read_file(ENTREE / "carres_15m.geojson").to_crs(CRS_GX)
VERITE = gpd.read_file(ENTREE / "verite_comptage.geojson").to_crs(CRS_GX)
VERITE = pd.DataFrame({"carre": VERITE.carre, "x": VERITE.geometry.x, "y": VERITE.geometry.y})
ZONE = dict(zip(CARRES.carre, CARRES.zone))
CARRES_O = [c for c in CARRES.carre if ZONE[c] == "ouest_reglage"]
CARRES_E = [c for c in CARRES.carre if ZONE[c] == "est_test"]
print(VERITE.groupby("carre").size().rename("pieds_annotes").to_frame().assign(zone=lambda t: t.index.map(ZONE)))

def decouper(res, geometrie, marge_m=0.0):
    \"\"\"Morceau de l'image RGB autour d'une géométrie. Renvoie (image, transform du morceau).\"\"\"
    xmin, ymin, xmax, ymax = geometrie.bounds
    inverse = ~TRANSFORM[res]
    c0, r0 = inverse * (xmin - marge_m, ymax + marge_m)
    c1, r1 = inverse * (xmax + marge_m, ymin - marge_m)
    h, w = RGB_GX[res].shape[:2]
    c0, r0 = max(int(np.floor(c0)), 0), max(int(np.floor(r0)), 0)
    c1, r1 = min(int(np.ceil(c1)), w), min(int(np.ceil(r1)), h)
    return RGB_GX[res][r0:r1, c0:c1], TRANSFORM[res] * Affine.translation(c0, r0)
""")

md("""
## 4. Le réseau et les données d'entraînement

**Le réseau** : RetinaNet (ResNet-50), le détecteur de DeepForest. On charge directement ses poids
pré-entraînés sur les couronnes d'arbres (Hugging Face `weecology/deepforest-tree`). La boucle d'entraînement
est écrite ici, en 20 lignes, plutôt que cachée dans une bibliothèque : on sait exactement ce qui est fait.

Trois choix :
- toutes les fenêtres font **400 × 400 px** (les plus petites sont complétées par du noir). Le réseau agrandit
  chaque image à 800 px : avec une taille fixe, un bananier garde la même taille apparente à l'entraînement
  et à l'usage ;
- les couches de **normalisation sont figées** (comme dans DeepForest) et les premières couches du ResNet
  gelées : avec des lots de 4 images et peu de données, les ré-estimer serait instable (même leçon que pour
  le module maladies) ;
- **augmentations** : retournements et rotations de 90° (vu du ciel, il n'y a pas de haut ni de bas) et
  légères variations de couleur (la caméra d'Équateur n'est pas celle de Guangxi).
""")
code("""
def creer_reseau():
    \"\"\"RetinaNet ResNet-50 à 1 classe, avec les poids DeepForest « arbres ».\"\"\"
    # Même architecture que DeepForest : ResNet-50 + FPN, normalisation figée, couches basses gelées
    resnet = torchvision.models.resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(resnet, 3, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256))
    reseau = RetinaNet(backbone, num_classes=1, score_thresh=0.05, nms_thresh=NMS_IOU,
                       detections_per_img=500, min_size=2 * TAILLE_PATCH, max_size=2 * TAILLE_PATCH)
    if not MODE_TEST:
        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            poids = load_file(hf_hub_download(MODELE_HF, "model.safetensors"))
        except Exception as e:      # plan B : passer par la bibliothèque DeepForest elle-même
            print("Téléchargement direct impossible (", e, ") -> installation de deepforest")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "deepforest"], check=True)
            from deepforest.models.retinanet import RetinaNetHub
            poids = RetinaNetHub.from_pretrained(MODELE_HF).state_dict()
        poids = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in poids.items()}
        manquants, inattendus = reseau.load_state_dict(poids, strict=False)
        manquants = [k for k in manquants if not k.endswith("num_batches_tracked")]
        assert not manquants and not inattendus, (manquants[:5], inattendus[:5])
    return reseau.to(DEVICE)

def figer_bn(reseau):
    for m in reseau.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

def sauver(reseau, nom):
    # demi-précision : 2 fois plus léger (≈ 70 Mo), aucune différence mesurable sur les détections
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in reseau.state_dict().items()},
               MODELES / f"pieds_{nom}.pt")

def charger(nom):
    reseau = creer_reseau()
    etat = torch.load(MODELES / f"pieds_{nom}.pt", map_location=DEVICE)
    reseau.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in etat.items()})
    return reseau

def positions(n, taille, pas):
    \"\"\"Débuts des fenêtres le long d'un côté de n pixels (la dernière colle au bord).\"\"\"
    if n <= taille:
        return [0]
    p = list(range(0, n - taille + 1, pas))
    return p if p[-1] == n - taille else p + [n - taille]

def boites_dans(boites, x0, y0, w, h):
    \"\"\"Boîtes ramenées dans une fenêtre ; on garde celles dont au moins la moitié est dedans.\"\"\"
    if len(boites) == 0:
        return boites.reshape(0, 4)
    b = boites - np.array([x0, y0, x0, y0], "float32")
    c = b.copy()
    c[:, [0, 2]] = c[:, [0, 2]].clip(0, w); c[:, [1, 3]] = c[:, [1, 3]].clip(0, h)
    aire = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    aire_c = (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
    garde = (aire_c >= 0.5 * aire) & (c[:, 2] - c[:, 0] >= 4) & (c[:, 3] - c[:, 1] >= 4)
    return c[garde]

def completer(img, taille=TAILLE_PATCH):
    \"\"\"Complète avec du noir jusqu'à taille x taille (en haut à gauche : les boîtes ne bougent pas).\"\"\"
    h, w = img.shape[:2]
    if h >= taille and w >= taille:
        return img
    plein = np.zeros((max(h, taille), max(w, taille), 3), img.dtype)
    plein[:h, :w] = img
    return plein

def augmenter(img, b):
    h, w = img.shape[:2]
    if random.random() < 0.5:
        img = img[:, ::-1]; b = b.copy(); b[:, [0, 2]] = w - b[:, [2, 0]]
    if random.random() < 0.5:
        img = img[::-1]; b = b.copy(); b[:, [1, 3]] = h - b[:, [3, 1]]
    for _ in range(random.randint(0, 3)):           # rotation de 90° : (x, y) -> (y, w - x)
        h, w = img.shape[:2]
        img = np.rot90(img)
        b = np.c_[b[:, 1], w - b[:, 2], b[:, 3], w - b[:, 0]]
    img = img.astype("float32") * np.random.uniform(0.8, 1.2, 3) + np.random.uniform(-12, 12, 3)
    return np.clip(img, 0, 255).astype("uint8"), b.astype("float32")

class JeuFenetres(torch.utils.data.Dataset):
    \"\"\"Chaque exemple = (image complète, ses boîtes, x0, y0, largeur, hauteur de la fenêtre).\"\"\"
    def __init__(self, fenetres, augmentation=True):
        self.fenetres, self.augmentation = fenetres, augmentation
    def __len__(self):
        return len(self.fenetres)
    def __getitem__(self, i):
        img, boites, x0, y0, w, h = self.fenetres[i]
        morceau = img[y0:y0 + h, x0:x0 + w]
        b = boites_dans(boites, x0, y0, morceau.shape[1], morceau.shape[0])
        if self.augmentation:
            morceau, b = augmenter(morceau, b)
        morceau = completer(np.ascontiguousarray(morceau))
        image = torch.from_numpy(morceau).permute(2, 0, 1).float() / 255
        return image, {"boxes": torch.from_numpy(b.reshape(-1, 4)), "labels": torch.zeros(len(b), dtype=torch.int64)}

def fenetres_reforestree(resolutions, sites):
    liste = []
    for r in resolutions:
        images, boites, tuiles = REFO[r]
        for t in tuiles[tuiles.site.isin(sites)].itertuples():
            img = images[t.tuile]
            for y0 in positions(img.shape[0], TAILLE_PATCH, PAS_PATCH):
                for x0 in positions(img.shape[1], TAILLE_PATCH, PAS_PATCH):
                    liste.append((img, boites[t.tuile], x0, y0, TAILLE_PATCH, TAILLE_PATCH))
    return liste

def fenetres_guangxi_O(repetitions=8):
    \"\"\"Les carrés O1, O2 exactement (pas de marge : hors du carré, rien n'est annoté).
    Chaque point annoté devient une boîte de la taille médiane d'un bananier ReforesTree.\"\"\"
    liste = []
    for carre in CARRES_O:
        img, t = decouper(RESOLUTION_CIBLE, CARRES[CARRES.carre == carre].geometry.iloc[0])
        v = VERITE[VERITE.carre == carre]
        cols, lignes = (~t) * (v.x.to_numpy(), v.y.to_numpy())
        demi = COTE_BANANIER_M / RESOLUTION_CIBLE / 2
        boites = np.c_[cols - demi, lignes - demi, cols + demi, lignes + demi].astype("float32")
        boites = boites_dans(boites, 0, 0, img.shape[1], img.shape[0])
        liste += [(img, boites, 0, 0, img.shape[1], img.shape[0])] * repetitions
    return liste

def entrainer(reseau, fenetres, epoques, lr, nom):
    if MODE_TEST:
        fenetres = fenetres[:8]
    chargeur = torch.utils.data.DataLoader(JeuFenetres(fenetres), batch_size=TAILLE_LOT, shuffle=True,
                                           num_workers=2 if DEVICE == "cuda" else 0,
                                           collate_fn=lambda lot: tuple(zip(*lot)))
    reseau.train(); figer_bn(reseau)
    parametres = [p for p in reseau.parameters() if p.requires_grad]
    optimiseur = torch.optim.SGD(parametres, lr=lr, momentum=0.9, weight_decay=1e-4)
    planning = torch.optim.lr_scheduler.CosineAnnealingLR(optimiseur, T_max=epoques * len(chargeur))
    print(f"  {nom} : {len(fenetres)} fenêtres, {epoques} époques", flush=True)
    for epoque in range(epoques):
        reseau.train(); figer_bn(reseau)
        pertes, t0 = [], time.time()
        for images, cibles in chargeur:
            images = [i.to(DEVICE) for i in images]
            cibles = [{k: v.to(DEVICE) for k, v in c.items()} for c in cibles]
            perte = sum(reseau(images, cibles).values())
            optimiseur.zero_grad(); perte.backward()
            torch.nn.utils.clip_grad_norm_(parametres, 10.0)
            optimiseur.step(); planning.step()
            pertes.append(perte.item())
        print(f"    époque {epoque + 1}/{epoques} : perte {np.mean(pertes):.3f} ({time.time() - t0:.0f} s)", flush=True)
    reseau.eval()
    return reseau
""")

md("""
## 5. Prédiction et évaluation

**Prédiction sur une grande image** : fenêtres de 400 px qui se chevauchent, puis suppression des doublons
(deux boîtes qui se recouvrent à plus de 30 % = un seul pied). Un pied = **le centre** de sa boîte.

**Évaluation** : exactement la règle officielle, avec la **même fonction** que `scripts/evaluer_comptage.py`
(appariement un-à-un à 1 m). On mesure aussi l'erreur de **comptage** par carré : c'est elle qui compte
pour le chef de culture.

**Seuil de confiance** : choisi pour maximiser le F1 sur les carrés **ouest** (O1, O2), puis appliqué tel quel
à l'est. Pour les modèles affinés sur O, ce seuil est choisi sur des carrés vus à l'entraînement (signalé
dans le tableau) ; le test à l'est reste, lui, honnête.
""")
code("# --- copie de scripts/evaluer_comptage.py ---\nDISTANCE_MAX_M = DISTANCE_APPARIEMENT_M\n"
     + SOURCE_APPARIER + '''

@torch.no_grad()
def predire(reseau, img, device=None, taille=TAILLE_PATCH, pas=TAILLE_PATCH - 80):
    """Boîtes (pixels de img) + scores, sur une image de n'importe quelle taille."""
    device = device or DEVICE
    reseau.eval()
    toutes, scores = [], []
    fenetres = [(x0, y0) for y0 in positions(img.shape[0], taille, pas) for x0 in positions(img.shape[1], taille, pas)]
    for i in range(0, len(fenetres), 8):
        lot = fenetres[i:i + 8]
        images = [torch.from_numpy(completer(np.ascontiguousarray(img[y0:y0 + taille, x0:x0 + taille])))
                  .permute(2, 0, 1).float().div(255).to(device) for x0, y0 in lot]
        for (x0, y0), sortie in zip(lot, reseau(images)):
            toutes.append(sortie["boxes"].cpu() + torch.tensor([x0, y0, x0, y0]))
            scores.append(sortie["scores"].cpu())
    if not toutes or sum(len(s) for s in scores) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    boites, scores = torch.cat(toutes), torch.cat(scores)
    garde = nms(boites, scores, NMS_IOU)        # doublons dans les zones de chevauchement
    return boites[garde].numpy(), scores[garde].numpy()

def centres_carte(boites, transform):
    cols, lignes = (boites[:, 0] + boites[:, 2]) / 2, (boites[:, 1] + boites[:, 3]) / 2
    return np.c_[transform * (cols, lignes)] if len(boites) else np.zeros((0, 2))

def predictions_carres(reseau, res):
    """Toutes les détections (score >= 0,05) dont le centre tombe DANS chaque carré."""
    lignes = []
    for c in CARRES.itertuples():
        img, t = decouper(res, c.geometry, marge_m=5.0)   # 5 m de contexte autour du carré
        boites, scores = predire(reseau, img)
        for (x, y), s in zip(centres_carte(boites, t), scores):
            if c.geometry.contains(Point(x, y)):
                lignes.append({"carre": c.carre, "x": x, "y": y, "score": float(s)})
    return pd.DataFrame(lignes, columns=["carre", "x", "y", "score"])

def tableau_carres(pred, verite, seuil, carres):
    lignes = []
    for carre in carres:
        p = pred[(pred.carre == carre) & (pred.score >= seuil)][["x", "y"]].to_numpy()
        v = verite[verite.carre == carre][["x", "y"]].to_numpy()
        lignes.append({"carre": carre, "n_vrai": len(v), "n_pred": len(p),
                       "vp": apparier(p, v, DISTANCE_APPARIEMENT_M)})
    return pd.DataFrame(lignes)

def resumer(t):
    vp, n_pred, n_vrai = t.vp.sum(), t.n_pred.sum(), t.n_vrai.sum()
    p, r = vp / max(n_pred, 1), vp / max(n_vrai, 1)
    return {"precision": round(p, 3), "rappel": round(r, 3), "f1": round(2 * p * r / max(p + r, 1e-9), 3),
            "n_vrai": int(n_vrai), "n_pred": int(n_pred),
            "erreur_comptage": round((n_pred - n_vrai) / max(n_vrai, 1), 3),
            "erreur_abs_moy_par_carre": round(float((t.n_pred - t.n_vrai).abs().mean()), 1)}

SEUILS = np.round(np.arange(0.05, 0.96, 0.05), 2)

def choisir_seuil(pred, verite, carres):
    f1 = [resumer(tableau_carres(pred, verite, s, carres))["f1"] for s in SEUILS]
    return float(SEUILS[int(np.argmax(f1))])

def evaluer_guangxi(reseau, nom, res=RESOLUTION_CIBLE, seuil=None):
    pred = predictions_carres(reseau, res)
    pred.to_csv(RES / f"predictions_{nom}_{cm(res)}cm.csv", index=False)
    seuil = choisir_seuil(pred, VERITE, CARRES_O) if seuil is None else seuil
    detail = tableau_carres(pred, VERITE, seuil, CARRES.carre)
    detail["zone"] = detail.carre.map(ZONE)
    detail.to_csv(RES / f"carres_{nom}_{cm(res)}cm.csv", index=False)
    return {"seuil": seuil,
            "ouest": resumer(detail[detail.carre.isin(CARRES_O)]),
            "est_test": resumer(detail[detail.carre.isin(CARRES_E)])}

def evaluer_reforestree(reseau, res):
    """Site de validation d'Équateur à une résolution donnée (seuil choisi sur ce site : comparaison relative)."""
    images, boites, tuiles = REFO[res]
    pred, verite = [], []
    for t in tuiles[tuiles.site == SITE_VALIDATION].itertuples():
        b, s = predire(reseau, images[t.tuile])
        for (x, y), sc in zip(np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2] * res, s):
            pred.append({"carre": t.tuile, "x": x, "y": y, "score": float(sc)})
        v = boites[t.tuile]
        for x, y in np.c_[(v[:, 0] + v[:, 2]) / 2, (v[:, 1] + v[:, 3]) / 2] * res:
            verite.append({"carre": t.tuile, "x": x, "y": y})
    pred = pd.DataFrame(pred, columns=["carre", "x", "y", "score"])
    verite = pd.DataFrame(verite, columns=["carre", "x", "y"])
    noms = list(tuiles[tuiles.site == SITE_VALIDATION].tuile)
    seuil = choisir_seuil(pred, verite, noms)
    return {"seuil": seuil, **resumer(tableau_carres(pred, verite, seuil, noms))}
''')

md("""
## 6. La baseline de `run.py`, évaluée de la même façon

Le code ci-dessous est une **copie exacte** de `bananes/comptage.py` (copié automatiquement à la génération
du notebook) : sommets de NDVI lissé, distance minimale de 1,5 m entre deux pieds.
""")
code("# --- copie de bananes/comptage.py ---\n" + SOURCE_COMPTAGE + '''
# --- fin de la copie ---

def evaluer_baseline():
    ndvi = (PIR_ROUGE["pir"] - PIR_ROUGE["rouge"]) / (PIR_ROUGE["pir"] + PIR_ROUGE["rouge"] + 1e-6)
    lignes = []
    for c in CARRES.itertuples():
        xmin, ymin, xmax, ymax = c.geometry.buffer(5.0).bounds
        inverse = ~TRANSFORM_GX
        c0, r0 = [max(int(v), 0) for v in inverse * (xmin, ymax)]
        c1, r1 = [int(np.ceil(v)) for v in inverse * (xmax, ymin)]
        morceau = ndvi[r0:r1, c0:c1]
        img = {"taille_pixel": RESOLUTION_CIBLE, "nodata": np.zeros(morceau.shape, bool),
               "transform": TRANSFORM_GX * Affine.translation(c0, r0), "crs": CRS_GX}
        pieds = detecter_pieds(morceau, img, "ndvi")
        for p in pieds[pieds.within(c.geometry)].geometry:
            lignes.append({"carre": c.carre, "x": p.x, "y": p.y, "score": 1.0})
    pred = pd.DataFrame(lignes, columns=["carre", "x", "y", "score"])
    pred.to_csv(RES / "predictions_baseline_ndvi_8cm.csv", index=False)
    detail = tableau_carres(pred, VERITE, 0.5, CARRES.carre)
    detail["zone"] = detail.carre.map(ZONE)
    detail.to_csv(RES / "carres_baseline_ndvi_8cm.csv", index=False)
    return {"seuil": None, "ouest": resumer(detail[detail.carre.isin(CARRES_O)]),
            "est_test": resumer(detail[detail.carre.isin(CARRES_E)])}

baseline = evaluer_baseline()
print(json.dumps(baseline, indent=1))
''')

md("""
## 7. Les expériences (avec reprise automatique)

Chaque expérience terminée écrit `resultat_<nom>.json` et ses poids `pieds_<nom>.pt` sur le Drive.
Compte environ 10 à 15 min par entraînement sur ReforesTree (T4), 2 min par affinage.
""")
code("""
EXPERIENCES = {
    # nom : (point de départ, données d'entraînement, époques, lr)
    "arbres_brut":            (None, None, 0, 0),
    "refo_8cm":               (None, lambda: fenetres_reforestree([RESOLUTION_CIBLE], SITES_ENTRAINEMENT), EPOQUES_REFO, LR_REFO),
    "refo_multires":          (None, lambda: fenetres_reforestree(RESOLUTIONS, SITES_ENTRAINEMENT), EPOQUES_REFO, LR_REFO),
    "refo_multires_affine_O": ("refo_multires", fenetres_guangxi_O, EPOQUES_AFFINAGE, LR_AFFINAGE),
    "arbres_affine_O":        (None, fenetres_guangxi_O, EPOQUES_AFFINAGE, LR_AFFINAGE),
}
RETENU = "refo_multires_affine_O"
a_faire = list(EXPERIENCES) if FAIRE_ABLATIONS else ["refo_multires", RETENU]

def lancer(nom):
    fichier = RES / f"resultat_{nom}.json"
    if fichier.exists():
        print(f"✓ {nom} : déjà fait"); return json.loads(fichier.read_text())
    depart, donnees, epoques, lr = EXPERIENCES[nom]
    if depart is not None and not (MODELES / f"pieds_{depart}.pt").exists():
        lancer(depart)
    torch.manual_seed(GRAINE); random.seed(GRAINE); np.random.seed(GRAINE)
    reseau = charger(depart) if depart else creer_reseau()
    t0 = time.time()
    if donnees is not None:
        reseau = entrainer(reseau, donnees(), epoques, lr, nom)
    duree = time.time() - t0
    sauver(reseau, nom)
    resultat = {"nom": nom, "depart": depart or MODELE_HF, "duree_entrainement_s": round(duree),
                "seuil_choisi_sur_carres_vus": nom.endswith("affine_O"),
                "guangxi_8cm": evaluer_guangxi(reseau, nom),
                "reforestree_validation": {f"{cm(r)}cm": evaluer_reforestree(reseau, r) for r in RESOLUTIONS}}
    # Vol 2x plus haut : même seuil qu'à 8 cm (on ne re-règle rien)
    resultat["guangxi_16cm"] = evaluer_guangxi(reseau, nom, 0.16, seuil=resultat["guangxi_8cm"]["seuil"])
    fichier.write_text(json.dumps(resultat, indent=1))
    print(f"  {nom} — est (test) :", resultat["guangxi_8cm"]["est_test"])
    del reseau; torch.cuda.empty_cache() if DEVICE == "cuda" else None
    return resultat

RESULTATS = {nom: lancer(nom) for nom in a_faire}
""")

md("## 8. Tableaux pour le rapport")
code("""
lignes = [{"experience": "baseline_ndvi", "seuil": None, **{f"est_{k}": v for k, v in baseline["est_test"].items()}}]
for nom, r in RESULTATS.items():
    g8, g16 = r["guangxi_8cm"], r["guangxi_16cm"]
    lignes.append({"experience": nom, "seuil": g8["seuil"],
                   **{f"est_{k}": v for k, v in g8["est_test"].items()},
                   "est_f1_16cm": g16["est_test"]["f1"], "est_erreur_16cm": g16["est_test"]["erreur_comptage"],
                   "ouest_f1": g8["ouest"]["f1"], "seuil_vu_a_l_entrainement": r["seuil_choisi_sur_carres_vus"],
                   "entrainement_min": round(r["duree_entrainement_s"] / 60, 1)})
ablations = pd.DataFrame(lignes)
ablations.to_csv(RES / "ablations_comptage.csv", index=False)

resolution = pd.DataFrame([{"experience": nom, "resolution": k, **v}
                           for nom, r in RESULTATS.items() for k, v in r["reforestree_validation"].items()])
resolution.to_csv(RES / "resolution_reforestree.csv", index=False)

pd.set_option("display.width", 200)
print("=== Guangxi, zone EST (test), 8 cm ===")
print(ablations[["experience", "seuil", "est_precision", "est_rappel", "est_f1", "est_n_vrai", "est_n_pred",
                 "est_erreur_comptage", "est_erreur_abs_moy_par_carre", "est_f1_16cm", "est_erreur_16cm"]].to_string(index=False))
print("\\n=== ReforesTree, site de validation, selon la résolution (F1) ===")
print(resolution.pivot(index="experience", columns="resolution", values="f1").to_string())
""")

md("""
## 9. Temps de calcul sur **CPU** (sobriété)

On mesure le temps pour traiter **1 hectare** (100 m × 100 m) avec le modèle retenu, sur le processeur
seul — c'est ce dont disposera la chaîne `run.py` sur un ordinateur ordinaire.
""")
code("""
reseau_cpu = charger(RETENU).cpu().eval()
torch.set_num_threads(os.cpu_count())
h, w = RGB_GX[RESOLUTION_CIBLE].shape[:2]
cote = min(int(100 / RESOLUTION_CIBLE), h, w)
y0, x0 = (h - cote) // 2, (w - cote) // 2
bloc = RGB_GX[RESOLUTION_CIBLE][y0:y0 + cote, x0:x0 + cote]
t0 = time.time()
_ = predire(reseau_cpu, bloc, device="cpu")
surface_ha = (cote * RESOLUTION_CIBLE) ** 2 / 10_000
temps_cpu = {"secondes_par_ha": round((time.time() - t0) / surface_ha, 1), "coeurs_cpu": os.cpu_count(),
             "taille_modele_mo": round((MODELES / f"pieds_{RETENU}.pt").stat().st_size / 1e6, 1)}
(RES / "temps_cpu.json").write_text(json.dumps(temps_cpu, indent=1))
print(temps_cpu)
del reseau_cpu
""")

md("""
## 10. Comptage de toute la plantation (modèle retenu)

Le produit pour le terrain : un point par pied (ouvrable dans QGIS) et le **nombre de pieds par parcelle**.
""")
code("""
reseau = charger(RETENU)
seuil = RESULTATS[RETENU]["guangxi_8cm"]["seuil"]
boites, scores = predire(reseau, RGB_GX[RESOLUTION_CIBLE])
garde = scores >= seuil
xy = centres_carte(boites[garde], TRANSFORM_GX)
pieds = gpd.GeoDataFrame({"score": scores[garde].round(3)}, geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=CRS_GX)
parcelles = gpd.read_file(ENTREE / "parcelles.geojson").to_crs(CRS_GX)
pieds = gpd.sjoin(pieds, parcelles[["parcelle_id", "geometry"]], predicate="within").drop(columns="index_right")
comptes = pieds.groupby("parcelle_id").size().rename("n_pieds").reset_index()
comptes = parcelles[["parcelle_id", "geometry"]].merge(comptes, how="left").fillna({"n_pieds": 0})
comptes["surface_ha"] = (comptes.area / 10_000).round(3)
comptes["densite_pieds_ha"] = (comptes.n_pieds / comptes.surface_ha).round(0)
pieds.to_crs(4326).to_file(RES / "pieds_plantation.geojson", driver="GeoJSON")
comptes.drop(columns="geometry").to_csv(RES / "comptage_par_parcelle.csv", index=False)
print(f"{len(pieds)} pieds détectés (seuil {seuil})")
print(comptes.drop(columns="geometry").sort_values("parcelle_id").to_string(index=False))
""")

md("""
## 11. Images pour le rapport : vérité (vert) et détections (rouge)

Une ligne par carré de test. À gauche la baseline NDVI, à droite le modèle retenu.
**Regarde-les** : ce sont tes cas de réussite et d'échec à commenter à la soutenance.
""")
code("""
def dessiner(ax, carre, pred, seuil, titre):
    geom = CARRES[CARRES.carre == carre].geometry.iloc[0]
    img, t = decouper(RESOLUTION_CIBLE, geom)
    ax.imshow(img); ax.axis("off"); ax.set_title(titre, fontsize=9)
    inverse = ~t
    v = VERITE[VERITE.carre == carre]
    ax.scatter(*inverse * (v.x.to_numpy(), v.y.to_numpy()), s=40, facecolors="none", edgecolors="lime", linewidths=1.5)
    p = pred[(pred.carre == carre) & (pred.score >= (seuil or 0))]
    ax.scatter(*inverse * (p.x.to_numpy(), p.y.to_numpy()), s=12, c="red", marker="x")

pred_base = pd.read_csv(RES / "predictions_baseline_ndvi_8cm.csv")
pred_ret = pd.read_csv(RES / f"predictions_{RETENU}_8cm.csv")
detail_base = pd.read_csv(RES / "carres_baseline_ndvi_8cm.csv").set_index("carre")
detail_ret = pd.read_csv(RES / f"carres_{RETENU}_8cm.csv").set_index("carre")
fig, axes = plt.subplots(len(CARRES_E), 2, figsize=(8, 4 * len(CARRES_E)))
for i, carre in enumerate(CARRES_E):
    for j, (nom, pred, det, s) in enumerate([("baseline NDVI", pred_base, detail_base, None),
                                             ("modèle retenu", pred_ret, detail_ret, seuil)]):
        d = det.loc[carre]
        dessiner(axes[i, j], carre, pred, s, f"{carre} — {nom} : {d.n_pred} prédits / {d.n_vrai} vrais")
plt.tight_layout()
plt.savefig(RES / "exemples_detection.png", dpi=110)
""")

md("""
## Ce qu'il faut récupérer

Dans `MyDrive/piscine-bananeraie/` (**pas** `reforestree_reduit`, inutile pour le repo) :
- `models/pieds_refo_multires_affine_O.pt` → dossier `models/` du repo (≈ 70 Mo) ;
- tout `resultats_comptage/` → `docs/resultats_comptage/` : `ablations_comptage.csv`,
  `resolution_reforestree.csv`, `temps_cpu.json`, `comptage_par_parcelle.csv`, `pieds_plantation.geojson`,
  `exemples_detection.png`, `decoupage_reforestree.csv`.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = CELLULES
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}
cible = Path(__file__).with_name("comptage_colab.ipynb")
nbf.write(nb, cible)
print("Notebook écrit :", cible)
