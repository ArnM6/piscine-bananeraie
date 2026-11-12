"""Génère notebooks/stress_colab.ipynb (stress hydrique par parcelle, Sentinel-2).

Comme pour les autres modules, le notebook est écrit en Python pour être relu et versionné proprement.
Le code de calcul (bananes/stress.py) est COPIÉ dans le notebook au moment de la génération :
le notebook utilise exactement les mêmes fonctions que run.py.
Les parcelles, la vigueur drone (bulletin) et les points terrain sont aussi intégrés : rien à déposer sur le Drive.

Usage : python notebooks/build_stress.py
"""
import json
from pathlib import Path

import nbformat as nbf
import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
SOURCE_STRESS = (RACINE / "bananes" / "stress.py").read_text(encoding="utf-8")
PARCELLES = (RACINE / "data" / "guangxi_pret" / "parcelles.geojson").read_text(encoding="utf-8")

bulletin = pd.read_csv(RACINE / "resultats" / "bulletin.csv")
VIGUEUR_DRONE = dict(zip(bulletin.parcelle_id, bulletin.part_vigueur_faible.round(3)))
points = json.loads((RACINE / "data" / "guangxi_pret" / "reference_terrain.geojson").read_text(encoding="utf-8"))
FUSARIOSE = {}
SAINS = {}
for f in points["features"]:
    p = f["properties"]
    if p.get("parcelle_id"):
        cible = FUSARIOSE if p["etat"] == "fusariose" else SAINS
        cible[p["parcelle_id"]] = cible.get(p["parcelle_id"], 0) + 1

# NDVI médian du drone par parcelle (sortie de run.py) : sert à dater le vol avec Sentinel-2
NDVI_DRONE = {}
_tif = RACINE / "resultats" / "guangxi_plantation_ndvi.tif"
if _tif.exists():
    import numpy as np
    import geopandas as gpd
    import rasterio
    from rasterio.mask import mask
    _p = gpd.read_file(RACINE / "data" / "guangxi_pret" / "parcelles.geojson")
    with rasterio.open(_tif) as _src:
        for _, _r in _p.to_crs(_src.crs).iterrows():
            _a, _ = mask(_src, [_r.geometry], crop=True, nodata=np.nan)
            _v = _a[0][np.isfinite(_a[0])]
            NDVI_DRONE[_r.parcelle_id] = round(float(np.median(_v)), 3)

CELLULES = []


def md(texte):
    CELLULES.append(nbf.v4.new_markdown_cell(texte.strip()))


def code(texte):
    CELLULES.append(nbf.v4.new_code_cell(texte.strip()))


md("""
# Stress hydrique par parcelle — Sentinel-2

**Pourquoi** : le bananier manque d'eau très vite (feuilles immenses, racines superficielles). Un secteur
d'irrigation en panne ou un sol plus filtrant fait perdre du rendement en quelques semaines, sans signe visible
du bord de la parcelle au début. Le chef de culture veut savoir **quelle parcelle est plus sèche que les autres**,
pour aller vérifier les goutteurs **cette semaine**.

**Pourquoi le satellite** : l'eau des feuilles se mesure dans l'infrarouge moyen (SWIR, 1,6 µm). La caméra du
drone ne l'a pas ; Sentinel-2 oui, gratuitement, tous les 5 jours. On calcule le **NDMI** = (B8A − B11) / (B8A + B11)
sur la même plantation de Guangxi, de 2021 à 2023 (le relevé terrain date de décembre 2022).

**Ce que le notebook vérifie, faute de vérité terrain sur l'eau** (personne n'a mesuré l'eau des feuilles) :
1. **cohérence physique** : le NDMI de la plantation baisse-t-il quand le bilan pluie − évaporation des 30 jours
   précédents est négatif ? (météo ERA5 via Open-Meteo) ;
2. **saisons** : la saison sèche est-elle nettement plus sèche que la saison des pluies ?
3. **piège de la canopée** : le NDMI baisse aussi quand il y a moins de feuilles. On retire la part expliquée par le
   NDVI et on vérifie que le signal « eau » reste ;
4. **date du vol drone** (non fournie) : à quelle image Sentinel-2 le NDVI du drone, parcelle par parcelle,
   ressemble-t-il le plus ? Le bulletin combine comptage drone et stress : il faut le même moment ;
5. **stabilité** : une parcelle plus sèche à une date l'est-elle encore 5 à 15 jours plus tard ? Si non,
   l'alerte serait du bruit ;
6. **recoupement avec le drone** (exploratoire, 21 parcelles) : les parcelles sèches sont-elles celles où la
   vigueur est faible ou où la fusariose a été relevée ? (la fusariose bouche les vaisseaux qui montent l'eau).

**Avant de lancer** : rien à déposer. Pas besoin de GPU ni de compte (archive publique Sentinel-2 d'Element 84
sur AWS). *Exécution → Tout exécuter*. Durée : ~10 à 20 min, surtout le téléchargement des petites fenêtres.
Les scènes déjà lues sont gardées sur le Drive : si Colab coupe, relance et ça reprend.
""")

md("## 0. Réglages")
code("""
DEBUT, FIN = "2021-01-01", "2023-12-31"
DATE_REFERENCE = "auto"           # "auto" : date du vol drone estimée par Sentinel-2 ; sinon une date "AAAA-MM-JJ"
DATE_REPLI = "2022-12-15"         # relevé terrain, utilisée si la datation du vol n'est pas nette
RHO_DATATION_MIN = 0.75           # datation retenue au-delà de 0,75 : sur des données sans lien, le max sur ~100 dates atteint déjà ~0,65
MAX_NUAGES_SCENE = 80             # % de nuages sur la tuile entière (filtre grossier ; le vrai filtre est par pixel)
PART_PLANTATION_CLAIRE_MIN = 0.5  # scène gardée si la moitié de la plantation au moins est sans nuage
N_PARCELLES_MIN = 10              # une date ne sert que si 10 parcelles sur 21 y sont mesurées
ECART_MAX_JOURS = 20              # stabilité : deux dates claires à moins de 20 jours d'écart
FENETRE_DRONE_JOURS = 45          # recoupement drone : dates Sentinel-2 à ± 45 jours de la date de référence

STAC = "https://earth-search.aws.element84.com/v1"
COLLECTIONS = ["sentinel-2-c1-l2a", "sentinel-2-l2a"]   # par date : Collection 1 (retraitée) si elle existe,
                                                         # sinon l'ancienne (la Collection 1 a des trous, ex. 2022)
N_DATES_CONTROLE = 8              # dates présentes dans les deux archives, lues deux fois pour vérifier l'accord
BANDES = {"rouge": "red", "pir": "nir08", "swir": "swir16", "scl": "scl"}   # B04, B8A, B11, SCL
N_THREADS = 8
GRAINE = 42
MODE_TEST = False                 # fausses scènes et fausse météo pour vérifier que le notebook tourne (ne pas toucher)
""")

md("## 1. Installation et Google Drive")
code("""
import os, sys, json, time, random, subprocess, types
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

if not MODE_TEST:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pystac-client", "rasterio", "geopandas",
                    "matplotlib", "requests", "scipy"], check=True)

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from scipy.stats import spearmanr, mannwhitneyu
import requests
import matplotlib
import matplotlib.pyplot as plt

random.seed(GRAINE); np.random.seed(GRAINE)
os.environ.update({
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "5", "GDAL_HTTP_RETRY_DELAY": "2",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
})

try:                                   # sur Colab : résultats dans le Drive
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = Path("/content/drive/MyDrive/piscine-bananeraie")
except Exception:                      # sur ton PC (script lancé depuis le dossier du repo) : directement dans docs/
    BASE = Path("docs")
RES = BASE / ("resultats_stress_TEST" if MODE_TEST else "resultats_stress")
RES.mkdir(parents=True, exist_ok=True)
print("Résultats dans :", RES)
""")

md("""
## 2. Fonctions de calcul (copie de `bananes/stress.py`)
Ce sont les mêmes fonctions que `run.py` : ne pas modifier ici, modifier le repo puis régénérer le notebook.
""")
code(SOURCE_STRESS)

md("""
## 3. Parcelles, vigueur drone et points terrain (intégrés au notebook)
21 parcelles d'≈ 1 ha (une parcelle de 1 ha = ~25 pixels Sentinel-2 de 20 m ; les plus petites, ~0,3 ha, en ont ~6).
""")
code(f"""
PARCELLES_GEOJSON = r'''{PARCELLES}'''
VIGUEUR_DRONE = {json.dumps(VIGUEUR_DRONE)}     # part de la parcelle en vigueur faible (drone, run.py)
NDVI_DRONE = {json.dumps(NDVI_DRONE)}           # NDVI médian du drone par parcelle (datation du vol)
FUSARIOSE = {json.dumps(FUSARIOSE)}             # points de plants atteints relevés au sol (déc. 2022)
SAINS = {json.dumps(SAINS)}                     # points de plants sains relevés au sol

parcelles = gpd.read_file(PARCELLES_GEOJSON).to_crs(32648)    # UTM 48N, comme l'image drone
IDS = list(parcelles.parcelle_id)
plantation = parcelles.union_all() if hasattr(parcelles, "union_all") else parcelles.unary_union
CENTRE = gpd.GeoSeries([plantation.centroid], crs=32648).to_crs(4326).iloc[0]
BBOX_4326 = list(gpd.GeoSeries([plantation.buffer(100)], crs=32648).to_crs(4326).total_bounds)
print(len(parcelles), "parcelles,", round(plantation.area / 1e4, 1), "ha ; centre", round(CENTRE.y, 4), round(CENTRE.x, 4))
""")

md("""
## 4. Recherche des images Sentinel-2
Une seule tuile (la plus fréquente) pour que tous les pixels tombent sur la même grille ; une image par jour.
""")
code("""
def https(href):
    # s3://bucket/cle -> https://bucket.s3.us-west-2.amazonaws.com/cle (lecture sans compte)
    if href.startswith("s3://"):
        bucket, cle = href[5:].split("/", 1)
        return f"https://{bucket}.s3.us-west-2.amazonaws.com/{cle}"
    return href


def chercher():
    from pystac_client import Client
    catalogue = Client.open(STAC)
    tous = []
    for col in COLLECTIONS:
        try:
            items = list(catalogue.search(collections=[col], bbox=BBOX_4326, datetime=f"{DEBUT}/{FIN}",
                                          query={"eo:cloud_cover": {"lt": MAX_NUAGES_SCENE}}).items())
        except Exception as e:
            print(f"  {col} : erreur {e}")
            items = []
        print(f"  {col} : {len(items)} scènes")
        tous += items
    if not tous:
        raise SystemExit("Aucune scène trouvée : vérifie la connexion ou les dates.")
    return tous


if MODE_TEST:
    # Fausses scènes : petits GeoTIFF écrits sur le disque, même structure que les vrais objets STAC.
    from rasterio.transform import from_origin
    xmin, ymin, xmax, ymax = plantation.buffer(200).bounds
    TRANSFORM_TEST = from_origin(np.floor(xmin / 20) * 20, np.ceil(ymax / 20) * 20, 20, 20)
    H = int(np.ceil((ymax - ymin) / 20)) + 2; W = int(np.ceil((xmax - xmin) / 20)) + 2
    DOSSIER_TEST = RES / "fausses_scenes"; DOSSIER_TEST.mkdir(exist_ok=True)
    dates_test = pd.date_range(DEBUT, FIN, freq="10D")
    jour = np.array([d.dayofyear for d in dates_test])
    humidite = np.cos(2 * np.pi * (jour - 200) / 365)          # humide en juillet, sec en janvier
    fractions_test = fractions_parcelles(parcelles, TRANSFORM_TEST, (H, W))
    SECHE = "P07"                                                # parcelle volontairement plus sèche
    items = []
    rng = np.random.default_rng(GRAINE)
    for k, d in enumerate(dates_test):
        lai = 0.75 + 0.05 * humidite[k] + rng.normal(0, 0.01, (H, W))
        if d < pd.Timestamp("2022-01-01"):                           # P01 replantée : sol nu la 1re année
            lai = np.where(fractions_test[IDS.index("P01")] > 0, 0.15, lai)
        eau = 0.25 + 0.08 * humidite[k] + rng.normal(0, 0.01, (H, W))
        eau -= 0.06 * fractions_test[IDS.index(SECHE)]
        pir = 0.30 + 0.05 * lai
        rouge = pir * (1 - lai) / (1 + lai)
        swir = pir * (1 - eau) / (1 + eau)
        scl = np.full((H, W), 4, "uint8")
        if rng.random() < 0.3:
            scl[: H // 2] = 9                                         # moitié nord nuageuse
        assets = {}
        for nom, tab in {"red": rouge, "nir08": pir, "swir16": swir}.items():
            chemin = DOSSIER_TEST / f"{k}_{nom}.tif"
            with rasterio.open(chemin, "w", driver="GTiff", height=H, width=W, count=1, dtype="uint16",
                               crs="EPSG:32648", transform=TRANSFORM_TEST) as dst:
                dst.write(((tab + 0.1) * 10000).astype("uint16"), 1)
            assets[nom] = types.SimpleNamespace(href=str(chemin), extra_fields={"raster:bands": [{"scale": 1e-4, "offset": -0.1}]})
        chemin = DOSSIER_TEST / f"{k}_scl.tif"
        with rasterio.open(chemin, "w", driver="GTiff", height=H, width=W, count=1, dtype="uint8",
                           crs="EPSG:32648", transform=TRANSFORM_TEST) as dst:
            dst.write(scl, 1)
        assets["scl"] = types.SimpleNamespace(href=str(chemin), extra_fields={})
        items.append(types.SimpleNamespace(id=f"TEST_{k}", datetime=d.tz_localize("UTC"), assets=assets,
                                           collection_id=COLLECTIONS[k % 7 == 0],
                                           properties={"eo:cloud_cover": 10, "grid:code": "MGRS-48QYM",
                                                       "s2:processing_baseline": "05.09"}))
        if k % 5 == 0:   # même date dans l'autre archive : sert au contrôle d'accord
            items.append(types.SimpleNamespace(id=f"TEST_{k}_bis", datetime=d.tz_localize("UTC"), assets=assets,
                                               collection_id=COLLECTIONS[k % 7 != 0],
                                               properties={"eo:cloud_cover": 20, "grid:code": "MGRS-48QYM",
                                                           "s2:processing_baseline": "05.09"}))
else:
    items = chercher()

lignes = []
for it in items:
    p = it.properties
    lignes.append({"id": it.id, "collection": it.collection_id, "date": it.datetime.tz_convert("Asia/Shanghai").date() if hasattr(it.datetime, "tz_convert")
                   else it.datetime.date(), "tuile": p.get("grid:code") or p.get("s2:mgrs_tile") or p.get("mgrs:utm_zone"),
                   "nuages_tuile": p.get("eo:cloud_cover"), "baseline": p.get("s2:processing_baseline"),
                   "decalage_applique": p.get("earthsearch:boa_offset_applied")})
scenes = pd.DataFrame(lignes)
TUILE = scenes.tuile.mode().iloc[0]
scenes = scenes[scenes.tuile == TUILE].copy()
scenes["priorite"] = scenes.collection.map({c: i for i, c in enumerate(COLLECTIONS)})
toutes = scenes
scenes = toutes.sort_values(["priorite", "nuages_tuile"]).drop_duplicates("date").sort_values("date")
# Contrôle : quelques dates présentes dans les deux archives, lues aussi dans l'archive non retenue
doublons = toutes[~toutes.id.isin(scenes.id) & toutes.date.isin(scenes.date)].drop_duplicates("date")
CONTROLE = doublons.sample(min(N_DATES_CONTROLE, len(doublons)), random_state=GRAINE)
ITEMS = {it.id: it for it in items}
print(f"tuile {TUILE} | {len(scenes)} dates de {scenes.date.min()} à {scenes.date.max()} ; par archive :",
      scenes.collection.value_counts().to_dict())
print("Dates par année :", pd.to_datetime(scenes.date).dt.year.value_counts().sort_index().to_dict())
""")

md("""
## 5. Lecture des petites fenêtres autour de la plantation
On ne télécharge pas les tuiles entières (110 × 110 km) : seulement ~1 km² autour de la plantation, 4 bandes.
**Réflectance** : échelle et décalage lus dans les métadonnées de chaque image. Depuis janvier 2022, l'ESA ajoute
un décalage de 1 000 aux valeurs ; l'oublier fausserait le NDMI et créerait un « saut » artificiel en 2022.
""")
code("""
def echelle_decalage(item, cle):
    bandes = item.assets[cle].extra_fields.get("raster:bands") or [{}]
    echelle = bandes[0].get("scale", 1e-4)
    decalage = bandes[0].get("offset", 0.0)
    if item.collection_id == "sentinel-2-l2a":
        # ancienne collection : le décalage n'est réel que si la baseline >= 4.00 et qu'Element 84 ne l'a pas retiré
        base = float(item.properties.get("s2:processing_baseline") or 0)
        deja = item.properties.get("earthsearch:boa_offset_applied")
        decalage = -0.1 if (base >= 4.0 and not deja) else 0.0
    return echelle, decalage


GRILLE = {}

def lire_scene(item_id):
    item = ITEMS[item_id]
    with rasterio.open(https(item.assets[BANDES["swir"]].href)) as src:
        geom = gpd.GeoSeries([plantation.buffer(60)], crs=32648).to_crs(src.crs).iloc[0]
        fen = from_bounds(*geom.bounds, src.transform).round_offsets().round_lengths()
        transform = src.window_transform(fen)
        forme = (int(fen.height), int(fen.width))
        bornes = rasterio.windows.bounds(fen, src.transform)
        crs = src.crs
        swir_dn = src.read(1, window=fen)
    lu = {"swir": swir_dn}
    for nom in ("rouge", "pir", "scl"):
        with rasterio.open(https(item.assets[BANDES[nom]].href)) as src:
            r = Resampling.nearest if nom == "scl" else Resampling.average   # rouge : 10 m -> 20 m par moyenne
            lu[nom] = src.read(1, window=from_bounds(*bornes, src.transform), out_shape=forme, resampling=r)
    refl = {nom: reflectance(lu[nom], *echelle_decalage(item, BANDES[nom])) for nom in ("rouge", "pir", "swir")}
    cle = (str(crs), transform, forme)
    if cle not in GRILLE:
        GRILLE[cle] = fractions_parcelles(parcelles.to_crs(crs), transform, forme)
    fractions = GRILLE[cle]
    clair = np.isin(lu["scl"], SCL_CLAIR)
    carte_ndmi = ndmi(refl["pir"], refl["swir"])
    carte_ndvi = ndvi(refl["rouge"], refl["pir"])
    stats = stats_scene(carte_ndmi, carte_ndvi, clair, fractions, IDS)
    dans_plantation = fractions.max(axis=0) >= FRACTION_PIXEL_MIN
    stats["part_plantation_claire"] = round(float(clair[dans_plantation].mean()), 3)
    stats["rouge_median"] = float(np.nanmedian(refl["rouge"][dans_plantation & clair])) if (dans_plantation & clair).any() else np.nan
    stats["id"] = item_id
    return stats, (carte_ndmi, clair, transform, crs)


CACHE = RES / "lectures_scenes.csv"
deja = pd.read_csv(CACHE) if CACHE.exists() else pd.DataFrame(columns=["id"])
a_lire = [i for i in list(scenes.id) + list(CONTROLE.id) if i not in set(deja.id)]
print(f"{len(deja.id.unique()) if len(deja) else 0} scènes déjà lues, {len(a_lire)} à lire")
debut = time.time(); nouvelles = []; erreurs = []
with ThreadPoolExecutor(N_THREADS) as pool:
    futurs = {pool.submit(lire_scene, i): i for i in a_lire}
    for n, f in enumerate(as_completed(futurs), 1):
        try:
            nouvelles.append(f.result()[0])
        except Exception as e:
            erreurs.append((futurs[f], str(e)[:120]))
        if n % 20 == 0 or n == len(futurs):
            print(f"  {n}/{len(futurs)} scènes ({time.time() - debut:.0f} s)")
            if nouvelles:
                deja = pd.concat([deja, *nouvelles], ignore_index=True); nouvelles = []
                deja.to_csv(CACHE, index=False)
if erreurs:
    print(f"{len(erreurs)} scènes illisibles (relance la cellule pour réessayer) :", erreurs[:3])
DUREE_LECTURE = time.time() - debut

# Accord entre les deux archives sur les dates de contrôle (même date, même parcelle)
paires = CONTROLE[["id", "date"]].merge(scenes[["id", "date"]], on="date", suffixes=("_autre", "_retenue"))
acc = []
for _, r in paires.iterrows():
    a = deja[deja.id == r.id_autre].set_index("parcelle_id"); b = deja[deja.id == r.id_retenue].set_index("parcelle_id")
    acc.append((a.ndmi - b.ndmi).abs().dropna())
ECART_ARCHIVES = float(pd.concat(acc).median()) if acc and len(pd.concat(acc)) else float("nan")
print(f"Accord entre archives ({len(paires)} dates communes) : écart médian de NDMI par parcelle = {ECART_ARCHIVES:.3f}")

brut = deja.merge(scenes[["id", "date"]], on="id")
par_scene = brut.groupby(["id", "date"]).agg(part_plantation_claire=("part_plantation_claire", "first"),
                                             rouge_median=("rouge_median", "first")).reset_index()
gardees = par_scene[par_scene.part_plantation_claire >= PART_PLANTATION_CLAIRE_MIN]
brut = brut[brut.id.isin(gardees.id)]
scenes = scenes.merge(par_scene, on=["id", "date"], how="left")
scenes["gardee"] = scenes.id.isin(gardees.id)
scenes.to_csv(RES / "scenes.csv", index=False)
print(f"{len(gardees)} dates gardées sur {len(par_scene)} lues (plantation dégagée à ≥ {PART_PLANTATION_CLAIRE_MIN:.0%})")
""")

md("""
**Contrôle du décalage de réflectance** : la réflectance rouge médiane de la plantation ne doit pas faire de saut
au 25 janvier 2022 (changement de traitement ESA). Un saut d'environ 0,1 signalerait un décalage mal géré.
""")
code("""
g = gardees.copy(); g["date"] = pd.to_datetime(g["date"])
avant = g[(g.date < "2022-01-25") & (g.date >= "2021-10-01")].rouge_median.median()
apres = g[(g.date >= "2022-01-25") & (g.date < "2022-05-01")].rouge_median.median()
SAUT_ROUGE = float(apres - avant) if np.isfinite(avant) and np.isfinite(apres) else float("nan")
print(f"Rouge médian : {avant:.3f} avant / {apres:.3f} après le 25/01/2022 (écart {SAUT_ROUGE:+.3f})")
if abs(SAUT_ROUGE) > 0.05:
    print("ATTENTION : saut suspect, le décalage de réflectance est peut-être mal appliqué.")
""")

md("""
## 6. Météo : pluie et évaporation (ERA5, via Open-Meteo)
**Bilan hydrique sur 30 jours** = pluie − évapotranspiration de référence (FAO-56) des 30 jours précédant l'image.
Négatif = le sol perd plus d'eau qu'il n'en reçoit. Les **mois secs** sont définis par les données : pluie < ET0.
""")
code("""
def meteo_open_meteo():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {"latitude": round(CENTRE.y, 4), "longitude": round(CENTRE.x, 4),
              "start_date": (pd.Timestamp(DEBUT) - pd.Timedelta(days=40)).date().isoformat(), "end_date": FIN,
              "daily": "precipitation_sum,et0_fao_evapotranspiration", "timezone": "Asia/Shanghai"}
    r = requests.get(url, params=params, timeout=60); r.raise_for_status()
    d = r.json()["daily"]
    return pd.DataFrame({"date": pd.to_datetime(d["time"]), "pluie": d["precipitation_sum"],
                         "et0": d["et0_fao_evapotranspiration"]})

if MODE_TEST:
    jours = pd.date_range(pd.Timestamp(DEBUT) - pd.Timedelta(days=40), FIN)
    h = np.cos(2 * np.pi * (jours.dayofyear - 200) / 365)
    meteo = pd.DataFrame({"date": jours, "pluie": np.clip(4 + 6 * h + np.random.normal(0, 3, len(jours)), 0, None),
                          "et0": 3.5 + 1.0 * h})
else:
    try:
        meteo = meteo_open_meteo()
    except Exception as e:
        print("Météo indisponible :", e); meteo = None

if meteo is not None:
    meteo = meteo.fillna(0).set_index("date")
    meteo["bilan_30j"] = (meteo.pluie - meteo.et0).rolling(30).sum()
    clim = meteo.groupby(meteo.index.month)[["pluie", "et0"]].sum() / meteo.index.year.nunique()
    clim["sec"] = clim.pluie < clim.et0
    MOIS_SECS = [int(m) for m in clim.index[clim.sec]]
    print("Pluie et ET0 moyennes par mois (mm) :"); print(clim.round(0).astype({"pluie": int, "et0": int}).T)
    print("Mois secs (pluie < ET0) :", MOIS_SECS)
    meteo.reset_index().to_csv(RES / "meteo.csv", index=False)
else:
    MOIS_SECS = [11, 12, 1, 2, 3]   # repli : saison sèche habituelle du Guangxi (novembre à mars)
    print("Repli sur les mois secs habituels :", MOIS_SECS)
""")

md("""
## 7. Série par parcelle : NDMI, contrôle de la canopée, score relatif
`ndmi_residu` = NDMI − (a + b × NDVI) : l'eau en plus ou en moins **à quantité de feuilles égale**.
`z_stress` = écart du résidu à la médiane des parcelles **à la même date**, en écarts-types robustes.
""")
code("""
serie, (A, B) = calculer(brut[["parcelle_id", "date", "ndmi", "ndvi", "part_claire"]], N_PARCELLES_MIN)
cmp_ = serie[serie.comparable]
R2_CANOPEE = float(np.corrcoef(cmp_.ndvi, cmp_.ndmi)[0, 1] ** 2)
serie["saison"] = np.where(serie.date.dt.month.isin(MOIS_SECS), "sèche", "pluies")
print(f"{serie.date.nunique()} dates, {len(serie)} mesures parcelle × date ; canopée fermée (NDVI ≥ {COUVERT_MIN}) : "
      f"{serie.couvert.mean():.0%} des mesures ; {cmp_.date.nunique()} dates comparables")
print(f"NDMI ≈ {A:.3f} + {B:.3f} × NDVI : le NDVI explique {R2_CANOPEE:.0%} des variations du NDMI")
serie.to_csv(RES / "serie_stress_parcelles.csv", index=False)

couvert_plant = serie.groupby("date").agg(ndvi_tout=("ndvi", "median"), part_couvert=("couvert", "mean")).reset_index()
plant = serie[serie.comparable].groupby("date").agg(ndmi=("ndmi", "median"), ndvi=("ndvi", "median"),
                                  residu=("ndmi_residu", "median"), n=("parcelle_id", "count")).reset_index()
plant["saison"] = np.where(plant.date.dt.month.isin(MOIS_SECS), "sèche", "pluies")
if meteo is not None:
    plant["bilan_30j"] = meteo.bilan_30j.reindex(plant.date).to_numpy()
plant.to_csv(RES / "serie_plantation.csv", index=False)
""")

md("""
## 7 bis. Datation du vol drone
Pour chaque image, ressemblance (Spearman) entre le NDVI Sentinel-2 des 21 parcelles et le NDVI du drone.
Les parcelles replantées à des moments différents donnent à chaque date un « motif » propre : la date la plus
ressemblante est la plus probable pour le vol. Le niveau absolu ne se compare pas (8 cm contre 20 m, sol entre les plants).
""")
code("""
dat = []
if NDVI_DRONE:
    nd = pd.Series(NDVI_DRONE)
    for d, g in serie.groupby("date"):
        g = g.set_index("parcelle_id"); c = g.index.intersection(nd.index)
        if len(c) >= 15:
            dat.append({"date": d, "rho": spearmanr(g.loc[c, "ndvi"], nd[c]).statistic, "n": len(c)})
dat = pd.DataFrame(dat)
if len(dat):
    dat.to_csv(RES / "datation_drone.csv", index=False)
    meilleure = dat.loc[dat.rho.idxmax()]
    print("Dates les plus ressemblantes au drone :"); print(dat.nlargest(5, "rho").round(2).to_string(index=False))
    DATE_DRONE_ESTIMEE = str(meilleure.date.date()); RHO_DATATION = float(meilleure.rho)
else:
    DATE_DRONE_ESTIMEE, RHO_DATATION = None, float("nan")
if DATE_REFERENCE == "auto":
    DATE_REF = DATE_DRONE_ESTIMEE if RHO_DATATION >= RHO_DATATION_MIN else DATE_REPLI
else:
    DATE_REF = DATE_REFERENCE
print(f"Date retenue pour le bulletin et le recoupement : {DATE_REF}"
      + ("" if DATE_REF == DATE_DRONE_ESTIMEE else f" (datation pas assez nette : rho max {RHO_DATATION:.2f})"))
""")

md("""
## 8. Vérifications
Les intervalles sont des **bootstraps sur les dates** (1 000 tirages). Les dates proches se ressemblent, donc ces
intervalles sont un peu trop étroits : on les lit comme un ordre de grandeur, pas comme une preuve.
""")
code("""
def bootstrap(f, *colonnes, n=1000):
    rng = np.random.default_rng(GRAINE); m = len(colonnes[0]); valeurs = []
    for _ in range(n):
        i = rng.integers(0, m, m)
        valeurs.append(f(*[c[i] for c in colonnes]))
    return np.nanpercentile(valeurs, [2.5, 97.5])

VERIF = {"archives": scenes.collection.value_counts().to_dict(), "ecart_ndmi_entre_archives": round(ECART_ARCHIVES, 3),
         "dates_par_annee": {str(k): int(v) for k, v in pd.to_datetime(scenes.date).dt.year.value_counts().sort_index().items()},
         "part_mesures_canopee_fermee": round(float(serie.couvert.mean()), 2),
         "n_dates_comparables": int(cmp_.date.nunique()),
         "date_drone_estimee": DATE_DRONE_ESTIMEE, "rho_datation": round(RHO_DATATION, 2), "date_reference": DATE_REF,
         "tuile": TUILE, "n_dates": int(serie.date.nunique()),
         "n_scenes_lues": int(len(par_scene)), "saut_rouge_2022": round(SAUT_ROUGE, 3),
         "ndmi_vs_ndvi_r2": round(R2_CANOPEE, 3), "canopee_a": round(A, 4), "canopee_b": round(B, 4),
         "mois_secs": MOIS_SECS}

# 1. Cohérence physique : NDMI vs bilan hydrique des 30 jours
if meteo is not None:
    ok = plant.dropna(subset=["bilan_30j"])
    print("1. Lien avec le bilan pluie − ET0 sur 30 jours (Spearman, IC 95 %) :")
    for col in ("ndmi", "residu", "ndvi"):
        rho = spearmanr(ok[col], ok.bilan_30j).statistic
        ic = bootstrap(lambda x, y: spearmanr(x, y).statistic, ok[col].to_numpy(), ok.bilan_30j.to_numpy())
        VERIF[f"rho_bilan_{col}"] = [round(rho, 2), round(ic[0], 2), round(ic[1], 2)]
        print(f"   {col:7s} rho = {rho:+.2f}  [{ic[0]:+.2f} ; {ic[1]:+.2f}]")

# 2. Saison sèche contre saison des pluies (niveau plantation, une valeur par date)
s, p = plant[plant.saison == "sèche"].ndmi.to_numpy(), plant[plant.saison == "pluies"].ndmi.to_numpy()
diff = np.median(s) - np.median(p)
rng = np.random.default_rng(GRAINE)
ic = np.percentile([np.median(rng.choice(s, len(s))) - np.median(rng.choice(p, len(p))) for _ in range(1000)], [2.5, 97.5])
VERIF["ndmi_sec_moins_pluies"] = [round(diff, 3), round(ic[0], 3), round(ic[1], 3)]
VERIF["mannwhitney_p"] = float(mannwhitneyu(s, p).pvalue)
par_p = serie[serie.comparable].groupby(["parcelle_id", "saison"]).ndmi.median().unstack().dropna()
VERIF["part_parcelles_plus_seches_en_saison_seche"] = round(float((par_p["sèche"] < par_p["pluies"]).mean()), 2)
print(f"2. NDMI saison sèche − saison des pluies : {diff:+.3f} [{ic[0]:+.3f} ; {ic[1]:+.3f}] "
      f"({len(s)} et {len(p)} dates) ; {VERIF['part_parcelles_plus_seches_en_saison_seche']:.0%} des parcelles plus sèches en saison sèche")

# 3. Stabilité des écarts entre parcelles d'une date claire à la suivante
dates = sorted(cmp_.date.unique()); rhos = {"z_stress": [], "z_ndmi_brut": [], "hasard": []}
for d1, d2 in zip(dates[:-1], dates[1:]):
    if (pd.Timestamp(d2) - pd.Timestamp(d1)).days > ECART_MAX_JOURS:
        continue
    a = cmp_[cmp_.date == d1].set_index("parcelle_id"); b = cmp_[cmp_.date == d2].set_index("parcelle_id")
    commun = a.index.intersection(b.index)
    if len(commun) < N_PARCELLES_MIN:
        continue
    for col in ("z_stress", "z_ndmi_brut"):
        rhos[col].append(spearmanr(a.loc[commun, col], b.loc[commun, col]).statistic)
    rhos["hasard"].append(spearmanr(a.loc[commun, "z_stress"], np.random.permutation(b.loc[commun, "z_stress"].to_numpy())).statistic)
for col, v in rhos.items():
    VERIF[f"stabilite_{col}"] = round(float(np.median(v)), 2) if v else None
print(f"3. Stabilité (Spearman médian entre dates à ≤ {ECART_MAX_JOURS} j, {len(rhos['hasard'])} paires) : "
      f"z_stress {VERIF['stabilite_z_stress']}, NDMI brut {VERIF['stabilite_z_ndmi_brut']}, hasard {VERIF['stabilite_hasard']}")

# 4. Recoupement avec le drone et le terrain (exploratoire, 21 parcelles)
ref = pd.Timestamp(DATE_REF)
proche = serie[(serie.date - ref).abs() <= pd.Timedelta(days=FENETRE_DRONE_JOURS)]
z_ref = proche.groupby("parcelle_id").z_stress.mean()
comp = pd.DataFrame({"z_stress": z_ref, "vigueur_faible_drone": pd.Series(VIGUEUR_DRONE),
                     "fusariose": pd.Series(FUSARIOSE), "sains": pd.Series(SAINS)}).fillna({"fusariose": 0, "sains": 0})
comp["part_fusariose"] = comp.fusariose / (comp.fusariose + comp.sains).replace(0, np.nan)
comp = comp.dropna(subset=["z_stress"])
for col in ("vigueur_faible_drone", "part_fusariose"):
    c = comp.dropna(subset=[col])
    r = spearmanr(c.z_stress, c[col])
    VERIF[f"rho_z_{col}"] = [round(r.statistic, 2), round(r.pvalue, 3), int(len(c))]
    print(f"4. z_stress (± {FENETRE_DRONE_JOURS} j) vs {col} : rho = {r.statistic:+.2f} (p = {r.pvalue:.2f}, {len(c)} parcelles)")
VERIF["n_dates_fenetre_drone"] = int(proche.date.nunique())
comp.round(3).to_csv(RES / "recoupement_drone.csv")
""")

md("## 9. État par parcelle pour le bulletin, historique des alertes")
code("""
etat = etat_a_date(serie, DATE_REF)
etat.to_csv(RES / "stress_par_parcelle.csv", index=False)

serie = serie.sort_values(["parcelle_id", "date"])
serie["sec"] = serie.z_stress < Z_ALERTE
serie["alerte"] = serie.groupby("parcelle_id").sec.transform(lambda s: s & s.shift(1, fill_value=False))
hist = serie.groupby("parcelle_id").agg(n_dates=("comparable", "sum"), n_alertes=("alerte", "sum"),
                                        n_alertes_saison_seche=("alerte", lambda a: int(a[serie.loc[a.index, "saison"] == "sèche"].sum())),
                                        z_median=("z_stress", "median")).reset_index().sort_values("z_median")
hist.to_csv(RES / "historique_alertes.csv", index=False)
VERIF["parcelles_en_alerte_reference"] = etat.loc[etat.alerte_stress, "parcelle_id"].tolist()
VERIF["part_dates_avec_alerte"] = round(float(serie.groupby("date").alerte.any().mean()), 2)
print(f"État au {DATE_REF} (image du {etat.date_stress.max()}) :")
print(etat.sort_values("z_stress").head(6).to_string(index=False))
print(); print("Parcelles les plus souvent sèches sur 3 ans :"); print(hist.head(6).to_string(index=False))
""")

md("## 10. Figures")
code("""
matplotlib.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(plant.date, plant.ndmi, "o-", ms=3, color="#1f6fb4", label="NDMI médian (eau des feuilles)")
ax.plot(couvert_plant.date, couvert_plant.ndvi_tout, "-", lw=1, color="#3a9d4b", alpha=0.8,
        label="NDVI médian, toutes parcelles (quantité de feuilles)")
ax.axhline(COUVERT_MIN, color="#3a9d4b", ls=":", lw=0.8)
for annee in range(int(DEBUT[:4]), int(FIN[:4]) + 1):
    for m in MOIS_SECS:
        d = pd.Timestamp(annee, m, 1)
        ax.axvspan(d, d + pd.offsets.MonthEnd(0), color="#e8c07a", alpha=0.25, lw=0)
ax.set_ylabel("indice"); ax.legend(loc="upper left", fontsize=8)
if meteo is not None:
    ax2 = ax.twinx(); ax2.fill_between(meteo.index, meteo.bilan_30j, 0, color="grey", alpha=0.2)
    ax2.set_ylabel("pluie − ET0 sur 30 j (mm)", color="grey")
ax.set_title("Guangxi : NDMI des parcelles en canopée fermée, NDVI de toutes (beige = mois secs, pointillé = seuil de canopée)")
plt.tight_layout(); plt.savefig(RES / "serie_plantation.png", dpi=120); plt.show()

piv = serie.pivot_table(index="parcelle_id", columns="date", values="z_stress")
piv = piv.loc[hist.parcelle_id]
fig, ax = plt.subplots(figsize=(10, 4.5))
im = ax.imshow(piv.clip(-4, 4), aspect="auto", cmap="RdBu", vmin=-4, vmax=4, interpolation="nearest")
ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=7)
pos = np.linspace(0, piv.shape[1] - 1, 8).astype(int)
ax.set_xticks(pos); ax.set_xticklabels([pd.Timestamp(piv.columns[i]).strftime("%m/%Y") for i in pos])
plt.colorbar(im, label="z_stress (rouge = plus sec que les autres)")
ax.set_title("Écart d'eau de chaque parcelle aux autres, date par date (parcelles triées : les plus sèches en haut)")
plt.tight_layout(); plt.savefig(RES / "parcelles_dates.png", dpi=120); plt.show()

carte = parcelles.merge(etat, on="parcelle_id")
fig, ax = plt.subplots(figsize=(8, 4))
carte.plot(column="z_stress", cmap="RdBu", vmin=-3, vmax=3, legend=True, ax=ax, edgecolor="k",
           legend_kwds={"label": "z_stress"}, missing_kwds={"color": "lightgrey", "hatch": "//", "label": "sol nu / jeunes plants"})
carte[carte.alerte_stress].boundary.plot(ax=ax, color="red", lw=3)
for _, r in carte.iterrows():
    ax.annotate(r.parcelle_id, r.geometry.centroid.coords[0], ha="center", fontsize=7)
ax.set_title(f"Stress hydrique relatif, image du {etat.date_stress.max()} (rouge = alerte, hachuré = pas de canopée)"); ax.axis("off")
plt.tight_layout(); plt.savefig(RES / "carte_stress.png", dpi=120); plt.show()

VERIF["duree_lecture_s"] = round(DUREE_LECTURE, 1)
(RES / "verifications.json").write_text(json.dumps(VERIF, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps(VERIF, indent=2, ensure_ascii=False, default=float))
""")

md("""
## Ce qu'il faut récupérer

Tout `MyDrive/piscine-bananeraie/resultats_stress/` → `docs/resultats_stress/` du repo, **sauf**
`lectures_scenes.csv` (cache). Le fichier pour le bulletin est `stress_par_parcelle.csv` :

```bash
python run.py --images data/guangxi_pret/images --parcelles data/guangxi_pret/parcelles.geojson \\
    --out resultats/ --stress docs/resultats_stress/stress_par_parcelle.csv
```

Et envoie-moi le bloc JSON affiché à la fin : c'est ce qui va dans le rapport.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = CELLULES
nb["metadata"] = {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}
cible = Path(__file__).with_name("stress_colab.ipynb")
nbf.write(nb, cible)
print("Notebook écrit :", cible)

# Même code en script, pour le lancer sur un PC sans Colab : python scripts/stress_sentinel2.py
morceaux = ['"""Stress hydrique Sentinel-2 — version script de notebooks/stress_colab.ipynb (généré, ne pas modifier).\n\n'
            'Lancer depuis le dossier du repo : python scripts/stress_sentinel2.py\n'
            'Résultats dans docs/resultats_stress/. Durée : ~10-20 min (téléchargement), pas de GPU."""',
            "import sys\nsys.stdout.reconfigure(encoding='utf-8', errors='replace')   # console Windows",
            "import matplotlib\nmatplotlib.use('Agg')   # figures enregistrées, pas affichées"]
for c in CELLULES:
    if c.cell_type == "code":
        morceaux.append(c.source.replace("plt.show()", "plt.close()"))
    else:
        morceaux.append("\n".join("# " + l for l in c.source.splitlines()[:3]))
script = RACINE / "scripts" / "stress_sentinel2.py"
script.write_text("\n\n".join(morceaux) + "\n", encoding="utf-8")
print("Script écrit :", script)
