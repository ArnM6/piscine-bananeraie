"""Stress hydrique par parcelle, à partir de Sentinel-2.

Pourquoi le satellite et pas le drone : l'eau des feuilles se voit dans l'infrarouge moyen (SWIR, ~1,6 µm),
que la caméra multispectrale du drone n'a pas (bleu, vert, rouge, red-edge, PIR). Sentinel-2 a cette bande
(B11), gratuitement, tous les 5 jours, depuis 2017 : on peut donc comparer saison sèche et saison des pluies.

Justification physique :
- L'eau liquide des feuilles absorbe fortement le SWIR (B11, 1 610 nm) ; le proche infrarouge (B8A, 865 nm)
  est réfléchi par la structure des feuilles et presque pas absorbé par l'eau.
- NDMI = (B8A - B11) / (B8A + B11) : plus il est haut, plus la canopée contient d'eau.
- Piège : le NDMI baisse aussi quand il y a MOINS DE FEUILLES (jeunes rejets, récolte, effeuillage),
  pas seulement quand les feuilles sont sèches. On retire donc la part du NDMI expliquée par le NDVI
  (quantité de feuilles) : le résidu est l'eau « en plus ou en moins » à quantité de feuilles égale.
- Le bananier est très gourmand en eau (feuilles immenses, racines superficielles) : il ferme ses stomates
  et ses feuilles se replient dès que le sol s'assèche. C'est un signal précoce, avant la perte de rendement.

Ce qu'on signale : une parcelle nettement PLUS SÈCHE QUE LES AUTRES À LA MÊME DATE (même pluie, même
satellite, même atmosphère). C'est ce qui sert au chef de culture : un goutteur bouché, un secteur d'irrigation
en panne, un sol plus filtrant. Une sécheresse qui touche toute la plantation se lit sur la série temporelle,
pas dans l'alerte par parcelle.
"""
import numpy as np
import pandas as pd
from rasterio import features
from rasterio.transform import Affine

EPS = 1e-6
SCL_CLAIR = (4, 5)          # classes Sentinel-2 L2A : 4 végétation, 5 sol nu. Tout le reste (nuage, ombre…) exclu
FRACTION_PIXEL_MIN = 0.75   # un pixel de 20 m compte pour une parcelle s'il est dedans à 75 % au moins
PART_CLAIRE_MIN = 0.8       # une parcelle est mesurée à une date si 80 % de ses pixels sont sans nuage
Z_ALERTE = -2.0             # 2 « écarts-types robustes » plus sec que les autres parcelles
N_DATES_ALERTE = 2          # alerte seulement si c'est le cas sur 2 dates claires consécutives
COUVERT_MIN = 0.6           # NDVI sous lequel la parcelle n'est pas une canopée fermée (sol nu, jeunes plants,
                            # après récolte) : le NDMI y mesure surtout l'humidité du SOL, pas l'eau des feuilles.
                            # Sur Guangxi, une parcelle replantée descend à NDVI ≈ 0,1-0,2 ; une canopée fermée ≈ 0,8.
DISPERSION_MIN = 0.01       # plancher de dispersion (unités NDMI) : ordre de grandeur du bruit d'une médiane
                            # de parcelle Sentinel-2. Sans lui, un jour où toutes les parcelles se ressemblent,
                            # un écart de 0,005 deviendrait « -6 écarts-types ».


def reflectance(dn, echelle=1e-4, decalage=0.0):
    """Valeurs numériques Sentinel-2 -> réflectance 0-1 (échelle et décalage lus dans les métadonnées)."""
    r = dn.astype("float32") * echelle + decalage
    r[dn == 0] = np.nan       # 0 = pas de donnée
    return r


def ndmi(pir, swir):
    return (pir - swir) / (pir + swir + EPS)


def ndvi(rouge, pir):
    return (pir - rouge) / (pir + rouge + EPS)


def fractions_parcelles(parcelles, transform, forme, sur=10):
    """Pour chaque parcelle, la fraction de chaque pixel qui est dans la parcelle (0 à 1).

    On rasterise sur une grille 10 fois plus fine puis on moyenne par blocs : un pixel de 20 m à cheval
    sur une piste et une parcelle n'est pas compté comme « bananier ».
    Renvoie un tableau (n_parcelles, hauteur, largeur).
    """
    h, w = forme
    fine = transform * Affine.scale(1 / sur)
    sortie = np.zeros((len(parcelles), h, w), dtype="float32")
    for i, geom in enumerate(parcelles.geometry):
        m = features.rasterize([(geom, 1)], out_shape=(h * sur, w * sur), transform=fine, fill=0, dtype="uint8")
        sortie[i] = m.reshape(h, sur, w, sur).mean(axis=(1, 3))
    return sortie


def stats_scene(carte_ndmi, carte_ndvi, clair, fractions, ids,
                fraction_min=FRACTION_PIXEL_MIN, part_claire_min=PART_CLAIRE_MIN):
    """Médiane du NDMI et du NDVI de chaque parcelle pour une date, si la parcelle est assez dégagée."""
    lignes = []
    for i, pid in enumerate(ids):
        dedans = fractions[i] >= fraction_min
        n = int(dedans.sum())
        if n == 0:
            continue
        ok = dedans & clair & np.isfinite(carte_ndmi) & np.isfinite(carte_ndvi)
        part = ok.sum() / n
        lignes.append({
            "parcelle_id": pid,
            "n_pixels": n,
            "part_claire": round(float(part), 3),
            "ndmi": float(np.median(carte_ndmi[ok])) if part >= part_claire_min else np.nan,
            "ndvi": float(np.median(carte_ndvi[ok])) if part >= part_claire_min else np.nan,
        })
    return pd.DataFrame(lignes)


def ajuster_canopee(serie):
    """Droite NDMI = a + b·NDVI sur toutes les (parcelle, date) mesurées. Renvoie (a, b)."""
    ok = serie[["ndmi", "ndvi"]].dropna()
    b, a = np.polyfit(ok["ndvi"], ok["ndmi"], 1)
    return float(a), float(b)


def z_relatif(serie, colonne):
    """Par date : écart de chaque parcelle à la médiane des parcelles, divisé par la dispersion (MAD).

    Même logique que la carte de vigueur : on compare à la même date, donc la pluie, la saison et
    l'atmosphère s'annulent. -2 = nettement plus sec que le reste de la plantation.
    """
    def z(v):
        med = v.median()
        mad = (v - med).abs().median() * 1.4826          # 1.4826 : MAD -> écart-type
        return (v - med) / max(mad, DISPERSION_MIN)
    return serie.groupby("date")[colonne].transform(z)


def calculer(serie, n_parcelles_min=10, couvert_min=COUVERT_MIN):
    """Ajoute à la série (parcelle_id, date, ndmi, ndvi) : couvert, résidu « eau » et score relatif.

    Le stress n'est calculé que sur les parcelles en canopée fermée (NDVI >= couvert_min), et seulement aux
    dates où au moins `n_parcelles_min` parcelles le sont (sinon la médiane de comparaison n'a pas de sens).
    Les autres lignes sont gardées (utiles pour suivre les replantations) avec z_stress vide.
    """
    serie = serie.dropna(subset=["ndmi", "ndvi"]).copy()
    serie["date"] = pd.to_datetime(serie["date"])
    serie["couvert"] = serie["ndvi"] >= couvert_min
    n_couverts = serie.groupby("date")["couvert"].transform("sum")
    serie["comparable"] = serie["couvert"] & (n_couverts >= n_parcelles_min)
    ok = serie[serie["comparable"]]
    a, b = ajuster_canopee(ok)
    serie["ndmi_residu"] = np.where(serie["comparable"], serie["ndmi"] - (a + b * serie["ndvi"]), np.nan)
    serie["z_stress"] = np.nan
    serie["z_ndmi_brut"] = np.nan
    serie.loc[serie["comparable"], "z_stress"] = z_relatif(ok.assign(ndmi_residu=ok["ndmi"] - (a + b * ok["ndvi"])), "ndmi_residu")
    serie.loc[serie["comparable"], "z_ndmi_brut"] = z_relatif(ok, "ndmi")
    return serie.sort_values(["date", "parcelle_id"]).reset_index(drop=True), (a, b)


def etat_a_date(serie, date_ref, z_alerte=Z_ALERTE, n_dates=N_DATES_ALERTE):
    """État de chaque parcelle à une date de référence (la dernière image claire avant ou à cette date).

    alerte_stress = z_stress < z_alerte sur les `n_dates` dernières dates claires de la parcelle.
    Une seule date sèche peut venir d'un nuage mal masqué ou d'un arrosage décalé : on exige la répétition.
    Une parcelle sans canopée fermée n'est jamais en alerte : statut « sol nu ou jeunes plants ».
    """
    date_ref = pd.Timestamp(date_ref)
    avant = serie[serie["date"] <= date_ref].sort_values("date")
    lignes = []
    for pid, g in avant.groupby("parcelle_id"):
        dernieres = g.tail(n_dates)
        derniere = g.iloc[-1]
        if not derniere["couvert"]:
            statut = "sol nu ou jeunes plants"
        elif not derniere["comparable"]:
            statut = "trop peu de parcelles comparables"
        else:
            statut = "mesuré"
        alerte = bool(len(dernieres) == n_dates and dernieres["comparable"].all()
                      and (dernieres["z_stress"] < z_alerte).all())
        lignes.append({
            "parcelle_id": pid,
            "date_stress": derniere["date"].date().isoformat(),
            "ndmi": round(float(derniere["ndmi"]), 3),
            "ndvi": round(float(derniere["ndvi"]), 3),
            "z_stress": round(float(derniere["z_stress"]), 2) if pd.notna(derniere["z_stress"]) else np.nan,
            "alerte_stress": alerte,
            "statut_stress": statut,
        })
    return pd.DataFrame(lignes)


def lire_etat(chemin):
    """Lit le fichier produit par le notebook (stress_par_parcelle.csv) pour le bulletin."""
    etat = pd.read_csv(chemin)
    attendues = {"parcelle_id", "date_stress", "z_stress", "alerte_stress"}
    manquantes = attendues - set(etat.columns)
    if manquantes:
        raise ValueError(f"{chemin} : colonnes manquantes {sorted(manquantes)}")
    etat["alerte_stress"] = etat["alerte_stress"].astype(str).str.lower().isin(["true", "1"])
    colonnes = ["parcelle_id", "date_stress", "z_stress", "alerte_stress"]
    if "statut_stress" in etat.columns:
        colonnes.append("statut_stress")
    return etat[colonnes]
