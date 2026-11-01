"""Comptage des pieds — BASELINE CLASSIQUE (semaine 2 : à comparer ensuite à un détecteur appris).

Idée : vu du dessus, un bananier est une "étoile" de feuilles vertes autour du
pseudo-tronc. Si on floute l'image de verdure (NDVI, ou ExG en RGB) à l'échelle d'un plant,
chaque plant devient une "bosse" : on cherche les sommets de ces bosses.

Gestion des rejets : un rejet pousse collé au pied mère (< ~1,5 m). En imposant
une distance minimale entre deux sommets, le rejet est fusionné avec sa mère :
on compte des TOUFFES (le pied mère et ses rejets), pas des feuilles.
Limite connue : un gros rejet bien développé peut être compté à part.
"""
import geopandas as gpd
import numpy as np
from rasterio.transform import xy
from scipy.ndimage import gaussian_filter, maximum_filter

# Paramètres en MÈTRES (convertis en pixels selon la résolution de l'image)
RAYON_PLANT_M = 0.8       # flou appliqué : à peu près la taille d'un demi-plant
DISTANCE_MIN_M = 1.5      # deux pieds sont au moins à 1,5 m l'un de l'autre
# Valeur minimale de l'indice lissé pour accepter un sommet (sinon c'est du sol), selon l'indice utilisé
SEUILS_VERDURE = {"ndvi": 0.40, "exg": 0.05}


def detecter_pieds(indice, img, nom_indice="exg", rayon_m=RAYON_PLANT_M, distance_min_m=DISTANCE_MIN_M, seuil=None):
    """Renvoie un GeoDataFrame de points (un par pied détecté) avec un score de confiance."""
    px = img["taille_pixel"]
    seuil = SEUILS_VERDURE[nom_indice] if seuil is None else seuil

    lisse = gaussian_filter(np.nan_to_num(indice), sigma=rayon_m / px)
    lisse[img["nodata"]] = -np.inf

    # Un pixel est un sommet s'il est le maximum de son voisinage de rayon distance_min
    taille_fenetre = max(3, int(round(2 * distance_min_m / px)) | 1)  # impair
    est_sommet = (lisse == maximum_filter(lisse, size=taille_fenetre)) & (lisse > seuil)

    lignes, colonnes = np.nonzero(est_sommet)
    xs, ys = xy(img["transform"], lignes, colonnes)

    return gpd.GeoDataFrame(
        {"score": lisse[lignes, colonnes].astype(float)},
        geometry=gpd.points_from_xy(xs, ys),
        crs=img["crs"],
    )


def rattacher_aux_parcelles(pieds, parcelles):
    """Ajoute la colonne parcelle_id à chaque pied (et retire ceux hors parcelle)."""
    joint = gpd.sjoin(pieds, parcelles, how="inner", predicate="within")
    return joint.drop(columns="index_right").reset_index(drop=True)
