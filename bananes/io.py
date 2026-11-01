"""Lecture / écriture des données géoréférencées (RGB ou multispectral)."""
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features

# Ordre des bandes supposé selon leur nombre (modifiable avec --bandes dans run.py)
#   3 ou 4 bandes : orthomosaïque RGB classique (la 4e éventuelle = transparence, ignorée)
#   5 bandes      : caméra multispectrale type MicaSense RedEdge (bleu, vert, rouge, proche infrarouge, red-edge)
ORDRES_PAR_DEFAUT = {
    3: ["rouge", "vert", "bleu"],
    4: ["rouge", "vert", "bleu", "alpha"],
    5: ["bleu", "vert", "rouge", "pir", "rededge"],
}


def lister_images(dossier):
    """Renvoie la liste triée des GeoTIFF d'un dossier."""
    dossier = Path(dossier)
    return sorted(set(dossier.glob("*.tif")) | set(dossier.glob("*.tiff")))


def lire_image(chemin, ordre_bandes=None):
    """Lit un GeoTIFF et renvoie un dictionnaire :
      canaux : {"rouge": tableau 2D en réflectance 0-1, "vert": ..., "pir": ...}
      nodata : masque des pixels vides (bords d'orthomosaïque)
      + géoréférencement (transform, crs, profil, taille_pixel, surface_m2)

    On exige un système de coordonnées projeté (en mètres), sinon les distances
    (espacement des plants, appariement à 1 m) n'ont pas de sens.
    """
    with rasterio.open(chemin) as src:
        if src.crs is None:
            raise ValueError(f"{chemin} n'est pas géoréférencée")
        if src.crs.is_geographic:
            raise ValueError(f"{chemin} est en degrés ({src.crs}). Reprojette-la en UTM avant "
                             "(ex : gdalwarp -t_srs EPSG:326XX).")
        ordre = ordre_bandes or ORDRES_PAR_DEFAUT.get(src.count)
        if ordre is None or len(ordre) != src.count:
            raise ValueError(f"{chemin} a {src.count} bandes : précise leur ordre avec --bandes")

        bandes = src.read().astype("float32")
        nodata = src.read_masks(1) == 0
        if "alpha" in ordre:
            nodata |= bandes[ordre.index("alpha")] == 0

        # Mise à l'échelle en réflectance 0-1
        echelle = src.tags().get("echelle_reflectance")
        if echelle:                                   # produit par scripts/preparer_guangxi.py
            bandes /= float(echelle)
        elif src.dtypes[0] == "uint8":
            bandes /= 255.0
        elif src.dtypes[0] == "uint16":
            bandes /= float(np.percentile(bandes[:, ~nodata], 99.9))
        # (float : supposé déjà en réflectance)
        bandes = np.clip(bandes, 0, None)

        taille_pixel = abs(src.transform.a)  # mètres par pixel
        return {
            "canaux": {nom: bandes[i] for i, nom in enumerate(ordre) if nom != "alpha"},
            "nodata": nodata,
            "transform": src.transform,
            "crs": src.crs,
            "profil": src.profile,
            "taille_pixel": taille_pixel,
            "surface_m2": float((~nodata).sum()) * taille_pixel**2,
        }


def lire_parcelles(chemin, crs):
    """Lit les parcelles et les reprojette dans le système de l'image.

    Chaque parcelle reçoit un identifiant 'parcelle_id' (repris du fichier s'il existe).
    """
    parcelles = gpd.read_file(chemin).to_crs(crs)
    if "parcelle_id" not in parcelles.columns:
        parcelles["parcelle_id"] = [f"P{i + 1:02d}" for i in range(len(parcelles))]
    return parcelles[["parcelle_id", "geometry"]]


def rasteriser_parcelles(parcelles, img):
    """Image de la même taille que l'orthophoto : chaque pixel vaut le numéro de sa parcelle (0 = hors parcelle)."""
    hauteur, largeur = img["nodata"].shape
    formes = [(geom, i + 1) for i, geom in enumerate(parcelles.geometry)]
    masque = features.rasterize(formes, out_shape=(hauteur, largeur), transform=img["transform"],
                                fill=0, dtype="int32")
    masque[img["nodata"]] = 0
    return masque


def ecrire_geotiff(chemin, tableau, img):
    """Écrit une carte (1 bande, float32) avec le même géoréférencement que l'image source."""
    profil = img["profil"].copy()
    profil.update(driver="GTiff", count=1, dtype="float32", nodata=np.nan, compress="deflate",
                  tiled=True, blockxsize=512, blockysize=512)
    for cle in ("photometric", "predictor"):
        profil.pop(cle, None)
    tableau = np.where(img["nodata"], np.nan, tableau).astype("float32")
    with rasterio.open(chemin, "w", **profil) as dst:
        dst.write(tableau, 1)
