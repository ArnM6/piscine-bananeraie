"""Lecture / écriture des données géoréférencées."""
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features


def lister_images(dossier):
    """Renvoie la liste triée des GeoTIFF d'un dossier."""
    dossier = Path(dossier)
    return sorted(list(dossier.glob("*.tif")) + list(dossier.glob("*.tiff")))


def lire_image(chemin):
    """Lit un GeoTIFF et renvoie un dictionnaire avec l'image RGB (float 0-1) et son géoréférencement.

    On exige un système de coordonnées projeté (en mètres), sinon les distances
    (espacement des plants, appariement à 1 m) n'ont pas de sens.
    """
    with rasterio.open(chemin) as src:
        if src.crs is None:
            raise ValueError(f"{chemin} n'est pas géoréférencée")
        if src.crs.is_geographic:
            raise ValueError(
                f"{chemin} est en degrés ({src.crs}). Reprojette-la en UTM avant "
                "(ex : gdalwarp -t_srs EPSG:326XX)."
            )
        if src.count < 3:
            raise ValueError(f"{chemin} a {src.count} bande(s), il en faut au moins 3 (RGB)")

        rgb = src.read([1, 2, 3]).astype("float32")
        nodata = src.read_masks(1) == 0  # pixels hors image (bords d'orthomosaïque)

        # Normalisation en 0-1 selon le type de l'image (8 bits ou 16 bits)
        maximum = 255.0 if src.dtypes[0] == "uint8" else float(np.percentile(rgb, 99.9))
        rgb = np.clip(rgb / maximum, 0, 1)

        taille_pixel = abs(src.transform.a)  # mètres par pixel
        return {
            "rgb": rgb,
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
    hauteur, largeur = img["rgb"].shape[1:]
    formes = [(geom, i + 1) for i, geom in enumerate(parcelles.geometry)]
    return features.rasterize(
        formes, out_shape=(hauteur, largeur), transform=img["transform"], fill=0, dtype="int32"
    )


def ecrire_geotiff(chemin, tableau, img):
    """Écrit une carte (1 bande, float32) avec le même géoréférencement que l'image source."""
    profil = img["profil"].copy()
    profil.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")
    profil.pop("photometric", None)  # l'image source est en RGB, la carte n'a qu'une bande
    tableau = np.where(img["nodata"], np.nan, tableau).astype("float32")
    with rasterio.open(chemin, "w", **profil) as dst:
        dst.write(tableau, 1)
