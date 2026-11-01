"""Bulletin par parcelle : chiffres clés, score de santé, alertes rattachées à leur image."""
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from shapely.geometry import shape

SEUIL_FAIBLE = -2.0          # anomalie de vigueur sous laquelle une zone est "faible"
SURFACE_MIN_ALERTE_M2 = 10   # on ignore les zones faibles plus petites (bruit, ombres)
PART_FAIBLE_ALERTE = 0.10    # alerte si plus de 10 % de la parcelle est faible


def zones_faibles(anomalie, img):
    """Polygones des zones de vigueur anormalement faible (pour la carte des alertes)."""
    masque = (anomalie < SEUIL_FAIBLE).astype("uint8")
    polygones = [
        shape(geom)
        for geom, valeur in features.shapes(masque, mask=masque == 1, transform=img["transform"])
        if valeur == 1
    ]
    zones = gpd.GeoDataFrame(geometry=polygones, crs=img["crs"])
    return zones[zones.area >= SURFACE_MIN_ALERTE_M2].reset_index(drop=True)


def stats_parcelles(parcelles, pieds, anomalie, masque_parcelles, img):
    """Une ligne par parcelle : surface, nombre de pieds, densité, part de zones faibles, score."""
    surface_pixel = img["taille_pixel"] ** 2
    lignes = []
    for i, parcelle in enumerate(parcelles.itertuples()):
        dedans = masque_parcelles == i + 1
        surface_ha = dedans.sum() * surface_pixel / 10_000
        if surface_ha == 0:
            continue  # parcelle hors de cette image
        n_pieds = int((pieds["parcelle_id"] == parcelle.parcelle_id).sum())
        valeurs = anomalie[dedans]
        valeurs = valeurs[np.isfinite(valeurs)]            # végétation seulement (pistes, sol exclus)
        part_faible = float(np.mean(valeurs < SEUIL_FAIBLE)) if len(valeurs) else 0.0
        lignes.append({
            "parcelle_id": parcelle.parcelle_id,
            "surface_ha": round(surface_ha, 3),
            "n_pieds": n_pieds,
            "densite_pieds_ha": round(n_pieds / surface_ha, 1),
            "part_vigueur_faible": round(part_faible, 3),
        })
    return pd.DataFrame(lignes)


def alertes_vigueur(anomalie, parcelles, img):
    """Zones faibles découpées par parcelle : chaque alerte connaît sa parcelle."""
    zones = zones_faibles(anomalie, img)
    if len(zones) == 0:
        return gpd.GeoDataFrame(columns=["parcelle_id", "geometry"], geometry="geometry", crs=img["crs"])
    alertes = gpd.overlay(zones, parcelles, how="intersection")
    alertes["type"] = "vigueur_faible"
    alertes["surface_m2"] = alertes.area.round(1)
    return alertes


def score_sante(stats):
    """Score 0-100, volontairement simple et explicable.

    Il ne dépend pour l'instant que de la vigueur ; il sera enrichi par les
    modules stress hydrique et maladies, avec des poids documentés dans le rapport.
    """
    return (100 * (1 - stats["part_vigueur_faible"])).round(0)


def ecrire(stats, parcelles, alertes, out):
    """Écrit bulletin.csv, parcelles_bulletin.geojson, alertes.geojson et bulletin.html."""
    stats = stats.copy()
    stats["score_sante"] = score_sante(stats)
    stats["alerte_vigueur"] = stats["part_vigueur_faible"] > PART_FAIBLE_ALERTE
    stats = stats.sort_values("score_sante")  # les parcelles à visiter en premier en haut

    stats.to_csv(out / "bulletin.csv", index=False)

    carte = parcelles.merge(stats, on="parcelle_id", how="left")
    carte.to_crs(4326).to_file(out / "parcelles_bulletin.geojson", driver="GeoJSON")

    # Chaque alerte garde le nom de l'image source : c'est ce qui permet, dans la
    # démo, de cliquer sur une alerte et d'ouvrir l'image qui la justifie.
    if len(alertes):
        alertes.to_crs(4326).to_file(out / "alertes.geojson", driver="GeoJSON")

    html = stats.to_html(index=False, float_format=lambda v: f"{v:.2f}")
    (out / "bulletin.html").write_text(
        f"<html><head><meta charset='utf-8'><title>Bulletin</title></head>"
        f"<body><h1>Bulletin par parcelle</h1>{html}</body></html>",
        encoding="utf-8",
    )
