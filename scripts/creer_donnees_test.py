"""Crée une petite plantation SYNTHÉTIQUE pour tester la chaîne sans données réelles.

Ce n'est PAS une donnée d'évaluation : elle sert uniquement à vérifier que
run.py tourne de bout en bout (test de fumée). On connaît la vérité terrain
exacte (positions des plants), ce qui permet aussi de tester le calcul du F1.

Usage : python scripts/creer_donnees_test.py --out data/test_synthetique
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

TAILLE_PIXEL = 0.05   # 5 cm/pixel
COTE_M = 60           # zone de 60 m x 60 m
ESPACEMENT_M = 3.0    # un bananier tous les 3 m
ORIGINE = (500_000.0, 9_700_000.0)  # coordonnées UTM 36S (Rwanda / Tanzanie), arbitraires
CRS = "EPSG:32736"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/test_synthetique")
    parser.add_argument("--graine", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.graine)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    n = int(COTE_M / TAILLE_PIXEL)
    yy, xx = np.mgrid[0:n, 0:n] * TAILLE_PIXEL  # coordonnées en mètres depuis le coin haut-gauche

    # Sol brun + bruit
    rgb = np.stack([np.full((n, n), 0.45), np.full((n, n), 0.35), np.full((n, n), 0.25)])
    rgb += rng.normal(0, 0.03, rgb.shape)

    verite = []
    for cx in np.arange(1.5, COTE_M, ESPACEMENT_M):
        for cy in np.arange(1.5, COTE_M, ESPACEMENT_M):
            if rng.random() < 0.08:
                continue  # pied manquant
            cx_j, cy_j = cx + rng.normal(0, 0.2), cy + rng.normal(0, 0.2)
            rayon = rng.uniform(0.9, 1.3)
            # zone stressée : quart sud-est, plants plus jaunes
            stresse = cx > COTE_M * 0.65 and cy > COTE_M * 0.65
            vert = 0.35 if stresse else 0.6
            d2 = (xx - cx_j) ** 2 + (yy - cy_j) ** 2
            poids = np.exp(-d2 / (2 * (rayon / 2) ** 2))
            for bande, valeur in enumerate([0.15 if not stresse else 0.4, vert, 0.12]):
                rgb[bande] = rgb[bande] * (1 - poids) + valeur * poids
            verite.append((ORIGINE[0] + cx_j, ORIGINE[1] - cy_j))

    rgb = (np.clip(rgb, 0, 1) * 255).astype("uint8")
    transform = from_origin(ORIGINE[0], ORIGINE[1], TAILLE_PIXEL, TAILLE_PIXEL)
    with rasterio.open(
        out / "images" / "plantation_synthetique.tif", "w", driver="GTiff",
        height=n, width=n, count=3, dtype="uint8", crs=CRS, transform=transform,
        photometric="RGB", compress="deflate",
    ) as dst:
        dst.write(rgb)

    # Deux parcelles : moitié ouest et moitié est
    x0, y0 = ORIGINE
    parcelles = gpd.GeoDataFrame(
        {"parcelle_id": ["OUEST", "EST"]},
        geometry=[box(x0, y0 - COTE_M, x0 + COTE_M / 2, y0), box(x0 + COTE_M / 2, y0 - COTE_M, x0 + COTE_M, y0)],
        crs=CRS,
    )
    parcelles.to_crs(4326).to_file(out / "parcelles.geojson", driver="GeoJSON")

    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*zip(*verite)), crs=CRS)
    pts.to_crs(4326).to_file(out / "verite_terrain.geojson", driver="GeoJSON")
    print(f"{len(verite)} plants écrits dans {out}")


if __name__ == "__main__":
    main()
