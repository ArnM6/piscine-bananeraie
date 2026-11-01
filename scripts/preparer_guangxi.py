"""Prépare le jeu drone de Guangxi (Science Data Bank, DOI 10.57760/sciencedb.07000, CC BY 4.0).

L'orthomosaïque fournie (5 bandes en réflectance, float32, 8 cm/pixel, 2,3 Go) couvre bien plus que la
plantation. Ce script :
  1. découpe l'image sur le contour de la plantation (+ une marge) ;
  2. la convertit en entiers 16 bits (réflectance x 10 000), compressée : ~10x plus légère, sans perte utile ;
  3. découpe la plantation (18 ha) en parcelles régulières d'environ 1 ha -> parcelles.geojson ;
  4. sépare la plantation en deux ZONES (ouest / est) : l'une pour régler les méthodes, l'autre pour les tester,
     jamais mélangées (découpage honnête par zone) -> zones.geojson ;
  5. convertit les 80 points terrain (40 sains, 40 fusariose) -> reference_terrain.geojson.

Il lit l'image bande par bande pour ne pas saturer la mémoire.

Usage (depuis le dossier du repo) :
    python scripts/preparer_guangxi.py --source data/guangxi --out data/guangxi_pret
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box

MARGE_M = 10            # marge autour du contour de la plantation
COTE_PARCELLE_M = 100   # parcelles de 100 m x 100 m (1 ha)
ECHELLE = 10_000        # réflectance 0,1234 -> 1234


def trouver(dossier, motif):
    fichiers = sorted(Path(dossier).rglob(motif))
    if not fichiers:
        raise SystemExit(f"Aucun fichier '{motif}' dans {dossier} (as-tu bien décompressé les zips ?)")
    return fichiers[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/guangxi")
    parser.add_argument("--out", default="data/guangxi_pret")
    args = parser.parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    contour = gpd.read_file(trouver(args.source, "Boundary_of_banana_planting_region.shp"))
    points = gpd.read_file(trouver(args.source, "Ground_survey_data_of_banana_Fusarium_wilt.shp"))
    image = trouver(args.source, "UAV multispectral reflectance.tif")

    with rasterio.open(image) as src:
        contour = contour.to_crs(src.crs)
        x0, y0, x1, y1 = contour.total_bounds
        fenetre = from_bounds(x0 - MARGE_M, y0 - MARGE_M, x1 + MARGE_M, y1 + MARGE_M, src.transform)
        fenetre = fenetre.round_offsets().round_lengths()
        profil = src.profile.copy()
        profil.update(driver="GTiff", dtype="uint16", nodata=0, compress="deflate", predictor=2, tiled=True,
                      blockxsize=512, blockysize=512, width=int(fenetre.width), height=int(fenetre.height),
                      transform=src.window_transform(fenetre), BIGTIFF="IF_SAFER")
        profil.pop("photometric", None)
        cible = out / "images" / "guangxi_plantation.tif"
        print(f"Image source : {src.count} bandes, {src.width} x {src.height} px, {src.res[0]:.3f} m/px, {src.crs}")
        with rasterio.open(cible, "w", **profil) as dst:
            for b in range(1, src.count + 1):
                bande = src.read(b, window=fenetre).astype("float32")
                valide = np.isfinite(bande) & (bande > 0)
                entiers = np.where(valide, np.clip(np.round(bande * ECHELLE), 1, 65535), 0).astype("uint16")
                dst.write(entiers, b)
                dst.set_band_description(b, src.descriptions[b - 1] or f"bande_{b}")
                print(f"  bande {b} écrite")
            dst.update_tags(echelle_reflectance=str(ECHELLE), source="Science Data Bank 10.57760/sciencedb.07000 (CC BY 4.0)")
        print(f"-> {cible} ({cible.stat().st_size / 1e6:.0f} Mo)")

    # Parcelles : grille de 100 m, découpée par le contour (on garde les morceaux de plus de 0,3 ha)
    plantation = contour.geometry.union_all() if hasattr(contour.geometry, "union_all") else contour.unary_union
    cases = []
    for x in np.arange(x0, x1, COTE_PARCELLE_M):
        for y in np.arange(y0, y1, COTE_PARCELLE_M):
            morceau = box(x, y, x + COTE_PARCELLE_M, y + COTE_PARCELLE_M).intersection(plantation)
            if morceau.area >= 3_000:
                cases.append(morceau)
    parcelles = gpd.GeoDataFrame(geometry=cases, crs=contour.crs)
    parcelles = parcelles.assign(centre_x=parcelles.centroid.x, centre_y=parcelles.centroid.y).sort_values(["centre_y", "centre_x"], ascending=[False, True])
    parcelles["parcelle_id"] = [f"P{i + 1:02d}" for i in range(len(parcelles))]

    # Deux zones séparées par la médiane est-ouest : jamais de mélange entre réglage et test
    milieu = np.median(parcelles.centre_x)
    parcelles["zone"] = np.where(parcelles.centre_x < milieu, "ouest_reglage", "est_test")
    parcelles[["parcelle_id", "zone", "geometry"]].to_crs(4326).to_file(out / "parcelles.geojson", driver="GeoJSON")
    zones = parcelles.dissolve(by="zone").reset_index()[["zone", "geometry"]]
    zones.to_crs(4326).to_file(out / "zones.geojson", driver="GeoJSON")
    print(f"-> {len(parcelles)} parcelles (~1 ha) : {parcelles.zone.value_counts().to_dict()}")

    # Points terrain : 健康植株 = plant sain, 枯萎病植株 = plant atteint de fusariose
    traduction = {"健康植株": "sain", "枯萎病植株": "fusariose"}
    colonne = next(c for c in points.columns if points[c].astype(str).isin(traduction).any())
    reference = gpd.GeoDataFrame({"etat": points[colonne].map(traduction)}, geometry=points.geometry, crs=points.crs)
    reference = gpd.sjoin(reference, parcelles[["parcelle_id", "zone", "geometry"]].to_crs(reference.crs),
                          how="left", predicate="within").drop(columns="index_right")
    reference.to_crs(4326).to_file(out / "reference_terrain.geojson", driver="GeoJSON")
    print(f"-> {len(reference)} points terrain : {reference.groupby(['zone', 'etat']).size().to_dict()}")
    print("Terminé.")


if __name__ == "__main__":
    main()
