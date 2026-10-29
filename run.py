"""Point d'entrée de la chaîne de traitement.

Usage :
    python run.py --images DOSSIER --parcelles parcelles.geojson --out resultats/

Pour chaque GeoTIFF RGB du dossier :
  1. lit l'image et les parcelles (reprojetées dans le système de l'image)
  2. calcule la carte de vigueur (ExG, VARI) et les zones anormalement faibles
  3. détecte les pieds (baseline : maxima locaux de verdure)
  4. écrit les sorties géospatiales + le bulletin par parcelle
"""
import argparse
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd

from bananes import bulletin, comptage, io, vigueur


def main():
    parser = argparse.ArgumentParser(description="Suivi de bananeraie par imagerie aérienne")
    parser.add_argument("--images", required=True, help="dossier contenant les GeoTIFF")
    parser.add_argument("--parcelles", required=True, help="fichier GeoJSON des parcelles")
    parser.add_argument("--out", required=True, help="dossier de sortie")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    images = io.lister_images(args.images)
    if not images:
        raise SystemExit(f"Aucun GeoTIFF trouvé dans {args.images}")

    tous_les_pieds = []
    toutes_les_stats = []
    toutes_les_alertes = []

    for chemin in images:
        debut = time.time()
        print(f"[image] {chemin.name}")

        img = io.lire_image(chemin)
        parcelles = io.lire_parcelles(args.parcelles, img["crs"])

        # 1. Vigueur
        exg = vigueur.exg(img["rgb"])
        vari = vigueur.vari(img["rgb"])
        masque_parcelles = io.rasteriser_parcelles(parcelles, img)
        exg_lisse = vigueur.lisser(exg, img["taille_pixel"], rayon_m=2.0)
        anomalie = vigueur.anomalie_par_parcelle(exg_lisse, masque_parcelles)
        io.ecrire_geotiff(out / f"{chemin.stem}_exg.tif", exg, img)
        io.ecrire_geotiff(out / f"{chemin.stem}_vari.tif", vari, img)
        io.ecrire_geotiff(out / f"{chemin.stem}_vigueur_anomalie.tif", anomalie, img)

        # 2. Comptage des pieds
        pieds = comptage.detecter_pieds(exg, img)
        pieds = comptage.rattacher_aux_parcelles(pieds, parcelles)
        pieds["image"] = chemin.name
        tous_les_pieds.append(pieds)

        # 3. Statistiques par parcelle pour le bulletin
        stats = bulletin.stats_parcelles(parcelles, pieds, anomalie, masque_parcelles, img)
        stats["image"] = chemin.name
        toutes_les_stats.append(stats)

        alertes = bulletin.alertes_vigueur(anomalie, parcelles, img)
        alertes["image"] = chemin.name
        toutes_les_alertes.append(alertes)

        hectares = img["surface_m2"] / 10_000
        duree = time.time() - debut
        print(f"   {len(pieds)} pieds | {hectares:.2f} ha | {duree:.1f} s ({duree / max(hectares, 1e-9):.1f} s/ha)")

    pieds = gpd.GeoDataFrame(pd.concat(tous_les_pieds, ignore_index=True), crs=tous_les_pieds[0].crs)
    stats = pd.concat(toutes_les_stats, ignore_index=True)
    alertes = gpd.GeoDataFrame(pd.concat(toutes_les_alertes, ignore_index=True), crs=pieds.crs)

    pieds.to_crs(4326).to_file(out / "pieds.geojson", driver="GeoJSON")
    bulletin.ecrire(stats, parcelles, alertes, out)
    print(f"Terminé. Résultats dans {out.resolve()}")


if __name__ == "__main__":
    main()
