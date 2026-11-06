"""Point d'entrée de la chaîne de traitement.

Usage :
    python run.py --images DOSSIER --parcelles parcelles.geojson --out resultats/

Pour chaque GeoTIFF du dossier (RGB ou multispectral 5 bandes) :
  1. lit l'image et les parcelles (reprojetées dans le système de l'image)
  2. calcule les indices (NDVI, NDRE si multispectral ; ExG, VARI si RGB) et la carte de vigueur relative
  3. détecte les pieds : détecteur appris (RetinaNet / DeepForest, par défaut) ou baseline (maxima locaux)
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
    parser.add_argument("--bandes", help="ordre des bandes, ex : bleu,vert,rouge,pir,rededge (défaut : selon leur nombre)")
    parser.add_argument("--comptage", choices=["affine", "robuste", "baseline"], default="affine",
                        help="affine : meilleur sur une monoculture dense type Guangxi (défaut) ; "
                             "robuste : plus stable sur un site très différent ; baseline : maxima locaux, sans réseau")
    args = parser.parse_args()
    ordre_bandes = args.bandes.split(",") if args.bandes else None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    images = io.lister_images(args.images)
    if not images:
        raise SystemExit(f"Aucun GeoTIFF trouvé dans {args.images}")

    reseau, seuil = None, None
    if args.comptage != "baseline":
        try:
            from bananes import detecteur
            reseau, seuil = detecteur.charger(args.comptage)
            print(f"[comptage] détecteur « {args.comptage} » ({detecteur.MODELES[args.comptage]}), seuil {seuil}")
        except (ImportError, FileNotFoundError) as e:
            print(f"[comptage] détecteur indisponible ({e}) -> baseline")
            args.comptage = "baseline"

    tous_les_pieds = []
    toutes_les_stats = []
    toutes_les_alertes = []

    for chemin in images:
        debut = time.time()
        print(f"[image] {chemin.name}")

        img = io.lire_image(chemin, ordre_bandes)
        parcelles = io.lire_parcelles(args.parcelles, img["crs"])

        # 1. Vigueur : indices, puis écart à la médiane de la parcelle, sur la végétation seulement
        cartes, principal, vegetation = vigueur.indices(img)
        masque_parcelles = io.rasteriser_parcelles(parcelles, img)
        lisse = vigueur.lisser(cartes[principal], img["taille_pixel"], rayon_m=2.0, masque=vegetation)
        anomalie = vigueur.anomalie_par_parcelle(lisse, masque_parcelles, vegetation)
        for nom, carte in cartes.items():
            io.ecrire_geotiff(out / f"{chemin.stem}_{nom}.tif", carte, img)
        io.ecrire_geotiff(out / f"{chemin.stem}_vigueur_anomalie.tif", anomalie, img)
        print(f"   bandes : {', '.join(img['canaux'])} | indice de vigueur : {principal.upper()}")

        # 2. Comptage des pieds
        if reseau is not None:
            pieds = detecteur.detecter_pieds(img, reseau, seuil)
        else:   # baseline : sur le NDVI si disponible (meilleur contraste plant / sol que l'ExG)
            nom_comptage = "ndvi" if "ndvi" in cartes else "exg"
            pieds = comptage.detecter_pieds(cartes[nom_comptage], img, nom_comptage)
        pieds = comptage.rattacher_aux_parcelles(pieds, parcelles)
        pieds["image"] = chemin.name
        tous_les_pieds.append(pieds)
        del cartes, lisse

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
