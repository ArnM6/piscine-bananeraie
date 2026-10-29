"""Évalue le comptage comme l'orga : F1 avec appariement à 1 m + erreur de comptage par parcelle.

Usage :
    python scripts/evaluer_comptage.py --pred resultats/pieds.geojson \
        --verite data/test_synthetique/verite_terrain.geojson \
        --parcelles data/test_synthetique/parcelles.geojson
"""
import argparse

import geopandas as gpd
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

DISTANCE_MAX_M = 1.0


def apparier(pred_xy, verite_xy, distance_max=DISTANCE_MAX_M):
    """Appariement un-à-un (algorithme hongrois) : chaque vrai plant compte au plus une fois.

    Renvoie le nombre de vrais positifs.
    """
    if len(pred_xy) == 0 or len(verite_xy) == 0:
        return 0
    distances = cdist(pred_xy, verite_xy)
    # Les paires trop éloignées sont rendues "interdites" par un coût énorme
    cout = np.where(distances <= distance_max, distances, 1e6)
    lignes, colonnes = linear_sum_assignment(cout)
    return int((distances[lignes, colonnes] <= distance_max).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--verite", required=True)
    parser.add_argument("--parcelles", required=True)
    args = parser.parse_args()

    pred = gpd.read_file(args.pred)
    verite = gpd.read_file(args.verite)
    crs_metrique = pred.estimate_utm_crs()  # distances en mètres
    pred, verite = pred.to_crs(crs_metrique), verite.to_crs(crs_metrique)
    parcelles = gpd.read_file(args.parcelles).to_crs(crs_metrique)

    xy = lambda g: np.c_[g.geometry.x, g.geometry.y]
    vp = apparier(xy(pred), xy(verite))
    precision = vp / max(len(pred), 1)
    rappel = vp / max(len(verite), 1)
    f1 = 2 * precision * rappel / max(precision + rappel, 1e-9)
    print(f"Prédits : {len(pred)} | Vrais : {len(verite)} | Appariés : {vp}")
    print(f"Précision {precision:.3f} | Rappel {rappel:.3f} | F1 {f1:.3f}")

    if "parcelle_id" not in parcelles.columns:
        parcelles["parcelle_id"] = [f"P{i + 1:02d}" for i in range(len(parcelles))]
    print("\nErreur de comptage par parcelle :")
    for parcelle in parcelles.itertuples():
        n_pred = int(pred.within(parcelle.geometry).sum())
        n_vrai = int(verite.within(parcelle.geometry).sum())
        erreur = (n_pred - n_vrai) / max(n_vrai, 1)
        print(f"  {parcelle.parcelle_id:>8} : prédit {n_pred:4d} | vrai {n_vrai:4d} | erreur {erreur:+.1%}")


if __name__ == "__main__":
    main()
