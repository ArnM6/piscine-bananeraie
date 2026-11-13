"""Bonus : plan d'intervention de la semaine (où aller, dans quel ordre, combien de temps).

Usage (après run.py) :
    python scripts/plan_intervention.py --resultats resultats/

Logique :
  1. Points de visite = zones de vigueur faible, regroupées si elles sont à moins de 8 m (une seule halte).
  2. Priorité 1 : toutes les haltes des parcelles en alerte vigueur (> 10 % de la parcelle faible) et des parcelles
     signalées plus sèches par Sentinel-2. Priorité 2 : ailleurs, seulement les haltes d'au moins 50 m²
     (les petites taches isolées attendent le vol suivant).
  3. Tournée à pied : d'abord toutes les haltes de priorité 1, puis celles de priorité 2 (si le temps manque, on
     s'arrête après la priorité 1). Ordre optimisé (plus proche voisin + amélioration 2-opt).
  4. Temps estimé, comparé à une inspection complète de la plantation à pied.
Les hypothèses de temps sont des paramètres (voir --aide) : elles se règlent avec le chef de culture.

Sorties : resultats/plan_intervention.csv (une ligne par halte, dans l'ordre) et plan_intervention.geojson (haltes + trajet),
affichés par scripts/demo_carte.py s'ils existent.
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

REGROUPEMENT_M = 8          # deux zones à moins de 8 m = une seule halte
SURFACE_MIN_P2_M2 = 50      # hors parcelles en alerte : seulement les zones d'au moins 50 m²
VITESSE_KMH = 3.0           # marche dans les rangs de bananiers
DETOUR = 1.3                # on ne marche pas en ligne droite entre les rangs
MINUTES_PAR_HALTE = 10      # observer, photographier 5 à 10 feuilles (classifieur maladies), noter les pieds manquants
LARGEUR_INSPECTION_M = 12   # inspection complète : un agent voit ~6 m de chaque côté de son passage


def ordre_tournee(points, depart):
    """Ordre de visite court : plus proche voisin depuis `depart`, puis 2-opt (inverse des tronçons tant que ça raccourcit)."""
    n = len(points)
    if n == 0:
        return []
    restants = list(range(n))
    ordre, pos = [], np.asarray(depart)
    while restants:
        i = min(restants, key=lambda k: np.hypot(*(points[k] - pos)))
        ordre.append(i); restants.remove(i); pos = points[i]

    def longueur(o):
        chemin = [np.asarray(depart)] + [points[k] for k in o]
        return sum(np.hypot(*(b - a)) for a, b in zip(chemin[:-1], chemin[1:]))

    ameliore = True
    while ameliore:
        ameliore = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                essai = ordre[:i] + ordre[i:j + 1][::-1] + ordre[j + 1:]
                if longueur(essai) < longueur(ordre) - 1e-6:
                    ordre, ameliore = essai, True
    return ordre


def main():
    ap = argparse.ArgumentParser(description="Plan d'intervention de la semaine")
    ap.add_argument("--resultats", default="resultats", help="dossier de sortie de run.py")
    ap.add_argument("--depart", help="point de départ « lat,lon » (défaut : centre de la plantation)")
    args = ap.parse_args()
    res = Path(args.resultats)

    parcelles = gpd.read_file(res / "parcelles_bulletin.geojson")
    crs = parcelles.estimate_utm_crs()
    parcelles = parcelles.to_crs(crs)
    alertes = gpd.read_file(res / "alertes.geojson").to_crs(crs)
    alertes = alertes[alertes.surface_m2 >= 1]

    def vrai(col):
        return parcelles[col].fillna(False).astype(str).str.lower().isin(["true", "1"]) if col in parcelles else False
    parcelles["prioritaire"] = vrai("alerte_vigueur") | vrai("alerte_stress")
    prioritaires = set(parcelles.loc[parcelles.prioritaire, "parcelle_id"])

    # 1. Haltes : zones regroupées à moins de REGROUPEMENT_M
    groupes = gpd.GeoDataFrame(geometry=list(getattr(u := alertes.buffer(REGROUPEMENT_M / 2).union_all(), "geoms", [u])), crs=crs)
    alertes = gpd.sjoin(alertes, groupes, predicate="intersects").rename(columns={"index_right": "groupe"})
    haltes = []
    for g, z in alertes.groupby("groupe"):
        surface = float(z.surface_m2.sum())
        parcelle = z.groupby("parcelle_id").surface_m2.sum().idxmax()
        priorite = 1 if parcelle in prioritaires else (2 if surface >= SURFACE_MIN_P2_M2 else None)
        if priorite is None:
            continue
        centre = z.union_all().representative_point()   # toujours DANS une zone faible
        haltes.append({"parcelle_id": parcelle, "priorite": priorite, "n_zones": len(z),
                       "surface_m2": round(surface, 1), "x": centre.x, "y": centre.y})
    haltes = pd.DataFrame(haltes)
    if haltes.empty:
        raise SystemExit("Aucune zone à visiter cette semaine.")

    # 2. Tournée : priorité 1 puis priorité 2
    if args.depart:
        lat, lon = map(float, args.depart.split(","))
        dep = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(crs).iloc[0]
    else:
        dep = parcelles.union_all().centroid
    depart = np.array([dep.x, dep.y])
    ordre_final, pos = [], depart
    for p in (1, 2):
        idx = haltes.index[haltes.priorite == p].tolist()
        pts = [haltes.loc[i, ["x", "y"]].to_numpy(float) for i in idx]
        o = ordre_tournee(pts, pos)
        ordre_final += [idx[k] for k in o]
        if o:
            pos = pts[o[-1]]
    haltes = haltes.loc[ordre_final].reset_index(drop=True)
    haltes.insert(0, "ordre", range(1, len(haltes) + 1))

    xy = np.vstack([depart, haltes[["x", "y"]].to_numpy(float), depart])     # retour au départ
    pas = np.hypot(*np.diff(xy, axis=0).T) * DETOUR
    haltes["distance_depuis_precedente_m"] = pas[:-1].round(0)
    haltes["minutes_cumulees"] = ((np.cumsum(pas[:-1]) / 1000 / VITESSE_KMH * 60)
                                  + MINUTES_PAR_HALTE * haltes.ordre).round(0)
    haltes["action"] = np.where(haltes.priorite == 1,
                                "observer, photographier 5-10 feuilles (classifieur), noter les pieds manquants",
                                "contrôle rapide : la tache s'étend-elle ?")

    def duree(n, metres):
        return metres / 1000 / VITESSE_KMH * 60 + MINUTES_PAR_HALTE * n
    retour_p1 = haltes[haltes.priorite == 1]
    n1 = len(retour_p1)
    if n1:
        dernier = retour_p1[["x", "y"]].to_numpy(float)[-1]
        m_p1 = pas[:n1].sum() + np.hypot(*(dernier - depart)) * DETOUR
    else:
        m_p1 = 0.0
    m_tout = pas.sum()
    surface_m2 = float(parcelles.area.sum())
    m_complet = surface_m2 / LARGEUR_INSPECTION_M
    resume = {
        "parcelles_prioritaires": sorted(prioritaires),
        "haltes_priorite_1": n1, "haltes_priorite_2": int((haltes.priorite == 2).sum()),
        "duree_priorite_1_h": round(duree(n1, m_p1) / 60, 1),
        "duree_tournee_complete_h": round(duree(len(haltes), m_tout) / 60, 1),
        "distance_tournee_km": round(m_tout / 1000, 1),
        "inspection_complete_km": round(m_complet / 1000, 1),
        "inspection_complete_h": round(m_complet / 1000 / VITESSE_KMH, 1),
        "hypotheses": {"vitesse_kmh": VITESSE_KMH, "detour": DETOUR, "minutes_par_halte": MINUTES_PAR_HALTE,
                       "largeur_inspection_m": LARGEUR_INSPECTION_M, "regroupement_m": REGROUPEMENT_M,
                       "surface_min_priorite_2_m2": SURFACE_MIN_P2_M2},
    }
    resume["gain"] = (f"marche : {resume['inspection_complete_km']} km pour tout inspecter -> {resume['distance_tournee_km']} km "
                      f"ciblés ; priorités vues en {resume['duree_priorite_1_h']} h")

    # 3. Sorties
    pts = gpd.GeoSeries(gpd.points_from_xy(haltes.x, haltes.y), crs=crs)
    lat_lon = pts.to_crs(4326)
    haltes["lat"], haltes["lon"] = lat_lon.y.round(6), lat_lon.x.round(6)
    colonnes = ["ordre", "priorite", "parcelle_id", "n_zones", "surface_m2", "distance_depuis_precedente_m",
                "minutes_cumulees", "action", "lat", "lon"]
    haltes[colonnes].to_csv(res / "plan_intervention.csv", index=False)
    geo = gpd.GeoDataFrame(haltes[colonnes], geometry=list(pts), crs=crs)
    trajet = gpd.GeoDataFrame({"ordre": [0], "priorite": [0]}, geometry=[LineString(xy)], crs=crs)
    sortie = pd.concat([geo, trajet], ignore_index=True)
    sortie = gpd.GeoDataFrame(sortie, geometry="geometry", crs=crs).to_crs(4326)
    sortie.to_file(res / "plan_intervention.geojson", driver="GeoJSON")
    (res / "plan_intervention_resume.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Parcelles prioritaires : {', '.join(resume['parcelles_prioritaires']) or 'aucune'}")
    print(f"Priorité 1 : {n1} haltes, {resume['duree_priorite_1_h']} h aller-retour")
    print(f"Tournée complète : {len(haltes)} haltes, {resume['distance_tournee_km']} km, {resume['duree_tournee_complete_h']} h")
    print(f"Inspection de toute la plantation à pied : {resume['inspection_complete_km']} km, ~{resume['inspection_complete_h']} h (sans arrêts)")
    print(haltes[["ordre", "priorite", "parcelle_id", "surface_m2", "minutes_cumulees"]].to_string(index=False))


if __name__ == "__main__":
    main()
