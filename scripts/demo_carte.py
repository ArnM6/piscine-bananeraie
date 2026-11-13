"""Démo : une carte cliquable de la plantation, dans un seul fichier HTML (aucun serveur).

Usage (après run.py) :
    python scripts/demo_carte.py --image data/guangxi_pret/images/guangxi_plantation.tif --resultats resultats/

Produit resultats/carte.html, à ouvrir dans un navigateur (connexion Internet nécessaire pour la
bibliothèque de carte Leaflet et le fond satellite). Contenu :
  - l'image du drone en fond, et la carte de vigueur relative en option ;
  - les parcelles colorées selon leur score de santé ;
  - les zones de vigueur faible (alertes) : un clic montre la parcelle, la surface et l'IMAGE qui justifie
    l'alerte (vignette drone à 8 cm, zone entourée en rouge) ;
  - les pieds détectés ;
  - à gauche, la liste des parcelles à visiter en premier ; un clic centre la carte dessus.
Tout est intégré dans le HTML (images en base64) : le fichier peut être envoyé tel quel.
"""
import argparse
import base64
import io
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import from_bounds

ORDRE_BANDES = ["bleu", "vert", "rouge", "pir", "rededge"]   # même convention que bananes/io.py (5 bandes)


def en_base64(img, qualite=80):
    """Image -> texte intégrable dans la page. WebP : ~10× plus léger que PNG, transparence comprise."""
    tampon = io.BytesIO()
    img.save(tampon, format="WEBP", quality=qualite, method=6)
    return f"data:image/webp;base64,{base64.b64encode(tampon.getvalue()).decode()}"


def bornes_couleurs(rgb, valide):
    """Percentiles 2 et 98 de chaque bande (comme QGIS), calculés UNE fois sur toute la plantation."""
    return [tuple(np.percentile(rgb[i][valide], [2, 98])) for i in range(3)]


def etirer(rgb, valide, bornes):
    """Réflectance -> couleurs 0-255 avec les mêmes bornes partout : les vignettes gardent les vraies couleurs.
    Hors image (parcelle qui déborde du vol) : gris clair."""
    sortie = np.zeros(rgb.shape, dtype="uint8")
    for i, (bas, haut) in enumerate(bornes):
        sortie[i] = (np.clip((rgb[i] - bas) / (haut - bas + 1e-9), 0, 1) * 255).astype("uint8")
    sortie[:, ~valide] = 225
    return sortie


def lire_rgb(src, fenetre=None, out_shape=None, bornes_etirement=None):
    """Lit rouge, vert, bleu (réduits si out_shape) et renvoie (tableau 3×H×W en réflectance, masque valide)."""
    idx = [ORDRE_BANDES.index(n) + 1 for n in ("rouge", "vert", "bleu")]
    echelle = float(src.tags().get("echelle_reflectance", 1))
    kw = {"window": fenetre}
    if out_shape:
        kw.update(out_shape=(3, *out_shape), resampling=Resampling.average)
    brut = src.read(idx, **kw).astype("float32")
    valide = brut.sum(axis=0) > 0
    return brut / echelle, valide


def vue_ensemble(src, largeur_max=2400):
    """Image du drone reprojetée en Web Mercator (le système des cartes en ligne), pour la superposer exactement."""
    facteur = max(1, int(np.ceil(src.width / largeur_max)))
    h, w = src.height // facteur, src.width // facteur
    rgb, valide = lire_rgb(src, out_shape=(h, w))
    transform = src.transform * src.transform.scale(src.width / w, src.height / h)
    bornes = bornes_couleurs(rgb, valide)
    couleurs = etirer(rgb, valide, bornes)
    alpha = (valide * 255).astype("uint8")
    return reprojeter(np.concatenate([couleurs, alpha[None]]), transform, src.crs, Resampling.bilinear), bornes


def reprojeter(tableau, transform, crs, resampling):
    """Reprojette un tableau (bandes×H×W) en EPSG:3857 ; renvoie (image PIL RGBA ou L, bornes lat/lon)."""
    n, h, w = tableau.shape
    gauche, haut = transform * (0, 0)
    droite, bas = transform * (w, h)
    dst_t, dw, dh = calculate_default_transform(crs, "EPSG:3857", w, h, gauche, bas, droite, haut)
    sortie = np.zeros((n, dh, dw), dtype=tableau.dtype)
    for i in range(n):
        reproject(tableau[i], sortie[i], src_transform=transform, src_crs=crs, dst_transform=dst_t,
                  dst_crs="EPSG:3857", resampling=resampling)
    x0, y0 = dst_t * (0, 0)
    x1, y1 = dst_t * (dw, dh)
    coins = gpd.GeoSeries(gpd.points_from_xy([x0, x1], [y1, y0]), crs=3857).to_crs(4326)
    bornes = [[coins.iloc[0].y, coins.iloc[0].x], [coins.iloc[1].y, coins.iloc[1].x]]   # [[sud, ouest], [nord, est]]
    mode = "RGBA" if n == 4 else "L"
    img = Image.fromarray(np.moveaxis(sortie, 0, -1) if n > 1 else sortie[0], mode)
    return img, bornes


def carte_vigueur(chemin, largeur_max=2400):
    """Carte d'anomalie de vigueur (rouge = plus faible que la parcelle), transparente hors végétation."""
    with rasterio.open(chemin) as src:
        facteur = max(1, int(np.ceil(src.width / largeur_max)))
        h, w = src.height // facteur, src.width // facteur
        a = src.read(1, out_shape=(h, w), resampling=Resampling.average)
        transform = src.transform * src.transform.scale(src.width / w, src.height / h)
        crs = src.crs
    valide = np.isfinite(a)
    t = np.clip((np.nan_to_num(a) + 3) / 6, 0, 1)                 # -3 -> 0 (rouge), +3 -> 1 (vert)
    rouge = np.where(t < 0.5, 215, 215 - (t - 0.5) * 2 * 189).astype("uint8")
    vert = np.where(t < 0.5, 48 + t * 2 * 172, 220 - (t - 0.5) * 2 * 70).astype("uint8")
    bleu = np.where(t < 0.5, 39 + t * 2 * 60, 99 - (t - 0.5) * 2 * 40).astype("uint8")
    alpha = np.where(valide, np.clip(np.abs(np.nan_to_num(a)) / 2, 0.15, 1) * 230, 0).astype("uint8")
    return reprojeter(np.stack([rouge, vert, bleu, alpha]), transform, crs, Resampling.nearest)


def vignette(src, geom, contours, bornes, pieds=None, marge_m=6, taille_max=420):
    """Extrait de l'image drone autour de `geom`, avec les zones `contours` entourées en rouge."""
    xmin, ymin, xmax, ymax = geom.buffer(marge_m).bounds
    fen = from_bounds(xmin, ymin, xmax, ymax, src.transform).round_offsets().round_lengths()
    facteur = max(1.0, max(fen.width, fen.height) / taille_max)
    forme = (max(1, int(fen.height / facteur)), max(1, int(fen.width / facteur)))
    rgb, valide = lire_rgb(src, fenetre=fen, out_shape=forme)
    img = Image.fromarray(np.moveaxis(etirer(rgb, valide, bornes), 0, -1), "RGB")
    t = src.window_transform(fen)

    def px(x, y):
        c, r = ~t * (x, y)
        return c / facteur, r / facteur

    dessin = ImageDraw.Draw(img)
    if pieds is not None:
        for p in pieds:
            x, y = px(p.x, p.y)
            dessin.ellipse([x - 2, y - 2, x + 2, y + 2], outline=(255, 255, 255))
    for g in contours:
        for poly in getattr(g, "geoms", [g]):
            dessin.line([px(*c) for c in poly.exterior.coords], fill=(255, 40, 40), width=2)
    return en_base64(img, 65)


def main():
    ap = argparse.ArgumentParser(description="Carte cliquable de la plantation (HTML autonome)")
    ap.add_argument("--image", required=True, help="GeoTIFF du drone (celui donné à run.py)")
    ap.add_argument("--resultats", default="resultats", help="dossier de sortie de run.py")
    ap.add_argument("--titre", default="Plantation de Long'an (Guangxi)")
    args = ap.parse_args()
    res = Path(args.resultats)

    with rasterio.open(args.image) as src:
        crs = src.crs
        parcelles = gpd.read_file(res / "parcelles_bulletin.geojson").to_crs(crs)
        alertes = gpd.read_file(res / "alertes.geojson").to_crs(crs) if (res / "alertes.geojson").exists() \
            else gpd.GeoDataFrame(columns=["parcelle_id", "surface_m2", "geometry"], geometry="geometry", crs=crs)
        pieds = gpd.read_file(res / "pieds.geojson").to_crs(crs)
        alertes = alertes[alertes.surface_m2 >= 1].reset_index(drop=True)   # éclats de découpage < 1 m²
        alertes["alerte_id"] = [f"A{i + 1:02d}" for i in range(len(alertes))]

        print("Vue d'ensemble du drone…")
        (fond, bornes), couleurs = vue_ensemble(src)
        print(f"Vignettes : {len(parcelles)} parcelles, {len(alertes)} alertes…")
        vign_parcelles = {r.parcelle_id: vignette(src, r.geometry, list(alertes[alertes.parcelle_id == r.parcelle_id].geometry), couleurs,
                                                  marge_m=3, taille_max=440)
                          for r in parcelles.itertuples()}
        vign_alertes = {}
        for r in alertes.itertuples():
            proches = pieds[pieds.distance(r.geometry) < 12].geometry
            vign_alertes[r.alerte_id] = vignette(src, r.geometry, [r.geometry], couleurs, pieds=proches)

    vig_chemin = next(res.glob("*_vigueur_anomalie.tif"), None)
    couche_vigueur = None
    if vig_chemin:
        print("Carte de vigueur…")
        img, b = carte_vigueur(vig_chemin)
        couche_vigueur = {"url": en_base64(img, 75), "bornes": b}

    # Données pour la page (en degrés) ; géométries simplifiées pour alléger le fichier
    p4326 = parcelles.to_crs(4326)
    colonnes = [c for c in p4326.columns if c != "geometry"]
    for c in colonnes:                                    # dates -> texte (JSON)
        if pd.api.types.is_datetime64_any_dtype(p4326[c]):
            p4326[c] = p4326[c].dt.strftime("%Y-%m-%d")
    p4326[colonnes] = p4326[colonnes].astype(object).where(p4326[colonnes].notna(), None)
    a4326 = alertes.assign(geometry=alertes.geometry.simplify(0.3)).to_crs(4326)[["alerte_id", "parcelle_id", "surface_m2", "geometry"]]
    pieds_xy = [[round(g.y, 6), round(g.x, 6)] for g in pieds.to_crs(4326).geometry]

    bulletin = pd.read_csv(res / "bulletin.csv")
    resume = {
        "surface_ha": round(float(parcelles.area.sum()) / 1e4, 1),
        "n_pieds": int(len(pieds)),
        "n_parcelles": int(len(parcelles)),
        "n_alertes_vigueur": int(bulletin.get("alerte_vigueur", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "n_alertes_stress": int(bulletin.get("alerte_stress", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "date_stress": str(bulletin["date_stress"].dropna().max()) if "date_stress" in bulletin else None,
        "image": str(bulletin["image"].iloc[0]) if "image" in bulletin else "",
    }
    donnees = {
        "titre": args.titre, "resume": resume, "fond": {"url": en_base64(fond, 82), "bornes": bornes},
        "vigueur": couche_vigueur, "parcelles": json.loads(p4326.to_json()), "alertes": json.loads(a4326.to_json()),
        "pieds": pieds_xy, "vignettes_parcelles": vign_parcelles, "vignettes_alertes": vign_alertes,
    }
    # Bonus : tournée de la semaine (scripts/plan_intervention.py), si elle a été calculée
    if (res / "plan_intervention.geojson").exists():
        plan = gpd.read_file(res / "plan_intervention.geojson")
        resume_plan = json.loads((res / "plan_intervention_resume.json").read_text(encoding="utf-8"))
        donnees["plan"] = {"geo": json.loads(plan.to_json()), "resume": resume_plan}
        print(f"Tournée ajoutée : {len(plan) - 1} haltes")
    else:
        donnees["plan"] = None
    gabarit = (Path(__file__).with_name("demo_carte_gabarit.html")).read_text(encoding="utf-8")
    html = gabarit.replace("/*__DONNEES__*/null", json.dumps(donnees, ensure_ascii=False, allow_nan=False))
    sortie = res / "carte.html"
    sortie.write_text(html, encoding="utf-8")
    print(f"Carte écrite : {sortie.resolve()} ({sortie.stat().st_size / 1e6:.1f} Mo). Ouvre-la dans ton navigateur.")


if __name__ == "__main__":
    main()
