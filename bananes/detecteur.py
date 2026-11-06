"""Comptage des pieds par DÉTECTEUR APPRIS (RetinaNet, l'architecture de DeepForest).

Les poids viennent du notebook notebooks/comptage_colab.ipynb :
  models/pieds_refo_multires_affine_O.pt   « affine »  : DeepForest -> ReforesTree (4, 8, 16 cm) -> carrés ouest de Guangxi
  models/pieds_refo_multires.pt            « robuste » : DeepForest -> ReforesTree (4, 8, 16 cm), sans Guangxi

Quel modèle choisir (résultats du 05/11/2026, F1 à 1 m) :
  - affine  : meilleur sur une monoculture dense comme Guangxi (F1 0,52 sur la zone test),
              mais spécialisé : 0,17 à 0,31 sur un autre site (ReforesTree, Équateur) ;
  - robuste : un peu moins bon sur Guangxi (0,41) mais stable sur un autre site, à toutes
              les résolutions (0,37 à 0,41).

Préparation de l'image, IDENTIQUE au notebook :
  - multispectral : composition rouge/vert/bleu, étirement entre les centiles 2 et 98 calculés
    sur toute l'image, puis gamma 1/1,5 (une réflectance est linéaire, une photo ne l'est pas) ;
  - RGB : l'image est utilisée telle quelle (comme les photos de ReforesTree) ;
  - rééchantillonnage à 8 cm/pixel si besoin : c'est la résolution vue à l'entraînement.
    Au-delà de 10 cm/pixel, le comptage se dégrade fortement sur canopée dense (F1 0,12 à 16 cm).
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from PIL import Image

RESOLUTION_MODELE = 0.08    # m/pixel
TAILLE_PATCH = 400          # fenêtre vue par le réseau (32 m à 8 cm)
RECOUVREMENT = 80           # les fenêtres se chevauchent de 80 px
NMS_IOU = 0.3               # deux boîtes qui se recouvrent à plus de 30 % = un seul pied

DOSSIER_MODELES = Path(__file__).resolve().parent.parent / "models"
MODELES = {"affine": "refo_multires_affine_O", "robuste": "refo_multires"}


def config_modeles():
    return json.loads((DOSSIER_MODELES / "pieds_modeles.json").read_text(encoding="utf-8"))


def creer_reseau():
    """RetinaNet ResNet-50 + FPN à 1 classe, exactement comme dans le notebook."""
    import torchvision
    from torchvision.models.detection import RetinaNet
    from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
    from torchvision.ops.feature_pyramid_network import LastLevelP6P7
    from torchvision.ops.misc import FrozenBatchNorm2d

    resnet = torchvision.models.resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(resnet, 3, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256))
    return RetinaNet(backbone, num_classes=1, score_thresh=0.05, nms_thresh=NMS_IOU,
                     detections_per_img=500, min_size=2 * TAILLE_PATCH, max_size=2 * TAILLE_PATCH)


def charger(choix="affine"):
    """Renvoie (réseau sur CPU, seuil de confiance)."""
    import torch

    nom = MODELES[choix]
    chemin = DOSSIER_MODELES / f"pieds_{nom}.pt"
    if not chemin.exists():
        raise FileNotFoundError(f"{chemin} introuvable : copie-le depuis MyDrive/piscine-bananeraie/models/")
    reseau = creer_reseau()
    etat = torch.load(chemin, map_location="cpu")
    reseau.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in etat.items()})
    return reseau.eval(), config_modeles()[nom]["seuil"]


def image_rgb(img):
    """Image RGB 8 bits pour le réseau, à partir de lire_image()."""
    c = img["canaux"]
    rgb = np.stack([c["rouge"], c["vert"], c["bleu"]], -1)
    valide = ~img["nodata"]
    if "pir" in c:      # multispectral en réflectance : même recomposition que le notebook
        bas, haut = np.percentile(rgb[valide][::10], [2, 98], axis=0)
        rgb = np.clip((rgb - bas) / (haut - bas), 0, 1) ** (1 / 1.5)
    else:               # photo RGB : déjà « couleur appareil photo »
        rgb = np.clip(rgb, 0, 1)
    rgb = (rgb * 255).astype("uint8")
    rgb[~valide] = 0
    return rgb


def reechantillonner(rgb, transform, taille_pixel, cible=RESOLUTION_MODELE):
    """Ramène l'image à la résolution du modèle. Renvoie (image, transform, facteur)."""
    facteur = taille_pixel / cible          # > 1 : on agrandit (image plus grossière que 8 cm)
    if abs(facteur - 1) < 0.1:
        return rgb, transform, 1.0
    h, w = rgb.shape[:2]
    taille = (max(1, round(w * facteur)), max(1, round(h * facteur)))
    methode = Image.BOX if facteur < 1 else Image.BILINEAR
    rgb = np.array(Image.fromarray(rgb).resize(taille, methode))
    from rasterio.transform import Affine
    return rgb, transform * Affine.scale(w / taille[0], h / taille[1]), facteur


def _positions(n, taille, pas):
    if n <= taille:
        return [0]
    p = list(range(0, n - taille + 1, pas))
    return p if p[-1] == n - taille else p + [n - taille]


def _completer(fenetre, taille=TAILLE_PATCH):
    h, w = fenetre.shape[:2]
    if h >= taille and w >= taille:
        return fenetre
    plein = np.zeros((max(h, taille), max(w, taille), 3), fenetre.dtype)
    plein[:h, :w] = fenetre
    return plein


def predire(reseau, rgb, taille=TAILLE_PATCH, pas=TAILLE_PATCH - RECOUVREMENT, lot=4):
    """Boîtes (pixels) + scores sur une image de n'importe quelle taille, par fenêtres glissantes."""
    import torch
    from torchvision.ops import nms

    fenetres = [(x0, y0) for y0 in _positions(rgb.shape[0], taille, pas)
                for x0 in _positions(rgb.shape[1], taille, pas)]
    boites, scores = [], []
    with torch.no_grad():
        for i in range(0, len(fenetres), lot):
            paquet = fenetres[i:i + lot]
            # Fenêtres entièrement vides (hors orthomosaïque) : rien à chercher
            paquet = [(x0, y0) for x0, y0 in paquet if rgb[y0:y0 + taille, x0:x0 + taille].any()]
            if not paquet:
                continue
            images = [torch.from_numpy(_completer(np.ascontiguousarray(rgb[y0:y0 + taille, x0:x0 + taille])))
                      .permute(2, 0, 1).float().div(255) for x0, y0 in paquet]
            for (x0, y0), sortie in zip(paquet, reseau(images)):
                boites.append(sortie["boxes"] + torch.tensor([x0, y0, x0, y0]))
                scores.append(sortie["scores"])
    if not boites or sum(len(s) for s in scores) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    boites, scores = torch.cat(boites), torch.cat(scores)
    garde = nms(boites, scores, NMS_IOU)       # doublons dans les zones de chevauchement
    return boites[garde].numpy(), scores[garde].numpy()


def detecter_pieds(img, reseau, seuil):
    """Un point par pied détecté (centre de la boîte), avec son score. Même sortie que comptage.detecter_pieds."""
    rgb = image_rgb(img)
    rgb, transform, facteur = reechantillonner(rgb, img["transform"], img["taille_pixel"])
    if img["taille_pixel"] > 0.10:
        print(f"   ⚠ {img['taille_pixel'] * 100:.0f} cm/pixel : trop grossier pour séparer les touffes, "
              "comptage peu fiable (voler à 8 cm ou plus fin)")
    boites, scores = predire(reseau, rgb)
    garde = scores >= seuil
    boites, scores = boites[garde], scores[garde]
    cols = (boites[:, 0] + boites[:, 2]) / 2
    lignes = (boites[:, 1] + boites[:, 3]) / 2
    # Centres tombés dans le remplissage noir (bords de l'image) : écartés
    dedans = (cols < rgb.shape[1]) & (lignes < rgb.shape[0])
    cols, lignes, scores = cols[dedans], lignes[dedans], scores[dedans]
    xs, ys = transform * (cols, lignes) if len(cols) else (np.zeros(0), np.zeros(0))
    return gpd.GeoDataFrame({"score": np.round(scores.astype(float), 3)},
                            geometry=gpd.points_from_xy(xs, ys), crs=img["crs"])
