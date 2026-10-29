"""Indices de végétation RGB et carte de vigueur relative.

Pourquoi ces indices (justification physique, demandée par l'évaluation) :
- La chlorophylle absorbe le rouge et le bleu et réfléchit le vert.
  Une feuille vigoureuse est donc "plus verte que rouge et bleu".
- ExG (Excess Green) = 2g - r - b, calculé sur les couleurs normalisées
  (r = R / (R+G+B), etc.) : la normalisation retire en partie l'effet de
  l'éclairage global, ce qui compte car la lumière change d'un vol à l'autre.
- VARI = (G - R) / (G + R - B) : conçu pour limiter l'effet de l'atmosphère,
  utile en second avis.
Sans proche infrarouge, on ne peut PAS calculer de NDVI : ces indices sont
moins fiables, d'où une carte en écart relatif à la parcelle plutôt qu'en
valeur absolue.
"""
import numpy as np

EPS = 1e-6  # évite les divisions par zéro


def exg(rgb):
    """Excess Green sur couleurs normalisées. Valeurs ~[-1, 2], plus haut = plus vert."""
    somme = rgb.sum(axis=0) + EPS
    r, g, b = rgb[0] / somme, rgb[1] / somme, rgb[2] / somme
    return 2 * g - r - b


def vari(rgb):
    """Visible Atmospherically Resistant Index, borné à [-1, 1]."""
    r, g, b = rgb
    return np.clip((g - r) / (g + r - b + EPS), -1, 1)


def lisser(indice, taille_pixel, rayon_m=2.0):
    """Moyenne locale sur ~rayon_m mètres.

    À 2 cm/pixel, un pixel isolé (une feuille, une ombre) ne veut rien dire.
    On raisonne à l'échelle d'un plant (~2 m) : "cette zone est-elle moins verte ?".
    """
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(indice, sigma=rayon_m / taille_pixel)


def anomalie_par_parcelle(indice, masque_parcelles):
    """Score robuste : écart de chaque pixel à la médiane de SA parcelle, divisé par la dispersion (MAD).

    -2 signifie "nettement moins vert que le reste de la parcelle".
    On compare chaque parcelle à elle-même : une parcelle entière plus sombre
    à cause d'un nuage ou d'un autre vol n'est pas marquée comme anormale.
    """
    anomalie = np.full(indice.shape, np.nan, dtype="float32")
    for numero in np.unique(masque_parcelles):
        if numero == 0:
            continue  # hors parcelle
        dedans = masque_parcelles == numero
        valeurs = indice[dedans]
        mediane = np.median(valeurs)
        mad = np.median(np.abs(valeurs - mediane)) * 1.4826 + EPS  # 1.4826 : MAD -> écart-type
        anomalie[dedans] = (valeurs - mediane) / mad
    return anomalie
