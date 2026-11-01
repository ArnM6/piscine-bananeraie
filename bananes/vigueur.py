"""Indices de végétation et carte de vigueur relative.

Justification physique (demandée par l'évaluation) :
- La chlorophylle absorbe le rouge (et le bleu) ; la structure interne des feuilles réfléchit fortement
  le proche infrarouge (PIR). Une feuille vigoureuse = PIR élevé, rouge faible.
- NDVI = (PIR - R) / (PIR + R) : l'indice de référence. Il sature quand la canopée est dense
  (troisième génération de bananiers : feuilles qui se recouvrent).
- NDRE = (PIR - RE) / (PIR + RE) : le red-edge (~720 nm) est la zone où la réflectance remonte brutalement ;
  sa position bouge avec la teneur en chlorophylle. Le NDRE sature moins que le NDVI sur canopée dense et
  réagit plus tôt à un stress (fusariose : jaunissement qui commence par les vieilles feuilles).
- Si l'image n'a que du RGB : ExG = 2g - r - b sur couleurs normalisées (r = R/(R+G+B)), qui retire une
  partie de l'effet d'éclairage, et VARI = (G - R) / (G + R - B). Sans PIR, ces indices sont moins fiables.

La carte de vigueur est RELATIVE : chaque zone est comparée à la médiane de SA parcelle, et seulement sur la
végétation (les pistes, bâtiments et sol nu sont exclus, sinon ils seraient tous signalés comme « faibles »).
"""
import numpy as np
from scipy.ndimage import gaussian_filter

EPS = 1e-6  # évite les divisions par zéro


def ndvi(c):
    return (c["pir"] - c["rouge"]) / (c["pir"] + c["rouge"] + EPS)


def ndre(c):
    return (c["pir"] - c["rededge"]) / (c["pir"] + c["rededge"] + EPS)


def exg(c):
    """Excess Green sur couleurs normalisées. Valeurs ~[-1, 2], plus haut = plus vert."""
    somme = c["rouge"] + c["vert"] + c["bleu"] + EPS
    return (2 * c["vert"] - c["rouge"] - c["bleu"]) / somme


def vari(c):
    """Visible Atmospherically Resistant Index, borné à [-1, 1]."""
    return np.clip((c["vert"] - c["rouge"]) / (c["vert"] + c["rouge"] - c["bleu"] + EPS), -1, 1)


def indices(img):
    """Calcule tous les indices possibles selon les bandes disponibles.

    Renvoie (dictionnaire nom -> carte, nom de l'indice de vigueur retenu, masque de végétation).
    """
    c = img["canaux"]
    cartes = {}
    if "pir" in c and "rouge" in c:
        cartes["ndvi"] = ndvi(c)
        if "rededge" in c:
            cartes["ndre"] = ndre(c)
        principal = "ndre" if "ndre" in cartes else "ndvi"
        vegetation = cartes["ndvi"] > 0.3          # sol nu, pistes, eau : NDVI < 0,3
    else:
        cartes["exg"] = exg(c)
        cartes["vari"] = vari(c)
        principal = "exg"
        vegetation = cartes["exg"] > 0.05
    vegetation &= ~img["nodata"]
    return cartes, principal, vegetation


def lisser(indice, taille_pixel, rayon_m=2.0, masque=None):
    """Moyenne locale sur ~rayon_m mètres, calculée seulement sur les pixels du masque.

    À quelques cm/pixel, un pixel isolé (une feuille, une ombre) ne veut rien dire : on raisonne
    à l'échelle d'un plant (~2 m). Avec un masque, les pistes voisines ne « tirent » pas la moyenne vers le bas.
    """
    sigma = rayon_m / taille_pixel
    if masque is None:
        return gaussian_filter(indice, sigma=sigma)
    poids = gaussian_filter(masque.astype("float32"), sigma=sigma)
    somme = gaussian_filter(np.where(masque, indice, 0).astype("float32"), sigma=sigma)
    return np.where(poids > 0.05, somme / np.maximum(poids, EPS), np.nan)


def anomalie_par_parcelle(indice, masque_parcelles, vegetation=None):
    """Score robuste : écart de chaque pixel à la médiane de SA parcelle, divisé par la dispersion (MAD).

    -2 signifie « nettement moins vigoureux que le reste de la parcelle ».
    On compare chaque parcelle à elle-même : une parcelle entière plus sombre à cause d'un nuage ou
    d'un autre vol n'est pas marquée comme anormale. Hors végétation : NaN (ni bon ni mauvais).
    """
    anomalie = np.full(indice.shape, np.nan, dtype="float32")
    for numero in np.unique(masque_parcelles):
        if numero == 0:
            continue  # hors parcelle
        dedans = masque_parcelles == numero
        if vegetation is not None:
            dedans &= vegetation
        dedans &= np.isfinite(indice)
        if dedans.sum() < 100:
            continue
        valeurs = indice[dedans]
        mediane = np.median(valeurs)
        mad = np.median(np.abs(valeurs - mediane)) * 1.4826 + EPS  # 1.4826 : MAD -> écart-type
        anomalie[dedans] = (valeurs - mediane) / mad
    return anomalie
