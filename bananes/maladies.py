"""Classification de feuilles (sain / sigatoka / fusarium) avec rejet de l'inconnu.

Inférence CPU uniquement, via ONNX Runtime : pas besoin de PyTorch ni de GPU sur la station de terrain.
Le modèle et ses réglages viennent du notebook notebooks/maladies_colab.ipynb :
  models/maladies_<nom>.onnx         le réseau (sorties : logits, caracteristiques)
  models/maladies_<nom>.json         classes, température, méthode de rejet, seuils, normalisation
  models/maladies_<nom>.banque.npy   caractéristiques d'exemples d'entraînement (pour le kNN)

Règle de décision (« honnête ») :
  1. classe = celle de plus haute probabilité
  2. score de familiarité selon la méthode retenue :
       knn     : - distance cosinus au k-ième exemple d'entraînement le plus proche
       energie : T * log(somme(exp(logits / T)))
       msp     : probabilité maximale après calibration (température)
  3. si score < seuil  ->  « je_ne_sais_pas »
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

JE_NE_SAIS_PAS = "je_ne_sais_pas"


class ClassifieurFeuilles:
    def __init__(self, chemin_onnx):
        chemin_onnx = Path(chemin_onnx)
        self.config = json.loads(chemin_onnx.with_suffix(".json").read_text(encoding="utf-8"))
        self.session = ort.InferenceSession(str(chemin_onnx), providers=["CPUExecutionProvider"])
        self.classes = self.config["classes"]
        self.taille = self.config["taille_image"]
        self.moyenne = np.array(self.config["moyenne"], dtype=np.float32).reshape(3, 1, 1)
        self.ecart_type = np.array(self.config["ecart_type"], dtype=np.float32).reshape(3, 1, 1)
        # Ancien format (v1) : un seul seuil sur la probabilité maximale
        self.methode = self.config.get("methode_rejet", "msp")
        self.seuil = self.config["seuils"][self.methode] if "seuils" in self.config else self.config["seuil_rejet"]
        self.banque = None
        if self.methode == "knn":
            self.banque = np.load(chemin_onnx.with_suffix(".banque.npy")).astype(np.float32)

    def preparer(self, image):
        """Même préparation qu'à l'entraînement : 256x256 puis 224x224, normalisation ImageNet."""
        image = image.convert("RGB").resize((256, 256), Image.BILINEAR)
        image = image.resize((self.taille, self.taille), Image.BILINEAR)
        tableau = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return (tableau - self.moyenne) / self.ecart_type

    def familiarite(self, logits, caracteristiques):
        t = self.config["temperature"]
        z = logits / t
        z_max = z.max(axis=1, keepdims=True)
        log_somme = (z_max + np.log(np.exp(z - z_max).sum(axis=1, keepdims=True)))[:, 0]
        if self.methode == "energie":
            return t * log_somme
        if self.methode == "knn":
            f = caracteristiques / np.maximum(np.linalg.norm(caracteristiques, axis=1, keepdims=True), 1e-8)
            distances = 1 - f @ self.banque.T
            k = min(self.config["k_voisins"], self.banque.shape[0])
            return -np.sort(distances, axis=1)[:, k - 1]
        return np.exp(z.max(axis=1) - log_somme)   # msp

    def predire(self, images):
        """images : liste de PIL.Image. Renvoie une liste de dicts {classe, familiarite, p_<classe>...}."""
        lot = np.stack([self.preparer(img) for img in images]).astype(np.float32)
        sorties = self.session.run(None, {"image": lot})
        logits = sorties[0]
        caracteristiques = sorties[1] if len(sorties) > 1 else None
        z = logits / self.config["temperature"]
        probas = np.exp(z - z.max(axis=1, keepdims=True))
        probas /= probas.sum(axis=1, keepdims=True)
        scores = self.familiarite(logits, caracteristiques)
        resultats = []
        for p, s in zip(probas, scores):
            k = int(p.argmax())
            resultats.append({
                "classe": self.classes[k] if s >= self.seuil else JE_NE_SAIS_PAS,
                "classe_brute": self.classes[k],
                "familiarite": round(float(s), 5),
                **{f"p_{c}": round(float(v), 4) for c, v in zip(self.classes, p)},
            })
        return resultats
