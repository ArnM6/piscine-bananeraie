"""Classification de feuilles (sain / sigatoka / fusarium) avec rejet de l'inconnu.

Inférence CPU uniquement, via ONNX Runtime : pas besoin de PyTorch ni de GPU sur la station de terrain.
Le modèle et ses réglages viennent du notebook notebooks/maladies_colab.ipynb :
  models/maladies_<nom>.onnx   le réseau
  models/maladies_<nom>.json   classes, température, seuil de rejet, normalisation

Règle de décision (« honnête ») :
  1. probabilités = softmax(logits / température)   (température calibrée sur la validation Tanzanie)
  2. si la plus haute probabilité < seuil  ->  « je_ne_sais_pas »
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

    def preparer(self, image):
        """Même préparation qu'à l'entraînement : 256x256 puis 224x224, normalisation ImageNet."""
        image = image.convert("RGB").resize((256, 256), Image.BILINEAR)
        image = image.resize((self.taille, self.taille), Image.BILINEAR)
        tableau = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return (tableau - self.moyenne) / self.ecart_type

    def predire(self, images):
        """images : liste de PIL.Image. Renvoie une liste de dicts {classe, confiance, probas}."""
        lot = np.stack([self.preparer(img) for img in images]).astype(np.float32)
        logits = self.session.run(None, {"image": lot})[0] / self.config["temperature"]
        logits = logits - logits.max(axis=1, keepdims=True)  # stabilité numérique
        probas = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        resultats = []
        for p in probas:
            k = int(p.argmax())
            classe = self.classes[k] if p[k] >= self.config["seuil_rejet"] else JE_NE_SAIS_PAS
            resultats.append({
                "classe": classe,
                "confiance": round(float(p[k]), 4),
                **{f"p_{c}": round(float(v), 4) for c, v in zip(self.classes, p)},
            })
        return resultats
