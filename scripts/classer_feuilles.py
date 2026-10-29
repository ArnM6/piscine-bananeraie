"""Classe toutes les photos de feuilles d'un dossier (CPU) et écrit un CSV.

Usage :
    python scripts/classer_feuilles.py --images DOSSIER_FEUILLES \
        --modele models/maladies_large_preentraine_aug.onnx --out resultats/feuilles.csv
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bananes.maladies import ClassifieurFeuilles  # noqa: E402

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--modele", default="models/maladies_large_preentraine_aug.onnx")
    parser.add_argument("--out", default="resultats/feuilles.csv")
    parser.add_argument("--lot", type=int, default=16)
    args = parser.parse_args()

    chemins = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not chemins:
        raise SystemExit(f"Aucune image dans {args.images}")
    classifieur = ClassifieurFeuilles(args.modele)

    lignes, debut = [], time.time()
    for i in range(0, len(chemins), args.lot):
        lot = chemins[i:i + args.lot]
        for chemin, res in zip(lot, classifieur.predire([Image.open(p) for p in lot])):
            lignes.append({"image": str(chemin), **res})

    df = pd.DataFrame(lignes)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    duree = time.time() - debut
    print(df.classe.value_counts().to_string())
    print(f"{len(df)} images en {duree:.1f} s ({1000 * duree / len(df):.0f} ms/image) -> {args.out}")


if __name__ == "__main__":
    main()
