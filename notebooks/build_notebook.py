"""Génère notebooks/maladies_colab.ipynb à partir des cellules ci-dessous.

On écrit le notebook en Python pour pouvoir le relire et le versionner proprement.
Usage : python notebooks/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

CELLULES = []


def md(texte):
    CELLULES.append(nbf.v4.new_markdown_cell(texte.strip()))


def code(texte):
    CELLULES.append(nbf.v4.new_code_cell(texte.strip()))


md("""
# Module maladies — feuilles de bananier

**Entraînement sur la Tanzanie, test sur le Bangladesh**, avec rejet de l'inconnu.

| Rôle | Jeu | Classes |
|---|---|---|
| Entraînement + validation | Tanzanie (Mduma & Elinisa, Zenodo 7670326) | sain, sigatoka, fusarium |
| Test (autre pays) | BananaLSD, Bangladesh (dossier *original* seulement) | sain, sigatoka = **connues** ; cordana, pestalotiopsis = **inconnues** |

Le modèle doit :
1. reconnaître la cercosporiose (Sigatoka) ;
2. ne pas alerter sur les feuilles saines ;
3. dire **« je ne sais pas »** face à cordana et pestalotiopsis, qu'il n'a jamais vues.

**Avant de lancer** : *Exécution → Modifier le type d'exécution → GPU T4*, puis *Exécution → Tout exécuter*.
Tout ce qui compte (modèle ONNX, résultats, listes de découpage) est sauvegardé sur ton Google Drive
dans `MyDrive/piscine-bananeraie/`.
""")

md("## 0. Réglages")
code("""
# Mets False pour aller vite (un seul entraînement). True = ablations pour le rapport (~4 entraînements).
FAIRE_ABLATIONS = True

EPOQUES = 8
TAILLE_IMAGE = 224
GRAINE = 42
TAUX_ACCEPTATION_VAL = 0.95   # le seuil de rejet garde 95 % des bonnes réponses de validation

# Mode test : petites fausses données, pour vérifier que le notebook tourne (ne pas toucher)
MODE_TEST = False
""")

md("## 1. Installation et Google Drive")
code("""
import os, sys, json, time, random, shutil, zipfile, tarfile, hashlib
from pathlib import Path
from collections import Counter

import subprocess
if not MODE_TEST:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "imagehash",
                    "datasets", "onnxscript"], check=True)

import numpy as np
import pandas as pd
import torch, torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image, ImageFile
import imagehash

ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(GRAINE); np.random.seed(GRAINE); torch.manual_seed(GRAINE)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Calcul sur :", DEVICE)

TRAVAIL = Path("/content") if Path("/content").exists() else Path.cwd() / "colab_travail"
DONNEES = TRAVAIL / "donnees"
CACHE = TRAVAIL / "cache_256"

if Path("/content").exists() and not MODE_TEST:
    from google.colab import drive
    drive.mount("/content/drive")
    SORTIE = Path("/content/drive/MyDrive/piscine-bananeraie")
else:
    SORTIE = TRAVAIL / "sortie"
for d in [DONNEES, CACHE, SORTIE / "models", SORTIE / "resultats_maladies"]:
    d.mkdir(parents=True, exist_ok=True)
print("Sauvegardes dans :", SORTIE)
""")

md("""
## 2. Téléchargement

- **Tanzanie** : via l'API Zenodo. Le notebook liste les fichiers du dépôt, les télécharge et les décompresse.
- **Bangladesh** : BananaLSD, copie officielle du sous-ensemble *raw* (937 originales) publiée sur Hugging Face
  par Project AgML (CC BY 4.0). On **n'utilise jamais** le sous-ensemble *augmented* : il est fabriqué à partir
  des originales, ce serait une fuite.
""")
code("""
import requests

def telecharger(url, destination):
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(destination, "wb") as f:
            for morceau in r.iter_content(1 << 20):
                f.write(morceau)
    return destination

def decompresser(archive, dossier):
    archive = Path(archive)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dossier)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(dossier)

TANZANIE = DONNEES / "tanzanie"
TANZANIE.mkdir(exist_ok=True)

if not MODE_TEST:
    infos = requests.get("https://zenodo.org/api/records/7670326", timeout=60).json()
    print("Titre   :", infos["metadata"]["title"])
    print("Licence :", infos["metadata"].get("license", {}))
    print("Version :", infos["metadata"].get("version"), "| publié le", infos["metadata"].get("publication_date"))
    for f in infos["files"]:
        print(f"  - {f['key']}  ({f['size'] / 1e6:.0f} Mo)")
        chemin = telecharger(f["links"]["self"], TANZANIE / f["key"])
        decompresser(chemin, TANZANIE)
    json.dump(infos["metadata"], open(SORTIE / "resultats_maladies" / "zenodo_tanzanie_metadata.json", "w"), indent=2)
""")
code("""
BANGLADESH = DONNEES / "bangladesh"

if not MODE_TEST and not any(BANGLADESH.rglob("*.jpg")):
    from datasets import load_dataset
    ds = load_dataset("Project-AgML/BananaLSD_leaf_disease_classification", "raw", split="train")
    noms = ds.features["label"].names
    print("Classes BananaLSD :", noms, "| images :", len(ds))
    for i, ex in enumerate(ds):
        dossier = BANGLADESH / noms[ex["label"]]
        dossier.mkdir(parents=True, exist_ok=True)
        ex["image"].convert("RGB").save(dossier / f"bd_{i:04d}.jpg", quality=95)
""")

md("""
## 3. Inventaire : quelles images, quelles classes, quel district ?

Les classes sont déduites des **noms de dossiers** (mots-clés). Le district est cherché dans le chemin de
chaque image. S'il est trouvé, la validation se fait **par district entier** ; sinon, par **groupes de
quasi-doublons** (étape 4).
""")
code("""
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DISTRICTS = ["rungwe", "mbeya", "arumeru", "arusha"]

def classe_depuis_chemin(chemin, origine):
    texte = str(chemin).lower()
    if "health" in texte or "sain" in texte:
        return "sain"
    if "sigatoka" in texte:
        return "sigatoka"
    if "fusarium" in texte or "wilt" in texte:
        return "fusarium"
    if origine == "bangladesh" and ("cordana" in texte or "pestalo" in texte):
        return "inconnu"
    return None

def inventaire(racine, origine):
    lignes = []
    for p in racine.rglob("*"):
        if p.suffix.lower() not in EXTENSIONS or "__MACOSX" in p.parts:
            continue
        rel = p.relative_to(racine)
        sous_type = rel.parts[0] if origine == "bangladesh" else None
        district = next((d for d in DISTRICTS if d in str(rel).lower()), None)
        lignes.append({"chemin": str(p), "origine": origine, "classe": classe_depuis_chemin(rel, origine),
                       "district": district, "sous_type": sous_type})
    return pd.DataFrame(lignes)

if MODE_TEST:
    # Fausses données : couleurs différentes par classe, juste pour tester la mécanique
    def fausses_images(racine, classes, n, prefixe):
        for c in classes:
            (racine / c).mkdir(parents=True, exist_ok=True)
            for i in range(n):
                teinte = {"healthy": (40, 160, 40), "sigatoka": (90, 90, 30), "fusarium": (170, 150, 40),
                          "cordana": (120, 70, 60), "pestalotiopsis": (140, 110, 90)}[c]
                bruit = np.random.randint(-40, 40, (300, 300, 3))
                img = np.clip(np.array(teinte) + bruit, 0, 255).astype("uint8")
                Image.fromarray(img).save(racine / c / f"{prefixe}_{i}.jpg")
    fausses_images(TANZANIE / "Rungwe", ["healthy", "sigatoka", "fusarium"], 20, "ru")
    fausses_images(TANZANIE / "Arusha", ["healthy", "sigatoka", "fusarium"], 20, "ar")
    fausses_images(BANGLADESH, ["healthy", "sigatoka", "cordana", "pestalotiopsis"], 10, "bd")

tz = inventaire(TANZANIE, "tanzanie")
bd = inventaire(BANGLADESH, "bangladesh")

print("Tanzanie : exemples de dossiers trouvés")
print(pd.Series([str(Path(c).parent.relative_to(TANZANIE)) for c in tz.chemin]).value_counts().head(15))
print("\\nTanzanie par classe :\\n", tz.classe.value_counts(dropna=False))
print("\\nTanzanie par district :\\n", tz.district.value_counts(dropna=False))
print("\\nBangladesh :\\n", bd.groupby(["sous_type", "classe"]).size())

non_classees = tz[tz.classe.isna()]
if len(non_classees):
    print(f"\\n⚠️ {len(non_classees)} images Tanzanie sans classe reconnue (ignorées). Exemples :")
    print(non_classees.chemin.head(5).tolist())
tz = tz[tz.classe.notna()].reset_index(drop=True)
assert tz.classe.nunique() == 3, "Il faut les 3 classes sain/sigatoka/fusarium : regarde les noms de dossiers ci-dessus"
assert set(bd.classe) >= {"sain", "sigatoka", "inconnu"}, "Classes BananaLSD inattendues"
""")

md("""
## 4. Mise en cache, quasi-doublons et contrôle de fuite

Chaque image est réduite une fois en 256×256 (l'entraînement va ~10× plus vite).
On calcule une **empreinte perceptuelle** (pHash) : deux photos presque identiques de la même feuille ont des
empreintes proches. Elles sont mises dans le **même groupe**, et un groupe ne peut jamais être coupé entre
entraînement et validation.

On vérifie aussi qu'**aucune image du Bangladesh ne ressemble à une image de Tanzanie**.
""")
code("""
DISTANCE_DOUBLON = 6   # bits de différence max entre deux pHash (sur 64) pour parler de quasi-doublon

def mettre_en_cache(df):
    caches, empreintes = [], []
    for chemin in df.chemin:
        nom = hashlib.md5(chemin.encode()).hexdigest() + ".jpg"
        cible = CACHE / nom
        if not cible.exists():
            Image.open(chemin).convert("RGB").resize((256, 256), Image.BILINEAR).save(cible, quality=92)
        img = Image.open(cible)
        caches.append(str(cible))
        empreintes.append(imagehash.phash(img))
    df = df.copy()
    df["cache"] = caches
    df["phash"] = empreintes
    return df

t0 = time.time()
tz = mettre_en_cache(tz)
bd = mettre_en_cache(bd)
print(f"Cache prêt en {time.time() - t0:.0f} s")

def grouper_doublons(empreintes, distance=DISTANCE_DOUBLON):
    \"\"\"Union-find : deux images à moins de `distance` bits finissent dans le même groupe.\"\"\"
    bits = np.array([e.hash.flatten() for e in empreintes], dtype=bool)
    parent = list(range(len(bits)))
    def racine(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(bits)):
        d = (bits[i + 1:] != bits[i]).sum(axis=1)
        for j in np.nonzero(d <= distance)[0] + i + 1:
            parent[racine(j)] = racine(i)
    return np.array([racine(i) for i in range(len(bits))])

tz["groupe"] = grouper_doublons(list(tz.phash))
n_groupes = tz.groupe.nunique()
print(f"Tanzanie : {len(tz)} images -> {n_groupes} groupes ({len(tz) - n_groupes} images sont des quasi-doublons)")

# Contrôle de fuite entre pays
bits_tz = np.array([e.hash.flatten() for e in tz.phash], dtype=bool)
bits_bd = np.array([e.hash.flatten() for e in bd.phash], dtype=bool)
min_dist = np.array([(bits_tz != b).sum(axis=1).min() for b in bits_bd])
n_suspects = int((min_dist <= DISTANCE_DOUBLON).sum())
print(f"Images Bangladesh quasi-identiques à une image Tanzanie : {n_suspects}")
bd = bd[min_dist > DISTANCE_DOUBLON].reset_index(drop=True)   # retirées par prudence
""")

md("""
## 5. Découpage entraînement / validation (Tanzanie uniquement)

- Si les districts sont connus : **un district entier** sert de validation.
- Sinon : 15 % des **groupes** en validation, jamais d'images isolées.

La liste complète (fichier → split) est sauvegardée : c'est la preuve d'absence de fuite pour le carnet.
""")
code("""
from sklearn.model_selection import GroupShuffleSplit

if tz.district.notna().all() and tz.district.nunique() >= 2:
    # On prend comme validation le district le plus petit qui contient les 3 classes
    candidats = [d for d in tz.district.value_counts(ascending=True).index
                 if tz[tz.district == d].classe.nunique() == 3]
    district_val = candidats[0]
    tz["split"] = np.where(tz.district == district_val, "val", "train")
    STRATEGIE = f"par district (validation = {district_val})"
else:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=GRAINE)
    idx_train, idx_val = next(gss.split(tz, groups=tz.groupe))
    tz["split"] = "train"
    tz.loc[idx_val, "split"] = "val"
    STRATEGIE = "par groupes de quasi-doublons (districts absents des métadonnées)"
bd["split"] = "test"

assert not set(tz[tz.split == "train"].groupe) & set(tz[tz.split == "val"].groupe), "Fuite train/val !"
print("Stratégie :", STRATEGIE)
print(tz.groupby(["split", "classe"]).size().unstack())

colonnes = ["chemin", "origine", "classe", "district", "sous_type", "split"]
pd.concat([tz.assign(groupe=tz.groupe)[colonnes + ["groupe"]], bd[colonnes]]).to_csv(
    SORTIE / "resultats_maladies" / "decoupage.csv", index=False)
""")

md("""
## 6. Modèle et entraînement

**MobileNetV3** (torchvision), pré-entraîné sur ImageNet : ~5 Mo pour la version *small*, ~16 Mo pour la
*large*, rapide sur CPU. Augmentations choisies pour le terrain : cadrages variés, retournements,
**variations de lumière** (ColorJitter). Perte pondérée car les classes sont déséquilibrées.
""")
code("""
CLASSES = ["sain", "sigatoka", "fusarium"]
NORMALISATION = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

AUG = transforms.Compose([
    transforms.RandomResizedCrop(TAILLE_IMAGE, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.03),
    transforms.ToTensor(), NORMALISATION,
])
SANS_AUG = transforms.Compose([transforms.Resize((TAILLE_IMAGE, TAILLE_IMAGE)), transforms.ToTensor(), NORMALISATION])

class Feuilles(torch.utils.data.Dataset):
    def __init__(self, df, transformation):
        self.chemins = list(df.cache)
        self.labels = [CLASSES.index(c) if c in CLASSES else -1 for c in df.classe]
        self.t = transformation
    def __len__(self):
        return len(self.chemins)
    def __getitem__(self, i):
        return self.t(Image.open(self.chemins[i]).convert("RGB")), self.labels[i]

def chargeur(df, transformation, melanger=False):
    return torch.utils.data.DataLoader(Feuilles(df, transformation), batch_size=64, shuffle=melanger,
                                       num_workers=2 if DEVICE == "cuda" else 0, pin_memory=DEVICE == "cuda")

def creer_modele(architecture="mobilenet_v3_large", pre_entraine=True):
    poids = "IMAGENET1K_V1" if pre_entraine else None
    if MODE_TEST:
        poids = None   # pas de téléchargement en test local
    modele = getattr(torchvision.models, architecture)(weights=poids)
    modele.classifier[-1] = nn.Linear(modele.classifier[-1].in_features, len(CLASSES))
    # torchvision règle la BatchNorm de MobileNetV3 à momentum=0.01 : ses statistiques suivent
    # trop lentement nos images (surtout sans pré-entraînement). 0.1 = valeur standard PyTorch.
    for m in modele.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.1
    return modele.to(DEVICE)

@torch.no_grad()
def predire_logits(modele, df):
    modele.eval()
    sorties = [modele(x.to(DEVICE)).float().cpu() for x, _ in chargeur(df, SANS_AUG)]
    return torch.cat(sorties)

def entrainer(nom, architecture="mobilenet_v3_large", pre_entraine=True, augmentation=True, epoques=EPOQUES):
    modele = creer_modele(architecture, pre_entraine)
    train, val = tz[tz.split == "train"], tz[tz.split == "val"]
    effectifs = train.classe.value_counts().reindex(CLASSES).values
    poids = torch.tensor(effectifs.sum() / (len(CLASSES) * effectifs), dtype=torch.float32, device=DEVICE)
    critere = nn.CrossEntropyLoss(weight=poids, label_smoothing=0.05)
    optim = torch.optim.AdamW(modele.parameters(), lr=1e-3 if not pre_entraine else 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=optim.param_groups[0]["lr"],
                                                total_steps=epoques * len(chargeur(train, AUG)))
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    meilleur, meilleur_etat = -1, None
    for ep in range(epoques):
        modele.train()
        for x, y in chargeur(train, AUG if augmentation else SANS_AUG, melanger=True):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(device_type="cuda", enabled=DEVICE == "cuda"):
                perte = critere(modele(x), y)
            optim.zero_grad()
            scaler.scale(perte).backward()
            scaler.step(optim); scaler.update(); sched.step()
        logits = predire_logits(modele, val)
        y_val = torch.tensor([CLASSES.index(c) for c in val.classe])
        # Exactitude équilibrée : chaque classe compte pareil, même la moins nombreuse
        rappels = [(logits.argmax(1)[y_val == k] == k).float().mean().item() for k in range(len(CLASSES))]
        score = float(np.mean(rappels))
        print(f"[{nom}] époque {ep + 1}/{epoques} | perte {perte.item():.3f} | val équilibrée {score:.3f}")
        if score > meilleur:
            meilleur, meilleur_etat = score, {k: v.detach().cpu().clone() for k, v in modele.state_dict().items()}
    modele.load_state_dict(meilleur_etat)
    return modele
""")

md("""
## 7. Calibration et seuil de rejet (sur la validation Tanzanie uniquement)

1. **Température** : on « adoucit » les probabilités pour qu'une confiance de 80 % soit juste ~80 % du temps.
2. **Seuil** : on garde 95 % des bonnes réponses de validation ; en dessous du seuil, le modèle répond
   « je ne sais pas ».

Le Bangladesh n'est **jamais** utilisé pour régler quoi que ce soit : il ne sert qu'à mesurer.
""")
code("""
def calibrer(modele):
    val = tz[tz.split == "val"]
    logits = predire_logits(modele, val)
    y = torch.tensor([CLASSES.index(c) for c in val.classe])
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    def etape():
        opt.zero_grad()
        perte = nn.functional.cross_entropy(logits / log_t.exp(), y)
        perte.backward()
        return perte
    opt.step(etape)
    temperature = log_t.exp().item()
    proba = torch.softmax(logits / temperature, 1)
    confiance, pred = proba.max(1)
    bonnes = confiance[pred == y].numpy()
    seuil = float(np.quantile(bonnes, 1 - TAUX_ACCEPTATION_VAL))
    return temperature, seuil
""")

md("""
## 8. Évaluation sur le Bangladesh (pays jamais vu)

Métriques demandées par l'orga :
- **précision / rappel** sur la Sigatoka ;
- **taux de fausses alertes** sur feuilles saines (une feuille saine classée malade et acceptée) ;
- **rejet de l'inconnu** : part de cordana / pestalotiopsis pour lesquelles le modèle dit « je ne sais pas » ;
- **AUROC connu/inconnu** : capacité à séparer les deux, indépendamment du seuil.
""")
code("""
from sklearn.metrics import roc_auc_score, confusion_matrix

def evaluer(modele, temperature, seuil, nom):
    logits = predire_logits(modele, bd)
    proba = torch.softmax(logits / temperature, 1)
    confiance, idx = proba.max(1)
    confiance, idx = confiance.numpy(), idx.numpy()
    pred = np.where(confiance >= seuil, np.array(CLASSES)[idx], "je_ne_sais_pas")
    vrai = bd.classe.values

    connu = vrai != "inconnu"
    malade_vrai = vrai == "sigatoka"
    alerte = np.isin(pred, ["sigatoka", "fusarium"])          # toute alerte maladie acceptée
    vp = int((alerte & malade_vrai).sum())
    precision = vp / max(int((alerte & connu).sum()), 1)
    rappel = vp / max(int(malade_vrai.sum()), 1)
    fausses_alertes = float(alerte[vrai == "sain"].mean())
    rejet_inconnu = float((pred[vrai == "inconnu"] == "je_ne_sais_pas").mean())
    rejet_connu = float((pred[connu] == "je_ne_sais_pas").mean())
    auroc = float(roc_auc_score(connu.astype(int), confiance))
    acc_acceptees = float((pred[connu & (pred != "je_ne_sais_pas")] == vrai[connu & (pred != "je_ne_sais_pas")]).mean())

    resultat = dict(modele=nom, precision_sigatoka=precision, rappel_sigatoka=rappel,
                    fausses_alertes_sain=fausses_alertes, rejet_inconnu=rejet_inconnu,
                    rejet_a_tort_connu=rejet_connu, auroc_connu_inconnu=auroc,
                    exactitude_sur_acceptees=acc_acceptees, temperature=temperature, seuil=seuil)
    etiquettes = ["sain", "sigatoka", "inconnu"]
    colonnes = CLASSES + ["je_ne_sais_pas"]
    matrice = pd.crosstab(pd.Series(vrai, name="vérité"), pd.Series(pred, name="prédiction")).reindex(
        index=etiquettes, columns=colonnes, fill_value=0)
    return resultat, matrice, pd.DataFrame({"chemin": bd.chemin, "vrai": vrai, "pred": pred, "confiance": confiance})
""")

md("## 9. Expériences : modèle principal + ablations")
code("""
EXPERIENCES = [dict(nom="large_preentraine_aug", architecture="mobilenet_v3_large", pre_entraine=True, augmentation=True)]
if FAIRE_ABLATIONS:
    EXPERIENCES += [
        dict(nom="large_sans_preentrainement", architecture="mobilenet_v3_large", pre_entraine=False, augmentation=True),
        dict(nom="large_sans_augmentation", architecture="mobilenet_v3_large", pre_entraine=True, augmentation=False),
        dict(nom="small_preentraine_aug", architecture="mobilenet_v3_small", pre_entraine=True, augmentation=True),
    ]

resultats, modeles = [], {}
for exp in EXPERIENCES:
    t0 = time.time()
    modele = entrainer(exp["nom"], exp["architecture"], exp["pre_entraine"], exp["augmentation"],
                       epoques=1 if MODE_TEST else EPOQUES)
    temperature, seuil = calibrer(modele)
    res, matrice, detail = evaluer(modele, temperature, seuil, exp["nom"])
    res["minutes_entrainement"] = round((time.time() - t0) / 60, 1)
    res["parametres_millions"] = round(sum(p.numel() for p in modele.parameters()) / 1e6, 2)
    resultats.append(res)
    modeles[exp["nom"]] = (modele, temperature, seuil, exp)
    print(matrice, "\\n")
    detail.to_csv(SORTIE / "resultats_maladies" / f"predictions_{exp['nom']}.csv", index=False)
    matrice.to_csv(SORTIE / "resultats_maladies" / f"matrice_{exp['nom']}.csv")

tableau = pd.DataFrame(resultats).set_index("modele").round(3)
tableau.to_csv(SORTIE / "resultats_maladies" / "ablations.csv")
tableau
""")

md("""
## 10. Export ONNX (inférence CPU sobre) et mesure de vitesse

Le modèle principal est exporté en ONNX avec ses réglages (classes, température, seuil) dans un JSON.
C'est ce que `bananes/maladies.py` charge dans la chaîne, **sans PyTorch ni GPU**.
""")
code("""
import onnxruntime as ort

def exporter(nom):
    modele, temperature, seuil, exp = modeles[nom]
    modele = modele.cpu().eval()
    chemin_onnx = SORTIE / "models" / f"maladies_{nom}.onnx"
    exemple = torch.randn(1, 3, TAILLE_IMAGE, TAILLE_IMAGE)
    torch.onnx.export(modele, exemple, chemin_onnx, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "lot"}, "logits": {0: "lot"}}, opset_version=17, dynamo=False)
    # Vérification : ONNX et PyTorch donnent la même chose
    session = ort.InferenceSession(str(chemin_onnx), providers=["CPUExecutionProvider"])
    ecart = np.abs(session.run(None, {"image": exemple.numpy()})[0] - modele(exemple).detach().numpy()).max()
    # Vitesse CPU, image par image
    for _ in range(5):
        session.run(None, {"image": exemple.numpy()})
    t0 = time.time()
    for _ in range(50):
        session.run(None, {"image": exemple.numpy()})
    ms = (time.time() - t0) / 50 * 1000
    config = dict(classes=CLASSES, temperature=temperature, seuil_rejet=seuil, taille_image=TAILLE_IMAGE,
                  moyenne=[0.485, 0.456, 0.406], ecart_type=[0.229, 0.224, 0.225],
                  entraine_sur="Tanzanie (Zenodo 7670326)", decoupage=STRATEGIE, experience=exp)
    json.dump(config, open(chemin_onnx.with_suffix(".json"), "w"), indent=2, ensure_ascii=False)
    taille = chemin_onnx.stat().st_size / 1e6
    print(f"{nom}: {taille:.1f} Mo | {ms:.1f} ms/image sur CPU | écart ONNX/PyTorch {ecart:.1e}")
    return taille, ms

for nom in modeles:
    if nom in ("large_preentraine_aug", "small_preentraine_aug"):
        taille, ms = exporter(nom)
        tableau.loc[nom, "taille_onnx_mo"] = round(taille, 1)
        tableau.loc[nom, "ms_par_image_cpu"] = round(ms, 1)
tableau.to_csv(SORTIE / "resultats_maladies" / "ablations.csv")
tableau
""")

md("""
## 11. Pour le rapport : les erreurs les plus instructives

Les feuilles saines prises pour malades, les Sigatoka ratées, et les inconnues acceptées avec confiance.
Regarde-les : ce sont tes **cas d'échec** à analyser.
""")
code("""
import matplotlib.pyplot as plt

detail = pd.read_csv(SORTIE / "resultats_maladies" / "predictions_large_preentraine_aug.csv")
cas = {
    "Saines -> alerte": detail[(detail.vrai == "sain") & detail.pred.isin(["sigatoka", "fusarium"])],
    "Sigatoka ratées": detail[(detail.vrai == "sigatoka") & ~detail.pred.isin(["sigatoka", "fusarium"])],
    "Inconnues acceptées": detail[(detail.vrai == "inconnu") & (detail.pred != "je_ne_sais_pas")],
}
fig, axes = plt.subplots(3, 6, figsize=(18, 9))
for ligne, (titre, df) in enumerate(cas.items()):
    exemples = df.sort_values("confiance", ascending=False).head(6)
    for col in range(6):
        ax = axes[ligne, col]; ax.axis("off")
        if col < len(exemples):
            r = exemples.iloc[col]
            ax.imshow(Image.open(r.chemin)); ax.set_title(f"{r.pred} ({r.confiance:.2f})", fontsize=9)
    axes[ligne, 0].text(-0.1, 0.5, titre, transform=axes[ligne, 0].transAxes, ha="right", va="center", fontsize=11)
plt.tight_layout()
plt.savefig(SORTIE / "resultats_maladies" / "cas_echec.png", dpi=110)
print({k: len(v) for k, v in cas.items()})
""")

md("""
## Ce qu'il faut récupérer

Dans `MyDrive/piscine-bananeraie/` :
- `models/maladies_large_preentraine_aug.onnx` + `.json` → à copier dans le dossier `models/` du repo ;
- `resultats_maladies/ablations.csv` → tableau du rapport ;
- `resultats_maladies/decoupage.csv` → preuve du découpage pour le carnet de données ;
- `resultats_maladies/cas_echec.png` → analyse des échecs.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = CELLULES
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}
cible = Path(__file__).with_name("maladies_colab.ipynb")
nbf.write(nb, cible)
print("Notebook écrit :", cible)
