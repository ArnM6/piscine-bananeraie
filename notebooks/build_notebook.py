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
# Module maladies — feuilles de bananier (v2)

**Entraînement sur la Tanzanie, test sur le Bangladesh**, avec rejet de l'inconnu.

| Rôle | Jeu | Classes |
|---|---|---|
| Entraînement + validation | Tanzanie (Mduma & Elinisa, Zenodo 7670326) | sain, sigatoka, fusarium |
| Test (autre pays) | BananaLSD, Bangladesh (937 images *originales* seulement) | sain, sigatoka = **connues** ; cordana, pestalotiopsis = **inconnues** |

Le modèle doit :
1. reconnaître la cercosporiose (Sigatoka) ;
2. ne pas alerter sur les feuilles saines ;
3. dire **« je ne sais pas »** face à cordana et pestalotiopsis, qu'il n'a jamais vues.

**Avant de lancer** : *Exécution → Modifier le type d'exécution → GPU T4*, puis *Exécution → Tout exécuter*.

**Reprise automatique** : tout est sauvegardé au fur et à mesure dans `MyDrive/piscine-bananeraie/`.
Si Colab coupe, relance *Tout exécuter* : les archives déjà préparées et les expériences déjà terminées
sont sautées.

Nouveautés par rapport à la v1 (premier lancement, voir `docs/carnet_de_donnees.md`) :
- les photos sont réduites en 256×256 **une seule fois** et gardées sur le Drive (≈ 0,3 Go au lieu de 41 Go) ;
- la validation se fait sur des **sessions de prise de vue** entières (date EXIF si disponible, sinon blocs de
  photos consécutives), et non plus image par image ;
- trois façons de dire « je ne sais pas » sont comparées : confiance maximale (v1), énergie, et
  **distance aux exemples connus (kNN)**, retenue a priori pour la chaîne.
""")

md("## 0. Réglages")
code("""
# Mets False pour aller vite (un seul entraînement). True = ablations pour le rapport (~4 entraînements).
FAIRE_ABLATIONS = True

EPOQUES = 8
TAILLE_IMAGE = 224
GRAINE = 42
PART_VALIDATION = 0.2          # part des sessions de Tanzanie gardées pour la validation
TAUX_ACCEPTATION_VAL = 0.95    # le seuil de rejet garde 95 % des bonnes réponses de validation
K_VOISINS = 10                 # kNN : distance au 10e exemple d'entraînement le plus proche
TAILLE_BANQUE = 3000           # nombre d'exemples d'entraînement gardés pour le kNN (sobriété)

# Mode test : petites fausses données, pour vérifier que le notebook tourne (ne pas toucher)
MODE_TEST = False
""")

md("## 1. Installation et Google Drive")
code("""
import os, sys, json, time, random, shutil, zipfile, tarfile, hashlib, subprocess, re
from pathlib import Path

if not MODE_TEST:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "imagehash",
                    "datasets", "onnxscript"], check=True)

import numpy as np
import pandas as pd
import requests
import torch, torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image, ImageFile
import imagehash

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
random.seed(GRAINE); np.random.seed(GRAINE); torch.manual_seed(GRAINE)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Calcul sur :", DEVICE)

TRAVAIL = Path("/content") if Path("/content").exists() and not MODE_TEST else Path.cwd() / "colab_travail"
TEMP = TRAVAIL / "temp"             # archives téléchargées, effacées au fur et à mesure
CACHE = TRAVAIL / "cache_256"       # photos réduites, utilisées pour l'entraînement

if Path("/content").exists() and not MODE_TEST:
    from google.colab import drive
    drive.mount("/content/drive")
    SORTIE = Path("/content/drive/MyDrive/piscine-bananeraie")
else:
    SORTIE = TRAVAIL / "sortie"
DRIVE_DONNEES = SORTIE / "donnees_256"     # une archive .tar par archive Zenodo, déjà réduite
MODELES = SORTIE / "models"
RES = SORTIE / "resultats_maladies"
for d in [TEMP, CACHE, DRIVE_DONNEES, MODELES, RES]:
    d.mkdir(parents=True, exist_ok=True)
print("Sauvegardes dans :", SORTIE)
""")

md("""
## 2. Données de Tanzanie : téléchargées et réduites **une seule fois**

Pour chaque archive Zenodo (HEALTHY-1, BLACK SIGATOKA-2…) :
1. si sa version réduite existe déjà sur le Drive → on saute ;
2. sinon : téléchargement, décompression, lecture des **métadonnées EXIF** (date et GPS de la photo, utiles
   pour le découpage), réduction en 256×256, sauvegarde en `.tar` sur le Drive, puis effacement des gros fichiers.

Les dossiers `__MACOSX` (fichiers cachés créés par macOS lors de la compression) sont ignorés.
""")
code("""
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def telecharger(url, destination):
    \"\"\"Télécharge en affichant la progression tous les 500 Mo.\"\"\"
    if url.startswith("/"):   # mode test : fichier local
        shutil.copy(url, destination)
        return destination
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total, palier = 0, 500e6
        with open(destination, "wb") as f:
            for morceau in r.iter_content(1 << 22):
                f.write(morceau)
                total += len(morceau)
                if total >= palier:
                    print(f"     {total / 1e9:.1f} Go", flush=True)
                    palier += 500e6
    return destination

def lire_exif(img):
    \"\"\"Date de prise de vue et position GPS, si l'appareil les a enregistrées.\"\"\"
    date = lat = lon = None
    try:
        exif = img.getexif()
        date = exif.get_ifd(0x8769).get(36867) or exif.get(306)   # DateTimeOriginal, sinon DateTime
        gps = exif.get_ifd(0x8825)
        if gps and 2 in gps and 4 in gps:
            en_degres = lambda dms: float(dms[0]) + float(dms[1]) / 60 + float(dms[2]) / 3600
            lat = en_degres(gps[2]) * (-1 if gps.get(1) == "S" else 1)
            lon = en_degres(gps[4]) * (-1 if gps.get(3) == "W" else 1)
    except Exception:
        pass
    return date, lat, lon

def classe_depuis_nom(nom):
    nom = nom.upper()
    if "HEALTH" in nom:
        return "sain"
    if "SIGATOKA" in nom:
        return "sigatoka"
    if "FUSARIUM" in nom or "WILT" in nom:
        return "fusarium"
    return None

def preparer_archive(nom_fichier, url):
    archive = Path(nom_fichier).stem
    tar_drive = DRIVE_DONNEES / f"{archive}.tar"
    if tar_drive.exists():
        print(f"  ✓ {archive} : déjà préparée sur le Drive")
        return
    print(f"  ↓ {archive} : téléchargement…", flush=True)
    t0 = time.time()
    zip_local = telecharger(url, TEMP / nom_fichier)
    extrait = TEMP / "extrait"
    shutil.rmtree(extrait, ignore_errors=True)
    with zipfile.ZipFile(zip_local) as z:
        z.extractall(extrait)
    zip_local.unlink()

    dossier = CACHE / "tanzanie" / archive
    dossier.mkdir(parents=True, exist_ok=True)
    lignes = []
    for p in sorted(extrait.rglob("*")):
        if p.suffix.lower() not in EXTENSIONS or "__MACOSX" in p.parts or p.name.startswith("._"):
            continue
        try:
            img = Image.open(p)
            date, lat, lon = lire_exif(img)
            largeur, hauteur = img.size
            img.convert("RGB").resize((256, 256), Image.BILINEAR).save(dossier / (p.stem + ".jpg"), quality=92)
        except Exception as e:
            print("     image illisible ignorée :", p.name, e)
            continue
        numero = re.findall(r"\\d+", p.stem)
        lignes.append({"fichier": p.stem + ".jpg", "archive": archive, "classe": classe_depuis_nom(archive),
                       "numero": int(numero[-1]) if numero else -1, "date_exif": date,
                       "lat": lat, "lon": lon, "largeur_origine": largeur, "hauteur_origine": hauteur})
    pd.DataFrame(lignes).to_csv(dossier / "metadonnees.csv", index=False)
    shutil.rmtree(extrait)

    tar_local = TEMP / f"{archive}.tar"
    with tarfile.open(tar_local, "w") as t:
        t.add(dossier, arcname=archive)
    shutil.copy(tar_local, tar_drive)
    tar_local.unlink()
    print(f"  ✓ {archive} : {len(lignes)} photos prêtes en {(time.time() - t0) / 60:.0f} min", flush=True)

if MODE_TEST:
    # Fausses archives : chaque classe a une teinte, juste pour tester la mécanique
    faux = TRAVAIL / "faux_zenodo"; faux.mkdir(exist_ok=True)
    teintes = {"HEALTHY": (40, 160, 40), "BLACK SIGATOKA": (90, 90, 30), "FUSARIUM WILT": (170, 150, 40)}
    sources, compteur = [], {}
    for classe, teinte in teintes.items():
        for partie in (1, 2):
            nom = f"{classe}-{partie}.zip"
            with zipfile.ZipFile(faux / nom, "w") as z:
                for i in range(60):
                    n = compteur.get(classe, 0) + 1; compteur[classe] = n
                    img = np.clip(np.array(teinte) + np.random.randint(-40, 40, (300, 300, 3)), 0, 255)
                    tampon = TEMP / "x.jpg"; Image.fromarray(img.astype("uint8")).save(tampon)
                    z.write(tampon, f"{classe}-{partie}/{classe[:2]}_{n * 10}.jpg")
                z.writestr(f"__MACOSX/{classe}-{partie}/._{classe[:2]}_1.jpg", b"x")
            sources.append((nom, str(faux / nom)))
else:
    infos = requests.get("https://zenodo.org/api/records/7670326", timeout=60).json()
    json.dump(infos["metadata"], open(RES / "zenodo_tanzanie_metadata.json", "w"), indent=2)
    print("Titre :", infos["metadata"]["title"], "| licence :", infos["metadata"].get("license"))
    sources = [(f["key"], f["links"]["self"]) for f in infos["files"]]
    for f in infos["files"]:
        print(f"  - {f['key']} ({f['size'] / 1e6:.0f} Mo)")

for nom_fichier, url in sources:
    preparer_archive(nom_fichier, url)

# Copie locale des versions réduites (rapide : quelques centaines de Mo)
for tar_drive in sorted(DRIVE_DONNEES.glob("*.tar")):
    if not (CACHE / "tanzanie" / tar_drive.stem).exists():
        with tarfile.open(tar_drive) as t:
            t.extractall(CACHE / "tanzanie")
print("Tanzanie prête.")
""")

md("""
## 2 bis. Données du Bangladesh (test)

BananaLSD, copie officielle du sous-ensemble *raw* (937 originales) publiée sur Hugging Face par
Project AgML (CC BY 4.0). On **n'utilise jamais** le sous-ensemble *augmented* : il est fabriqué à partir
des originales, ce serait une fuite.
""")
code("""
BANGLADESH = CACHE / "bangladesh"

if not any(BANGLADESH.rglob("*.jpg")):
    if MODE_TEST:
        teintes = {"healthy": (40, 160, 40), "sigatoka": (90, 90, 30), "cordana": (120, 70, 60),
                   "pestalotiopsis": (140, 110, 90)}
        for c, teinte in teintes.items():
            (BANGLADESH / c).mkdir(parents=True, exist_ok=True)
            for i in range(12):
                img = np.clip(np.array(teinte) + np.random.randint(-40, 40, (224, 224, 3)), 0, 255)
                Image.fromarray(img.astype("uint8")).resize((256, 256)).save(BANGLADESH / c / f"bd_{c[:2]}{i}.jpg")
    else:
        from datasets import load_dataset
        ds = load_dataset("Project-AgML/BananaLSD_leaf_disease_classification", "raw", split="train")
        noms = ds.features["label"].names
        print("Classes BananaLSD :", noms, "| images :", len(ds))
        for i, ex in enumerate(ds):
            dossier = BANGLADESH / noms[ex["label"]]
            dossier.mkdir(parents=True, exist_ok=True)
            ex["image"].convert("RGB").resize((256, 256), Image.BILINEAR).save(dossier / f"bd_{i:04d}.jpg", quality=92)
print("Bangladesh prêt.")
""")

md("""
## 3. Inventaire
""")
code("""
tz = pd.concat([pd.read_csv(f) for f in sorted((CACHE / "tanzanie").glob("*/metadonnees.csv"))], ignore_index=True)
tz["cache"] = [str(CACHE / "tanzanie" / a / f) for a, f in zip(tz.archive, tz.fichier)]
tz["origine"] = "tanzanie"
tz = tz[tz.classe.notna()].reset_index(drop=True)

bd_lignes = []
for p in sorted(BANGLADESH.rglob("*.jpg")):
    sous_type = p.parent.name.lower()
    classe = {"healthy": "sain", "sigatoka": "sigatoka"}.get(sous_type, "inconnu")
    bd_lignes.append({"cache": str(p), "origine": "bangladesh", "classe": classe, "sous_type": sous_type})
bd = pd.DataFrame(bd_lignes)

print("Tanzanie par classe :\\n", tz.classe.value_counts().to_string())
print("\\nTanzanie par archive :\\n", tz.archive.value_counts().sort_index().to_string())
print(f"\\nPhotos avec date EXIF : {tz.date_exif.notna().mean():.0%} | avec GPS : {tz.lat.notna().mean():.0%}")
if tz.date_exif.notna().any():
    jours = tz.date_exif.dropna().str[:10]
    print(f"Prises de vue du {jours.min()} au {jours.max()}, sur {jours.nunique()} jours différents")
print("\\nBangladesh :\\n", bd.groupby(["sous_type", "classe"]).size().to_string())
assert tz.classe.nunique() == 3, "Il faut les 3 classes sain/sigatoka/fusarium"
assert set(bd.classe) >= {"sain", "sigatoka", "inconnu"}, "Classes BananaLSD inattendues"
""")

md("""
## 4. Quasi-doublons et contrôle de fuite entre pays

Empreinte perceptuelle (pHash) : deux photos presque identiques ont des empreintes proches.
Toute image du Bangladesh trop proche d'une image de Tanzanie est retirée du test.
""")
code("""
DISTANCE_DOUBLON = 6   # bits de différence max entre deux pHash (sur 64)

tz["phash"] = [imagehash.phash(Image.open(c)) for c in tz.cache]
bd["phash"] = [imagehash.phash(Image.open(c)) for c in bd.cache]

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

tz["groupe_doublons"] = grouper_doublons(list(tz.phash))
print(f"Tanzanie : {len(tz)} images, {len(tz) - tz.groupe_doublons.nunique()} quasi-doublons regroupés")

bits_tz = np.array([e.hash.flatten() for e in tz.phash], dtype=bool)
bits_bd = np.array([e.hash.flatten() for e in bd.phash], dtype=bool)
min_dist = np.array([(bits_tz != b).sum(axis=1).min() for b in bits_bd])
print(f"Images Bangladesh quasi-identiques à une image Tanzanie : {int((min_dist <= DISTANCE_DOUBLON).sum())} (retirées)")
bd = bd[min_dist > DISTANCE_DOUBLON].reset_index(drop=True)
""")

md("""
## 5. Découpage entraînement / validation par **sessions de prise de vue**

Des photos prises le même jour, dans le même champ, se ressemblent (lumière, arrière-plan, plants).
Les couper entre entraînement et validation rend la validation trop facile : c'est ce qui s'est passé en v1,
et le seuil de rejet en est sorti beaucoup trop strict.

- si la plupart des photos ont une **date EXIF** : une session = un jour de prise de vue ;
- sinon : une session = un **bloc de 100 photos consécutives** de la même classe (les fichiers sont numérotés
  dans l'ordre de collecte).

20 % des sessions vont en validation ; un groupe de quasi-doublons n'est jamais coupé.
""")
code("""
from sklearn.model_selection import GroupShuffleSplit

if tz.date_exif.notna().mean() >= 0.8:
    tz["session"] = tz.classe + "_" + tz.date_exif.fillna("inconnue").str[:10]
    STRATEGIE = "par jour de prise de vue (date EXIF)"
else:
    tz["session"] = tz.classe + "_bloc" + (tz.numero // 100).astype(str)
    STRATEGIE = "par blocs de 100 photos consécutives (pas de date EXIF)"
# Un groupe de quasi-doublons prend la session de son premier membre : il n'est jamais coupé
tz["groupe"] = tz.groupby("groupe_doublons")["session"].transform("first")

for essai in range(50):
    gss = GroupShuffleSplit(n_splits=1, test_size=PART_VALIDATION, random_state=GRAINE + essai)
    idx_train, idx_val = next(gss.split(tz, groups=tz.groupe))
    part_val = tz.iloc[idx_val].classe.value_counts() / tz.classe.value_counts()
    if part_val.reindex(tz.classe.unique()).fillna(0).min() >= 0.1:   # chaque classe a ≥ 10 % en validation
        break
tz["split"] = "train"
tz.loc[tz.index[idx_val], "split"] = "val"
bd["split"] = "test"

assert not set(tz[tz.split == "train"].groupe) & set(tz[tz.split == "val"].groupe), "Fuite train/val !"
print("Stratégie :", STRATEGIE, f"| {tz.groupe.nunique()} sessions")
print(tz.groupby(["split", "classe"]).size().unstack())

colonnes = ["cache", "origine", "classe", "split"]
pd.concat([tz[colonnes + ["archive", "fichier", "session", "date_exif", "lat", "lon"]],
           bd[colonnes + ["sous_type"]]]).to_csv(RES / "decoupage.csv", index=False)
""")

md("""
## 6. Modèle et entraînement

**MobileNetV3** (torchvision) pré-entraîné sur ImageNet : ~16 Mo (*large*) ou ~6 Mo (*small*), rapide sur CPU.
Le réseau renvoie aussi son **vecteur de caractéristiques** (avant-dernière couche) : c'est ce qui sert au kNN.
Augmentations choisies pour le terrain : cadrages variés, retournements, **variations de lumière**.
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

class Reseau(nn.Module):
    \"\"\"MobileNetV3 dont on garde le vecteur de caractéristiques en plus des scores de classes.\"\"\"
    def __init__(self, architecture, pre_entraine):
        super().__init__()
        poids = "IMAGENET1K_V1" if pre_entraine and not MODE_TEST else None
        base = getattr(torchvision.models, architecture)(weights=poids)
        self.extracteur = base.features
        self.pool = base.avgpool
        self.tete = base.classifier[:-1]                     # Linear + Hardswish + Dropout
        self.sortie = nn.Linear(base.classifier[-1].in_features, len(CLASSES))
        # torchvision règle la BatchNorm de MobileNetV3 à momentum=0.01 : ses statistiques suivent
        # trop lentement nos images. 0.1 = valeur standard PyTorch (bug trouvé en testant la v1).
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.1
    def forward(self, x):
        caracteristiques = self.tete(torch.flatten(self.pool(self.extracteur(x)), 1))
        return self.sortie(caracteristiques), caracteristiques

@torch.no_grad()
def predire(modele, df):
    \"\"\"Renvoie (logits, caractéristiques) pour toutes les images du DataFrame.\"\"\"
    modele.eval()
    logits, feats = [], []
    for x, _ in chargeur(df, SANS_AUG):
        l, f = modele(x.to(DEVICE))
        logits.append(l.float().cpu()); feats.append(f.float().cpu())
    return torch.cat(logits), torch.cat(feats)

def entrainer(nom, architecture, pre_entraine, augmentation, epoques):
    modele = Reseau(architecture, pre_entraine).to(DEVICE)
    train, val = tz[tz.split == "train"], tz[tz.split == "val"]
    effectifs = train.classe.value_counts().reindex(CLASSES).values
    poids = torch.tensor(effectifs.sum() / (len(CLASSES) * effectifs), dtype=torch.float32, device=DEVICE)
    critere = nn.CrossEntropyLoss(weight=poids, label_smoothing=0.05)
    lr = 3e-4 if pre_entraine else 1e-3
    optim = torch.optim.AdamW(modele.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=lr, total_steps=epoques * len(chargeur(train, AUG)))
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    y_val = torch.tensor([CLASSES.index(c) for c in val.classe])
    meilleur, meilleur_etat = -1, None
    for ep in range(epoques):
        modele.train()
        for x, y in chargeur(train, AUG if augmentation else SANS_AUG, melanger=True):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(device_type="cuda", enabled=DEVICE == "cuda"):
                perte = critere(modele(x)[0], y)
            optim.zero_grad()
            scaler.scale(perte).backward()
            scaler.step(optim); scaler.update(); sched.step()
        pred = predire(modele, val)[0].argmax(1)
        # Exactitude équilibrée : chaque classe compte pareil, même la moins nombreuse
        score = float(np.mean([(pred[y_val == k] == k).float().mean().item() for k in range(len(CLASSES))]))
        print(f"[{nom}] époque {ep + 1}/{epoques} | perte {perte.item():.3f} | val équilibrée {score:.3f}", flush=True)
        if score > meilleur:
            meilleur, meilleur_etat = score, {k: v.detach().cpu().clone() for k, v in modele.state_dict().items()}
    modele.load_state_dict(meilleur_etat)
    return modele, meilleur
""")

md("""
## 7. Trois façons de dire « je ne sais pas » (réglées sur la validation Tanzanie uniquement)

Pour chaque image, un **score de familiarité** (plus il est haut, plus l'image ressemble à ce qu'on connaît) :

| Méthode | Score | Idée |
|---|---|---|
| `msp` (v1) | probabilité maximale | « je suis sûr à 98 % » — mais un réseau peut être sûr de lui à tort |
| `energie` | T · log Σ exp(logits / T) | utilise tous les scores de classes, pas seulement le plus haut |
| `knn` **(retenu)** | − distance au 10e exemple d'entraînement le plus proche | « ai-je déjà vu une feuille qui ressemble à celle-là ? » |

Le kNN est choisi **a priori** (avant de voir le Bangladesh) : il ne dépend pas de la confiance du
classifieur, qui s'est montrée trop élevée en v1. Les deux autres sont gardés comme comparaison.

Le seuil de chaque méthode garde 95 % des bonnes réponses de validation. Le Bangladesh ne sert **jamais** à régler.
""")
code("""
METHODES = ["msp", "energie", "knn"]
METHODE_RETENUE = "knn"

def normaliser(f):
    return f / f.norm(dim=1, keepdim=True).clamp_min(1e-8)

def construire_banque(modele):
    \"\"\"Sous-échantillon équilibré des caractéristiques d'entraînement (gardé pour l'inférence).\"\"\"
    train = tz[tz.split == "train"]
    par_classe = TAILLE_BANQUE // len(CLASSES)
    echantillon = pd.concat([g.sample(min(len(g), par_classe), random_state=GRAINE) for _, g in train.groupby("classe")])
    return normaliser(predire(modele, echantillon)[1])

def scores(logits, feats, temperature, banque):
    t = temperature
    proba = torch.softmax(logits / t, 1)
    distances = 1 - normaliser(feats) @ banque.T                     # distance cosinus
    k = min(K_VOISINS, banque.shape[0])
    dist_k = distances.topk(k, dim=1, largest=False).values[:, -1]
    return {"msp": proba.max(1).values.numpy(),
            "energie": (t * torch.logsumexp(logits / t, 1)).numpy(),
            "knn": (-dist_k).numpy()}, proba.argmax(1).numpy()

def calibrer(modele):
    val = tz[tz.split == "val"]
    logits, feats = predire(modele, val)
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
    banque = construire_banque(modele)
    s, pred = scores(logits, feats, temperature, banque)
    bonnes = pred == y.numpy()
    seuils = {m: float(np.quantile(s[m][bonnes], 1 - TAUX_ACCEPTATION_VAL)) for m in METHODES}
    return temperature, seuils, banque
""")

md("""
## 8. Évaluation sur le Bangladesh (pays jamais vu)

- **précision / rappel** des alertes maladie sur la Sigatoka ;
- **fausses alertes** sur feuilles saines ;
- **rejet de l'inconnu** (cordana, pestalotiopsis → « je ne sais pas ») et **rejet à tort** des feuilles connues ;
- **AUROC connu/inconnu** : capacité à séparer les deux, indépendamment du seuil (0,5 = hasard, 1 = parfait).
""")
code("""
from sklearn.metrics import roc_auc_score

def evaluer(modele, temperature, seuils, banque):
    logits, feats = predire(modele, bd)
    s, idx = scores(logits, feats, temperature, banque)
    vrai = bd.classe.values
    connu = vrai != "inconnu"
    resultats, details = {}, {}
    for m in METHODES:
        pred = np.where(s[m] >= seuils[m], np.array(CLASSES)[idx], "je_ne_sais_pas")
        alerte = np.isin(pred, ["sigatoka", "fusarium"])
        vp = int((alerte & (vrai == "sigatoka")).sum())
        acceptees = connu & (pred != "je_ne_sais_pas")
        resultats[m] = dict(
            precision_alerte=vp / max(int((alerte & connu).sum()), 1),
            rappel_sigatoka=vp / max(int((vrai == "sigatoka").sum()), 1),
            fausses_alertes_sain=float(alerte[vrai == "sain"].mean()),
            rejet_inconnu=float((pred[~connu] == "je_ne_sais_pas").mean()),
            rejet_a_tort_connu=float((pred[connu] == "je_ne_sais_pas").mean()),
            exactitude_sur_acceptees=float((pred[acceptees] == vrai[acceptees]).mean()) if acceptees.any() else 0.0,
            auroc_connu_inconnu=float(roc_auc_score(connu.astype(int), s[m])),
            seuil=seuils[m])
        details[m] = pred
    detail = pd.DataFrame({"chemin": bd.cache, "sous_type": bd.sous_type, "vrai": vrai,
                           "classe_brute": np.array(CLASSES)[idx],
                           **{f"score_{m}": s[m] for m in METHODES}, **{f"pred_{m}": details[m] for m in METHODES}})
    return resultats, detail
""")

md("""
## 9. Expériences (modèle principal + ablations), sauvegardées une par une

Chaque expérience terminée écrit son résultat sur le Drive. Si Colab coupe, elle n'est pas refaite.
Les modèles destinés à la chaîne sont exportés en **ONNX** tout de suite, avec leurs réglages et la banque kNN.
""")
code("""
import onnxruntime as ort

class PourExport(nn.Module):
    def __init__(self, modele):
        super().__init__(); self.modele = modele
    def forward(self, x):
        return self.modele(x)

def exporter(nom, modele, temperature, seuils, banque, exp):
    modele = modele.cpu().eval()
    chemin = MODELES / f"maladies_{nom}.onnx"
    exemple = torch.randn(1, 3, TAILLE_IMAGE, TAILLE_IMAGE)
    torch.onnx.export(PourExport(modele).eval(), exemple, chemin, input_names=["image"],
                      output_names=["logits", "caracteristiques"],
                      dynamic_axes={"image": {0: "lot"}, "logits": {0: "lot"}, "caracteristiques": {0: "lot"}},
                      opset_version=17, dynamo=False)
    modele.eval()   # l'export peut remettre le réseau en mode entraînement : on le repasse en évaluation
    np.save(chemin.with_suffix(".banque.npy"), banque.numpy().astype(np.float16))
    session = ort.InferenceSession(str(chemin), providers=["CPUExecutionProvider"])
    ecart = np.abs(session.run(None, {"image": exemple.numpy()})[0] - modele(exemple)[0].detach().numpy()).max()
    for _ in range(5):
        session.run(None, {"image": exemple.numpy()})
    t0 = time.time()
    for _ in range(50):
        session.run(None, {"image": exemple.numpy()})
    ms = (time.time() - t0) / 50 * 1000
    config = dict(classes=CLASSES, temperature=temperature, methode_rejet=METHODE_RETENUE, seuils=seuils,
                  k_voisins=K_VOISINS, taille_image=TAILLE_IMAGE,
                  moyenne=[0.485, 0.456, 0.406], ecart_type=[0.229, 0.224, 0.225],
                  entraine_sur="Tanzanie (Zenodo 7670326)", decoupage=STRATEGIE, experience=exp)
    json.dump(config, open(chemin.with_suffix(".json"), "w"), indent=2, ensure_ascii=False)
    taille = (chemin.stat().st_size + chemin.with_suffix(".banque.npy").stat().st_size) / 1e6
    print(f"   ONNX {nom}: {taille:.1f} Mo (modèle + banque) | {ms:.1f} ms/image CPU | écart ONNX/PyTorch {ecart:.1e}")
    return taille, ms

EXPERIENCES = [dict(nom="large_preentraine_aug", architecture="mobilenet_v3_large", pre_entraine=True,
                    augmentation=True, exporter=True)]
if FAIRE_ABLATIONS:
    EXPERIENCES += [
        dict(nom="large_sans_preentrainement", architecture="mobilenet_v3_large", pre_entraine=False, augmentation=True, exporter=False),
        dict(nom="large_sans_augmentation", architecture="mobilenet_v3_large", pre_entraine=True, augmentation=False, exporter=False),
        dict(nom="small_preentraine_aug", architecture="mobilenet_v3_small", pre_entraine=True, augmentation=True, exporter=True),
    ]

for exp in EXPERIENCES:
    fichier = RES / f"resultat_{exp['nom']}.json"
    if fichier.exists():
        print(f"✓ {exp['nom']} : déjà fait, on saute")
        continue
    t0 = time.time()
    modele, val_equilibree = entrainer(exp["nom"], exp["architecture"], exp["pre_entraine"], exp["augmentation"],
                                       epoques=2 if MODE_TEST else EPOQUES)
    temperature, seuils, banque = calibrer(modele)
    resultats, detail = evaluer(modele, temperature, seuils, banque)
    detail.to_csv(RES / f"predictions_{exp['nom']}.csv", index=False)
    bilan = dict(experience=exp, val_equilibree=val_equilibree, temperature=temperature,
                 minutes_entrainement=round((time.time() - t0) / 60, 1),
                 parametres_millions=round(sum(p.numel() for p in modele.parameters()) / 1e6, 2),
                 methodes=resultats)
    if exp["exporter"]:
        bilan["taille_mo"], bilan["ms_par_image_cpu"] = exporter(exp["nom"], modele, temperature, seuils, banque, exp)
    json.dump(bilan, open(fichier, "w"), indent=2)
    r = resultats[METHODE_RETENUE]
    print(f"✓ {exp['nom']} terminé | rappel sigatoka {r['rappel_sigatoka']:.2f} | fausses alertes {r['fausses_alertes_sain']:.2f}"
          f" | rejet inconnu {r['rejet_inconnu']:.2f} | rejet à tort {r['rejet_a_tort_connu']:.2f}\\n", flush=True)
    del modele
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
""")

md("""
## 10. Tableaux pour le rapport

1. **Ablations** : chaque expérience, avec la méthode de rejet retenue (kNN).
2. **Méthodes de rejet** : msp / énergie / kNN sur le modèle principal.
""")
code("""
lignes_ablation, lignes_methodes = [], []
for exp in EXPERIENCES:
    fichier = RES / f"resultat_{exp['nom']}.json"
    if not fichier.exists():
        continue
    b = json.load(open(fichier))
    lignes_ablation.append({"modele": exp["nom"], **b["methodes"][METHODE_RETENUE],
                            "auroc_msp": b["methodes"]["msp"]["auroc_connu_inconnu"],
                            "val_equilibree": b["val_equilibree"], "parametres_millions": b["parametres_millions"],
                            "taille_mo": b.get("taille_mo"), "ms_par_image_cpu": b.get("ms_par_image_cpu"),
                            "minutes_entrainement": b["minutes_entrainement"]})
    if exp["nom"] == "large_preentraine_aug":
        for m in METHODES:
            lignes_methodes.append({"methode": m, **b["methodes"][m]})

ablations = pd.DataFrame(lignes_ablation).set_index("modele").round(3)
methodes = pd.DataFrame(lignes_methodes).set_index("methode").round(3)
ablations.to_csv(RES / "ablations.csv")
methodes.to_csv(RES / "methodes_rejet.csv")
print("=== Ablations (rejet kNN) ===")
print(ablations.T.to_string())
print("\\n=== Méthodes de rejet (modèle principal) ===")
print(methodes.T.to_string())
""")

md("""
## 11. Cas d'échec (modèle principal, rejet kNN)

Les feuilles saines prises pour malades, les Sigatoka ratées, et les inconnues acceptées.
Regarde-les : ce sont tes **cas d'échec** à analyser dans le rapport.
""")
code("""
import matplotlib
import matplotlib.pyplot as plt

detail = pd.read_csv(RES / "predictions_large_preentraine_aug.csv")
pred = detail[f"pred_{METHODE_RETENUE}"]
cas = {
    "Saines -> alerte": detail[(detail.vrai == "sain") & pred.isin(["sigatoka", "fusarium"])],
    "Sigatoka ratées": detail[(detail.vrai == "sigatoka") & ~pred.isin(["sigatoka", "fusarium"])],
    "Inconnues acceptées": detail[(detail.vrai == "inconnu") & (pred != "je_ne_sais_pas")],
}
fig, axes = plt.subplots(3, 6, figsize=(18, 9))
for ligne, (titre, df) in enumerate(cas.items()):
    exemples = df.sort_values(f"score_{METHODE_RETENUE}", ascending=False).head(6)
    for col in range(6):
        ax = axes[ligne, col]; ax.axis("off")
        if col < len(exemples):
            r = exemples.iloc[col]
            ax.imshow(Image.open(r.chemin)); ax.set_title(f"{r.sous_type} → {pred[r.name]}", fontsize=9)
    axes[ligne, 0].text(-0.1, 0.5, titre, transform=axes[ligne, 0].transAxes, ha="right", va="center", fontsize=11)
plt.tight_layout()
plt.savefig(RES / "cas_echec.png", dpi=110)
print({k: len(v) for k, v in cas.items()})
print("Matrice (vérité × prédiction) :")
print(pd.crosstab(detail.sous_type, pred))
""")

md("""
## Ce qu'il faut récupérer

Dans `MyDrive/piscine-bananeraie/` (télécharge le dossier entier, **sauf `donnees_256`**) :
- `models/maladies_large_preentraine_aug.onnx` + `.json` + `.banque.npy` → dossier `models/` du repo ;
- `resultats_maladies/ablations.csv` et `methodes_rejet.csv` → tableaux du rapport ;
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
