# Carnet de données

Pour chaque source : provenance, licence, résolution, date, composition, usage
(train / val / test), risques de fuite, biais et limites.
**Règle de découpage : jamais aléatoire entre images d'une même zone — par site ou par pays.**

## Vue d'ensemble

| # | Source | Type | Pays | Licence | Usage prévu | Statut |
|---|---|---|---|---|---|---|
| L1 | Mduma & Elinisa 2025 — Banana Leaves Imagery Dataset (Zenodo 10.5281/zenodo.7670326) | Feuilles, smartphone | Tanzanie | à vérifier sur Zenodo | Maladies : **train + val** | à télécharger |
| L2 | Mduma et al. 2023 — Banana leaves & stem (Data in Brief, PMC10333424) | Feuilles + tiges, smartphone | Tanzanie | CC BY (article) — données à vérifier | **Même source que L1** (voir fuites) | à vérifier |
| L3 | BananaLSD — Arman et al. 2023 (Mendeley 10.17632/9tb7k297ff.1) | Feuilles, 3 smartphones | Bangladesh | CC BY 4.0 | Maladies : **test** (autre pays) + classes inconnues | à télécharger |
| D1 | DS-v1 / DS-v2 — UAV Black Sigatoka (Zenodo 10.5281/zenodo.20945958) | Tuiles drone RGB + boîtes YOLO | à vérifier | CC BY 4.0 | Drone : à évaluer (contenu des boîtes ?) | à vérifier |
| M1 | Courageous Land — agroforestry-inventory, `banana.pth` (Hugging Face) | Modèle pré-entraîné | Brésil | Poids CC BY 4.0, code GPL-3.0 | Comptage : détecteur pré-entraîné | à tester |
| S1 | Sentinel-2 L2A (Copernicus Data Space) | Satellite, 10–20 m | à choisir | Licence Copernicus (libre) | Vigueur NDVI, stress NDMI | à télécharger |
| T1 | Plantation synthétique (`scripts/creer_donnees_test.py`) | Simulation | — | — | Tests de la chaîne uniquement | ✅ |

Exclu : données figshare de Neupane et al. 2019 (Thaïlande) — **embargo permanent**,
non téléchargeables. L'article reste une référence de méthode.

---

## Fiches détaillées

### L1 — Banana Leaves Imagery Dataset (Tanzanie)
- **Provenance** : Nelson Mandela African Institution of Science and Technology. Districts Rungwe, Mbeya, Arumeru, Arusha.
- **Date d'acquisition** : novembre 2022 – janvier 2023.
- **Capteur / résolution** : smartphone haute résolution, prise au sol, de près.
- **Composition** : 11 767 images — saines 3 339, Black Sigatoka 3 496, Fusarium Wilt Race 1 4 932.
- **Licence** : _à recopier depuis la page Zenodo_.
- **Découpage** : train/val **par district** si le district figure dans les chemins des fichiers ;
  sinon par **groupes de quasi-doublons** (pHash, ≤ 6 bits d'écart), un groupe n'étant jamais coupé.
  Stratégie réellement appliquée et liste fichier → split : `resultats_maladies/decoupage.csv` (notebook).
- **Contrôle de fuite entre pays** : toute image du Bangladesh à ≤ 6 bits (pHash) d'une image de Tanzanie est retirée du test.
- **Fuites** : recoupement très probable avec L2 (même autrice, même région). Doublons
  quasi identiques possibles → dédoublonnage par hash perceptuel avant découpage.
- **Biais / limites** : fonds, cadrage et éclairage propres à une campagne ; Fusarium
  surreprésenté ; pas de distinction de stade.

### L2 — Banana leaves and stem (Tanzanie, 2023)
- Même équipe et même région que L1. **Traité comme la même source** : jamais L1 en train et L2 en test.
- Usage : éventuellement complément d'entraînement après dédoublonnage, sinon non utilisé.

### L3 — BananaLSD (Bangladesh)
- **Provenance** : ferme de la Bangabandhu Sheikh Mujibur Rahman Agricultural University + champs voisins, Gazipur.
- **Date** : juin 2021. **Capteurs** : Huawei ELE-L29, Xiaomi Redmi 8A, Xiaomi Redmi Note 9.
- **Composition (originales)** : Sigatoka 473, Cordana 162, Pestalotiopsis 173, saines 129 = 937.
- **Téléchargement** : copie du sous-ensemble *raw* publiée sur Hugging Face par Project AgML
  (`Project-AgML/BananaLSD_leaf_disease_classification`, config `raw`, CC BY 4.0), images déjà en 224×224.
- **Usage** : **test uniquement**. Sain + Sigatoka = classes connues ; Cordana + Pestalotiopsis = **inconnues** (test du rejet).
- **Fuites** : le dossier « augmented » est dérivé des originales → **exclu**.
- **Limites** : Sigatoka noire et jaune non distinguées ; images redimensionnées 224×224 dans le jeu traité.

### M1 — Modèle Courageous Land `banana.pth`
- Entraîné sur la ferme Raízes (São Paulo), orthophotos RTK ~1,66 cm/pixel, 133 étiquettes.
- Performances publiées sur leur site : précision 0,59, rappel 0,93 (appariement à 1 m).
- Défaut connu : compte les rejets d'une même touffe comme des pieds distincts → surcomptage.
- **Usage documenté** : inférence seule, puis fusion des détections < 1,4 m ; évalué sur nos propres annotations.

### S1 — Sentinel-2
- Bandes utilisées : B04 (rouge), B08 (PIR), B05 (red-edge) et B11 (SWIR) pour NDMI.
- Zone, dates (saison sèche / saison des pluies) et couverture nuageuse : _à remplir_.

---

## Écart sol / drone (point de vigilance du sujet)
Les images de feuilles sont prises au sol, de près (quelques cm). Un drone à 40 m
voit ~1–2 cm/pixel : les lésions de quelques millimètres sont invisibles ou réduites
à quelques pixels. Le classifieur de feuilles **ne peut pas être appliqué tel quel**
aux orthomosaïques. _À compléter : conséquences et protocole terrain proposé._
