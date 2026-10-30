# Carnet de données

Pour chaque source : provenance, licence, résolution, date, composition, usage
(train / val / test), risques de fuite, biais et limites.
**Règle de découpage : jamais aléatoire entre images d'une même zone — par site ou par pays.**

## Vue d'ensemble

| # | Source | Type | Pays | Licence | Usage prévu | Statut |
|---|---|---|---|---|---|---|
| L1 | Mduma & Elinisa 2025 — Banana Leaves Imagery Dataset (Zenodo 10.5281/zenodo.7670326) | Feuilles, smartphone | Tanzanie | CC BY 4.0 | Maladies : **train + val** | ✅ utilisé |
| L2 | Mduma et al. 2023 — Banana leaves & stem (Data in Brief, PMC10333424) | Feuilles + tiges, smartphone | Tanzanie | CC BY (article) | **Même source que L1** (voir fuites) | exclu volontairement |
| L3 | BananaLSD — Arman et al. 2023 (Mendeley 10.17632/9tb7k297ff.1) | Feuilles, 3 smartphones | Bangladesh | CC BY 4.0 | Maladies : **test** (autre pays) + classes inconnues | ✅ utilisé |
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
- **Licence** : CC BY 4.0 (lue via l'API Zenodo, publié le 23/02/2023).
- **Organisation réelle** : 9 archives zip, découpées **par classe puis par tranches de numéros**
  (HEALTHY-1 = HEALTHY_1 à HEALTHY_999, HEALTHY-2 = 1001 à 2000, etc.). Total ≈ 41 Go.
  **Aucune information de district** dans les noms de fichiers. Les archives contiennent des dossiers
  `__MACOSX` (métadonnées macOS, ignorées).
- **Découpage v1 (premier lancement)** : aucun district, et le pHash n'a trouvé qu'**une seule paire** de
  quasi-doublons → le découpage s'est fait de fait **image par image**. La validation ressemblait trop à
  l'entraînement (mêmes séances de prise de vue), d'où un seuil de rejet trop strict (voir résultats v1).
  Pas de fuite vers le test pour autant : le test reste un autre pays.
- **Découpage v2** : par **sessions de prise de vue** — un jour (date EXIF) si ≥ 80 % des photos en ont une,
  sinon des **blocs de 100 photos consécutives** de la même classe (les numéros suivent l'ordre de collecte).
  20 % des sessions en validation, chaque classe ≥ 10 % ; un groupe de quasi-doublons n'est jamais coupé.
  Stratégie appliquée et liste fichier → split (avec date et GPS EXIF) : `resultats_maladies/decoupage.csv`.
- **Contrôle de fuite entre pays** : toute image du Bangladesh à ≤ 6 bits (pHash) d'une image de Tanzanie est retirée du test.
- **Fuites** : recoupement très probable avec L2 (même autrice, même région). Doublons
  quasi identiques possibles → dédoublonnage par hash perceptuel avant découpage.
- **Biais / limites** : fonds, cadrage et éclairage propres à une campagne ; Fusarium
  surreprésenté ; pas de distinction de stade.

#### Résultats du premier lancement (v1, 30/10/2026) — test Bangladesh, rejet par probabilité maximale
| Métrique | Valeur |
|---|---|
| Précision des alertes maladie | 0,95 |
| Rappel Sigatoka | 0,49 |
| Fausses alertes sur feuilles saines | 9 % (12/129) |
| Rejet de l'inconnu (cordana + pestalotiopsis) | 75 % |
| Feuilles connues rejetées à tort | 54 % |
| AUROC connu/inconnu | 0,63 |

Constats : modèle trop sûr de lui (seuil calibré à 0,996), et « fusarium » sert de classe fourre-tout
(81 inconnues et 12 saines classées fusarium ; le fusarium n'existe pas dans le jeu du Bangladesh).
Ces constats ont motivé la v2 (découpage par sessions, rejet par distance kNN).

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
