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
| D2 | Ye et al. 2023 — UAV multispectral images for banana Fusarium wilt survey (Science Data Bank, DOI 10.57760/sciencedb.07000, V2) | Orthomosaïque drone multispectrale + points terrain + contour | Chine (Guangxi) | CC BY 4.0 | Socle : vigueur, comptage, cohérence avec la référence terrain | ✅ utilisé |
| D3 | ReforesTree — Reiersen et al. 2022 (Zenodo 10.5281/zenodo.6813783) | Orthophotos drone RGB 2 cm + boîtes d'arbres | Équateur | CC BY 4.0 (à vérifier sur Zenodo) | Comptage : apprendre « bananier » + **validation hors site** | ✅ utilisé |
| M1 | Courageous Land — agroforestry-inventory, `banana.pth` (Hugging Face) | Modèle pré-entraîné | Brésil | Poids CC BY 4.0, code GPL-3.0 | Comptage : détecteur pré-entraîné | non retenu (voir fiche) |
| M2 | DeepForest `weecology/deepforest-tree` (Hugging Face) | Détecteur RetinaNet de couronnes d'arbres | États-Unis | MIT (à vérifier sur Hugging Face) | Comptage : point de départ du détecteur | ✅ utilisé |
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

#### Ce que révèlent les métadonnées EXIF (v2.1, 31/10/2026) — biais majeur
100 % des photos ont une date EXIF, aucune n'a de GPS. **Chaque classe a été photographiée à des dates différentes** :

| Classe | Jours de prise de vue |
|---|---|
| sain | 22 et 23/11/2022 |
| fusarium | 06, 09, 19, 20 et 22/12/2022 |
| sigatoka | 12 et 13/01/2023 |

La classe est donc **confondue avec la session** (lumière, champ, arrière-plan, appareil). Un modèle peut obtenir
un excellent score en Tanzanie en reconnaissant la session plutôt que la maladie : c'est pourquoi la validation
interne (0,99) n'a pas de valeur et que **seul le test au Bangladesh mesure la vraie performance**.
Découpage appliqué : par jour pour fusarium (5 jours), par blocs de 100 photos consécutives pour sain et sigatoka
(2 jours seulement chacun, découpage par jour impossible). Validation : sain 640, sigatoka 700, fusarium 584 images.

#### Résultats v2.1 (31/10/2026) — test Bangladesh, MobileNetV3-large pré-entraîné, rejet kNN
| Métrique | v1 (msp) | **v2.1 (kNN)** |
|---|---|---|
| Précision des alertes maladie | 0,95 | **0,99** |
| Rappel Sigatoka | 0,49 | **0,80** |
| Fausses alertes sur feuilles saines | 9 % | **3 %** (4/129) |
| Rejet de l'inconnu | 75 % | **73 %** |
| Feuilles connues rejetées à tort | 54 % | **31 %** |
| AUROC connu/inconnu | 0,63 | **0,78** |

Fichiers : `docs/resultats_maladies/` (ablations, méthodes de rejet, découpage complet, planche d'échecs).

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
- **Qualité des étiquettes** : au moins une feuille « healthy » présente des taches visibles, et elle apparaît
  en double dans le jeu (voir `cas_echec.png`). Quelques erreurs de « fausses alertes » viennent sans doute d'étiquettes discutables.

### D2 — Bananeraie de Long'an, Guangxi (drone multispectral)
- **Citation** : Huichun YE, Senzheng CHEN, Anting GUO, et al. *A dataset of UAV multispectral images for banana
  Fusarium wilt survey* [DS/OL]. V2. Science Data Bank, 2023. https://doi.org/10.57760/sciencedb.07000
- **Licence** : CC BY 4.0 (vérifiée sur la page du jeu, publié le 07/06/2023).
- **Provenance** : plantation de bananiers du district de Long'an, Guangxi, Chine.
- **Capteur / résolution** : drone multispectral, vol à 120 m, ≈ 8 cm/pixel (selon l'article associé, *China Scientific Data* 2024).
- **Fichiers** : `1_UAV multispectral reflectance.zip` (585,5 Mo), `2_Ground survey data of banana Fusarium wilt.zip`
  (6,6 Ko, points GPS de plants malades recalés au centre des plants), `3_Boundary of banana planting region.zip` (3,2 Ko).
- **Usage prévu** : carte de vigueur (NDVI / NDRE), cohérence des zones faibles avec les plants malades relevés au sol,
  comptage des pieds (annotation manuelle d'une zone dans QGIS, test sur une autre zone).
- **Découpage** : par zones géographiques séparées de la plantation, jamais par tuiles aléatoires.
- **Image** : 5 bandes en réflectance (float32), 0,08 m/pixel, UTM 48N (EPSG:32648), 14 420 × 7 885 px.
  **Ordre des bandes non documenté** : déduit des spectres de la végétation (bande 4 la plus réfléchissante = PIR,
  bande 5 entre rouge et PIR = red-edge) → Bleu, Vert, Rouge, PIR, Red-edge (ordre des caméras MicaSense RedEdge).
- **Contour** : un polygone de 18,06 ha. **Points terrain** : 80 (40 plants sains `健康植株`, 40 atteints de fusariose
  `枯萎病植株`), relevés au GPS en décembre 2022 (d'après l'historique ArcGIS des fichiers).
- **Préparation** (`scripts/preparer_guangxi.py`) : découpe sur le contour + 10 m, conversion en uint16
  (réflectance × 10 000, compression deflate : 2,3 Go → 365 Mo), grille de 21 parcelles d'≈ 1 ha.
- **Découpage honnête** : deux zones séparées par la médiane est-ouest — `ouest_reglage` (10 parcelles,
  35 points terrain) pour régler les méthodes, `est_test` (11 parcelles, 42 points) pour les tester. 3 points
  tombent dans des morceaux de parcelle trop petits (< 0,3 ha) et sont hors parcelles.
- **Annotation du comptage** : 4 carrés de 30 × 30 m annotés à la main dans QGIS (un pied = un point au centre de
  la touffe) — O1, O2 (zone ouest) et E1, E2 (zone est). E1 contient un bosquet d'arbres non-bananiers (piège).
- **Limites** : date de vol exacte non fournie ; pas de bande thermique ; une seule date (pas de suivi temporel).
  Baseline de comptage par maxima locaux du NDVI : ≈ 200 pieds/ha détectés alors que la canopée en montre
  visuellement 3 à 4 fois plus → la canopée fermée rend le NDVI uniforme et les centres de touffes invisibles.

### D3 — ReforesTree (Équateur)
- **Citation** : Reiersen et al., *ReforesTree: A Dataset for Estimating Tropical Forest Carbon Stock with
  Deep Learning and Aerial Imagery*, AAAI 2022. Zenodo 10.5281/zenodo.6813783 (archive 7,5 Go, md5 contrôlé).
- **Contenu** : 6 sites d'agroforesterie (cacao, agrumes, bananiers…), orthophotos RGB ≈ 2 cm/pixel,
  boîtes autour de chaque arbre avec son espèce. Seules les boîtes `banana` sont des cibles ; les autres
  arbres restent dans l'image comme **exemples négatifs**.
- **Préparation** : réduction à 4, 8 et 16 cm/pixel (moyenne par blocs = même caméra volant 2, 4, 8 fois plus haut).
  Cinq tuiles écartées comme le recommandent les auteurs (annotations incomplètes).
- **Découpage par site** (`docs/resultats_comptage/decoupage_reforestree.csv`) : 5 sites en entraînement
  (1 283 bananiers), **Carlos Vera Arteaga en validation** (221 bananiers, 522 autres arbres), jamais vu.
- **Limite** : bananiers isolés sous couvert agroforestier, très différents d'une monoculture fermée (Guangxi).

### M2 — DeepForest (point de départ du détecteur)
- RetinaNet ResNet-50 + FPN pré-entraîné sur des couronnes d'arbres de forêts américaines (~10 cm/pixel).
  Poids chargés directement depuis Hugging Face ; boucle d'entraînement écrite dans le notebook.
- Utilisé brut (`arbres_brut`) puis affiné : sur ReforesTree, puis sur les carrés O1, O2 de Guangxi.
- **Fuite évitée** : les carrés de test E1–E4 ne servent ni à l'entraînement ni au réglage du seuil.

### M1 — Modèle Courageous Land `banana.pth`
- Entraîné sur la ferme Raízes (São Paulo), orthophotos RTK ~1,66 cm/pixel, 133 étiquettes.
- Performances publiées sur leur site : précision 0,59, rappel 0,93 (appariement à 1 m).
- Défaut connu : compte les rejets d'une même touffe comme des pieds distincts → surcomptage.
- **Usage documenté** : inférence seule, puis fusion des détections < 1,4 m ; évalué sur nos propres annotations.
- **Non retenu** : entraîné à 1,7 cm/pixel (contre 8 cm à Guangxi) et sur 133 étiquettes ; DeepForest peut être
  affiné avec nos annotations et ReforesTree, ce qui teste directement l'effet de la résolution.

### S1 — Sentinel-2
- Bandes utilisées : B04 (rouge), B08 (PIR), B05 (red-edge) et B11 (SWIR) pour NDMI.
- Zone, dates (saison sèche / saison des pluies) et couverture nuageuse : _à remplir_.

---

## Écart sol / drone (point de vigilance du sujet)
Les images de feuilles sont prises au sol, de près (quelques cm). Un drone à 40 m
voit ~1–2 cm/pixel : les lésions de quelques millimètres sont invisibles ou réduites
à quelques pixels. Le classifieur de feuilles **ne peut pas être appliqué tel quel**
aux orthomosaïques. _À compléter : conséquences et protocole terrain proposé._
