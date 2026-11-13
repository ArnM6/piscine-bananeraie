# Rapport — Suivi de bananeraie par imagerie aérienne (4 pages max)

## 1. Problème et choix techniques
**Utilisateur** : le chef de culture d'une bananeraie de 17 ha (21 parcelles d'≈ 1 ha, Guangxi, Chine). Sa question
chaque semaine : **quelles parcelles visiter, et pour quoi faire ?** La chaîne transforme une orthomosaïque drone
en bulletin par parcelle trié par priorité, plus une carte cliquable où chaque alerte montre l'image qui la justifie.

Les modules ont été choisis selon **la décision terrain qu'ils permettent**, pas selon leur difficulté :

| Module | Décision permise | Méthode | Pourquoi ce choix |
|---|---|---|---|
| Vigueur (socle) | Où regarder en premier | NDRE drone, comparé à la médiane de la parcelle | Physique simple, validée au sol (§ 4) ; relatif = robuste à la lumière |
| Comptage | Pieds manquants à replanter, prévision de récolte | RetinaNet (DeepForest) affiné | Seul module où un modèle appris bat nettement la méthode classique (§ 3) |
| Maladies | Sigatoka (fongicide) ou fusariose (arrachage, isolement) | MobileNetV3 sur photos de feuilles + rejet kNN | Confirme au sol ce que le drone signale ; dit « je ne sais pas » plutôt que se tromper |
| Stress hydrique | Vérifier l'irrigation d'une parcelle | NDMI Sentinel-2, comparé aux autres parcelles | Le drone n'a ni SWIR ni thermique ; gratuit, sans entraînement |

**Non traité** : le suivi temporel par drone (une seule date de vol disponible). **Principes communs** : chaque modèle
est testé sur un site ou un pays **jamais vu** ; tout tourne sur CPU (comptage : 71 s/ha ; maladies : 8 ms/image) ;
chaque score du bulletin est explicable en une phrase.

## 2. Données
Détail complet (licences, dates, composition, risques) : `docs/carnet_de_donnees.md`. Toutes les sources sont libres
(CC BY 4.0 ou licence Copernicus).

| Source | Rôle | Découpage honnête |
|---|---|---|
| Drone multispectral, Guangxi (Ye et al. 2023), 8 cm, 5 bandes, 80 plants relevés au sol | Vigueur, comptage, démo | Zone **ouest** pour régler, zone **est** pour tester ; 115 pieds annotés à la main dans QGIS |
| ReforesTree (Équateur), drone RGB 2 cm, 6 sites | Apprendre « bananier » au détecteur | Un site entier gardé pour la validation hors site |
| Feuilles, Tanzanie (Mduma & Elinisa), 11 767 photos | Entraînement maladies | Par **séances de prise de vue**, pas par image |
| Feuilles, Bangladesh (BananaLSD), 937 photos | Test maladies + maladies inconnues | Autre pays, jamais vu |
| Sentinel-2 L2A et météo ERA5, 2021–2023 | Stress hydrique et sa vérification | Rien n'est entraîné |

**Fuites et biais évités** :
- **Biais de session (Tanzanie)** : chaque maladie a été photographiée à des dates différentes (saines en novembre,
  fusariose en décembre, Sigatoka en janvier). Un modèle peut reconnaître la séance plutôt que la maladie : la
  validation interne (0,99) ne vaut rien, seul le test au Bangladesh compte.
- Un second jeu tanzanien de la même équipe a été **exclu** (recoupement probable) ; toute photo du Bangladesh
  quasi identique à une photo de Tanzanie (hash perceptuel) est retirée du test.
- Les seuils (rejet de l'inconnu, détection) sont réglés **sur la validation uniquement**, jamais sur le test.
- **Date du vol drone inconnue** : le relevé terrain date de décembre 2022, mais Sentinel-2 montre P01 et P08 en sol nu
  à cette date alors que le drone y compte 115 et 431 pieds. Le vol n'a donc pas eu lieu en décembre 2022.

## 3. Comptage des pieds
Vérité terrain : 6 carrés de 15 × 15 m annotés à la main dans QGIS, un point au centre de chaque touffe
(pied mère + rejets = 1 pied). 46 pieds en zone **ouest** (réglage, affinage), 69 en zone **est** (test, jamais vue).
Validation hors site : ReforesTree, site d'Équateur exclu de l'entraînement (221 bananiers).

**Détecteur** : RetinaNet ResNet-50, l'architecture de DeepForest, à partir de ses poids « arbres ».

| Approche (Guangxi est, 8 cm/px) | Précision | Rappel | F1 (1 m) | Erreur comptage | F1 hors site (ReforesTree 8 cm) |
|---|---|---|---|---|---|
| Baseline maxima locaux (NDVI lissé) | 0,47 | 0,10 | 0,17 | −78 % | — |
| DeepForest brut | 0,28 | 0,52 | 0,36 | +87 % | 0,10 |
| + ReforesTree à 8 cm | 0,58 | 0,22 | 0,32 | −62 % | 0,41 |
| + ReforesTree à 4, 8, 16 cm (**robuste**) | 0,41 | 0,42 | 0,41 | +3 %* | 0,37 |
| **+ affiné sur Guangxi ouest (affine, par défaut)** | 0,62 | 0,45 | **0,52** | −28 % | 0,31 |
| DeepForest affiné sur Guangxi ouest seul | 0,57 | 0,41 | 0,48 | −29 % | 0,08 |

\* erreurs qui se compensent entre carrés (erreur absolue moyenne : 7,5 pieds par carré).

Sobriété : 64,6 Mo, **71 s/ha sur 2 cœurs CPU** (≈ 20 min pour les 18 ha), sans GPU. Affinage : 2 min sur GPU T4.

À retenir :
- le F1 est **multiplié par 3** par rapport à la baseline, qui échoue sur canopée fermée (NDVI uniforme) ;
- l'affinage sur 46 pieds est le gain le moins cher, mais il **spécialise** : hors site, le F1 tombe de 0,37 à 0,31
  (et à 0,17–0,18 à 4 et 16 cm). Sans ReforesTree, le modèle affiné surcompte 7 à 18 fois ailleurs ;
- d'où **deux modèles** dans `run.py` : `affine` pour une monoculture dense proche de Guangxi, `robuste` pour un
  site inconnu très différent.

**Effet de la résolution** (vol 2 × plus haut, même seuil) :

| Modèle | Guangxi 8 cm | Guangxi 16 cm | ReforesTree 4 / 8 / 16 cm |
|---|---|---|---|
| robuste | 0,41 | 0,17 | 0,40 / 0,37 / 0,41 |
| affine | 0,52 | 0,13 | 0,18 / 0,31 / 0,17 |

La résolution utile dépend de la **densité** : 16 cm suffit pour des bananiers isolés (ReforesTree), pas pour une
monoculture fermée, où les touffes voisines se confondent (même sans seuil, 35 détections pour 69 pieds).
**Recommandation de vol : 8 cm/pixel ou plus fin** sur une bananeraie dense.

## 4. Vigueur et stress hydrique
**Vigueur.** La chlorophylle absorbe le rouge et le red-edge ; une feuille saine réfléchit fortement le proche
infrarouge. On retient le **NDRE** = (PIR − RE)/(PIR + RE) plutôt que le NDVI : il sature moins sur une canopée dense
et réagit plus tôt au jaunissement. La carte est lissée sur 2 m (la taille d'un plant), puis chaque zone est comparée
à la **médiane de sa parcelle** : une parcelle entière plus sombre (nuage, autre vol) n'est pas signalée.
Vérification sur les 80 plants relevés au sol (40 sains, 40 atteints de fusariose) :

| Indice | Sains (médiane) | Fusariose (médiane) | AUC sain / malade |
|---|---|---|---|
| NDVI | 0,56 | 0,36 | 0,99 |
| **NDRE (retenu)** | 0,22 | 0,05 | **1,00** (zone de test aussi) |
| Écart relatif à la parcelle | +1,1 | −0,8 | 0,93 |

Réserve : les plants relevés étaient sans doute **visiblement** malades ; le score est optimiste pour une détection précoce.

**Stress hydrique.** L'eau des feuilles absorbe l'infrarouge moyen (SWIR, B11 à 1,6 µm), absent de la caméra du drone.
On utilise Sentinel-2 (20 m, tous les 5 jours, gratuit) : NDMI = (B8A − B11)/(B8A + B11). Deux précautions :
le NDMI baisse aussi quand il y a **moins de feuilles**, donc on retire la part expliquée par le NDVI, et on ne mesure
que les parcelles en **canopée fermée** (NDVI ≥ 0,6), car sur sol nu le NDMI mesure l'humidité du sol. Chaque parcelle
est comparée aux autres **à la même date** (même pluie, même atmosphère) ; alerte si elle est nettement plus sèche
(z < −2) sur 2 images de suite. Données : 100 images 2021–2023 lues, 55 sans nuage sur la plantation, 27 avec assez de
parcelles en canopée fermée. Contrôles de qualité : deux archives Sentinel-2 concordantes (écart de NDMI 0,001), pas de
saut au changement de traitement ESA de 2022.

| Question | Attendu si le NDMI mesure l'eau | Résultat |
|---|---|---|
| Suit-il la pluie des 30 jours précédents ? | lien positif | −0,20 [−0,53 ; +0,21] |
| À quantité de feuilles égale ? | lien positif | **−0,44** [−0,72 ; −0,09], à l'envers |
| Saison sèche plus sèche ? | NDMI plus bas | +0,087, plus humide (p = 0,06) |
| Les écarts entre parcelles sont-ils réels ? | stables d'une image à l'autre | **0,80** (hasard : −0,07) |
| Lien avec la vigueur drone ou la fusariose ? | lien | aucun (−0,11 et −0,08) |

**Conclusion** : la chaîne est correcte et les écarts entre parcelles sont réels, mais **rien ne montre qu'ils viennent
de l'eau** sur ce site. Explications plausibles : plantation irriguée en saison sèche, cycle de culture dominant
(canopée fermée sur 48 % des mesures seulement, replantations décalées), journées claires de saison des pluies
souvent précédées de jours secs. **Décision** : le stress est affiché dans le bulletin (« à vérifier sur le terrain »)
mais **ne compte pas dans le score de santé**, qui repose sur la vigueur seule, validée au sol.

## 5. Maladies
Entraînement : Tanzanie (sain / sigatoka / fusarium). Test : Bangladesh, jamais vu ; cordana et pestalotiopsis
servent de maladies **inconnues**. Seuil de rejet réglé sur la validation Tanzanie uniquement.

**Ablations** (rejet kNN, test Bangladesh) :

| Modèle | Précision alertes | Rappel Sigatoka | Fausses alertes (sain) | Rejet inconnu | Rejet à tort (connu) | AUROC connu/inconnu | Taille (Mo) | ms/image CPU |
|---|---|---|---|---|---|---|---|---|
| **MobileNetV3-large, pré-entraîné, augmentations** | 0,99 | **0,80** | 3 % | **73 %** | 31 % | 0,78 | 24,5 | 8 |
| sans pré-entraînement | 0,92 | 0,73 | 23 % | 25 % | 30 % | 0,50 | — | — |
| sans augmentations | 0,99 | 0,76 | 2 % | 60 % | 30 % | 0,71 | — | — |
| MobileNetV3-small, pré-entraîné, augmentations | 1,00 | 0,74 | 1 % | 71 % | 29 % | 0,78 | 12,2 | 3 |

**Méthode de rejet** (modèle principal) :

| Méthode | Rappel Sigatoka | Fausses alertes | Rejet inconnu | Rejet à tort | AUROC |
|---|---|---|---|---|---|
| Probabilité max. (msp) | 0,68 | 12 % | 70 % | 36 % | 0,73 |
| Énergie | 0,61 | 12 % | 74 % | 42 % | 0,72 |
| **Distance kNN (retenue a priori)** | **0,80** | **3 %** | 73 % | **31 %** | **0,78** |

À retenir :
- le **pré-entraînement est indispensable** : sans lui, le modèle ne distingue plus l'inconnu (AUROC 0,50 = hasard)
  et les fausses alertes passent de 3 % à 23 % ;
- les **augmentations** (lumière, cadrage) améliorent le rejet de l'inconnu de 60 % à 73 % ;
- le **kNN** bat la confiance du classifieur sur toutes les métriques utiles ;
- la version **small** fait presque aussi bien, 2× plus légère et 2,5× plus rapide : bon candidat pour la station de terrain.

## 6. Analyse des échecs
_Comptage (voir `docs/resultats_comptage/exemples_detection.png`)_ :
- **Arbres non-bananiers comptés** (E1, bosquet) : 11 détections pour 2 bananiers, contre 41 pour DeepForest
  brut. L'affinage réduit le piège sans le supprimer ; plus d'exemples négatifs annotés aideraient.
- **Touffes collées manquées** (E3, E4) : ≈ 12 détections pour 24 pieds. Sur canopée fermée, deux touffes
  voisines forment une seule tache : sous-comptage systématique.
- **Densité sur la plantation** : ≈ 450 pieds/ha prédits, contre ≈ 850/ha dans les carrés annotés (choisis dans des
  zones bien couvertes) : le sous-comptage réel est probablement plus fort que −28 %.

_Maladies (voir `docs/resultats_maladies/cas_echec.png`)_ :
- **Sigatoka rejetées** (« je ne sais pas ») : photos du Bangladesh plus ternes et grisâtres que celles de Tanzanie.
  Le modèle préfère s'abstenir plutôt que se tromper : c'est le comportement honnête attendu.
- **Pestalotiopsis acceptée comme Sigatoka** (50 cas) : taches sombres à halo jaune, visuellement très proches.
  Sur le terrain, l'alerte resterait utile (une équipe passe vérifier), mais le diagnostic serait faux.
- **Feuilles « saines » en alerte** (4 cas) : au moins une présente des taches visibles, en double dans le jeu
  → probablement une étiquette discutable.
- **Biais du jeu d'entraînement** : chaque classe a été photographiée à des dates différentes (sain en novembre,
  fusarium en décembre, sigatoka en janvier). Le modèle peut apprendre la session au lieu de la maladie ;
  seul le test dans un autre pays le révèle.

## 7. Limites et passage en conditions réelles
- **Comptage** : 69 pieds de test seulement (l'écart de +0,05 entre affinage avec et sans ReforesTree n'est pas
  significatif) ; seuil des modèles affinés choisi sur les carrés ouest, déjà vus à l'entraînement ; une seule
  plantation dense pour l'affinage. Annoter 2 ou 3 carrés sur chaque nouveau site et affiner (2 min) est la voie
  la plus rentable. Voler à **8 cm/pixel ou plus fin** sur une bananeraie dense.
- **Écart feuilles au sol / drone** : le classifieur de maladies apprend sur des photos prises à quelques centimètres ;
  à 8 cm/pixel, une lésion de Sigatoka (quelques mm) fait moins d'un pixel. Il **ne s'applique pas** à l'orthomosaïque.
  Protocole proposé : le drone (vigueur) désigne les zones faibles, un agent photographie 5 à 10 feuilles par zone avec
  un smartphone, le classifieur tranche entre Sigatoka et fusariose, ou répond « je ne sais pas » → expert.
- **Vigueur** : validée sur des plants déjà visiblement malades ; les plants très desséchés passent sous le seuil de
  végétation et sortent de la carte. Une seule date de vol : impossible de suivre une zone dans le temps.
- **Stress hydrique** : non validé sur ce site (§ 4) ; à 20 m, Sentinel-2 raisonne par parcelle, jamais par plant.
  Pour le valider : quelques **sondes d'humidité du sol** et le calendrier d'irrigation. Pour descendre au niveau du
  plant : une **caméra thermique** sur le drone (une feuille qui manque d'eau ferme ses stomates et chauffe).
- **Date du vol inconnue** : comptage et vigueur (drone) ne datent pas du même jour que l'état Sentinel-2 du bulletin.
  En production, il faut enregistrer la date de chaque vol et prendre l'image satellite la plus proche.

## Bonus — Plan d'intervention
_Parcelles à visiter cette semaine, ordre, gain de temps._
