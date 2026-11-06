# Rapport — Suivi de bananeraie par imagerie aérienne (4 pages max)

## 1. Problème et choix techniques
_Socle + modules choisis (maladies, stress hydrique) et pourquoi._

## 2. Données
_Résumé du carnet ; découpage par pays/site ; fuites évitées._

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
_Indices choisis, justification physique, comparaison saison sèche / humide._

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
  la plus rentable.
- _Écart feuilles au sol / drone : à compléter._

## Bonus — Plan d'intervention
_Parcelles à visiter cette semaine, ordre, gain de temps._
