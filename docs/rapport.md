# Rapport — Suivi de bananeraie par imagerie aérienne (4 pages max)

## 1. Problème et choix techniques
_Socle + modules choisis (maladies, stress hydrique) et pourquoi._

## 2. Données
_Résumé du carnet ; découpage par pays/site ; fuites évitées._

## 3. Comptage des pieds
| Approche | F1 (1 m) | Erreur comptage | Taille | Temps CPU / ha |
|---|---|---|---|---|
| Baseline maxima locaux (NDVI lissé) — zone test Guangxi, 8 cm/px | **0,17** (P 0,47 / R 0,10) | −78 % | 0 Mo | 2,4 s/ha |
| Détecteur pré-entraîné (Courageous Land) | | | | |
| Détecteur affiné | | | | |

Vérité terrain : 6 carrés de 15 × 15 m annotés à la main (46 pieds en zone ouest pour le réglage, 69 en zone est
pour le test). La baseline échoue sur canopée fermée : le NDVI y est uniforme et le centre des touffes ne forme
pas de « bosse », d'où 3 à 4 fois trop peu de détections → justifie un détecteur appris.

Effet de la résolution : _résultats à 8, 16, 24 cm/pixel._

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
_Écart feuilles au sol / drone ; transfert entre sites._

## Bonus — Plan d'intervention
_Parcelles à visiter cette semaine, ordre, gain de temps._
