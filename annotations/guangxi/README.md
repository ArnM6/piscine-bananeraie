# Vérité terrain du comptage — plantation de Guangxi

Annotée à la main dans QGIS par Arnaud (01/11/2026) sur l'orthomosaïque à 8 cm/pixel, affichée en fausses
couleurs infrarouge (bandes 4-3-2).

| Fichier | Contenu |
|---|---|
| `carres_15m.geojson` | 6 carrés de 15 × 15 m : O1, O2 (zone ouest, réglage / entraînement), E1 à E4 (zone est, test) |
| `clics_bruts.geojson` | tous les clics d'annotation, tels quels |
| `verite_comptage.geojson` | vérité terrain utilisée : clics à moins de 0,8 m fusionnés (règle « une touffe = un point ») |

Règles d'annotation : un point au centre de chaque touffe (pied mère + rejets = 1 pied) ; les arbres qui ne
sont pas des bananiers (bosquet de E1) ne sont pas annotés ; chaque carré est annoté en entier.

| Carré | Zone | Clics | Pieds après fusion |
|---|---|---|---|
| O1_15m | ouest_reglage | 24 | 22 |
| O2_15m | ouest_reglage | 34 | 24 |
| E1_15m | est_test | 2 | 2 (carré occupé surtout par d'autres arbres) |
| E2_15m | est_test | 27 | 17 |
| E3_15m | est_test | 26 | 26 |
| E4_15m | est_test | 24 | 24 |

Total : 46 pieds en zone ouest (réglage / entraînement), 69 pieds en zone est (test).

Limites : 8 cm/pixel rend certaines touffes ambiguës même pour un humain ; échantillons modestes (une seule personne a annoté, pas de double annotation pour mesurer l'accord inter-annotateurs).
Les carrés initiaux de 30 m ont été abandonnés (annotation complète irréaliste à cette résolution).
