"""Génère docs/rapport.pdf à partir de docs/rapport.md et des figures de docs/.
Usage : python scripts/rapport_pdf.py . docs/rapport.pdf   (pip install weasyprint markdown)"""
import sys, re, markdown, weasyprint
from pathlib import Path
repo = Path(sys.argv[1]); out = Path(sys.argv[2])
md = (repo/"docs/rapport.md").read_text(encoding="utf-8")
title, md = md.split("\n",1)
title = title.lstrip("# ").replace(" (4 pages max)","")

def fig(path, cap, cls="full"):
    return f'\n\n<figure class="{cls}"><img src="{path}"><figcaption>{cap}</figcaption></figure>\n\n'
def fig2(a, b, cap):
    return (f'\n\n<figure class="full duo"><div><img src="{a}"><img src="{b}"></div>'
            f'<figcaption>{cap}</figcaption></figure>\n\n')

# Stress figures: before the stress validation table
md = md.replace("| Question | Attendu si le NDMI mesure l'eau",
  fig("resultats_stress/serie_plantation.png",
      "Figure 1 — NDMI médian des parcelles en canopée fermée, NDVI et bilan pluie − ETP sur 30 jours (2021–2023).")
  + fig2("resultats_stress/carte_stress.png","resultats_stress/parcelles_dates.png",
      "Figure 2 — Gauche : stress relatif par parcelle (11/12/2022, rouge = alerte, hachuré = pas de canopée). "
      "Droite : écart de chaque parcelle aux autres, date par date — écarts stables d'une image à l'autre.")
  + "| Question | Attendu si le NDMI mesure l'eau", 1)

md = md.replace("_Comptage (voir `docs/resultats_comptage/exemples_detection.png`)_ :",
  fig("resultats_comptage/exemples_detection.png",
      "Figure 3 — Carrés de test E1–E4 : baseline NDVI (gauche) et modèle retenu (droite). "
      "Cercles verts = pieds annotés, croix rouges = détections.", "side")
  + "_Comptage (Figure 3)_ :", 1)
md = md.replace("_Maladies (voir `docs/resultats_maladies/cas_echec.png`)_ :", "_Maladies (Figure 4)_ :", 1)
md = md.replace("## 7. Limites",
  fig("resultats_maladies/cas_echec.png",
      "Figure 4 — Cas d'échec sur le test Bangladesh : saines en alerte, Sigatoka rejetées, "
      "Pestalotiopsis (inconnue) acceptée comme Sigatoka.") + "## 7. Limites", 1)

md = md.replace("_Parcelles à visiter cette semaine, ordre, gain de temps._",
  "_Parcelles à visiter cette semaine, ordre, gain de temps._ (`scripts/plan_intervention.py`, après `run.py`)\n\n"
  "Les zones de vigueur faible deviennent des **haltes** (regroupées à moins de 8 m). **Priorité 1** : toutes celles des "
  "parcelles en alerte vigueur et des parcelles signalées plus sèches par Sentinel-2 ; **priorité 2** : les taches d'au moins "
  "50 m² ailleurs (les petites taches isolées attendent le vol suivant). La tournée à pied passe d'abord par la priorité 1, "
  "dans un ordre optimisé (plus proche voisin + 2-opt), et son temps est comparé à une inspection complète de la plantation. "
  "Sorties : `plan_intervention.csv` (ordre, parcelle, surface, heure d'arrivée, action) et `plan_intervention.geojson`, "
  "affichés sur la carte de démo. Hypothèses de temps (3 km/h, 10 min par halte…) réglables avec le chef de culture.")
# blank line before lists so Markdown renders them
lines=md.split("\n"); outl=[]
for l in lines:
    if l.startswith("- ") and outl and outl[-1].strip() and not outl[-1].startswith(("- ","  ")):
        outl.append("")
    outl.append(l)
md="\n".join(outl)
body = markdown.markdown(md, extensions=["tables"])
css = """
@page { size: A4; margin: 1.5cm 1.6cm 1.6cm;
  @bottom-center { content: counter(page) " / " counter(pages); font: 8pt Inter; color:#888; } }
body { font-family: Inter, sans-serif; font-size: 8.6pt; line-height: 1.38; color:#1d2327; }
h1 { font-size: 16pt; margin: 0 0 2pt; color:#1f5130; }
.sub { color:#666; font-size:8.5pt; margin-bottom:10pt; border-bottom:1.5pt solid #2e7d4f; padding-bottom:6pt;}
h2 { font-size: 11.5pt; color:#1f5130; margin: 12pt 0 4pt; break-after: avoid; }
p { margin: 3pt 0 5pt; } ul { margin: 2pt 0 5pt; padding-left: 14pt; } li { margin-bottom: 1.5pt; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 7pt; font-size: 7.6pt; break-inside: avoid; }
th { background:#e6f0e9; color:#1f5130; text-align:left; font-weight:600; }
th, td { border: 0.5pt solid #c9d6cd; padding: 2.5pt 4pt; vertical-align: top; }
tr:nth-child(even) td { background:#f7faf8; }
code { font-family: 'DejaVu Sans Mono'; font-size: 7.4pt; background:#f1f3f2; padding:0 2pt; }
strong { color:#111; }
figure { margin: 6pt 0 8pt; break-inside: avoid; }
figure.full img { width: 100%; }
figure.duo div { display:flex; gap:6pt; align-items:center; }
figure.duo img:first-child { width: 44%; } figure.duo img:last-child { width: 56%; }
figure.side { float: right; width: 36%; margin: 0 0 6pt 10pt; }
figure.side img { width: 100%; }
figcaption { font-size: 7.3pt; color:#555; margin-top: 2pt; font-style: italic; }
"""
html = f"""<html><head><meta charset="utf-8"><style>{css}</style></head><body>
<h1>{title}</h1><div class="sub">Piscine Intelligence Lab — ECE Paris · Arnaud · novembre 2026 ·
Code : github.com/ArnM6/piscine-bananeraie</div>{body}</body></html>"""
weasyprint.HTML(string=html, base_url=str(repo/"docs")+"/").write_pdf(out)
print("ok")
