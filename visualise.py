"""
Charts of the corpus and the extracted flora and fauna.

Run:  python visualise.py      (needs matplotlib; reads corpus/ and results/)
Writes figures/corpus_share.png, figures/flora_fauna_categories.png,
figures/checking_outcome.png and figures/checking_by_book.png
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from rag_pipeline import BOOKS, load_chunks

HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
INK, INK_2 = "#1b1f2a", "#5b6275"
ROSE = LinearSegmentedColormap.from_list("rose", ["#fbf4f7", "#f1cfdd", "#de95b2", "#c25d87", "#b5446e", "#6e2342"])
plt.rcParams.update({"font.family": "Helvetica", "text.color": INK, "axes.labelcolor": INK_2,
                     "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white"})
SHORT = {b: b.split(", ")[-1] for b in BOOKS.values()}
ORDER = list(BOOKS.values())

ext = json.loads((HERE / "results/extractions.json").read_text(encoding="utf-8"))
val = json.loads((HERE / "results/validated.json").read_text(encoding="utf-8"))
records = val["records"]


def pie(ax, labels, values, colors, title):
    ax.pie(values, colors=colors, startangle=90, counterclock=False,
           wedgeprops={"edgecolor": "white", "linewidth": 2},
           autopct=lambda p: f"{p:.0f}%" if p >= 6 else "", pctdistance=0.72,
           textprops={"fontsize": 10, "color": "white", "fontweight": "bold"})
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=9)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


# 1. Corpus share (words)
words = Counter()
for c in load_chunks():
    words[c["book"]] += len(c["text"].split()[:200])          # passages overlap by 50 words
fig, ax = plt.subplots(figsize=(5.5, 5.2))
pie(ax, [SHORT[b] for b in ORDER], [words[b] for b in ORDER],
    ["#e07b24", "#2e8b57", "#3e6fb0", "#b5446e", "#6b5ba8"], "The corpus: share of words")
save(fig, "corpus_share.png")

# 2. Flora and fauna by category
FL = {"tree": "Trees", "flower": "Flowers", "herb_or_grass": "Herbs and grasses",
      "crop_or_fruit": "Crops and fruit", "creeper": "Creepers"}
FA = {"mammal": "Mammals", "bird": "Birds", "reptile": "Reptiles", "fish_or_aquatic": "Fish",
      "insect": "Insects", "amphibian": "Amphibians"}
fc = Counter(r["category"] for r in records if r["kingdom"] == "flora")
ac = Counter(r["category"] for r in records if r["kingdom"] == "fauna")
fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
fk, ak = [k for k in FL if fc[k]], [k for k in FA if ac[k]]
pie(axes[0], [FL[k] for k in fk], [fc[k] for k in fk],
    ["#2e8b57", "#7bbf8e", "#a7c957", "#c9a227", "#5b8c5a"], f"Flora ({sum(fc.values())} names)")
pie(axes[1], [FA[k] for k in ak], [ac[k] for k in ak],
    ["#e07b24", "#f2a65a", "#b5446e", "#3e6fb0", "#c9a227", "#6b5ba8"], f"Fauna ({sum(ac.values())} names)")
save(fig, "flora_fauna_categories.png")

# 3. Outcome of every name the LLM proposed
decision = {d["candidate"]: d["decision"] for d in val["decisions"]}
real = sum(1 for d in decision.values() if d)
not_living = sum(1 for d in decision.values() if not d)
invented = len(ext["rejected_not_in_text"])
fig, ax = plt.subplots(figsize=(5.5, 5.2))
pie(ax, [f"Real plants or animals ({real})", f"Not living things ({not_living})",
         f"Not in the passage ({invented})"], [real, not_living, invented],
    ["#2e8b57", "#e07b24", "#b5446e"], f"All {real + not_living + invented} names the LLM proposed")
save(fig, "checking_outcome.png")

# 4. Heatmap: outcome by book (row %)
book_of = lambda src: next(b for b in ORDER if src.startswith(b.split(",")[0]))
out = {b: Counter() for b in ORDER}
for r in ext["records"]:
    for b in r["books"]:
        out[b]["real" if decision.get(r["name"]) else "not living"] += 1
for r in ext["rejected_not_in_text"]:
    out[book_of(r["source"])]["not in passage"] += 1
cols = ["real", "not living", "not in passage"]
counts = np.array([[out[b][c] for c in cols] for b in ORDER])
M = 100 * counts / counts.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.imshow(M, cmap=ROSE, vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(3), ["Real plant\nor animal", "Not a living\nthing", "Not in the\npassage"])
ax.set_yticks(range(len(ORDER)), [SHORT[b] for b in ORDER])
ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
ax.set_yticks(np.arange(-.5, len(ORDER), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2)
ax.tick_params(which="both", length=0)
for s in ax.spines.values():
    s.set_visible(False)
for i in range(len(ORDER)):
    for j in range(3):
        ax.text(j, i, f"{M[i, j]:.0f}%", ha="center", va="center", fontsize=10,
                color="white" if M[i, j] > 55 else INK)
    ax.text(2.6, i, f"  n = {counts[i].sum()}", va="center", ha="left", fontsize=9, color=INK_2)
ax.set_xlim(-0.5, 3.4)
ax.set_title("What happened to the LLM's names, by book (row %)", loc="left", fontsize=12, fontweight="bold")
save(fig, "checking_by_book.png")

print({b: dict(out[b]) for b in ORDER})
