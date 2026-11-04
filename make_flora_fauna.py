"""
Final step of the RAG pipeline: write flora_fauna.py.

Reads the validated names in results/validated.json and writes them in
the format given in the assignment instructions (two lists,
flora_RAG_extraction and fauna_RAG_extraction), followed by the same names
grouped by category and the books each name was found in.

Run:  python make_flora_fauna.py
"""

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
data = json.loads((HERE / "results" / "validated.json").read_text(encoding="utf-8"))
records = data["records"]

HEADER = f'''"""
============================================================
Topic: Extraction of Indian Cultural Elements Using RAG
============================================================

Author Name(s): Khushi Jha

Short Description: Names of Indian flora and fauna extracted with a
Retrieval-Augmented Generation (RAG) pipeline from a repository of five
Indian literary texts in public-domain English translation:
  1. Kalidasa, Shakuntala and Other Works (Sanskrit; tr. Arthur W. Ryder)
  2. Buddhist Birth Stories, or Jataka Tales, vol. 1 (Pali; tr. T. W.
     Rhys Davids)
  3. Bankim Chandra Chattopadhyay, The Poison Tree (Bengali novel
     Vishabriksha; tr. Miriam S. Knight)
  4. Sarojini Naidu, The Golden Threshold (written in English)
  5. Rabindranath Tagore, The Gardener (Bengali; tr. by the author)
Passages were embedded with {data["embedding_model"]}, the passages of
each book most relevant to 12 flora/fauna queries were retrieved, and a
local LLM ({data["model"]},
run through Ollama) listed the plant and animal names in each passage.
Every name was then checked against the passage it came from, and
names not found in the text were discarded. A validation pass (the LLM
plus recorded author corrections) removed non-living items such as
ornaments, rivers and characters, and merged variants such as
"jasmine-blossoms" under one base name. See rag_pipeline.py and
validate_results.py.

Semester: Monsoon Semester 2026

Affiliation: Department of Humanities and Social Sciences,
Indian Institute of Technology (Indian School of Mines) Dhanbad
============================================================
"""
'''


def py_list(name, items, per_line=5):
    lines = [f"{name} = ["]
    for i in range(0, len(items), per_line):
        lines.append("    " + " ".join(f'"{x}",' for x in items[i:i + per_line]))
    lines.append("]")
    return "\n".join(lines)


def main():
    flora = sorted(r["name"].title() for r in records if r["kingdom"] == "flora")
    fauna = sorted(r["name"].title() for r in records if r["kingdom"] == "fauna")

    by_category = defaultdict(list)
    for r in records:
        by_category[(r["kingdom"], r["category"])].append(r["name"].title())

    parts = [HEADER, py_list("flora_RAG_extraction", flora), "",
             py_list("fauna_RAG_extraction", fauna), "", "",
             "# ---- Additional detail: names grouped by category ----", ""]
    for kingdom in ("flora", "fauna"):
        parts.append(f"{kingdom}_by_category = {{")
        for (k, cat), names in sorted(by_category.items()):
            if k == kingdom:
                parts.append(f'    "{cat}": {json.dumps(sorted(names), ensure_ascii=False)},')
        parts.append("}\n")

    parts.append("# Books in which each name was found")
    parts.append("found_in = {")
    for r in sorted(records, key=lambda r: r["name"]):
        parts.append(f'    "{r["name"].title()}": {json.dumps(r["books"])},')
    parts.append("}\n")

    parts.append('if __name__ == "__main__":')
    parts.append('    print("Flora:", len(flora_RAG_extraction), "names")')
    parts.append('    print("Fauna:", len(fauna_RAG_extraction), "names")')

    out = HERE / "flora_fauna.py"
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote {out.name}: {len(flora)} flora, {len(fauna)} fauna")


if __name__ == "__main__":
    main()
