"""
Step 6b of the RAG pipeline: validation and normalisation.

The raw extraction (results/extractions.json) contains the right names, but
also noise typical of a small model: objects ("anklets", "lute", "ruby"),
places ("ganges river", "hermitage"), characters ("shakuntala"), animals
filed as plants ("buffalo"), and many variants of one name ("jasmine",
"jasmines", "jasmine-blossom", "jasmine-vine").

This script
  1. asks the local LLM, for each candidate and the sentence it came from,
     whether it is a real plant or animal and what its base name is;
  2. applies the author's corrections listed in AUTHOR_CORRECTIONS
     (cases the model still got wrong, each one recorded here);
  3. merges variants under the base name and writes
     results/validated.json, which make_flora_fauna.py turns into
     flora_fauna.py.

Run:  python validate_results.py      (Ollama must be running)
"""

import json
from collections import defaultdict
from pathlib import Path

from rag_pipeline import LLM_MODEL, ollama

HERE = Path(__file__).parent
RESULTS = HERE / "results"

CATEGORIES = ["tree", "flower", "herb_or_grass", "crop_or_fruit", "creeper",
              "mammal", "bird", "reptile", "fish_or_aquatic", "insect", "amphibian"]

PROMPT = """Decide whether each CANDIDATE below, as used in its SENTENCE from
an Indian literary text, names a real plant or a real animal.

For each candidate return:
- "candidate": the candidate exactly as given
- "is_real": true only for a real plant or animal species or kind
  (false for objects, places, rivers, mountains, celestial bodies, people,
  gods, mythical beings, body parts, ornaments and weapons)
- "base_name": the plain singular common name of the plant or animal,
  without words like tree, blossom, flower, bud, leaf, pollen, vine, branch
  (e.g. "jasmine-blossoms" -> "jasmine", "mango-tree" -> "mango",
  "coral-trees" -> "coral tree", "lioness" -> "lion", "fawns" -> "deer")
- "kingdom": "flora" or "fauna"
- "category": one of {categories}

CANDIDATES:
{items}

Return JSON: {{"results": [ ... ]}}"""

SCHEMA = {"type": "object", "properties": {"results": {"type": "array", "items": {
    "type": "object",
    "properties": {"candidate": {"type": "string"}, "is_real": {"type": "boolean"},
                   "base_name": {"type": "string"},
                   "kingdom": {"type": "string", "enum": ["flora", "fauna"]},
                   "category": {"type": "string", "enum": CATEGORIES}},
    "required": ["candidate", "is_real", "base_name", "kingdom", "category"]}}},
    "required": ["results"]}

# Author's corrections after reading the model's validation against the text.
# candidate -> None (reject) or (base_name, kingdom, category)
AUTHOR_CORRECTIONS = {
    # not plants or animals
    "fairies": None, "goddess": None, "hermit-girl": None, "holiest hermits": None,
    "malini": None, "sarasvati": None, "somatirtha": None, "reva river": None,
    "jumna river": None, "ganges river": None, "ganges": None, "himalaya mountains": None,
    "hill": None, "range": None, "stream": None, "river": None, "ocean": None,
    "cloud": None, "moon": None, "mountain-echo": None, "thunderbolt": None,
    "fires": None, "coals": None, "emerald": None, "ruby": None, "conch": None,
    "columns": None, "picture": None, "lute": None, "quiver": None, "arrow": None,
    "chariot": None, "anklets": None, "girdle": None, "ribbon": None,
    "fevered limbs": None, "stumbling feet": None, "misfortunes": None,
    "budding charms": None, "hermitage": None, "pious grove": None,
    "shakuntala": None, "serpent-ring": None, "peacock-train": None,
    "bark": None, "bark dress": None, "barken dress": None, "roots": None,
    "cocoa-nut casket": None, "cuscus-salve": None, "lotus-bracelet": None,
    "perfumed lotus-chain": None, "lily-dotted lakes": None, "lily-pond": None,
    "tree of wishing": None, "wishing-tree": None,      # mythical tree
    "blossoms": None, "budding blossoms": None, "blossom-burdened trees": None,
    "shade-trees": None, "vines": None, "toughest vine": None, "vine-bride": None,
    # wrong kingdom or base name
    "buffalo": ("buffalo", "fauna", "mammal"),
    "cow": ("cow", "fauna", "mammal"),
    "serpent": ("snake", "fauna", "reptile"),
    "frog": ("frog", "fauna", "amphibian"),
    "fawn": ("deer", "fauna", "mammal"), "fawns": ("deer", "fauna", "mammal"),
    "antelope": ("antelope", "fauna", "mammal"),
    "lion cub": ("lion", "fauna", "mammal"), "lioness": ("lion", "fauna", "mammal"),
    "coral-tree": ("coral tree", "flora", "tree"), "coral-trees": ("coral tree", "flora", "tree"),
    "rose-apples": ("rose-apple", "flora", "crop_or_fruit"),
    "karkandhu-fruit": ("jujube", "flora", "crop_or_fruit"),
    "jujube-trees": ("jujube", "flora", "tree"),
    "rice-stems": ("rice", "flora", "crop_or_fruit"),
    "plantain-buds": ("plantain", "flora", "crop_or_fruit"),
    "mushroom": ("mushroom", "flora", "herb_or_grass"),
    "sacred grass": ("sacred grass (kusha)", "flora", "herb_or_grass"),
    "spring-creeper": ("spring creeper", "flora", "creeper"),
    "spring-creepers": ("spring creeper", "flora", "creeper"),
    "jasmine-vine": ("jasmine", "flora", "creeper"),
    "madder-bush": ("madder", "flora", "herb_or_grass"),
    "nipa plants": ("nipa", "flora", "tree"),
    "ketak": ("ketak (screw-pine)", "flora", "flower"),
    "ketak-blossom": ("ketak (screw-pine)", "flora", "flower"),
    "panic grass": ("panic grass", "flora", "herb_or_grass"),
    "reed": ("reed", "flora", "herb_or_grass"),
    "flowers of grasses": None,
    "jasmine blossom": ("jasmine", "flora", "flower"),
    "jasmine-blossoms": ("jasmine", "flora", "flower"),
    "starling": ("starling", "fauna", "bird"),
    "ape": ("ape", "fauna", "mammal"),
    "ruddy goose": ("ruddy goose (chakravaka)", "fauna", "bird"),
    # ---- added after the five-book run (5 November 2026) ----
    # not plants or animals: objects, food, places, generic words
    # (bonti = a fish-cutter; Bambu grove = a monastery; tilka = the tilak mark)
    "blossom-dotted couch": None, "branches": None, "curry-stuffs": None, "dead leaves": None,
    "withered leaves": None, "firewood": None, "flowering shrubs": None, "garden trees": None,
    "poisonous trees": None, "holy peak": None, "molasses": None, "pebble": None,
    "waggons": None, "quail-catcher": None, "roast mutton": None, "insects": None,
    "reptiles": None, "yellow bird": None, "topmost shoots": None, "twig": None,
    "thorn bush": None, "weeds": None, "creepers": None, "water-plant": None, "spice": None,
    "spices": None, "bonti": None, "canes": None, "bambu grove": None, "tilka": None,
    # wrong base name, kingdom or category (kul = jujube, jak = jackfruit,
    # bel = wood-apple, keova = screw-pine, dhadikulas 'sing' so a bird)
    "paddy birds": ("paddy bird (pond heron)", "fauna", "bird"),
    "thoroughbreds": ("horse", "fauna", "mammal"),
    "tawny heifers": ("cow", "fauna", "mammal"),
    "lamb": ("sheep", "fauna", "mammal"),
    "lambs": ("sheep", "fauna", "mammal"),
    "stag": ("deer", "fauna", "mammal"),
    "roe": ("deer", "fauna", "mammal"),
    "monkey deer": ("deer", "fauna", "mammal"),
    "kurunga": ("antelope", "fauna", "mammal"),
    "bel": ("bel (wood-apple)", "flora", "crop_or_fruit"),
    "jak": ("jackfruit", "flora", "crop_or_fruit"),
    "kul": ("jujube", "flora", "crop_or_fruit"),
    "asoka": ("ashoka", "flora", "tree"),
    "asoka-tree": ("ashoka", "flora", "tree"),
    "sirisha": ("siris", "flora", "tree"),
    "koel": ("koel", "fauna", "bird"),
    "koil": ("koel", "fauna", "bird"),
    "kokila": ("koel", "fauna", "bird"),
    "papiya": ("papiya (hawk-cuckoo)", "fauna", "bird"),
    "dahuk": ("dahuk (waterhen)", "fauna", "bird"),
    "dhadikulas": ("dhadikula", "fauna", "bird"),
    "oriole": ("oriole", "fauna", "bird"),
    "mice": ("mouse", "fauna", "mammal"),
    "pigeon": ("pigeon", "fauna", "bird"),
    "pigeons": ("pigeon", "fauna", "bird"),
    "partridge": ("partridge", "fauna", "bird"),
    "kamala flowers": ("lotus", "flora", "flower"),
    "lotus-buds": ("lotus", "flora", "flower"),
    "lotus-flower": ("lotus", "flora", "flower"),
    "keovas": ("ketak (screw-pine)", "flora", "flower"),
    "sandal": ("sandalwood", "flora", "tree"),
    "cocoanut": ("coconut", "flora", "tree"),
    "cocoanut palms": ("coconut", "flora", "tree"),
    "cocoanut trees": ("coconut", "flora", "tree"),
    "fan palm": ("palm", "flora", "tree"),
    "babla tree": ("babul (acacia)", "flora", "tree"),
    "musk-rats": ("musk-rat", "fauna", "mammal"),
    "kusm flowers": ("kusum", "flora", "flower"),
    "gourd-flower": ("gourd", "flora", "crop_or_fruit"),
    "vegetable pumpkins": ("pumpkin", "flora", "crop_or_fruit"),
    "dabba grass": ("durva grass", "flora", "herb_or_grass"),   # dabba = durva, the sacred grass
    "kunda": None,   # "the flower, Kunda": the heroine Kunda Nandini, a metaphor, not the plant
}

FAUNA_CATEGORIES = {"mammal", "bird", "reptile", "fish_or_aquatic", "insect", "amphibian"}


def validate(batch):
    items = "\n".join(f'- "{r["name"]}": {r["evidence"]["sentence"][:240]}' for r in batch)
    out = ollama("chat", {
        "model": LLM_MODEL, "stream": False, "format": SCHEMA,
        "options": {"temperature": 0, "num_ctx": 8192},
        "messages": [{"role": "user", "content": PROMPT.format(
            categories=", ".join(CATEGORIES), items=items)}]})
    return {v["candidate"].lower(): v for v in json.loads(out["message"]["content"])["results"]}


def main():
    data = json.loads((RESULTS / "extractions.json").read_text(encoding="utf-8"))
    records = data["records"]

    model_view = {}
    for i in range(0, len(records), 15):
        model_view.update(validate(records[i:i + 15]))
        print(f"  validated {min(i + 15, len(records))}/{len(records)}")

    merged = defaultdict(lambda: {"variants": set(), "books": set(), "works": set(),
                                  "evidence": None})
    log = []
    for r in records:
        name = r["name"]
        m = model_view.get(name)
        decision = m and ((m["base_name"].lower().strip(), m["kingdom"], m["category"])
                          if m["is_real"] else None)
        source = "model"
        if name in AUTHOR_CORRECTIONS:
            corrected = AUTHOR_CORRECTIONS[name]
            if corrected != decision:
                source = "author"
            decision = corrected
        log.append({"candidate": name, "decision": decision, "by": source})
        if decision is None:
            continue
        base, kingdom, category = decision
        kingdom = "fauna" if category in FAUNA_CATEGORIES else "flora"   # category decides
        entry = merged[(base, kingdom)]
        entry["category"] = entry.get("category") or category
        entry["variants"].add(name)
        entry["books"].update(r["books"])
        entry["works"].update(r["sources"])
        entry["evidence"] = entry["evidence"] or r["evidence"]

    final = [{"name": base, "kingdom": k, "category": e["category"],
              "variants": sorted(e["variants"]), "books": sorted(e["books"]),
              "sources": sorted(e["works"]),
              "evidence": e["evidence"]}
             for (base, k), e in sorted(merged.items())]
    (RESULTS / "validated.json").write_text(json.dumps(
        {"model": data["model"], "embedding_model": data["embedding_model"],
         "records": final, "decisions": log}, ensure_ascii=False, indent=2), encoding="utf-8")
    kept = sum(d["decision"] is not None for d in log)
    print(f"{len(records)} candidates -> {kept} accepted -> {len(final)} distinct names "
          f"({sum(d['by'] == 'author' for d in log)} author corrections)")


if __name__ == "__main__":
    main()
