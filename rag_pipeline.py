"""
============================================================
Topic: Extraction of Indian Cultural Elements Using RAG
       (Retrieval-Augmented Generation pipeline)
============================================================
Author Name(s): Khushi Jha
Semester: Monsoon Semester 2026
Affiliation: Department of Humanities and Social Sciences,
Indian Institute of Technology (Indian School of Mines) Dhanbad

Pipeline (all models run locally through Ollama):
  1. LOAD      read the five books in corpus/ (cleaned by preprocess.py);
               Kalidasa's book is split into its seven works and its
               translator's introduction is left out
  2. CHUNK     split each text into overlapping passages (~250 words)
  3. EMBED     turn every passage into a vector with nomic-embed-text
  4. RETRIEVE  for each flora/fauna query, find the most similar passages
               in EACH book (so that the longest book cannot crowd out
               the shorter ones)
  5. GENERATE  ask the local LLM (qwen2.5:7b) to list only the plant and
               animal names that appear in those passages
  6. VERIFY    keep a name only if it really occurs in its passage
               (guards against the LLM inventing names)
  7. SAVE      write results/extractions.json and ../flora_fauna.py

Requirements: Ollama running locally with the two models pulled:
    ollama pull nomic-embed-text
    ollama pull qwen2.5:7b
Run:  python rag_pipeline.py
============================================================
"""

import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
RESULTS = HERE / "results"
OLLAMA = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:7b"

CHUNK_WORDS = 250
OVERLAP_WORDS = 50
TOP_K = 10          # passages retrieved per query, per book

# Retrieval queries: one per category of flora and fauna.
QUERIES = {
    "flora": [
        "trees in the forest or village: mango, banyan, asoka, sal, palm, bamboo",
        "flowers and blossoms: lotus, jasmine, kadamba, champak, rose",
        "creepers, vines, grass and herbs",
        "crops and grains in the fields: rice, barley, sugar-cane, corn",
        "fruits, gardens and orchards",
        "sandalwood, henna, spices and fragrant plants",
    ],
    "fauna": [
        "wild animals of the jungle: tiger, lion, elephant, deer, jackal",
        "birds singing: cuckoo, koel, peacock, parrot, swan, crane",
        "snakes, crocodiles, tortoises and other reptiles",
        "fish and creatures of the river and the sea",
        "bees, insects and butterflies",
        "domestic animals: cow, horse, buffalo, goat, dog, monkey",
    ],
}

# Generic words that name a category, not a species; not counted as results.
GENERIC = {
    "tree", "trees", "plant", "plants", "flower", "flowers", "herb", "herbs",
    "crop", "crops", "fruit", "fruits", "grass", "creeper", "vine", "leaf",
    "leaves", "bud", "blossom", "seed", "forest", "wood", "garden", "grain",
    "animal", "animals", "beast", "bird", "birds", "fish", "insect", "reptile",
    "creature", "mammal", "game", "cattle", "herd", "flock", "fowl", "prey",
    "bough", "branch", "root", "thorn", "weed", "shrub", "bush", "flesh",
    "young", "calf", "cub", "egg", "nest", "feather", "hide", "horn", "tusk",
}

EXTRACTION_PROMPT = """You are a careful annotator of Indian literature.
Read the PASSAGE and list every specific plant or animal that is named in it.

Rules:
- Include only names that literally appear in the passage. Do not guess.
- Use the singular form, as written in the passage (e.g. "lotus", "mango",
  "elephant", "cuckoo", "chataka bird").
- Skip generic words such as tree, flower, bird, animal, beast, fish.
- Skip metaphors where no real plant or animal is meant, and skip
  mythical beings (e.g. naga, garuda, gandharva).
- kingdom is "flora" for plants and "fauna" for animals.
- category is one of: tree, flower, herb_or_grass, crop_or_fruit,
  creeper (flora); mammal, bird, reptile, fish_or_aquatic, insect (fauna).

Return JSON: {{"items": [{{"name": "...", "kingdom": "...", "category": "..."}}]}}
Return {{"items": []}} if nothing is named.

PASSAGE (from {source}):
\"\"\"{passage}\"\"\"
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kingdom": {"type": "string", "enum": ["flora", "fauna"]},
                    "category": {"type": "string"},
                },
                "required": ["name", "kingdom", "category"],
            },
        }
    },
    "required": ["items"],
}


def ollama(endpoint, payload):
    req = urllib.request.Request(
        f"{OLLAMA}/api/{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


# ---- 1-2. Load and chunk -------------------------------------------------
# corpus file -> how its passages are labelled in the results
BOOKS = {
    "kalidasa_shakuntala_and_other_works": "Kalidasa",
    "jataka_buddhist_birth_stories": "Jataka Tales",
    "bankim_the_poison_tree": "Bankim Chandra Chattopadhyay, The Poison Tree",
    "naidu_the_golden_threshold": "Sarojini Naidu, The Golden Threshold",
    "tagore_the_gardener": "Rabindranath Tagore, The Gardener",
}

# Kalidasa's book holds seven works; each begins at a heading on a line of
# its own (after the contents page).
KALIDASA_WORKS = [
    ("SHAKUNTALA", "Shakuntala"),
    ("THE STORY OF SHAKUNTALA", "The Story of Shakuntala"),
    ("THE TWO MINOR DRAMAS", "Malavika and Agnimitra; Urvashi"),
    ("THE DYNASTY OF RAGHU", "The Dynasty of Raghu"),
    ("THE BIRTH OF THE WAR-GOD", "The Birth of the War-god"),
    ("THE CLOUD-MESSENGER", "The Cloud-Messenger"),
    ("THE SEASONS", "The Seasons"),
]


def split_kalidasa(text):
    """Return [(work title, work text)], skipping the introduction and contents."""
    lines = text.splitlines()
    body_start = next(i for i, l in enumerate(lines) if l.strip() == "A PLAY IN SEVEN ACTS") - 1
    starts = []
    for heading, title in KALIDASA_WORKS:
        idx = next(i for i, l in enumerate(lines)
                   if i >= body_start and l.strip() == heading
                   and all(i > s for s, _ in starts))
        starts.append((idx, f"Kalidasa, {title}"))
    starts.append((len(lines), None))
    return [(title, "\n".join(lines[s:e])) for (s, title), (e, _) in zip(starts, starts[1:])]


def load_chunks():
    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    for name, book in BOOKS.items():
        text = (CORPUS / f"{name}.txt").read_text(encoding="utf-8")
        parts = split_kalidasa(text) if name.startswith("kalidasa") else [(book, text)]
        for title, part in parts:
            words = part.split()
            for i in range(0, max(len(words) - OVERLAP_WORDS, 1), step):
                chunks.append({
                    "id": len(chunks),
                    "book": book,
                    "source": title,
                    "text": " ".join(words[i:i + CHUNK_WORDS]),
                })
    return chunks


# ---- 3. Embed (cached, so re-runs are fast) ------------------------------
def embed(texts, prefix):
    vectors = []
    for i in range(0, len(texts), 32):
        batch = [prefix + t for t in texts[i:i + 32]]
        vectors += ollama("embed", {"model": EMBED_MODEL, "input": batch})["embeddings"]
    v = np.array(vectors, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def chunk_vectors(chunks):
    cache = RESULTS / "chunk_embeddings.npy"
    if cache.exists():
        v = np.load(cache)
        if len(v) == len(chunks):
            return v
    v = embed([c["text"] for c in chunks], "search_document: ")
    np.save(cache, v)
    return v


# ---- 4. Retrieve ---------------------------------------------------------
def retrieve(chunks, vectors):
    """Top TOP_K passages per query within each book."""
    book_of = np.array([c["book"] for c in chunks])
    hits = {}
    for kingdom, queries in QUERIES.items():
        qv = embed(queries, "search_query: ")
        scores = qv @ vectors.T
        for q, row in zip(queries, scores):
            for book in BOOKS.values():
                idxs = np.where(book_of == book)[0]
                for idx in idxs[np.argsort(-row[idxs])[:TOP_K]]:
                    hits.setdefault(int(idx), []).append(q)
    return hits


# ---- 5. Generate ---------------------------------------------------------
def extract(chunk):
    prompt = EXTRACTION_PROMPT.format(source=chunk["source"], passage=chunk["text"])
    out = ollama("chat", {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "format": SCHEMA,
        "stream": False,
        "options": {"temperature": 0},
    })
    try:
        return json.loads(out["message"]["content"]).get("items", [])
    except json.JSONDecodeError:
        return []


# ---- 6. Verify -----------------------------------------------------------
def normalise(name):
    name = re.sub(r"[^a-z\- ]", "", name.lower()).strip()
    return re.sub(r"^(the|a|an) ", "", name)


def find_in_text(name, text):
    """Return the sentence in which `name` (or its plural) occurs, else None."""
    pattern = re.compile(rf"\b{re.escape(name)}(s|es)?\b", re.IGNORECASE)
    for sentence in re.split(r"(?<=[.!?;])\s+", text):
        if pattern.search(sentence):
            return sentence.strip()
    return None


def main():
    RESULTS.mkdir(exist_ok=True)
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} passages from {len(BOOKS)} books "
          f"({len(set(c['source'] for c in chunks))} texts)")

    vectors = chunk_vectors(chunks)
    hits = retrieve(chunks, vectors)
    print(f"Retrieved {len(hits)} unique passages for {sum(map(len, QUERIES.values()))} queries")

    found = defaultdict(lambda: {"kingdom": None, "categories": defaultdict(int),
                                 "books": set(), "sources": set(), "evidence": None})
    rejected = []
    for n, idx in enumerate(sorted(hits), 1):
        chunk = chunks[idx]
        for item in extract(chunk):
            name = normalise(item.get("name", ""))
            if not name or name in GENERIC or len(name) < 3:
                continue
            sentence = find_in_text(name, chunk["text"])
            if sentence is None:                      # hallucinated / not in text
                rejected.append({"name": name, "source": chunk["source"]})
                continue
            entry = found[name]
            entry["kingdom"] = entry["kingdom"] or item.get("kingdom")
            entry["categories"][item.get("category", "other")] += 1
            entry["books"].add(chunk["book"])
            entry["sources"].add(chunk["source"])
            entry["evidence"] = entry["evidence"] or {"source": chunk["source"],
                                                      "sentence": sentence}
        if n % 20 == 0:
            print(f"  processed {n}/{len(hits)} passages, {len(found)} names so far")

    records = []
    for name, e in sorted(found.items()):
        records.append({
            "name": name,
            "kingdom": e["kingdom"],
            "category": max(e["categories"], key=e["categories"].get),
            "books": sorted(e["books"]),
            "sources": sorted(e["sources"]),
            "evidence": e["evidence"],
        })

    (RESULTS / "extractions.json").write_text(
        json.dumps({"model": LLM_MODEL, "embedding_model": EMBED_MODEL,
                    "passages_retrieved": len(hits), "records": records,
                    "rejected_not_in_text": rejected},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Kept {len(records)} verified names; rejected {len(rejected)} not found in text")


if __name__ == "__main__":
    main()
