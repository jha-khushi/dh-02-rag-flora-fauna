"""
Step 1 of the RAG pipeline: pre-processing the corpus.

For each raw Project Gutenberg download (corpus/raw_<author>_<title>.txt):
  1. remove the Gutenberg licence header and footer;
  2. keep only the literary text, cutting the editors' and translators'
     prefaces, introductions, indexes and notes (see BOOKS below);
  3. remove footnote markers such as [237], italics markers and
     illustration tags, and collapse extra blank lines;
  4. save the result as corpus/<author>_<title>.txt and delete the raw file.

Kalidasa's book keeps its translator's introduction here; rag_pipeline.py
skips it while splitting that book into its seven works.

Run:  python preprocess.py
"""

import re
from pathlib import Path

CORPUS = Path(__file__).parent / "corpus"

START = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*")
END = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*")

# file name -> (line where the literary text starts, which occurrence of that
#               line, line where it ends or None for the end of the book)
BOOKS = {
    # the publisher's list of other Everyman books follows the last poem
    "kalidasa_shakuntala_and_other_works": ("EVERYMAN'S LIBRARY", 1, "EVERYMAN'S LIBRARY"),
    "jataka_buddhist_birth_stories": ("BOOK I.", 1, "INDEX."),
    "bankim_the_poison_tree": ("CHAPTER I.", 2, "THE END."),
    "naidu_the_golden_threshold": ("FOLK SONGS", 2, None),
    "tagore_the_gardener": ("1", 1, "INDEX OF FIRST WORDS"),
}


def strip_gutenberg(text):
    text = text.replace("\r\n", "\n")
    start, end = START.search(text), END.search(text)
    if start and end:
        text = text[start.end():end.start()]
    return text


def keep_literary_text(name, text):
    if name not in BOOKS:
        return text
    start_line, occurrence, end_line = BOOKS[name]
    lines = text.splitlines()
    starts = [i for i, l in enumerate(lines) if l.strip() == start_line]
    s = starts[occurrence - 1]
    e = next((i for i, l in enumerate(lines) if i > s and l.strip() == end_line),
             len(lines))
    return "\n".join(lines[s:e])


def tidy(text):
    text = text.replace("_", "")                     # Gutenberg italics markers
    text = re.sub(r"\[Illustration[^\]]*\]", "", text)
    text = re.sub(r"\[Frontispiece[^\]]*\]", "", text)
    text = re.sub(r"\[\d+\]", "", text)              # footnote markers
    text = re.sub(r"\n{3,}", "\n\n", text)           # at most one blank line
    return text.strip() + "\n"


if __name__ == "__main__":
    for raw in sorted(CORPUS.glob("raw_*.txt")):
        name = raw.stem.removeprefix("raw_")
        text = strip_gutenberg(raw.read_text(encoding="utf-8"))
        cleaned = tidy(keep_literary_text(name, text))
        out = CORPUS / f"{name}.txt"
        out.write_text(cleaned, encoding="utf-8")
        raw.unlink()
        print(f"{out.name}: {len(cleaned.split()):,} words")
