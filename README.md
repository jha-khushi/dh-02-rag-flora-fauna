# Assignment 02: Extraction of Indian Cultural Elements Using RAG

Names of Indian flora and fauna extracted from a repository of five Indian literary texts with a Retrieval-Augmented Generation (RAG) pipeline that runs entirely on a local machine.

**Result file:** [flora_fauna.py](flora_fauna.py) (`flora_RAG_extraction` and `fauna_RAG_extraction`)

## Corpus (`corpus/`, public domain, Project Gutenberg)
1. Kalidasa, *Shakuntala and Other Works*, translated by Arthur W. Ryder (1912)
2. *Buddhist Birth Stories (Jataka Tales)*, vol. 1, translated by T. W. Rhys Davids (1880)
3. Bankim Chandra Chattopadhyay, *The Poison Tree* (*Vishabriksha*), translated by Miriam S. Knight (1884)
4. Sarojini Naidu, *The Golden Threshold* (1905)
5. Rabindranath Tagore, *The Gardener* (1913)

## Pipeline
| Step | Script |
|---|---|
| 1. Pre-process: remove licence text, prefaces, introductions, indexes and footnotes | [preprocess.py](preprocess.py) |
| 2–6. Chunk, embed (nomic-embed-text), retrieve per book, generate (qwen2.5:7b), verify names against their passage | [rag_pipeline.py](rag_pipeline.py) |
| 7. Validate with the LLM plus recorded author corrections, then merge variants | [validate_results.py](validate_results.py) |
| 8. Write flora_fauna.py | [make_flora_fauna.py](make_flora_fauna.py) |

Every name has its evidence sentence in [results/validated.json](results/validated.json).

```bash
ollama pull nomic-embed-text && ollama pull qwen2.5:7b
python rag_pipeline.py && python validate_results.py && python make_flora_fauna.py
```

The five books are compared in [dh-06-comparing-books](https://github.com/jha-khushi/dh-06-comparing-books).

---

**Khushi Jha** · MA Digital Humanities · Department of Humanities and Social Sciences, IIT (ISM) Dhanbad · Monsoon Semester 2026
Portfolio: https://jha-khushi.github.io/digital-humanities-website/
