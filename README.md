# Assignment 02: Extraction of Indian Cultural Elements Using RAG

Names of Indian flora and fauna extracted from a repository of five Indian literary texts with a Retrieval-Augmented Generation (RAG) pipeline that runs entirely on a local machine.

**Result file:** [flora_fauna.py](flora_fauna.py) (`flora_RAG_extraction` and `fauna_RAG_extraction`)

## Corpus (`corpus/`, public domain, Project Gutenberg)
1. Kalidasa, *Shakuntala and Other Works*, translated by Arthur W. Ryder (1912)
2. *Buddhist Birth Stories (Jataka Tales)*, vol. 1, translated by T. W. Rhys Davids (1880)
3. Bankim Chandra Chattopadhyay, *The Poison Tree* (*Vishabriksha*), translated by Miriam S. Knight (1884)
4. Sarojini Naidu, *The Golden Threshold* (1905)
5. Rabindranath Tagore, *The Gardener* (1913)

## RAG pipeline ([rag_pipeline.py](rag_pipeline.py))

1. **Chunk:** split each book into overlapping passages of about 250 words.
2. **Embed:** turn every passage into a vector with `nomic-embed-text`.
3. **Retrieve:** for each query about plants and animals (trees, flowers, birds, mammals and so on), take the 10 closest passages from each book.
4. **Generate:** `qwen2.5:7b` reads each retrieved passage and lists the plants and animals named in it.

Both models run locally through Ollama. Each name is kept with the sentence it came from in [results/validated.json](results/validated.json).

```bash
ollama pull nomic-embed-text && ollama pull qwen2.5:7b
python rag_pipeline.py
```

Charts: `python visualise.py` (writes [figures/](figures/)).

The five books are compared in [dh-06-comparing-books](https://github.com/jha-khushi/dh-06-comparing-books).

---

**Khushi Jha** · MA Digital Humanities · Department of Humanities and Social Sciences, IIT (ISM) Dhanbad · Monsoon Semester 2026
Portfolio: https://jha-khushi.github.io/digital-humanities-website/
