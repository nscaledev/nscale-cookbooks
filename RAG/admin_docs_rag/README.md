# Admin-docs RAG: answering questions about bills and contracts, and measuring what helps

A runnable cookbook that builds a question-answering assistant over a synthetic household's bills, invoices, contracts, insurance policies and letters, then benchmarks a ladder of retrieval techniques on 148 questions with known answers. The LLMs run on [Nscale serverless inference](https://www.nscale.com/product/serverless); everything else runs locally on a CPU-only laptop.

- **Read it**: [`admin_docs_rag.md`](admin_docs_rag.md) (source of truth, no outputs)
- **Run it**: [`admin_docs_rag.ipynb`](admin_docs_rag.ipynb) (paired with the Markdown by Jupytext, executed outputs included)
- **Results without running anything**: [`results/benchmark.csv`](results/benchmark.csv)

## What's inside

| File | Purpose |
| --- | --- |
| `admin_docs_rag.md` / `.ipynb` | The cookbook |
| `dataset/` | `world.yaml` (the fictional household) and the generator that renders it into `corpus/` (46 PDFs, 65 pages, 9 scans) and `questions.jsonl` (148 questions with answers and evidence pages). The notebook runs it on first use. |
| `docker-compose.yml` | Local Weaviate |
| `requirements.txt` | Pinned, CPU-only dependencies |
| `check_cookbook.py` | Structure checks for the `.md` / `.ipynb` pair |

## Setup (about 10 minutes)

Requirements: Python 3.12, Docker, 16 GB RAM, an Nscale account.

```bash
cd RAG/admin_docs_rag
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then paste your Nscale service token (Settings -> Service Tokens)
docker compose up -d          # starts Weaviate on localhost:8080
jupyter lab admin_docs_rag.ipynb
```

The first run downloads the embedding and reranker models (about 1.4 GB) and Docling's layout models.

## Runtime and cost

Measured on an Apple-silicon laptop, CPU only:

| Full run (148 questions) | |
| --- | --- |
| Time, first run | about 15 minutes, plus model downloads |
| Time, re-run with warm caches | about 10 minutes (mostly the CPU reranker) |
| Nscale cost, first run | $0.93 |
| Nscale cost, re-run | $0 (all LLM calls are cached) |

`QUICK_RUN = True` uses a 50-question subset and costs roughly a third as much; it was not timed separately. LLM responses, parsed documents and embeddings are cached in `cache/`.

## Headline results

| Configuration | Recall@5 | Answer correctness |
| --- | --- | --- |
| Hybrid search (BM25-weighted, alpha 0.25) | 0.897 | 0.886 |
| + metadata filters | 0.895 | |
| + cross-encoder reranker | 0.909 | |
| + contextual retrieval | 0.939 | 0.932 |
| + HyDE | 0.939 | |

Full table, with latency and cost: [`results/benchmark.csv`](results/benchmark.csv).

## Editing the cookbook

Edit `admin_docs_rag.md`, never the notebook, then sync and re-run:

```bash
jupytext --sync admin_docs_rag.md
jupyter nbconvert --to notebook --execute --inplace admin_docs_rag.ipynb
jupytext --sync admin_docs_rag.ipynb
python check_cookbook.py admin_docs_rag.md admin_docs_rag.ipynb
```
