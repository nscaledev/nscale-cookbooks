# Dataset generator

The notebook runs `generate_corpus.py` automatically when `corpus/` is missing, so you don't need to run anything here.

To change the fictional household, edit `world.yaml`, delete `corpus/`, `questions.jsonl` and `cache/`, and re-run the notebook (or run `python dataset/generate_corpus.py` from the cookbook folder). Output is byte-identical for the same `world.yaml`.
