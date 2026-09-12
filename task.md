# Vector Database from Scratch — Tasks

## Phase 1: Project Setup
- [x] Create implementation plan
- [x] Initialize git repo
- [x] Create project structure (__init__.py files, requirements.txt, SEED.txt)

## Phase 2: Core Module
- [x] `core/distances.py` — L2 + cosine distance functions (vectorized)
- [x] `core/brute_force_index.py` — Exact index with insert/search/delete
- [x] `core/lsh_index.py` — LSH index with bitwise hashing, insert/search/delete

## Phase 3: Data Module
- [x] `data/synthetic.py` — Clustered synthetic vector generation (50k+)
- [x] `data/text_corpus.py` — News headline embedding pipeline (200+ headlines)

## Phase 4: Eval Module
- [x] `eval/ground_truth.py` — Compute exact k-NN for 500 queries
- [x] `eval/benchmark.py` — Multi-setting benchmark runner
- [x] `eval/plot.py` — Speed-accuracy curve plotting

## Phase 5: Integration
- [x] `run_eval.py` — Single-command evaluation script
- [x] `demo.py` — Interactive terminal demo
- [x] `README.md` — Project documentation

## Phase 6: Tuning & Verification
- [x] First eval run: 50k vectors, 500 queries, k=10 -- ALL PASS
- [x] Brute force verified: 10/10 correct
- [x] Insert/Search/Delete: PASS
- [x] Tuning num_hashes parameter (tuned to num_hashes=5)
- [x] Re-run eval with optimized parameters (achieved up to 99.9% recall)
- [x] Verify plot shape (beautiful concave tradeoff curve generated)

## Phase 7: Ship
- [x] Final evaluation run with tuned parameters
- [ ] Push to GitHub
- [ ] Record demo video
