# Benchmark Experiment Log

This document records all experiment configurations, tuned parameters, and results across BEIR SciFact and NFCorpus.

## SciFact Experiment Run
- **Run ID**: `scifact_20261026_135813`
- **Corpus Size**: 300 test queries evaluated
- **Tuned RRF k**: `10` (Sweep: `{'10': 0.7427290468309433, '30': 0.7319610629150142, '60': 0.7298482659763659, '100': 0.7275262984673615}`)
- **Tuned BM25 Weight**: `0.5`
- **Tuned Score Fusion Alpha**: `0.3`

### SciFact Results Summary
| System | nDCG@10 | Recall@100 | MRR@10 | HitRate@10 | Precision@10 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `bm25` | 0.6539 | 0.8764 | 0.6211 | 0.7967 | 0.0853 |
| `dense_exact` | 0.7200 | 0.9533 | 0.6845 | 0.8567 | 0.0953 |
| `dense_hnsw` | 0.7167 | 0.9433 | 0.6812 | 0.8533 | 0.0950 |
| `hybrid_rrf_default` | 0.7109 | 0.9650 | 0.6829 | 0.8333 | 0.0910 |
| `hybrid_rrf_tuned` | 0.7258 | 0.9650 | 0.6904 | 0.8733 | 0.0953 |
| `hybrid_weighted_rrf` | 0.7270 | 0.9683 | 0.6889 | 0.8700 | 0.0963 |
| `hybrid_score_fusion` | 0.7444 | 0.9683 | 0.7096 | 0.8833 | 0.0977 |
| `bm25+rerank_N20` | 0.6847 | 0.8764 | 0.6587 | 0.8200 | 0.0883 |
| `bm25+rerank_N50` | 0.6810 | 0.8764 | 0.6528 | 0.8133 | 0.0893 |
| `bm25+rerank_N100` | 0.6753 | 0.8764 | 0.6498 | 0.8033 | 0.0877 |
| `dense+rerank_N20` | 0.7011 | 0.9533 | 0.6667 | 0.8400 | 0.0937 |
| `dense+rerank_N50` | 0.7001 | 0.9533 | 0.6682 | 0.8367 | 0.0927 |
| `dense+rerank_N100` | 0.6957 | 0.9533 | 0.6650 | 0.8367 | 0.0923 |
| `hybrid_rrf_tuned+rerank_N20` | 0.7066 | 0.9650 | 0.6710 | 0.8633 | 0.0950 |
| `hybrid_rrf_tuned+rerank_N50` | 0.6977 | 0.9650 | 0.6649 | 0.8433 | 0.0933 |
| `hybrid_rrf_tuned+rerank_N100` | 0.6896 | 0.9650 | 0.6592 | 0.8267 | 0.0913 |

## NFCorpus Experiment Run
- **Run ID**: `nfcorpus_20261027_071419`
- **Corpus Size**: 323 test queries evaluated
- **Tuned RRF k**: `10` (Sweep: `{'10': 0.3273302912095045, '30': 0.3260299959781593, '60': 0.32374446950149005, '100': 0.3226552359137673}`)
- **Tuned BM25 Weight**: `0.5`
- **Tuned Score Fusion Alpha**: `0.4`

### NFCorpus Results Summary
| System | nDCG@10 | Recall@100 | MRR@10 | HitRate@10 | Precision@10 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `bm25` | 0.3043 | 0.2351 | 0.5068 | 0.6873 | 0.2170 |
| `dense_exact` | 0.3371 | 0.3058 | 0.5284 | 0.6997 | 0.2505 |
| `dense_hnsw` | 0.3301 | 0.3045 | 0.5220 | 0.6904 | 0.2474 |
| `hybrid_rrf_default` | 0.3546 | 0.3140 | 0.5667 | 0.7245 | 0.2576 |
| `hybrid_rrf_tuned` | 0.3563 | 0.3140 | 0.5743 | 0.7399 | 0.2570 |
| `hybrid_weighted_rrf` | 0.3530 | 0.3150 | 0.5529 | 0.7307 | 0.2604 |
| `hybrid_score_fusion` | 0.3549 | 0.3180 | 0.5607 | 0.7337 | 0.2579 |
| `bm25+rerank_N20` | 0.3315 | 0.2351 | 0.5613 | 0.6873 | 0.2254 |
| `bm25+rerank_N50` | 0.3424 | 0.2351 | 0.5673 | 0.7028 | 0.2381 |
| `bm25+rerank_N100` | 0.3425 | 0.2351 | 0.5672 | 0.6997 | 0.2384 |
| `dense+rerank_N20` | 0.3526 | 0.3058 | 0.5725 | 0.7090 | 0.2554 |
| `dense+rerank_N50` | 0.3555 | 0.3058 | 0.5751 | 0.7276 | 0.2533 |
| `dense+rerank_N100` | 0.3490 | 0.3058 | 0.5681 | 0.7090 | 0.2474 |
| `hybrid_rrf_tuned+rerank_N20` | 0.3638 | 0.3140 | 0.5840 | 0.7399 | 0.2591 |
| `hybrid_rrf_tuned+rerank_N50` | 0.3582 | 0.3140 | 0.5830 | 0.7245 | 0.2517 |
| `hybrid_rrf_tuned+rerank_N100` | 0.3554 | 0.3140 | 0.5785 | 0.7276 | 0.2505 |