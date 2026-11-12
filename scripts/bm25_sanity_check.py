"""BM25 sanity check: Compare custom BM25 on SciFact and NFCorpus against published BEIR baselines."""

from rag_bench.data import load_beir_dataset
from rag_bench.metrics import evaluate_run
from rag_bench.retrievers.bm25 import BM25Retriever

# Published BEIR paper numbers (Thakur et al. 2021, Table 2 / Table 3):
# SciFact BM25 nDCG@10: ~0.665 (Lucene / Anserini BM25)
# NFCorpus BM25 nDCG@10: ~0.325 (Lucene / Anserini BM25)
PUBLISHED_BEIR = {
    "scifact": {
        "published_ndcg10": 0.665,
        "source": "Thakur et al., 'BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models', NeurIPS 2021",
    },
    "nfcorpus": {
        "published_ndcg10": 0.325,
        "source": "Thakur et al., 'BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models', NeurIPS 2021",
    },
}


def run_sanity_check():
    print("=" * 60)
    print("RUNNING BM25 BASELINE SANITY CHECK AGAINST PUBLISHED BEIR NUMBERS")
    print("=" * 60)

    for dataset_name in ["scifact", "nfcorpus"]:
        print(f"\n--- Loading {dataset_name} ---")
        dataset = load_beir_dataset(dataset_name)
        print(f"Corpus size: {len(dataset.corpus)} docs")
        print(f"Test queries: {len(dataset.test_qrels)} queries")
        print(f"Checksums: {dataset.checksums}")

        print("Building BM25 sparse index...")
        retriever = BM25Retriever(k1=1.2, b=0.75)
        retriever.index(dataset.corpus)

        print(f"Running BM25 search on {len(dataset.test_qrels)} test queries...")
        test_queries = {qid: dataset.queries[qid] for qid in dataset.test_qrels if qid in dataset.queries}
        run_results = retriever.batch_search(test_queries, top_k=100)

        mean_metrics, _ = evaluate_run(dataset.test_qrels, run_results)
        our_ndcg = mean_metrics["ndcg@10"]
        published_ndcg = PUBLISHED_BEIR[dataset_name]["published_ndcg10"]
        abs_diff = our_ndcg - published_ndcg
        rel_diff_pct = (abs_diff / published_ndcg) * 100.0

        print(f"\nResults for {dataset_name}:")
        print(f"  Custom BM25 nDCG@10:    {our_ndcg:.4f}")
        print(f"  Published BEIR nDCG@10: {published_ndcg:.4f}")
        print(f"  Difference:             {abs_diff:+.4f} ({rel_diff_pct:+.2f}%)")
        print(f"  Recall@100:             {mean_metrics['recall@100']:.4f}")
        print(f"  Reference Citation:     {PUBLISHED_BEIR[dataset_name]['source']}")

        # Verify sanity margin (allow small deviation due to Lucene Porter stemming vs regex whitespace)
        # Typically within ~0.05-0.08 points
        if abs(abs_diff) > 0.10:
            print(f"WARNING: Large deviation detected on {dataset_name}: {abs_diff:+.4f}")
        else:
            print(f"SANITY CHECK PASSED: {dataset_name} is within expected margin of published baseline.")


if __name__ == "__main__":
    run_sanity_check()
