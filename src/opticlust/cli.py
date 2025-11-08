from __future__ import annotations
import argparse, json, sys
import pandas as pd

from .pipeline import TextClusterer, EmbeddingConfig, ReductionConfig, ClusterConfig, ReportingConfig

def main():
    ap = argparse.ArgumentParser(description="Unsupervised text clustering and reporting")
    ap.add_argument("--input", "-i", required=True, help="Path to CSV or Parquet file")
    ap.add_argument("--text-col", default="text", help="Name of text column in the input DataFrame")
    ap.add_argument("--id-col", default=None, help="Optional id column (not used for clustering)")
    ap.add_argument("--output-prefix", "-o", default="out", help="Output prefix for CSV/JSON")

    # Embedding
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="Transformers model name")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto", help="Compute device")
    ap.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    ap.add_argument("--max-length", type=int, default=256, help="Max token length")

    # Reduction
    ap.add_argument("--var-ratio", type=float, default=0.95, help="PCA retained variance for clustering space")
    ap.add_argument("--max-components", type=int, default=256, help="Max PCA components")

    # Clustering
    ap.add_argument("--min-samples", type=int, default=5, help="DBSCAN min_samples")
    ap.add_argument("--metric", default="euclidean", help="DBSCAN metric")
    ap.add_argument("--eps-start", type=float, default=0.1, help="DBSCAN sweep start")
    ap.add_argument("--eps-end", type=float, default=5.0, help="DBSCAN sweep end")
    ap.add_argument("--eps-lr", type=float, default=0.1, help="DBSCAN sweep step")
    ap.add_argument("--use-tsne", action="store_true", help="Preproject with t-SNE before DBSCAN (overrides metric to euclidean)")
    ap.add_argument("--tsne-perplexity", type=float, default=30.0, help="t-SNE perplexity")
    ap.add_argument("--tsne-random-state", type=int, default=42, help="t-SNE random state")

    ap.add_argument("--top-representatives", type=int, default=5, help="Top representative examples per cluster in the report")

    args = ap.parse_args()

    # Read input
    if args.input.lower().endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    embedding = EmbeddingConfig(model_name=args.model, device=args.device, batch_size=args.batch_size, max_length=args.max_length)
    reduction = ReductionConfig(var_ratio=args.var_ratio, max_components=args.max_components)
    clustering = ClusterConfig(min_samples=args.min_samples, metric=args.metric,
                               eps_start=args.eps_start, eps_end=args.eps_end, eps_lr=args.eps_lr,
                               use_tsne=args.use_tsne, tsne_perplexity=args.tsne_perplexity,
                               tsne_random_state=args.tsne_random_state)
    reporting_cfg = ReportingConfig(top_representatives=args.top_representatives)

    tc = TextClusterer(embedding=embedding, reduction=reduction, clustering=clustering, reporting_cfg=reporting_cfg)
    fit = tc.fit(df, text_col=args.text_col, id_col=args.id_col)

    fit.result_df.to_csv(f"{args.output_prefix}_assignments.csv", index=False)
    with open(f"{args.output_prefix}_report.json", "w", encoding="utf-8") as f:
        json.dump(fit.report, f, ensure_ascii=False, indent=2)

    print(f"Saved {args.output_prefix}_assignments.csv and {args.output_prefix}_report.json")

if __name__ == "__main__":
    main()
