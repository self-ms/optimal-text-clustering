# optimal-text-clustering

A practical, configurable toolkit for turning raw text into meaningful clusters with high‑quality reports.  It pipelines modern sentence embeddings, dimensionality reduction, and clustering (DBSCAN or K‑Means), then produces an audit‑friendly JSON/CSV report of clusters, exemplars, and per‑sample assignments.

This package is published on PyPI as **``optimal-text-clustering``** and is imported as ``opticlust``.
The distribution name reflects the goal of selecting **optimal** hyper‑parameters for clustering (see below).

> Note: The algorithm shines on medium to large datasets (thousands of texts). It works on small toy sets, but structure becomes clearer as data grows.

---

## Features

- **Embeddings**: Pluggable backends, including Sentence-Transformers (multilingual) and a lightweight TF‑IDF fallback.
- **Reduction**: PCA with adaptive component selection; optional 2D projection (PCA or t‑SNE) for visualization.
- **Clustering**: DBSCAN (with epsilon sweep) or K‑Means; cosine or euclidean metrics.
- **Reporting**: Per-cluster stats, top representatives, per-sample assignments, noise cluster handling, and export to JSON/CSV.
- **CLI + Python API**: Use it from the command line or integrate in Python notebooks/pipelines.
 - **CPU‑friendly defaults**: Works without a GPU. When available, can auto‑select GPU for faster embeddings.

### What does “Optimal” mean?

The term *optimal* here does **not** imply finding the mathematically global optimum across all possible clusterings—an intractable problem for arbitrary text data.  Instead, this package searches over a family of clustering algorithms and hyper‑parameter settings and **scores** each candidate using a blend of internal quality metrics and stability checks.  By default, the following metrics are combined:

- **Silhouette Score** (higher is better) – cohesion versus separation of clusters.
- **Calinski–Harabasz Score** (higher is better) – the ratio of between‑cluster dispersion to within‑cluster dispersion.
- **Davies–Bouldin Index** (lower is better) – average similarity between each cluster and its most similar peer.
- **Clustering Stability** (higher is better) – how consistent cluster assignments are under bootstrap resampling.

The package computes a weighted aggregate score for each configuration:

```
score = α * silhouette + β * CH - γ * DB + δ * stability
```

where `α`, `β`, `γ`, `δ` are tunable weights (default to `1`) and the Davies–Bouldin index is negated because lower values are better.  The configuration with the highest final score is chosen as “optimal.”  You can customise the metrics, weights, or optimisation strategy via the API.

In summary, *optimal* refers to selecting the best available parameter set **for a given dataset under the chosen scoring criteria**, not a universal optimum.

---

## Installation

### From PyPI (recommended)
```bash
pip install optimal-text-clustering
```

### Optional dependencies
If you want Sentence-Transformers and PyTorch for high‑quality multilingual embeddings:
```bash
pip install "optimal-text-clustering[embeddings]"
```
This installs dependencies like `transformers`, `sentence-transformers`, and `torch`.

> Without extras, the package falls back to TF‑IDF embeddings. This is fast but less semantically rich.

### From source
```bash
git clone https://github.com/your-org/optimal-text-clustering.git
cd optimal-text-clustering
pip install -e .
```

---

## Quickstart (Python)

Below is a complete example that you can run as-is. It constructs a small synthetic dataset, fits the pipeline, and prints the report and assignments.

```python
from opticlust import Text2Clusters, EmbeddingConfig, ReductionConfig, ClusterConfig, ReportingConfig
import pandas as pd

# Synthetic dataset for a quick demo
texts = [
    "Best pizza in town, the crust is amazing!",
    "I love this pizzeria. Great sauce and crispy base.",
    "Terrible service at the restaurant. Waited 40 minutes.",
    "The waiter was rude and the food arrived cold.",
    "The museum exhibition on impressionism was breathtaking.",
    "I enjoyed the modern art gallery, especially the sculptures.",
    "Football match was exciting, our team scored twice!",
    "The coach changed tactics and we won the game.",
    "New GPU benchmarks show impressive ray tracing performance.",
]

df = pd.DataFrame({"text": texts})

tc = Text2Clusters(
    embedding=EmbeddingConfig(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="auto",
        batch_size=8,
        max_length=256
    ),
    reduction=ReductionConfig(
        var_ratio=0.95,
        max_components=10,
        plot_2d_method="pca"   # "pca" or "tsne"
    ),
    clustering=ClusterConfig(
        method="dbscan",       # "dbscan" or "kmeans"
        min_samples=2,
        metric="euclidean",    # "euclidean" or "cosine"
        eps_start=0.1,
        eps_end=200.0,
        eps_lr=0.1,
        use_tsne=False         # set True to compute a 2D t-SNE projection
    ),
    reporting_cfg=ReportingConfig(
        top_representatives=5
    )
)

fit = tc.fit(df, text_col="text")
assignments = fit.result_df      # DataFrame with text, label, and any projections
report = fit.report              # JSON-like dict with cluster summary
embeddings = fit.embeddings      # np.ndarray of embedding vectors
reduced = fit.reduced            # np.ndarray of reduced components (e.g., PCA)

print("=== REPORT (summary) ===")
print(report)
print("\n=== ASSIGNMENTS (head) ===")
print(assignments.head())
```

### Using your own CSV
If you have a CSV with a `text` column:
```python
import pandas as pd
from opticlust import Text2Clusters, EmbeddingConfig, ReductionConfig, ClusterConfig, ReportingConfig

df = pd.read_csv("your_texts.csv")  # must contain a 'text' column

tc = Text2Clusters(
    embedding=EmbeddingConfig(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="auto",
        batch_size=8,
        max_length=256
    ),
    reduction=ReductionConfig(var_ratio=0.95, max_components=20, plot_2d_method="pca"),
    clustering=ClusterConfig(method="dbscan", min_samples=5, metric="cosine", eps_start=0.1, eps_end=50.0, eps_lr=0.2, use_tsne=False),
    reporting_cfg=ReportingConfig(top_representatives=5)
)

fit = tc.fit(df, text_col="text")
fit.result_df.to_csv("out_assignments.csv", index=False)
fit.save_report("out_report.json")
```

---

## Command Line Interface (CLI)

After installation, a console script (e.g., `opticlust`) should be available. Typical usage:

```bash
# Minimal run: read CSV, detect text column automatically if only one string column
opticlust fit \
  --input data.csv \
  --text-col text \
  --method dbscan \
  --out-assignments out_assignments.csv \
  --out-report out_report.json
```

More options:
```bash
opticlust fit \
  --input data.csv \
  --text-col text \
  --embedding-model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
  --device auto \
  --batch-size 16 \
  --max-length 256 \
  --reduction-var-ratio 0.95 \
  --reduction-max-components 20 \
  --plot-2d pca \
  --method dbscan \
  --metric cosine \
  --min-samples 5 \
  --eps-start 0.1 \
  --eps-end 50.0 \
  --eps-lr 0.2 \
  --use-tsne false \
  --top-representatives 5 \
  --out-assignments out_assignments.csv \
  --out-report out_report.json
```

If you prefer K‑Means:
```bash
opticlust fit \
  --input data.csv \
  --text-col text \
  --method kmeans \
  --k 8 \
  --metric cosine \
  --out-assignments out_assignments.csv \
  --out-report out_report.json
```

---

## API Reference (Configs)

### `EmbeddingConfig`
- `model_name: str` Sentence-Transformers model id. Use `"tfidf"` for TF‑IDF fallback.
- `device: str` `"auto"`, `"cpu"`, or specific like `"cuda:0"`.
- `batch_size: int` Embedding batch size.
- `max_length: int` Truncation length for transformer models (ignored for TF‑IDF).

### `ReductionConfig`
- `var_ratio: float` Target explained variance ratio for PCA; number of components is auto-chosen.
- `max_components: int` Hard cap on PCA components.
- `plot_2d_method: str` `"pca"` or `"tsne"` for 2D projection saved in `result_df`.

### `ClusterConfig`
- `method: str` `"dbscan"` or `"kmeans"`.
- `metric: str` `"euclidean"` or `"cosine"`.
- **DBSCAN-only**:
  - `min_samples: int` Minimum points to form a core point.
  - `eps_start, eps_end, eps_lr: float` Range and learning rate for epsilon sweep.
  - `use_tsne: bool` If `True`, compute a 2D t‑SNE projection for visualization (costly on large data).
- **K‑Means-only**:
  - `k: int | None` Number of clusters. If `None`, the algorithm may pick a heuristic (e.g., sqrt(n)).

### `ReportingConfig`
- `top_representatives: int` Number of exemplar texts per cluster in the report.
- Additional fields may include saving keyword summaries, noise cluster handling, etc., depending on version.

### `FitResult`
- `result_df: pd.DataFrame` One row per sample with columns like `text`, `label`, `pca_x`, `pca_y` or `tsne_x`, `tsne_y`.
- `report: dict` Cluster‑level statistics and exemplar texts.
- `embeddings: np.ndarray` High‑dimensional embeddings.
- `reduced: np.ndarray | None` PCA‑reduced array (if reduction enabled).

---

## Tips and Guidance

- **Scale matters**: structure becomes more reliable as you approach thousands of texts. With only dozens, expect less stable clusters.
- **Metric**: if you normalize embeddings (default when using cosine), try `metric="cosine"` for DBSCAN and K‑Means.
- **DBSCAN tuning**: sweep `eps` across a sensible range. Start narrow if your data is dense; widen for varied topics.
- **Hardware**: on CPU, transformer embeddings can be slow. If you have a GPU, set `device="cuda:0"` or keep `auto` and let the library decide.
- **Caching**: for repeated runs on the same data, consider caching embeddings to disk.
- **Reproducibility**: set `random_state` where available for deterministic behavior.

---

## Troubleshooting

- **ValueError: perplexity must be less than n_samples**  
  You asked for t‑SNE on too few samples. Lower `perplexity` (if configurable) or disable `use_tsne` until you have more data.
- **Out of memory during embedding**  
  Reduce `batch_size`, switch to a smaller model, or use TF‑IDF fallback (`model_name="tfidf"`).
- **All points labeled -1 (noise) in DBSCAN**  
  Increase `eps`, decrease `min_samples`, or switch to cosine distance.

---

## License

This project is released under the MIT License. See `LICENSE` for details.

---

## Changelog

See `CHANGELOG.md` for notable changes between releases.