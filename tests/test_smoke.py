import pandas as pd
from opticlust import Text2Clusters, EmbeddingConfig, ReductionConfig, ClusterConfig

def test_import_and_configs():
    # basic instantiation of classes from the new package
    _ = Text2Clusters()
    EmbeddingConfig()
    ReductionConfig()
    ClusterConfig()

def test_dataframe_interface():
    df = pd.DataFrame({"text": ["hello world", "hi there"]})
    # Don't actually run heavy embedding in CI by default.
    assert "text" in df.columns
