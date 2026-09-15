"""TokenGS Early Reconstruction--Understanding dual-stream modules."""

from .types import TokenERUOutput
from .pair_adapter import ZeroInitPairAdapter
from .dual_stream_decoder import TokenGSEarlyDualStreamDecoder
from .dino_metric import (
    DINOUnitEvidence,
    FrozenDINOv2Extractor,
    HistoricalDINOUnitEncoder,
    DINOUnitFusion,
    MetricEmbeddingHead,
)
from .historical_unit_infonce import MetricLossOutput, historical_soft_unit_infonce
from .metric_clustering import (
    MetricClusterOutput,
    historical_metric_cluster,
    historical_metric_cluster_oracle_audit,
)
from .unit_3d_anchor import (
    FixedFourierPositionEncoding,
    Unit3DAnchor,
    Unit3DAnchorOutput,
    compute_opacity_weighted_unit_centers,
    normalize_unit_centers,
    reshape_child_gaussian_attributes,
)
from .query_metric_coupling import QueryMetricCoupling, QueryMetricCouplingOutput

__all__ = [
    "TokenERUOutput",
    "ZeroInitPairAdapter",
    "TokenGSEarlyDualStreamDecoder",
    "DINOUnitEvidence",
    "FrozenDINOv2Extractor",
    "HistoricalDINOUnitEncoder",
    "DINOUnitFusion",
    "MetricEmbeddingHead",
    "MetricLossOutput",
    "historical_soft_unit_infonce",
    "MetricClusterOutput",
    "historical_metric_cluster",
    "historical_metric_cluster_oracle_audit",
    "FixedFourierPositionEncoding",
    "Unit3DAnchor",
    "Unit3DAnchorOutput",
    "compute_opacity_weighted_unit_centers",
    "normalize_unit_centers",
    "reshape_child_gaussian_attributes",
    "QueryMetricCoupling",
    "QueryMetricCouplingOutput",
]
