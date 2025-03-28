"""
GPUQuantile: Efficient Quantile Computation for Anomaly Detection

This package provides APIs and algorithms to efficiently compute quantiles for anomaly detection
in service and system logs. It implements multiple sketching algorithms optimized for:

- Low memory footprint
- Fast updates and queries
- Distributed computation support through mergeable sketches
- Accuracy guarantees for quantile approximation

The package includes three main implementations:

1. DDSketch: A deterministic algorithm with relative error guarantees
2. MomentSketch: A moment-based algorithm using maximum entropy optimization
3. HDRHistogram: A high dynamic range histogram for tracking values across wide ranges

All implementations are designed to handle high-throughput data streams and provide
accurate quantile estimates with minimal memory overhead.
"""
from GPUQuantile.ddsketch.core import DDSketch
from GPUQuantile.momentsketch.core import MomentSketch
from GPUQuantile.hdrhistogram.core import HDRHistogram

# Also import the main components for direct access
from GPUQuantile.ddsketch import (
    LogarithmicMapping,
    LinearInterpolationMapping,
    CubicInterpolationMapping,
    ContiguousStorage,
    SparseStorage,
    BucketManagementStrategy
)

__version__ = "0.0.4"
__all__ = [
    "DDSketch",
    "MomentSketch",
    "HDRHistogram",
    "LogarithmicMapping",
    "LinearInterpolationMapping",
    "CubicInterpolationMapping",
    "ContiguousStorage",
    "SparseStorage",
    "BucketManagementStrategy"
]

if __name__ == "__main__":
    print("This is root of GPUQuantile module. API not to be exposed as a script!")