"""
GPUQuantile: Efficient Quantile Computation for Anomaly Detection

This package provides APIs and algorithms to efficiently compute quantiles for anomaly detection
in service and system logs. It implements multiple sketching algorithms optimized for:

- Low memory footprint
- Fast updates and queries
- Distributed computation support through mergeable sketches
- Accuracy guarantees for quantile approximation

The package includes two main implementations:

1. DDSketch: A deterministic algorithm with relative error guarantees
2. MomentSketch: A moment-based algorithm using maximum entropy optimization

Both implementations are designed to handle high-throughput data streams and provide
accurate quantile estimates with minimal memory overhead.
"""
from GPUQuantile.ddsketch.core import DDSketch
from GPUQuantile.momentsketch.core import MomentSketch

__version__ = "0.0.1"
__all__ = [
  "DDSketch",
  "MomentSketch"
]

if __name__ == "__main__":
    print("This is root of GPUQuantile module. API not to be exposed as a script!")