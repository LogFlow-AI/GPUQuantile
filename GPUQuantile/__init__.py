"""
  This package provides API and functionality to efficiently compute quantiles for anomaly detection in service/system logs.
"""

__version__ = "0.1.0"

from .ddsketch.core import DDSketch

__all__ = ['DDSketch']

if __name__ == "__main__":
    print("This is root of GPUQuantile module. API not to be exposed as a script!")