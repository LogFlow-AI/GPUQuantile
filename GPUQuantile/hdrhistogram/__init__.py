"""
High Dynamic Range Histogram implementation for quantile approximation.

This module provides an efficient implementation of HDR Histogram for tracking values
across a wide range using logarithmic bucketing. It supports streaming data and
provides accurate quantile estimates with configurable precision.
"""

from .core import HDRHistogram

__all__ = ['HDRHistogram'] 