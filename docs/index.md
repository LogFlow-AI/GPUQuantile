# GPUQuantile

> A Python package for efficient quantile estimation in data streams and large datasets.

[![Latest Version on PyPI](https://img.shields.io/pypi/v/GPUQuantile.svg)](https://pypi.python.org/pypi/GPUQuantile/)
![Build Status](https://github.com/LogFlow-AI/GPUQuantile/actions/workflows/test.yaml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/GPUQuantile/badge/?version=latest)](https://GPUQuantile.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/LogFlow-AI/GPUQuantile/badge.svg?branch=main)](https://coveralls.io/github/LogFlow-AI/GPUQuantile?branch=main)
[![Built with PyPi Template](https://img.shields.io/badge/PyPi_Template-v0.8.0-blue.svg)](https://github.com/christophevg/pypi-template)

GPUQuantile provides algorithms for computing approximate quantiles with guaranteed error bounds. It is designed for efficiency in streaming data applications and anomaly detection in logs and time-series data.

## Key Features

- **Multiple Algorithms**: Includes both DDSketch and MomentSketch implementations
- **Memory Efficient**: Uses compact data structures regardless of data stream size
- **Mergeable**: Supports distributed processing by merging sketches
- **Accuracy Guarantees**: Provides configurable error bounds
- **Fast Operations**: O(1) insertions and efficient quantile queries
- **Python API**: Simple and intuitive interface for Python applications

## Contents

```{toctree}
:maxdepth: 2

whats-in-the-box.md
getting-started.md
examples.md
api/index.md
api/ddsketch.md
api/momentsketch.md
contributing.md
code.md
```


