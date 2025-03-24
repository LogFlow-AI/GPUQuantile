# API Reference

This section provides detailed documentation for all the public classes and methods available in the GPUQuantile package.

## Core Components

### DDSketch

```{eval-rst}
.. autoclass:: GPUQuantile.DDSketch
   :members:
   :undoc-members:
   :show-inheritance:
```

### MomentSketch

```{eval-rst}
.. autoclass:: GPUQuantile.MomentSketch
   :members:
   :undoc-members:
   :show-inheritance:
```

## Mapping Classes

These classes implement different mapping strategies for DDSketch:

```{eval-rst}
.. autoclass:: GPUQuantile.LogarithmicMapping
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: GPUQuantile.LinearInterpolationMapping
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: GPUQuantile.CubicInterpolationMapping
   :members:
   :undoc-members:
   :show-inheritance:
```

## Storage Classes

These classes implement different storage strategies for DDSketch:

```{eval-rst}
.. autoclass:: GPUQuantile.ContiguousStorage
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: GPUQuantile.SparseStorage
   :members:
   :undoc-members:
   :show-inheritance:
```

## Utility Classes

```{eval-rst}
.. autoclass:: GPUQuantile.BucketManagementStrategy
   :members:
   :undoc-members:
   :show-inheritance:
``` 