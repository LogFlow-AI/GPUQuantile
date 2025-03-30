import pytest
import numpy as np
from GPUQuantile.ddsketch.core import DDSketch
from GPUQuantile.ddsketch.storage.base import BucketManagementStrategy

def test_ddsketch_initialization():
    # Test valid initialization
    sketch = DDSketch(relative_accuracy=0.01)
    assert sketch.relative_accuracy == 0.01
    assert sketch.cont_neg is True
    
    # Test invalid relative accuracy
    with pytest.raises(ValueError):
        DDSketch(relative_accuracy=0)
    with pytest.raises(ValueError):
        DDSketch(relative_accuracy=1)
    with pytest.raises(ValueError):
        DDSketch(relative_accuracy=-0.1)

def test_insert_positive():
    sketch = DDSketch(relative_accuracy=0.01)
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    for v in values:
        sketch.insert(v)
    assert sketch.count == len(values)
    
    # Test median (should be approximately 3.0)
    assert abs(sketch.quantile(0.5) - 3.0) <= 3.0 * 0.01  # Within relative accuracy

def test_insert_negative():
    sketch = DDSketch(relative_accuracy=0.01)
    values = [-1.0, -2.0, -3.0, -4.0, -5.0]
    for v in values:
        sketch.insert(v)
    assert sketch.count == len(values)
    
    # Test median (should be approximately -3.0)
    assert abs(sketch.quantile(0.5) - (-3.0)) <= abs(-3.0) * 0.01

def test_insert_mixed():
    sketch = DDSketch(relative_accuracy=0.01)
    values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    for v in values:
        sketch.insert(v)
    assert sketch.count == len(values)
    assert sketch.zero_count == 1
    
    # Test median (should be approximately 0.0)
    assert abs(sketch.quantile(0.5)) <= 0.01

def test_negative_values_disabled():
    sketch = DDSketch(relative_accuracy=0.01, cont_neg=False)
    sketch.insert(1.0)  # Should work
    with pytest.raises(ValueError):
        sketch.insert(-1.0)

def test_delete():
    sketch = DDSketch(relative_accuracy=0.01)
    values = [1.0, 2.0, 2.0, 3.0]
    for v in values:
        sketch.insert(v)
    
    sketch.delete(2.0)
    assert sketch.count == len(values) - 1
    
    # Delete non-existent value (should not affect count)
    sketch.delete(10.0)
    assert sketch.count == len(values) - 1

def test_quantile_edge_cases():
    sketch = DDSketch(relative_accuracy=0.01)
    
    # Empty sketch
    with pytest.raises(ValueError):
        sketch.quantile(0.5)
    
    # Invalid quantile values
    sketch.insert(1.0)
    with pytest.raises(ValueError):
        sketch.quantile(-0.1)
    with pytest.raises(ValueError):
        sketch.quantile(1.1)

def test_merge():
    sketch1 = DDSketch(relative_accuracy=0.01)
    sketch2 = DDSketch(relative_accuracy=0.01)

    # Generate Pareto distribution with shape parameter a=3 (finite variance)
    np.random.seed(42)
    values = (1 / (1 - np.random.random(1000)) ** (1/3))  # Inverse CDF method for Pareto
    values = np.sort(values)  # Sort to make splitting deterministic
    median_idx = len(values) // 2
    true_median = values[median_idx]

    # Split values between sketches
    for v in values[:median_idx]:
        sketch1.insert(v)
    for v in values[median_idx:]:
        sketch2.insert(v)

    # Merge sketch2 into sketch1
    sketch1.merge(sketch2)
    assert sketch1.count == len(values)

    # Test median of merged sketch
    assert abs(sketch1.quantile(0.5) - true_median) <= true_median * 0.01

    # Also test other quantiles
    q1_idx = len(values) // 4
    q3_idx = 3 * len(values) // 4
    true_q1 = values[q1_idx]
    true_q3 = values[q3_idx]
    assert abs(sketch1.quantile(0.25) - true_q1) <= true_q1 * 0.01  # Q1
    assert abs(sketch1.quantile(0.75) - true_q3) <= true_q3 * 0.01  # Q3

def test_merge_incompatible():
    sketch1 = DDSketch(relative_accuracy=0.01)
    sketch2 = DDSketch(relative_accuracy=0.02)
    
    with pytest.raises(ValueError):
        sketch1.merge(sketch2)

def test_different_mapping_types():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Test all mapping types
    for mapping_type in ['logarithmic', 'lin_interpol', 'cub_interpol']:
        sketch = DDSketch(relative_accuracy=0.01, mapping_type=mapping_type)
        for v in values:
            sketch.insert(v)
        
        # Verify median
        assert abs(sketch.quantile(0.5) - 3.0) <= 3.0 * 0.01

def test_different_storage_types():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Test both storage strategies
    for strategy in [BucketManagementStrategy.FIXED, BucketManagementStrategy.COLLAPSE]:
        sketch = DDSketch(relative_accuracy=0.01, bucket_strategy=strategy)
        for v in values:
            sketch.insert(v)
        
        # Verify median
        assert abs(sketch.quantile(0.5) - 3.0) <= 3.0 * 0.01

def test_extreme_values():
    sketch = DDSketch(relative_accuracy=0.01)
    
    # Test very large and very small positive values
    sketch.insert(1e-100)
    sketch.insert(1e100)
    
    # Should handle these values without issues
    assert sketch.count == 2
    
    # Test quantiles
    assert sketch.quantile(0) > 0  # Should be close to 1e-100
    assert sketch.quantile(1) < float('inf')  # Should be close to 1e100

def test_accuracy_guarantee():
    # Test that the relative error guarantee is maintained
    sketch = DDSketch(relative_accuracy=0.01)
    
    # Generate log-normal distribution
    np.random.seed(42)
    values = np.random.lognormal(0, 1, 1000)
    
    # Insert values
    for v in values:
        sketch.insert(v)
    
    # Test various quantiles
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        true_quantile = np.quantile(values, q)
        approx_quantile = sketch.quantile(q)
        
        # Verify relative error guarantee
        relative_error = abs(approx_quantile - true_quantile) / true_quantile
        assert relative_error <= 0.01, f"Relative error exceeded at q={q}" 