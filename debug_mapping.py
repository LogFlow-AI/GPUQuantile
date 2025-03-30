"""
Debug script for DDSketch mappings and quantile calculations.

This script provides diagnostic tools to test and verify the behavior of various
mapping schemes used in the DDSketch implementation:
- LogarithmicMapping: The canonical implementation using logarithmic buckets
- LinearInterpolationMapping: An approximation using linear interpolation between powers of 2
- CubicInterpolationMapping: An approximation using cubic interpolation for improved accuracy

For each mapping scheme, the script:
1. Tests the bucket index computation and value reconstruction
2. Checks that relative error guarantees are maintained
3. Examines behavior with extreme values
4. Verifies quantile calculations with a simple test dataset

Usage: python debug_mapping.py
"""

import numpy as np
from GPUQuantile.ddsketch.mapping.logarithmic import LogarithmicMapping
from GPUQuantile.ddsketch.mapping.linear_interpolation import LinearInterpolationMapping
from GPUQuantile.ddsketch.mapping.cubic_interpolation import CubicInterpolationMapping
from GPUQuantile.ddsketch.core import DDSketch

def test_mapping(mapping_class, alpha=0.01):
    """Test a mapping scheme."""
    print(f"\nTesting {mapping_class.__name__} with alpha={alpha}")
    mapping = mapping_class(alpha)
    
    # Test various values
    test_values = [0.1, 1.0, 5.0, 10.0, 100.0, 1000.0]
    for value in test_values:
        bucket = mapping.compute_bucket_index(value)
        back_value = mapping.compute_value_from_index(bucket)
        rel_error = abs(back_value - value) / value
        print(f"Value: {value:10.4g} -> Bucket: {bucket:4d} -> Value: {back_value:10.4g}, Rel Error: {rel_error:8.6f}")
        # Check if error is within alpha
        if rel_error > alpha:
            print(f"  WARNING: Relative error {rel_error} exceeds alpha {alpha}")
            
    # Test extreme values
    print("\nExtreme values:")
    for value in [1e-10, 1e-5, 1e5, 1e10]:
        bucket = mapping.compute_bucket_index(value)
        back_value = mapping.compute_value_from_index(bucket)
        rel_error = abs(back_value - value) / value
        print(f"Value: {value:10.4g} -> Bucket: {bucket:8d} -> Value: {back_value:10.4g}, Rel Error: {rel_error:8.6f}")

def test_sketch():
    """Test DDSketch quantile calculations."""
    print("\nTesting DDSketch with LogarithmicMapping")
    sketch = DDSketch(relative_accuracy=0.01, mapping_type='logarithmic')
    
    # Insert values
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    for v in values:
        sketch.insert(v)
        
    # Test quantiles
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    print("\nQuantile calculations:")
    for q in quantiles:
        true_q = np.quantile(values, q)
        approx_q = sketch.quantile(q)
        rel_error = abs(approx_q - true_q) / true_q if true_q != 0 else 0
        print(f"Quantile {q:.2f}: True={true_q:.4f}, Approx={approx_q:.4f}, Rel Error={rel_error:.6f}")
        
    # Print internal state
    print("\nInternal state:")
    print(f"Count: {sketch.count}")
    print(f"Min: {sketch.min_value}, Max: {sketch.max_value}")
    
    # Print buckets
    print("\nPositive store buckets:")
    for idx, count in sketch._get_specific_storage_items(sketch.positive_store):
        value = sketch.mapping.compute_value_from_index(idx)
        print(f"Bucket {idx}: Count={count}, Value={value:.4f}")

if __name__ == "__main__":
    # Test all mappings
    test_mapping(LogarithmicMapping)
    test_mapping(LinearInterpolationMapping)
    test_mapping(CubicInterpolationMapping)
    
    # Test sketch
    test_sketch() 