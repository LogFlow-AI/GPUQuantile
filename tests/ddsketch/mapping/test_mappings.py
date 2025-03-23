import pytest
import numpy as np
from GPUQuantile.ddsketch.mapping.logarithmic import LogarithmicMapping
from GPUQuantile.ddsketch.mapping.linear_interpolation import LinearInterpolationMapping
from GPUQuantile.ddsketch.mapping.cubic_interpolation import CubicInterpolationMapping

@pytest.fixture(params=[0.01, 0.001, 0.1])
def relative_accuracy(request):
    return request.param

@pytest.fixture(params=[
    LogarithmicMapping,
    LinearInterpolationMapping,
    CubicInterpolationMapping
])
def mapping_class(request):
    return request.param

def test_mapping_initialization(mapping_class, relative_accuracy):
    mapping = mapping_class(relative_accuracy)
    assert mapping.relative_accuracy == relative_accuracy

def test_bucket_index_monotonicity(mapping_class, relative_accuracy):
    mapping = mapping_class(relative_accuracy)
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    indices = [mapping.compute_bucket_index(v) for v in values]
    
    # Check that indices are monotonically increasing
    assert all(indices[i] <= indices[i+1] for i in range(len(indices)-1))

def test_value_reconstruction(mapping_class, relative_accuracy):
    mapping = mapping_class(relative_accuracy)
    test_values = [0.1, 1.0, 10.0, 100.0]
    
    for value in test_values:
        bucket_index = mapping.compute_bucket_index(value)
        reconstructed = mapping.compute_value_from_index(bucket_index)
        
        # Check relative error guarantee
        relative_error = abs(reconstructed - value) / value
        assert relative_error <= relative_accuracy

def test_negative_values(mapping_class, relative_accuracy):
    mapping = mapping_class(relative_accuracy)
    
    # Negative values should raise ValueError
    with pytest.raises(ValueError):
        mapping.compute_bucket_index(-1.0)

def test_zero_value(mapping_class, relative_accuracy):
    mapping = mapping_class(relative_accuracy)
    
    # Zero should raise ValueError
    with pytest.raises(ValueError):
        mapping.compute_bucket_index(0.0)

def test_extreme_values(mapping_class, relative_accuracy):
    mapping = mapping_class(relative_accuracy)
    
    # Test very small and very large values
    small_value = 1e-100
    large_value = 1e100
    
    small_index = mapping.compute_bucket_index(small_value)
    large_index = mapping.compute_bucket_index(large_value)
    
    # Indices should be different
    assert small_index < large_index
    
    # Reconstructed values should maintain relative accuracy
    small_reconstructed = mapping.compute_value_from_index(small_index)
    large_reconstructed = mapping.compute_value_from_index(large_index)
    
    assert abs(small_reconstructed - small_value) / small_value <= relative_accuracy
    assert abs(large_reconstructed - large_value) / large_value <= relative_accuracy

def test_consecutive_buckets(mapping_class, relative_accuracy):
    mapping = mapping_class(relative_accuracy)
    
    # Test that consecutive bucket indices give values within relative accuracy
    value = 1.0
    index = mapping.compute_bucket_index(value)
    
    next_value = mapping.compute_value_from_index(index + 1)
    prev_value = mapping.compute_value_from_index(index - 1)
    
    # Check that consecutive buckets maintain relative accuracy bounds
    assert next_value / value <= 1 + relative_accuracy
    assert value / prev_value <= 1 + relative_accuracy

def test_specific_mapping_features():
    """Test specific features of each mapping type"""
    
    # Test LogarithmicMapping
    log_mapping = LogarithmicMapping(0.01)
    assert hasattr(log_mapping, 'gamma')
    assert log_mapping.gamma > 1
    
    # Test LinearInterpolationMapping
    lin_mapping = LinearInterpolationMapping(0.01)
    assert hasattr(lin_mapping, 'interpolation_multiplier')
    
    # Test CubicInterpolationMapping
    cubic_mapping = CubicInterpolationMapping(0.01)
    assert hasattr(cubic_mapping, 'coefficients')
    
    # Test that each mapping type gives different results
    test_value = 2.0
    log_index = log_mapping.compute_bucket_index(test_value)
    lin_index = lin_mapping.compute_bucket_index(test_value)
    cubic_index = cubic_mapping.compute_bucket_index(test_value)
    
    # The indices should be different due to different mapping strategies
    assert len({log_index, lin_index, cubic_index}) > 1

def test_mapping_consistency(mapping_class, relative_accuracy):
    """Test that mapping is consistent across multiple calls"""
    mapping = mapping_class(relative_accuracy)
    
    # Test multiple calls with same value
    value = 1.234
    indices = [mapping.compute_bucket_index(value) for _ in range(10)]
    
    # All indices should be identical
    assert len(set(indices)) == 1
    
    # Test reconstruction is consistent
    reconstructions = [mapping.compute_value_from_index(indices[0]) for _ in range(10)]
    
    # All reconstructions should be identical
    assert len(set(reconstructions)) == 1 