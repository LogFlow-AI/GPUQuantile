import pytest
from GPUQuantile.ddsketch.storage.base import BucketManagementStrategy
from GPUQuantile.ddsketch.storage.contiguous import ContiguousStorage
from GPUQuantile.ddsketch.storage.sparse import SparseStorage

@pytest.fixture(params=[ContiguousStorage, SparseStorage])
def storage_class(request):
    return request.param

@pytest.fixture(params=[32, 64, 128])
def max_buckets(request):
    return request.param

@pytest.fixture(params=[
    BucketManagementStrategy.FIXED,
    BucketManagementStrategy.COLLAPSE
])
def bucket_strategy(request):
    return request.param

def test_storage_initialization(storage_class, max_buckets, bucket_strategy):
    if storage_class == ContiguousStorage and bucket_strategy != BucketManagementStrategy.FIXED:
        pytest.skip("ContiguousStorage only supports FIXED strategy")
    
    storage = storage_class(max_buckets, bucket_strategy)
    assert storage.max_buckets == max_buckets
    if hasattr(storage, 'strategy'):
        assert storage.strategy == bucket_strategy

def test_add_and_get_count(storage_class, max_buckets, bucket_strategy):
    if storage_class == ContiguousStorage and bucket_strategy != BucketManagementStrategy.FIXED:
        pytest.skip("ContiguousStorage only supports FIXED strategy")
    
    storage = storage_class(max_buckets, bucket_strategy)
    
    # Add counts to some buckets
    test_buckets = {0: 1, 5: 3, 10: 2}
    for bucket, count in test_buckets.items():
        for _ in range(count):
            storage.add(bucket)
    
    # Verify counts
    for bucket, expected_count in test_buckets.items():
        assert storage.get_count(bucket) == expected_count
    
    # Test non-existent bucket
    assert storage.get_count(999) == 0

def test_remove(storage_class, max_buckets, bucket_strategy):
    if storage_class == ContiguousStorage and bucket_strategy != BucketManagementStrategy.FIXED:
        pytest.skip("ContiguousStorage only supports FIXED strategy")
    
    storage = storage_class(max_buckets, bucket_strategy)
    
    # Add and remove counts
    bucket_idx = 5
    storage.add(bucket_idx)
    storage.add(bucket_idx)
    storage.remove(bucket_idx)
    
    assert storage.get_count(bucket_idx) == 1
    
    # Remove from empty bucket
    storage.remove(999)  # Should not raise error
    assert storage.get_count(999) == 0

def test_merge(storage_class, max_buckets, bucket_strategy):
    if storage_class == ContiguousStorage and bucket_strategy != BucketManagementStrategy.FIXED:
        pytest.skip("ContiguousStorage only supports FIXED strategy")
    
    storage1 = storage_class(max_buckets, bucket_strategy)
    storage2 = storage_class(max_buckets, bucket_strategy)
    
    # Add counts to both storages
    storage1.add(0)
    storage1.add(5)
    storage2.add(5)
    storage2.add(10)
    
    # Merge storage2 into storage1
    storage1.merge(storage2)
    
    assert storage1.get_count(0) == 1
    assert storage1.get_count(5) == 2
    assert storage1.get_count(10) == 1

def test_bucket_limit(storage_class, max_buckets, bucket_strategy):
    if storage_class == ContiguousStorage and bucket_strategy != BucketManagementStrategy.FIXED:
        pytest.skip("ContiguousStorage only supports FIXED strategy")
    
    storage = storage_class(max_buckets, bucket_strategy)
    
    # Add more buckets than max_buckets
    for i in range(max_buckets + 10):
        storage.add(i)
    
    # Count total non-zero buckets
    non_zero_buckets = sum(1 for i in range(max_buckets + 10) if storage.get_count(i) > 0)
    
    if bucket_strategy == BucketManagementStrategy.FIXED:
        assert non_zero_buckets <= max_buckets
    else:
        # For COLLAPSE strategy, some buckets should have been merged
        assert non_zero_buckets <= max_buckets

def test_contiguous_storage_specific():
    """Test ContiguousStorage-specific features"""
    storage = ContiguousStorage(max_buckets=32)
    
    # Test bucket range limits
    min_bucket = -16
    max_bucket = 15
    
    # Add values at extremes
    storage.add(min_bucket)
    storage.add(max_bucket)
    
    # Verify counts
    assert storage.get_count(min_bucket) == 1
    assert storage.get_count(max_bucket) == 1
    
    # Test out of range buckets
    with pytest.raises(ValueError):
        storage.add(min_bucket - 1)
    with pytest.raises(ValueError):
        storage.add(max_bucket + 1)

def test_sparse_storage_specific():
    """Test SparseStorage-specific features"""
    storage = SparseStorage(max_buckets=32, strategy=BucketManagementStrategy.COLLAPSE)
    
    # Test collapse behavior
    # Add many buckets to force collapse
    for i in range(100):
        storage.add(i)
    
    # Verify that some buckets were merged
    assert len(storage.counts) <= 32
    
    # Test that we can still add to collapsed buckets
    last_bucket = max(storage.counts.keys())
    initial_count = storage.get_count(last_bucket)
    storage.add(last_bucket)
    assert storage.get_count(last_bucket) == initial_count + 1

def test_storage_clear(storage_class, max_buckets, bucket_strategy):
    if storage_class == ContiguousStorage and bucket_strategy != BucketManagementStrategy.FIXED:
        pytest.skip("ContiguousStorage only supports FIXED strategy")
    
    storage = storage_class(max_buckets, bucket_strategy)
    
    # Add some counts
    storage.add(0)
    storage.add(5)
    storage.add(10)
    
    # Clear the storage
    if hasattr(storage, 'clear'):
        storage.clear()
        
        # Verify all counts are zero
        assert storage.get_count(0) == 0
        assert storage.get_count(5) == 0
        assert storage.get_count(10) == 0

def test_negative_buckets(storage_class, max_buckets, bucket_strategy):
    if storage_class == ContiguousStorage and bucket_strategy != BucketManagementStrategy.FIXED:
        pytest.skip("ContiguousStorage only supports FIXED strategy")
    
    storage = storage_class(max_buckets, bucket_strategy)
    
    # Test adding negative bucket indices
    if storage_class == SparseStorage:
        storage.add(-1)
        storage.add(-5)
        assert storage.get_count(-1) == 1
        assert storage.get_count(-5) == 1
    else:
        # ContiguousStorage should handle negative indices within its range
        min_bucket = -(max_buckets // 2)
        storage.add(min_bucket)
        assert storage.get_count(min_bucket) == 1 