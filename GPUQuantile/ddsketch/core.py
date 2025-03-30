"""Core DDSketch implementation."""

from typing import Literal, Union
from .mapping.logarithmic import LogarithmicMapping
from .mapping.linear_interpolation import LinearInterpolationMapping
from .mapping.cubic_interpolation import CubicInterpolationMapping
from .storage.base import BucketManagementStrategy
from .storage.contiguous import ContiguousStorage
from .storage.sparse import SparseStorage
import numpy as np

class DDSketch:
    """
    DDSketch implementation for quantile approximation with relative-error guarantees.
    
    This implementation supports different mapping schemes and storage types for
    optimal performance in different scenarios. It can handle both positive and
    negative values, and provides configurable bucket management strategies.
    
    Reference:
        "DDSketch: A Fast and Fully-Mergeable Quantile Sketch with Relative-Error Guarantees"
        by Charles Masson, Jee E. Rim and Homin K. Lee
    """
    
    def __init__(
        self,
        relative_accuracy: float,
        mapping_type: Literal['logarithmic', 'lin_interpol', 'cub_interpol'] = 'logarithmic',
        max_buckets: int = 2048,
        bucket_strategy: BucketManagementStrategy = BucketManagementStrategy.FIXED,
        cont_neg: bool = True
    ):
        """
        Initialize DDSketch.
        
        Args:
            relative_accuracy: The relative accuracy guarantee (alpha).
                             Must be between 0 and 1.
            mapping_type: The type of mapping scheme to use:
                        - 'logarithmic': Basic logarithmic mapping
                        - 'lin_interpol': Linear interpolation mapping
                        - 'cub_interpol': Cubic interpolation mapping
            max_buckets: Maximum number of buckets per store (default 2048).
                        If cont_neg is True, each store will have max_buckets buckets.
            bucket_strategy: Strategy for managing bucket count.
                           If FIXED, uses ContiguousStorage, otherwise uses SparseStorage.
            cont_neg: Whether to handle negative values (default True).
        
        Raises:
            ValueError: If relative_accuracy is not between 0 and 1.
        """
        if not 0 < relative_accuracy < 1:
            raise ValueError("Relative accuracy must be between 0 and 1")
            
        self.relative_accuracy = relative_accuracy
        self.cont_neg = cont_neg
        self.count = 0
        self.zero_count = 0  # Count of zero values (stored separately)
        self.min_value = None  # Minimum value seen
        self.max_value = None  # Maximum value seen
        
        
        # Initialize mapping scheme
        if mapping_type == 'logarithmic':
            self.mapping = LogarithmicMapping(relative_accuracy)
        elif mapping_type == 'lin_interpol':
            self.mapping = LinearInterpolationMapping(relative_accuracy)
        elif mapping_type == 'cub_interpol':
            self.mapping = CubicInterpolationMapping(relative_accuracy)
        else:
            raise ValueError(f"Unknown mapping type: {mapping_type}")
            
        # Adjust max_buckets if handling negative values
        store_max_buckets = max_buckets // 2 if cont_neg else max_buckets
        
        # Choose storage type based on strategy
        if bucket_strategy == BucketManagementStrategy.FIXED:
            self.positive_store = ContiguousStorage(max_buckets)
            self.negative_store = ContiguousStorage(max_buckets) if cont_neg else None
        else:
            self.positive_store = SparseStorage(strategy=bucket_strategy)
            self.negative_store = SparseStorage(strategy=bucket_strategy) if cont_neg else None
            
        self.count = 0
        self.zero_count = 0
    
    def insert(self, value: Union[int, float]) -> None:
        """
        Insert a value into the sketch.
        
        Args:
            value: The value to insert.
            
        Raises:
            ValueError: If the value is negative and cont_neg is False.
        """
        if not self.cont_neg and value < 0:
            raise ValueError("Negative values are not supported with cont_neg=False")
            
        # Handle min/max tracking
        if self.count == 0:
            self.min_value = value
            self.max_value = value
        else:
            if value < self.min_value:
                self.min_value = value
            if value > self.max_value:
                self.max_value = value
                
        # Handle zero values
        if value == 0:
            self.zero_count += 1
            self.count += 1
            return
            
        # Determine the bucket index
        bucket_index = self.mapping.compute_bucket_index(abs(value))
        
        # Print bucket index for debugging - remove in production
        # print(f"Value: {value}, Bucket index: {bucket_index}")
            
        # Insert into the appropriate store based on sign
        if value > 0:
            self.positive_store.add(bucket_index)
        else:  # value < 0
            self.negative_store.add(bucket_index)
            
        self.count += 1
    
    def delete(self, value: Union[int, float]) -> None:
        """
        Delete a value from the sketch.
        
        This is an approximate operation. It removes the value from the appropriate
        bucket, but does not guarantee that the exact value is removed.
        
        Args:
            value: The value to delete.
        """
        if self.count == 0:
            return
            
        deleted = False
        if value == 0 and self.zero_count > 0:
            self.zero_count -= 1
            deleted = True
        elif value > 0:
            bucket_idx = self.mapping.compute_bucket_index(value)
            deleted = self.positive_store.remove(bucket_idx)
        elif value < 0 and self.cont_neg:
            bucket_idx = self.mapping.compute_bucket_index(-value)
            deleted = self.negative_store.remove(bucket_idx)
        elif value < 0:
            raise ValueError("Negative values not supported when cont_neg is False")
            
        if deleted:
            self.count -= 1
    
    def quantile(self, q: float) -> float:
        """
        Compute the approximate quantile.
        
        Args:
            q (float): The quantile to compute, must be between 0 and 1.

        Returns:
            float: The approximate value at the given quantile.

        Raises:
            ValueError: If the sketch is empty or q is outside [0, 1].
        """
        if self.count == 0:
            raise ValueError("Cannot compute quantile of empty sketch")
            
        rank = q * (self.count - 1)
        
        if self.cont_neg:
            neg_count = self.negative_store.total_count
            if rank < neg_count:
                # Handle negative values
                curr_count = 0
                if self.negative_store.min_index is not None:
                    for idx in range(self.negative_store.max_index, self.negative_store.min_index - 1, -1):
                        bucket_count = self.negative_store.get_count(idx)
                        curr_count += bucket_count
                        if curr_count > rank:
                            return -self.mapping.compute_value_from_index(idx)
            rank -= neg_count
            
        if rank < self.zero_count:
            return 0
        rank -= self.zero_count
        
        curr_count = 0
        if self.positive_store.min_index is not None:
            for idx in range(self.positive_store.min_index, self.positive_store.max_index + 1):
                bucket_count = self.positive_store.get_count(idx)
                curr_count += bucket_count
                if curr_count > rank:
                    return self.mapping.compute_value_from_index(idx)
                
        return float('inf')
    
    def merge(self, other: 'DDSketch') -> None:
        """
        Merge another DDSketch into this one.
        
        Args:
            other: Another DDSketch to merge with this one.
            
        Returns:
            self: This sketch after the merge.
            
        Raises:
            ValueError: If the sketches have different relative accuracy.
        """
        if self.relative_accuracy != other.relative_accuracy:
            raise ValueError("Cannot merge sketches with different relative accuracy")
            
        # Merge zero counts 
        self.zero_count += other.zero_count
        
        # Merge negative store if present
        if other.negative_store is not None and other.cont_neg:
            for bucket_index, count in other._get_specific_storage_items(other.negative_store):
                if count > 0:
                    self.negative_store.add(bucket_index, count)
                    
        # Merge positive store
        for bucket_index, count in other._get_specific_storage_items(other.positive_store):
            if count > 0:
                self.positive_store.add(bucket_index, count)
                
        # Update count
        self.count += other.count
        
        # Update min/max values
        if other.min_value is not None:
            if self.min_value is None or other.min_value < self.min_value:
                self.min_value = other.min_value
                
        if other.max_value is not None:
            if self.max_value is None or other.max_value > self.max_value:
                self.max_value = other.max_value
                
        return self
    
    def _update_min_max_after_delete(self, deleted_value: float) -> None:
        """
        Update min and max values after a deletion operation.
        This is called only when the min or max value was deleted.
        
        Args:
            deleted_value: The value that was deleted.
        """
        # If we deleted the min value, recalculate min
        if deleted_value == self.min_value:
            if self.zero_count > 0:
                self.min_value = 0
            else:
                # Find the smallest bucket with non-zero count
                min_bucket = None
                for bucket_index, count in self._get_specific_storage_items(self.positive_store):
                    if count > 0 and (min_bucket is None or bucket_index < min_bucket):
                        min_bucket = bucket_index
                
                if min_bucket is not None:
                    self.min_value = self.mapping.compute_value_from_index(min_bucket)
                else:
                    # No more values in the sketch
                    self.min_value = None
        
        # If we deleted the max value, recalculate max
        if deleted_value == self.max_value:
            if self.zero_count > 0:
                self.max_value = 0
            else:
                # Find the largest bucket with non-zero count
                max_bucket = None
                for bucket_index, count in self._get_specific_storage_items(self.positive_store):
                    if count > 0 and (max_bucket is None or bucket_index > max_bucket):
                        max_bucket = bucket_index
                
                if max_bucket is not None:
                    self.max_value = self.mapping.compute_value_from_index(max_bucket)
                else:
                    # No more values in the sketch
                    self.max_value = None 