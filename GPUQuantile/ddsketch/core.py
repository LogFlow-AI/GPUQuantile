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
            relative_accuracy: The relative accuracy guarantee (0 < alpha < 1).
            mapping_type: The mapping scheme to use:
                - 'logarithmic': Basic logarithmic mapping
                - 'lin_interpol': Linear interpolation mapping (more accurate)
                - 'cub_interpol': Cubic interpolation mapping (most accurate, more compute)
            max_buckets: Maximum number of buckets in storage (default 2048).
            bucket_strategy: Strategy for bucket management (FIXED or DYNAMIC).
            cont_neg: Whether to track negative values (default True).
        
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
        
        # Create appropriate mapping scheme
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
            self.positive_store = ContiguousStorage(store_max_buckets)
            self.negative_store = ContiguousStorage(store_max_buckets) if cont_neg else None
        else:
            self.positive_store = SparseStorage(store_max_buckets, bucket_strategy)
            self.negative_store = SparseStorage(store_max_buckets, bucket_strategy) if cont_neg else None
    
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
        if value == 0:
            if self.zero_count > 0:
                self.zero_count -= 1
                self.count -= 1
        elif value > 0:
            bucket_index = self.mapping.compute_bucket_index(value)
            if self.positive_store.get_count(bucket_index) > 0:
                self.positive_store.remove(bucket_index)
                self.count -= 1
                
                # Update min/max if needed
                if value == self.min_value or value == self.max_value:
                    # Recalculate min/max from buckets
                    self._update_min_max_after_delete(value)
        elif value < 0 and self.cont_neg:
            bucket_idx = self.mapping.compute_bucket_index(-value)
            if self.negative_store.get_count(bucket_idx) > 0:
                self.negative_store.remove(bucket_idx)
                self.count -= 1
        # If value not found or negative with cont_neg=False, do nothing
    
    def _get_storage_count(self, storage):
        """Helper method to get the total count from a storage, handling both dict and array counts."""
        if hasattr(storage.counts, 'values'):
            # Dict-based storage (SparseStorage)
            return sum(storage.counts.values())
        else:
            # Array-based storage (ContiguousStorage)
            return int(np.sum(storage.counts))
    
    def _find_bucket_at_rank(self, rank):
        """Find the bucket at a given rank.

        Args:
            rank (float): The rank to find the bucket for (1-based rank).
                If rank <= 0, returns the smallest bucket index.
                If rank > self.count, returns the largest bucket index.

        Returns:
            int: The bucket index at the given rank.
        """
        if rank <= 0:
            # Return the smallest non-zero bucket
            min_bucket = float('inf')
            for bucket_index, count in self._get_specific_storage_items(self.negative_store):
                if count > 0 and bucket_index < min_bucket:
                    min_bucket = bucket_index
            for bucket_index, count in self._get_specific_storage_items(self.positive_store):
                if count > 0 and bucket_index < min_bucket:
                    min_bucket = bucket_index
            return float('-inf') if min_bucket == float('inf') else min_bucket

        # Collect all bucket indices and counts
        neg_buckets = [(idx, count) for idx, count in self._get_specific_storage_items(self.negative_store)]
        pos_buckets = [(idx, count) for idx, count in self._get_specific_storage_items(self.positive_store)]
        
        # Sort negative buckets in descending order (most negative first)
        neg_buckets.sort(reverse=True)
        # Sort positive buckets in ascending order (least positive first)
        pos_buckets.sort()
        
        # Combine buckets, negative first then positive
        sorted_buckets = neg_buckets + pos_buckets
        
        if not sorted_buckets:
            return 0  # Default if no buckets
        
        # Find the bucket at the rank
        cumulative = 0
        for bucket_index, count in sorted_buckets:
            cumulative += count
            if cumulative >= rank:
                return bucket_index
        
        # If we get here, return the last bucket
        return sorted_buckets[-1][0] if sorted_buckets else 0
    
    def quantile(self, q):
        """Return the approximate value at a given quantile.

        Args:
            q (float): The quantile to compute, must be between 0 and 1.

        Returns:
            float: The approximate value at the given quantile.

        Raises:
            ValueError: If the sketch is empty or q is outside [0, 1].
        """
        if self.count == 0:
            raise ValueError("Cannot compute quantile of empty sketch")
        
        if q < 0 or q > 1:
            raise ValueError(f"Quantile must be between 0 and 1, got {q}")
        
        # Handle edge cases directly
        if q == 0:
            return self.min_value
        if q == 1:
            return self.max_value
            
        # Handle zero counts
        if self.zero_count > 0:
            # Calculate the rank corresponding to the requested quantile (1-based)
            rank = 1 + q * (self.count - 1)
            
            # Count values in negative store
            neg_count = 0
            if self.negative_store is not None:
                neg_count = self._get_storage_count(self.negative_store)
                
            # Count values in positive store
            pos_count = self._get_storage_count(self.positive_store)
            
            # Determine where q falls
            if neg_count > 0 and rank <= neg_count:
                # q falls in negative range
                bucket_index = self._find_bucket_at_rank(rank)
                return -self.mapping.compute_value_from_index(bucket_index)
            elif pos_count > 0 and rank > neg_count + self.zero_count:
                # q falls in positive range
                pos_rank = rank - (neg_count + self.zero_count)
                bucket_index = self._find_bucket_at_rank(pos_rank)
                return self.mapping.compute_value_from_index(bucket_index)
            else:
                # q falls in zero range
                return 0.0
        else:
            # No zeros, simpler calculation
            # Calculate rank directly
            rank = 1 + q * (self.count - 1)
            
            # Check if we need to look in negative or positive storage
            neg_count = 0
            if self.negative_store is not None:
                neg_count = self._get_storage_count(self.negative_store)
                
            if rank <= neg_count:
                # The rank is in the negative storage
                bucket_index = self._find_bucket_at_rank(rank)
                return -self.mapping.compute_value_from_index(bucket_index)
            else:
                # The rank is in the positive storage
                pos_rank = rank - neg_count
                bucket_index = self._find_bucket_at_rank(pos_rank)
                return self.mapping.compute_value_from_index(bucket_index)
    
    def _get_specific_storage_items(self, storage):
        """
        Get (bucket_index, count) pairs from a specific storage.
        
        Args:
            storage: The storage to get items from.
            
        Returns:
            A list or generator of (bucket_index, count) pairs.
        """
        if hasattr(storage.counts, 'items'):
            # Dictionary-based storage
            return storage.counts.items()
        else:
            # Array-based storage
            non_zero_indices = np.nonzero(storage.counts)[0]
            min_idx = storage.min_index
            return [(idx + min_idx, storage.counts[idx]) for idx in non_zero_indices]
    
    def merge(self, other: 'DDSketch') -> 'DDSketch':
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