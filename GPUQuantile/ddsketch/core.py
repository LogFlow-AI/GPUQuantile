"""Core DDSketch implementation."""

from typing import Literal, Union
from .mapping.base import MappingScheme
from .mapping.logarithmic import LogarithmicMapping
from .mapping.linear_interpolation import LinearInterpolationMapping
from .mapping.cubic_interpolation import CubicInterpolationMapping
from .storage.base import BucketManagementStrategy
from .storage.contiguous import ContiguousStorage
from .storage.sparse import SparseStorage

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
                        If cont_neg is True, each store will have max_buckets/2 buckets.
            bucket_strategy: Strategy for managing bucket count.
                           If FIXED, uses ContiguousStorage, otherwise uses SparseStorage.
            cont_neg: Whether to handle negative values (default True).
        
        Raises:
            ValueError: If relative_accuracy is not between 0 and 1.
        """
        if not 0 < relative_accuracy < 1:
            raise ValueError("relative_accuracy must be between 0 and 1")
            
        self.relative_accuracy = relative_accuracy
        self.cont_neg = cont_neg
        
        # Adjust max_buckets if handling negative values
        store_max_buckets = max_buckets // 2 if cont_neg else max_buckets
        
        # Initialize mapping scheme
        if mapping_type == 'logarithmic':
            self.mapping = LogarithmicMapping(relative_accuracy)
        elif mapping_type == 'lin_interpol':
            self.mapping = LinearInterpolationMapping(relative_accuracy)
        elif mapping_type == 'cub_interpol':
            self.mapping = CubicInterpolationMapping(relative_accuracy)
            
        # Choose storage type based on strategy
        if bucket_strategy == BucketManagementStrategy.FIXED:
            self.positive_store = ContiguousStorage(store_max_buckets)
            self.negative_store = ContiguousStorage(store_max_buckets) if cont_neg else None
        else:
            self.positive_store = SparseStorage(store_max_buckets, bucket_strategy)
            self.negative_store = SparseStorage(store_max_buckets, bucket_strategy) if cont_neg else None
            
        self.count = 0
        self.zero_count = 0
    
    def insert(self, value: Union[int, float]) -> None:
        """
        Insert a value into the sketch.
        
        Args:
            value: The value to insert.
            
        Raises:
            ValueError: If value is negative and cont_neg is False.
        """
        if value == 0:
            self.zero_count += 1
        elif value > 0:
            bucket_idx = self.mapping.compute_bucket_index(value)
            self.positive_store.add(bucket_idx)
        elif value < 0 and self.cont_neg:
            bucket_idx = self.mapping.compute_bucket_index(-value)
            self.negative_store.add(bucket_idx)
        elif value < 0:
            raise ValueError("Negative values not supported when cont_neg is False")
        self.count += 1
    
    def delete(self, value: Union[int, float]) -> None:
        """
        Delete a value from the sketch.
        
        Args:
            value: The value to delete.
            
        Raises:
            ValueError: If value is negative and cont_neg is False.
        """
        if self.count == 0:
            return
            
        if value == 0 and self.zero_count > 0:
            self.zero_count -= 1
        elif value > 0:
            bucket_idx = self.mapping.compute_bucket_index(value)
            self.positive_store.remove(bucket_idx)
        elif value < 0 and self.cont_neg:
            bucket_idx = self.mapping.compute_bucket_index(-value)
            self.negative_store.remove(bucket_idx)
        elif value < 0:
            raise ValueError("Negative values not supported when cont_neg is False")
        self.count = max(0, self.count - 1)
    
    def quantile(self, q: float) -> float:
        """
        Compute the approximate quantile.
        
        Args:
            q: The desired quantile (between 0 and 1).
            
        Returns:
            The approximate value at the specified quantile.
            
        Raises:
            ValueError: If q is not between 0 and 1 or if the sketch is empty.
        """
        if not 0 <= q <= 1:
            raise ValueError("Quantile must be between 0 and 1")
        if self.count == 0:
            raise ValueError("Cannot compute quantile of empty sketch")
            
        rank = q * self.count
        
        if self.cont_neg:
            neg_count = sum(self.negative_store.counts.values())
            if rank < neg_count:
                # Handle negative values
                curr_count = 0
                for idx in sorted(self.negative_store.counts.keys(), reverse=True):
                    curr_count += self.negative_store.get_count(idx)
                    if curr_count > rank:
                        return -self.mapping.compute_value_from_index(idx)
            rank -= neg_count
            
        if rank < self.zero_count:
            return 0
        rank -= self.zero_count
        
        curr_count = 0
        for idx in sorted(self.positive_store.counts.keys()):
            curr_count += self.positive_store.get_count(idx)
            if curr_count > rank:
                return self.mapping.compute_value_from_index(idx)
                
        return float('inf')
    
    def merge(self, other: 'DDSketch') -> None:
        """
        Merge another DDSketch into this one.
        
        Args:
            other: Another DDSketch instance to merge with this one.
            
        Raises:
            ValueError: If the sketches are incompatible.
        """
        if self.relative_accuracy != other.relative_accuracy:
            raise ValueError("Cannot merge sketches with different relative accuracies")
            
        self.positive_store.merge(other.positive_store)
        if self.cont_neg and other.cont_neg:
            self.negative_store.merge(other.negative_store)
        elif other.cont_neg and sum(other.negative_store.counts.values()) > 0:
            raise ValueError("Cannot merge sketch containing negative values when cont_neg is False")
            
        self.zero_count += other.zero_count
        self.count += other.count 