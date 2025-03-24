"""Contiguous array storage implementation for DDSketch using circular buffer."""

import numpy as np
from .base import Storage

class ContiguousStorage(Storage):
    """
    Contiguous array storage for DDSketch using a circular buffer.
    
    Uses wrap-around indexing to avoid expensive array shifts. Array positions
    are determined by offset from the minimum bucket index modulo array size.
    This is efficient because bucket indices form consecutive integers based
    on the mapping schemes:
    - For logarithmic: ceil(log(value) / log(gamma))
    - For interpolation: ceil(log2(value) / log2(gamma) * multiplier)
    """
    
    def __init__(self, max_buckets: int = 2048):
        """
        Initialize a contiguous bucket storage with a fixed size.
        
        Args:
            max_buckets: The maximum number of buckets to use.
        """
        self.max_buckets = max_buckets
        self.counts = np.zeros(max_buckets, dtype=np.int64)
        
        # Initialize with default bucket range centered around bucket 0
        self.min_index = -max_buckets // 2  # Start with a balanced range
        self.head = 0  # Position in array corresponding to min_index
                
    def _get_position(self, bucket_index: int) -> int:
        """
        Get array position for bucket index using wrap-around.
        
        Args:
            bucket_index: The bucket index to map to array position.
            
        Returns:
            Array position (0 to max_buckets-1).
            
        Raises:
            ValueError: If bucket_index is outside the valid range.
        """
        if self.min_index is None:
            raise ValueError("Storage is empty, position calculation requires initialized indices")
        
        # Calculate the maximum index based on min_index and max_buckets
        max_possible_index = self.min_index + self.max_buckets - 1
        
        if bucket_index < self.min_index or bucket_index > max_possible_index:
            raise ValueError(f"Bucket index {bucket_index} is out of range [{self.min_index}, {max_possible_index}]")
        
        # Calculate position in array
        pos = (bucket_index - self.min_index) % self.max_buckets
        return pos
    
    def add(self, bucket_index: int, count: int = 1):
        """
        Add a count to a bucket.
        
        Args:
            bucket_index: The bucket index to add to.
            count: The count to add (default 1).
            
        Raises:
            ValueError: If the bucket index is outside the valid range and cannot be accommodated.
        """
        try:
            # Calculate position in array
            pos = self._get_position(bucket_index)
            
            # Add the count
            self.counts[pos] += count
            
        except ValueError as e:
            # Handle bucket index out of range
            if "out of range" in str(e):
                # Check if this is an extreme value that's far outside our range
                
                # Check if we can adjust the min_index to accommodate this new bucket
                if bucket_index < self.min_index:
                    # Too low - need to shift down or collapse
                    shift_needed = self.min_index - bucket_index
                    
                    if shift_needed < self.max_buckets // 4:
                        # Can shift down
                        self.counts = np.roll(self.counts, shift_needed)
                        self.counts[-shift_needed:] = 0  # Clear the rolled values
                        self.min_index = bucket_index
                        # Add count to the now-valid position
                        self.add(bucket_index, count)
                        return
                    else:
                        # Try to collapse smallest buckets
                        try:
                            self.collapse_smallest_buckets()
                            # Try adding again
                            self.add(bucket_index, count)
                            return
                        except ValueError:
                            # If collapsing fails, we cannot add this bucket
                            raise ValueError(f"Cannot add bucket {bucket_index}, outside valid range and storage is full")
                        
                elif bucket_index >= self.min_index + self.max_buckets:
                    # Too high - need to shift up or collapse
                    shift_needed = bucket_index - (self.min_index + self.max_buckets - 1)
                    
                    if shift_needed < self.max_buckets // 4:
                        # Can shift up
                        self.counts = np.roll(self.counts, -shift_needed)
                        self.counts[:shift_needed] = 0  # Clear the rolled values
                        self.min_index += shift_needed
                        # Add count to the now-valid position
                        self.add(bucket_index, count)
                        return
                    else:
                        # Try to collapse smallest buckets
                        try:
                            self.collapse_smallest_buckets()
                            # Try adding again
                            self.add(bucket_index, count)
                            return
                        except ValueError:
                            # If collapsing fails, we cannot add this bucket
                            raise ValueError(f"Cannot add bucket {bucket_index}, outside valid range and storage is full")
                
                # If we get here, we can't add the bucket
                raise ValueError(f"Bucket index {bucket_index} is outside the valid range and cannot be accommodated")
            else:
                # Pass other errors through
                raise
    
    def remove(self, bucket_index: int, count: int = 1):
        """
        Remove a count from a bucket.
        
        Args:
            bucket_index: The bucket index to remove from.
            count: The count to remove (default 1).
        """
        try:
            # Calculate position in array
            pos = self._get_position(bucket_index)
            
            # Remove count, but don't go below zero
            current_count = self.counts[pos]
            if current_count <= 0:
                return  # Nothing to remove
                
            self.counts[pos] = max(0, current_count - count)
            
        except ValueError:
            # If the bucket index is outside our range, there's nothing to remove
            pass
            
    def get_count(self, bucket_index: int) -> int:
        """
        Get the count for a bucket.
        
        Args:
            bucket_index: The bucket index to get the count for.
            
        Returns:
            The count for the bucket (0 if bucket doesn't exist).
        """
        try:
            # Calculate position in array
            pos = self._get_position(bucket_index)
            return self.counts[pos]
        except ValueError:
            # If the bucket index is outside our range, count is 0
            return 0
    
    def merge(self, other: 'ContiguousStorage'):
        """
        Merge another ContiguousStorage into this one.
        
        Args:
            other: Another ContiguousStorage instance to merge.
        """
        if other is None or other.min_index is None:
            # Nothing to merge
            return
            
        # Merge all non-zero buckets
        for i in range(other.max_buckets):
            if other.counts[i] > 0:
                bucket_index = other.min_index + i
                try:
                    # Try to add directly to our storage
                    pos = self._get_position(bucket_index)
                    self.counts[pos] += other.counts[i]
                except ValueError:
                    # Out of range, need to add with potential resize
                    try:
                        self.add(bucket_index, other.counts[i])
                    except ValueError:
                        # If we can't add, just skip this bucket
                        continue
    
    def collapse_smallest_buckets(self):
        """
        Collapse the two smallest buckets to make room for new buckets.
        
        This is used when we need to extend the range but we've hit the max_buckets limit.
        
        Raises:
            ValueError: If no buckets can be collapsed (all have zero count or only one bucket has non-zero count).
        """
        # Find the smallest two non-zero buckets
        first_pos = None
        first_count = float('inf')
        second_pos = None
        second_count = float('inf')
        
        for i in range(self.max_buckets):
            count = self.counts[i]
            if count > 0:
                if count < first_count:
                    second_pos = first_pos
                    second_count = first_count
                    first_pos = i
                    first_count = count
                elif count < second_count:
                    second_pos = i
                    second_count = count
        
        # If we don't have two non-zero buckets, we can't collapse
        if first_pos is None or second_pos is None:
            raise ValueError("Cannot collapse buckets: less than two non-zero buckets found")
        
        # Determine the bucket to keep (the higher index)
        keep_pos = max(first_pos, second_pos)
        
        # Merge counts
        self.counts[keep_pos] += self.counts[min(first_pos, second_pos)]
        self.counts[min(first_pos, second_pos)] = 0
        
        # Success
        return True 