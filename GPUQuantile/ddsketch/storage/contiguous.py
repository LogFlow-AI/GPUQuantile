"""Contiguous array storage implementation for DDSketch using circular buffer."""

import numpy as np
from .base import Storage, BucketManagementStrategy

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
        Initialize contiguous storage.
        
        Args:
            max_buckets: Maximum number of buckets (default 2048).
        """
        if max_buckets <= 0:
            raise ValueError("max_buckets must be positive for ContiguousStorage")
        super().__init__(max_buckets, BucketManagementStrategy.FIXED)
        self.counts = np.zeros(max_buckets, dtype=np.int64)
        self.min_index = None  # Minimum bucket index seen
        self.max_index = None  # Maximum bucket index seen
        self.num_buckets = 0   # Number of non-zero buckets
        self.head = 0          # Array position corresponding to min_index
        self.offset = 0        # Offset from the physical array position to logical position
    
    def _get_position(self, bucket_index: int) -> int:
        """
        Get array position for bucket index using wrap-around.
        
        Args:
            bucket_index: The bucket index to map to array position.
            
        Returns:
            The array position using circular buffer indexing.
        """
        if self.min_index is None:
            return 0
        # Add offset to the physical position calculation
        offset = bucket_index - self.min_index
        return (self.head + offset + self.offset) % len(self.counts)
    
    def add(self, bucket_index: int, count: int = 1):
        """
        Add count to bucket_index.
        
        Args:
            bucket_index: The bucket index to add to.
            count: The count to add (default 1).
        """
        if count <= 0:
            return
            
        if self.min_index is None:
            # First insertion
            self.min_index = bucket_index
            self.max_index = bucket_index
            self.counts[0] = count
            self.num_buckets = 1
            self.head = 0
        else:
            # Check if new bucket extends the range
            if bucket_index < self.min_index or bucket_index > self.max_index:
                # Calculate new range
                new_min = min(self.min_index, bucket_index)
                new_max = max(self.max_index, bucket_index)
                new_span = new_max - new_min + 1
                
                # First center the data around the new range
                self._center_data(new_min, new_max)
                
                # Then collapse if needed to fit the new range
                if new_span > len(self.counts):
                    self._collapse_to_fit(new_min, new_max)
            
            # Add the count
            pos = self._get_position(bucket_index)
            was_zero = self.counts[pos] == 0
            self.counts[pos] += count
            if was_zero:
                self.num_buckets += 1
                
        # Check if we need to collapse due to too many buckets
        if self.num_buckets > self.max_buckets:
            self.collapse_smallest_buckets()
                
        self.total_count += count
    
    def _center_data(self, new_min: int, new_max: int):
        """Center the data in the circular buffer around the new range."""
        # Calculate the center point of the new range
        center = (new_min + new_max) // 2
        
        # Calculate how much to shift the data
        shift = center - (self.min_index + self.max_index) // 2
        
        # Update offset and range
        self.offset = (self.offset + shift) % len(self.counts)
        self.min_index = new_min
        self.max_index = new_max
        self.head = (self.head + shift) % len(self.counts)
    
    def _collapse_to_fit(self, new_min: int, new_max: int):
        """Collapse buckets to fit the new range."""
        # Calculate how many buckets we need to collapse
        current_span = self.max_index - self.min_index + 1
        new_span = new_max - new_min + 1
        buckets_to_collapse = new_span - len(self.counts)
        
        if buckets_to_collapse <= 0:
            return
            
        # Collapse buckets starting from the lowest index
        # This maintains the shape of the distribution by collapsing from one end
        for i in range(buckets_to_collapse):
            # Find the next non-zero bucket after min_index
            next_bucket = self.min_index + 1
            while next_bucket <= self.max_index and self.get_count(next_bucket) == 0:
                next_bucket += 1
                
            if next_bucket > self.max_index:
                break  # No more buckets to collapse into
                
            # Merge min_index into next_bucket if min_index has counts
            if self.get_count(self.min_index) > 0:
                self.counts[self._get_position(next_bucket)] += self.counts[self._get_position(self.min_index)]
                self.counts[self._get_position(self.min_index)] = 0
                self.num_buckets -= 1
                
            # Move min_index up
            self.min_index = next_bucket
            self.head = self._get_position(self.min_index)
    
    def remove(self, bucket_index: int, count: int = 1):
        """
        Remove count from bucket_index.
        
        Args:
            bucket_index: The bucket index to remove from.
            count: The count to remove (default 1).
        """
        if count <= 0 or self.min_index is None:
            return
            
        if self.min_index <= bucket_index <= self.max_index:
            pos = self._get_position(bucket_index)
            old_count = self.counts[pos]
            self.counts[pos] = max(0, old_count - count)
            self.total_count = max(0, self.total_count - count)
            
            if old_count > 0 and self.counts[pos] == 0:
                self.num_buckets -= 1
                if self.num_buckets == 0:
                    self.min_index = None
                    self.max_index = None
                elif bucket_index == self.min_index:
                    # Find new minimum index
                    for i in range(self.max_index - self.min_index + 1):
                        pos = (self.head + i) % len(self.counts)
                        if self.counts[pos] > 0:
                            self.min_index += i
                            self.head = pos
                            break
                elif bucket_index == self.max_index:
                    # Find new maximum index
                    for i in range(self.max_index - self.min_index + 1):
                        pos = (self.head + (self.max_index - self.min_index - i)) % len(self.counts)
                        if self.counts[pos] > 0:
                            self.max_index -= i
                            break
    
    def get_count(self, bucket_index: int) -> int:
        """
        Get count for bucket_index.
        
        Args:
            bucket_index: The bucket index to get count for.
            
        Returns:
            The count at the specified bucket index.
        """
        if self.min_index is None or bucket_index < self.min_index or bucket_index > self.max_index:
            return 0
        pos = self._get_position(bucket_index)
        return int(self.counts[pos])
    
    def merge(self, other: 'ContiguousStorage'):
        """
        Merge another storage into this one.
        
        Args:
            other: Another ContiguousStorage instance to merge with this one.
        """
        if other.min_index is None:
            return
            
        # Add each non-zero bucket
        for i in range(other.max_index - other.min_index + 1):
            pos = (other.head + i) % len(other.counts)
            if other.counts[pos] > 0:
                bucket_index = other.min_index + i
                self.add(bucket_index, int(other.counts[pos]))
    
    def collapse_smallest_buckets(self):
        """Collapse the two buckets with smallest indices."""
        if self.num_buckets < 2:
            return
            
        # Find first two non-zero buckets starting from min_index
        first_pos = self.head
        second_pos = None
        
        for i in range(self.max_index - self.min_index + 1):
            pos = (self.head + i) % len(self.counts)
            if self.counts[pos] > 0:
                if first_pos == self.head:  # Still at initial position
                    first_pos = pos
                else:
                    second_pos = pos
                    break
                    
        # If we didn't find two non-zero buckets, nothing to collapse
        if second_pos is None:
            return
                    
        # Merge counts into second bucket
        self.counts[second_pos] += self.counts[first_pos]
        self.counts[first_pos] = 0
        self.num_buckets -= 1
        
        # Update min_index and head
        self.min_index += second_pos - first_pos
        self.head = second_pos 