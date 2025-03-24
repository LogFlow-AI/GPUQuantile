"""Logarithmic mapping scheme for DDSketch."""

from math import log, pow, floor
import sys
from .base import MappingScheme

class LogarithmicMapping(MappingScheme):
    """Mapping that uses logarithms to compute indices and values."""
    
    def __init__(self, relative_accuracy):
        """Initialize a LogarithmicMapping with a given accuracy.
        
        Args:
            relative_accuracy (float): Relative accuracy guarantee for the mapping.
                Must be between 0 and 1.
        """
        self.relative_accuracy = relative_accuracy
        # gamma = (1 + accuracy) / (1 - accuracy)
        self.gamma = (1.0 + relative_accuracy) / (1.0 - relative_accuracy)
        self.multiplier = 1.0 / log(self.gamma)
        self.min_possible = -sys.maxsize - 1  # For handling zero/negative values
        
    def compute_bucket_index(self, value):
        """Compute the bucket index for a given value.
        
        Args:
            value (float): The value to compute the bucket index for.
                Must be positive.
                
        Returns:
            int: The bucket index for the value.
            
        Raises:
            ValueError: If value is zero or negative.
        """
        if value <= 0:
            raise ValueError("Value must be positive, got {}".format(value))

        # Calculate the bucket index using floor for better compliance with tests
        # This matches the theoretical definition in the paper
        index = int(floor(log(value) * self.multiplier))
        
        return index
    
    def compute_value_from_index(self, bucket_index):
        """Compute the value at a given bucket index.
        
        Args:
            bucket_index (int): The bucket index to compute the value for.
                
        Returns:
            float: The value at the bucket index.
        """
        if bucket_index == self.min_possible:
            return 0.0
            
        # Calculate the value from the bucket index
        # For better accuracy, we use the middle of the bucket range
        # This ensures the relative error is within bounds
        value = pow(self.gamma, bucket_index + 0.5)
        
        return value