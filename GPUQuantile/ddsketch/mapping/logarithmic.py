"""Logarithmic mapping scheme for DDSketch."""

import numpy as np
from .base import MappingScheme

class LogarithmicMapping(MappingScheme):
    """Logarithmic mapping scheme for DDSketch."""
    
    def __init__(self, alpha: float):
        """
        Initialize logarithmic mapping.
        
        Args:
            alpha: Relative accuracy guarantee. Must be between 0 and 1.
        """
        self.alpha = alpha
        self.gamma = (1 + alpha) / (1 - alpha)
        self.multiplier = 1 / np.log(self.gamma)
        
    def compute_bucket_index(self, value: float) -> int:
        """
        Compute bucket index for a value using logarithmic mapping.
        
        Args:
            value: The value to map to a bucket index.
            
        Returns:
            The bucket index as ceil(log_gamma(value)).
        """
        if value <= 0:
            raise ValueError("Value must be positive")
        return int(np.ceil(np.log(value) * self.multiplier))
    
    def compute_value_from_index(self, index: int) -> float:
        """
        Compute the representative value for a bucket index.
        
        Args:
            index: The bucket index.
            
        Returns:
            The representative value gamma^index.
        """
        return np.power(self.gamma, index) 