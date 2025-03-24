"""
Linear interpolation mapping scheme for DDSketch.

This implementation approximates the memory-optimal logarithmic mapping by:
1. Extracting the floor value of log2 from binary representation
2. Linearly interpolating the logarithm between consecutive powers of 2
"""

import numpy as np
from .base import MappingScheme

class LinearInterpolationMapping(MappingScheme):
    """Linear interpolation mapping scheme for DDSketch."""
    
    def __init__(self, alpha: float):
        """
        Initialize linear interpolation mapping with relative accuracy alpha.
        
        Args:
            alpha: Relative accuracy parameter (0 < alpha < 1).
        """
        self.relative_accuracy = alpha
        self.alpha = alpha
        self.gamma = (1 + alpha) / (1 - alpha)
        self.log2_gamma = np.log2(self.gamma)
        # Use a higher multiplier for improved accuracy with small values
        self.interpolation_multiplier = 2.0  
        
    def _extract_exponent(self, value: float) -> tuple[int, float]:
        """
        Extract the binary exponent and normalized fraction from an IEEE 754 float.
        
        Args:
            value: The float value to decompose.
            
        Returns:
            tuple: (exponent, normalized_fraction)
            where normalized_fraction is in [1, 2)
        """
        # Convert float to its binary representation
        bits = np.frexp(value)
        exponent = bits[1] - 1  # frexp returns 2's exponent, we need floor(log2)
        normalized_fraction = bits[0] * 2  # Scale to [1, 2)
        return exponent, normalized_fraction
        
    def compute_bucket_index(self, value: float) -> int:
        """
        Compute bucket index for a value.
        
        Args:
            value: The value to map to a bucket index.
            
        Returns:
            Bucket index.
            
        Raises:
            ValueError: If value is zero or negative.
        """
        if value <= 0:
            raise ValueError("Value must be positive, got {}".format(value))
            
        # Get binary exponent and normalized fraction
        exponent, normalized_fraction = self._extract_exponent(value)
        
        # Linear interpolation between powers of 2
        # normalized_fraction is in [1, 2), so we interpolate log_gamma
        log2_fraction = normalized_fraction - 1  # Map [1, 2) to [0, 1)
        
        # Compute final index using change of base and linear interpolation
        # index = floor(log_gamma(value))
        # log_gamma(value) = log2(value) / log2(gamma)
        log2_value = exponent + log2_fraction
        
        # Use an additional scaling factor for better accuracy
        # This helps with small values and extreme values
        scaled_value = log2_value / self.log2_gamma * self.interpolation_multiplier
        return int(np.floor(scaled_value))
        
    def compute_value_from_index(self, index: int) -> float:
        """
        Compute representative value for a bucket index.
        
        Args:
            index: Bucket index.
            
        Returns:
            Representative value for the bucket.
        """
        if index == np.iinfo(np.int32).min:
            return 0.0
            
        # Convert back from index to value using gamma
        # Use the middle of the bucket for better accuracy
        # Adjust for the interpolation multiplier
        adjusted_index = (index + 0.5) / self.interpolation_multiplier
        return np.power(self.gamma, adjusted_index)