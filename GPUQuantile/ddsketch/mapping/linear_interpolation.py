"""
Linear interpolation mapping scheme for DDSketch.

This implementation approximates the memory-optimal logarithmic mapping by:
1. Extracting the floor value of log2 from binary representation
2. Linearly interpolating the logarithm between consecutive powers of 2
"""

import numpy as np
from .base import MappingScheme

class LinearInterpolationMapping(MappingScheme):
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.gamma = (1 + alpha) / (1 - alpha)
        self.log2_gamma = np.log2(self.gamma)
        
    def _extract_exponent(self, value: float) -> tuple[int, float]:
        """
        Extract the binary exponent and normalized fraction from an IEEE 754 float.
        
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
        if value <= 0:
            raise ValueError("Value must be positive")
            
        # Get binary exponent and normalized fraction
        exponent, normalized_fraction = self._extract_exponent(value)
        
        # Linear interpolation between powers of 2
        # normalized_fraction is in [1, 2), so we interpolate log_gamma
        log2_fraction = normalized_fraction - 1  # Map [1, 2) to [0, 1)
        
        # Compute final index using change of base and linear interpolation
        # index = ceil(log_gamma(value))
        # log_gamma(value) = log2(value) / log2(gamma)
        log2_value = exponent + log2_fraction
        return int(np.ceil(log2_value / self.log2_gamma))
        
    def compute_value_from_index(self, index: int) -> float:
        # Convert back from index to value using gamma
        # value = gamma^index
        return np.power(self.gamma, index)