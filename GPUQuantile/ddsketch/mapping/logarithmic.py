"""Logarithmic mapping scheme for DDSketch."""

import numpy as np
from .base import MappingScheme

class LogarithmicMapping(MappingScheme):
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.gamma = (1 + alpha) / (1 - alpha)
        self.multiplier = 1 / np.log(self.gamma)
        
    def compute_bucket_index(self, value: float) -> int:
        # ceil(log_gamma(value) = ceil(log(value) / log(gamma))
        return int(np.ceil(np.log(value) * self.multiplier))
    
    def compute_value_from_index(self, index: int) -> float:
        return np.power(self.gamma, index)