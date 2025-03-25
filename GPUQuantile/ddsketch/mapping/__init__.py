"""Mapping schemes for DDSketch."""

from .base import MappingScheme
from .logarithmic import LogarithmicMapping
from .linear_interpolation import LinearInterpolationMapping
from .cubic_interpolation import CubicInterpolationMapping

__all__ = [
    'MappingScheme',
    'LogarithmicMapping',
    'LinearInterpolationMapping',
    'CubicInterpolationMapping'
] 