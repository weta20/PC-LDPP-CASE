"""Fused shake and variable-neighborhood descent for CBIS."""

from .engine import solve as search
from .models import line_model


def solve(initial, matrix, seed, options, progress=None):
    return search(initial, line_model(matrix), seed, options, progress)
