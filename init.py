"""
polydiff: minimal polynomial-mode autodiff MVP.

Exports:
  - PolyTensor

This MVP only supports elementwise add/sub/mul/neg (and in-place variants),
plus detach support to make backward work. Everything else raises.
"""

from .polytensor import PolyTensor

__all__ = ["PolyTensor"]
