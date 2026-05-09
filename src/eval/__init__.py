"""Evaluation harness for Code-Diffusion-200M.

Decoupled from src.model and src.data; talks to the model only through the
Sampler protocol defined in src.eval.sampler_interface.
"""

from src.eval.sampler_interface import Sampler

__all__ = ["Sampler"]
