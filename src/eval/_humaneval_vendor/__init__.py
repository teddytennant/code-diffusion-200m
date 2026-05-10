"""Vendored subset of openai/human-eval (MIT licensed).

We only need ``check_correctness`` for HumanEval/MBPP scoring. Vendored
because the upstream ``human-eval`` PyPI install is broken on modern pip
(invalid console_scripts entry point).

Source: https://github.com/openai/human-eval
"""
from .execution import check_correctness

__all__ = ["check_correctness"]
