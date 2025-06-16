"""
Test Suite Generator Package

A package for generating test suites for Python projects using AI agents.
"""

__version__ = "1.0.0"
__author__ = "Leith Chakroun"
__email__ = "leith.chakroun@epita.fr"

from .api import create_app
from .generator import TestSuiteGenerator
from .config import config

__all__ = ["create_app", "TestSuiteGenerator", "config"]