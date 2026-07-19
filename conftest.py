"""Minimal conftest: put the repo root on sys.path for test imports."""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
