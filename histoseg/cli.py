"""
HistoSeg CLI module.

Provides command-line interface for training, validation, and testing.
"""

# Import the main CLI from the root
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from cli import cli
except ImportError:
    # Create a simple placeholder CLI
    def cli():
        """Placeholder CLI function."""
        print("HistoSeg CLI - placeholder implementation")
        print("Use the main cli.py script in the repository root")

__all__ = ['cli']
