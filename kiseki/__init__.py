"""
Kiseki (軌跡) — Lightweight motion trajectory visualization from .npy features.

A minimal-dependency package for converting motion feature files to 
animated skeleton videos.

Example:
    >>> from kiseki import visualize
    >>> visualize("motion.npy", focus_joints='both_hands', fixed_view='front')
"""

from .api import visualize
from .core import JOINT_GROUPS
from .visualize import VIEW_PRESETS

__version__ = "0.1.0"
__all__ = ['visualize', 'JOINT_GROUPS', 'VIEW_PRESETS']