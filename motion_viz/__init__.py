"""
Motion Visualization - Lightweight motion visualization from .npy features.

A minimal-dependency package for converting motion feature files to 
animated skeleton videos.

Example:
    >>> from motion_viz import visualize
    >>> visualize("motion.npy", focus_joints='both_hands', fixed_view='front')
"""

from .api import visualize
from .core import JOINT_GROUPS

__version__ = "0.1.0"
__all__ = ['visualize', 'JOINT_GROUPS']