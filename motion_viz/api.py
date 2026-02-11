"""
Main API for motion visualization package.
"""

from pathlib import Path
from typing import Optional, Union, Tuple, List
import numpy as np

from .core import (
    load_features,
    find_bvh_file,
    parse_bvh_hierarchy,
    reconstruct_positions,
)
from .visualize import create_animation, create_frame_grid


def visualize(npy_path: Union[str, Path],
             output_path: Optional[Union[str, Path]] = None,
             bvh_path: Optional[Union[str, Path]] = None,
             norm_path: Optional[Union[str, Path]] = None,
             fps: int = 30,
             downsample: int = 1,
             tracking: bool = True,
             title: Optional[str] = None,
             focus_joints: Optional[Union[List[int], str]] = None,
             fixed_view: Optional[Union[Tuple[float, float], str]] = None,
             hand_point_size: float = 8,
             save_grid: bool = False,
             grid_frames: int = 9) -> Path:
    """
    Convert .npy motion file to animated video.
    
    Args:
        npy_path: Input .npy motion file
        output_path: Output video path (default: input_name.mp4)
        bvh_path: Reference BVH file (auto-detected if None)
        norm_path: Normalization file for denormalization
        fps: Frames per second
        downsample: Downsample factor
        tracking: Camera follows root
        title: Video title
        focus_joints: Focus on specific joints (list or group name)
                     Options: 'both_hands', 'left_hand', 'right_hand',
                             'both_arms', 'fingers', 'upper_body'
        fixed_view: Fixed camera view (tuple or preset)
                   Presets: 'front', 'side', 'top', 'front_down'
        hand_point_size: Point size for hand joints
        save_grid: Also save frame grid image
        grid_frames: Number of frames in grid
    
    Returns:
        Path to saved video
        
    Example:
        >>> from motion_viz import visualize
        >>> output = visualize("motion.npy", focus_joints='both_hands', 
        ...                    fixed_view='front')
    """
    # Setup paths
    npy_path = Path(npy_path)
    if output_path is None:
        output_path = npy_path.with_suffix('.mp4')
    else:
        output_path = Path(output_path)
    
    if title is None:
        title = f"Motion: {npy_path.stem}"
    
    # Find BVH
    if bvh_path is None:
        bvh_path = find_bvh_file()
        if not bvh_path:
            raise FileNotFoundError(
                "No BVH file found. Please provide bvh_path argument."
            )
    
    print(f"Using BVH: {bvh_path}")
    
    # Load data
    features = load_features(str(npy_path), norm_path)
    print(f"Loaded: {features.shape[0]} frames, {features.shape[1]} features")
    
    # Parse BVH hierarchy
    joint_names, parents = parse_bvh_hierarchy(str(bvh_path))
    num_joints = len(joint_names)
    
    # Reconstruct
    print("Reconstructing skeleton...")
    positions, hierarchy = reconstruct_positions(features, num_joints, parents)
    print(f"Reconstructed: {positions.shape[0]} frames, {positions.shape[1]} joints")
    
    # Downsample
    if downsample > 1:
        positions = positions[::downsample]
        print(f"Downsampled to {positions.shape[0]} frames")
    
    # Create animation
    print("Creating animation...")
    result_path = create_animation(
        positions, hierarchy, output_path,
        fps=fps, title=title, tracking=tracking,
        focus_joints=focus_joints, fixed_view=fixed_view,
        hand_point_size=hand_point_size
    )
    print(f"Saved: {result_path}")
    
    # Optional grid
    if save_grid:
        grid_path = output_path.with_name(f"{output_path.stem}_frames.png")
        create_frame_grid(positions, hierarchy, grid_path, 
                         num_frames=grid_frames, title=title)
        print(f"Saved grid: {grid_path}")
    
    return result_path


__all__ = ['visualize']