#!/usr/bin/env python3
"""
Command-line interface for motion visualization.
"""

import argparse
import sys
from pathlib import Path

from .api import visualize


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize motion from .npy feature files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  motion-viz -i motion.npy
  
  # Focus on hands with front view
  motion-viz -i motion.npy --focus both_hands --view front
  
  # High FPS with downsampling
  motion-viz -i motion.npy --fps 60 --downsample 2
  
  # With normalization and grid
  motion-viz -i motion.npy --norm normalization.npz --grid
        """
    )
    
    parser.add_argument("-i", "--input", required=True,
                       help="Input .npy motion file")
    parser.add_argument("-o", "--output", default=None,
                       help="Output video path (default: input_name.mp4)")
    parser.add_argument("--bvh", default=None,
                       help="Reference BVH file (auto-detected if omitted)")
    parser.add_argument("--norm", default=None,
                       help="Normalization .npz file")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second (default: 30)")
    parser.add_argument("--downsample", type=int, default=1,
                       help="Downsample factor (default: 1)")
    parser.add_argument("--title", default=None,
                       help="Video title")
    
    # View options
    parser.add_argument("--static", action="store_true",
                       help="Use static camera (no tracking)")
    parser.add_argument("--focus", default=None,
                       choices=['both_hands', 'left_hand', 'right_hand',
                               'both_arms', 'left_arm', 'right_arm',
                               'fingers', 'upper_body'],
                       help="Focus on specific body part")
    parser.add_argument("--view", default=None,
                       choices=['front', 'back', 'side', 'left_side',
                               'top', 'front_down', 'three_quarter'],
                       help="Fixed camera angle")
    parser.add_argument("--hand-size", type=float, default=8,
                       help="Point size for hand joints (default: 8)")
    
    # Grid option
    parser.add_argument("--grid", action="store_true",
                       help="Also save frame grid as PNG")
    parser.add_argument("--grid-frames", type=int, default=9,
                       help="Number of frames in grid (default: 9)")
    
    args = parser.parse_args()
    
    try:
        output_path = visualize(
            npy_path=args.input,
            output_path=args.output,
            bvh_path=args.bvh,
            norm_path=args.norm,
            fps=args.fps,
            downsample=args.downsample,
            tracking=not args.static,
            title=args.title,
            focus_joints=args.focus,
            fixed_view=args.view,
            hand_point_size=args.hand_size,
            save_grid=args.grid,
            grid_frames=args.grid_frames
        )
        
        print("\n✓ Done!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())