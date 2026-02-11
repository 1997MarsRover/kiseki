# Motion Viz 🎬

Lightweight motion visualization package - convert `.npy` motion feature files to animated skeleton videos with **minimal dependencies**.

## Features ✨

- **Minimal Dependencies**: Only requires `numpy` and `matplotlib` (removed torch and pymotion dependencies)
- **Easy to Use**: Simple Python API and CLI
- **Flexible Views**: Focus on specific body parts, set camera angles
- **Fast**: Optimized rendering with downsampling support
- **Pip Installable**: Install directly from PyPI or GitHub

## Installation 📦

```bash
# Install from PyPI (when published)
pip install motion-viz

# Or install from source
pip install .

# Development installation
pip install -e ".[dev]"
```

## Quick Start 🚀

### Python API

```python
from motion_viz import visualize

# Basic usage
visualize("motion.npy")

# Focus on hands with front view
visualize("motion.npy", 
          focus_joints='both_hands', 
          fixed_view='front',
          fps=60)

# With normalization
visualize("motion.npy",
          norm_path="normalization.npz",
          save_grid=True)
```

### Command Line

```bash
# Basic usage
motion-viz -i motion.npy

# Focus on hands with front view
motion-viz -i motion.npy --focus both_hands --view front

# High FPS with downsampling and grid
motion-viz -i motion.npy --fps 60 --downsample 2 --grid

# With normalization
motion-viz -i motion.npy --norm normalization.npz
```

## API Reference 📚

### `visualize()`

Main function for creating motion visualizations.

**Parameters:**
- `npy_path` (str|Path): Input .npy motion file
- `output_path` (str|Path, optional): Output video path (default: input_name.mp4)
- `bvh_path` (str|Path, optional): Reference BVH file (auto-detected if None)
- `norm_path` (str|Path, optional): Normalization file
- `fps` (int): Frames per second (default: 30)
- `downsample` (int): Downsample factor for faster rendering (default: 1)
- `tracking` (bool): Camera follows root joint (default: True)
- `title` (str, optional): Video title
- `focus_joints` (list|str, optional): Focus on specific joints
  - Options: `'both_hands'`, `'left_hand'`, `'right_hand'`, `'both_arms'`, `'fingers'`, `'upper_body'`
- `fixed_view` (tuple|str, optional): Fixed camera view
  - Presets: `'front'`, `'side'`, `'top'`, `'front_down'`, `'three_quarter'`
  - Or tuple: `(elevation, azimuth)` in degrees
- `hand_point_size` (float): Point size for hand joints (default: 8)
- `save_grid` (bool): Also save frame grid image (default: False)
- `grid_frames` (int): Number of frames in grid (default: 9)

**Returns:** Path to saved video

## Advanced Usage 🔧

### Focus Groups

Available focus groups in `JOINT_GROUPS`:

```python
from motion_viz import JOINT_GROUPS

print(JOINT_GROUPS.keys())
# dict_keys(['left_hand', 'right_hand', 'both_hands', 
#            'left_arm', 'right_arm', 'both_arms', 
#            'upper_body', 'fingers'])
```

### Custom Joint Selection

```python
# Focus on specific joint indices
visualize("motion.npy", focus_joints=[20, 21, 22, 39, 40, 41])
```

### Camera Views

```python
# Preset views
visualize("motion.npy", fixed_view='front')
visualize("motion.npy", fixed_view='three_quarter')

# Custom angle (elevation, azimuth)
visualize("motion.npy", fixed_view=(30, 90))
```

## Dependency Comparison 📊

**Original Script:**
- numpy
- matplotlib
- torch ❌ (removed)
- pymotion ❌ (removed, lightweight BVH parser built-in)

**Motion Viz Package:**
- numpy ✅
- matplotlib ✅

**Reduction:** From 4 dependencies to 2 core dependencies!

## File Structure 📁

```
motion_viz/
├── __init__.py          # Package initialization
├── core.py              # Core functions (quaternions, reconstruction, BVH parser)
├── visualize.py         # Visualization functions
├── api.py               # Main API
└── cli.py               # Command-line interface
```

## Requirements 📋

- Python >= 3.8
- numpy >= 1.20.0
- matplotlib >= 3.3.0

## License 📄

MIT License

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog 📝

### v0.1.0 (Initial Release)
- Removed torch dependency (pure numpy quaternion operations)
- Removed pymotion dependency (built-in lightweight BVH parser)
- Modular package structure
- Comprehensive CLI and Python API
- Focus views and camera presets
- Frame grid generation