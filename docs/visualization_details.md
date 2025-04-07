# Dance Visualization Details

## Overview
This document explains the technical implementation of the 3D visualization system for dance motion capture data. The system transforms raw `.npy` motion capture files into intuitive, animated 3D visualizations that display a dancer's movements over time.

## Data Structure
The motion capture data files have the shape `(# joints, # timesteps, # dimensions)`:
- **Joints**: Points on the body captured by the motion capture system (typically 20-25 points)
- **Timesteps**: Sequential frames of the motion capture session
- **Dimensions**: 3D coordinates (x, y, z) for each joint

## Core Visualization Components

### 1. Skeleton Representation
The visualization connects joints to form a human skeleton using predefined connections:

```python
# Define connections between joints to form a skeleton
SKELETON_CONNECTIONS = [
    # Torso
    (8, 9), (9, 10),
    # Arms
    (8, 14), (14, 15), (15, 16),  # Left arm
    (8, 11), (11, 12), (12, 13),  # Right arm
    # Legs
    (0, 1), (1, 2), (2, 3),       # Right leg
    (0, 4), (4, 5), (5, 6),       # Left leg
    # Spine to head
    (8, 7)
]
```

### 2. Animation Function
The main animation function processes sequences frame by frame:

```python
def animate_dance_sequence(motion_data, interval=50, save_gif=False, output_path=None):
    """
    Create and display an animation of dance motion capture data
    
    Parameters:
    motion_data: numpy array of shape (joints, timesteps, 3)
    interval: time between frames in milliseconds
    save_gif: whether to save the animation as a GIF
    output_path: path to save the GIF (if save_gif is True)
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initial plot setup
    set_axes_equal(ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Create the animation
    num_frames = motion_data.shape[1]
    ani = FuncAnimation(
        fig, update_frame, frames=num_frames, 
        fargs=(motion_data, ax, SKELETON_CONNECTIONS),
        interval=interval, blit=False
    )
    
    if save_gif and output_path:
        ani.save(output_path, writer='pillow', fps=1000/interval)
    
    return ani
```

### 3. Frame Update Function
This function updates the skeleton position for each frame:

```python
def update_frame(frame_idx, motion_data, ax, connections):
    """Update the plot for a specific frame"""
    ax.clear()
    
    # Get the current frame's joint positions
    joint_positions = motion_data[:, frame_idx, :]
    
    # Plot the joints as points
    ax.scatter(
        joint_positions[:, 0], 
        joint_positions[:, 1], 
        joint_positions[:, 2], 
        c=JOINT_COLORS, 
        s=50
    )
    
    # Draw the skeleton connections
    for connection in connections:
        start_idx, end_idx = connection
        ax.plot(
            [joint_positions[start_idx, 0], joint_positions[end_idx, 0]],
            [joint_positions[start_idx, 1], joint_positions[end_idx, 1]],
            [joint_positions[start_idx, 2], joint_positions[end_idx, 2]],
            color='black', linewidth=2
        )
    
    # Set consistent axes limits
    set_axes_limits(ax, motion_data)
    set_axes_equal(ax)
    
    return ax
```

## Enhanced Visualization Features

### 1. Color Coding
Different body parts are color-coded for easier interpretation:

```python
# Color map for different body parts
JOINT_COLORS = [
    'blue',   # 0: Pelvis 
    'blue',   # 1: Right hip
    'blue',   # 2: Right knee
    'blue',   # 3: Right foot
    'green',  # 4: Left hip
    'green',  # 5: Left knee
    'green',  # 6: Left foot
    'red',    # 7: Head
    'purple', # 8: Spine
    # ... and so on for other joints
]
```

### 2. Motion Trajectories
Optional trajectory lines show the path of key joints:

```python
def add_trajectories(ax, motion_data, tracked_joints=[7, 13, 16], frames=30):
    """Add motion trajectory lines for specified joints"""
    for joint_idx in tracked_joints:
        # Get recent positions of the joint
        traj_start = max(0, motion_data.shape[1] - frames)
        trajectory = motion_data[joint_idx, traj_start:, :]
        
        # Plot the trajectory
        ax.plot(
            trajectory[:, 0], 
            trajectory[:, 1], 
            trajectory[:, 2],
            linewidth=1.5,
            alpha=0.7,
            color=JOINT_COLORS[joint_idx]
        )
```

### 3. Multiple View Angles
The visualization includes functions to view the dance from different angles:

```python
def rotate_view(ax, elevation, azimuth):
    """Change the viewpoint of the 3D plot"""
    ax.view_init(elev=elevation, azim=azimuth)
    
def create_multi_angle_view(motion_data, frame_idx, angles=[(30, 0), (0, 90), (0, 180)]):
    """Create a figure with multiple viewing angles of the same frame"""
    fig = plt.figure(figsize=(15, 5))
    
    for i, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        plot_frame(motion_data, frame_idx, ax)
        rotate_view(ax, elev, azim)
        ax.set_title(f"Elevation: {elev}°, Azimuth: {azim}°")
    
    plt.tight_layout()
    return fig
```

## Optimization Techniques

1. **Dynamic Axes Scaling**: Automatically adjusts axes limits based on the dancer's movement range
2. **Equal Aspect Ratio**: Ensures the skeleton isn't distorted in the visualization
3. **Memory Management**: Efficiently clears previous frames to prevent memory issues with long sequences

## Usage Example

```python
# Load motion capture data
motion_data = np.load('dance_sequence.npy')

# Create and display animation
animation = animate_dance_sequence(
    motion_data,
    interval=50,  # 20 frames per second
    save_gif=True,
    output_path='dance_visualization.gif'
)

# Display in Jupyter notebook
from IPython.display import display, HTML
display(HTML(animation.to_jshtml()))
```

The visualization system successfully converts abstract motion capture data into intuitive animations that preserve the dancer's movements, making it easier for researchers and artists to analyze and understand dance sequences.