# src/visualization/animator.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.colors as mcolors
import logging

logger = logging.getLogger(__name__)

class DanceAnimator:
    """Class for creating and customizing 3D animations of dance sequences"""
    
    # Default skeleton connections (adjust based on actual data)
    DEFAULT_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3),  # Spine to head
        (1, 4), (4, 5), (5, 6),  # Right arm
        (1, 7), (7, 8), (8, 9),  # Left arm
        (0, 10), (10, 11), (11, 12),  # Right leg
        (0, 13), (13, 14), (14, 15)   # Left leg
    ]
    
    # Joint groups for different body parts
    DEFAULT_JOINT_GROUPS = {
        'head': [2, 3],
        'torso': [0, 1],
        'right_arm': [4, 5, 6],
        'left_arm': [7, 8, 9],
        'right_leg': [10, 11, 12],
        'left_leg': [13, 14, 15]
    }
    
    def __init__(
        self, 
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 100,
        background_color: str = 'white',
        joint_color: Union[str, Dict[str, str]] = 'blue',
        joint_size: Union[int, Dict[str, int]] = 50,
        line_color: str = 'red',
        line_width: int = 2,
        axis_visibility: bool = True,
        view_elevation: int = 20,
        view_azimuth: int = 30,
        draw_floor: bool = True,
        floor_alpha: float = 0.3,
        show_trajectory: bool = False,
        trajectory_joints: List[int] = None,
        title: str = 'Dance Motion Capture Visualization'
    ):
        """Initialize animator with customizable visualization parameters"""
        self.figsize = figsize
        self.dpi = dpi
        self.background_color = background_color
        self.joint_color = joint_color
        self.joint_size = joint_size
        self.line_color = line_color
        self.line_width = line_width
        self.axis_visibility = axis_visibility
        self.view_elevation = view_elevation
        self.view_azimuth = view_azimuth
        self.draw_floor = draw_floor
        self.floor_alpha = floor_alpha
        self.show_trajectory = show_trajectory
        self.trajectory_joints = trajectory_joints if trajectory_joints is not None else [3, 6, 9]  # Default to head and hands
        self.title = title
        
        # Internal state
        self.fig = None
        self.ax = None
        self.scatter = None
        self.lines = None
        self.trajectories = None
        self.floor = None
        
        logger.info("DanceAnimator initialized with custom settings")
    
    def animate_sequence(self, motion_data: np.ndarray, frame_range: Tuple[int, int] = None, 
                         interval: int = 50, save_path: Optional[str] = None, 
                         color_mode: str = 'default'):
        """
        Create animation for motion data sequence
        
        Args:
            motion_data: Motion capture data with shape (n_joints, n_frames, 3)
            frame_range: Range of frames to animate (start, end)
            interval: Interval between frames in milliseconds
            save_path: Path to save animation (if None, animation is not saved)
            color_mode: Coloring method ('default', 'velocity', 'height', 'groups')
            
        Returns:
            The created animation object
        """
        # Setup figure and axes
        self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi, facecolor=self.background_color)
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor=self.background_color)
        
        # Prepare the data
        if frame_range is not None:
            start, end = frame_range
            data = motion_data[:, start:end, :]
        else:
            data = motion_data
        
        n_joints, n_frames, n_dims = data.shape
        
        # Set view angle
        self.ax.view_init(elev=self.view_elevation, azim=self.view_azimuth)
        
        # Set title
        self.ax.set_title(self.title, fontsize=14, pad=20)
        
        # Initialize scatter plot for joints with initial positions
        joint_positions = data[:, 0, :]
        self.scatter = self.ax.scatter(
            joint_positions[:, 0], 
            joint_positions[:, 1], 
            joint_positions[:, 2], 
            c=self.joint_color, 
            s=self.joint_size, 
            marker='o'
        )
        
        # Set axis limits
        self._set_axis_limits(data)
        
        # Draw floor if requested
        if self.draw_floor:
            self._add_floor(data)
        
        # Initialize lines for skeleton
        self.lines = []
        for joint1, joint2 in self.DEFAULT_CONNECTIONS:
            if joint1 < n_joints and joint2 < n_joints:  # Ensure joints exist
                line, = self.ax.plot(
                    [joint_positions[joint1, 0], joint_positions[joint2, 0]],
                    [joint_positions[joint1, 1], joint_positions[joint2, 1]],
                    [joint_positions[joint1, 2], joint_positions[joint2, 2]],
                    color=self.line_color,
                    linewidth=self.line_width
                )
                self.lines.append(line)
        
        # Initialize trajectories if requested
        if self.show_trajectory:
            self.trajectories = []
            for joint_idx in self.trajectory_joints:
                if joint_idx < n_joints:
                    trajectory, = self.ax.plot([], [], [], 'o-', alpha=0.5, markersize=2)
                    self.trajectories.append((joint_idx, trajectory))
        
        # Animation update function
        def update(frame):
            # Update joint positions
            new_positions = data[:, frame, :]
            self.scatter._offsets3d = (new_positions[:, 0], new_positions[:, 1], new_positions[:, 2])
            
            # Update skeleton lines
            for i, (joint1, joint2) in enumerate(self.DEFAULT_CONNECTIONS):
                if i < len(self.lines) and joint1 < n_joints and joint2 < n_joints:
                    self.lines[i].set_data(
                        [new_positions[joint1, 0], new_positions[joint2, 0]],
                        [new_positions[joint1, 1], new_positions[joint2, 1]]
                    )
                    self.lines[i].set_3d_properties(
                        [new_positions[joint1, 2], new_positions[joint2, 2]]
                    )
            
            # Update trajectories if showing
            if self.show_trajectory and frame > 0:
                for idx, (joint_idx, trajectory) in enumerate(self.trajectories):
                    # Get trajectory history (last N frames)
                    history_length = min(20, frame + 1)
                    history_frames = range(frame - history_length + 1, frame + 1)
                    
                    # Extract trajectory points
                    x = data[joint_idx, history_frames, 0]
                    y = data[joint_idx, history_frames, 1]
                    z = data[joint_idx, history_frames, 2]
                    
                    trajectory.set_data(x, y)
                    trajectory.set_3d_properties(z)
            
            return [self.scatter] + self.lines + [traj for _, traj in self.trajectories] if self.show_trajectory else []
        
        # Create animation
        ani = animation.FuncAnimation(
            self.fig, update, frames=n_frames, interval=interval, blit=True
        )
        
        # Save animation if requested
        if save_path:
            ani.save(save_path, writer='pillow' if save_path.endswith('.gif') else 'ffmpeg', dpi=self.dpi)
            logger.info(f"Animation saved to {save_path}")
        
        return ani
    
    def _set_axis_limits(self, data: np.ndarray):
        """Set axis limits based on data range"""
        # Calculate data range
        min_val = np.min(data)
        max_val = np.max(data)
        range_val = max_val - min_val
        
        # Add buffer to ensure all points are visible
        buffer = range_val * 0.1
        
        # Set axis limits
        self.ax.set_xlim([min_val - buffer, max_val + buffer])
        self.ax.set_ylim([min_val - buffer, max_val + buffer])
        self.ax.set_zlim([min_val - buffer, max_val + buffer])
        
        # Control axis visibility
        if not self.axis_visibility:
            self.ax.set_axis_off()
    
    def _add_floor(self, data: np.ndarray):
        """Add a floor plane beneath the dancer"""
        # Find minimum z-value (lowest point)
        min_z = np.min(data[:, :, 2])
        
        # Get x and y ranges for the floor
        min_x, max_x = np.min(data[:, :, 0]), np.max(data[:, :, 0])
        min_y, max_y = np.min(data[:, :, 1]), np.max(data[:, :, 1])
        
        # Add some padding
        padding = max(max_x - min_x, max_y - min_y) * 0.2
        min_x -= padding
        max_x += padding
        min_y -= padding
        max_y += padding
        
        # Create a grid for the floor
        xx, yy = np.meshgrid([min_x, max_x], [min_y, max_y])
        zz = np.ones_like(xx) * (min_z - 0.05)  # Place slightly below lowest point
        
        # Plot the floor with a grid
        self.floor = self.ax.plot_surface(
            xx, yy, zz, 
            alpha=self.floor_alpha,
            color='gray', 
            rstride=1, cstride=1, 
            edgecolors='darkgray',
            linewidth=0.5,
            antialiased=True
        )