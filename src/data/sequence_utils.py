# src/data/sequence_utils.py
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

def create_sequences(motion_data: np.ndarray, 
                    seq_length: int = 30, 
                    stride: int = 10,
                    min_motion: float = 0.01) -> Tuple[np.ndarray, List[int]]:
    """
    Split motion data into fixed-length sequences
    
    Args:
        motion_data: Motion data with shape (joints, timesteps, dimensions)
        seq_length: Length of each sequence (number of timesteps)
        stride: Number of timesteps to move forward for each new sequence
        min_motion: Minimum motion threshold to filter static sequences
        
    Returns:
        Tuple containing:
        - Sequence array with shape (n_sequences, joints, seq_length, dimensions)
        - List of start indices for each sequence
    """
    n_joints, n_frames, n_dims = motion_data.shape
    
    # Calculate how many sequences we can extract
    n_sequences = max(0, (n_frames - seq_length) // stride + 1)
    logger.info(f"Creating {n_sequences} sequences of length {seq_length} with stride {stride}")
    
    if n_sequences == 0:
        logger.warning(f"Cannot create sequences: data has {n_frames} frames but seq_length is {seq_length}")
        return np.empty((0, n_joints, seq_length, n_dims)), []
    
    # Create empty array for sequences
    sequences = np.zeros((n_sequences, n_joints, seq_length, n_dims))
    sequence_indices = []
    
    # Extract sequences
    valid_count = 0
    for i in range(n_sequences):
        start_idx = i * stride
        end_idx = start_idx + seq_length
        
        # Extract sequence
        seq = motion_data[:, start_idx:end_idx, :]
        
        # Calculate motion amount (average displacement between adjacent frames)
        motion_amount = np.mean(np.sqrt(np.sum(np.diff(seq, axis=1)**2, axis=2)))
        
        # Only keep sequences with sufficient motion
        if motion_amount >= min_motion:
            sequences[valid_count] = seq
            sequence_indices.append(start_idx)
            valid_count += 1
    
    # If some sequences were filtered, resize array
    if valid_count < n_sequences:
        logger.info(f"Filtered out {n_sequences - valid_count} sequences with insufficient motion")
        sequences = sequences[:valid_count]
        
    return sequences, sequence_indices

def extract_features(sequence: np.ndarray) -> np.ndarray:
    """
    Extract features from a dance sequence to enable comparison
    
    Args:
        sequence: Dance sequence with shape (joints, timesteps, dimensions)
        
    Returns:
        Feature vector representing key characteristics of the sequence
    """
    n_joints, seq_length, n_dims = sequence.shape
    features = []
    
    # 1. Position-based features
    # Mean position of each joint
    mean_pos = np.mean(sequence, axis=1)
    features.append(mean_pos.flatten())
    
    # 2. Motion-based features
    # Calculate velocities (first derivative)
    velocities = np.diff(sequence, axis=1)
    mean_vel = np.mean(np.abs(velocities), axis=1)
    features.append(mean_vel.flatten())
    
    # Calculate accelerations (second derivative)
    accels = np.diff(velocities, axis=1)
    mean_accel = np.mean(np.abs(accels), axis=1)
    features.append(mean_accel.flatten())
    
    # 3. Range of motion for each joint
    motion_range = np.max(sequence, axis=1) - np.min(sequence, axis=1)
    features.append(motion_range.flatten())
    
    # Concatenate all features
    return np.concatenate(features)

def augment_sequence(sequence: np.ndarray, augmentation_type: str = 'mirror') -> np.ndarray:
    """
    Apply data augmentation to a sequence
    
    Args:
        sequence: Dance sequence with shape (joints, timesteps, dimensions)
        augmentation_type: Type of augmentation to apply
            - 'mirror': Mirror along X axis
            - 'reverse': Reverse time dimension
            - 'rotate': Rotate around vertical axis
            - 'noise': Add small random noise
        
    Returns:
        Augmented sequence with same shape as input
    """
    augmented = sequence.copy()
    
    if augmentation_type == 'mirror':
        # Mirror along X axis
        augmented[:, :, 0] = -augmented[:, :, 0]
    
    elif augmentation_type == 'reverse':
        # Reverse time dimension
        augmented = augmented[:, ::-1, :]
    
    elif augmentation_type == 'rotate':
        # Rotate around vertical axis (Z)
        theta = np.radians(90)
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        
        for j in range(augmented.shape[0]):
            for t in range(augmented.shape[1]):
                augmented[j, t] = rot_matrix @ augmented[j, t]
    
    elif augmentation_type == 'noise':
        # Add small random noise
        noise_level = 0.02
        noise = np.random.normal(0, noise_level, augmented.shape)
        augmented += noise
    
    return augmented