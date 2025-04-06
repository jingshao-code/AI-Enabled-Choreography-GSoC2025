# src/labeling/feature_extractor.py
import numpy as np
from typing import List, Dict, Tuple
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def extract_sequence_features(sequence: np.ndarray) -> np.ndarray:
    """
    Extract meaningful features from a dance sequence
    
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
    # Calculate velocities
    velocities = np.diff(sequence, axis=1)
    mean_vel = np.mean(np.abs(velocities), axis=1)
    features.append(mean_vel.flatten())
    
    # Calculate accelerations
    accels = np.diff(velocities, axis=1) if velocities.shape[1] > 1 else np.zeros((n_joints, 1, n_dims))
    mean_accel = np.mean(np.abs(accels), axis=1)
    features.append(mean_accel.flatten())
    
    # 3. Range of motion
    motion_range = np.max(sequence, axis=1) - np.min(sequence, axis=1)
    features.append(motion_range.flatten())
    
    # 4. Joint coordination features
    # Calculate distance between pairs of joints
    coordination_features = []
    for j1 in range(n_joints):
        for j2 in range(j1+1, n_joints):
            joint_distances = np.sqrt(np.sum((sequence[j1] - sequence[j2])**2, axis=1))
            coordination_features.extend([
                np.mean(joint_distances),
                np.std(joint_distances)
            ])
    
    features.append(np.array(coordination_features))
    
    # Concatenate all features
    return np.concatenate(features)

def compute_all_features(sequences: np.ndarray) -> np.ndarray:
    """
    Compute features for all sequences
    
    Args:
        sequences: Array of sequences with shape (n_sequences, joints, timesteps, dimensions)
        
    Returns:
        Array of feature vectors for all sequences
    """
    n_sequences = sequences.shape[0]
    
    # Extract features for the first sequence to determine feature vector length
    first_features = extract_sequence_features(sequences[0])
    feature_length = len(first_features)
    
    # Initialize features array
    all_features = np.zeros((n_sequences, feature_length))
    all_features[0] = first_features
    
    # Extract features for the remaining sequences
    for i in range(1, n_sequences):
        all_features[i] = extract_sequence_features(sequences[i])
    
    logger.info(f"Computed features for {n_sequences} sequences, feature dim: {feature_length}")
    return all_features

def analyze_principal_components(features: np.ndarray, n_components: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform PCA on sequence features for dimensionality reduction
    
    Args:
        features: Feature vectors for all sequences
        n_components: Number of principal components to compute
        
    Returns:
        Tuple containing:
        - Transformed features
        - Explained variance ratios
        - PCA components
    """
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply PCA
    pca = PCA(n_components=min(n_components, scaled_features.shape[1], scaled_features.shape[0]))
    transformed = pca.fit_transform(scaled_features)
    
    logger.info(f"PCA reduced dimension from {features.shape[1]} to {transformed.shape[1]}")
    logger.info(f"Explained variance: {np.sum(pca.explained_variance_ratio_):.2f}")
    
    return transformed, pca.explained_variance_ratio_, pca.components_