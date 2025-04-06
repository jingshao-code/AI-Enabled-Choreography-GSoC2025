# src/labeling/label_propagation.py
import numpy as np
from typing import List, Dict, Tuple, Any
import logging
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import Counter
import random

logger = logging.getLogger(__name__)

def propagate_labels(features: np.ndarray, labeled_indices: List[int], 
                    labels: List[str], k: int = 5) -> List[str]:
    """
    Propagate labels from a small subset to all sequences using k-nearest neighbors
    
    Args:
        features: Feature vectors for all sequences
        labeled_indices: Indices of manually labeled sequences
        labels: List of labels for labeled sequences
        k: Number of nearest neighbors to consider
        
    Returns:
        List of labels for all sequences
    """
    n_sequences = features.shape[0]
    
    # Get features of labeled sequences
    labeled_features = features[labeled_indices]
    
    # Fit nearest neighbors model on labeled features
    nn_model = NearestNeighbors(n_neighbors=min(k, len(labeled_indices)))
    nn_model.fit(labeled_features)
    
    # Initialize all labels as None
    all_labels = [None] * n_sequences
    
    # Set known labels
    for i, idx in enumerate(labeled_indices):
        all_labels[idx] = labels[i]
    
    # For each unlabeled sequence, propagate label from nearest neighbors
    for i in range(n_sequences):
        if all_labels[i] is None:  # Only process unlabeled sequences
            # Find k nearest neighbors
            distances, indices = nn_model.kneighbors([features[i]], n_neighbors=min(k, len(labeled_indices)))
            
            # Get labels of nearest neighbors
            neighbor_labels = [labels[idx] for idx in indices[0]]
            
            # Assign most common label
            label_counts = Counter(neighbor_labels)
            most_common_label = label_counts.most_common(1)[0][0]
            all_labels[i] = most_common_label
    
    logger.info(f"Propagated {len(labeled_indices)} manual labels to {n_sequences} sequences")
    return all_labels

def suggest_candidates_for_labeling(features: np.ndarray, n_clusters: int = 10, 
                                   n_candidates: int = 20) -> List[int]:
    """
    Suggest diverse sequence candidates for manual labeling
    
    Args:
        features: Feature vectors for all sequences
        n_clusters: Number of clusters to form
        n_candidates: Total number of candidates to suggest
        
    Returns:
        List of suggested sequence indices for manual labeling
    """
    # Apply KMeans clustering to group similar sequences
    kmeans = KMeans(n_clusters=min(n_clusters, features.shape[0]), random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    
    # For each cluster, find the sequence closest to the centroid
    candidates = []
    for cluster_idx in range(n_clusters):
        # Get sequences in this cluster
        cluster_mask = (cluster_labels == cluster_idx)
        if np.sum(cluster_mask) == 0:
            continue
            
        cluster_sequences = np.where(cluster_mask)[0]
        cluster_features = features[cluster_mask]
        
        # Find distance to centroid
        centroid = kmeans.cluster_centers_[cluster_idx]
        distances = np.sqrt(np.sum((cluster_features - centroid)**2, axis=1))
        
        # Sequence closest to centroid
        closest_idx = cluster_sequences[np.argmin(distances)]
        candidates.append(closest_idx)
        
        # Also add a random sequence from the cluster for diversity
        if len(cluster_sequences) > 1:
            random_idx = random.choice([i for i in cluster_sequences if i != closest_idx])
            candidates.append(random_idx)
    
    # If we need more candidates, add random sequences
    if len(candidates) < n_candidates:
        remaining = n_candidates - len(candidates)
        all_indices = set(range(features.shape[0]))
        unused_indices = list(all_indices - set(candidates))
        
        if len(unused_indices) >= remaining:
            additional = random.sample(unused_indices, remaining)
            candidates.extend(additional)
    
    # If we have too many, trim to the requested number
    candidates = candidates[:n_candidates]
    
    logger.info(f"Suggested {len(candidates)} diverse sequences for manual labeling")
    return candidates

def create_manual_labels(candidates: List[int]) -> Dict[int, str]:
    """
    Placeholder for creating manual labels (in practice, this would involve human annotation)
    
    Args:
        candidates: List of sequence indices to label
        
    Returns:
        Dictionary mapping sequence indices to labels
    """
    # In a real implementation, you would show each sequence to a human annotator
    # Here we just create placeholder labels for demonstration
    movement_types = [
        "spin", "jump", "reach", "bend", "walk", "sidestep", "turn", 
        "spiral", "kick", "lunge", "arch", "wave"
    ]
    
    movement_qualities = [
        "smooth", "sharp", "fluid", "rigid", "slow", "fast", 
        "light", "heavy", "balanced", "off-balance"
    ]
    
    # Randomly assign labels for demonstration
    labels = {}
    for idx in candidates:
        # Create a label combining movement type and quality
        movement = random.choice(movement_types)
        quality = random.choice(movement_qualities)
        labels[idx] = f"{quality} {movement}"
    
    return labels