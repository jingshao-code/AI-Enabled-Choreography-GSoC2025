# src/generation/text_from_dance.py
import torch
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)

def retrieve_text_by_dance(
    dance_sequence: np.ndarray,
    dance_encoder: torch.nn.Module,
    text_embeddings: np.ndarray,
    text_labels: List[str],
    top_k: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[Tuple[int, float, str]]:
    """
    Retrieve text descriptions similar to a dance sequence
    
    Args:
        dance_sequence: Input dance sequence
        dance_encoder: Trained dance encoder model
        text_embeddings: Pre-computed text embeddings
        text_labels: List of all text labels
        top_k: Number of results to return
        device: Device to run inference on
        
    Returns:
        List of tuples (index, similarity_score, text)
    """
    # Ensure dance sequence has batch dimension
    if dance_sequence.ndim == 3:
        dance_sequence = np.expand_dims(dance_sequence, axis=0)
    
    # Convert to tensor
    dance_tensor = torch.tensor(dance_sequence, dtype=torch.float32, device=device)
    
    # Encode the dance sequence
    dance_encoder.eval()
    with torch.no_grad():
        dance_embedding = dance_encoder(dance_tensor)
        dance_embedding = dance_embedding.cpu().numpy()
    
    # Convert text embeddings to numpy if needed
    if isinstance(text_embeddings, torch.Tensor):
        text_embeddings = text_embeddings.cpu().numpy()
    
    # Compute similarities
    similarities = np.dot(text_embeddings, dance_embedding.T).squeeze()
    
    # Get top-k results
    if top_k == 1:
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        return [(best_idx, float(best_score), text_labels[best_idx])]
    else:
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append((idx, float(similarities[idx]), text_labels[idx]))
        return results

def generate_composite_description(
    dance_sequence: np.ndarray,
    dance_encoder: torch.nn.Module,
    text_embeddings: np.ndarray,
    text_labels: List[str],
    num_components: int = 3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    """
    Generate a composite text description for a dance sequence
    by combining multiple retrieved descriptions
    
    Args:
        dance_sequence: Input dance sequence
        dance_encoder: Trained dance encoder model
        text_embeddings: Pre-computed text embeddings
        text_labels: List of all text labels
        num_components: Number of text components to combine
        device: Device to run inference on
        
    Returns:
        Composite text description
    """
    # Retrieve top-k similar texts
    top_results = retrieve_text_by_dance(
        dance_sequence, 
        dance_encoder, 
        text_embeddings, 
        text_labels, 
        top_k=num_components,
        device=device
    )
    
    # Extract components and their scores
    descriptions = []
    scores = []
    for _, score, text in top_results:
        descriptions.append(text)
        scores.append(score)
    
    # Normalize scores to use as weights
    total_score = sum(scores)
    if total_score > 0:
        weights = [score / total_score for score in scores]
    else:
        weights = [1.0 / len(scores)] * len(scores)
    
    # Extract movement types and qualities from each description
    movement_words = set()
    quality_words = set()
    
    # Common movement and quality terms (extend as needed)
    movement_terms = ["spin", "turn", "jump", "leap", "reach", "bend", "stretch", 
                     "walk", "run", "kick", "swing", "twist", "roll", "slide"]
    quality_terms = ["slow", "fast", "smooth", "sharp", "fluid", "rigid", "light", 
                     "heavy", "gentle", "powerful", "balanced", "dynamic", "graceful"]
    
    # Extract terms with their weights
    weighted_terms = {}
    
    for desc, weight in zip(descriptions, weights):
        words = desc.lower().split()
        for word in words:
            if word in movement_terms:
                weighted_terms[word] = weighted_terms.get(word, 0) + weight
            elif word in quality_terms:
                weighted_terms[word] = weighted_terms.get(word, 0) + weight
    
    # Sort terms by weight
    sorted_terms = sorted(weighted_terms.items(), key=lambda x: x[1], reverse=True)
    
    # Take top terms (up to 5)
    top_terms = sorted_terms[:5]
    
    # Construct composite description
    if not top_terms:
        return descriptions[0]  # Fallback to top description
    
    # Separate qualities and movements
    qualities = [term for term, _ in top_terms if term in quality_terms]
    movements = [term for term, _ in top_terms if term in movement_terms]
    
    # Construct description
    if qualities and movements:
        description = f"{' and '.join(qualities)} {' and '.join(movements)}"
    elif movements:
        description = f"{' and '.join(movements)}"
    elif qualities:
        description = f"{' and '.join(qualities)} movement"
    else:
        description = descriptions[0]  # Fallback
    
    return description