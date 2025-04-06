# src/generation/dance_from_text.py
import torch
import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def retrieve_dance_by_text(
    text_query: str,
    text_encoder: torch.nn.Module,
    text_tokenizer: callable,
    dance_sequences: np.ndarray,
    dance_embeddings: np.ndarray,
    top_k: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[Tuple[int, float, np.ndarray]]:
    """
    Retrieve dance sequences similar to a text query
    
    Args:
        text_query: Text description to search for
        text_encoder: Trained text encoder model
        text_tokenizer: Function to tokenize text
        dance_sequences: Array of all dance sequences
        dance_embeddings: Pre-computed dance embeddings
        top_k: Number of results to return
        device: Device to run inference on
        
    Returns:
        List of tuples (index, similarity_score, sequence)
    """
    # Tokenize the text query
    tokens, lengths = text_tokenizer([text_query])
    tokens = torch.tensor(tokens, device=device)
    lengths = torch.tensor(lengths, device=device) if lengths is not None else None
    
    # Encode the text query
    text_encoder.eval()
    with torch.no_grad():
        if lengths is not None:
            text_embedding = text_encoder(tokens, lengths)
        else:
            text_embedding = text_encoder(tokens)
        text_embedding = text_embedding.cpu().numpy()
    
    # Convert dance embeddings to numpy if needed
    if isinstance(dance_embeddings, torch.Tensor):
        dance_embeddings = dance_embeddings.cpu().numpy()
    
    # Compute similarities
    similarities = np.dot(dance_embeddings, text_embedding.T).squeeze()
    
    # Get top-k results
    if top_k == 1:
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        return [(best_idx, float(best_score), dance_sequences[best_idx])]
    else:
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append((idx, float(similarities[idx]), dance_sequences[idx]))
        return results

def interpolate_dance_sequences(
    sequences: List[np.ndarray],
    weights: Optional[List[float]] = None
) -> np.ndarray:
    """
    Interpolate between multiple dance sequences
    
    Args:
        sequences: List of dance sequences to interpolate
        weights: Optional weights for each sequence (normalized if not already)
        
    Returns:
        Interpolated dance sequence
    """
    if len(sequences) == 0:
        raise ValueError("No sequences provided for interpolation")
    
    if len(sequences) == 1:
        return sequences[0]
    
    # Ensure all sequences have the same shape
    shapes = [seq.shape for seq in sequences]
    if len(set(shapes)) > 1:
        raise ValueError(f"Sequences have different shapes: {shapes}")
    
    # Default to equal weights if none provided
    if weights is None:
        weights = [1.0 / len(sequences)] * len(sequences)
    else:
        # Normalize weights to sum to 1
        total = sum(weights)
        weights = [w / total for w in weights]
    
    # Check if weights match number of sequences
    if len(weights) != len(sequences):
        raise ValueError(f"Number of weights ({len(weights)}) doesn't match number of sequences ({len(sequences)})")
    
    # Interpolate sequences
    result = np.zeros_like(sequences[0])
    for seq, weight in zip(sequences, weights):
        result += seq * weight
    
    return result