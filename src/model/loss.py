# src/model/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
    This is a form of contrastive loss used for self-supervised learning
    """
    def __init__(self, temperature: float = 0.07):
        """
        Initialize the loss function
        
        Args:
            temperature: Temperature parameter to scale the similarity scores
        """
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, 
                dance_embeddings: torch.Tensor, 
                text_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute bidirectional contrastive loss between dance and text embeddings
        
        Args:
            dance_embeddings: Embeddings from dance encoder (batch_size, embedding_dim)
            text_embeddings: Embeddings from text encoder (batch_size, embedding_dim)
            
        Returns:
            Tuple containing:
            - Total loss (dance-to-text + text-to-dance)
            - Dance-to-text loss
            - Text-to-dance loss
        """
        # Compute similarity matrix
        # Each element [i,j] represents similarity between i-th dance and j-th text
        similarity = torch.matmul(dance_embeddings, text_embeddings.T) / self.temperature
        
        # Ground truth is the diagonal (paired dance-text)
        batch_size = dance_embeddings.shape[0]
        labels = torch.arange(batch_size, device=dance_embeddings.device)
        
        # Compute loss in both directions
        dance_to_text_loss = self.criterion(similarity, labels)  # Dance -> text
        text_to_dance_loss = self.criterion(similarity.T, labels)  # Text -> dance
        
        # Total loss is the average of the two directions
        total_loss = (dance_to_text_loss + text_to_dance_loss) / 2
        
        return total_loss, dance_to_text_loss, text_to_dance_loss


class TripletLoss(nn.Module):
    """
    Triplet loss for contrastive learning
    Pushes anchor close to positive and far from negative examples
    """
    def __init__(self, margin: float = 1.0):
        """
        Initialize the triplet loss
        
        Args:
            margin: Minimum desired distance between positive and negative pairs
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, 
                anchor: torch.Tensor, 
                positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss
        
        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            negative: Negative embeddings (batch_size, embedding_dim)
            
        Returns:
            Triplet loss value
        """
        # Distance between anchor and positive
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        
        # Distance between anchor and negative
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        # Compute triplet loss with margin
        losses = F.relu(pos_dist - neg_dist + self.margin)
        
        # Return mean loss
        return losses.mean()