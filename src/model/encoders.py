# src/model/encoders.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DanceEncoder(nn.Module):
    """
    Encoder for dance sequences to embed them into a shared space with text
    """
    def __init__(self, n_joints: int, seq_length: int, n_dims: int, embedding_dim: int = 128):
        """
        Initialize the dance encoder
        
        Args:
            n_joints: Number of joints in motion capture data
            seq_length: Length of sequence (number of frames)
            n_dims: Dimensionality of each joint position (typically 3 for x,y,z)
            embedding_dim: Size of the output embedding
        """
        super().__init__()
        
        # Calculate input dimension
        self.input_dim = n_joints * n_dims  # Flatten joints and dimensions
        self.seq_length = seq_length
        
        # Temporal convolution to capture motion patterns
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1)  # Global pooling
        )
        
        # Projection to embedding space
        self.projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder
        
        Args:
            x: Input tensor of shape (batch_size, n_joints, seq_length, n_dims)
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        batch_size = x.shape[0]
        
        # Reshape: (batch_size, n_joints, seq_length, n_dims) -> (batch_size, n_joints * n_dims, seq_length)
        x = x.reshape(batch_size, -1, self.seq_length)
        
        # Apply convolution layers
        x = self.conv_layers(x)  # -> (batch_size, 256, 1)
        x = x.squeeze(-1)  # -> (batch_size, 256)
        
        # Project to embedding space
        x = self.projection(x)  # -> (batch_size, embedding_dim)
        
        # Normalize embedding to unit length
        x = F.normalize(x, p=2, dim=1)
        
        return x


class TextEncoder(nn.Module):
    """
    Encoder for text descriptions to embed them into a shared space with dance
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256):
        """
        Initialize the text encoder
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Size of the output embedding (should match dance encoder)
            hidden_dim: Size of hidden layers
        """
        super().__init__()
        
        # Word embedding
        self.embedding = nn.Embedding(vocab_size, hidden_dim // 2)
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            hidden_dim // 2, 
            hidden_dim // 2, 
            bidirectional=True, 
            batch_first=True
        )
        
        # Projection to embedding space
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder
        
        Args:
            x: Input tensor of shape (batch_size, seq_length) containing token indices
            lengths: Tensor of shape (batch_size) containing sequence lengths
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        # Word embedding
        embedded = self.embedding(x)  # -> (batch_size, seq_length, hidden_dim//2)
        
        # Pack padded sequences for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Process with GRU
        _, hidden = self.gru(packed)  # hidden: (2, batch_size, hidden_dim//2)
        
        # Concat forward and backward hidden states
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)  # -> (batch_size, hidden_dim)
        
        # Project to embedding space
        x = self.projection(hidden)  # -> (batch_size, embedding_dim)
        
        # Normalize embedding to unit length
        x = F.normalize(x, p=2, dim=1)
        
        return x


class SimpleTextEncoder(nn.Module):
    """
    A simpler text encoder for cases when tokenization is handled separately
    """
    def __init__(self, input_dim: int, embedding_dim: int = 128):
        """
        Initialize the simple text encoder
        
        Args:
            input_dim: Dimension of input text representation (e.g., TF-IDF or BoW)
            embedding_dim: Size of the output embedding
        """
        super().__init__()
        
        # MLP for embedding
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
        # Apply MLP
        x = self.layers(x)
        
        # Normalize embedding to unit length
        x = F.normalize(x, p=2, dim=1)
        
        return x