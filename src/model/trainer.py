# src/model/trainer.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging
import time
from tqdm import tqdm

from src.model.encoders import DanceEncoder, TextEncoder, SimpleTextEncoder
from src.model.loss import NTXentLoss

logger = logging.getLogger(__name__)

class ContrastiveTrainer:
    """
    Trainer for dance-text contrastive learning model
    """
    def __init__(
        self,
        dance_encoder: torch.nn.Module,
        text_encoder: torch.nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        temperature: float = 0.07
    ):
        """
        Initialize the trainer
        
        Args:
            dance_encoder: Dance sequence encoder model
            text_encoder: Text description encoder model
            device: Device to run training on ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            temperature: Temperature parameter for contrastive loss
        """
        self.device = device
        
        # Move models to device
        self.dance_encoder = dance_encoder.to(device)
        self.text_encoder = text_encoder.to(device)
        
        # Set up loss function
        self.criterion = NTXentLoss(temperature=temperature).to(device)
        
        # Set up optimizer
        self.optimizer = optim.Adam(
            list(dance_encoder.parameters()) + list(text_encoder.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train the model for one epoch
        
        Args:
            train_loader: DataLoader with training data
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.dance_encoder.train()
        self.text_encoder.train()
        
        total_loss = 0.0
        total_acc = 0.0
        
        # Progress bar
        pbar = tqdm(train_loader, desc="Training")
        
        for batch in pbar:
            # Get batch data
            dance_seqs, text_data, text_lengths = batch
            
            # Move data to device
            dance_seqs = dance_seqs.to(self.device)
            text_data = text_data.to(self.device)
            if text_lengths is not None:
                text_lengths = text_lengths.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass through encoders
            dance_embeddings = self.dance_encoder(dance_seqs)
            if text_lengths is not None:
                text_embeddings = self.text_encoder(text_data, text_lengths)
            else:
                text_embeddings = self.text_encoder(text_data)
            
            # Compute loss
            loss, d2t_loss, t2d_loss = self.criterion(dance_embeddings, text_embeddings)
            
            # Compute accuracy (percentage of correct top-1 predictions)
            accuracy = self._compute_accuracy(dance_embeddings, text_embeddings)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            total_acc += accuracy.item()
            
            # Update progress bar
            pbar.set_postfix(loss=loss.item(), acc=accuracy.item())
        
        # Calculate averages
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_acc / len(train_loader)
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model
        
        Args:
            val_loader: DataLoader with validation data
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.dance_encoder.eval()
        self.text_encoder.eval()
        
        total_loss = 0.0
        total_acc = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Get batch data
                dance_seqs, text_data, text_lengths = batch
                
                # Move data to device
                dance_seqs = dance_seqs.to(self.device)
                text_data = text_data.to(self.device)
                if text_lengths is not None:
                    text_lengths = text_lengths.to(self.device)
                
                # Forward pass through encoders
                dance_embeddings = self.dance_encoder(dance_seqs)
                if text_lengths is not None:
                    text_embeddings = self.text_encoder(text_data, text_lengths)
                else:
                    text_embeddings = self.text_encoder(text_data)
                
                # Compute loss
                loss, _, _ = self.criterion(dance_embeddings, text_embeddings)
                
                # Compute accuracy
                accuracy = self._compute_accuracy(dance_embeddings, text_embeddings)
                
                # Update statistics
                total_loss += loss.item()
                total_acc += accuracy.item()
        
        # Calculate averages
        avg_loss = total_loss / len(val_loader)
        avg_acc = total_acc / len(val_loader)
        
        return avg_loss, avg_acc
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None,
        callback: Optional[Callable[[Dict], None]] = None
    ) -> Dict:
        """
        Train the model for multiple epochs
        
        Args:
            train_loader: DataLoader with training data
            val_loader: Optional DataLoader with validation data
            num_epochs: Number of epochs to train
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            save_path: Optional path to save the best model
            callback: Optional callback function called after each epoch
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate if validation data is provided
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
            else:
                val_loss, val_acc = train_loss, train_acc
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_accuracy'].append(val_acc)
            
            # Log progress
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s - "
                       f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
                       f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
            # Check for improvement for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model if path is provided
                if save_path is not None:
                    self._save_model(save_path)
                    logger.info(f"Model saved to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping after {epoch+1} epochs")
                    break
            
            # Call callback if provided
            if callback is not None:
                callback({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'best_val_loss': best_val_loss
                })
        
        return self.history
    
    def _compute_accuracy(self, dance_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute accuracy of matching dance to text
        
        Args:
            dance_embeddings: Dance embeddings (batch_size, embedding_dim)
            text_embeddings: Text embeddings (batch_size, embedding_dim)
            
        Returns:
            Top-1 accuracy
        """
        # Compute similarity matrix
        similarity = torch.matmul(dance_embeddings, text_embeddings.T)
        
        # Get predictions (indices of highest similarity for each dance)
        predictions = torch.argmax(similarity, dim=1)
        
        # Ground truth is the diagonal (paired dance-text)
        targets = torch.arange(dance_embeddings.size(0), device=dance_embeddings.device)
        
        # Compute accuracy
        correct = (predictions == targets).float().sum()
        accuracy = correct / dance_embeddings.size(0)
        
        return accuracy
    
    def _save_model(self, path: str):
        """
        Save model parameters to file
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'dance_encoder': self.dance_encoder.state_dict(),
            'text_encoder': self.text_encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'history': self.history
        }, path)
    
    def load_model(self, path: str):
        """
        Load model parameters from file
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.dance_encoder.load_state_dict(checkpoint['dance_encoder'])
        self.text_encoder.load_state_dict(checkpoint['text_encoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.history = checkpoint['history']
        
        logger.info(f"Model loaded from {path}")
    
    def compute_embeddings(self, loader: DataLoader, type: str = 'dance') -> Tuple[np.ndarray, List]:
        """
        Compute embeddings for all items in the loader
        
        Args:
            loader: DataLoader with data
            type: Type of embeddings to compute ('dance' or 'text')
            
        Returns:
            Tuple of (embeddings array, original items)
        """
        if type not in ['dance', 'text']:
            raise ValueError(f"Invalid embedding type: {type}. Must be 'dance' or 'text'")
        
        # Set models to evaluation mode
        self.dance_encoder.eval()
        self.text_encoder.eval()
        
        all_embeddings = []
        all_items = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Computing {type} embeddings"):
                # Get batch data
                dance_seqs, text_data, text_lengths = batch
                
                if type == 'dance':
                    # Move data to device
                    dance_seqs = dance_seqs.to(self.device)
                    
                    # Compute dance embeddings
                    embeddings = self.dance_encoder(dance_seqs)
                    
                    # Store original items
                    for seq in dance_seqs.cpu().numpy():
                        all_items.append(seq)
                else:  # text
                    # Move data to device
                    text_data = text_data.to(self.device)
                    if text_lengths is not None:
                        text_lengths = text_lengths.to(self.device)
                    
                    # Compute text embeddings
                    if text_lengths is not None:
                        embeddings = self.text_encoder(text_data, text_lengths)
                    else:
                        embeddings = self.text_encoder(text_data)
                    
                    # Store original items
                    for text in text_data.cpu().numpy():
                        all_items.append(text)
                
                # Collect embeddings
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate all embeddings
        return np.concatenate(all_embeddings), all_items