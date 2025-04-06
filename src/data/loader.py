# src/data/loader.py
import numpy as np
import os
import glob
from typing import List, Optional, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MotionDataLoader:
    """Class for loading and processing dance motion capture data"""
    
    def __init__(self, data_dir: str):
        """
        Initialize the loader with directory containing .npy files
        
        Args:
            data_dir: Directory path containing motion capture .npy files
        """
        self.data_dir = data_dir
        self.data_files = self._find_data_files()
        self.data_cache = {}  # Cache loaded data to avoid reloading
        
        logger.info(f"Found {len(self.data_files)} data files in {data_dir}")
    
    def _find_data_files(self) -> List[str]:
        """Find all .npy files in the data directory"""
        pattern = os.path.join(self.data_dir, "*.npy")
        files = glob.glob(pattern)
        if not files:
            logger.warning(f"No .npy files found in {self.data_dir}")
        return files
    
    def get_file_list(self) -> List[str]:
        """Get list of all available files"""
        return [os.path.basename(f) for f in self.data_files]
    
    def load_file(self, filename: str, normalize: bool = True) -> Optional[np.ndarray]:
        """
        Load a specific .npy file
        
        Args:
            filename: File name to load (without path)
            normalize: Whether to normalize the data
            
        Returns:
            Loaded motion data, or None if file doesn't exist
        """
        # Check cache first
        if filename in self.data_cache:
            logger.info(f"Using cached data for {filename}")
            return self.data_cache[filename]
        
        # Find full path
        filepath = None
        for f in self.data_files:
            if os.path.basename(f) == filename:
                filepath = f
                break
        
        if filepath is None:
            logger.error(f"File {filename} not found in {self.data_dir}")
            return None
        
        try:
            # Load the data
            data = np.load(filepath)
            logger.info(f"Successfully loaded {filename}, shape: {data.shape}")
            
            # Normalize if requested
            if normalize:
                data = self._normalize_data(data)
                logger.info(f"Data normalized")
            
            # Cache data for future use
            self.data_cache[filename] = data
            return data
            
        except Exception as e:
            logger.error(f"Error loading {filename}: {str(e)}")
            return None
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data for better visualization and analysis
        
        This centers the data around origin and scales to appropriate range
        """
        # Calculate center point at each timestep (average of all joints)
        centers = np.mean(data, axis=0)
        
        # Subtract center from each joint position
        centered_data = data - centers[np.newaxis, :, :]
        
        # Scale the data to a reasonable range
        max_abs_val = np.max(np.abs(centered_data))
        scaled_data = centered_data / max_abs_val if max_abs_val > 0 else centered_data
        
        return scaled_data
    
    def get_data_info(self, filename: str) -> Dict[str, Any]:
        """Get basic information about a data file"""
        data = self.load_file(filename, normalize=False)
        if data is None:
            return {}
        
        info = {
            "filename": filename,
            "shape": data.shape,
            "n_joints": data.shape[0],
            "n_frames": data.shape[1],
            "min_value": float(np.min(data)),
            "max_value": float(np.max(data)),
            "mean_value": float(np.mean(data)),
            "std_value": float(np.std(data))
        }
        
        return info