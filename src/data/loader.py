"""Data loading utilities for predictive maintenance system"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

class DataLoader:
    """
    Handles data loading and initial preprocessing for the predictive maintenance system.
    """
    
    def __init__(self, data_path: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to the CSV data file
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
        """
        self.data_path = Path(data_path)
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            DataFrame containing the loaded data
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        self.data = pd.read_csv(self.data_path)
        print(f"Data loaded successfully. Shape: {self.data.shape}")
        return self.data
    
    def get_basic_info(self) -> dict:
        """
        Get basic information about the loaded data.
        
        Returns:
            Dictionary with data info
        """
        if self.data is None:
            self.load_data()
        
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'statistics': self.data.describe().to_dict()
        }
    
    def get_train_test_split(self, target_column: str = 'failure'):
        """
        Split data into training and testing sets.
        
        Args:
            target_column: Name of the target column
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            self.load_data()
        
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"Train set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_data_summary(self) -> str:
        """
        Get a summary of the data.
        
        Returns:
            Summary string
        """
        if self.data is None:
            self.load_data()
        
        return self.data.describe().to_string()
