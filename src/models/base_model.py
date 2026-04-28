"""Base model class for all ML models"""

from abc import ABC, abstractmethod
import joblib
from pathlib import Path

class BaseModel(ABC):
    """
    Abstract base class for all predictive models.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize base model.
        
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        pass
    
    @abstractmethod
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    def save_model(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
        """
        self.model = joblib.load(path)
        self.is_trained = True
        print(f"Model loaded from {path}")
