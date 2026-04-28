"""Unit tests for ML models"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.models.random_forest import RandomForestPredictiveModel

class TestRandomForestModel:
    """
    Test cases for Random Forest model.
    """
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        return X, y
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = RandomForestPredictiveModel()
        assert model.model is not None
        assert model.is_trained is False
    
    def test_model_training(self, sample_data):
        """Test model training."""
        X, y = sample_data
        model = RandomForestPredictiveModel(n_estimators=10)
        model.train(X, y)
        assert model.is_trained is True
    
    def test_model_prediction(self, sample_data):
        """Test model prediction."""
        X, y = sample_data
        model = RandomForestPredictiveModel(n_estimators=10)
        model.train(X, y)
        
        predictions = model.predict(X[:5])
        assert predictions.shape[0] == 5
        assert all(p in [0, 1] for p in predictions)
    
    def test_model_evaluation(self, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        model = RandomForestPredictiveModel(n_estimators=10)
        model.train(X, y)
        
        metrics = model.evaluate(X, y)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
