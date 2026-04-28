"""Random Forest model implementation"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .base_model import BaseModel

class RandomForestPredictiveModel(BaseModel):
    """
    Random Forest classifier for predictive maintenance.
    """
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 15, 
                 min_samples_split: int = 5, min_samples_leaf: int = 2,
                 random_state: int = 42):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples in leaf
            random_state: Random state for reproducibility
        """
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
    
    def train(self, X_train, y_train):
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Random Forest model trained successfully")
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get probability predictions.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1': f1_score(y_test, predictions),
            'auc': roc_auc_score(y_test, probabilities)
        }
        
        return metrics
    
    def get_feature_importance(self):
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        return self.model.feature_importances_
