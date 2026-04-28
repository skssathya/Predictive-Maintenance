"""Data preprocessing module for predictive maintenance"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    """
    Handles data preprocessing including cleaning, scaling, and feature transformation.
    """
    
    def __init__(self, scaling_method: str = 'standard', missing_strategy: str = 'mean'):
        """
        Initialize DataPreprocessor.
        
        Args:
            scaling_method: 'standard' or 'minmax'
            missing_strategy: 'mean', 'median', or 'forward_fill'
        """
        self.scaling_method = scaling_method
        self.missing_strategy = missing_strategy
        self.scaler = None
        self.imputer = None
        
    def handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        Args:
            X: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        if self.missing_strategy == 'forward_fill':
            X = X.fillna(method='ffill').fillna(method='bfill')
        else:
            strategy = 'mean' if self.missing_strategy == 'mean' else 'median'
            self.imputer = SimpleImputer(strategy=strategy)
            X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        
        return X
    
    def remove_outliers(self, X: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers from the data.
        
        Args:
            X: Input DataFrame
            method: 'iqr' or 'zscore'
            
        Returns:
            DataFrame without outliers
        """
        if method == 'iqr':
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            X = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
        elif method == 'zscore':
            from scipy import stats
            X = X[(np.abs(stats.zscore(X)) < 3).all(axis=1)]
        
        return X
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale features using specified method.
        
        Args:
            X: Input DataFrame
            fit: Whether to fit the scaler
            
        Returns:
            Scaled DataFrame
        """
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns)
    
    def preprocess(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Apply all preprocessing steps.
        
        Args:
            X: Input DataFrame
            fit: Whether to fit transformers
            
        Returns:
            Preprocessed DataFrame
        """
        X = self.handle_missing_values(X)
        X = self.scale_features(X, fit=fit)
        return X
