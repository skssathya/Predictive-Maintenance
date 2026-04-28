"""Feature engineering utilities for predictive maintenance"""

import pandas as pd
import numpy as np
from scipy import stats

class FeatureEngineer:
    """
    Handles feature engineering and creation of new features from sensor data.
    """
    
    @staticmethod
    def create_statistical_features(data: pd.DataFrame, columns: list, window: int = 5) -> pd.DataFrame:
        """
        Create statistical features from time series data.
        
        Args:
            data: Input DataFrame
            columns: Columns to create features from
            window: Rolling window size
            
        Returns:
            DataFrame with new statistical features
        """
        df = data.copy()
        
        for col in columns:
            # Rolling statistics
            df[f'{col}_mean'] = df[col].rolling(window=window).mean()
            df[f'{col}_std'] = df[col].rolling(window=window).std()
            df[f'{col}_max'] = df[col].rolling(window=window).max()
            df[f'{col}_min'] = df[col].rolling(window=window).min()
            
            # Rate of change
            df[f'{col}_diff'] = df[col].diff()
            
        return df
    
    @staticmethod
    def create_domain_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific features for machinery monitoring.
        
        Args:
            data: Input DataFrame with sensor readings
            
        Returns:
            DataFrame with new domain features
        """
        df = data.copy()
        
        # Vibration to temperature ratio
        if 'vibration' in df.columns and 'temperature' in df.columns:
            df['vibration_temp_ratio'] = df['vibration'] / (df['temperature'] + 1e-6)
        
        # Temperature anomaly
        if 'temperature' in df.columns:
            df['temp_anomaly'] = np.abs(df['temperature'] - df['temperature'].mean()) / df['temperature'].std()
        
        # Operating stress index
        if 'motor_current' in df.columns and 'run_time' in df.columns:
            df['stress_index'] = df['motor_current'] * df['run_time']
        
        return df
    
    @staticmethod
    def select_features(X: pd.DataFrame, y: pd.Series, method: str = 'correlation', top_k: int = 10) -> list:
        """
        Select most important features.
        
        Args:
            X: Features DataFrame
            y: Target Series
            method: 'correlation' or 'mutual_info'
            top_k: Number of top features to return
            
        Returns:
            List of selected feature names
        """
        if method == 'correlation':
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            return list(correlations.head(top_k).index)
        
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_classif
            mi_scores = mutual_info_classif(X, y, random_state=42)
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': mi_scores
            }).sort_values('importance', ascending=False)
            return list(feature_importance.head(top_k)['feature'])
        
        return list(X.columns[:top_k])
