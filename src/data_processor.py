"""Data processing module for feature engineering and transformation"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create and engineer features for sales forecasting"""
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, 
                           target_col: str, 
                           lags: List[int] = [1, 7, 14, 30]) -> pd.DataFrame:
        """
        Create lagged features for time series
        
        Args:
            df: Input DataFrame (must be sorted by date)
            target_col: Column name for which to create lags
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        result = df.copy()
        
        for lag in lags:
            result[f'{target_col}_lag_{lag}'] = result[target_col].shift(lag)
        
        logger.info(f"Created {len(lags)} lag features for {target_col}")
        return result
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame,
                               target_col: str,
                               windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """
        Create rolling window features (mean, std, min, max)
        
        Args:
            df: Input DataFrame (must be sorted by date)
            target_col: Column name for which to create rolling features
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        result = df.copy()
        
        for window in windows:
            result[f'{target_col}_rolling_mean_{window}'] = result[target_col].rolling(
                window=window, min_periods=1
            ).mean()
            result[f'{target_col}_rolling_std_{window}'] = result[target_col].rolling(
                window=window, min_periods=1
            ).std()
            result[f'{target_col}_rolling_min_{window}'] = result[target_col].rolling(
                window=window, min_periods=1
            ).min()
            result[f'{target_col}_rolling_max_{window}'] = result[target_col].rolling(
                window=window, min_periods=1
            ).max()
        
        logger.info(f"Created rolling window features for {target_col}")
        return result
    
    @staticmethod
    def create_seasonal_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Create seasonal features from date column
        
        Args:
            df: Input DataFrame
            date_col: Name of date column
            
        Returns:
            DataFrame with seasonal features
        """
        result = df.copy()
        result[date_col] = pd.to_datetime(result[date_col])
        
        # Seasonal decomposition features
        result['day_of_year'] = result[date_col].dt.dayofyear
        result['week_of_year'] = result[date_col].dt.isocalendar().week
        result['month'] = result[date_col].dt.month
        result['quarter'] = result[date_col].dt.quarter
        result['day_of_week'] = result[date_col].dt.dayofweek
        
        # Cyclical encoding for seasonality
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        result['day_of_week_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['day_of_week_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        
        logger.info("Created seasonal features")
        return result
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame,
                                   feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features from pairs of features
        
        Args:
            df: Input DataFrame
            feature_pairs: List of (col1, col2) tuples for interaction
            
        Returns:
            DataFrame with interaction features
        """
        result = df.copy()
        
        for col1, col2 in feature_pairs:
            if col1 in result.columns and col2 in result.columns:
                result[f'{col1}_x_{col2}'] = result[col1] * result[col2]
        
        logger.info(f"Created {len(feature_pairs)} interaction features")
        return result
    
    @staticmethod
    def create_trend_features(df: pd.DataFrame,
                             target_col: str,
                             window: int = 30) -> pd.DataFrame:
        """
        Create trend features using linear regression slopes
        
        Args:
            df: Input DataFrame (sorted by date)
            target_col: Column for trend calculation
            window: Window size for trend calculation
            
        Returns:
            DataFrame with trend features
        """
        result = df.copy()
        
        def calculate_trend(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            z = np.polyfit(x, series, 1)
            return z[0]  # slope
        
        result[f'{target_col}_trend'] = result[target_col].rolling(
            window=window, min_periods=2
        ).apply(calculate_trend, raw=False)
        
        logger.info(f"Created trend features for {target_col}")
        return result


class DataScaler:
    """Scale and normalize data"""
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize scaler
        
        Args:
            scaler_type: 'standard' for StandardScaler or 'minmax' for MinMaxScaler
        """
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        self.scaler_type = scaler_type
        self.fitted = False
    
    def fit_transform(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit scaler and transform data
        
        Args:
            df: Input DataFrame
            columns: Columns to scale (if None, scales all numeric columns)
            
        Returns:
            Scaled DataFrame
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        result = df.copy()
        result[columns] = self.scaler.fit_transform(df[columns])
        self.fitted = True
        
        logger.info(f"Fitted and transformed {len(columns)} columns with {self.scaler_type} scaler")
        return result
    
    def transform(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Transform data using fitted scaler
        
        Args:
            df: Input DataFrame
            columns: Columns to scale
            
        Returns:
            Scaled DataFrame
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        result = df.copy()
        result[columns] = self.scaler.transform(df[columns])
        
        return result
    
    def inverse_transform(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Inverse transform scaled data
        
        Args:
            df: Scaled DataFrame
            columns: Columns to inverse transform
            
        Returns:
            Original scale DataFrame
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted.")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        result = df.copy()
        result[columns] = self.scaler.inverse_transform(df[columns])
        
        return result


class TimeSeriesSplitter:
    """Split time series data for training and testing"""
    
    @staticmethod
    def train_test_split(df: pd.DataFrame,
                        date_col: str,
                        test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split time series data chronologically
        
        Args:
            df: Input DataFrame (sorted by date)
            date_col: Name of date column
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Split data: train={train_df.shape[0]}, test={test_df.shape[0]}")
        return train_df, test_df
    
    @staticmethod
    def create_rolling_windows(df: pd.DataFrame,
                              lookback: int = 30,
                              lookahead: int = 7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create rolling windows for supervised learning
        
        Args:
            df: Input DataFrame (1D array expected after processing)
            lookback: Number of previous time steps to use as input
            lookahead: Number of future time steps to predict
            
        Returns:
            Tuple of (X, y) arrays
        """
        data = df.values if isinstance(df, pd.Series) else df
        
        X, y = [], []
        
        for i in range(len(data) - lookback - lookahead + 1):
            X.append(data[i:(i + lookback)])
            y.append(data[(i + lookback):(i + lookback + lookahead)])
        
        logger.info(f"Created {len(X)} rolling windows (lookback={lookback}, lookahead={lookahead})")
        return np.array(X), np.array(y)


class OutlierDetector:
    """Detect and handle outliers"""
    
    @staticmethod
    def identify_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
        """
        Identify outliers using Interquartile Range method
        
        Args:
            series: Input Series
            multiplier: IQR multiplier (1.5 is standard)
            
        Returns:
            Boolean Series marking outliers
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    @staticmethod
    def identify_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """
        Identify outliers using Z-score method
        
        Args:
            series: Input Series
            threshold: Z-score threshold (3.0 is standard)
            
        Returns:
            Boolean Series marking outliers
        """
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    @staticmethod
    def handle_outliers(df: pd.DataFrame,
                       column: str,
                       method: str = 'iqr',
                       action: str = 'cap') -> pd.DataFrame:
        """
        Handle outliers using specified method
        
        Args:
            df: Input DataFrame
            column: Column to process
            method: 'iqr' or 'zscore'
            action: 'remove', 'cap', or 'ffill'
            
        Returns:
            DataFrame with handled outliers
        """
        result = df.copy()
        
        if method == 'iqr':
            outliers = OutlierDetector.identify_outliers_iqr(result[column])
            Q1 = result[column].quantile(0.25)
            Q3 = result[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        else:
            outliers = OutlierDetector.identify_outliers_zscore(result[column])
            lower_bound = result[column].mean() - 3 * result[column].std()
            upper_bound = result[column].mean() + 3 * result[column].std()
        
        if action == 'remove':
            result = result[~outliers]
        elif action == 'cap':
            result.loc[result[column] < lower_bound, column] = lower_bound
            result.loc[result[column] > upper_bound, column] = upper_bound
        elif action == 'ffill':
            result.loc[outliers, column] = np.nan
            result[column] = result[column].fillna(method='ffill')
        
        logger.info(f"Handled {outliers.sum()} outliers in {column} using {method}/{action}")
        return result
