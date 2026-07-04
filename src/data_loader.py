"""Data loading module for Walmart and Tesco datasets"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class WalmartDataLoader:
    """Load and process Walmart Sales Forecasting dataset"""
    
    def __init__(self, data_dir: str = "data/walmart"):
        """
        Initialize Walmart data loader
        
        Args:
            data_dir: Directory containing Walmart CSV files
        """
        self.data_dir = Path(data_dir)
        self.train = None
        self.test = None
        self.features = None
        self.stores = None
    
    def load_all_files(self) -> Dict[str, pd.DataFrame]:
        """
        Load all Walmart dataset files
        
        Returns:
            Dictionary containing train, test, features, and stores DataFrames
        """
        try:
            self.train = pd.read_csv(self.data_dir / "train.csv")
            self.test = pd.read_csv(self.data_dir / "test.csv")
            self.features = pd.read_csv(self.data_dir / "features.csv")
            self.stores = pd.read_csv(self.data_dir / "stores.csv")
            
            logger.info(f"Loaded Walmart data - Train: {self.train.shape}, Test: {self.test.shape}")
            
            return {
                'train': self.train,
                'test': self.test,
                'features': self.features,
                'stores': self.stores
            }
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
    
    def preprocess_train_data(self) -> pd.DataFrame:
        """
        Preprocess training data with feature engineering
        
        Returns:
            Processed training DataFrame
        """
        if self.train is None:
            self.load_all_files()
        
        df = self.train.copy()
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Merge with features
        if self.features is not None:
            self.features['date'] = pd.to_datetime(self.features['date'])
            df = df.merge(self.features, on=['store', 'date'], how='left', suffixes=('', '_features'))
        
        # Merge with store metadata
        if self.stores is not None:
            df = df.merge(self.stores, on='store', how='left')
        
        # Handle missing values in markdown columns
        markdown_cols = ['markdown1', 'markdown2', 'markdown3', 'markdown4', 'markdown5']
        for col in markdown_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill other missing values with forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Create temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['week'] = df['date'].dt.isocalendar().week
        df['day_of_week'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        
        logger.info(f"Preprocessed data shape: {df.shape}")
        
        return df
    
    def get_store_department_data(self, store_id: int, dept_id: int) -> pd.DataFrame:
        """
        Get data for specific store and department
        
        Args:
            store_id: Store identifier
            dept_id: Department identifier
            
        Returns:
            Filtered DataFrame
        """
        if self.train is None:
            self.load_all_files()
        
        return self.train[(self.train['store'] == store_id) & 
                         (self.train['dept'] == dept_id)].copy()


class TescoDataLoader:
    """Load and process Tesco retail data"""
    
    def __init__(self, data_dir: str = "data/tesco"):
        """
        Initialize Tesco data loader
        
        Args:
            data_dir: Directory containing Tesco data files
        """
        self.data_dir = Path(data_dir)
        self.sales_data = None
        self.inventory_data = None
        self.product_data = None
    
    def load_sales_data(self, filename: str = "sales.csv") -> pd.DataFrame:
        """
        Load Tesco sales data
        
        Args:
            filename: Name of sales CSV file
            
        Returns:
            Sales DataFrame
        """
        try:
            self.sales_data = pd.read_csv(self.data_dir / filename)
            self.sales_data['date'] = pd.to_datetime(self.sales_data['date'])
            logger.info(f"Loaded Tesco sales data: {self.sales_data.shape}")
            return self.sales_data
        except FileNotFoundError as e:
            logger.error(f"Tesco sales file not found: {e}")
            raise
    
    def load_inventory_data(self, filename: str = "inventory.csv") -> pd.DataFrame:
        """
        Load Tesco inventory data
        
        Args:
            filename: Name of inventory CSV file
            
        Returns:
            Inventory DataFrame
        """
        try:
            self.inventory_data = pd.read_csv(self.data_dir / filename)
            self.inventory_data['date'] = pd.to_datetime(self.inventory_data['date'])
            logger.info(f"Loaded Tesco inventory data: {self.inventory_data.shape}")
            return self.inventory_data
        except FileNotFoundError as e:
            logger.error(f"Tesco inventory file not found: {e}")
            raise
    
    def load_product_data(self, filename: str = "products.csv") -> pd.DataFrame:
        """
        Load Tesco product master data
        
        Args:
            filename: Name of product CSV file
            
        Returns:
            Product DataFrame
        """
        try:
            self.product_data = pd.read_csv(self.data_dir / filename)
            logger.info(f"Loaded Tesco product data: {self.product_data.shape}")
            return self.product_data
        except FileNotFoundError as e:
            logger.error(f"Tesco product file not found: {e}")
            raise
    
    def preprocess_sales_data(self) -> pd.DataFrame:
        """
        Preprocess Tesco sales data
        
        Returns:
            Processed sales DataFrame
        """
        if self.sales_data is None:
            self.load_sales_data()
        
        df = self.sales_data.copy()
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Merge with product data if available
        if self.product_data is not None:
            df = df.merge(self.product_data, on='product_id', how='left')
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Create temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['week'] = df['date'].dt.isocalendar().week
        df['day_of_week'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        logger.info(f"Preprocessed Tesco sales data: {df.shape}")
        
        return df
    
    def merge_sales_inventory(self) -> pd.DataFrame:
        """
        Merge sales and inventory data
        
        Returns:
            Merged DataFrame
        """
        if self.sales_data is None:
            self.load_sales_data()
        if self.inventory_data is None:
            self.load_inventory_data()
        
        merged = self.sales_data.merge(
            self.inventory_data,
            on=['store_id', 'product_id', 'date'],
            how='inner'
        )
        
        logger.info(f"Merged sales and inventory: {merged.shape}")
        return merged


class DataValidator:
    """Validate data quality and consistency"""
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
        """
        Check missing value percentages
        
        Args:
            df: Input DataFrame
            threshold: Alert if missing % exceeds this value
            
        Returns:
            Dictionary of column names and missing percentages
        """
        missing_pct = (df.isnull().sum() / len(df)) * 100
        missing_cols = missing_pct[missing_pct > threshold]
        
        if len(missing_cols) > 0:
            logger.warning(f"Columns with >{threshold}% missing: {missing_cols.to_dict()}")
        
        return missing_pct[missing_pct > 0].to_dict()
    
    @staticmethod
    def check_duplicates(df: pd.DataFrame, subset: Optional[list] = None) -> int:
        """
        Check for duplicate rows
        
        Args:
            df: Input DataFrame
            subset: Columns to check for duplicates
            
        Returns:
            Number of duplicate rows
        """
        duplicates = df.duplicated(subset=subset).sum()
        
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
        
        return duplicates
    
    @staticmethod
    def check_data_types(df: pd.DataFrame) -> Dict[str, str]:
        """
        Check and return data types
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of column names and data types
        """
        return df.dtypes.to_dict()
    
    @staticmethod
    def check_numeric_ranges(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Check numeric column ranges
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary of column statistics
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        ranges = {}
        for col in numeric_df.columns:
            ranges[col] = {
                'min': numeric_df[col].min(),
                'max': numeric_df[col].max(),
                'mean': numeric_df[col].mean(),
                'std': numeric_df[col].std()
            }
        
        return ranges
