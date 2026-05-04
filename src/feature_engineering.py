import pandas as pd
import numpy as np

def add_rolling_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Adds rolling mean and rolling standard deviation for sensor columns.
    
    Args:
        df (pd.DataFrame): Dataframe containing sensor columns and 'unit_id'.
        window (int): Size of the rolling window.
        
    Returns:
        pd.DataFrame: Dataframe with added rolling features.
    """
    df = df.copy()
    sensor_cols = [col for col in df.columns if 'sensor' in col]
    
    for col in sensor_cols:
        df[f'{col}_roll_mean'] = df.groupby('unit_id')[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'{col}_roll_std'] = df.groupby('unit_id')[col].transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        
    return df

def add_trend_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Adds trend (slope) features for sensor columns using a rolling window.
    Trend is calculated as the difference between the current value and the value
    at the start of the window, normalized by the window size.
    
    Args:
        df (pd.DataFrame): Dataframe containing sensor columns and 'unit_id'.
        window (int): Size of the rolling window.
        
    Returns:
        pd.DataFrame: Dataframe with added trend features.
    """
    df = df.copy()
    sensor_cols = [col for col in df.columns if 'sensor' in col]
    
    def calculate_trend(series):
        return (series - series.shift(window - 1)) / window

    for col in sensor_cols:
        df[f'{col}_trend'] = df.groupby('unit_id')[col].transform(calculate_trend).fillna(0)
        
    return df

def add_custom_health_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a custom Health Index (1.0 to 0.0) based on the Remaining Useful Life (RUL).
    Health Index = RUL / (RUL + cycle), representing the percentage of life remaining.
    
    Args:
        df (pd.DataFrame): Dataframe containing 'RUL' and 'cycle'.
        
    Returns:
        pd.DataFrame: Dataframe with 'health_index' and 'health_status' added.
    """
    df = df.copy()
    
    if 'RUL' in df.columns and 'cycle' in df.columns:
        # Quantitative Health Index: 1.0 (new) -> 0.0 (failed)
        # Using RUL/(RUL+cycle) is more robust for test data where we know ground truth RUL
        df['health_index'] = df['RUL'] / (df['RUL'] + df['cycle'])
    else:
        # Fallback to simple cycle-based progress if RUL isn't present
        max_cycles = df.groupby('unit_id')['cycle'].transform('max')
        df['health_index'] = 1 - (df['cycle'] / max_cycles)
        
    # Categorical Health Status (Traffic Light logic)
    def classify_status(rul):
        if rul > 100:
            return 'Healthy'
        elif 30 < rul <= 100:
            return 'Warning'
        else:
            return 'Critical'
            
    if 'RUL' in df.columns:
        df['health_status'] = df['RUL'].apply(classify_status)
        
    return df

def apply_feature_engineering(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Executes the full feature engineering pipeline.
    """
    df = add_rolling_features(df, window)
    df = add_trend_features(df, window)
    df = add_custom_health_index(df)
    return df
