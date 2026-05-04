import pandas as pd
from sklearn.preprocessing import StandardScaler

COLUMNS = ['unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3',
           'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
           'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
           'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
           'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21']

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the CMAPSS text data into a pandas DataFrame with predefined column headers.
    
    Args:
        filepath (str): Path to the text file dataset (e.g., train_FD001.txt)
        
    Returns:
        pd.DataFrame: Formatted dataframe containing the operational settings and sensor readings.
    """
    return pd.read_csv(filepath, sep=r'\s+', header=None, names=COLUMNS)

def create_rul(df: pd.DataFrame, clip_rul: int = None) -> pd.DataFrame:
    """
    Calculates Remaining Useful Life (RUL) for each engine.
    RUL = max_cycle - current_cycle for a specific unit.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'unit_id' and 'cycle' columns.
        clip_rul (int, optional): Maximum RUL value to clip at (piecewise linear).
        
    Returns:
        pd.DataFrame: DataFrame with the calculated 'RUL' column added.
    """
    # Group by unit_id to find the maximum cycle length for each engine
    max_cycles = df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    
    # Merge back and compute RUL
    df = df.merge(max_cycles, on='unit_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    
    if clip_rul is not None:
        df['RUL'] = df['RUL'].clip(upper=clip_rul)
        
    # Drop intermediate max_cycle column
    df = df.drop('max_cycle', axis=1)
    return df

def create_test_rul(df_test: pd.DataFrame, filepath_truth: str, clip_rul: int = None) -> pd.DataFrame:
    """
    Calculates RUL for the test set using the ground truth file.
    
    Args:
        df_test (pd.DataFrame): Testing dataframe.
        filepath_truth (str): Path to the true RUL text file (e.g., RUL_FD001.txt).
        clip_rul (int, optional): Maximum RUL value to clip at.
        
    Returns:
        pd.DataFrame: Testing dataframe with 'RUL' column added.
    """
    truth_df = pd.read_csv(filepath_truth, sep=r'\s+', header=None, names=['true_rul'])
    truth_df['unit_id'] = truth_df.index + 1
    
    max_cycles = df_test.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    
    df_test = df_test.merge(max_cycles, on='unit_id')
    df_test = df_test.merge(truth_df, on='unit_id')
    
    df_test['RUL'] = df_test['max_cycle'] - df_test['cycle'] + df_test['true_rul']
    
    if clip_rul is not None:
        df_test['RUL'] = df_test['RUL'].clip(upper=clip_rul)
        
    df_test = df_test.drop(['max_cycle', 'true_rul'], axis=1)
    return df_test

def drop_constant_columns(df_train: pd.DataFrame, df_test: pd.DataFrame = None):
    """
    Drops columns that have zero variance (constant values) in the training set.
    
    Args:
        df_train (pd.DataFrame): Training dataframe.
        df_test (pd.DataFrame, optional): Testing dataframe.
        
    Returns:
        pd.DataFrame or tuple: Dataframe(s) with constant columns removed.
    """
    # Find columns with only 1 unique value in training set
    constant_cols = [col for col in df_train.columns if df_train[col].nunique() <= 1]
    
    df_train_dropped = df_train.drop(columns=constant_cols)
    
    if df_test is not None:
        df_test_dropped = df_test.drop(columns=constant_cols, errors='ignore')
        return df_train_dropped, df_test_dropped
    
    return df_train_dropped


def clean_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    """
    Scales the feature columns using StandardScaler based on the training dataset.
    
    Args:
        df_train (pd.DataFrame): Training dataset with features.
        df_test (pd.DataFrame): Testing dataset with features.
        
    Returns:
        tuple: (df_train_scaled, df_test_scaled)
    """
    # Identify columns to scale (exclude identifiers and targets)
    exclude_cols = ['unit_id', 'cycle', 'RUL', 'health_index', 'health_status']
    scale_cols = [col for col in df_train.columns if col not in exclude_cols]
    
    scaler = StandardScaler()
    
    # Fit scaler on training data and transform both sets
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    
    df_train_scaled[scale_cols] = scaler.fit_transform(df_train_scaled[scale_cols])
    df_test_scaled[scale_cols] = scaler.transform(df_test_scaled[scale_cols])
    
    return df_train_scaled, df_test_scaled

