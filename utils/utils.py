Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>># Corresponding Section: 4.1 Context Information and Environmental Data Processing, Definition 6-7
import pandas as pd
import numpy as np
import pm4py

def sliding_window_preprocess(
    time_series: pd.Series,
    timestamps: pd.Series,
    window_minutes: int = 5,
    agg_func: str = "median"
) -> pd.Series:
    """
    Sliding window preprocessing for continuous environmental time series.
    Suppresses high-frequency noise and ensures causal consistency.

    Parameters
    ----------
    time_series : pd.Series
        Raw environmental observation values
    timestamps : pd.Series
        Corresponding datetime timestamps for each observation
    window_minutes : int
        Length of the sliding window in minutes
    agg_func : str
        Aggregation function: 'median'/'mean' for continuous data, 'last' for discrete data

    Returns
    -------
    pd.Series
        Smoothed observation values after sliding window preprocessing
    """
    # Sort data by timestamp to ensure temporal order
    df = pd.DataFrame({"value": time_series, "timestamp": timestamps}).sort_values("timestamp")
    df = df.set_index("timestamp")
    
    # Apply rolling window aggregation
    if agg_func == "median":
        smoothed = df.rolling(f"{window_minutes}min", closed="both").median()["value"]
    elif agg_func == "mean":
        smoothed = df.rolling(f"{window_minutes}min", closed="both").mean()["value"]
    elif agg_func == "last":
        smoothed = df.rolling(f"{window_minutes}min", closed="both").last()["value"]
    else:
        raise ValueError("Only 'median', 'mean', and 'last' aggregation functions are supported")
    
    # Linear interpolation for missing values
    smoothed = smoothed.interpolate(method="linear", limit_direction="both")
    return smoothed.reset_index(drop=True)

def semantic_discretization(
    values: pd.Series,
    normal_threshold: float,
    is_continuous: bool = True,
    discrete_mapping: dict = None
) -> pd.Series:
    """
    Maps continuous environmental observations to discrete semantic states (normal/abnormal).

    Parameters
    ----------
    values : pd.Series
        Preprocessed observation values
    normal_threshold : float
        Threshold for normal operation range
    is_continuous : bool
        Whether the variable is continuous (True) or discrete (False)
    discrete_mapping : dict
        Mapping rule for discrete variables (required if is_continuous=False)

    Returns
    -------
    pd.Series
        Discretized context state series ('normal'/'abnormal')
    """
    if is_continuous:
        # Continuous variable: values above threshold are classified as abnormal
        return values.apply(lambda x: "abnormal" if x > normal_threshold else "normal")
    else:
        # Discrete variable: map to semantic states via predefined rules
        if discrete_mapping is None:
            raise ValueError("discrete_mapping must be provided for discrete variables")
        return values.map(discrete_mapping)

def standardize_event_log(log_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes raw event logs into pm4py-compatible format with unified field names.
    Follows the XES event log standard for process mining.

    Parameters
    ----------
    log_df : pd.DataFrame
        Raw event log DataFrame

    Returns
    -------
    pd.DataFrame
        Standardized event log with consistent field naming and sorting
    """
    # Map standard XES fields to unified names
    standard_mapping = {
        "case:concept:name": "case_id",
        "concept:name": "activity",
        "time:timestamp": "timestamp",
        "org:resource": "resource"
    }
    log_df = log_df.rename(columns=standard_mapping)
    
    # Convert timestamp to UTC datetime format
    log_df["timestamp"] = pd.to_datetime(log_df["timestamp"], utc=True)
    # Sort log by case ID and timestamp
    log_df = log_df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    return log_df
