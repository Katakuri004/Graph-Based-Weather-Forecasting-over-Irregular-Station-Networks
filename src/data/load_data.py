"""
Data loading utilities for weather forecasting project.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def load_openmeteo_data(data_file: Optional[str] = None) -> pd.DataFrame:
    """
    Load Open-Meteo data from file.
    
    Parameters:
    -----------
    data_file : str, optional
        Path to data file. If None, uses default location.
    
    Returns:
    --------
    pd.DataFrame
        Weather data with columns: timestamp, station_name, latitude, longitude,
        temperature_2m, relative_humidity_2m, wind_speed_10m, etc.
    """
    if data_file is None:
        data_file = RAW_DATA_DIR / "openmeteo_raw_data.parquet"
    else:
        data_file = Path(data_file)
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    # Try parquet first, then CSV
    if data_file.suffix == '.parquet':
        df = pd.read_parquet(data_file)
    else:
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def load_station_metadata(metadata_file: Optional[str] = None) -> pd.DataFrame:
    """
    Load station metadata.
    
    Parameters:
    -----------
    metadata_file : str, optional
        Path to metadata file. If None, uses default location.
    
    Returns:
    --------
    pd.DataFrame
        Station metadata with columns: station_name, latitude, longitude, elevation
    """
    if metadata_file is None:
        metadata_file = RAW_DATA_DIR / "openmeteo_station_metadata.csv"
    else:
        metadata_file = Path(metadata_file)
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    return pd.read_csv(metadata_file)


def get_station_data(df: pd.DataFrame, station_name: str) -> pd.DataFrame:
    """
    Get data for a specific station.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full weather data
    station_name : str
        Name of the station
    
    Returns:
    --------
    pd.DataFrame
        Data for the specified station
    """
    return df[df['station_name'] == station_name].copy()


def get_time_range_data(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Get data for a specific time range.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full weather data
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    
    Returns:
    --------
    pd.DataFrame
        Data for the specified time range
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
    return df[mask].copy()
