#!/usr/bin/env python3
"""
Standalone script to download and process NOAA ISD data.

This script can be run independently of Jupyter notebooks and handles:
- Robust HTTPS downloads with retries
- Progress saving/resume capability
- Better error handling and logging
- Alternative data sources if primary fails

Usage:
    python scripts/download_noaa_isd_data.py --stations 725030-14732,722950-23174 --years 2022,2023
    python scripts/download_noaa_isd_data.py --stations-file stations.txt --years 2022,2023
"""

import os
import sys
import argparse
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import requests
import pandas as pd
import numpy as np
import gzip
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('noaa_isd_download.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "https://www.ncei.noaa.gov/data/global-hourly/access"
MAX_RETRIES = 5
RETRY_DELAYS = [2, 4, 8, 16, 32]  # Exponential backoff in seconds
REQUEST_TIMEOUT = (30, 600)  # (connect, read) timeout in seconds
CHUNK_SIZE = 8192

# Alternative endpoints to try if primary fails
ALTERNATIVE_ENDPOINTS = [
    "https://www1.ncei.noaa.gov/pub/data/noaa/{year}/{filename}",
    "https://ftp.ncei.noaa.gov/pub/data/noaa/{year}/{filename}",
]


def parse_isd_line(line: str) -> Optional[Dict]:
    """Parse a single ISD line - handles both full and lite formats."""
    if len(line) < 80:
        return None
    
    try:
        # Extract basic fields
        usaf = line[0:6].strip()
        wban = line[7:12].strip() if len(line) > 11 else ''
        year_str = line[13:17].strip() if len(line) > 16 else ''
        month_str = line[17:19].strip() if len(line) > 18 else ''
        day_str = line[19:21].strip() if len(line) > 20 else ''
        hour_str = line[21:23].strip() if len(line) > 22 else ''
        
        if not all([year_str.isdigit(), month_str.isdigit(), day_str.isdigit(), hour_str.isdigit()]):
            return None
        
        year, month, day, hour = int(year_str), int(month_str), int(day_str), int(hour_str)
        
        if not (1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31 and 0 <= hour <= 23):
            return None
        
        try:
            timestamp = datetime(year, month, day, hour)
        except ValueError:
            return None
        
        # Parse meteorological variables
        is_full_format = len(line) > 200
        temperature_2m = np.nan
        dewpoint_2m = np.nan
        surface_pressure = np.nan
        wind_direction_10m = np.nan
        wind_speed_10m = np.nan
        
        if is_full_format:
            # Full ISD format - use more precise parsing
            # Temperature: Look in ADD section around position 87-92
            # Format: +NNNN (tenths of degrees C)
            temp_patterns = [
                (87, 92),  # Standard position
                (80, 100),  # Search wider area
            ]
            for start, end in temp_patterns:
                if len(line) > end:
                    temp_str = line[start:end].strip()
                    # Look for +NNNN or -NNNN pattern
                    match = re.search(r'([+-]\d{4,5})', temp_str)
                    if match:
                        try:
                            temp_val = float(match.group(1)) / 10.0
                            if -100 <= temp_val <= 100:
                                temperature_2m = temp_val
                                break
                        except:
                            pass
            
            # Dew point: similar approach
            for start, end in [(93, 98), (90, 100)]:
                if len(line) > end:
                    dew_str = line[start:end].strip()
                    match = re.search(r'([+-]\d{3,4})', dew_str)
                    if match:
                        try:
                            dew_val = float(match.group(1)) / 10.0
                            if -100 <= dew_val <= 100:
                                dewpoint_2m = dew_val
                                break
                        except:
                            pass
            
            # Pressure: position 99-104
            for start, end in [(99, 104), (95, 110)]:
                if len(line) > end:
                    press_str = line[start:end].strip()
                    match = re.search(r'([+-]\d{4,5})', press_str)
                    if match:
                        try:
                            press_val = float(match.group(1)) / 10.0
                            if 500 <= press_val <= 1100:
                                surface_pressure = press_val
                                break
                        except:
                            pass
            
            # Wind direction: position 60-63
            if len(line) > 63:
                wind_dir_str = line[60:63].strip()
                if wind_dir_str.isdigit() and wind_dir_str != '999':
                    try:
                        wind_dir_val = float(wind_dir_str)
                        if 0 <= wind_dir_val <= 360:
                            wind_direction_10m = wind_dir_val
                    except:
                        pass
            
            # Wind speed: position 65-69
            if len(line) > 69:
                wind_speed_str = line[65:69].strip()
                if wind_speed_str.isdigit() and wind_speed_str != '9999':
                    try:
                        wind_speed_val = float(wind_speed_str) / 10.0
                        if 0 <= wind_speed_val <= 200:
                            wind_speed_10m = wind_speed_val
                    except:
                        pass
        else:
            # ISD-Lite format - fixed positions
            if len(line) > 28:
                temp_str = line[24:29].strip()
                if temp_str and temp_str not in ['+9999', '99999', '+999']:
                    try:
                        temperature_2m = float(temp_str) / 10.0
                        if not (-100 <= temperature_2m <= 100):
                            temperature_2m = np.nan
                    except:
                        temperature_2m = np.nan
            
            if len(line) > 33:
                dewpoint_str = line[30:34].strip()
                if dewpoint_str and dewpoint_str not in ['+999', '9999']:
                    try:
                        dewpoint_2m = float(dewpoint_str) / 10.0
                        if not (-100 <= dewpoint_2m <= 100):
                            dewpoint_2m = np.nan
                    except:
                        dewpoint_2m = np.nan
            
            if len(line) > 38:
                pressure_str = line[35:39].strip()
                if pressure_str and pressure_str not in ['+999', '9999']:
                    try:
                        surface_pressure = float(pressure_str) / 10.0
                        if not (500 <= surface_pressure <= 1100):
                            surface_pressure = np.nan
                    except:
                        surface_pressure = np.nan
            
            if len(line) > 43:
                wind_dir_str = line[40:44].strip()
                if wind_dir_str and wind_dir_str not in ['999', '9999']:
                    try:
                        wind_direction_10m = float(wind_dir_str)
                        if not (0 <= wind_direction_10m <= 360):
                            wind_direction_10m = np.nan
                    except:
                        wind_direction_10m = np.nan
            
            if len(line) > 50:
                wind_speed_str = line[46:51].strip()
                if wind_speed_str and wind_speed_str not in ['9999', '99999']:
                    try:
                        wind_speed_10m = float(wind_speed_str) / 10.0
                        if not (0 <= wind_speed_10m <= 200):
                            wind_speed_10m = np.nan
                    except:
                        wind_speed_10m = np.nan
        
        # Calculate relative humidity
        if not np.isnan(temperature_2m) and not np.isnan(dewpoint_2m):
            try:
                e_sat = 6.112 * np.exp(17.67 * temperature_2m / (temperature_2m + 243.5))
                e_act = 6.112 * np.exp(17.67 * dewpoint_2m / (dewpoint_2m + 243.5))
                relative_humidity_2m = 100.0 * (e_act / e_sat)
                if not (0 <= relative_humidity_2m <= 100):
                    relative_humidity_2m = np.nan
            except:
                relative_humidity_2m = np.nan
        else:
            relative_humidity_2m = np.nan
        
        # Only return if we have at least one valid measurement
        if all(np.isnan([temperature_2m, dewpoint_2m, surface_pressure, wind_speed_10m, wind_direction_10m])):
            return None
        
        return {
            'station_id': f"{usaf}-{wban}",
            'timestamp': timestamp,
            'temperature_2m': temperature_2m,
            'dewpoint_2m': dewpoint_2m,
            'relative_humidity_2m': relative_humidity_2m,
            'wind_speed_10m': wind_speed_10m,
            'wind_direction_10m': wind_direction_10m,
            'surface_pressure': surface_pressure
        }
    except Exception as e:
        return None


def download_file(url: str, output_path: Path, session: Optional[requests.Session] = None) -> Tuple[bool, Optional[str]]:
    """Download a file with retries and error handling."""
    if session is None:
        session = requests.Session()
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Downloading {url} (attempt {attempt + 1}/{MAX_RETRIES})")
            
            response = session.get(
                url,
                stream=True,
                timeout=REQUEST_TIMEOUT,
                headers={'User-Agent': 'Mozilla/5.0 (compatible; NOAA-ISD-Downloader/1.0)'}
            )
            response.raise_for_status()
            
            # Download to temp file first
            temp_file = output_path.with_suffix('.tmp')
            total_size = 0
            expected_size = int(response.headers.get('Content-Length', 0))
            
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)
            
            # Verify file size
            if expected_size > 0 and abs(total_size - expected_size) > 1024:  # Allow 1KB difference
                logger.warning(f"Size mismatch: expected {expected_size}, got {total_size}")
            
            if total_size < 1024:
                temp_file.unlink()
                return False, f"File too small: {total_size} bytes"
            
            # Rename temp file
            if output_path.exists():
                output_path.unlink()
            temp_file.rename(output_path)
            
            logger.info(f"Successfully downloaded {output_path.name} ({total_size:,} bytes)")
            return True, None
            
        except requests.exceptions.Timeout as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAYS[attempt]
                logger.warning(f"Timeout, retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                return False, f"Timeout after {MAX_RETRIES} attempts: {e}"
                
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                if e.response.status_code == 404:
                    return False, "File not found (404)"
            
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAYS[attempt]
                logger.warning(f"Network error: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            else:
                return False, f"Network error after {MAX_RETRIES} attempts: {e}"
                
        except Exception as e:
            return False, f"Unexpected error: {e}"
    
    return False, "Max retries exceeded"


def download_station_year(usaf: str, wban: str, year: int, output_dir: Path) -> Optional[pd.DataFrame]:
    """Download and parse data for a station-year."""
    station_id = f"{usaf}-{wban}"
    filename = f"{usaf}-{wban}-{year}.gz"
    local_file = output_dir / filename
    
    # Try primary endpoint first
    url = f"{BASE_URL}/{year}/{filename}"
    success, error = download_file(url, local_file)
    
    # Try alternative endpoints if primary fails
    if not success and "not found" in error.lower() or "404" in error:
        for alt_template in ALTERNATIVE_ENDPOINTS:
            alt_url = alt_template.format(year=year, filename=filename)
            logger.info(f"Trying alternative endpoint: {alt_url}")
            success, error = download_file(alt_url, local_file)
            if success:
                break
    
    if not success:
        logger.error(f"Failed to download {filename}: {error}")
        return None
    
    # Parse file
    observations = []
    try:
        with gzip.open(local_file, 'rt', encoding='latin-1', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                obs = parse_isd_line(line)
                if obs:
                    observations.append(obs)
                # Progress indicator every 10000 lines
                if line_num % 10000 == 0:
                    logger.debug(f"Parsed {line_num} lines, {len(observations)} valid observations")
    except Exception as e:
        logger.error(f"Error parsing {filename}: {e}")
        return None
    
    if len(observations) == 0:
        logger.warning(f"No valid observations found in {filename}")
        return None
    
    df = pd.DataFrame(observations)
    df['usaf'] = usaf
    df['wban'] = wban
    logger.info(f"Parsed {len(df)} observations from {filename}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Download NOAA ISD data')
    parser.add_argument('--stations', type=str, help='Comma-separated station IDs (USAF-WBAN)')
    parser.add_argument('--stations-file', type=Path, help='File with station IDs (one per line)')
    parser.add_argument('--years', type=str, required=True, help='Comma-separated years (e.g., 2022,2023)')
    parser.add_argument('--output-dir', type=Path, default=Path('data/raw/noaa_isd'), help='Output directory')
    parser.add_argument('--output-file', type=Path, help='Output CSV/Parquet file')
    parser.add_argument('--resume', action='store_true', help='Resume from progress file')
    
    args = parser.parse_args()
    
    # Parse stations
    stations = []
    if args.stations:
        stations = [s.strip() for s in args.stations.split(',')]
    elif args.stations_file:
        with open(args.stations_file, 'r') as f:
            stations = [line.strip() for line in f if line.strip()]
    else:
        logger.error("Must provide --stations or --stations-file")
        return 1
    
    # Parse years
    years = [int(y.strip()) for y in args.years.split(',')]
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Progress tracking
    progress_file = args.output_dir / 'download_progress.json'
    if args.resume and progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        logger.info(f"Resuming from progress file: {progress}")
    else:
        progress = {'completed': [], 'failed': []}
    
    # Download data
    all_data = []
    session = requests.Session()
    
    for station_id in stations:
        parts = station_id.split('-')
        if len(parts) != 2:
            logger.error(f"Invalid station ID format: {station_id} (expected USAF-WBAN)")
            continue
        
        usaf, wban = parts[0].zfill(6), parts[1].zfill(5)
        station_key = f"{usaf}-{wban}"
        
        logger.info(f"Processing station: {station_key}")
        
        for year in years:
            task_key = f"{station_key}-{year}"
            if task_key in progress['completed']:
                logger.info(f"Skipping {task_key} (already completed)")
                continue
            
            try:
                df = download_station_year(usaf, wban, year, args.output_dir)
                if df is not None and len(df) > 0:
                    all_data.append(df)
                    progress['completed'].append(task_key)
                    logger.info(f"✓ {task_key}: {len(df)} observations")
                else:
                    progress['failed'].append(task_key)
                    logger.warning(f"✗ {task_key}: No data")
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                # Save progress
                with open(progress_file, 'w') as f:
                    json.dump(progress, f, indent=2)
                return 1
            except Exception as e:
                progress['failed'].append(task_key)
                logger.error(f"✗ {task_key}: {e}")
            
            # Save progress periodically
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
            
            time.sleep(1)  # Be polite to server
    
    # Combine and save results
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"\nTotal observations: {len(combined_df)}")
        logger.info(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
        logger.info(f"Stations: {combined_df['station_id'].nunique()}")
        
        if args.output_file:
            output_path = args.output_file
        else:
            output_path = args.output_dir / 'noaa_isd_combined.parquet'
        
        if output_path.suffix == '.parquet':
            combined_df.to_parquet(output_path, index=False)
        else:
            combined_df.to_csv(output_path, index=False)
        
        logger.info(f"Saved to: {output_path}")
    else:
        logger.warning("No data downloaded")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
