@echo off
REM Example batch script to download NOAA ISD data
REM Run this from the project root directory

REM Activate virtual environment (adjust path if needed)
call venv-earth-sgnn\Scripts\activate.bat

REM Download data for example stations
python scripts\download_noaa_isd_data.py ^
    --stations 725030-14732,722950-23174,725300-14734 ^
    --years 2022,2023 ^
    --output-dir data\raw\noaa_isd

pause
