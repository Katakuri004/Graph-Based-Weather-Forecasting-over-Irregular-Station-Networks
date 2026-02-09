# Data Download Scripts

## NOAA ISD Data Downloader

Standalone script to download and process NOAA ISD data with robust error handling.

### Usage

```bash
# Download data for specific stations and years
python scripts/download_noaa_isd_data.py \
    --stations 725030-14732,722950-23174,725300-14734 \
    --years 2022,2023 \
    --output-dir data/raw/noaa_isd

# Use a stations file
python scripts/download_noaa_isd_data.py \
    --stations-file stations.txt \
    --years 2022,2023

# Resume interrupted download
python scripts/download_noaa_isd_data.py \
    --stations 725030-14732 \
    --years 2022,2023 \
    --resume

# Save to specific output file
python scripts/download_noaa_isd_data.py \
    --stations 725030-14732 \
    --years 2022 \
    --output-file data/processed/noaa_isd_2022.parquet
```

### Features

- **Robust downloads**: HTTPS with retries and exponential backoff
- **Progress tracking**: Saves progress to resume interrupted downloads
- **Multiple endpoints**: Tries alternative URLs if primary fails
- **Error handling**: Distinguishes network errors from missing files
- **Logging**: Detailed logs saved to `noaa_isd_download.log`
- **Format support**: Handles both full ISD and ISD-Lite formats

### Station IDs

Format: `USAF-WBAN` (e.g., `725030-14732` for New York)

Common stations:
- New York: `725030-14732`
- Los Angeles: `722950-23174`
- Chicago: `725300-14734`
- Houston: `722430-12960`
- Phoenix: `722780-23183`

### Output

- Raw files: Saved to `--output-dir` as `{USAF}-{WBAN}-{YEAR}.gz`
- Combined data: Saved as Parquet or CSV (specify with `--output-file`)
- Progress: Saved to `download_progress.json` in output directory
- Logs: Saved to `noaa_isd_download.log` in current directory
