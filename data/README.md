# Data Related Instruction

## Data Source
- **Get raw data from [kaggle](https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge/data)**

## Data Aggregating and Transforming
- using follow script:
```bash
python utils/data_transfom.py
```
- the parameter reference:
  - data_dir: the raw data relative path / absolute path
  - output_dir: the aggregated data saving relative path / absolute path

- **Finally you can get the aggregated data files with 2 types: CSV/Parquet**