import os
import sys
import json
import argparse
import logging

import pandas as pd
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, default = "data/raw_data")
    parser.add_argument('--songs_data_path', type = str, default = "songs.csv")
    parser.add_argument('--members_data_path', type = str, default = "members.csv")
    parser.add_argument('--songs_extra_info_data_path', type = str, default = "song_extra_info.csv")
    parser.add_argument('--output_dir', type = str, default = "data/aggregated_data")
    return parser.parse_args()

def load_csv_data(data_dir: str, data_files: str):
    data_dir = os.path.abspath(data_dir)
    data_path = os.path.join(data_dir, data_files)
    logger.info(f"Loading data from {data_path}")
    # Dataset Version
    # try:
    #     data = load_dataset(
    #         data_files,
    #         data_dir,
    #         split="train",
    #     )
    #     logger.info(f"Data has been successful loaded")
    #     return data
    # except Exception as e:
    #     logger.error(f"Failed to load data: {e}")
    #     logger.info(f"Error data path: {data_path}")
    #     return None
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data has been successfully loaded")
        logger.info(f"Data shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {data_path}, error: {e}")
        return None


def data_aggregated(meta_data: pd.DataFrame, songs_data: pd.DataFrame, members_data: pd.DataFrame, songs_extra_info: pd.DataFrame):
    try:
        df = pd.merge(meta_data, songs_data, on='song_id', how='left')
        df = pd.merge(df, members_data, on='msno', how='left')
        df = pd.merge(df, songs_extra_info, on='song_id', how='left')
        logger.info(f"Data has been successfully aggregated")
        logger.info(f"Data shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to aggregate data, error: {e}")
        return None

def data_transformed(aggregated_data: pd.DataFrame, output_dir: str, mode: str = "train"):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    transformed_msg = "FAIL"
    if aggregated_data is None:
        logger.info(f"No aggregated data, transformed msg: {transformed_msg}")
        return transformed_msg

    data = aggregated_data.copy()
    data.to_csv(os.path.join(output_dir, f"{mode}_aggregated.csv"), index=False)

    data = data.where(pd.notnull(data), None)
    data_list = data.to_dict(orient='records')
    dataset = Dataset.from_list(data_list)
    dataset.to_parquet(os.path.join(output_dir, f"{mode}.parquet"))
    # data.to_parquet(os.path.join(output_dir, f"{mode}.parquet"))
    logger.info(f"Data has been successfully transformed")

    transformed_msg = "SUCCESS"
    logger.info(f"Transformed Message: {transformed_msg}")
    return transformed_msg

def data_transformed_pipeline(
        data_dir: str,
        waiting_list: list,
        songs_data_path: str,
        members_data_path: str,
        songs_extra_info_data_path: str,
        output_dir: str,
):
    songs_data = load_csv_data(data_dir, songs_data_path)
    members_data = load_csv_data(data_dir, members_data_path)
    songs_extra_info_data = load_csv_data(data_dir, songs_extra_info_data_path)
    for waiting_data_path in waiting_list:
        mode = "train" if waiting_data_path.startswith("train") else "test"
        meta_data = load_csv_data(data_dir, waiting_data_path)
        aggregated_data = data_aggregated(meta_data, songs_data, members_data, songs_extra_info_data)
        transformed_msg = data_transformed(aggregated_data, output_dir, mode)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    args = parse_args()
    data_dir = args.data_dir
    songs_data_path = args.songs_data_path
    members_data_path = args.members_data_path
    songs_extra_info_data_path = args.songs_extra_info_data_path
    output_dir = args.output_dir

    train_raw_data = "train.csv"
    test_raw_data = "test.csv"
    waiting_list = [train_raw_data, test_raw_data]

    data_transformed_pipeline(data_dir, waiting_list, songs_data_path, members_data_path, songs_extra_info_data_path, output_dir)



