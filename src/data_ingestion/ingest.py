import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import os
import pandas as pd

def open_yaml(file_path):
    with open(file_path, "r") as f:
        yaml_file = yaml.safe_load(f)

    return yaml_file

def open_df(file_path):
    df = pd.read_csv(file_path)
    return df

def save_df(file_path, filename, df):
    os.makedirs(file_path, exist_ok=True)
    save_path = os.path.join(file_path, filename)
    df.to_csv(save_path, index=False)
    return True

def loads_cofigs(yaml):
    save_path = yaml["data_ingestion"]["path"]

    return save_path

def split_data(df):
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    return train_data, test_data

def main():
    yaml_file = open_yaml("config.yaml")
    save_path = loads_cofigs(yaml_file)

    df = open_df("data/bbc_data.csv")

    train, test = split_data(df)

    save_df(save_path, "train_df.csv", train)
    save_df(save_path, "test_df.csv", test)

if __name__ == "__main__":
    main()

