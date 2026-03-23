import numpy as np
from sklearn.model_selection import train_test_split
from utils import open_df, open_yaml, save_df

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

    save_df(save_path, "train_df", train)
    save_df(save_path, "test_df", test)

if __name__ == "__main__":
    main()

