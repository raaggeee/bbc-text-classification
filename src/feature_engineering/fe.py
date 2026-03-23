import pandas as pd
from utils import open_yaml, open_df, save_df
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def get_yaml_content(yaml_content):
    data_path = yaml_content["feature_engineering"]["data_path"]
    save_path = yaml_content["feature_engineering"]["save_path"]

    return data_path, save_path

def apply_feature_engineering(train_df, test_df):
    tfidf = TfidfVectorizer()
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    X_train = tfidf.fit_transform(X_train)
    X_test = tfidf.transform(X_test)

    train_vectorized = pd.DataFrame(X_train)
    train_vectorized["label"] = y_train
    test_vectorized = pd.DataFrame(X_test)
    test_vectorized["label"] = y_test
    
    return train_vectorized, test_vectorized


def main():
    yaml_file = open_yaml("config.yaml")
    data_path, save_path = get_yaml_content(yaml_file)

    train_df = open_df(os.path.join(data_path, "train_df.csv"))
    test_df = open_df(os.path.join(data_path, "test_df.csv"))

    train_df, test_df = apply_feature_engineering(train_df, test_df)

    save_df(save_path, "train_df", train_df)
    save_df(save_path, "test_df", test_df)

if __name__ == "__main__":
    main()