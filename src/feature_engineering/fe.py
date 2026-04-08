import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
import yaml
import os
import pandas as pd

"""
Checklist
[] - Apply PCA (check which is better)
[] - Apply LDA (check which is better)
[] - 
"""

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
def get_yaml_content(yaml_content):
    data_path = yaml_content["feature_engineering"]["data_path"]
    save_path = yaml_content["feature_engineering"]["save_path"]

    return data_path, save_path

def apply_feature_engineering(train_df, test_df):
    tfidf = TfidfVectorizer(max_features=100)
    label = LabelEncoder()
    
    X_train = train_df["data"]
    y_train = train_df["labels"]

    y_train = label.fit_transform(y_train)
    X_test = test_df["data"]
    y_test = test_df["labels"]
    y_test = label.transform(y_test)

    X_train = tfidf.fit_transform(X_train)
    print(X_train.shape)
    X_test = tfidf.transform(X_test)



    train_vectorized = pd.DataFrame(X_train.toarray())
    print(train_vectorized.shape)
    train_vectorized["label"] = label.fit_transform(y_train)
    test_vectorized = pd.DataFrame(X_test.toarray())
    test_vectorized["label"] = label.fit_transform(y_test)
    
    return train_vectorized, test_vectorized


def main():
    yaml_file = open_yaml("config.yaml")
    data_path, save_path = get_yaml_content(yaml_file)

    train_df = open_df(os.path.join(data_path, "train_df.csv"))
    test_df = open_df(os.path.join(data_path, "test_df.csv"))

    train_df, test_df = apply_feature_engineering(train_df, test_df)

    save_df(save_path, "train_df.csv", train_df)
    save_df(save_path, "test_df.csv", test_df)

if __name__ == "__main__":
    main()