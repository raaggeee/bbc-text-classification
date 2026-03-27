import numpy as np
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string
import re
import yaml
import os
import pandas as pd
import nltk
nltk.download('wordnet')


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

def remove_punc(text):
    punc = string.punctuation
    return text.translate(str.maketrans("", "", punc))

def remove_html(text):
    pattern = re.compile(r"<.*?>")
    return pattern.sub(r'', text)

def remove_url(text):
    pattern = re.compile(r"https?.//\S+|www\.\S+")
    return pattern.sub(r'', text)

def remove_number(text):
    remove_number = [i for i in text if not i.isdigit()]
    return "".join(remove_number)

def handling_stopwords(text):
    new_text = []
    stopword = stopwords.words("english")

    for word in text:
        if word in stopword:
            new_text.append(" ")
        else:
            new_text.append(word)

    return "".join(new_text)

def lemmatization(text):
    wordnet = WordNetLemmatizer()
    
    text = text.split()

    text = [wordnet.lemmatize(word) for word in text]

    return " ".join(text)

def stemming(text):
    port_stem = PorterStemmer()

    text = text.split()

    text = [port_stem.stem(x) for x in text]

    return " ".join(text)

def apply_preprocessing(df):
    df["data"] = df["data"].apply(remove_punc)
    df["data"] = df["data"].apply(remove_number)
    df["data"] = df["data"].apply(remove_html)
    df["data"] = df["data"].apply(remove_url)
    # df["data"] = df["data"].apply(handling_stopwords)
    df["data"] = df["data"].apply(lemmatization)
    df["data"] = df["data"].apply(stemming)

    return df

def get_yaml_content(yaml_content):
    data_path = yaml_content["data_preprocessing"]["data_path"]
    save_path = yaml_content["data_preprocessing"]["save_path"]

    return data_path, save_path

def main():
    yaml_file = open_yaml("config.yaml")
    data_path, save_path = get_yaml_content(yaml_file)

    train_df = open_df(os.path.join(data_path, "train_df.csv"))
    test_df = open_df(os.path.join(data_path, "test_df.csv"))

    train_df = apply_preprocessing(train_df)
    test_df = apply_preprocessing(test_df)

    save_df(save_path, "train_df.csv", train_df)
    save_df(save_path, "test_df.csv", test_df)

if __name__ == "__main__":
    main()
