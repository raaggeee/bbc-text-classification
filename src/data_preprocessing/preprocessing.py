import numpy as np
import os
from utils import open_df, open_yaml, save_df
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string
import re

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
    remove_number = [i for i in text if i.isdigit()]
    return " ".join(remove_number)

def handling_stopwords(text):
    new_text = []
    stopword = stopwords.words("english")

    for word in text:
        if word in stopword:
            new_text.append(" ")
        else:
            new_text.append(word)

    return " ".join(new_text)

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
    df["data"] = df["data"].apply(lambda x: remove_punc(x))
    df["data"] = df["data"].apply(lambda x: remove_number(x))
    df["data"] = df["data"].apply(lambda x: remove_html(x))
    df["data"] = df["data"].apply(lambda x: remove_url(x))
    df["data"] = df["data"].apply(lambda x: handling_stopwords(x))
    df["data"] = df["data"].apply(lambda x: lemmatization(x))
    df["data"] = df["data"].apply(lambda x: stemming(x))

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

    save_df(save_path, "train_df", train_df)
    save_df(save_path, "test_df", test_df)

if __name__ == "__main__":
    main()
