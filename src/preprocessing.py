import re
from datasets import load_dataset, DatasetDict
import pandas as pd


def load_imdb_data_as_dataframe() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loading IMDB dataset from Hugging Face and converting it to pandas DataFrame"""
    dataset = load_dataset("stanfordnlp/imdb")
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])
    return train_df, test_df

def load_imdb_raw_dataset() -> DatasetDict:
    """Loading and returning Hugging Face dataset"""
    return load_dataset("stanfordnlp/imdb")

def filter_overlapping_texts(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Identifies and filters out reviews in the test set tha also appear in the training ser (to prevent data leakage)"""
    common_text = set(train_df["text"]).intersection(set(test_df["text"]))

    test_df_clean = test_df[~test_df["text"].isin(common_text)].copy()

    return test_df_clean


def clean_text(text) -> str:
    """Cleaning the reviews by removing HTML tags and converting all the characters to lowercase""" 
    text = re.sub(r'<.*?>', ' ', text)
    text = text.lower()

    return text

def prepare_data_for_logreg() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Loads the dataser, removes overlapping reviews from the test set,
    cleans the text and returns X y for both training and testing"""
    train_df, test_df_full = load_imdb_data_as_dataframe()

    test_df = filter_overlapping_texts(train_df, test_df_full)

    X_train = train_df["text"].apply(clean_text)
    y_train = train_df["label"]

    X_test = test_df["text"].apply(clean_text)
    y_test = test_df["label"]

    return X_train, y_train, X_test, y_test