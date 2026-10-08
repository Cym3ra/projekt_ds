import re
from datasets import load_dataset, DatasetDict
import pandas as pd
from sklearn.model_selection import train_test_split


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

def prepare_data_for_logreg(test_size: float = 0.2, random_state: int = 42) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Loads the dataser, filter overlap,
    cleans the text and splits tarin into train/validation sets"""
    train_df, test_df_full = load_imdb_data_as_dataframe()

    test_df = filter_overlapping_texts(train_df, test_df_full)

    X_full = train_df["text"].apply(clean_text)
    y_full = train_df["label"]

    X_test = test_df["text"].apply(clean_text)
    y_test = test_df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X_full,
        y_full,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, y_train, X_test, y_test