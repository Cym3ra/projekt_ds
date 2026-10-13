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

def prepare_data_for_logreg(
        train_df: pd.DataFrame,
        test_df_full: pd.DataFrame,
        test_size: float = 0.2, 
        random_state: int = 42
    ):
    """Filter overlapping texts, cleans reviews and splits train into train/validation sets"""

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

    return X_train, X_val, X_test, y_train, y_val, y_test

def load_imdb_raw_dataset() -> DatasetDict:
    """Loading and returning Hugging Face dataset"""
    return load_dataset("stanfordnlp/imdb")


def prepare_data_for_distilbert(test_size: float = 0.2, seed: int = 42):
    """Loads raw Hugging Face datasets, removes train/test overlaps,
    and splits the training data into train and validation sets"""
    dataset = load_imdb_raw_dataset()
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    train_text = set(train_dataset["text"])
    test_dataset = test_dataset.filter(
        lambda example: example["text"] not in train_text
    )

    split = train_dataset.train_test_split(test_size=test_size, seed=seed)
    train_dataset = split["train"]
    val_dataset = split["test"]

    return train_dataset, val_dataset, test_dataset