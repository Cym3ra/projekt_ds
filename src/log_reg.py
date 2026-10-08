import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def vectorize_text(X_train: pd.Series, X_val: pd.Series, X_test: pd.Series) -> tuple[TfidfVectorizer, any, any, any]:
    """Fits a TF-IDF vectorizer on training data and transforms train, val, and test sets"""
    tfidf = TfidfVectorizer()

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    X_test_tfidf = tfidf.transform(X_test)

    return tfidf, X_train_tfidf, X_val_tfidf, X_test_tfidf

