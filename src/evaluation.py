import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


def calculate_metrics(y_true, y_pred):
    """
    Calculate the main classification metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
    }


def create_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    Create a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"],
    )

    disp.plot(ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    plt.show()
    

    return cm

def create_classification_report(y_true, y_pred):
    """Return a classification report."""
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        zero_division=0,
    )


def get_prediction_errors(
    y_true,
    y_pred,
    texts,
    prediction_probabilities=None
):
    """
    return misclassified reviews, with probabilities if possible
    """
    errors = pd.DataFrame({
        "text": list(texts),
        "actual": list(y_true),
        "predicted": list(y_pred),
    })

    predicted_prob = None

    # Return probabilities if the model has them
    if prediction_probabilities is not None:
        probabilities = np.asarray(prediction_probabilities, dtype="float64")
        rows = np.arange(len(errors))
        predicted_prob = pd.Series(probabilities[rows, y_pred], index=errors.index)

        # assumes the classes are divided as 0 = Negative, 1 = Positive
        if probabilities.shape[1] == 2:
            errors["prob_negative"] = probabilities[:, 0] * 100
            errors["prob_positive"] = probabilities[:, 1] * 100

    # Only keep the wmisclassified reviews
    errors = errors.loc[
        errors["actual"] != errors["predicted"]
    ].copy()

    # Sort to show where it was most confidently wrong
    if predicted_prob is not None:
        order = predicted_prob.loc[errors.index].sort_values(ascending=False).index
        errors = errors.loc[order]

    percentage_columns = ["prob_negative", "prob_positive"]

    for column in percentage_columns:
        if column in errors.columns:
            errors[column] = errors[column].round(2)

    return errors