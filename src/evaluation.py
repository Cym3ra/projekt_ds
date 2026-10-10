import pandas as pd
import matplotlib.pyplot as plt

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
    """Return a classification report as a dictionary."""
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        output_dict=True,
        zero_division=0,
    )


def get_prediction_errors(y_true, y_pred, texts):
    """Return only incorrectly classified reviews."""
    errors = pd.DataFrame({
        "text": list(texts),
        "actual": list(y_true),
        "predicted": list(y_pred),
    })

    return errors.loc[
        errors["actual"] != errors["predicted"]
    ].copy()


"""
För att köra i notebook ex:

from src.evaluation import (
    calculate_metrics,
    create_confusion_matrix,
    create_classification_report,
    get_prediction_errors,
)

errors = get_prediction_errors(
    y_val,
    y_val_pred,
    X_val,
)

display(errors.head(10))


cm = create_confusion_matrix(y_val, y_val_pred, title="LogisticRegression Confusion Matrix")
cm

"""