from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
import pandas as pd


def create_multilabel_target(df):
    """
    Create a multilabel target DataFrame from the original dataset
    """
    y_multilabel = pd.DataFrame()
    for sentiment in ['fuel', 'machine', 'part']:
        for label in ['negative', 'neutral', 'positive']:
            col_name = f"{sentiment}_{label}"
            y_multilabel[col_name] = (df[sentiment] == label).astype(int)
    return y_multilabel


def get_multilabel_classifier(model_name, **params):
    """
    Return a multi-label classifier with the specified base model
    """
    if model_name == "SVM":
        base_classifier = SVC(
            C=params.get('C', 1.0),
            probability=True,
            random_state=42
        )
    elif model_name == "Multinomial Naive Bayes":
        base_classifier = MultinomialNB(
            alpha=params.get('alpha', 1.0)
        )
    elif model_name == "K-Nearest Neighbors":  
        base_classifier = KNeighborsClassifier(
            n_neighbors=params.get('n_neighbors', 5)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return BinaryRelevance(classifier=base_classifier)


def create_vectorizer(max_features=5000):
    """
    Create a TF-IDF vectorizer
    """
    return TfidfVectorizer(max_features=max_features)


def evaluate_multilabel_model(model, X_test, y_test, label_columns):
    """
    Evaluate a multi-label model and return performance metrics
    """
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    mcm = multilabel_confusion_matrix(y_test, y_pred.toarray())

    # Create comparison DataFrame
    comparison_df = pd.DataFrame()

    # Add columns for actual and predicted values
    for i, label_col in enumerate(label_columns):
        comparison_df[f'{label_col}_actual'] = y_test[label_col].reset_index(
            drop=True)
        comparison_df[f'{label_col}_predicted'] = y_pred.toarray()[:, i]
        comparison_df[f'{label_col}_match'] = comparison_df[f'{label_col}_actual'] == comparison_df[f'{label_col}_predicted']

    return accuracy, mcm, comparison_df, y_pred