import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st


def plot_label_distribution(df, column):
    """
    Create a bar plot showing the distribution of labels
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    df[column].value_counts().plot(kind='bar', ax=ax)
    plt.title(f"Distribution of {column}")
    plt.tight_layout()
    return fig


def plot_multilabel_confusion_matrix(mcm, label_name):
    """
    Plot a confusion matrix for a specific label
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(mcm, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.title(f'Confusion Matrix: {label_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    return fig
