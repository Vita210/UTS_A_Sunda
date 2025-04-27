import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from models.multi_label_classifiers import get_multilabel_classifier, create_vectorizer, evaluate_multilabel_model, create_multilabel_target
from utils.visualization import plot_multilabel_confusion_matrix
from utils.data_loader import load_data

st.set_page_config(page_title="Model Training", layout="wide")
st.title("Model Training")

if 'df' not in st.session_state:
    st.session_state.df = load_data()

# Access data from session state
df = st.session_state.df

# Model selection
model_option = st.selectbox(
    "Select Model", ["SVM", "Multinomial Naive Bayes", "K-Nearest Neighbors"]
)

# Vectorization parameters
st.subheader("Text Vectorization Parameters")
max_features = st.slider(
    "Max Features", min_value=1000, max_value=10000, value=5000, step=1000)

# Training parameters
test_size = st.slider("Test Size", min_value=0.1,
                      max_value=0.5, value=0.2, step=0.05)

# Model-specific parameters
model_params = {}
if model_option == "SVM":
    model_params['C'] = st.slider("Regularization Parameter (C)",
                                  min_value=0.01, max_value=10.0, value=1.0, step=0.01)
elif model_option == "Multinomial Naive Bayes":
    model_params['alpha'] = st.slider("Smoothing Parameter (alpha)",
                                      min_value=0.01, max_value=1.0, value=1.0, step=0.01)
elif model_option == "K-Nearest Neighbors":
    model_params['n_neighbors'] = st.slider("Number of Neighbors (k)", 
                                            min_value=1, max_value=50, value=5, step=1)
    
# Processing
if st.button("Train Model"):
    st.info("Training in progress...")

    # Split data
    X = df['Sentence']

    # Define label columns
    label_columns = ['fuel_negative', 'fuel_neutral', 'fuel_positive',
                     'machine_negative', 'machine_neutral', 'machine_positive',
                     'part_negative', 'part_neutral', 'part_positive']

    # Create multilabel target
    y_multilabel = create_multilabel_target(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_multilabel, test_size=test_size, random_state=42)

    # Vectorize text
    vectorizer = create_vectorizer(max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train model
    model = get_multilabel_classifier(model_option, **model_params)
    model.fit(X_train_tfidf, y_train)

    # Save model, vectorizer, and parameters to session state
    st.session_state.trained_model = model
    st.session_state.vectorizer = vectorizer
    st.session_state.model_name = model_option
    st.session_state.label_columns = label_columns

    # Evaluate model
    accuracy, mcm, comparison_df, y_pred = evaluate_multilabel_model(
        model, X_test_tfidf, y_test, label_columns)

    # Display results
    st.success("Training complete!")
    st.subheader("Model Performance")

    # Display accuracy
    st.write(f"Overall Accuracy: {accuracy:.4f}")

    # Side by side comparison
    st.subheader("Side-by-Side Comparison")

    comparison_df['Text'] = X_test.reset_index(drop=True)
    st.dataframe(comparison_df.head(10))

    # Confusion matrices
    st.subheader("Confusion Matrices")

    # Create 3 rows of 3 columns for all 9 matrices
    for row in range(3):
        cols = st.columns(3)

        for col in range(3):
            label_idx = row * 3 + col
            if label_idx < len(label_columns):
                with cols[col]:
                    fig = plot_multilabel_confusion_matrix(
                        mcm[label_idx], label_columns[label_idx])
                    st.pyplot(fig)
                    plt.close(fig)
