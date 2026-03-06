import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Student Dropout ML Dashboard", layout="wide")

st.title("🎓 Student Dropout Prediction System")

st.markdown("Upload a dataset and compare machine learning models.")

# Sidebar
st.sidebar.header("Controls")

file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if file is not None:

    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Encode categorical columns
    label = LabelEncoder()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = label.fit_transform(df[col])

    df = df.fillna(df.mean())

    st.subheader("Cleaned Dataset")
    st.dataframe(df.head())

    target = st.sidebar.selectbox("Select Target Column", df.columns)

    if target:

        X = df.drop(target, axis=1)
        y = df[target]

        test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier()
        }

        results = {}

        st.subheader("Model Comparison")

        for name, model in models.items():

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)

            results[name] = acc

            st.write(f"{name} Accuracy:", acc)

        best_model_name = max(results, key=results.get)

        st.success(f"🏆 Best Model: {best_model_name}")

        best_model = models[best_model_name]

        # Accuracy Bar Chart
        st.subheader("Model Accuracy Comparison")

        fig, ax = plt.subplots()

        ax.bar(results.keys(), results.values())

        ax.set_ylabel("Accuracy")

        st.pyplot(fig)

        # Target Distribution
        st.subheader("Target Distribution")

        fig2, ax2 = plt.subplots()

        df[target].value_counts().plot(kind="bar", ax=ax2)

        st.pyplot(fig2)

        # Correlation Heatmap
        st.subheader("Feature Correlation")

        corr = df.corr()

        fig3, ax3 = plt.subplots()

        cax = ax3.matshow(corr)

        fig3.colorbar(cax)

        ax3.set_xticks(range(len(corr.columns)))
        ax3.set_yticks(range(len(corr.columns)))

        ax3.set_xticklabels(corr.columns, rotation=90)
        ax3.set_yticklabels(corr.columns)

        st.pyplot(fig3)

        # Prediction panel
        st.subheader("Make Prediction")

        input_data = []

        for col in X.columns:
            val = st.number_input(f"{col}")
            input_data.append(val)

        if st.button("Predict"):

            prediction = best_model.predict([input_data])

            st.success(f"Prediction Result: {prediction[0]}")

else:

    st.info("Upload a dataset to start.")