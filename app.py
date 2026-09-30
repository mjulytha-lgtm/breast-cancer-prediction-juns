# Deployment using Streamlit
# upload local using your computer, so no need to create account
# An account needed if you want to host it online using
# Streamlit Community Cloud

# Requirement:
# pip install streamlit tensorflow scikit-learn joblib

import os
from tensorflow.keras.models import load_model
import joblib
import streamlit as st
import numpy as np
import pandas as pd

# ==========================================
# 1. Load Model and Scaler
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    model = load_model(os.path.join(BASE_DIR, "model", "breast_cancer_model.keras"))
    scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))
    return model, scaler

model, scaler = load_artifacts()

# ==========================================
# 2. Application Title
# ==========================================
st.title("Breast Cancer Prediction App")

st.write(
    "Use either manual input or upload a CSV file "
    "to perform breast cancer prediction."
)

# ==========================================
# 3. Feature Names
# ==========================================
features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]

# Nama kolom pada file TEST-breast-cancer-pred.csv berbeda dengan nama fitur
# scikit-learn di atas (contoh: "radius_mean" untuk "mean radius").
# Urutan kolomnya sama, sehingga cukup dibuat pemetaan (mapping) nama kolom.
csv_columns = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]
csv_to_features = dict(zip(csv_columns, features))

# ==========================================
# 4. Select Input Method
# ==========================================
input_method = st.radio(
    "Select Input Method:",
    ["Manual Input", "CSV File"],
    horizontal=True
)

# ==========================================
# 5. Manual Input
# ==========================================
if input_method == "Manual Input":
    st.subheader("Enter Feature Values")
    input_data = []

    # Use two columns to make the interface more compact
    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):
        if i % 2 == 0:
            with col1:
                value = st.number_input(
                    feature,
                    min_value=0.0,
                    value=0.0,
                    step=0.001,
                    format="%.6f",
                    key=feature
                )
        else:
            with col2:
                value = st.number_input(
                    feature,
                    min_value=0.0,
                    value=0.0,
                    step=0.001,
                    format="%.6f",
                    key=feature
                )
        input_data.append(value)

# ==========================================
# 6. CSV File Input
# ==========================================
else:
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        csv_data = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(csv_data.head())

        # Samakan nama kolom CSV (radius_mean, dst.) dengan nama fitur model
        csv_features = csv_data.rename(columns=csv_to_features)

        # Check whether all required features exist
        missing_features = [
            feature for feature in features
            if feature not in csv_features.columns
        ]

        if missing_features:
            st.error(
                "The CSV file is missing the following features:"
            )
            st.write(missing_features)

        else:
            st.success(
                f"CSV file loaded successfully. " f"{len(csv_data)} records found."
            )

# ==========================================
# 7. Run Prediction
# ==========================================
st.divider()

if st.button("Run Predict", type="primary"):

    # --------------------------------------
    # Manual Input Prediction
    # --------------------------------------
    if input_method == "Manual Input":

        # Convert input to DataFrame
        input_df = pd.DataFrame(
            [input_data], columns=features
        )

        # Apply scaler
        input_scaled = scaler.transform(input_df.values)

        # Model inference
        prediction = model.predict(
            input_scaled,
            verbose=0
        )

        # Output sigmoid = probabilitas kelas 1 (Benign),
        # karena pada dataset scikit-learn: 0 = Malignant, 1 = Benign
        probability = prediction[0][0]

        # Convert probability to class
        if probability >= 0.5:
            result = "Benign"
        else:
            result = "Malignant"

        # Display result
        st.subheader("Prediction Result")

        if result == "Malignant":
            st.error(f"Prediction: **{result}**")

        else:
            st.success(f"Prediction: **{result}**")

        st.write(
            f"Prediction probability (Benign): **{probability:.4f}**"
        )

    # --------------------------------------
    # CSV Prediction
    # --------------------------------------
    else:
        if uploaded_file is None:
            st.warning(
                "Please upload a CSV file first."
            )
        elif missing_features:
            st.error(
                "Prediction cannot be performed because "
                "some required features are missing."
            )

        else:
            # Select only required features
            X_new = csv_features[features]

            # Apply the same scaler
            X_scaled = scaler.transform(X_new.values)

            # Model inference
            predictions = model.predict( X_scaled, verbose=0 )

            # Convert probabilities to classes
            probabilities = predictions.flatten()

            results = np.where(
                probabilities >= 0.5, "Benign", "Malignant"
            )

            # Create result dataframe
            result_df = csv_data.copy()
            result_df["Prediction Probability"] = probabilities
            result_df["Prediction"] = results

            # Display results
            st.subheader("Prediction Results")
            st.caption("Prediction Probability = probabilitas kelas Benign.")
            st.dataframe(result_df)

            # Summary
            malignant_count = np.sum(results == "Malignant")
            benign_count = np.sum(results == "Benign")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Benign", benign_count)
            with col2:
                st.metric("Malignant", malignant_count)

            # Jika CSV memiliki kolom diagnosis (M/B), bandingkan dengan prediksi
            if "diagnosis" in csv_data.columns:
                actual = csv_data["diagnosis"].map({"M": "Malignant", "B": "Benign"})
                accuracy = np.mean(actual.values == results)
                st.metric("Accuracy vs diagnosis column", f"{accuracy:.2%}")

            # Download prediction results
            csv_output = result_df.to_csv(
                index=False
            )

            st.download_button(
                label="Download Prediction Results",
                data=csv_output,
                file_name="breast_cancer_predictions.csv",
                mime="text/csv"
            )
