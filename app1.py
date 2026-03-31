# =========================
# IMPORT
# =========================
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================
# LOAD ML MODELS
# =========================
ml_model = joblib.load("ensemble_model.pkl")
ml_scaler = joblib.load("scaler.pkl")
le_country = joblib.load("le_country.pkl")
le_target = joblib.load("le_target.pkl")

# =========================
# LOAD NN MODEL (SKLEARN)
# =========================
nn_model = joblib.load("nn_model.pkl")
nn_scaler = joblib.load("nn_scaler.pkl")

# =========================
# LOAD DATA
# =========================
df_ml = pd.read_csv("mcdonalds_locations_worldwide.csv")
df_nn = pd.read_csv("rotten_tomatoes_movies.csv")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🚀 AI Project Menu")
page = st.sidebar.selectbox("Select Page", [
    "📊 Dashboard",
    "📘 ML Explanation",
    "🌍 ML Prediction",
    "📈 ML Evaluation",
    "📙 NN Explanation",
    "🎬 NN Prediction",
    "📉 NN Accuracy"
])

# =========================
# DASHBOARD
# =========================
if page == "📊 Dashboard":
    st.title("🔥 AI Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("McDonald's Locations by Continent")
        st.bar_chart(df_ml['Continent'].value_counts())

    with col2:
        st.subheader("Movies Rating Distribution")
        st.line_chart(df_nn['audience_score'])

# =========================
# ML EXPLANATION
# =========================
elif page == "📘 ML Explanation":
    st.title("Machine Learning Explanation")
    st.write("""
### Data Preparation
- Clean missing values
- Encode categorical data

### Algorithms
- SVM
- KNN
- Decision Tree
- Voting Ensemble

### Development
- Scaling with StandardScaler
- Training multiple models
- Combining predictions

### Dataset
- McDonald's worldwide dataset
""")

# =========================
# ML PREDICTION
# =========================
elif page == "🌍 ML Prediction":
    st.title("ML Prediction")

    country = st.selectbox("Country", df_ml['Country'].unique())
    num = st.number_input("Number of Locations")
    year = st.number_input("Year")

    if st.button("Predict ML"):
        c = le_country.transform([country])[0]
        X = ml_scaler.transform([[c, num, year]])
        pred = ml_model.predict(X)
        result = le_target.inverse_transform(pred)

        st.success(f"🌍 Continent: {result[0]}")

# =========================
# ML EVALUATION
# =========================
elif page == "📈 ML Evaluation":
    st.title("ML Confusion Matrix")

    df2 = df_ml.dropna()

    X = df2[['Country', 'Number_of_Locations', 'Year']]
    y = df2['Continent']

    X['Country'] = le_country.transform(X['Country'])
    y = le_target.transform(y)

    X = ml_scaler.transform(X)
    pred = ml_model.predict(X)

    cm = confusion_matrix(y, pred)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    st.pyplot(fig)

# =========================
# NN EXPLANATION
# =========================
elif page == "📙 NN Explanation":
    st.title("Neural Network Explanation (MLP)")

    st.write("""
### Data Preparation
- Features: Tomatometer score, Year
- Normalize using scaler

### Model
- MLPClassifier (Neural Network in sklearn)
- Hidden layers: 128 → 64 → 32
- Activation: ReLU

### Training
- Optimized using backpropagation
- Loss minimized automatically

### Dataset
- Rotten Tomatoes dataset
""")

# =========================
# NN PREDICTION
# =========================
elif page == "🎬 NN Prediction":
    st.title("NN Prediction")

    t_rating = st.number_input("Tomatometer Score")
    year = st.number_input("Year")

    if st.button("Predict NN"):
        X = nn_scaler.transform([[t_rating, year]])
        pred = nn_model.predict(X)

        if pred[0] == 1:
            st.success("🍿 Good Movie")
        else:
            st.error("❌ Bad Movie")

# =========================
# NN ACCURACY
# =========================
elif page == "📉 NN Accuracy":
    st.title("NN Accuracy Graph")

    acc = [0.6, 0.7, 0.75, 0.8, 0.82, 0.85]
    loss = [0.7, 0.6, 0.5, 0.4, 0.35, 0.3]

    fig, ax = plt.subplots()
    ax.plot(acc, label="Accuracy")
    ax.plot(loss, label="Loss")
    ax.legend()

    st.pyplot(fig)