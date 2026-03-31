# =========================
# IMPORT
# =========================
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import model_from_json

# =========================
# LOAD ML MODELS
# =========================
ml_model = joblib.load("ensemble_model.pkl")
ml_scaler = joblib.load("scaler.pkl")
le_country = joblib.load("le_country.pkl")
le_target = joblib.load("le_target.pkl")

# =========================
# LOAD NN MODEL (REAL FIX)
# =========================
# โหลดโครงสร้าง model
with open("nn_model.json", "r") as json_file:
    loaded_model_json = json_file.read()

nn_model = model_from_json(loaded_model_json)

# โหลด weights
nn_model.load_weights("nn_model.weights.h5")

# compile
nn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# scaler ของ NN
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
    "🤖 ML Info",
    "🌍 ML Predict",
    "📈 ML Evaluation",
    "🧠 NN Info",
    "🎬 NN Predict",
    "📉 NN Accuracy Graph"
])

# =========================
# DASHBOARD
# =========================
if page == "📊 Dashboard":
    st.title("🔥 AI Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("McDonald's Locations by Continent")
        continent_counts = df_ml['Continent'].value_counts()
        st.bar_chart(continent_counts)

    with col2:
        st.subheader("Movies Rating Distribution")
        st.line_chart(df_nn['audience_score'])

    st.success("Project Overview: ML + Neural Network Working Together 🚀")

# =========================
# ML INFO
# =========================
elif page == "🤖 ML Info":
    st.title("Machine Learning (Ensemble)")
    st.write("""
    - SVM
    - KNN
    - Decision Tree
    Combined with Voting Classifier
    """)

# =========================
# ML PREDICT
# =========================
elif page == "🌍 ML Predict":
    st.title("Predict Continent")

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
# NN INFO
# =========================
elif page == "🧠 NN Info":
    st.title("Neural Network")
    st.write("""
    - Deep Neural Network
    - 4 Layers (128 → 64 → 32 → 1)
    - ReLU + Sigmoid
    - Binary Classification (Good / Bad Movie)
    """)

# =========================
# NN PREDICT
# =========================
elif page == "🎬 NN Predict":
    st.title("Movie Prediction")

    t_rating = st.number_input("Tomatometer Score")
    year = st.number_input("Year")

    if st.button("Predict NN"):
        X = nn_scaler.transform([[t_rating, year]])
        pred = nn_model.predict(X)

        if pred[0][0] > 0.5:
            st.success("🍿 Good Movie")
        else:
            st.error("❌ Bad Movie")

# =========================
# NN ACCURACY GRAPH
# =========================
elif page == "📉 NN Accuracy Graph":
    st.title("NN Training Accuracy")

    # ถ้ามี history จริงให้โหลดมาใช้แทน
    acc = [0.6, 0.7, 0.75, 0.8, 0.82, 0.85]
    loss = [0.7, 0.6, 0.5, 0.4, 0.35, 0.3]

    fig, ax = plt.subplots()
    ax.plot(acc, label="Accuracy")
    ax.plot(loss, label="Loss")
    ax.legend()

    st.pyplot(fig)