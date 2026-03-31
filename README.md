# 🎯 AI Prediction Project (Machine Learning + Neural Network)

## 📌 Overview
This project demonstrates the use of **Machine Learning** and **Neural Network (MLP)** models to solve real-world prediction problems.

The system is deployed as a **web application** using Streamlit, allowing users to interact with both models.

---

## 🚀 Features
- 📊 Dashboard visualization
- 🤖 Machine Learning model (Ensemble)
- 🧠 Neural Network model (MLPClassifier)
- 🌍 ML Prediction (Continent Prediction)
- 🎬 NN Prediction (Movie Quality Prediction)
- 📈 Confusion Matrix Visualization
- 📉 Accuracy Graph

---

## 📊 Dataset

### 1. McDonald's Locations Dataset
- Features:
  - Country
  - Number of Locations
  - Year
- Target:
  - Continent

### 2. Rotten Tomatoes Movies Dataset
- Features:
  - Tomatometer Score
  - Year
- Target:
  - Good / Bad Movie

---

## 🤖 Machine Learning Model

### Algorithms Used:
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree

### Technique:
- Ensemble Learning (Voting Classifier)

### Steps:
1. Data Cleaning
2. Encoding Categorical Data
3. Feature Scaling
4. Model Training
5. Model Evaluation

---

## 🧠 Neural Network Model

### Model:
- MLPClassifier (Multi-Layer Perceptron)

### Architecture:
- Hidden Layers: (128 → 64 → 32)
- Activation: ReLU

### Training:
- Optimized using backpropagation
- Automatically minimizes loss

---

## 🌐 Web Application

Built with **Streamlit**

### Pages:
- Dashboard
- ML Explanation
- ML Prediction
- ML Evaluation
- NN Explanation
- NN Prediction
- NN Accuracy

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
