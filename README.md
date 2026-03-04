# Used Car Price Prediction  
### End-to-End Machine Learning Project with Deployment

Predict the resale value of used cars using Machine Learning.  
This project builds a complete ML pipeline using the CarDekho dataset and deploys it using Streamlit for real-time predictions.

---

## 📌 Business Problem

Car resale platforms like **CarDekho** require accurate pricing systems to:

- Help sellers price vehicles competitively
- Assist buyers in evaluating fair market value
- Increase trust and transaction efficiency
- Reduce negotiation gaps using data-driven pricing

### 🎯 Objective

Build a machine learning model that accurately predicts the **selling price** of used cars based on vehicle attributes.

---

## 📊 Dataset Overview

The dataset contains **15,000+ used car listings** with the following features:

### 🔹 Numerical Features
- Vehicle Age  
- Kilometers Driven  
- Mileage  
- Engine Capacity  
- Max Power  
- Seats  

### 🔹 Categorical Features
- Car Name  
- Brand  
- Model  
- Seller Type  
- Fuel Type  
- Transmission Type  

### 🎯 Target Variable
- `selling_price`

---

## 🔎 Exploratory Data Analysis (EDA)

### 📌 Key Insights

- 🔥 **Max Power (0.75 correlation)** has the strongest positive relationship with price.
- 🚗 Engine size and car model significantly influence resale value.
- 📉 Vehicle age negatively impacts price (depreciation effect).
- 🛣 Km driven has weaker impact than expected.
- 🏷 Brand value plays a major role in premium pricing.

---

## ⚙️ Feature Engineering

### 🧠 Encoding Strategy

A hybrid encoding approach was implemented:

### 🔹 Target Encoding (High Cardinality Features)
- `car_name`
- `brand`
- `model`

Target encoding helps capture brand/model pricing behavior without exploding feature space.

### 🔹 One-Hot Encoding (Low Cardinality Features)
- `seller_type`
- `fuel_type`
- `transmission_type`

This ensures:
- No multicollinearity
- Better generalization
- Prevention of data leakage (inside Pipeline)

---

## 🤖 Models Implemented & Comparison

| Model              | Train R² | Test R² |
|--------------------|----------|---------|
| Linear Regression  | ~0.67    | ~0.67   |
| Decision Tree      | ~0.93    | ~0.88   |
| Random Forest      | ~0.95    | ~0.93   |

---

## 🏆 Final Model: Random Forest Regressor

Random Forest was selected because:

- Handles non-linear relationships effectively
- Reduces overfitting via ensemble learning
- Performs best on unseen test data
- Robust to outliers and feature scaling

---

## 📈 Evaluation Metrics

Model performance was evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

### ✅ Final Test R² ≈ 0.93

This indicates strong predictive performance and good generalization.

---

## 🧠 ML Pipeline Architecture

The entire workflow is built using a **Scikit-Learn Pipeline**:

```
Data Cleaning
      ↓
Duplicate Removal
      ↓
Feature Encoding (Target + OneHot)
      ↓
Train-Test Split
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Saving (.pkl)
      ↓
Streamlit Deployment
```

✔ All preprocessing steps are included inside the pipeline  
✔ Prevents data leakage  
✔ Ensures production-ready structure  

---

## 🌐 Streamlit Web Application

An interactive web app allows users to:

- Input car details
- Predict resale price instantly
- View estimated market value in real-time

### ▶ Run Locally

```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

---

## 📂 Project Structure

```
CarDekho_Price_Prediction/
│
├── cardekho_imputated.csv
├── train.py
├── app.py
├── car_price_model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Skills Demonstrated

- Data Cleaning & Preprocessing  
- Exploratory Data Analysis (EDA)  
- Target Encoding & One-Hot Encoding  
- Model Comparison & Evaluation  
- Pipeline Design  
- Model Serialization (.pkl)  
- Streamlit Deployment  
- End-to-End ML Project Development  

---

## 👨‍💻 Author

**Vivek Sonawane**  
Aspiring Data Analyst / Data Scientist / ML Engineer  

---

⭐ If you found this project useful, feel free to star the repository!
