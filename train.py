# ============================================================
# Car Price Prediction - Training Script (With Target Encoding)
# ============================================================

import pandas as pd
import numpy as np
import joblib
import logging

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import category_encoders as ce

# ============================================================
# Logging Configuration
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Starting training pipeline...")

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("cardekho_imputated.csv")

# Drop unwanted column
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)

# Fix seat issue
df.loc[(df['car_name'] == 'Honda City') & (df['seats'] == 0), 'seats'] = 5
df.loc[(df['car_name'] == 'Nissan Kicks') & (df['seats'] == 0), 'seats'] = 5
df = df[df['seats'] > 0]

logging.info(f"Dataset cleaned. Shape: {df.shape}")

# ============================================================
# Split Features & Target
# ============================================================

X = df.drop("selling_price", axis=1)
y = df["selling_price"]

# ============================================================
# Feature Groups
# ============================================================

# High cardinality → Target Encoding
target_encode_features = ["car_name", "brand", "model"]

# Low cardinality → OneHot Encoding
onehot_features = ["seller_type", "fuel_type", "transmission_type"]

# Numerical features
numerical_features = [
    "vehicle_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats"
]

# ============================================================
# Preprocessing
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "target_enc",
            ce.TargetEncoder(smoothing=0.3),
            target_encode_features
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
            onehot_features
        ),
        (
            "num",
            "passthrough",
            numerical_features
        )
    ]
)

# ============================================================
# Model
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# ============================================================
# Full Pipeline
# ============================================================

pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("model", model)
])

# ============================================================
# Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# ============================================================
# Train Model
# ============================================================

logging.info("Training started...")
pipeline.fit(X_train, y_train)
logging.info("Training completed.")

# ============================================================
# Evaluation
# ============================================================

train_pred = pipeline.predict(X_train)
test_pred = pipeline.predict(X_test)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)
mae = mean_absolute_error(y_test, test_pred)
rmse = np.sqrt(mean_squared_error(y_test, test_pred))

logging.info(f"Train R2: {train_r2:.4f}")
logging.info(f"Test R2: {test_r2:.4f}")
logging.info(f"MAE: {mae:.2f}")
logging.info(f"RMSE: {rmse:.2f}")

# ============================================================
# Save Model
# ============================================================

joblib.dump(pipeline, "car_price_model.pkl")

logging.info("Model saved successfully!")