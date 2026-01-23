import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---------------------------
# Load data
# ---------------------------
df = pd.read_csv("visa_data_cleaned.csv")

# ---------------------------
# Encode categorical features
# ---------------------------
encoders = {}

cat_cols = [
    "education_of_employee",
    "has_job_experience",
    "requires_job_training",
    "unit_of_wage",
    "full_time_position",
    "case_status"
]

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ---------------------------
# Features & target
# ---------------------------
X = df[
    [
        "education_of_employee",
        "has_job_experience",
        "requires_job_training",
        "prevailing_wage",
        "unit_of_wage",
        "full_time_position"
    ]
]

y = df["case_status"]

# ---------------------------
# Train regression model
# ---------------------------
model = LinearRegression()
model.fit(X, y)

# ---------------------------
# Save model + encoders
# ---------------------------
joblib.dump(
    {
        "model": model,
        "encoders": encoders
    },
    "model.joblib"
)

print("✅ Model trained and saved as model.joblib")
