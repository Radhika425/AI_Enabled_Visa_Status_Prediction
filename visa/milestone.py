import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("Visa_Status_Prediction_Dataset.csv")   # change filename accordingly

# -----------------------------
# 2. DROP UNUSED COLUMNS
# -----------------------------
df.drop(columns=["case_id"], inplace=True)


# -----------------------------
# 3. HANDLE MISSING VALUES
# -----------------------------
# Categorical columns
categorical_cols = ["continent", "education_of_employee", "region_of_employment", "unit_of_wage"]
for col in categorical_cols:
    df[col] = df[col].fillna("Unknown")


# Binary Y/N columns
binary_cols = ["has_job_experience", "requires_job_training", "full_time_position"]
for col in binary_cols:
    df[col] = df[col].fillna("N")

# Numeric columns
numeric_cols = ["no_of_employees", "yr_of_estab", "prevailing_wage"]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# -----------------------------
# 4. CONVERT Y/N TO 0/1
# -----------------------------
for col in binary_cols:
    df[col] = df[col].map({"Y": 1, "N": 0})

# -----------------------------
# 5. FEATURE ENGINEERING
# -----------------------------
# Company age from establishment year
df["company_age"] = 2025 - df["yr_of_estab"]

# Normalize wages (optional)
# Idea: Convert Hour → Year equivalent (assuming 2080 hours/year)
df.loc[df["unit_of_wage"] == "Hour", "prevailing_wage"] *= 2080

# Education Ordinal Encoding
education_order = {
    "High School": 1,
    "Bachelor's": 2,
    "Master's": 3,
    "Doctorate": 4,
    "Unknown": 0
}
df["education_level"] = df["education_of_employee"].map(education_order)

# -----------------------------
# 6. LABEL ENCODE OTHER CATEGORICAL VARIABLES
# -----------------------------
label_encode_cols = ["continent", "region_of_employment", "unit_of_wage"]

le = LabelEncoder()
for col in label_encode_cols:
    df[col] = le.fit_transform(df[col])

# -----------------------------
# 7. ENCODE TARGET VARIABLE
# -----------------------------
# Convert case_status into binary: Certified = 1, Denied = 0
df["case_status"] = df["case_status"].map({
    "Certified": 1,
    "Denied": 0
})

# Remove rows with other statuses (optional)
df = df[df["case_status"].isin([0, 1])]

# -----------------------------
# 8. SAVE CLEANED DATASET
# -----------------------------
df.to_csv("visa_data_cleaned.csv", index=False)

print("Preprocessing complete! Saved as visa_data_cleaned.csv")
df.head()

### Milestone2

import numpy as np

np.random.seed(42)

# Simulated processing time (days)
df["processing_time_days"] = np.random.randint(30, 180, size=len(df))

# Simulated application month (seasonality)
df["application_month"] = np.random.randint(1, 13, size=len(df))

### Processing Time Distribution

plt.figure()
plt.hist(df["processing_time_days"], bins=30)
plt.title("Processing Time Distribution (Days)")
plt.xlabel("Days")
plt.ylabel("Number of Applications")
plt.show()

### Processing Time Across Regions

plt.figure()
sns.boxplot(x="region_of_employment", y="processing_time_days", data=df)
plt.title("Processing Time by Employment Region")
plt.xlabel("Region")
plt.ylabel("Processing Time (Days)")
plt.show()

### Seasonal Trends (Month-wise)

plt.figure()
sns.lineplot(x="application_month", y="processing_time_days", data=df)
plt.title("Seasonal Trend in Processing Time")
plt.xlabel("Application Month")
plt.ylabel("Avg Processing Time (Days)")
plt.show()

### Applicant Origin vs Visa Outcome

plt.figure()
sns.countplot(x="continent", hue="case_status", data=df)
plt.title("Visa Status by Applicant Continent")
plt.xlabel("Continent")
plt.ylabel("Count")
plt.show()

### Workload Indicators

plt.figure()
sns.scatterplot(x="no_of_employees", y="processing_time_days", data=df)
plt.title("Employer Size vs Processing Time")
plt.xlabel("Number of Employees")
plt.ylabel("Processing Time (Days)")
plt.show()

### Feature Importance Analysis

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Drop the original 'education_of_employee' column as it's now encoded in 'education_level'
X = df.drop(columns=["case_status", "education_of_employee", "unit_of_wage"])
y = df["case_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

importances = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

### Plot Feature Importance

plt.figure()
sns.barplot(x="Importance", y="Feature", data=importance_df.head(10))
plt.title("Top 10 Important Features for Visa Status Prediction")
plt.show()

### Milestone3

#Prepare Data for Regression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Features & target
X = df.drop(columns=["processing_time_days", "case_status", "education_of_employee"])
y = df["processing_time_days"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale numerical features (important for Linear Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Training Regression Models

#A. Linear Regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

#B. Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

#C. Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
gbr.fit(X_train, y_train)

y_pred_gbr = gbr.predict(X_test)

###Model Evaluation

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return [model_name, mae, rmse, r2]

results = []
results.append(evaluate_model(y_test, y_pred_lr, "Linear Regression"))
results.append(evaluate_model(y_test, y_pred_rf, "Random Forest"))
results.append(evaluate_model(y_test, y_pred_gbr, "Gradient Boosting"))

results_df = pd.DataFrame(
    results, columns=["Model", "MAE", "RMSE", "R2 Score"]
)

results_df

##Model Comparison Visualization

import matplotlib.pyplot as plt

plt.figure()
plt.bar(results_df["Model"], results_df["RMSE"])
plt.title("RMSE Comparison of Regression Models")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.show()

###Hyperparameter Tuning (Best Model)
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_


####added

#Evaluating Tuned Model
y_pred_best_rf = best_rf.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_best_rf)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_best_rf))
r2 = r2_score(y_test, y_pred_best_rf)

print("Tuned Random Forest Results")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

##Feature Importance (Regression Model)
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

importance_df.head(10)

plt.figure()
plt.barh(importance_df["Feature"][:10], importance_df["Importance"][:10])
plt.title("Top Features Influencing Processing Time")
plt.gca().invert_yaxis()
plt.show()


###Milestone 4

#Train Quantile Gradient Boosting Models
from sklearn.ensemble import GradientBoostingRegressor

# Quantile models
lower_model = GradientBoostingRegressor(
    loss="quantile", alpha=0.1, random_state=42
)

upper_model = GradientBoostingRegressor(
    loss="quantile", alpha=0.9, random_state=42
)

median_model = GradientBoostingRegressor(
    loss="squared_error", random_state=42
)

# Train models
lower_model.fit(X_train, y_train)
upper_model.fit(X_train, y_train)
median_model.fit(X_train, y_train)

#Build the Prediction Engine Function
def estimate_processing_time(user_input_df):
    """
    Input: DataFrame with same feature columns as training data
    Output: Estimated processing time range with confidence interval
    """

    # Predict quantiles
    lower = lower_model.predict(user_input_df)[0]
    median = median_model.predict(user_input_df)[0]
    upper = upper_model.predict(user_input_df)[0]

    return {
        "Estimated Processing Time (Days)": round(median),
        "Range": f"{round(lower)} - {round(upper)} days",
        "Confidence Level": "80%"
    }

#Example User Input
user_input = pd.DataFrame([{
    "continent": 2,
    "has_job_experience": 1,
    "requires_job_training": 1,
    "no_of_employees": 500,
    "yr_of_estab": 2005,
    "region_of_employment": 1,
    "prevailing_wage": 85000,
    "unit_of_wage": 1,
    "full_time_position": 1,
    "company_age": 2025 - 2005,
    "education_level": 3,
    "application_month": 6
}], columns=X.columns)

estimate_processing_time(user_input)

estimate_processing_time(user_input)

###UI

import joblib

joblib.dump(lower_model, "lower_quantile_model.pkl")
joblib.dump(median_model, "median_model.pkl")
joblib.dump(upper_model, "upper_quantile_model.pkl")
joblib.dump(scaler, "scaler.pkl")
