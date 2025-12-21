# AI_Enabled_Visa_Status_Prediction #
AI-Enabled Visa Status Prediction and Processing Time Estimator uses machine learning to analyze applicant, employer, and wage factors to predict visa approval outcomes and estimate processing duration. The system helps reduce uncertainty and provides data-driven insights for applicants and immigration teams.

✨ Features
Visa approval status prediction (Certified / Denied)
Processing time estimation (in days)
Fully automated data preprocessing pipeline
Handling missing values and encoding categorical fields
Feature engineering (company age, wage normalization, education levels)
ML models for both classification and regression
Performance evaluation with visualizations
Export of cleaned dataset

🗂️ Dataset Details
Total Columns: 12 (e.g., continent, education, job experience, region, wage, etc.)
Target Variable: case_status
Dataset Type: Public/synthetic (based on historical visa attributes)

Preprocessing Steps:
Filling missing values (categorical → “Unknown”, numerical → median)
Converting Y/N values to binary (0/1)
Label Encoding & One-Hot Encoding where required
Feature Engineering:
company_age = 2025 − employer establishment year
Wage normalization (Hourly → Yearly)
Education level ordinal mapping

Exploratory Data Analysis: Exploratory Data Analysis revealed significant variations in visa processing time across regions and seasons. Mid-year application periods exhibited increased processing delays, likely due to higher workloads. Feature importance analysis highlighted prevailing wage, employer characteristics, and education level as key determinants of visa approval outcomes.
