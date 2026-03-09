import os
import glob
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------------
# WINDOWS FIX — must be before SparkSession
# ---------------------------------------------------------------
os.environ["HADOOP_HOME"] = r"C:\hadoop"
os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ.get("PATH", "")

os.environ["PYSPARK_PYTHON"] = r"C:\Users\HP\AppData\Local\Programs\Python\Python311\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\HP\AppData\Local\Programs\Python\Python311\python.exe"

from pyspark.sql import SparkSession

# ===============================================================
# 1. START SPARK SESSION
# ===============================================================
spark = (
    SparkSession.builder
    .appName("RealTime Fraud Detection")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
print("=" * 60)
print("   Real-Time Bank Transaction Fraud Detection")
print("=" * 60)
print(" Spark Started...")

# ===============================================================
# 2. LOAD THE PRE-TRAINED MODEL
# ===============================================================
MODEL_PATH = r"C:\Users\HP\Desktop\Coderzone\fraud_anomaly_detection\fraud_model.pkl"
model = joblib.load(MODEL_PATH)
print(" Model Loaded...")

# ===============================================================
# 3. READ TRANSACTION DATA (CSV file or folder of CSVs)
# ===============================================================
DATA_PATH = r"C:\Users\HP\Desktop\Coderzone\fraud_anomaly_detection\realtime_transactions"

if os.path.isdir(DATA_PATH):
    csv_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {DATA_PATH}")
    pdf = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    print(f" Loaded {len(csv_files)} CSV file(s) — {len(pdf)} transactions total")
else:
    pdf = pd.read_csv(DATA_PATH)
    print(f" Loaded CSV — {len(pdf)} transactions total")

print(f"   Columns found: {list(pdf.columns)}\n")

# ===============================================================
# 4. SHOW RAW DATA VIA SPARK
#    Only select columns that actually exist in the file
# ===============================================================
spark_df = spark.createDataFrame(pdf)

# Show whichever of these columns are present
preview_cols = [c for c in [
    "Transaction_Amount", "Account_Balance", "Transaction_Type",
    "Location", "Merchant_Category", "Authentication_Method"
] if c in pdf.columns]

print(" Sample Raw Transactions:")
spark_df.select(preview_cols).show(5, truncate=False)

# ===============================================================
# 5. FEATURE ENGINEERING
#    Drop non-feature columns (only if they exist)
# ===============================================================
DROP_COLS = ["Transaction_ID", "User_ID", "Timestamp", "Fraud_Label"]
existing_drop = [c for c in DROP_COLS if c in pdf.columns]
pdf_feat = pdf.drop(columns=existing_drop)

# ---------------------------------------------------------------
# 5a. Extract time features from Timestamp (if present)
# ---------------------------------------------------------------
if "Timestamp" in pdf.columns:
    pdf["Timestamp"] = pd.to_datetime(pdf["Timestamp"])
    pdf_feat["Hour"]      = pdf["Timestamp"].dt.hour
    pdf_feat["DayOfWeek"] = pdf["Timestamp"].dt.dayofweek
    pdf_feat["Month"]     = pdf["Timestamp"].dt.month

# ---------------------------------------------------------------
# 5b. One-Hot Encode all categorical columns
# ---------------------------------------------------------------
CATEGORICAL_COLS = [
    "Transaction_Type",
    "Device_Type",
    "Location",
    "Merchant_Category",
    "Card_Type",
    "Authentication_Method"
]

cat_cols_present = [c for c in CATEGORICAL_COLS if c in pdf_feat.columns]
pdf_encoded = pd.get_dummies(pdf_feat, columns=cat_cols_present)
print(f" Encoded {len(cat_cols_present)} categorical columns")

# ===============================================================
# 6. ALIGN FEATURES TO EXACTLY MATCH THE MODEL
# ===============================================================
if hasattr(model, "feature_names_in_"):
    expected_features = model.feature_names_in_.tolist()

    # Add any missing columns as 0
    for col in expected_features:
        if col not in pdf_encoded.columns:
            pdf_encoded[col] = 0

    # Select features in the exact order the model expects
    X = pdf_encoded[expected_features]
    print(f" Features aligned: {X.shape[1]} features matched to model\n")
else:
    X = pdf_encoded.select_dtypes(include=[np.number])
    print(f" Using {X.shape[1]} numeric features\n")

# ===============================================================
# 7. PREDICT FRAUD
# ===============================================================
predictions = model.predict(X)
pdf["Fraud_Prediction"] = predictions

if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X)[:, 1]
    pdf["Fraud_Probability"] = (proba * 100).round(2)

    def risk_level(p):
        if p >= 75:   return "HIGH RISK"
        elif p >= 40: return "MEDIUM RISK"
        else:         return "LOW RISK"

    pdf["Risk_Level"] = pdf["Fraud_Probability"].apply(risk_level)

# ===============================================================
# 8. DISPLAY FULL RESULTS VIA SPARK
#    Only include columns that exist in the dataframe
# ===============================================================
result_cols = [c for c in [
    "Transaction_Amount", "Account_Balance",
    "Transaction_Type", "Location", "Merchant_Category",
    "Authentication_Method", "Fraud_Probability",
    "Risk_Level", "Fraud_Prediction"
] if c in pdf.columns]

result_spark = spark.createDataFrame(pdf[result_cols])
print(" Full Prediction Results:")
result_spark.show(truncate=False)

# ===============================================================
# 9. SUMMARY
# ===============================================================
fraud_df = pdf[pdf["Fraud_Prediction"] == 1]
legit_df = pdf[pdf["Fraud_Prediction"] == 0]

print(f"\n{'=' * 60}")
print(f"   FRAUD DETECTION SUMMARY")
print(f"{'=' * 60}")
print(f"  Total Transactions    : {len(pdf)}")
print(f"   Legitimate          : {len(legit_df)}")
print(f"   Fraudulent          : {len(fraud_df)}")
print(f"  Fraud Rate            : {len(fraud_df)/len(pdf)*100:.2f}%")
if "Fraud_Probability" in pdf.columns and len(fraud_df) > 0:
    print(f"  Avg Fraud Probability : {fraud_df['Fraud_Probability'].mean():.2f}%")
print(f"{'=' * 60}\n")

# ===============================================================
# 10. SHOW ONLY FRAUD TRANSACTIONS
# ===============================================================
if not fraud_df.empty:
    print(" FRAUDULENT TRANSACTIONS DETECTED:")
    fraud_spark = spark.createDataFrame(fraud_df[result_cols])
    fraud_spark.show(truncate=False)
else:
    print(" No fraudulent transactions detected in this batch.")

# ===============================================================
# 11. BREAKDOWN BY CATEGORY
# ===============================================================
if not fraud_df.empty and "Transaction_Type" in pdf.columns:
    print("\nFraud Breakdown by Transaction Type:")
    breakdown = (
        pdf.groupby("Transaction_Type")["Fraud_Prediction"]
        .agg(Total="count", Fraudulent="sum")
        .assign(Fraud_Rate_Pct=lambda x: (x["Fraudulent"] / x["Total"] * 100).round(2))
    )
    print(breakdown.to_string())

if not fraud_df.empty and "Location" in pdf.columns:
    print("\n Fraud Breakdown by Location:")
    loc_breakdown = (
        pdf.groupby("Location")["Fraud_Prediction"]
        .agg(Total="count", Fraudulent="sum")
        .assign(Fraud_Rate_Pct=lambda x: (x["Fraudulent"] / x["Total"] * 100).round(2))
        .sort_values("Fraudulent", ascending=False)
    )
    print(loc_breakdown.to_string())

# ===============================================================
# 12. SAVE FRAUD ALERTS TO CSV
# ===============================================================
OUTPUT_PATH = r"C:\Users\HP\Desktop\Coderzone\fraud_anomaly_detection\fraud_alerts.csv"
fraud_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n Fraud alerts saved to: {OUTPUT_PATH}")

# ===============================================================
# 13. STOP SPARK
# ===============================================================
spark.stop()
print("\n Done. Spark stopped.")
