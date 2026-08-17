import subprocess
import sys


# ==========================================
# 1. CLEAN THE DATA
# ==========================================

print("=" * 60)
print("STEP 1: CLEANING RESTAURANT DATA")
print("=" * 60)

clean_result = subprocess.run(
    [
        sys.executable,
        "src/data_processing/clean_data.py"
    ]
)

if clean_result.returncode != 0:
    print("Data cleaning failed.")
    sys.exit(1)


# ==========================================
# 2. TRAIN THE MODEL
# ==========================================

print()
print("=" * 60)
print("STEP 2: TRAINING AI MODEL")
print("=" * 60)

train_result = subprocess.run(
    [
        sys.executable,
        "src/prediction/waste_prediction.py"
    ]
)

if train_result.returncode != 0:
    print("Model training failed.")
    sys.exit(1)


# ==========================================
# 3. FINISHED
# ==========================================

print()
print("=" * 60)
print("RETRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("The trained model is saved at:")
print("models/waste_prediction_model.pkl") 