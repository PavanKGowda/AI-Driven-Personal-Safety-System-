import os
import numpy as np
import librosa
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Parameters
DATASET_PATH = "datasets"
MODEL_SAVE_PATH = "ai_safety_system/model/svm_model.pkl"
FEATURE_SAVE_PATH = "data"
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(FEATURE_SAVE_PATH, exist_ok=True)

def extract_mfcc(file_path, n_mfcc=13, min_length=1.0):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        if len(y)/sr < min_length:  # Skip files shorter than min_length seconds
            print(f"Skipping short file: {file_path}")
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Initialize data holders
X = []
y = []

# Loop through dataset
for label, category in enumerate(["non_scream", "scream"]):
    folder_path = os.path.join(DATASET_PATH, category)
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            features = extract_mfcc(file_path)
            if features is not None:
                X.append(features)
                y.append(label)

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print(f"\n✅ Extracted {len(X)} feature vectors")

# Save features
np.save(os.path.join(FEATURE_SAVE_PATH, "features.npy"), X)
np.save(os.path.join(FEATURE_SAVE_PATH, "labels.npy"), y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"\n🎯 Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, MODEL_SAVE_PATH)
print(f"\n💾 Model saved at: {MODEL_SAVE_PATH}")
