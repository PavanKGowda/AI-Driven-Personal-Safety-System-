import joblib
import librosa
import numpy as np

MODEL_PATH = "ai_safety_system/model/svm_model.pkl"

def extract_mfcc(file_path, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"❌ Error extracting MFCC: {e}")
        return None

def predict(file_path):
    try:
        model = joblib.load(MODEL_PATH)
        features = extract_mfcc(file_path)
        if features is None:
            print("⚠️ Could not extract features.")
            return False

        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0]

        print("🎧 Prediction:", "SCREAM" if prediction == 1 else "NON-SCREAM")
        print("📈 Confidence:", f"{max(probability) * 100:.2f}%")

        return prediction == 1
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return False
