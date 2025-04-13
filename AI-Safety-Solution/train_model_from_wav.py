import os
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import sys
import io
from sklearn.metrics import roc_curve, auc
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
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"\n🎯 Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, MODEL_SAVE_PATH)
print(f"\n💾 Model saved at: {MODEL_SAVE_PATH}")

# === 🔍 VISUALIZATION === #

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Scream", "Scream"])
disp.plot(cmap=plt.cm.Blues)
cm_path = os.path.join(FEATURE_SAVE_PATH, "confusion_matrix.png")
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.savefig(cm_path)
print(f"🖼️ Confusion matrix saved at: {cm_path}")
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()

# 2. Class Distribution (Bar Plot)
class_counts = [y.count(0) if isinstance(y, list) else np.sum(y == 0),
                y.count(1) if isinstance(y, list) else np.sum(y == 1)]

plt.figure(figsize=(6, 4))
sns.barplot(x=["Non-Scream", "Scream"], y=class_counts, palette="Set2")
plt.title("Class Distribution in Dataset")
plt.ylabel("Number of Samples")
plt.xlabel("Class")
plt.tight_layout()

class_dist_path = os.path.join(FEATURE_SAVE_PATH, "class_distribution.png")
plt.savefig(class_dist_path)
print(f"🖼️ Class distribution saved at: {class_dist_path}")
plt.show()

# === 3. ROC Curve === #
y_score = model.decision_function(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

roc_path = os.path.join(FEATURE_SAVE_PATH, "roc_curve.png")
plt.savefig(roc_path)
print(f"🖼️ ROC curve saved at: {roc_path}")
plt.show()


# === 4. MFCC Feature Importance Heatmap === #
# Only applicable for linear SVM
if model.kernel == "linear":
    plt.figure(figsize=(8, 4))
    sns.heatmap(np.array([coef]), cmap="coolwarm", annot=True, xticklabels=feature_names,
            yticklabels=["Importance"], cbar=True)
    plt.title("Feature Importance from SVM Coefficients (MFCCs)")
    plt.tight_layout()

    importance_path = os.path.join(FEATURE_SAVE_PATH, "mfcc_feature_importance.png")
    plt.savefig(importance_path)
    print(f"🖼️ Feature importance heatmap saved at: {importance_path}")
    plt.show()
else:
    print("Feature importance heatmap only supported for linear SVM kernel.")

# === 5. Precision vs Recall Curve for Each Class === #
y_score_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the "Scream" class

precision, recall, _ = precision_recall_curve(y_test, y_score_prob)
avg_precision = average_precision_score(y_test, y_score_prob)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, color="purple", lw=2, label=f"Precision-Recall curve (AP = {avg_precision:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision vs Recall Curve (Scream Class)")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()

# Binarize the output
y_test_binarized = label_binarize(y_test, classes=[0, 1])  # shape: (n_samples, n_classes)
n_classes = y_test_binarized.shape[1]

# Get probability scores for both classes
y_score_prob_all = model.predict_proba(X_test)

# Precision-Recall per class
plt.figure(figsize=(7, 5))

colors = ["blue", "orange"]
class_labels = ["Non-Scream", "Scream"]

for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_score_prob_all[:, i])
    avg_precision = average_precision_score(y_test_binarized[:, i], y_score_prob_all[:, i])
    
    plt.plot(recall, precision, color=colors[i], lw=2,
             label=f"{class_labels[i]} (AP = {avg_precision:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves (Non-Scream Class)")
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()

# === 🖼️ Save the figure === #
output_image_path = os.path.join(FEATURE_SAVE_PATH, "precision_recall_per_class.png")
plt.savefig(output_image_path)
print(f"\n🖼️ Precision-Recall curve image saved at: {output_image_path}")

plt.show()