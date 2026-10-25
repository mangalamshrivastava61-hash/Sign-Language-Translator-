import os
import sys
import subprocess

# Auto-redirect to .venv Python if executed outside virtual environment
_base_dir = os.path.dirname(os.path.abspath(__file__))
_venv_py = os.path.join(_base_dir, ".venv", "Scripts", "python.exe")
if os.path.exists(_venv_py) and sys.executable.lower() != os.path.abspath(_venv_py).lower():
    sys.exit(subprocess.call([_venv_py] + sys.argv))

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

DATA_FILE = os.path.join(_base_dir, "data", "gestures_data.pickle")
MODEL_DIR = os.path.join(_base_dir, "model")
MODEL_FILE = os.path.join(MODEL_DIR, "sign_classifier.p")
CONFUSION_MATRIX_FILE = os.path.join(MODEL_DIR, "confusion_matrix.png")

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(DATA_FILE):
    print(f"Error: Dataset '{DATA_FILE}' not found! Please run '01_collect_data.py' first.")
    exit(1)

print("Loading dataset...")
with open(DATA_FILE, 'rb') as f:
    dataset = pickle.load(f)

X = np.array(dataset['features'])
y = np.array(dataset['labels'])
classes = np.unique(y)

if len(classes) < 2:
    print(f"\nError: You have only recorded {len(classes)} sign ({classes}).")
    print("A classifier needs at least 2 different signs to learn! Please run '01_collect_data.py' and add another word.")
    exit(1)

print("=" * 60)
print(f"Dataset summary:")
print(f" - Total samples: {len(X)}")
print(f" - Feature dimension: {X.shape[1]} (21 normalized 3D landmarks)")
print(f" - Classes ({len(classes)}): {list(classes)}")
for c in classes:
    print(f"   * '{c}': {np.sum(y == c)} samples")
print("=" * 60)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples | Testing set: {len(X_test)} samples")

# Train Random Forest
print("\nTraining RandomForestClassifier...")
clf = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("=" * 60)
print(f"MODEL ACCURACY ON TEST SET: {acc * 100:.2f}%")
print("=" * 60)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
plt.title(f"Random Forest - Sign Language Confusion Matrix ({acc*100:.1f}%)")
plt.tight_layout()
plt.savefig(CONFUSION_MATRIX_FILE, dpi=300)
print(f"Confusion matrix plot saved: {CONFUSION_MATRIX_FILE}")

# Save Model
model_payload = {
    'model': clf,
    'classes': classes,
    'feature_dim': X.shape[1]
}
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model_payload, f)

print(f"Trained model saved to: {MODEL_FILE}")
print("\nAll done! Now run 'python 03_realtime_translator.py' to test your signs in real time!")
