import bootstrap                                  # venv redirect + DLL setup
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from config import BASE_DIR, DATA_FILE, MODEL_FILE, CM_FILE, N_ESTIMATORS, TEST_SIZE

os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

# ── Load dataset ──────────────────────────────────────────────────────────────
if not os.path.exists(DATA_FILE):
    print(f"Error: '{DATA_FILE}' not found. Run '01_collect_data.py' first."); exit(1)

with open(DATA_FILE, 'rb') as f:
    ds = pickle.load(f)

raw_features, raw_labels = ds['features'], ds['labels']
if not raw_features:
    print("Error: Dataset is empty. Record signs first."); exit(1)

# Auto-pad to uniform feature dimension (handles mixed 63-/126-dim samples)
target_dim = max(len(f) for f in raw_features)
X = np.array([np.pad(np.array(f, dtype=np.float32), (0, target_dim - len(f))) for f in raw_features])
y = np.array(raw_labels)
classes = np.unique(y)

if len(classes) < 2:
    print(f"Need at least 2 sign classes (found: {list(classes)}). Add another word."); exit(1)

print("=" * 60)
print(f"Dataset: {len(X)} samples | {X.shape[1]}D features | {len(classes)} classes: {list(classes)}")
for c in classes:
    print(f"  '{c}': {np.sum(y == c)} samples")
print("=" * 60)

# ── Train ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42, stratify=y)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = clf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion matrix plot
cm   = confusion_matrix(y_test, y_pred, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
plt.title(f"Confusion Matrix ({acc*100:.1f}%)")
plt.tight_layout()
plt.savefig(CM_FILE, dpi=300)
print(f"Confusion matrix saved: {CM_FILE}")

# ── Save model ────────────────────────────────────────────────────────────────
with open(MODEL_FILE, 'wb') as f:
    pickle.dump({'model': clf, 'classes': classes, 'feature_dim': X.shape[1]}, f)

print(f"Model saved: {MODEL_FILE}")
print("Run '03_realtime_translator.py' to test live!")
