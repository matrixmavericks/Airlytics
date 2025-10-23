<<<<<<< HEAD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, log_loss
)
from sklearn.preprocessing import RobustScaler, OneHotEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.calibration import calibration_curve

import warnings
warnings.filterwarnings("ignore")

# --- Load Data ---
df = pd.read_csv('test.csv')
if 'id' in df.columns:
    df = df.drop('id', axis=1)
df = df.dropna(subset=['satisfaction'])

y = df['satisfaction'].astype('category').cat.codes
X = df.drop('satisfaction', axis=1)
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = [col for col in X.columns if col not in cat_cols]

# --- Preprocessing ---
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

# --- Split & SMOTE ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42, stratify=y
)
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)
X_train_proc, y_train_proc = SMOTE(random_state=42).fit_resample(X_train_proc, y_train)

# --- Model ---
gb = GradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=2,
    n_estimators=100,
    random_state=42
)
gb.fit(X_train_proc, y_train_proc)
y_train_pred = gb.predict(X_train_proc)
y_test_pred = gb.predict(X_test_proc)
y_test_proba = gb.predict_proba(X_test_proc)

# --- Statistics ---
train_f1 = f1_score(y_train_proc, y_train_pred, average='macro')
test_f1 = f1_score(y_test, y_test_pred, average='macro')
train_acc = accuracy_score(y_train_proc, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
fit_diff = train_f1 - test_f1
logloss = log_loss(y_test, y_test_proba)
print(f"Train F1: {train_f1:.4f}")
print(f"Test F1: {test_f1:.4f}")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Fit Difference: {fit_diff:.4f}")
print(f"Test Log Loss: {logloss:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# --- Prepare for plotting ---
n_classes = len(np.unique(y))
feature_names = (
    num_cols + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols))
)
indices = np.argsort(gb.feature_importances_)[::-1]
y_test_bin = label_binarize(y_test, classes=np.arange(n_classes)) if n_classes > 2 else None

# --- 1. Confusion Matrix ---
plt.figure(figsize=(7,6))
sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# --- 2. ROC Curve ---
plt.figure(figsize=(7,6))
if n_classes == 2:
    fpr, tpr, _ = roc_curve(y_test, y_test_proba[:, 1])
    auc = roc_auc_score(y_test, y_test_proba[:, 1])
    plt.plot(fpr, tpr, label=f'AUC={auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
else:
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_proba[:, i])
        plt.plot(fpr, tpr, label=f'Class {i} (AUC={roc_auc_score(y_test_bin[:, i], y_test_proba[:, i]):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve (multi-class)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
plt.show()

# --- 3. Precision-Recall Curve ---
plt.figure(figsize=(7,6))
if n_classes == 2:
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba[:, 1])
    plt.plot(recall, precision, label='PR Curve')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
else:
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_test_proba[:, i])
        plt.plot(recall, precision, label=f'Class {i}')
    plt.title('Precision-Recall Curve (multi-class)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
plt.show()

# --- 4. Feature Importances ---
plt.figure(figsize=(12,6))
plt.bar(range(len(indices)), gb.feature_importances_[indices], align="center")
plt.xticks(range(len(indices)), np.array(feature_names)[indices], rotation=90)
plt.title("Feature Importances")
plt.tight_layout()
plt.show()

# --- 5. Permutation Importance ---
perm = permutation_importance(gb, X_test_proc, y_test, n_repeats=10, random_state=42)
sorted_idx = perm.importances_mean.argsort()[::-1]
plt.figure(figsize=(12,6))
plt.bar(range(len(sorted_idx)), perm.importances_mean[sorted_idx], align="center")
plt.xticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx], rotation=90)
plt.title("Permutation Importances")
plt.tight_layout()
plt.show()

# --- 6. Learning Curve ---
train_sizes, train_scores, test_scores = learning_curve(
    gb, X_train_proc, y_train_proc, cv=5, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scoring='f1_macro'
)
plt.figure(figsize=(8,6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Train F1")
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="CV F1")
plt.xlabel("Training examples")
plt.ylabel("F1 Score")
plt.title("Learning Curve")
plt.legend()
plt.show()

# --- 7. Partial Dependence Plot (top 2 features) ---
try:
    fig, ax = plt.subplots(1, 2, figsize=(14,6))
    PartialDependenceDisplay.from_estimator(
        gb, X_test_proc, [indices[0], indices[1]], feature_names=np.array(feature_names)[indices[:2]], ax=ax
    )
    plt.show()
except Exception as e:
    print("Partial dependence plot error:", e)

# --- 8. Prediction Probability Histogram ---
plt.figure(figsize=(7,6))
plt.hist(np.max(y_test_proba, axis=1), bins=20, color='skyblue')
plt.title("Histogram of Max Prediction Probabilities (Test)")
plt.xlabel("Max Probability")
plt.ylabel("Frequency")
plt.show()

# --- 9. Actual vs Predicted Counts ---
plt.figure(figsize=(7,6))
sns.countplot(x=y_test, label='Actual', color='blue', alpha=0.5)
sns.countplot(x=y_test_pred, label='Predicted', color='red', alpha=0.5)
plt.title("Actual vs Predicted Class Counts")
plt.legend(['Actual', 'Predicted'])
plt.show()

# --- 10. Calibration Curve ---
plt.figure(figsize=(7,6))
prob_true, prob_pred = calibration_curve(y_test == y_test_pred, np.max(y_test_proba, axis=1), n_bins=10)
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1],'k--')
plt.title("Calibration Curve")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.show()

# --- 11. Cumulative Gains Chart ---
plt.figure(figsize=(7,6))
fpr, tpr, thresholds = roc_curve((y_test == y_test_pred), np.max(y_test_proba, axis=1))
plt.plot(tpr, label='Cumulative Gains')
plt.plot([0,1],[0,1],'k--')
plt.title("Cumulative Gains Chart")
plt.xlabel("Proportion of Samples")
plt.ylabel("Proportion of Positives")
plt.legend()
plt.show()

print("All statistics and visualizations complete for Model 1.")
=======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, log_loss
)
from sklearn.preprocessing import RobustScaler, OneHotEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.calibration import calibration_curve

import warnings
warnings.filterwarnings("ignore")

# --- Load Data ---
df = pd.read_csv('test.csv')
if 'id' in df.columns:
    df = df.drop('id', axis=1)
df = df.dropna(subset=['satisfaction'])

y = df['satisfaction'].astype('category').cat.codes
X = df.drop('satisfaction', axis=1)
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = [col for col in X.columns if col not in cat_cols]

# --- Preprocessing ---
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

# --- Split & SMOTE ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42, stratify=y
)
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)
X_train_proc, y_train_proc = SMOTE(random_state=42).fit_resample(X_train_proc, y_train)

# --- Model ---
gb = GradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=2,
    n_estimators=100,
    random_state=42
)
gb.fit(X_train_proc, y_train_proc)
y_train_pred = gb.predict(X_train_proc)
y_test_pred = gb.predict(X_test_proc)
y_test_proba = gb.predict_proba(X_test_proc)

# --- Statistics ---
train_f1 = f1_score(y_train_proc, y_train_pred, average='macro')
test_f1 = f1_score(y_test, y_test_pred, average='macro')
train_acc = accuracy_score(y_train_proc, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
fit_diff = train_f1 - test_f1
logloss = log_loss(y_test, y_test_proba)
print(f"Train F1: {train_f1:.4f}")
print(f"Test F1: {test_f1:.4f}")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Fit Difference: {fit_diff:.4f}")
print(f"Test Log Loss: {logloss:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# --- Prepare for plotting ---
n_classes = len(np.unique(y))
feature_names = (
    num_cols + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols))
)
indices = np.argsort(gb.feature_importances_)[::-1]
y_test_bin = label_binarize(y_test, classes=np.arange(n_classes)) if n_classes > 2 else None

# --- 1. Confusion Matrix ---
plt.figure(figsize=(7,6))
sns.heatmap(confusion_matrix(y_test, y_test_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# --- 2. ROC Curve ---
plt.figure(figsize=(7,6))
if n_classes == 2:
    fpr, tpr, _ = roc_curve(y_test, y_test_proba[:, 1])
    auc = roc_auc_score(y_test, y_test_proba[:, 1])
    plt.plot(fpr, tpr, label=f'AUC={auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
else:
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_proba[:, i])
        plt.plot(fpr, tpr, label=f'Class {i} (AUC={roc_auc_score(y_test_bin[:, i], y_test_proba[:, i]):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve (multi-class)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
plt.show()

# --- 3. Precision-Recall Curve ---
plt.figure(figsize=(7,6))
if n_classes == 2:
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba[:, 1])
    plt.plot(recall, precision, label='PR Curve')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
else:
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_test_proba[:, i])
        plt.plot(recall, precision, label=f'Class {i}')
    plt.title('Precision-Recall Curve (multi-class)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
plt.show()

# --- 4. Feature Importances ---
plt.figure(figsize=(12,6))
plt.bar(range(len(indices)), gb.feature_importances_[indices], align="center")
plt.xticks(range(len(indices)), np.array(feature_names)[indices], rotation=90)
plt.title("Feature Importances")
plt.tight_layout()
plt.show()

# --- 5. Permutation Importance ---
perm = permutation_importance(gb, X_test_proc, y_test, n_repeats=10, random_state=42)
sorted_idx = perm.importances_mean.argsort()[::-1]
plt.figure(figsize=(12,6))
plt.bar(range(len(sorted_idx)), perm.importances_mean[sorted_idx], align="center")
plt.xticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx], rotation=90)
plt.title("Permutation Importances")
plt.tight_layout()
plt.show()

# --- 6. Learning Curve ---
train_sizes, train_scores, test_scores = learning_curve(
    gb, X_train_proc, y_train_proc, cv=5, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scoring='f1_macro'
)
plt.figure(figsize=(8,6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Train F1")
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="CV F1")
plt.xlabel("Training examples")
plt.ylabel("F1 Score")
plt.title("Learning Curve")
plt.legend()
plt.show()

# --- 7. Partial Dependence Plot (top 2 features) ---
try:
    fig, ax = plt.subplots(1, 2, figsize=(14,6))
    PartialDependenceDisplay.from_estimator(
        gb, X_test_proc, [indices[0], indices[1]], feature_names=np.array(feature_names)[indices[:2]], ax=ax
    )
    plt.show()
except Exception as e:
    print("Partial dependence plot error:", e)

# --- 8. Prediction Probability Histogram ---
plt.figure(figsize=(7,6))
plt.hist(np.max(y_test_proba, axis=1), bins=20, color='skyblue')
plt.title("Histogram of Max Prediction Probabilities (Test)")
plt.xlabel("Max Probability")
plt.ylabel("Frequency")
plt.show()

# --- 9. Actual vs Predicted Counts ---
plt.figure(figsize=(7,6))
sns.countplot(x=y_test, label='Actual', color='blue', alpha=0.5)
sns.countplot(x=y_test_pred, label='Predicted', color='red', alpha=0.5)
plt.title("Actual vs Predicted Class Counts")
plt.legend(['Actual', 'Predicted'])
plt.show()

# --- 10. Calibration Curve ---
plt.figure(figsize=(7,6))
prob_true, prob_pred = calibration_curve(y_test == y_test_pred, np.max(y_test_proba, axis=1), n_bins=10)
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1],[0,1],'k--')
plt.title("Calibration Curve")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.show()

# --- 11. Cumulative Gains Chart ---
plt.figure(figsize=(7,6))
fpr, tpr, thresholds = roc_curve((y_test == y_test_pred), np.max(y_test_proba, axis=1))
plt.plot(tpr, label='Cumulative Gains')
plt.plot([0,1],[0,1],'k--')
plt.title("Cumulative Gains Chart")
plt.xlabel("Proportion of Samples")
plt.ylabel("Proportion of Positives")
plt.legend()
plt.show()

print("All statistics and visualizations complete for Model 1.")
>>>>>>> 223af30 (Initial commit — Airlytics Streamlit app)
