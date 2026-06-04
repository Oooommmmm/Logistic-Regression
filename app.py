import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, classification_report

df = pd.read_csv('data.csv')
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("Standard Threshold Metrics")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

custom_threshold = 0.25
y_pred_custom = (y_prob >= custom_threshold).astype(int)

cm_c = confusion_matrix(y_test, y_pred_custom)
precision_c = precision_score(y_test, y_pred_custom)
recall_c = recall_score(y_test, y_pred_custom)

print(f"Custom Tuned Threshold ({custom_threshold}) Metrics")
print(f"Precision: {precision_c:.4f}")
print(f"Recall: {recall_c:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
axes[0].set_title('Confusion Matrix (Threshold = 0.5)', fontsize=12)
axes[0].set_xlabel('Predicted Diagnostic Class')
axes[0].set_ylabel('True Diagnostic Class')

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate ')
axes[1].set_ylabel('True Positive Rate ')
axes[1].set_title('Receiver Operating Characteristic (ROC)', fontsize=12)
axes[1].legend(loc="lower right")
axes[1].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('logistic_regression_evaluation.png')

z = np.linspace(-7, 7, 200)
sigmoid_values = 1 / (1 + np.exp(-z))

fig_sig, ax_sig = plt.subplots(figsize=(7, 4.5))
ax_sig.plot(z, sigmoid_values, color='darkblue', lw=2.5, label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$')
ax_sig.axhline(0.5, color='red', linestyle='--', lw=1, label='Standard Threshold = 0.5')
ax_sig.axhline(0.25, color='green', linestyle=':', lw=1, label='Custom Threshold = 0.25')
ax_sig.axvline(0, color='gray', linestyle='--', lw=1)
ax_sig.set_title('The Sigmoid Function Transformation', fontsize=13)
ax_sig.set_xlabel('Linear Input / Log-Odds ($z$)')
ax_sig.set_ylabel('Mapped Output Probability ($P$)')
ax_sig.legend(loc='upper left')
ax_sig.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sigmoid_function.png')