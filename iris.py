import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, 0]  # Sepal length
y = (iris.target == 2).astype(int)  # Classify whether species is 'virginica'

# Define a threshold for classification
threshold = 4.5  # Example threshold
y_pred = (X > threshold).astype(int)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Accuracy Calculation
accuracy = accuracy_score(y_test, (X_test > threshold).astype(int))
print(f'Accuracy: {accuracy:.2f}')
# Compute predicted probabilities (in this case, simply the feature value)
probs = X_test

# Vary the threshold
thresholds = np.linspace(min(X), max(X), 100)
tpr = []
fpr = []

# Compute TPR and FPR for each threshold
for t in thresholds:
    y_pred = (probs > t).astype(int)
    tp = np.sum((y_pred == 1) & (y_test == 1))
    fn = np.sum((y_pred == 0) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    tn = np.sum((y_pred == 0) & (y_test == 0))
    
    tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC Curve)')
plt.legend(loc='lower right')
plt.grid()
plt.show()
