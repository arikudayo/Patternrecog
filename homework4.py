import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize

iris = load_iris()
X = iris.data
y = iris.target

y_bin = label_binarize(y, classes=np.unique(y))
n_classes = y_bin.shape[1]

lda = LinearDiscriminantAnalysis()
log_reg = LogisticRegression(max_iter=1000, solver='liblinear')
nb = GaussianNB()
lda.fit(X, y)
log_reg.fit(X, y)
nb.fit(X, y)

y_lda_pred = lda.predict(X)
y_log_reg_pred = log_reg.predict(X)
y_nb_pred = nb.predict(X)

y_lda_pred_prob = lda.predict_proba(X)
y_log_reg_pred_prob = log_reg.predict_proba(X)
y_nb_pred_prob = nb.predict_proba(X)

def calculate_roc_auc_class1(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true[:, 1], y_pred_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

fpr_lda, tpr_lda, roc_auc_lda = calculate_roc_auc_class1(y_bin, y_lda_pred_prob)
fpr_log_reg, tpr_log_reg, roc_auc_log_reg = calculate_roc_auc_class1(y_bin, y_log_reg_pred_prob)
fpr_nb, tpr_nb, roc_auc_nb = calculate_roc_auc_class1(y_bin, y_nb_pred_prob)

f1_lda = f1_score(y, y_lda_pred, average='weighted')
f1_log_reg = f1_score(y, y_log_reg_pred, average='weighted')
f1_nb = f1_score(y, y_nb_pred, average='weighted')

accuracy_lda = accuracy_score(y, y_lda_pred)
accuracy_log_reg = accuracy_score(y, y_log_reg_pred)
accuracy_nb = accuracy_score(y, y_nb_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lda, tpr_lda, label=f'LDA (AUC = {roc_auc_lda:.2f})')
plt.plot(fpr_log_reg, tpr_log_reg, label=f'Logistic Regression (AUC = {roc_auc_log_reg:.2f})')
plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC = {roc_auc_nb:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Class 1 on Iris Dataset')
plt.legend(loc="lower right")
plt.show()

print(f"LDA - F1 Score: {f1_lda:.2f}, Accuracy: {accuracy_lda:.2f}")
print(f"Logistic Regression - F1 Score: {f1_log_reg:.2f}, Accuracy: {accuracy_log_reg:.2f}")
print(f"Naive Bayes - F1 Score: {f1_nb:.2f}, Accuracy: {accuracy_nb:.2f}")