import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

#load the dataset
diabetes = pd.read_csv('diabetes.csv')
feature_names=diabetes.columns
X = diabetes[feature_names]
y = diabetes.Outcome




# Splitting features and target datasets into: train and test

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)



# Training a Linear Regression model with fit()

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train, y_train)



# Predicting the results for our test dataset

predicted_values = lr.predict(X_test)



# Printing the residuals: difference between real and predicted

for (real, predicted) in list(zip(y_test, predicted_values)):

    print(f'Value: {real}, pred: {predicted} {"is different" if real != predicted else ""}')



# Printing accuracy score(mean accuracy) from 0 - 1

print(f'Accuracy score is {lr.score(X_test, y_test):.2f}/1 \n')



# Printing the classification report

from sklearn.metrics import classification_report, confusion_matrix, f1_score

print('Classification Report')

print(classification_report(y_test, predicted_values))



# Printing the classification confusion matrix (diagonal is true)

print('Confusion Matrix')

print(confusion_matrix(y_test, predicted_values))



print('Overall f1-score')

print(f1_score(y_test, predicted_values, average="macro"))



# calculating ROC and drawing the ROC figure





logit_roc_auc = roc_auc_score(y_test, lr.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')
