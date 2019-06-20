from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))
#load the dataset
diabetes = pd.read_csv('diabetes.csv')
feature_names=diabetes.columns
X = diabetes[feature_names]
y = diabetes.Outcome

#the accuracy of machine learning algorithms
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = diabetes.Outcome, random_state=0)
names = []
scores = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)
tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)

axis = sns.barplot(x='Name', y='Score', data=tr_split)
axis.set(xlabel='Classifier', ylabel='Accuracy')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width() / 2, height + 0.005, '{:1.4f}'.format(height), ha="center")

plt.savefig(f'plots/1/model_scores.png')

## a list of accuracy scores for each of the features selected
from sklearn.feature_selection import RFECV
logreg_model = LogisticRegression()
rfecv = RFECV(estimator=logreg_model, step=1, scoring='accuracy')
rfecv.fit(X, y)
plt.figure()
plt.title('Logistic Regression CV score vs No of Features')
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.savefig(f'plots/1/feature_impor.png')

#the most important features
feature_importance = list(zip(feature_names, rfecv.support_))
new_features = []
for key, value in enumerate(feature_importance):
    if (value[1]) == True:
        new_features.append(value[0])

print(new_features)

# compare the accuracy before and after feature selection.
X_new = diabetes[new_features]
initial_score = cross_val_score(logreg_model, X, y,  scoring='accuracy').mean()
print("Initial accuracy : {} ".format(initial_score))
fe_score = cross_val_score(logreg_model, X_new, y,  scoring='accuracy').mean()
print("Accuracy after Feature Selection : {} ".format(fe_score))
