
# cebd1160_project_template
Instructions and template for final projects.

| Name | Date |
|:-------|:---------------|
|Wagdy Elashry | June 19,2019 |

-----

### Resources
Your repository should include the following:

- Python script for your analysis
- Results figure/saved file
- Dockerfile for your experiment
- runtime-instructions in a file named RUNME.md

-----

## Research Question
How can we use machine learning to prdict and classify the diabetes?.


### Abstract

4 sentence longer explanation about your research question. Include:

- opportunity (diabetes data set )
- challenge ( our challenge is to  predict diabetes)
- action (we will solve this problem by using a machine learning model with a pproporiate algorithm "DT Classifier")
- resolution (using "DT Classifier" and "Logistic Regression")

### Introduction
Diabetes dataset which is a Ten baseline variables + new veriable"Outcome", age, sex, body mass index, average blood pressure
and six blood serum measurements were obtained for each of n = 442 diabetes patients, as well as the response of interest, a quantitative measure of disease progression one year after baseline, and here we will create a machine learning model based on the dataset we have in order to help us to predict a diabetes.

### Methods
based on the BMI and its levels, i have created a new feature called Outcome and fill it up with approporiate value(0"means the patient doesnt have diabetes" or 1"means the patient has a diabetes" and use it in order to help me predict the diabetes 
 and here is a Brief Analysis of the diabetes data
![alt text](https://raw.githubusercontent.com/welashry/CEBD1160_Project/master/Plots/pairplot.png)

 then using heat map to find the correlation between the features
 ![alt text](https://raw.githubusercontent.com/welashry/CEBD1160_Project/master/Plots/Heatmap.png)
 
 
 
 to solve the problem i have used KNeighborsClassifier, LogisticRegression, DecisionTreeClassifier,GaussianNB...etc and i also compaired these models in order to know the accuracy of machine learning algorithms and therefor choose the accurate ones.
  
  ![alt text](https://raw.githubusercontent.com/welashry/CEBD1160_Project/master/Plots/model_scores.png)
### Results

 The most important features and the accuracy before and after feature selection
  ![alt text](https://raw.githubusercontent.com/welashry/CEBD1160_Project/master/Plots/features_bef_after.PNG)
  here is the logistec regression
![alt text](https://raw.githubusercontent.com/welashry/CEBD1160_Project/master/Plots/logreg.png)

the classification
![alt text](https://raw.githubusercontent.com/welashry/CEBD1160_Project/master/Plots/Log_ROC.png)
![alt text](https://raw.githubusercontent.com/welashry/CEBD1160_Project/master/Plots/classification.PNG)


### Discussion
As a result of the figures above and the accuracy of the models i have usedto classify the diabetes and predict it.
And yes i can say that we have probably been able to solve it, i have also tried lasso model but i wasn't sure about the results if it solved the problem or not.

### References
All of the links
1-https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html
2-https://towardsdatascience.com/machine-learning-for-diabetes-562dd7df4d42

-------
