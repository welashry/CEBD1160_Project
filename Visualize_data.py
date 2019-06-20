import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

diabetes = pd.read_csv('diabetes.csv')
print(diabetes.columns)

plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.
p=sns.heatmap(diabetes.corr(), annot=True,cmap ='RdYlGn')

plt.savefig(f'plots/1/Heatmap.png')


p = diabetes.hist(figsize = (20,20))
plt.savefig(f'plots/1/Hist.png')

from pandas.plotting import scatter_matrix
p=scatter_matrix(diabetes,figsize=(25, 25))
plt.savefig(f'plots/1/pairplot.png')
p=scatter_matrix(diabetes,figsize=(25, 25))

