import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fruits = pd.read_csv("smell_dataset.csv", index_col='index')
fruits.head()
print(fruits.head())
fruits = fruits.drop('class_label', axis=1)
fruits = fruits.drop('temperature', axis=1)

fig, ax = plt.subplots(1,3, figsize=(18, 8))

corr1 = fruits.corr('pearson')[['humidity']].sort_values(by='humidity', ascending=False)
corr2 = fruits.corr('spearman')[['humidity']].sort_values(by='humidity', ascending=False)
corr3 = fruits.corr('kendall')[['humidity']].sort_values(by='humidity', ascending=False)


sns.heatmap(corr1, ax=ax[0], annot=True).set_title('Correlation: pearson')
sns.heatmap(corr2, ax=ax[1], annot=True).set_title('Correlation: spearman')
sns.heatmap(corr3, ax=ax[2], annot=True).set_title('Correlation: kendall')
plt.savefig('results/corr.jpg')

plt.show()