import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# humidity
fruits = pd.read_csv("smell_dataset.csv", index_col='index')
fruits = fruits.drop('class_label', axis=1)
fruits = fruits.drop('temperature', axis=1)

# temperature
fruitsT = pd.read_csv("smell_dataset.csv", index_col='index')
fruitsT = fruitsT.drop('class_label', axis=1)
fruitsT = fruitsT.drop('humidity', axis=1)

fig, ax = plt.subplots(1,2, figsize=(18, 8))
corr1 = fruits.corr('pearson')[['humidity']].sort_values(by='humidity', ascending=False)
corr4 = fruitsT.corr('pearson')[['temperature']].sort_values(by='temperature', ascending=False)


sns.heatmap(corr1, ax=ax[0], annot=True).set_title('Pearson: Correlation with humidity')
sns.heatmap(corr4, ax=ax[1], annot=True).set_title('Pearson: Correlation with temperature')

plt.savefig('results/corrHT.jpg')

plt.show()