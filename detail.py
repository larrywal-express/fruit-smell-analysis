import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import json

import warnings
warnings.filterwarnings('ignore')

# normalize data load
data = pd.read_csv("smell_dataset.csv", index_col='index', dtype = str)
class_data = data['class_label']

f = open('smell_dataset_metadata.json')
df = json.load(f)

# feature scalling
scaler = preprocessing.MinMaxScaler()
data = data.drop('class_label', axis=1)
fruits = scaler.fit_transform(data)


headers = list(data.columns)
print(headers)
for i in headers:
    print(i)                       # copy the header list to columns 
    #columns = list(i)
    columns=['ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10', 'ch11', 'ch12', 'ch13',
    'ch14', 'ch22', 'ch23', 'ch24', 'ch25', 'ch26', 'ch27', 'ch28', 'ch29', 'ch30', 'ch35',
    'ch36', 'ch37', 'ch38', 'ch39', 'ch40', 'ch41', 'ch42', 'ch43', 'ch44', 'ch45', 'ch46',
    'ch51', 'ch52', 'ch53', 'ch54', 'ch55', 'ch56', 'ch57', 'ch58', 'ch59', 'ch60', 'ch61',
    'ch62', 'humidity', 'temperature']
    fruits = pd.DataFrame(fruits, columns=columns)
    
    # add the class_label again
    fruits['class_label'] = class_data.replace(df)
    print(fruits)


# saving the dataframe
#fruits.to_csv('results/smell_dataset_normalize.csv')

print(fruits.head())
print(fruits.info())
print(fruits.describe())
print(fruits.describe().T)
print(fruits.corr()['humidity']) # Y=class_label, humidity
print(fruits.isnull().sum())

# histogram to visualise the distribution of the fruits
for col in fruits.columns:
  if fruits[col].isnull().sum() > 0:
    fruits[col] = fruits[col].fillna(fruits[col].mean())
fruits.isnull().sum().sum()
fruits.hist(bins=20, figsize=(10, 10))
plt.subplots_adjust(left=0.035,right=0.990,bottom=0.055,top=0.960,wspace=0.375,hspace=0.910)
plt.show()


# Heatmap of correlation matrix
plt.figure(figsize=(10,10))
sns.heatmap(fruits.corr(),annot=True,linewidth=0.03,center=0,cmap='viridis') # ocean, coolwarm, Oranges, Blues, rainbow
plt.show()


# distributions
plt.hist(fruits['class_label'], bins=6, alpha=0.5, histtype='bar', ec='black') # Y= fruits.class_label
plt.title('Distribution of fruit')
plt.xlabel('class_label')
plt.ylabel('count')
plt.show()


# Boxplot of fruits and humidity
ax = sns.boxplot(x='class_label', y='humidity', data=fruits, palette='GnBu_d') # palette= GnBu_d or coolwarm, Y = 'class_label'
plt.title("Boxplot of humidity")
plt.show()

ax = sns.boxplot(x='class_label',y='ch36',data=fruits, palette='GnBu_d') # palette= GnBu_d or coolwarm, Y = 'class_label'
plt.title("Boxplot of ch36")
plt.show()

# ## Distplot:
sns.distplot(fruits['humidity'])
fruits.plot(kind ='box',subplots = True, layout =(7,7),sharex = False)
fruits.plot(kind ='density',subplots = True, layout =(7,7),sharex = False)
plt.show()

sns.distplot(fruits['ch36'])
plt.show()
