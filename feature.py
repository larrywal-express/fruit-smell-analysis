import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pickle 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
#from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import svm
from matplotlib.colors import ListedColormap
from sklearn import preprocessing
import json


import warnings
warnings.filterwarnings('ignore')

# normalize data load
data = pd.read_csv("smell_dataset.csv", index_col='index', dtype = str)
class_data = data['class_label']

f = open('smell_dataset_metadata.json')
df = json.load(f)

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
    fruits = pd.DataFrame(data, columns=columns)
    
    # add the class_label again
    fruits['class_label'] = class_data.replace(df)
    #print(fruits)

"""
fruits = pd.read_csv("smell_dataset.csv", index_col='index')
print(str(len(fruits)) + ' records')
print(fruits.head())
"""

X = fruits.loc[:,['humidity', 'ch36']]
y = fruits.loc[:, ['class_label']]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y, random_state=10)

print(len(X_train))
print('-'*30)
print(X_train.head())
print('-'*30)
print(len(y_train))
print('-'*30)
print(y_train.head())
print(len(X_test))
print('-'*30)
print(X_test.head())
print(len(y_test))
print('-'*30)
print(y_test.head())
print('-'*30)

list = y_test['class_label'].values.tolist()
print('air     :' + str(list.count(0)))
print('apple :' + str(list.count(1)))
print('bag  :' + str(list.count(2)))
print('banana  :' + str(list.count(3)))
print('mango  :' + str(list.count(4)))
print('orange  :' + str(list.count(5)))
print('pear  :' + str(list.count(6)))


# MinMaxScaler
scaler = MinMaxScaler()
X_trainSc = scaler.fit_transform(X_train)
X_testSc = scaler.transform(X_test)


print(X_test)
print(X_testSc)

tmpdf = pd.DataFrame(X_trainSc)
print('-'*30)
print(tmpdf.describe())

# show histograms of the features
print('-'*30)
tmpdf.hist()
plt.show()


# Nearest Neighbors classification decision boundaries
N_NEIGHBORS = 2  #17

# Create color maps
#cmap_light = ListedColormap(["pink", "lightblue", "tan", "thistle", "white", "wheat", "gray"])
cmap_light = ListedColormap(["white", "white", "white", "white", "white", "white", "white"])
cmap_bold = ["cyan", "magenta", "red", "blue", "green", "k", "purple"]


# create K Neighbours Classifier and fit data.
classifier = [LogisticRegression(), KNeighborsClassifier(n_neighbors=3), SVC(kernel='rbf', probability=True),
    DecisionTreeClassifier(criterion='entropy',random_state=7), GaussianNB(), RandomForestClassifier(random_state=1),
    SGDClassifier(penalty=None)] #XGBClassifier(), 

for i in range(len(classifier)):
    # plot boundaires
    _, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        classifier[i].fit(X_trainSc, y_train['class_label'].values.tolist()),
        X_trainSc,
        cmap=cmap_light,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        xlabel=X.columns[0],
        ylabel=X.columns[1],
        shading="auto",
    )

    # Plot training points
    scatter = sns.scatterplot(
        x=X_trainSc[:, 0],
        y=X_trainSc[:, 1],
        hue=y_train['class_label'],
        palette=cmap_bold,
        alpha=1.0,
        edgecolor="black",
    )
    scatter.set_xlim(left=-0.1, right=1.1)
    scatter.set_ylim(bottom=-0.1, top=1.1);
        
    plt.title("Classification Boundary")
    #plt.title(f'classification boundary{classifier[i]}')
    plt.savefig(f'results/features/1/{classifier[i]}.jpg')
    plt.show()

    # Predict the test set results
    y_pred = classifier[i].predict(X_testSc)
    print(y_pred)

    labels = ['air', 'apple', 'bag', 'banana', 'mango', 'orange', 'pear']
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    
    plt.title('confustion matrix')
    #plt.title(f'confustion matrix{classifier[i]}')
    plt.savefig(f'results/features/2/{classifier[i]}.jpg')
    plt.show()
    print(f'{classifier[i]} : ')
    print('f1_score       : ' + str(f1_score(y_test, y_pred, average=None)))
    print('accuracy_score : ' + str(accuracy_score(y_test, y_pred)))