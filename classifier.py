import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
#from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, f1_score, ConfusionMatrixDisplay
 
import warnings
warnings.filterwarnings('ignore')

# load csv data
data = pd.read_csv("smell_dataset.csv", index_col='index')
Y = data['class_label']
X = data.drop('class_label', axis=1)


#train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, stratify=Y, random_state=10)

# balance it by adding repetitive rows of minority class.
ros = RandomOverSampler(sampling_strategy='minority', random_state=20)
X_train, Y_train = ros.fit_resample(X_train, Y_train)

# Normalizing the features for stable and fast training.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = [LogisticRegression(), KNeighborsClassifier(n_neighbors=3), SVC(kernel='rbf', probability=True),
    DecisionTreeClassifier(criterion='entropy',random_state=7), GaussianNB(), RandomForestClassifier(random_state=1),
    SGDClassifier(penalty=None)] # XGBClassifier()

labels = ['air', 'apple', 'bag', 'banana', 'mango', 'orange', 'pear']
 
for i in range(len(classifier)):
      classifier[i].fit(X_train, Y_train)
      print(f'{classifier[i]} : ')
      train_preds = classifier[i].predict(X_train) 
      print("Training Accuracy : ", accuracy_score(Y_train, train_preds))
     
      test_preds = classifier[i].predict(X_test) 
      print("Validation Accuracy : ", accuracy_score(Y_test, test_preds))
      
      print("Training f1_score: ", f1_score(Y_train, train_preds, average=None))
      print("Validation f1_score: ", f1_score(Y_test, test_preds, average=None))
      
for i in range(len(classifier)):
    #metrics.plot_confusion_matrix(classifier[i], X_test, Y_test)
    test_preds = classifier[i].predict(X_test)
    cm = confusion_matrix(Y_test, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    #plt.title(f'confustion matrix{classifier[i]}')
    plt.title('confustion matrix')
    plt.savefig(f'results/{classifier[i]}.jpg')
    plt.show()
    
for i in range(len(classifier)):
    print(f'{classifier[i]} : ')
    print(metrics.classification_report(Y_test, classifier[i].predict(X_test)))