#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import data
dataset = pd.read_csv("smell_dataset.csv", index_col='index')

# Check for missing values
dataset.isnull().sum()

# value counts of fruits quality
print(dataset.class_label.value_counts())

y = dataset['class_label']
X = dataset.drop('class_label', axis=1)

# y is of int type. Change it to categorical
y = y.astype('object')

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#  one hot encoding
from keras.utils import np_utils
y = np_utils.to_categorical(y)

# Splitting the dataset 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)


# balance it by adding repetitive rows of minority class.
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(sampling_strategy='minority', random_state=20)
X_train, y_train = ros.fit_resample(X_train, y_train)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Import Keras libraries 
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow._api.v2.compat.v1 as tf


# ANN
model = Sequential()
model.add(Dense(units = 512, kernel_initializer = 'uniform', activation = 'relu', input_dim = (X_train.shape[1]))) # No. of features count starting from 0 = 46
model.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))
#model.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'softmax')) # no. of class = 7
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) # categorical_crossentropy, binary_crossentropy, adam, sgd
model.summary()

logger=tf.keras.callbacks.CSVLogger('results/logkeras.csv', separator=",", append=True)
cl = model.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size = 10, epochs = 500, callbacks=[logger]) # Lesser no of epochs - Basic Model


fig, ax = plt.subplots(figsize=(17,8))
plt.title('Accuracy')
plt.plot(cl.history['accuracy'], label='accuracy')
plt.plot(cl.history['val_accuracy'], label='val_accuracy', linestyle='--')
plt.legend()

fig, ax = plt.subplots(figsize=(17,8))
plt.title('Loss')
plt.plot(cl.history['loss'], label='loss')
plt.plot(cl.history['val_loss'], label='val_loss', linestyle='--')
plt.legend()
plt.show()

ModelLoss, ModelAccuracy = model.evaluate(X_test, y_test)

print(f'Test Loss is {ModelLoss}')
print(f'Test Accuracy is {ModelAccuracy}')


y_pred = model.predict(X_test)
y_test_list=list(y_test)
total=len(y_test_list)
correct=0


for i in range(total):
  if(np.argmax(y_pred[i])==y_test_list[i]).any():
    correct+=1

print(f'{correct}/{total}')
print(correct/total)


# save
model.save('results/model.h5')
print('model Saved!')
 
# load model
from tensorflow.keras.models import load_model
savedModel=load_model('results/model.h5')
savedModel.summary()

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
predict=model.predict(X_test)


y_pred=[]
for i in range(len(predict)):
    y_pred.append(np.argmax(predict[i]))

cr = classification_report(y_test.argmax(axis=1), y_pred)
print(cr)


p_test = model.predict(X_test).argmax(axis=1)
cm = confusion_matrix(y_test.argmax(axis=1), p_test)
print(cm)

labels = ['air', 'apple', 'bag', 'banana', 'mango', 'orange', 'pear']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()

import seaborn as sns
plt.title('Confustion matrix: ANN')
plt.savefig('results/ANN.jpg')
plt.show()