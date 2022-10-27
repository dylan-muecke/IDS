### This Code can also be found in the Google Colab File found Below ###
### https://colab.research.google.com/drive/1gQ47iWN1MPSZpJlr5jWGfDrCSzaazNg-?usp=sharing ###


from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, auc, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sn

###insert code to import data from drive here. You should created two dataframes: df1 and df2 ###
tf1 = pd.read_csv('/content/drive/My Drive/FinalProject315/IoT-23_Normal.csv', iterator=True, low_memory=False)
df1 = pd.concat(tf1, ignore_index=True)
tf2 = pd.read_csv('/content/drive/My Drive/FinalProject315/IoT-23_C&C.csv', iterator=True, low_memory=False)
df2 = pd.concat(tf2, ignore_index=True)

data = pd.DataFrame()
data = df1.append(df2)

data['Label'].value_counts().plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend(bbox_to_anchor=(1, 1))
print(data['Label'].value_counts())
print(data['Label'].value_counts().sum())

df1 = df1.sample(23981)

data = pd.DataFrame()
data = df1.append(df2)

data['Label'].value_counts().plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend(bbox_to_anchor=(1, 1))
print(data['Label'].value_counts())
print(data['Label'].value_counts().sum())
print(data.shape) #prints out the dimensions of the DataFrame (rows, columns)

data.drop(['Flow_ID','Src_IP','Dst_IP','Timestamp','Sub_Cat'],axis=1,inplace=True)
#.drop function reference: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
print(data.shape)

nancheck = data.isnull().values.any() #only checks for NaN values
print(nancheck)

#to be safe, always run
data = data[~data.isin([np.nan,np.inf,-np.inf]).any(1)] 
print(data['Label'].value_counts())

#make y_data 
attack_type = data['Label'].unique()
attack_type = attack_type.tolist()
y_data = data['Label'].apply(attack_type.index)
y_data = y_data.drop(columns='index')

#print to make sure the labels were converted into integers
print(y_data.value_counts())

#remove 'Label' from x_data using the .drop() function 
data.drop(['Label'],axis=1,inplace=True)

#train/test split reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(data, y_data, test_size=23981)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model_dummy = DummyClassifier(strategy="uniform", random_state=1)
model_dummy.fit(X_train, y_train)

y_predict_dummy = model_dummy.predict(X_test)
print('Baseline Accuracy Score: ', accuracy_score(y_test, y_predict_dummy))
print('Baseline Precision Score: ', precision_score(y_test, y_predict_dummy))

#make and train model
clf = tree.DecisionTreeClassifier(random_state=0) ###Insert code for parameters here###
clf = clf.fit(X_train, y_train)

#print accuracy and precision score
y_pred = clf.predict(X_test)
print("Test Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Test Precision Score: ", precision_score(y_test, y_pred))
#print classification report
CR=classification_report(y_test, y_pred)
print("Classification Report: ")
print(CR)

#print confusion matrix
cm=confusion_matrix(y_test,y_pred)
df_cm=pd.DataFrame(cm,columns=np.unique(y_test),index=np.unique(y_test))
df_cm.index.name='Actual'
df_cm.columns.name='Predicted'
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm,cmap='Blues',annot=True,annot_kws={'size':16})

###make and train model###
clf2 = RandomForestClassifier(random_state=0) ###Insert code for parameters here###
clf2 = clf2.fit(X_train, y_train)

#print accuracy and precision score
y_pred = clf2.predict(X_test)
print("Test Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Test Precision Score: ", precision_score(y_test, y_pred))

#print classification report
CR=classification_report(y_test, y_pred)
print("Classification Report: ")
print(CR)

#print confusion matrix
#import seaborn as sn
cm=confusion_matrix(y_test,y_pred)
df_cm=pd.DataFrame(cm,columns=np.unique(y_test),index=np.unique(y_test))
df_cm.index.name='Actual'
df_cm.columns.name='Predicted'
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm,cmap='Blues',annot=True,annot_kws={'size':16})

#Torri Attack#

tf1 = pd.read_csv('/content/drive/My Drive/FinalProject315/IoT-23_Normal.csv', iterator=True, low_memory=False)
df1 = pd.concat(tf1, ignore_index=True)
tf2 = pd.read_csv('/content/drive/My Drive/FinalProject315/IoT-23_Torii.csv', iterator=True, low_memory=False)
df2 = pd.concat(tf2, ignore_index=True)

data = pd.DataFrame()
data = df1.append(df2)

data['Label'].value_counts().plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend(bbox_to_anchor=(1, 1))
print(data['Label'].value_counts())
print(data['Label'].value_counts().sum())

df1 = df1.sample(33858)
data = pd.DataFrame()
data = df1.append(df2)

data['Label'].value_counts().plot(kind = 'pie', autopct='%1.1f%%', figsize=(8, 8)).legend(bbox_to_anchor=(1, 1))
print(data['Label'].value_counts())
print(data['Label'].value_counts().sum())
print(data.shape) #prints out the dimensions of the DataFrame (rows, columns)

data.drop(['Flow_ID','Src_IP','Dst_IP','Timestamp','Sub_Cat'],axis=1,inplace=True)
print(data.shape)

nancheck = data.isnull().values.any() #only checks for NaN values
print(nancheck)

data = data[~data.isin([np.nan,np.inf,-np.inf]).any(1)] 
print(data['Label'].value_counts())

#make y_data 
attack_type = data['Label'].unique()
attack_type = attack_type.tolist()
y_data = data['Label'].apply(attack_type.index)
y_data = y_data.drop(columns='index')

#print to make sure the labels were converted into integers
print(y_data.value_counts())

#remove 'Label' from x_data using the .drop() function 
data.drop(['Label'],axis=1,inplace=True)

#train/test split reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(data, y_data, test_size=33858)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model_dummy = DummyClassifier(strategy="uniform", random_state=1)
model_dummy.fit(X_train, y_train)
y_predict_dummy = model_dummy.predict(X_test)
print('Baseline Accuracy Score: ', accuracy_score(y_test, y_predict_dummy))
print('Baseline Precision Score: ', precision_score(y_test, y_predict_dummy))

#make and train model
clf = tree.DecisionTreeClassifier(random_state=0) ###Insert code for parameters here###
clf = clf.fit(X_train, y_train)

#print accuracy and precision score
y_pred = clf.predict(X_test)
print("Test Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Test Precision Score: ", precision_score(y_test, y_pred))

#print classification report
CR=classification_report(y_test, y_pred)
print("Classification Report: ")
print(CR)

#print confusion matrix
cm=confusion_matrix(y_test,y_pred)
df_cm=pd.DataFrame(cm,columns=np.unique(y_test),index=np.unique(y_test))
df_cm.index.name='Actual'
df_cm.columns.name='Predicted'
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm,cmap='Blues',annot=True,annot_kws={'size':16})

###make and train model###
clf2 = RandomForestClassifier(random_state=0) ###Insert code for parameters here###
clf2 = clf2.fit(X_train, y_train)

#print accuracy and precision score
y_pred = clf2.predict(X_test)
print("Test Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Test Precision Score: ", precision_score(y_test, y_pred))

#print classification report
CR=classification_report(y_test, y_pred)
print("Classification Report: ")
print(CR)

#print confusion matrix
#import seaborn as sn
cm=confusion_matrix(y_test,y_pred)
df_cm=pd.DataFrame(cm,columns=np.unique(y_test),index=np.unique(y_test))
df_cm.index.name='Actual'
df_cm.columns.name='Predicted'
plt.figure(figsize=(10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm,cmap='Blues',annot=True,annot_kws={'size':16})

