#%% Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from collections import Counter

#%% Read the dataset

data = pd.read_csv("healthcare-dataset-stroke-data.csv")


#%% EDA (Exploratory Data Analysis)

data.shape # (5110, 12)
data.info()


"""
Variable Description
1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12) stroke: 1 if the patient had a stroke or 0 if not
*Note: "Unknown" in smoking_status means that the information is unavailable for this patient

"""


# In this dataset , stroke column is target, I'm going to change the column name as a Target

data.rename({"stroke":"Target"},axis = 1,inplace = True)
data = data.drop(["id"],axis = 1)

data.describe()


#%% Missing Values

data.columns[data.isnull().any()]
data.isnull().sum() # bmi column have 201 missing values
#%% Filling Values
data["bmi"] = data["bmi"].fillna(data["bmi"].mean()) # Here I filled average of the bmi column  instance of missing values (Nan)

#%% Categorical Variables = gender,ever_married,work_type,Residence_type,smoking_status,Target,hypertension

def bar_plot(variable):
    
    # get value
    var = data[variable]
    
    # count number of the values
    varValue = var.value_counts()
    
    # visualize
    plt.figure(figsize = (10,10))
    plt.bar(varValue.index,varValue)
    plt.xticks(varValue.index,varValue.index.values)
    plt.ylabel("Frequency")
    plt.title(variable)
    plt.show()

categorical_variables = ["gender","ever_married","work_type","Residence_type","smoking_status","Target","hypertension"]
for Q in categorical_variables:
    bar_plot(Q)
    
    
#%% Numerical Variables = age,avg_glucose_level,bmi

def plot_hist(numerical_variable):
        plt.figure(figsize = (10,10))
        plt.hist(data[numerical_variable],bins = 50)
        plt.xlabel(numerical_variable)
        plt.ylabel("Frequency")
        plt.title("{} distribution with hist".format(numerical_variable))
        plt.show()
numerical_variables = ["age","avg_glucose_level","bmi"]
for T in numerical_variables:
    plot_hist(T)

#%% Correlation Matrix

f,ax = plt.subplots(figsize = (10,10))
sns.heatmap(data.corr(),annot = True,linewidths = 5,fmt = ".2f",ax = ax)
plt.show()

#%% Label Encoder Operation (We are going to convert categorical to numerical)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

columnsToEncode = list(data.select_dtypes(include = ["category","object"]))
for feature in columnsToEncode:
    data[feature] = le.fit_transform(data[feature])

#%% Get X and Y Coordinates

y = data.Target.values
x_data = data.drop(["Target"],axis = 1)


# Normalization Opeariton
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))
#%% Train-Test Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#%% Random Forest Classification

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000,random_state=1) # n_estimators , which means number of trees
rf.fit(x_train,y_train)

print("Accuracy of the Random Forest Classification:  ",rf.score(x_test,y_test))

"""
Accuracy of the Random Forest Classification:   0.9393346379647749
"""

#%% K-Nearst Neighbor Classification

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train,y_train)
print("Accuracy of the KNN for k=3: {}".format(knn.score(x_test,y_test)))

"""
Accuracy of the KNN for k=3: 0.9305283757338552
"""

# Let's find best k value (n_neighbors)

score_list = []

for each in range(1,100):
    knn2 = KNeighborsClassifier(n_neighbors = each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
plt.plot(range(1,100),score_list)
plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("K value vs Accuracy")
plt.show()

knn3 = KNeighborsClassifier(n_neighbors = 2)
knn3.fit(x_train,y_train)
print("Accuracy of the KNN for k = 2 : {}".format(knn3.score(x_test,y_test)))

"""
Accuracy of the KNN for k = 2 : 0.9403131115459883
"""
#%% K-Fold Cross Validation
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = knn, X = x_train,y = y_train,cv = 10)

print("average accuracy = ",np.mean(accuracies))
print("average std = ",np.std(accuracies))

"""
average accuracy =  0.9469173977659524
average std =  0.004387961032384926
"""


#%% Confusion Matrix
y_pred = knn3.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)

# Confusion Matrix Visualize
f,ax = plt.subplots(figsize = (10,10))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax = ax)
plt.ylabel("y_pred")
plt.xlabel("y_true")
plt.title("Confusion Matrix")
plt.show()





