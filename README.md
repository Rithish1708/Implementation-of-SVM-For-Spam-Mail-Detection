# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries (pandas, chardet, sklearn, etc.).
2. Detect the encoding of the CSV file using chardet.
3. Read the CSV file with the correct encoding.
4. Check the data for structure and missing values.
5. Split the data into input (x = messages) and output (y = labels).   
6. Divide the data into training and testing sets.
7. Divide the data into training and testing sets.
8. Train an SVM model, make predictions, and calculate accuracy.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: RITHISH R
RegisterNumber:  212224040278
*/
```

```
import chardet
file='spam.csv'
with open(file,'rb')as rawdata:
    result=chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')
data.head()

data.info()

data.isnull().sum()

x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
x_train
x_test

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy 
*/
```

## Output:


## Result output

<img width="921" height="34" alt="image" src="https://github.com/user-attachments/assets/55fe477a-e16f-4251-8be6-1fc35486f5ad" />

## data.head()

<img width="1027" height="312" alt="image" src="https://github.com/user-attachments/assets/7e9a2e89-da4e-493d-90e8-98bd51cc11c2" />

## data.info()

<img width="495" height="326" alt="image" src="https://github.com/user-attachments/assets/196cb454-5223-4ffe-970c-24aa18183f96" />

## data.isnull().sum()

<img width="243" height="172" alt="image" src="https://github.com/user-attachments/assets/298d2b97-686b-4cd2-b868-dac9a76a1995" />

## x_train and y_train
<img width="924" height="216" alt="image" src="https://github.com/user-attachments/assets/19f2838a-fadd-4519-8ff6-1fcb5a4d4966" />


## y_pred

<img width="799" height="48" alt="image" src="https://github.com/user-attachments/assets/31441962-9f3e-4662-989b-632312b69b09" />

## accuracy()

<img width="299" height="59" alt="image" src="https://github.com/user-attachments/assets/53be1a0a-8ada-48bf-8b73-6b7821a25c5c" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
