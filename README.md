# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries for data handling, visualization, and model building.
2. Load the dataset and inspect the first and last few records to understand the data structure.
3. Prepare the data by separating the independent variable (hours studied) and the dependent variable (marks scored).
4. Split the dataset into training and testing sets to evaluate the model's performance.
5. Initialize and train a linear regression model using the training data.
6. Predict the marks for the test set using the trained model.
7. Evaluate the model by comparing the predicted marks with the actual marks from the test set.
8. Visualize the results for both the training and test sets by plotting the actual data points and the regression line
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: K SARANYA 
RegisterNumber: 212224040298  
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import  mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```                           
## Output:
![simple linear regression model for predicting the marks scored](sam.png)
<img width="281" height="361" alt="image" src="https://github.com/user-attachments/assets/b0546100-ec2f-420a-a5ab-2cee3145fa94" />
```
dataset.info()
```
<img width="484" height="252" alt="image" src="https://github.com/user-attachments/assets/f70977c5-c313-4a75-a05f-1b15749c65ec" />

```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
```
<img width="884" height="727" alt="image" src="https://github.com/user-attachments/assets/6e0416b8-16dc-498b-a783-6edf7cdf48ed" />


```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train.shape)
print(X_test.shape)
```

<img width="338" height="85" alt="image" src="https://github.com/user-attachments/assets/fb316afe-be50-4b7c-848a-493406e797b0" />

```
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
```
<img width="873" height="93" alt="image" src="https://github.com/user-attachments/assets/c0525c41-744d-4fdf-91c8-8d3d8adb8528" />

```
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,reg.predict(X_train),color="green")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
<img width="867" height="688" alt="image" src="https://github.com/user-attachments/assets/7490e7f4-d232-4d6c-a250-d64d7122ef0b" />


```
plt.scatter(X_test, Y_test,color="blue")
plt.plot(X_test, reg.predict(X_test), color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
<img width="868" height="695" alt="image" src="https://github.com/user-attachments/assets/ead5ffd7-0643-431d-b889-77a628ffc87d" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
