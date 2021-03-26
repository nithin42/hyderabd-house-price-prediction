import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

data = pd.read_csv("hyd_cost.csv")
data.head()

data = data.drop(['Unnamed: 0','Locality'], axis=1)

data.head()

X = data.drop(['Price'], axis=1)
y = data['Price'].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
multiple_reg = LinearRegression()
multiple_reg.fit(X_train, y_train)

y_pred = multiple_reg.predict(X_test)

#multiple_reg.coef_

#multiple_reg.intercept_

#Calculating the R squared value
#from sklearn.metrics import r2_score
#r2_score(y_test, y_pred)

#print("Enter the details of the house:")
#Bedroom = float(input("Bedroom : "))
#Bathroom = float(input("Bathroom : "))
#Area = float(input("Area : "))


#predicting the sales with respect to the inputs
#output = multiple_reg.predict([[Bedroom,Bathroom,Area]])

#print("you will get Rs {:.2f}"\
#      .format(output[0][0]))


if not os.path.exists('models'):
    os.makedirs('models')
    
MODEL_PATH = "models/multiple_reg.sav"
pickle.dump(multiple_reg, open(MODEL_PATH, 'wb'))

model=pickle.load(open('model.pkl','rb'))

