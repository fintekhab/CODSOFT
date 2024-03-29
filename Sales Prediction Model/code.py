import pandas as pd
import matplotlib.pyplot as plt
import sklearn 
import numpy as np
import seaborn as sns

df = pd.read_csv(r"C:\Users\Fatima\Documents\Sales Prediction Model\advertising.csv")#importing file 
print(df.head())#viewing first 5 elements
print(df.tail())#viewing last 5 elements

print("\n We will first remove unnecessary values from data")
df_cleaned=df.dropna()#removing empty rows to proceed

print("\n")
print("The Statistial Measures of Cleaned Data are: \n", df_cleaned.describe())

print("\n"+"The minimum spent on TV is {} and the maximum earnt in sales is {}".format(min(df_cleaned['TV']), max(df_cleaned['Sales'])))

fig, axs = plt.subplots(1, 3, figsize = (10,5))
axs[0].bar(df_cleaned.index, df_cleaned['Sales'], color = 'yellow', label = 'Sales')
axs[0].bar(df_cleaned.index, df_cleaned['TV'], color = 'r', label = 'TV')
axs[0].set_title('Sales vs TV')
axs[0].set_xlabel('Observation')
axs[0].set_ylabel('Amount')
axs[0].legend()

axs[1].bar(df_cleaned.index,df_cleaned['Sales'], color = 'yellow', label ='Sales')
axs[1].bar(df_cleaned.index, df_cleaned['Radio'], color = 'maroon', label= 'Radio')
axs[1].set_title('Sales vs Radio')
axs[1].set_xlabel('Observation')
axs[1].set_ylabel('Amount')
axs[1].legend()

axs[2].bar(df_cleaned.index, df_cleaned['Sales'], color = 'yellow', label = 'Sales')
axs[2].bar(df_cleaned.index, df_cleaned['Newspaper'], color = 'b', label = 'Newspaper')
axs[2].set_title('Sales vs Newspaper')
axs[2].set_xlabel('Observation')
axs[2].set_ylabel('Amount')
axs[2].legend()

plt.show()

x = df_cleaned['TV'].values.reshape(-1,1)
y = df_cleaned['Sales']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print("\n Mean square error:", mse)

plt.figure(figsize=(10, 5))
plt.scatter(df_cleaned['TV'], df_cleaned['Sales'], color='b')
plt.plot(x_test, y_pred, color='r')
plt.title('Sales vs TV (Linear Regression)')
plt.xlabel('TV Advertising Spending')
plt.ylabel('Sales')
plt.show()

