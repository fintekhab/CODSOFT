import pandas as pd
import sklearn
import matplotlib.pyplot as plt

print("\n  TITANIC SURVIVAL PREDICTION")

df = pd.read_csv(r"C:\Users\Fatima\Documents\Titanic Survival Model\Titanic-Dataset.csv") #Csv into a Pandas DataFrame

selected_cols = ['Name', 'Sex', 'Age', 'Fare', 'Survived']
subset_df = df[selected_cols] #selecting relevant columns from DataFrame

subset_df_cleaned = subset_df.dropna() #Cleaning by removing missing value rows
print("\n")
print("The given data set can be viewed in terms of 'Age', 'Sex' and 'Fare' \n")
print(subset_df_cleaned.head())      #viewing first 5 rows of this data

print("\n")
print("Statistical Measures of the Cleaned Data: \n")

print(subset_df_cleaned.describe())  #statitical measures

#Observing how Age, Sex and Fare varied among passengers by plotting pie charts
print("\n")
print("The pie charts show proportion within the categories of age/sex/fare")

fig, axs = plt.subplots(1,3, figsize=(100,10))

sex_count = subset_df_cleaned['Sex'].value_counts()
axs[0].pie(sex_count, labels = sex_count.index, startangle = 90)
axs[0].set_title('Distribution Based On Sex', color='r')


age_count = subset_df_cleaned['Age'].value_counts()
axs[1].pie(age_count, labels = age_count.index, startangle = 90)
axs[1].set_title('Distribution based on Age', color='r')


fare_count = subset_df_cleaned['Fare'].value_counts()
axs[2].pie(fare_count, labels = fare_count.index, startangle = 90)
axs[2].set_title('Distribution Based on Fare', color='r')


plt.suptitle("Distribution of Passengers within Three Categories", color='k')
plt.show()
print("\n")

print("Using Logistic Regression Model we test if the considered features influenced survival of a passenger.")

#model
subset_df_cleaned.loc[:,'Sex'] = subset_df_cleaned['Sex'].map({'male':0, 'female':1})

x = subset_df_cleaned[['Sex', 'Age', 'Fare']]
y = subset_df_cleaned['Survived']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=42) 

from sklearn.linear_model import LogisticRegression #selecting logistic regression model for testing
model = LogisticRegression()

model.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model is:", accuracy*100, "%")

