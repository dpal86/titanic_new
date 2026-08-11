import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error,root_mean_squared_error

df = pd.read_csv("juProject/titanic.csv", encoding="latin-1")

# Display the first 5 records
print("First 5 records:")
print(df.head(5))

# Display the dataset dimensions
print("\nDataset dimensions (rows, columns):")
print(df.shape)

# Missing values
print(df.isnull().sum())

df['age']=df['age'].fillna(df['age'].median(), inplace=True)

df = df.drop_duplicates()

# Updated dataset
print(df.isnull().sum())

# Total number of passengers
print("Total passengers:", df.shape[0])

# Calculate average age
average_age = df['age'].mean()

print("Average Age of Passengers:", round(average_age, 2))

# Maximum and minimum fare
max_fare = df['fare'].max()
min_fare = df['fare'].min()

print("Maximum Fare:", max_fare)
print("Minimum Fare:", min_fare)

# Filter: female passengers in first class
female_first_class = df[(df['sex'] == 'female') & (df['pclass'] == 1)]

# Display result
print(female_first_class[['survived','pclass','sex','age','fare','embarked']].head(10))
print("Total Female First-Class Passengers:", female_first_class.shape[0])

# Average fare by passenger class
avg_fare_by_class = df.groupby('pclass')['fare'].mean()


# Features (X) and Target (y)
X = df[['age']]   # predictor
y = df['fare']    # target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", root_mean_squared_error(y_test, y_pred))
print("Co-effient Slope:",model.coef_[0])
print("Intercept:",model.intercept_)


# Example: Predict fare for a passenger aged 30
predicted_fare = model.predict([[30]])
print("Predicted Fare for Age 30:", predicted_fare[0])

# Histogram of age
plt.figure(figsize=(8,5))
sns.histplot(df['age'], bins=20, kde=True)
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

#  Age vs Fare
plt.figure(figsize=(8,5))
sns.scatterplot(x='age', y='fare', data=df)
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # reference line
plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")
plt.title("Actual vs Predicted Age")
#plt.show()


