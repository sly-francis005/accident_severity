import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset (Replace 'road_accidents.csv' with your actual dataset file)
df = pd.read_csv("road_accidents.csv")

# Selecting relevant columns
columns = ["Accident_Severity", "Weather_Conditions", "Road_Type", "Speed_Limit", 
           "Number_of_Vehicles_Involved", "Lighting_Conditions", "Day_of_Week", "Time_of_Accident"]
df = df[columns]

# Convert categorical variables into numerical using one-hot encoding
df = pd.get_dummies(df, columns=["Weather_Conditions", "Road_Type", "Lighting_Conditions", "Day_of_Week"], drop_first=True)

# Splitting data into X (independent variables) and y (dependent variable)
X = df.drop("Accident_Severity", axis=1)
y = df["Accident_Severity"]

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# accident_severity
