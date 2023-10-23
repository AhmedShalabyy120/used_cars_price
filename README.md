# Used Car Price Prediction

This repository contains a machine learning project that aims to predict used car prices based on various features. The project includes data preprocessing, model training, and evaluation.

## Getting Started

### Prerequisites

Make sure you have the following Python libraries installed:

- pandas
- seaborn
- plotly.express
- scikit-learn
- datasist
- joblib
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import RobustScaler, StandardScaler
from datasist.structdata import detect_outliers
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import joblib
import warnings 
warnings.simplefilter("ignore")
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("train.csv")

# Data Preprocessing
# The 'New_Price' column, which has 86% null values, was dropped.
df.drop('New_Price', axis=1, inplace=True)
df.dropna(axis=0, inplace=True)

# Data distribution and pairwise relationships were visualized.
sns.distplot(df['Price'])
sns.pairplot(df)

# Feature Engineering
outliers = detect_outliers(df, 0, Num_Columns)
for col in Num_Columns:
    df.loc[df.index.isin(outliers), col] = no_outliers[col].mean()

Nominal=['Location', 'Fuel_Type', 'Transmission', 'Brand']
Ordinal=['Owner_Type']
df = pd.get_dummies(df, columns=Nominal, drop_first=True)

scaler = StandardScaler()
df.iloc[:, :6] = scaler.fit_transform(df.iloc[:, :6])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2)

# Model Training
model = LinearRegression()
model.fit(xtrain, ytrain)
training_score = model.score(xtrain, ytrain)

# Linear Regression
# Achieved a training R-squared score of approximately 0.69.
print("Training R-squared:", training_score)

KN = KNeighborsRegressor(n_neighbors=6)
KN.fit(xtrain, ytrain)
training_score_knn = KN.score(xtrain, ytrain)
testing_score_knn = KN.score(xtest, ytest)

# K-Nearest Neighbors Regression (KNN)
# Achieved training R-squared score 87%
# Achieved a testing R-squared score of 83%.
print("KNN Training R-squared:", training_score_knn)
print("KNN Testing R-squared:", testing_score_knn)

## Results

The models were trained to predict used car prices, with the Linear Regression model achieving an R-squared score of 0.69 on the training data. The K-Nearest Neighbors model achieved a testing R-squared score of 83%.
