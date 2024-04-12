################################
# Credit Risk Prediction with Machine Learning
################################

# Problem:

# This project aims to estimate the probability
# that an individual or a client will make regular loan payments
# over a given period of time.
# The main problem of the credit risk prediction project is to determine
# whether loan applicants are likely
# to experience delays in loan payments in the future.

# Dataset story:

# The story of the dataset is for a study to estimate the credit risk of financial institutions when assessing loan applications.
# This dataset is used to estimate the likelihood that an individual or customer will make regular loan payments over a given period of time.
# Financial institutions take into account their customers' credit history, income level, debts and other financial information when assessing loan applications.
# However, it can be difficult to accurately estimate credit risk and predict future payment performance based on this data.
# This dataset is used to understand the challenges faced by such financial institutions and provide more accurate credit risk prediction.

#SeriousDlqin2yrs: A binary variable indicating whether the credit applicant has experienced serious delinquency in the last two years. (0: No delinquency, 1: Delinquency)
#RevolvingUtilizationOfUnsecuredLines: The ratio of the credit limit used by the credit applicant.
#age: The age of the credit applicant.
#NumberOfTime30-59DaysPastDueNotWorse: The number of times the credit applicant has been 30-59 days past due in the last two years.
#DebtRatio: The ratio of the credit applicant's debt to income.
#MonthlyIncome: The monthly income of the credit applicant.
#NumberOfOpenCreditLinesAndLoans: The number of open credit lines and loans the credit applicant has.
#NumberOfTimes90DaysLate: The number of times the credit applicant has been 90 or more days late in the last two years.
#NumberRealEstateLoansOrLines: The number of real estate loans or lines of credit the credit applicant has.
#NumberOfTime60-89DaysPastDueNotWorse: The number of times the credit applicant has been 60-89 days past due in the last two years.
#NumberOfDependents: The number of dependents the credit applicant has.


# Required library and functions
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# load the data set
data = pd.read_csv("C:/Users/USER/Desktop/python_temelleri/cs-training.csv")
print(data.head())

# Data preprocessing and feature engineering
# Fill in missing values
data.fillna(data.median(), inplace=True)

# Separating independent and dependent variables
X = data.drop(['SeriousDlqin2yrs', 'Unnamed: 0'], axis=1)
y = data['SeriousDlqin2yrs']

# Divide the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=42)

# Model building and training (Random Forest)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

#Evaluating model performance (Random Forest)
predictions_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, predictions_rf)
print('Random Forest Accuracy:', accuracy_rf)

# The output gives an accuracy value of 0.9364.
# This means that the model makes correct predictions on about 93.64% of the data in the test dataset.
# So, it can be said that the model performs well overall and generalizes well on the data.

# Model building and training (XGBoost)
model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)

#Evaluating model performance (XGBoost)
predictions_xgb = model_xgb.predict(X_test)
accuracy_xgb = accuracy_score(y_test, predictions_xgb)
print('XGBoost Accuracy:', accuracy_xgb)

# The accuracy value of 0.9366 indicates
# that the model made correct predictions on about 93.66% of the data in the test dataset.
# This indicates that the XGBoost model also performs well overall and generalizes well over the data.

# DATA VISUALIZATION
# Feature Analysis
sns.pairplot(data.sample(1000), hue="SeriousDlqin2yrs", diag_kind="kde")
plt.show()

# I just want to obtain graphs showing the relationships of the variables "age", "MonthlyIncome" and "DebtRatio".
sns.pairplot(data.sample(1000), vars=["age", "MonthlyIncome", "DebtRatio"], hue="SeriousDlqin2yrs", diag_kind="kde")
plt.show()

# Class Distribution
sns.countplot(x="SeriousDlqin2yrs", data=data)
plt.show()

# Model performance
sns.heatmap(confusion_matrix(y_test, predictions_rf), annot=True, cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix (Random Forest)")
plt.show()

sns.heatmap(confusion_matrix(y_test, predictions_xgb), annot=True, cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix (XGBoost)")
plt.show()





















