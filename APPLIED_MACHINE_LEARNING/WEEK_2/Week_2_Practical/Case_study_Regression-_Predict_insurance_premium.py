r"""°°°
# Predicting Insurance Premiums

- Our simple dataset contains a few attributes for each person such as
- Age, Sex, BMI, Children, Smoker, Region and their charges

## Aim
- To use this info to predict charges for new customers
°°°"""
# |%%--%%| <xUAR1RAODH|jTrLK2yHeS>

import pandas as pd

# Uncomment this line if using this notebook locally
# insurance = pd.read_csv('./data/insurance/insurance.csv')

file_name = "https://raw.githubusercontent.com/rajeevratan84/datascienceforbusiness/master/insurance.csv"
insurance = pd.read_csv(file_name)

# Preview our data
insurance.head()

# |%%--%%| <jTrLK2yHeS|93cB6ihKHI>

insurance.info()

# |%%--%%| <93cB6ihKHI|vWkLOOKdrQ>

insurance.describe()

# |%%--%%| <vWkLOOKdrQ|0qHTQftC5a>

print("Rows     : ", insurance.shape[0])
print("Columns  : ", insurance.shape[1])
print("\nFeatures : \n", insurance.columns.tolist())
print("\nMissing values :  ", insurance.isnull().sum().values.sum())
print("\nUnique values :  \n", insurance.nunique())

# |%%--%%| <0qHTQftC5a|d3og6JzqzT>

insurance[["age", "bmi", "children", "charges"]].corr()

# |%%--%%| <d3og6JzqzT|SNbi9TEW7z>

import matplotlib.pyplot as plt


def plot_corr(df, size=10):
    """Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot"""

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.legend()
    cax = ax.matshow(corr)
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation="vertical")
    plt.yticks(range(len(corr.columns)), corr.columns)


plot_corr(insurance[["age", "bmi", "children", "charges"]])

# |%%--%%| <SNbi9TEW7z|KSJtu7RKQq>

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
insurance.plot(kind="hist", y="age", bins=70, color="b", ax=axes[0][0])
insurance.plot(kind="hist", y="bmi", bins=100, color="r", ax=axes[0][1])
insurance.plot(kind="hist", y="children", bins=6, color="g", ax=axes[1][0])
insurance.plot(kind="hist", y="charges", bins=100, color="orange", ax=axes[1][1])
plt.show()

# |%%--%%| <KSJtu7RKQq|JID84vVnGN>

insurance["sex"].value_counts().plot(kind="bar")

# |%%--%%| <JID84vVnGN|XqAuHgDEsa>

insurance["smoker"].value_counts().plot(kind="bar")

# |%%--%%| <XqAuHgDEsa|bl01bN0pYV>

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
insurance.plot(
    kind="scatter",
    x="age",
    y="charges",
    alpha=0.5,
    color="green",
    ax=axes[0],
    title="Age vs. Charges",
)
insurance.plot(
    kind="scatter",
    x="bmi",  # should this be changed to sex?
    y="charges",
    alpha=0.5,
    color="red",
    ax=axes[1],
    title="Sex vs. Charges",
)
insurance.plot(
    kind="scatter",
    x="children",
    y="charges",
    alpha=0.5,
    color="blue",
    ax=axes[2],
    title="Children vs. Charges",
)
plt.show()

# |%%--%%| <bl01bN0pYV|WYj6CpeX6M>

import seaborn as sns  # Imorting Seaborn library

pal = ["#FA5858", "#58D3F7"]
sns.scatterplot(x="bmi", y="charges", data=insurance, palette=pal, hue="smoker")

# |%%--%%| <WYj6CpeX6M|M5KwHYk89D>

pal = ["#FA5858", "#58D3F7"]
sns.catplot(
    x="sex", y="charges", hue="smoker", kind="violin", data=insurance, palette=pal
)

# |%%--%%| <M5KwHYk89D|9wz7Xn6MLF>

import seaborn as sns

sns.set(style="ticks")
pal = ["#FA5858", "#58D3F7"]

sns.pairplot(insurance, hue="smoker", palette=pal)
plt.title("Smokers")

# |%%--%%| <9wz7Xn6MLF|Zgv2XYOnI2>
r"""°°°
# Preparing Data for Machine Learning Algorithms
°°°"""
# |%%--%%| <Zgv2XYOnI2|rZeMsd1ayZ>

insurance.head()

# |%%--%%| <rZeMsd1ayZ|NLLHP1PsVG>

insurance["region"].unique()

# |%%--%%| <NLLHP1PsVG|Y7BBcoZ5c4>

insurance.drop(["region"], axis=1, inplace=True)
insurance.head()

# |%%--%%| <Y7BBcoZ5c4|vwzTapC5GF>

# Changing binary categories to 1s and 0s
insurance["sex"] = insurance["sex"].map(lambda s: 1 if s == "female" else 0)
insurance["smoker"] = insurance["smoker"].map(lambda s: 1 if s == "yes" else 0)

insurance.head()

# |%%--%%| <vwzTapC5GF|5FhsJicj8d>

X = insurance.drop(["charges"], axis=1)
y = insurance.charges

# |%%--%%| <5FhsJicj8d|CLlO0ZpPV8>
r"""°°°
# Modeling our Data
°°°"""
# |%%--%%| <CLlO0ZpPV8|z154gnmLF3>

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

print(lr.score(X_test, y_test))

# |%%--%%| <z154gnmLF3|Fx0FAV73uC>
r"""°°°
**Score** is the R2 score, which varies between 0 and 100%. It is closely related to the MSE but not the same. 

Wikipedia defines r2 like this, ” … is the proportion of the variance in the dependent variable that is predictable from the independent variable(s).” Another definition is “(total variance explained by model) / total variance.” So if it is 100%, the two variables are perfectly correlated, i.e., with no variance at all. A low value would show a low level of correlation, meaning a regression model that is not valid, but not in all cases.
°°°"""
# |%%--%%| <Fx0FAV73uC|NsFqkhGTc7>

results = pd.DataFrame({"Actual": y_test, "Predicted": y_test_pred})
results

# |%%--%%| <NsFqkhGTc7|OPSrfMtJv2>

# Normalize the data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# |%%--%%| <OPSrfMtJv2|sF73nd9M0Z>

pd.DataFrame(X_train).head()

# |%%--%%| <sF73nd9M0Z|ilELyGdaFH>

pd.DataFrame(y_train).head()

# |%%--%%| <ilELyGdaFH|j8GTCrClsw>

from sklearn.linear_model import LinearRegression  # Import Linear Regression model

multiple_linear_reg = LinearRegression(
    fit_intercept=False
)  # Create a instance for Linear Regression model
multiple_linear_reg.fit(X_train, y_train)  # Fit data to the model

# |%%--%%| <j8GTCrClsw|ENRGs7gw0J>

from sklearn.preprocessing import PolynomialFeatures

polynomial_features = PolynomialFeatures(
    degree=3
)  # Create a PolynomialFeatures instance in degree 3
x_train_poly = polynomial_features.fit_transform(
    X_train
)  # Fit and transform the training data to polynomial
x_test_poly = polynomial_features.fit_transform(
    X_test
)  # Fit and transform the testing data to polynomial

polynomial_reg = LinearRegression(
    fit_intercept=False
)  # Create a instance for Linear Regression model
polynomial_reg.fit(x_train_poly, y_train)  # Fit data to the model

# |%%--%%| <ENRGs7gw0J|CwVQe6bDI5>

from sklearn.tree import DecisionTreeRegressor  # Import Decision Tree Regression model

decision_tree_reg = DecisionTreeRegressor(
    max_depth=5, random_state=13
)  # Create a instance for Decision Tree Regression model
decision_tree_reg.fit(X_train, y_train)  # Fit data to the model

# |%%--%%| <CwVQe6bDI5|vkYALu0u8n>

from sklearn.ensemble import (
    RandomForestRegressor,
)  # Import Random Forest Regression model

random_forest_reg = RandomForestRegressor(
    n_estimators=400, max_depth=5, random_state=13
)  # Create a instance for Random Forest Regression model
random_forest_reg.fit(X_train, y_train)  # Fit data to the model

# |%%--%%| <vkYALu0u8n|O4DcF7ugTw>
r"""°°°
**NOTE:**
**n_estimators** represents the number of trees in the forest. Usually the higher the number of trees the better to learn the data. However, adding a lot of trees can slow down the training process considerably, therefore we do a parameter search to find the sweet spot.
°°°"""
# |%%--%%| <O4DcF7ugTw|2HqzhRBcp6>

from sklearn.svm import SVR  # Import SVR model

support_vector_reg = SVR(
    gamma="auto", kernel="linear", C=1000
)  # Create a instance for Support Vector Regression model
support_vector_reg.fit(X_train, y_train)  # Fit data to the model

# |%%--%%| <2HqzhRBcp6|DvUB36I12D>

from sklearn.model_selection import cross_val_predict  # For K-Fold Cross Validation
from sklearn.metrics import r2_score  # For find accuracy with R2 Score
from sklearn.metrics import mean_squared_error  # For MSE
from math import sqrt  # For squareroot operation

# |%%--%%| <DvUB36I12D|PzxAdTK9Fo>
r"""°°°
### Evaluating Multiple Linear Regression Model
°°°"""
# |%%--%%| <PzxAdTK9Fo|G2d061sDz2>

# Prediction with training dataset:
y_pred_MLR_train = multiple_linear_reg.predict(X_train)

# Prediction with testing dataset:
y_pred_MLR_test = multiple_linear_reg.predict(X_test)

# Find training accuracy for this model:
accuracy_MLR_train = r2_score(y_train, y_pred_MLR_train)
print("Training Accuracy for Multiple Linear Regression Model: ", accuracy_MLR_train)

# Find testing accuracy for this model:
accuracy_MLR_test = r2_score(y_test, y_pred_MLR_test)
print("Testing Accuracy for Multiple Linear Regression Model: ", accuracy_MLR_test)

# Find RMSE for training data:
RMSE_MLR_train = sqrt(mean_squared_error(y_train, y_pred_MLR_train))
print("RMSE for Training Data: ", RMSE_MLR_train)

# Find RMSE for testing data:
RMSE_MLR_test = sqrt(mean_squared_error(y_test, y_pred_MLR_test))
print("RMSE for Testing Data: ", RMSE_MLR_test)

# Prediction with 10-Fold Cross Validation:
y_pred_cv_MLR = cross_val_predict(multiple_linear_reg, X, y, cv=10)

# Find accuracy after 10-Fold Cross Validation
accuracy_cv_MLR = r2_score(y, y_pred_cv_MLR)
print(
    "Accuracy for 10-Fold Cross Predicted Multiple Linaer Regression Model: ",
    accuracy_cv_MLR,
)

# |%%--%%| <G2d061sDz2|uRyGnhqizw>
r"""°°°
###  Evaluating Polynomial Regression Model
°°°"""
# |%%--%%| <uRyGnhqizw|7Rgf8s6O81>

# Prediction with training dataset:
y_pred_PR_train = polynomial_reg.predict(x_train_poly)

# Prediction with testing dataset:
y_pred_PR_test = polynomial_reg.predict(x_test_poly)

# Find training accuracy for this model:
accuracy_PR_train = r2_score(y_train, y_pred_PR_train)
print("Training Accuracy for Polynomial Regression Model: ", accuracy_PR_train)

# Find testing accuracy for this model:
accuracy_PR_test = r2_score(y_test, y_pred_PR_test)
print("Testing Accuracy for Polynomial Regression Model: ", accuracy_PR_test)

# Find RMSE for training data:
RMSE_PR_train = sqrt(mean_squared_error(y_train, y_pred_PR_train))
print("RMSE for Training Data: ", RMSE_PR_train)

# Find RMSE for testing data:
RMSE_PR_test = sqrt(mean_squared_error(y_test, y_pred_PR_test))
print("RMSE for Testing Data: ", RMSE_PR_test)

# Prediction with 10-Fold Cross Validation:
y_pred_cv_PR = cross_val_predict(
    polynomial_reg, polynomial_features.fit_transform(X), y, cv=10
)

# Find accuracy after 10-Fold Cross Validation
accuracy_cv_PR = r2_score(y, y_pred_cv_PR)
print(
    "Accuracy for 10-Fold Cross Predicted Polynomial Regression Model: ", accuracy_cv_PR
)

# |%%--%%| <7Rgf8s6O81|UjvldMQCSZ>
r"""°°°
###  Evaluating Decision Tree Regression Model
°°°"""
# |%%--%%| <UjvldMQCSZ|2KvlIomU0p>

# Prediction with training dataset:
y_pred_DTR_train = decision_tree_reg.predict(X_train)

# Prediction with testing dataset:
y_pred_DTR_test = decision_tree_reg.predict(X_test)

# Find training accuracy for this model:
accuracy_DTR_train = r2_score(y_train, y_pred_DTR_train)
print("Training Accuracy for Decision Tree Regression Model: ", accuracy_DTR_train)

# Find testing accuracy for this model:
accuracy_DTR_test = r2_score(y_test, y_pred_DTR_test)
print("Testing Accuracy for Decision Tree Regression Model: ", accuracy_DTR_test)

# Find RMSE for training data:
RMSE_DTR_train = sqrt(mean_squared_error(y_train, y_pred_DTR_train))
print("RMSE for Training Data: ", RMSE_DTR_train)

# Find RMSE for testing data:
RMSE_DTR_test = sqrt(mean_squared_error(y_test, y_pred_DTR_test))
print("RMSE for Testing Data: ", RMSE_DTR_test)

# Prediction with 10-Fold Cross Validation:
y_pred_cv_DTR = cross_val_predict(decision_tree_reg, X, y, cv=10)

# Find accuracy after 10-Fold Cross Validation
accuracy_cv_DTR = r2_score(y, y_pred_cv_DTR)
print(
    "Accuracy for 10-Fold Cross Predicted Decision Tree Regression Model: ",
    accuracy_cv_DTR,
)

# |%%--%%| <2KvlIomU0p|qeTPWyMfUR>
r"""°°°
### Evaluating Random Forest Regression Model
°°°"""
# |%%--%%| <qeTPWyMfUR|EqHbxSUwCV>

# Prediction with training dataset:
y_pred_RFR_train = random_forest_reg.predict(X_train)

# Prediction with testing dataset:
y_pred_RFR_test = random_forest_reg.predict(X_test)

# Find training accuracy for this model:
accuracy_RFR_train = r2_score(y_train, y_pred_RFR_train)
print("Training Accuracy for Random Forest Regression Model: ", accuracy_RFR_train)

# Find testing accuracy for this model:
accuracy_RFR_test = r2_score(y_test, y_pred_RFR_test)
print("Testing Accuracy for Random Forest Regression Model: ", accuracy_RFR_test)

# Find RMSE for training data:
RMSE_RFR_train = sqrt(mean_squared_error(y_train, y_pred_RFR_train))
print("RMSE for Training Data: ", RMSE_RFR_train)

# Find RMSE for testing data:
RMSE_RFR_test = sqrt(mean_squared_error(y_test, y_pred_RFR_test))
print("RMSE for Testing Data: ", RMSE_RFR_test)

# Prediction with 10-Fold Cross Validation:
y_pred_cv_RFR = cross_val_predict(random_forest_reg, X, y, cv=10)

# Find accuracy after 10-Fold Cross Validation
accuracy_cv_RFR = r2_score(y, y_pred_cv_RFR)
print(
    "Accuracy for 10-Fold Cross Predicted Random Forest Regression Model: ",
    accuracy_cv_RFR,
)

# |%%--%%| <EqHbxSUwCV|m5fHmTE5p6>
r"""°°°
### Evaluating Support Vector Regression Model
°°°"""
# |%%--%%| <m5fHmTE5p6|yuVjlSbBcA>

# Prediction with training dataset:
y_pred_SVR_train = support_vector_reg.predict(X_train)

# Prediction with testing dataset:
y_pred_SVR_test = support_vector_reg.predict(X_test)

# Find training accuracy for this model:
accuracy_SVR_train = r2_score(y_train, y_pred_SVR_train)
print("Training Accuracy for Support Vector Regression Model: ", accuracy_SVR_train)

# Find testing accuracy for this model:
accuracy_SVR_test = r2_score(y_test, y_pred_SVR_test)
print("Testing Accuracy for Support Vector Regression Model: ", accuracy_SVR_test)

# Find RMSE for training data:
RMSE_SVR_train = sqrt(mean_squared_error(y_train, y_pred_SVR_train))
print("RMSE for Training Data: ", RMSE_SVR_train)

# Find RMSE for testing data:
RMSE_SVR_test = sqrt(mean_squared_error(y_test, y_pred_SVR_test))
print("RMSE for Testing Data: ", RMSE_SVR_test)

# Prediction with 10-Fold Cross Validation:
y_pred_cv_SVR = cross_val_predict(support_vector_reg, X, y, cv=10)

# Find accuracy after 10-Fold Cross Validation
accuracy_cv_SVR = r2_score(y, y_pred_cv_SVR)
print(
    "Accuracy for 10-Fold Cross Predicted Support Vector Regression Model: ",
    accuracy_cv_SVR,
)

# |%%--%%| <yuVjlSbBcA|O3APev0UMU>

# Compare all results in one table
training_accuracies = [
    accuracy_MLR_train,
    accuracy_PR_train,
    accuracy_DTR_train,
    accuracy_RFR_train,
    accuracy_SVR_train,
]
testing_accuracies = [
    accuracy_MLR_test,
    accuracy_PR_test,
    accuracy_DTR_test,
    accuracy_RFR_test,
    accuracy_SVR_test,
]
training_RMSE = [
    RMSE_MLR_train,
    RMSE_PR_train,
    RMSE_DTR_train,
    RMSE_RFR_train,
    RMSE_SVR_train,
]
testing_RMSE = [
    RMSE_MLR_test,
    RMSE_PR_test,
    RMSE_DTR_test,
    RMSE_RFR_test,
    RMSE_SVR_test,
]
cv_accuracies = [
    accuracy_cv_MLR,
    accuracy_cv_PR,
    accuracy_cv_DTR,
    accuracy_cv_RFR,
    accuracy_cv_SVR,
]

parameters = [
    "fit_intercept=False",
    "fit_intercept=False",
    "max_depth=5",
    "n_estimators=400, max_depth=5",
    "kernel=”linear”, C=1000",
]

table_data = {
    "Parameters": parameters,
    "Training Accuracy": training_accuracies,
    "Testing Accuracy": testing_accuracies,
    "Training RMSE": training_RMSE,
    "Testing RMSE": testing_RMSE,
    "10-Fold Score": cv_accuracies,
}
model_names = [
    "Multiple Linear Regression",
    "Polynomial Regression",
    "Decision Tree Regression",
    "Random Forest Regression",
    "Support Vector Regression",
]

table_dataframe = pd.DataFrame(data=table_data, index=model_names)
table_dataframe

# |%%--%%| <O3APev0UMU|YrReT1SLmE>
r"""°°°
### Our best classifier is our Random Forests using 400 estimators and a max_depth of 5
°°°"""
# |%%--%%| <YrReT1SLmE|ZpbobG3Jg3>
r"""°°°
**R^2 (coefficient of determination) regression score function.**

Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.
°°°"""
# |%%--%%| <ZpbobG3Jg3|TzR4Qm79an>
r"""°°°
# Let's test our best regression on some new data
°°°"""
# |%%--%%| <TzR4Qm79an|Bb2bASJB1g>

input_data = {
    "age": [35],
    "sex": ["male"],
    "bmi": [26],
    "children": [0],
    "smoker": ["no"],
    "region": ["southeast"],
}

input_data = pd.DataFrame(input_data)
input_data

# |%%--%%| <Bb2bASJB1g|2N8mT5XGdv>

# Our simple pre-processing
input_data.drop(["region"], axis=1, inplace=True)
input_data["sex"] = input_data["sex"].map(lambda s: 1 if s == "female" else 0)
input_data["smoker"] = input_data["smoker"].map(lambda s: 1 if s == "yes" else 0)
input_data

# |%%--%%| <2N8mT5XGdv|vkGpJDg2Yi>

# Scale our input data
input_data = sc.transform(input_data)
input_data

# |%%--%%| <vkGpJDg2Yi|DhrMGmlWRP>

# Reshape our input data in the format required by sklearn models
input_data = input_data.reshape(1, -1)
print(input_data.shape)
input_data

# |%%--%%| <DhrMGmlWRP|lKpK30D74z>

# Get our predicted insurance rate for our new customer
random_forest_reg.predict(input_data)

# |%%--%%| <lKpK30D74z|UexMp7qfUw>

# Note Standard Scaler remembers your inputs so you can use it still here
print(sc.mean_)
print(sc.scale_)

# |%%--%%| <UexMp7qfUw|aphpmBYBDy>
