r"""°°°
# Method 1- Linear Regression

# Linear Regression on Olympic 100m Gold Times

!cclt text](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQTrYpw1FCRAqMEsZJRBr30sotdaZvia4NgQAWiJEuK13DAgnsZ)
°°°"""
# |%%--%%| <H5h8uxiFGS|ldhbaQp8eY>

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

file_name = "https://raw.githubusercontent.com/rajeevratan84/datascienceforbusiness/master/olympic100m.csv"

# read the file into python using read_csv
df = pd.read_csv(file_name)

# print out the first rows
df.head(40)

# |%%--%%| <ldhbaQp8eY|oAzc7p83oU>

df.shape

# |%%--%%| <oAzc7p83oU|lJhodwMBAJ>

x = df["year"]
x.shape

# |%%--%%| <lJhodwMBAJ|KAC2gY4L0J>

# Format data into correct shape
x_train = np.array(x).reshape((-1, 1))
x_train.shape

# |%%--%%| <KAC2gY4L0J|fJmAQUQ7BV>

y_train = np.array(df["time"])

# |%%--%%| <fJmAQUQ7BV|nzQmPu5HAm>

y_train.shape

# |%%--%%| <nzQmPu5HAm|loVgxFHces>

import numpy as np
from sklearn.linear_model import LinearRegression

# Let's create the model object using LinearRegression
model = LinearRegression()

# Fit our model to our input data x and y
model.fit(x_train, y_train)

y_pred = model.predict(x_train)
plt.scatter(x_train, y_train)
plt.plot(x, y_pred, color="r")

# |%%--%%| <loVgxFHces|7c0EsAARyD>

# Predict for 2020 Olympics

x_2020 = np.array([2020]).reshape(-1, 1)
x_2020.shape

# |%%--%%| <7c0EsAARyD|2W7i3ykmfO>

model.predict(x_2020)

# |%%--%%| <2W7i3ykmfO|UFCvaldLNh>

# How to apply polynomial regressions to this dataset?
# Step 1: Extract X and Y
# x = df.iloc[:, 1:2].values

# Extract our y or target variable Pressure
# y = df.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly1 = PolynomialFeatures(degree=9)
model_Poly1 = poly1.fit_transform(x_train)

lin_poly1 = LinearRegression()
lin_poly1.fit(model_Poly1, y_train)

lin_poly1.predict(poly1.fit_transform(x_2020))


# |%%--%%| <UFCvaldLNh|xz8SNkcV8M>
r"""°°°
# Polynomial Regressions
°°°"""
# |%%--%%| <xz8SNkcV8M|i75iyo5BC1>

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
file_name = "https://raw.githubusercontent.com/rajeevratan84/datascienceforbusiness/master/polylinearregression.csv"
df = pd.read_csv(file_name)
df.head(7)

# |%%--%%| <i75iyo5BC1|xid88sdgqm>

# Extract our x values, the column Temperature
x = df.iloc[:, 1:2].values

# Extract our y or target variable Pressure
y = df.iloc[:, 2].values

# |%%--%%| <xid88sdgqm|ciuGBZPqOp>

x

# |%%--%%| <ciuGBZPqOp|Q0dVIfkq8w>

# Fitting Polynomial Regression to the dataset
# Fitting the Polynomial Regression model on two components X and y.
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=9)
model_Poly = poly.fit_transform(x)

lin_poly = LinearRegression()
lin_poly.fit(model_Poly, y)

# |%%--%%| <Q0dVIfkq8w|FeIEf9nSW1>

# Visualising the Polynomial Regression results
plt.scatter(x, y, color="blue")

plt.plot(x, lin_poly.predict(model_Poly), color="red")
plt.title("Polynomial Regression")
plt.xlabel("Temperature")
plt.ylabel("Pressure")

plt.show()

# |%%--%%| <FeIEf9nSW1|J5DdUqcNZE>
r"""°°°
# Multivariate Linear Regression
°°°"""
# |%%--%%| <J5DdUqcNZE|PFBQTICSyk>

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
file_name = "https://raw.githubusercontent.com/rajeevratan84/datascienceforbusiness/master/auto-mpg.csv"
auto_df = pd.read_csv(file_name)
auto_df.head()

# |%%--%%| <PFBQTICSyk|GPCgqSmiQZ>

# Check for the rows that contain "?"
auto_df[auto_df["horsepower"] == "?"]

# |%%--%%| <GPCgqSmiQZ|W8uEMJ6Vpz>

# Get the indexes that have "?" instead of numbers
indexNames = auto_df[auto_df["horsepower"] == "?"].index

# Delete these row indexes from dataFrame
auto_df.drop(indexNames, inplace=True)

# |%%--%%| <W8uEMJ6Vpz|3BhXMnvAhg>

# Just checking to see if they've been removed
auto_df[auto_df["horsepower"] == "?"]

# |%%--%%| <3BhXMnvAhg|anVLTON8WS>

auto_df["horsepower"] = auto_df["horsepower"].astype(float)

# |%%--%%| <anVLTON8WS|xGyuMMgODj>

auto_df.info()

# |%%--%%| <xGyuMMgODj|57w5Lss6m1>

x = auto_df.iloc[:, 1:8].values
y = auto_df.iloc[:, 0].values

# |%%--%%| <57w5Lss6m1|kMBBCwZV6u>

x.shape

# |%%--%%| <kMBBCwZV6u|SfPY3XiIHF>

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# |%%--%%| <SfPY3XiIHF|SAQjTLTuJG>

# cylinders	displacement	horsepower	weight	acceleration	model year	origin

# Data for Honda Prelude actual mpg is 24
Trial_Data = np.array([8, 220, 195, 3042, 6.7, 98, 3])
Trial_Data = Trial_Data.reshape((-1, 7))
Trial_Data = Trial_Data.astype(float)

# |%%--%%| <SAQjTLTuJG|ajFBVIyLUl>

regressor.predict(X_test)

# |%%--%%| <ajFBVIyLUl|vv9ynN1A5q>
r"""°°°
# Support Vector Regression
°°°"""
# |%%--%%| <vv9ynN1A5q|FjY6FxWV3t>

from sklearn.svm import SVR

Support_vector_regressor = SVR()
Support_vector_regressor.fit(X_train, Y_train)

Support_vector_regressor.predict(X_test)

# |%%--%%| <FjY6FxWV3t|DDwCZNERfR>

from sklearn.metrics import r2_score, mean_squared_error

R2_LR = r2_score(regressor.predict(X_test), Y_test)
print(R2_LR)
MSE_LR = mean_squared_error(regressor.predict(X_test), Y_test)
print(MSE_LR)

# |%%--%%| <DDwCZNERfR|zoKElGJ2Je>

R2_SVR = r2_score(Support_vector_regressor.predict(X_test), Y_test)
print(R2_SVR)
MSE_SVR = mean_squared_error(Support_vector_regressor.predict(X_test), Y_test)
print(MSE_SVR)

# |%%--%%| <zoKElGJ2Je|ECN0jr6and>
r"""°°°
# Which Model is better?
°°°"""
# |%%--%%| <ECN0jr6and|3CAXsBcFb0>
