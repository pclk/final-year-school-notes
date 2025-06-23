r"""°°°
# SDGRegressor to Predict Crowdness at the Gym

I love going to the Gym, however, I hate it when it's crowded because I cannot follow my plan at my rythm. I often have to wait for the machine I need to free up, and it becomes next to impossible to follow my routine.

Because of this, I decided to build a predictive model using Machine Learninhg, especifically a linear regressor using Stochastic Gradient Decsent.

Using a dataset with over 60,000 observations and 11 featres including day, hour, temperature and other details, I will be creating a model that can predict how many people will be at the gym at a particular day and time. That way, I will be able to enjoy my excersise routine without waiting times.
°°°"""

# |%%--%%| <SVrCrg8aHA|2rbfUf7FmP>
r"""°°°

### Import Libraries and Load the Data

The first step, is loading the libraries I will need to use to load and explore the data. I will be using the following ones:

- Numpy
- Pandas
- Matplotlib
- Seaborn

Also nore I am using the magic inline command to plot graphs directly on to the notebook.
°°°"""
# |%%--%%| <2rbfUf7FmP|WBbdz7YxzR>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# |%%--%%| <WBbdz7YxzR|JJYMeXgtyo>
r"""°°°
The dataset csv file is called 'crowdness_gym_data'. I am using the Pandas `read_csv` comand to load it into a dataframe called **df**.
°°°"""
# |%%--%%| <JJYMeXgtyo|QdWt1F4AC5>

df = pd.read_csv("crowdness_gym_data.csv")

# |%%--%%| <QdWt1F4AC5|nsIwnZUSfo>
r"""°°°
### EDA and Cleaning up the Data

To get a sense of the data, I ran some basic exploration commands, starting with `.head()` to get a general sense of the data.
°°°"""
# |%%--%%| <nsIwnZUSfo|4CvoRy2bNH>

df.head()

# |%%--%%| <4CvoRy2bNH|n8qnSIdtpD>
r"""°°°
Everything looks fairly straightforward and clean. Let's check out the shape of the data with `.shape`.
°°°"""
# |%%--%%| <n8qnSIdtpD|CL9FYYh3hw>

print("Data contains", df.shape[0], "rows and", df.shape[1], "columns")

# |%%--%%| <CL9FYYh3hw|GccNqBpsQS>
r"""°°°
There are 62,184 rows, or observations and 11 columns, or features.

Now to get somne info on each of the features with `.info()`
°°°"""
# |%%--%%| <GccNqBpsQS|k42DxAax6o>

df.info()

# |%%--%%| <k42DxAax6o|KCTz94nM0B>
r"""°°°
Most of the data is numeric and integers, with the exceptions of the temperature, which is a float (expected as the temperature is seldom a whole number) and the date object, which could be a problem.

We can get some more information with `.describe()` which gives us some basic statistics about the data.
°°°"""
# |%%--%%| <KCTz94nM0B|6WM5n5bfmr>

df.describe()

# |%%--%%| <6WM5n5bfmr|DkjecE51mY>
r"""°°°
It all looks fairly straighforward. The date column, as it is an object, has no statistics, and the timestamp seems to be wierd to work with. Most of the others seem good, with some of the features like **is_holiday** and **is_weekend** being binary features.

Finally, I will check to see if we have any empty (**Null**) values in the dataset with the `is.null()` and `.sum()` functions.
°°°"""
# |%%--%%| <DkjecE51mY|pOCUaMY87w>

df.isnull().sum()

# |%%--%%| <pOCUaMY87w|5jHWpdKWnL>
r"""°°°
There are no null valies in the dataset.

At the moment, the only feature that I feel is definitely problematic is the date column, since it is an object and because we already have other features that give us the specific day and time, I will get rid of the date column completely using `.drop()`.  Then run `.head()` again just to check the date column is gone.
°°°"""
# |%%--%%| <5jHWpdKWnL|uHYP6boxEu>

df = df.drop("date", axis=1)
df.head()

# |%%--%%| <uHYP6boxEu|4isK4a8f1q>
r"""°°°
### Plots

Now I am ready to do some EDA (Exploratory Data Analysis).

I will start by doing Univariate Analysis on some of the features. This means we will take a deeper look at the distributions of specific features.

I will plot histograms for the month, day and hour, since they probably have the largest influence on the ammount of people.
°°°"""
# |%%--%%| <4isK4a8f1q|LAFbccn9lT>

plt.figure(figsize=(8, 8))

plt.hist(df["month"])
plt.title("Observations per Month of the Year")
plt.xlabel("Month")
plt.ylabel("No. of Observavtions")
plt.show()


# |%%--%%| <LAFbccn9lT|wZOYQYxtHv>
r"""°°°
Here we can see that December and January are the months with the most observations, probably because they are the most popular months to go to the gym. We can also see more obervations at the begining of the semester (August), then at the end, probably because everyone is very excited at the beguning and very busy at the end (March, April).
°°°"""
# |%%--%%| <wZOYQYxtHv|3iUMpdDQoi>

plt.figure(figsize=(8, 8))

plt.hist(df["day_of_week"])
plt.title("Observations per Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("No. of Observavtions")
plt.show()


# |%%--%%| <3iUMpdDQoi|n5xirYdV04>
r"""°°°
This one looks strange, but it sometimes happens with plots. Instead of fighting with it, I will replot it using the Seaborn library and a ditribution plot.
°°°"""
# |%%--%%| <n5xirYdV04|YzaTEIU9EA>

plt.figure(figsize=(8, 8))

sns.displot(df["day_of_week"], color="g")
plt.title("Observations per Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("No. of Observavtions")
plt.show()

# |%%--%%| <YzaTEIU9EA|hRd9sXZ6TP>
r"""°°°
Not a lot of information here, except that there is not a huge diference in the number of observations for each day of the week. Tuesday (1) seems to be the most common day, but not by much.
°°°"""
# |%%--%%| <hRd9sXZ6TP|87JHBH3iDU>

plt.figure(figsize=(8, 8))

plt.hist(df["hour"])
plt.title("Observations per Day of the Week")
plt.xlabel("Hour")
plt.ylabel("No. of Observavtions")
plt.show()

# |%%--%%| <87JHBH3iDU|80GNQPuR6x>
r"""°°°
This is much more interesting, but just in case, there are a lot of observations at early morning and mid afternoon, which is expected, but the one at midnight are a surprise. Seems like there are night owls going to the gym.

This is interesting, but since I am building a model to predict the ammount of people (target variable), I can get more information from using Bivariate Analysis, meaning we confront two variables at the same time to see if there is any correlation between them.

Let's plot the relations between month, day and hour compared to the number of people.
°°°"""
# |%%--%%| <80GNQPuR6x|R0bdLjHteP>

plt.figure(figsize=(8, 8))

plt.scatter(df["month"], df["number_people"])
plt.title("Number of People VS Month")
plt.xlabel("Month")
plt.ylabel("Number of People")
plt.show()

# |%%--%%| <R0bdLjHteP|I1d2uChiOj>
r"""°°°
A clearer version of the relationship. We can see again that August and January are the months with the bigger peaks of people, and once again, that the begining of the semester has larger peaks than the end of it.
°°°"""
# |%%--%%| <I1d2uChiOj|0y53PVsECc>

plt.figure(figsize=(8, 8))

plt.scatter(df["day_of_week"], df["number_people"])
plt.title("Number of People VS Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Number of People")
plt.show()

# |%%--%%| <0y53PVsECc|9GL68gUn8b>
r"""°°°
Here we can now see the largest peaks on Monday and Wednesday. And the lower peaks on Saturdays.
°°°"""
# |%%--%%| <9GL68gUn8b|0Xy25ozck7>

plt.figure(figsize=(8, 8))

plt.scatter(df["hour"], df["number_people"])
plt.title("Number of People VS Hour")
plt.xlabel("Hour")
plt.ylabel("Number of People")
plt.show()

# |%%--%%| <0Xy25ozck7|mW0Xt2MKPy>
r"""°°°
Now we can see a lot clearer, that the largest peaks of people are during the afternoon, evening, and still surprising, large peaks late at night. ALso we see very small peaks from 2am to 5am.

I can go on with each variable, but to make it short, I will use a set of tools, from correlation tables, pairplots and a heatmap, to quickly see the relationship between each variable and out target (number of people).

I will start with the Correlation Table.
°°°"""
# |%%--%%| <mW0Xt2MKPy|646PwhTtCH>

df.corr()

# |%%--%%| <646PwhTtCH|iW3D2ehfvM>
r"""°°°
This table gives is a sense of the correlation (positive or negative), between each factor and each other variable. Since we are mostly interested in the number of people, we can stick to the first column of the table.

We cans ee how the **hour**, **temperature** and interestingly the **is_during_semester** variables have the largest weight. Also we can see that the timestamp and hour variables have a very similar weight, wich means they could be redundant.

Other variables have weaker correlations as well, like **is_weekend** and **day_of_week** are negatively correlated which is very interesting.

Another way to look at this is to use `pairplot()` function on seaborn, since it gives you a scatterplot for each pair of variables. It can be harder to read than the table though, but it can help us see some interesting pattern emerge.
°°°"""
# |%%--%%| <iW3D2ehfvM|EbkaDjECyj>

sns.pairplot(df)

# |%%--%%| <EbkaDjECyj|sxXpXodvSB>
r"""°°°
There is not a lot of additional info to discover from the paired plots here. Still, and just to make sure, I want to try one more visual, the `heatmap()` from seaborn.

In this case, we can create a heatmap using the correlation table, this will help us see the correlations with much more ease.
°°°"""
# |%%--%%| <sxXpXodvSB|nLgdrF7CXR>

plt.figure(figsize=(9, 9))
sns.heatmap(df.corr())

# |%%--%%| <nLgdrF7CXR|ODaqaZiFoF>
r"""°°°
This simply confirms our previous suspicions, that **temperature**, **hour** and **is_during_semester** variables are the most important.

Another thing, the **timestamp** seems to be redundant, since it has the same correlation as the hour, and we already have all the information on the month, day and time. So I will remove the **timestamp** column before moving on to building the model.

Also, check with `.head()` to make sure the column was removed.
°°°"""
# |%%--%%| <ODaqaZiFoF|bjDf4gH91p>

df = df.drop("timestamp", axis=1)
df.head()

# |%%--%%| <bjDf4gH91p|g9pIDgehJN>
r"""°°°
## Getting ready to build our model with Stochastic Gradient Descent

Now that the dataset is ready and we have our features, I need to import the tools needed to build the model. In this case, from the **Scikit Lean** library, the `train_test_split` and `SGDRegressor`.
°°°"""
# |%%--%%| <g9pIDgehJN|FA8Rqj0AQU>

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor

# |%%--%%| <FA8Rqj0AQU|lUNXaHSphv>
r"""°°°
I need to split the data into train and test sets. I am using a test size of 30% (70% of the data for training and 30% for testing). I am also setting the random state, so as to be able to replicate in the future.
°°°"""
# |%%--%%| <lUNXaHSphv|PxmBUyg9vi>

data = df.values
X = data[:, 1:]
y = data[:, 0]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# |%%--%%| <PxmBUyg9vi|4AudcII060>
r"""°°°
Check to make sure the shape of each set is correct.
°°°"""
# |%%--%%| <4AudcII060|u7Sxg9s0xU>

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")
print(f"Training label shape: {y_train.shape}")
print(f"Testing label shape: {y_test.shape}")

# |%%--%%| <u7Sxg9s0xU|q1RYKX9gJ1>
r"""°°°
Build the model object with `SGDRegressor`. Setting the learning rate to `optimal`, the loss function to hubber loss and using elasticnet for the penalty.

The fitting the model with the training data.
I set the `random_state` so as to be able to reproduce the training.
°°°"""
# |%%--%%| <q1RYKX9gJ1|gq090jBySG>

sgd_v1 = SGDRegressor(
    alpha=0.0001,
    learning_rate="optimal",
    loss="huber",
    penalty="elasticnet",
    random_state=52,
)

# |%%--%%| <gq090jBySG|lMVPr2Ghlv>

sgd_v1.fit(X_train, y_train)

# |%%--%%| <lMVPr2Ghlv|IXcS5ayPRk>
r"""°°°
## Measure the Performance of the Mode

Now that we have trained our model, it is time to predict the target variable with the test data. I will be using Mean Squared Error, Mean Absolute Error and R Sqared.
°°°"""
# |%%--%%| <IXcS5ayPRk|DOf1iSvnw6>

y_pred_v1 = sgd_v1.predict(X_test)  # Predict labels

# |%%--%%| <DOf1iSvnw6|ZMZe0yRwez>

# Let's evaluate the performance of the model.

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# The mean squared error
print(f"Mean squared error: {round( mean_squared_error(y_test, y_pred_v1),3)}")
# Explained variance score: 1 is perfect prediction
print(f"R2 score: {round(r2_score(y_test, y_pred_v1),3)}")
# Mean Absolute Error
print(f"Mean absolute error: { round(mean_absolute_error(y_test, y_pred_v1),3)}")

# |%%--%%| <ZMZe0yRwez|ZGPpnIGT32>
r"""°°°
1Mean Squared Error and Mean Absolute Error are fairly high (the closer to 0 the higher the accuracy), meaning the model is not incredibly accurate. With the R2 Score we can see there is a correlation of 0.506, wich is not terrible, but not that good either since we want it to be as close to 1 as possible.

To try and imprive the model, we can scale the features to normalize them from -1 to 1, this mught help improve the model. For this, I will import the `StandardScaler` from Scikit Learn.
°°°"""
# |%%--%%| <ZGPpnIGT32|3QJ58krX3c>

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# |%%--%%| <3QJ58krX3c|FmOm3rDxiJ>
r"""°°°
Now we can build another model with the scaled data and see if we can improve it.

I am using the same `random_state` for consistent results.
°°°"""
# |%%--%%| <FmOm3rDxiJ|FvWCR8vIlv>

sgd_v2 = SGDRegressor(
    alpha=0.0001,
    learning_rate="optimal",
    loss="huber",
    penalty="elasticnet",
    random_state=52,
)

# |%%--%%| <FvWCR8vIlv|YMP68MRtCn>

sgd_v2.fit(X_train_scaled, y_train)

# |%%--%%| <YMP68MRtCn|5qBqPsXrA9>

y_pred_v2 = sgd_v2.predict(X_test_scaled)  # Predict labels

# |%%--%%| <5qBqPsXrA9|VpLTbqg3S2>

# The mean squared error
print(f"Mean squared error: {round( mean_squared_error(y_test, y_pred_v2),3)}")
# Explained variance score: 1 is perfect prediction
print(f"R2 score: {round(r2_score(y_test, y_pred_v2),3)}")
# Mean Absolute Error
print(f"Mean absolute error: { round(mean_absolute_error(y_test, y_pred_v2),3)}")

# |%%--%%| <VpLTbqg3S2|A8YpQWL74j>
r"""°°°
With the scaled data, the model performs slightly better, decresing the Mean Squared Error and Mean Absolute Error and increasing the R2 score by 0.001.
°°°"""
# |%%--%%| <A8YpQWL74j|TOcCJd53Kk>
r"""°°°
## Visualizing the Results

To see how our model performs, the best way is to visualize it. Here is the plot from our first model, using line plots with the actual test data on the back and the predicted data on the front. The parts where the plots converge are the points where the model performed well, and the divergence in the plots is where the model performed poorly.
°°°"""
# |%%--%%| <TOcCJd53Kk|Hy91Q9c2PG>

plt.figure(figsize=(15, 15))

x_ax = range(len(y_test))
plt.plot(x_ax, y_test, linewidth=1, label="original")
plt.plot(x_ax, y_pred_v1, linewidth=1.1, label="predicted")
plt.title("y-test and y-predicted data Model 1")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend(loc="best", fancybox=True, shadow=True)
plt.grid(True)
plt.show()


# |%%--%%| <Hy91Q9c2PG|NVFsGQHUCI>

# Model v2

plt.figure(figsize=(15, 15))

x_ax = range(len(y_test))
plt.plot(x_ax, y_test, linewidth=1, label="original")
plt.plot(x_ax, y_pred_v2, linewidth=1.1, label="predicted")
plt.title("y-test and y-predicted data Model 2")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend(loc="best", fancybox=True, shadow=True)
plt.grid(True)
plt.show()

# |%%--%%| <NVFsGQHUCI|D5SEOH9a32>
r"""°°°
## Summarize your Results

We can clearly see there is a lot of room for improvement. However, a linear regression model using Stochastic Gradient Descent is a good place to start for building such a prediction model.

We can improve the model by making some changes. Regarding the data, I decided to remove the timestamp variable since I believed it to be redundant, nonetheless, maybe that redundancy might help the model get higher accuracy.

Also, I might changing and testing other hyperparameters might be interesting, especially changing the loss function from `huber` to `squared_epsilon_insensitive` and maybe exploring changing the learning rate and penalty.

In general, from the data and the model, for someone like me who likes to go to the gym often without having too many people there, any day at 5am seems like a safe bet.


°°°"""
