r"""°°°
<a href="https://colab.research.google.com/github/nyp-sit/sdaai-iti103/blob/master/session-4/classification_winequality.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
°°°"""

# |%%--%%| <yP5FJm7VaF|ciPpnaHC9I>
r"""°°°
# Multi-class Classification
°°°"""
# |%%--%%| <ciPpnaHC9I|AwHjgKXF8H>
r"""°°°
## Introduction

We will be using the wine quality data set for this exercise. This data set contains various chemical properties of wine, such as acidity, sugar, pH, alcohol, as well as color. It also contains a quality metric (3-9, with highest being better). 

Using what you have learnt in the previous exercises, you will now build a classification model to predict the quality of the wine, given the various chemical properties and color.
°°°"""
# |%%--%%| <AwHjgKXF8H|LugmycTuFp>
r"""°°°
## Getting the Data

You can download the data from the following link:

https://nyp-aicourse.s3-ap-southeast-1.amazonaws.com/datasets/Wine_Quality_Data.csv

°°°"""
# |%%--%%| <LugmycTuFp|5TnyKKkBtS>

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data_url = "https://nyp-aicourse.s3-ap-southeast-1.amazonaws.com/datasets/Wine_Quality_Data.csv"
df = pd.read_csv(data_url)
df.head()

# |%%--%%| <5TnyKKkBtS|HLlnqHZtU0>
r"""°°°
## Data Exploration

Find out the following: 
- how many samples we have? 
- are there any missing values? 
- are there any categorical data? 
- how many different grades (qualities) of wine. 
°°°"""
# |%%--%%| <HLlnqHZtU0|1fDFih9VGx>

## Write your code here
df.info()

# |%%--%%| <1fDFih9VGx|53AUvMBvDj>

df["color"].unique()

# |%%--%%| <53AUvMBvDj|DeDe7IjSUQ>
r"""°°°
## Data Preparation

As part of data prep, you will need some of the following:
- Encode any categorical columns if necessary
- Handle any missing values
- Scaling if necessary
- Split the datasets into train/val/test

Decide if you want to do K-fold cross-validation or set aside a dedicated validation set. Explain your choice.

Think about the splitting strategy, do you need stratified split?
°°°"""
# |%%--%%| <DeDe7IjSUQ|wgaTKiOMMU>

label_map = {"red": 0, "white": 1}

df["color"] = df["color"].map(label_map)

# |%%--%%| <wgaTKiOMMU|iqlaZynelC>

df["color"].head()

# |%%--%%| <iqlaZynelC|0AN7BxUCIK>

df["color"].value_counts()
# is skewed to white

# |%%--%%| <0AN7BxUCIK|FlrnXkpVOK>
from sklearn.model_selection import train_test_split

X = df[
    [
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "chlorides",
        "free_sulfur_dioxide",
        "total_sulfur_dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
        "quality",
    ]
]
y = df["color"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# |%%--%%| <FlrnXkpVOK|of1XrV3uAP>
r"""°°°
## Build and validate your model

For this exercise, use SVM as a start. You do not neeed to understand what the parameters mean at this point, as you will learn more during the ML Algorithms module. 

What do you notice about the validation accuracy/recall/precision? You can just use classification report to get more info about the performance of each class. Analyse the report and explain your result. 
°°°"""
# |%%--%%| <of1XrV3uAP|6K2YkmouqO>

## Write your code here
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

svc = LinearSVC(random_state=42)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(accuracy_score(y_test, y_pred))


# |%%--%%| <6K2YkmouqO|HAAnOKRYZc>

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# |%%--%%| <HAAnOKRYZc|Yq9sf86r3W>
r"""°°°
## Improve your model

Based on your analysis above, what do you think you can do to improve the model? 

Try to implement ONE possible change to improve your model.  Has the model improved in validation performance? 

Test it now on your test set. Do you get similar result as your validation result?
°°°"""
# |%%--%%| <Yq9sf86r3W|wCj49g6b6K>

## Write your code here
