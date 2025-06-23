r"""°°°
## Week 4: Imbalance Data
°°°"""

# |%%--%%| <l4VGr3Dyzi|KDF50O2EAy>
r"""°°°
## Load dataset
°°°"""
# |%%--%%| <KDF50O2EAy|nXxd4kXcVx>

import pandas as pd

data = pd.read_csv("./creditcard.csv")

# |%%--%%| <nXxd4kXcVx|FLSNqVDk19>

data.head()

# |%%--%%| <FLSNqVDk19|IlGKxmy4Ij>

data.shape

# |%%--%%| <IlGKxmy4Ij|BS1Yrnfr1T>

data.Class.value_counts()

# |%%--%%| <BS1Yrnfr1T|yiNjsapsda>

# check the number of 1s and 0s
count = data["Class"].value_counts()

print('Fraudulent "1" :', count[1])
print('Not Fraudulent "0":', count[0])

# print the percentage of question where target == 1
print(count[1] / count[0] * 100)

# |%%--%%| <yiNjsapsda|60nxF9g9OX>
r"""°°°
This show  that the data is highly imbalance. Only 0.17% of the data is belong to fraud
°°°"""
# |%%--%%| <60nxF9g9OX|WUEAN9qiqd>

import seaborn as sns
import matplotlib.pyplot as plt

# plot the no of 1's and 0's
g = sns.countplot(x="Class", data=data)
g.set_xticklabels(["Not Fraud", "Fraud"])
plt.show()

# |%%--%%| <WUEAN9qiqd|dCVIQmAwI0>

# check for null values
data.isnull().sum()

# |%%--%%| <dCVIQmAwI0|z3sZvREzyW>
r"""°°°
## Respose and Target variable
°°°"""
# |%%--%%| <z3sZvREzyW|ZfmukbhSbk>

import numpy as np

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

# check length of 1's and 0's
one = np.where(y == 1)
zero = np.where(y == 0)
len(one[0]), len(zero[0])

# |%%--%%| <ZfmukbhSbk|NXAKNv1Tty>
r"""°°°
## Train test split
°°°"""
# |%%--%%| <NXAKNv1Tty|pLX9BdHcZW>

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# |%%--%%| <pLX9BdHcZW|MU9zz3tQJX>

Train_one = np.where(y_train == 1)
Train_zero = np.where(y_train == 0)
len(Train_one[0]), len(Train_zero[0])

# |%%--%%| <MU9zz3tQJX|dN2f2pLV67>

Test_one = np.where(y_test == 1)
Test_zero = np.where(y_test == 0)
len(Test_one[0]), len(Test_zero[0])


# |%%--%%| <dN2f2pLV67|USYmW9TmQy>
r"""°°°
## Fit the model using Logitic Regression
°°°"""
# |%%--%%| <USYmW9TmQy|bVhz0fPRtm>

# create the object
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=10000)

model.fit(x, y)

y_predict = model.predict(x)

# |%%--%%| <bVhz0fPRtm|lfdyjoIexz>

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

accuracy_score(y_predict, y)

# |%%--%%| <lfdyjoIexz|yjTGSdWm78>
r"""°°°
Accuray of 99.89% is achieved with this dataset. Let have a look of the confusion matrix
°°°"""
# |%%--%%| <yjTGSdWm78|NXWKoXOV85>

confusion_matrix(y_predict, y)

# |%%--%%| <NXWKoXOV85|u96XEOWcsw>

roc_auc_score(y_predict, y)

# |%%--%%| <u96XEOWcsw|otsaTV8h72>

fpr, tpr, thresholds = roc_curve(y_predict, y)
plt.plot(fpr, tpr)

# |%%--%%| <otsaTV8h72|OWO1E0z2VS>

f1_score(y_predict, y)

# |%%--%%| <OWO1E0z2VS|ulq9JNOfpo>
r"""°°°
### What can you conclude from the confusion matrix, ROC and F1 score?
°°°"""
# |%%--%%| <ulq9JNOfpo|BN00AycbhJ>


# |%%--%%| <BN00AycbhJ|53mN7JVCrQ>
r"""°°°
##  Resampling Technique
°°°"""
# |%%--%%| <53mN7JVCrQ|yEkYb6lm6i>

# class count
class_count_0, class_count_1 = data["Class"].value_counts()

# divie class
class_0 = data[data["Class"] == 0]
class_1 = data[data["Class"] == 1]

# |%%--%%| <yEkYb6lm6i|d88HDgaI8u>

# print the shape of the class
print("class 0:", class_0.shape)
print("\nclass 1:", class_1.shape)

# |%%--%%| <d88HDgaI8u|h6pewT8poB>
r"""°°°
## 1. Random under sampling
°°°"""
# |%%--%%| <h6pewT8poB|K9bvE7Kkpu>

class_0_under = class_0.sample(class_count_1)

test_under = pd.concat([class_0_under, class_1], axis=0)

print("total class of 1 and 0:\n", test_under["Class"].value_counts())

test_under["Class"].value_counts().plot(kind="bar", title="Count (target)")
plt.show()

# |%%--%%| <K9bvE7Kkpu|8drrwggZ22>

# Let split the data into train and testing
# Train the model using Logistic Regression
# Get the performance matrix: Accuray, Confusion matrix, ROC and F1 Score

# |%%--%%| <8drrwggZ22|5pnWWq2zRT>
r"""°°°
## 2. Random over sampling
°°°"""
# |%%--%%| <5pnWWq2zRT|IYoSc4iaYF>

class_1_over = class_1.sample(class_count_0, replace=True)

test_under = pd.concat([class_1_over, class_0], axis=0)

# print the number of class count
print("class count of 1 and 0:\n", test_under["Class"].value_counts())

# plot the count
test_under["Class"].value_counts().plot(kind="bar", title="Count (target)")
plt.show()

# |%%--%%| <IYoSc4iaYF|lA5PIkUNfW>

# Let split the data into train and testing
# Train the model using Logistic Regression
# Get the performance matrix: Accuracy, Confusion matrix, ROC and F1 Score, ROC Curve


# |%%--%%| <lA5PIkUNfW|4VnGWC4zc8>
r"""°°°
## Balance data with imbalance learn module
°°°"""
# |%%--%%| <4VnGWC4zc8|bsSODnkW3t>

# import library
import imblearn

# |%%--%%| <bsSODnkW3t|adgR31EvTb>
r"""°°°
## 3. Random under-sampling with imblearn
°°°"""
# |%%--%%| <adgR31EvTb|CqjKlM2OEm>

# import library
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42, replacement=True)

# fit predictor and target varialbe
x_rus, y_rus = rus.fit_resample(x, y)

print("original dataset shape:", Counter(y))
print("Resample dataset shape", Counter(y_rus))

# |%%--%%| <CqjKlM2OEm|u2tFDGzvSM>

# Let split the data into train and testing
# Train the model using Logistic Regression
# Get the performance matrix: Accuracy, Confusion matrix, ROC and F1 Score, ROC Curve


# |%%--%%| <u2tFDGzvSM|VMDGWo4gJd>
r"""°°°
## 4. Random over-sampling with imblearn
°°°"""
# |%%--%%| <VMDGWo4gJd|jk5L7FGt6e>

# import library
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)

# fit predictor and target varaible
x_ros, y_ros = ros.fit_resample(x, y)

print("Original dataset shape", Counter(y))
print("Resample dataset shape", Counter(y_ros))

# |%%--%%| <jk5L7FGt6e|TeX871qc0j>

# Let split the data into train and testing
# Train the model using Logistic Regression
# Get the performance matrix: Accuracy, Confusion matrix, ROC and F1 Score, ROC Curve


# |%%--%%| <TeX871qc0j|FZOPgxQOF0>
r"""°°°
## 5. Under-sampling Tomek links
°°°"""
# |%%--%%| <FZOPgxQOF0|ebArxuAnAz>

# load library
from imblearn.under_sampling import TomekLinks

tl = TomekLinks(sampling_strategy="majority")

# fit predictor and target variable
x_tl, y_tl = tl.fit_resample(x, y)

print("Original dataset shape:", Counter(y))
print("Resample dataset shape:", Counter(y_tl))

# |%%--%%| <ebArxuAnAz|czle1rbCkX>

# Let split the data into train and testing
# Train the model using Logistic Regression
# Get the performance matrix: Accuracy, Confusion matrix, ROC and F1 Score, ROC Curve


# |%%--%%| <czle1rbCkX|ymMutx3yqv>
r"""°°°
## 6. Synthetic minority over-sampling technique (SMOTE)
°°°"""
# |%%--%%| <ymMutx3yqv|yHjGuhfQ4N>

# load library
from imblearn.over_sampling import SMOTE

smote = SMOTE()

# fit target and predictor variable
x_smote, y_smote = smote.fit_resample(x, y)

print("Origianl dataset shape:", Counter(y))
print("Resampple dataset shape:", Counter(y_smote))

# |%%--%%| <yHjGuhfQ4N|P6t1zxaHEi>

# Let split the data into train and testing
# Train the model using Logistic Regression
# Get the performance matrix: Accuracy, Confusion matrix, ROC and F1 Score, ROC Curve


# |%%--%%| <P6t1zxaHEi|m8nPiaFk4H>
r"""°°°
## 7. NearMiss
°°°"""
# |%%--%%| <m8nPiaFk4H|MCh9PwIPFy>

from imblearn.under_sampling import NearMiss

nm = NearMiss()

x_nm, y_nm = nm.fit_resample(x, y)

print("Original dataset shape:", Counter(y))
print("Resample dataset shape:", Counter(y_nm))

# |%%--%%| <MCh9PwIPFy|94kKQan4eX>

# Let split the data into train and testing
# Train the model using Logistic Regression
# Get the performance matrix: Accuracy, Confusion matrix, ROC and F1 Score, ROC Curve


# |%%--%%| <94kKQan4eX|UnKmxLmrKS>
r"""°°°
Let check out what is NearMiss
°°°"""
# |%%--%%| <UnKmxLmrKS|U5BE5CbP3c>


# |%%--%%| <U5BE5CbP3c|Khg6uGTcTX>
r"""°°°
## 8. penalize algorithm (cost-sensitive training)
°°°"""
# |%%--%%| <Khg6uGTcTX|9x3qWSyJSs>

"""
# load library
from sklearn.svm import SVC

# we can add class_weight='balanced' to add panalize mistake
svc_model = SVC(class_weight='balanced', probability=True)

svc_model.fit(x_train, y_train)

svc_predict = svc_model.predict(x_test)
"""

# |%%--%%| <9x3qWSyJSs|9nP4VGmfZf>

# Let split the data into train and testing
# Train the model using Logistic Regression
# Get the performance matrix: Accuracy, Confusion matrix, ROC and F1 Score, ROC Curve


# |%%--%%| <9nP4VGmfZf|id3mOTNiqz>
r"""°°°
## 10. Tree based algorithm

While in every machine learning problem, it’s a good rule of thumb to try a variety of algorithms, it can be especially beneficial with imbalanced datasets.

Decision trees frequently perform well on imbalanced data. In modern machine learning, tree ensembles (Random Forests, Gradient Boosted Trees, etc.) almost always outperform singular decision trees, so we’ll jump right into those:

Tree base algorithm work by learning a hierarchy of if/else questions. This can force both classes to be addressed.
°°°"""
# |%%--%%| <id3mOTNiqz|UVSgFcNG8P>

# Let Train the model using Random Forest
# Get the performance matrix: Accuracy, Confusion matrix, ROC and F1 Score, ROC Curve
# fit the predictor and target

# |%%--%%| <UVSgFcNG8P|EbC3kzNuwg>


# |%%--%%| <EbC3kzNuwg|XetDvhAirs>

# Let Train the model using XGBoost
# Get the performance matrix: Accuracy, Confusion matrix, ROC and F1 Score, ROC Curve
# fit the predictor and target

# |%%--%%| <XetDvhAirs|mC8iBxyRtQ>


# |%%--%%| <mC8iBxyRtQ|1sNKBAeQyw>
r"""°°°
What is the advantages and disadvantage of using under-sampling and over-sampling?
°°°"""
# |%%--%%| <1sNKBAeQyw|gUrWVvYPBD>
