r"""°°°
<a href="https://colab.research.google.com/github/nyp-sit/sdaai-iti103/blob/master/session-4/classification_text.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
°°°"""

# |%%--%%| <0|LCXs2bfKf9>
r"""°°°
# Classification

We have worked with regression type of problem in the previous exercise. Let us now take a closer look at classification type of problem.  

We will work with both binary classification and multi-class classification problems, and learn to compute different metrics to evaluate a classification model. 


°°°"""
# |%%--%%| <LCXs2bfKf9|FShVUjNFYB>
r"""°°°
## Binary Classification
°°°"""
# |%%--%%| <FShVUjNFYB|Toe0b8Qqg8>
r"""°°°
### Dataset

We will be using an SMS spam/ham dataset and build a binary classification model to help us predict if a text message is a spam or not. 

Let's go head and load the data into a panda dataframe and look at the first few samples.

°°°"""
# |%%--%%| <Toe0b8Qqg8|pOENwOuiUS>
r"""°°°

pip install pandas numpy matplotlib scikit-learn 

°°°"""
# |%%--%%| <pOENwOuiUS|MYVajFkRCt>

import pandas as pd

data_url = "https://nyp-aicourse.s3-ap-southeast-1.amazonaws.com/datasets/smsspamcollection.tsv"
df = pd.read_csv(data_url, sep="\t")
# I am the king of nvim
df.head()

# |%%--%%| <MYVajFkRCt|xidJaLS23P>
r"""°°°
### Data Preparation
°°°"""
# |%%--%%| <xidJaLS23P|hI9WvvUQfs>

df.tail()

# |%%--%%| <hI9WvvUQfs|qMYIAZCMRj>
r"""°°°
Let's see what are the different labels we have. 
°°°"""
# |%%--%%| <qMYIAZCMRj|T6fGPtIZdG>

df["label"].unique()

# |%%--%%| <T6fGPtIZdG|UTJOirLrSs>
r"""°°°
You will notice that we have two different labels: 'ham' and 'spam', both a text string (dtype=object)

As most of the evaluation metrics in scikit-learn assume (by default) positive label as 1 and negative label as 0, for convenience, we will first convert the label to 1 and 0. As our task is to detect spam, the positive label (label 1) in our case will be for spam.  

Let's create a mapping to map the string label to its corresponding numeric label and use the pandas [map()](https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html)  function to change the label to numeric label. 
°°°"""
# |%%--%%| <UTJOirLrSs|D2H3hOyQ3S>

labelmap = {"ham": 0, "spam": 1}

df["label"] = df["label"].map(labelmap)

# |%%--%%| <D2H3hOyQ3S|RTkkEHdyQo>

df.head()

# |%%--%%| <RTkkEHdyQo|1gZaKqQa6T>
r"""°°°
Always a good practice to check if there is any missing values, using ``isnull()`` method of dataframe.
°°°"""
# |%%--%%| <1gZaKqQa6T|IvPtGSEaV3>

df.isnull().sum()

# |%%--%%| <IvPtGSEaV3|Z5K4pInK7v>
r"""°°°
Let's get a sense of the distribution of positive and negative cases to see if we are dealing with imbalanced dataset.
°°°"""
# |%%--%%| <Z5K4pInK7v|RL2AuV5EQq>

df["label"].value_counts()

# |%%--%%| <RL2AuV5EQq|g7xlFr7t2Q>
r"""°°°
You will see that we have a lot more 'ham' messages than 'spam' messages: 4825 out of 5572 messages, or 86.6%, are ham. This means that any text classification model we create has to perform **better than 86.6%** to beat random chance. 
°°°"""
# |%%--%%| <g7xlFr7t2Q|JVMqXo2gfy>
r"""°°°
### Split data into train and test set

We will have to first decide what we want to use as features. For this lab, let us just start simply, only use the text message and ignore others like punctuation and message length. 

We then split the data randomly into 80-20 split of train and test set.
°°°"""
# |%%--%%| <JVMqXo2gfy|e0gWbxnbnq>

from sklearn.model_selection import train_test_split

X = df["message"]  # this time we want to look at the text
y = df["label"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# |%%--%%| <e0gWbxnbnq|efBWYkVkNK>
r"""°°°
### Text Pre-processing 

We cannot use text string directly as our input features for training our model. It has to be converted into numeric features first. There are many ways to do this, from simple bag-of-words approach to more sophisticated dense embedding using modern neural model. 

In this example, we will use the TF-IDF to represent our string as numeric vector. Text usually has to be pre-processed first, for example removal of punctuation marks, stop words, lower-casing, etc, before convert to numeric vector. Scikit-learn's [TFIDFVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) class conveniently do all these for us, transforming our collection of text into document matrix.

By default TfidfVectorizer will lowercase the text and remove punctuation. We have also removed the English stop_words such as 'the', 'is', etc. and also specify that only words that occurs 2 times or more should be included as part of the vocabulary (min_df=2). By keeping our vocubalary small, we are keeping our number of features small. 
°°°"""
# |%%--%%| <efBWYkVkNK|oaIpsNi0bV>

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(stop_words="english", min_df=2)

# We will first fit the vectorizer to the training text,
# and transform the training text into dcoument matrix
X_train_vect = tfidf_vect.fit_transform(X_train)
print(X_train_vect.shape)


# |%%--%%| <oaIpsNi0bV|qSgjrJZfZg>
r"""°°°
You can print out the vocabulary learnt by the TfidfVectorizer by accessing the instance variable `vocabulary_`. Notice that the vocbulary size is the feature size of your vectorized X_train. 
°°°"""
# |%%--%%| <qSgjrJZfZg|gkDVWhTdBv>

## printout a subset of vocabulary
print("Vocabulary size : ", len(tfidf_vect.vocabulary_))
print("Some words in the vocab : \n", list(tfidf_vect.vocabulary_.items())[:5])

# |%%--%%| <gkDVWhTdBv|M4k7Dvz9Fb>
r"""°°°
We will need to transform our X_test as well. We will use the TfidfVectorizer already fitted on train data to transform. There maybe a chance that certain words in the test set are not found in the vocabulary derived from the train set. In this case, the TfidfVectorizer will just ignore the unknown words.
°°°"""
# |%%--%%| <M4k7Dvz9Fb|MIs4Qvqpd4>

X_test_vect = tfidf_vect.transform(X_test)

# |%%--%%| <MIs4Qvqpd4|JmFpIQjwbn>
r"""°°°
Now we have gotten our features. Let's go ahead and train our model! 
°°°"""
# |%%--%%| <JmFpIQjwbn|rFQut24TB0>
r"""°°°
## Train a classifier 

We will now train a binary classifier capable of distinguishing between ham and spam. 

* Use Logistic Regression and train it on the whole training set. (use liblinear as solver and 42 as random_state)
* Use the trained classifier to predict the test set 
* Calculate the accuracy score 
°°°"""
# |%%--%%| <rFQut24TB0|cCbY2rVNMq>

# import the logistic regressor

from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(solver="liblinear", random_state=42)
lr_clf.fit(X_train_vect, y_train)
y_pred = lr_clf.predict(X_test_vect)


# |%%--%%| <cCbY2rVNMq|blc4fMcbKu>

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))

# |%%--%%| <blc4fMcbKu|rwzqKpQyxp>

from sklearn.svm import LinearSVC

svc = LinearSVC(random_state=42)
svc.fit(X_train_vect, y_train)
y_pred = svc.predict(X_test_vect)
print(accuracy_score(y_test, y_pred))

# |%%--%%| <rwzqKpQyxp|AZ84SCeTYK>
r"""°°°
Our accuracy on the chosen test set seems quite decent. But how do we know if it is because we are lucky to pick a 'easy' test set. Since our test set is pretty small, it may not be an accurate reflection of the accuracy of our model. A better way is to use cross-validation.
°°°"""
# |%%--%%| <AZ84SCeTYK|eo3fXDqwLF>
r"""°°°
### Measuring Accuracy using Cross-Validation

°°°"""
# |%%--%%| <eo3fXDqwLF|FYSGcKxYIj>
r"""°°°

Evaluate the **accuracy** of the model using cross-validation on the **train** data set with the `cross_val_score()` function, with 5 folds. 

**Exercise 1:**

What do you observe? What is the average validation accuracy?

<details><summary>Click here for answer</summary>

```python
    
val_accuracies = cross_val_score(lr_clf, X_train_vect, y_train, cv=5, scoring="accuracy")
print(val_accuracies)
print(np.mean(val_accuracies))
    
```
</details>
°°°"""
# |%%--%%| <FYSGcKxYIj|mrtGo6YDQI>

import numpy as np
from sklearn.model_selection import cross_val_score

# Complete your code here


# |%%--%%| <mrtGo6YDQI|VNPQzkTlRq>
r"""°°°
### Confusion Matrix


A much better way to understand how a trained classifier perform is to look at the confusion matrix. We will do the following: 
*   Generate a set of predictions using `cross_val_predict()` on the train data set
*   Compute the confusion matrix using the `confusion_matrix()` function.  Use ConfusionMatrixDisplay to plot the confusion matrix graphically.

°°°"""
# |%%--%%| <VNPQzkTlRq|QOvtVa2qfw>

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(lr_clf, X_train_vect, y_train, cv=5)


# |%%--%%| <QOvtVa2qfw|372C8gMQXb>

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(y_train, y_train_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=lr_clf.classes_)
disp.plot()

# |%%--%%| <372C8gMQXb|RwXdMBgnIC>
r"""°°°
**Exercise 2:**

What can you tell from the confusion matrix? What kind of errors does the model more frequently make? 
<br/>
<details><summary>Click here for answer</summary>
It predicts 400 spam messages correctly but got 198 wrong, represents only 66.8% recall rate for 'spam' class. It did however, better at predicting ham messages, which is not suprising, given we have a lot more ham messages in our training set.

<p><br/>
Important lesson here: Just looking at accuracy alone will not give you a full picture of the performance of your model. 
    
</details>
°°°"""
# |%%--%%| <RwXdMBgnIC|WcPx8GFpxJ>
r"""°°°
### Precision and Recall

**Exercise 3:**

From the confusion matrix above, compute the precision, recall and F1 score **manually** using the following formula:

- `recall = TP/(TP+FN)`
- `precision = TP/(TP+FP)`
- `F1 = 2*precision*recall/(precision + recall)`

<details><summary>Click here for answer</summary>
    
By convention, we use label 1 as positive case and label 0 as negative case. 
    
From the confusion matrix, we can obtain the following: 
- TP = 400
- FN = 198
- FP = 9
- TN = 3850

Now we can calculate recall, precision, and f1 easily: 

- recall = TP/(TP+FN) = 400/(400+198) = 0.67
- precision = TP/(TP+FP) = 400/(400+9) = 0.98
- f1 = 2\*precision\*recall/(precision+recall) = 0.8

</details>
°°°"""
# |%%--%%| <WcPx8GFpxJ|qJzNdUSzt6>
r"""°°°
Now we use the scikit learn's metric function to compute recall, precision and f1_score and compare the values with those manually computed: 
- recall_score()
- precision_score()
- f1_score()

Are they the same as your calculation? 
°°°"""
# |%%--%%| <qJzNdUSzt6|zo1XntU1Wu>

from sklearn.metrics import f1_score, precision_score, recall_score

print(recall_score(y_train, y_train_pred))
print(precision_score(y_train, y_train_pred))
print(f1_score(y_train, y_train_pred))

# |%%--%%| <zo1XntU1Wu|IcVaNofHYS>
r"""°°°
The is a another useful function called `classification_report()` in scikit-learn that gives all the metrics in one glance. Note that the ``classification_report()`` provides the precision/recall/f1-score values for each of the class. 

Note that we have different precison and recall scores for each class (0 and 1). 
°°°"""
# |%%--%%| <IcVaNofHYS|uQFF1QFPls>

from sklearn.metrics import classification_report

print(classification_report(y_train, y_train_pred))

# |%%--%%| <uQFF1QFPls|KIVQF6Qikd>
r"""°°°
Also note that we have different averages for precision, recall and f1 : macro average and weighted average in the classication_report. What is the difference between the two ? You can refer to this [link](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report) for info.  Manually calculate the macro and weighted average to check your understanding. 


°°°"""
# |%%--%%| <KIVQF6Qikd|HTlJkuQNOt>
r"""°°°
### Precision and Recall tradeoff

The confusion matrix and the classification report provide a very detailed analysis of
a particular set of predictions. However, the predictions themselves already threw
away a lot of information that is contained in the model. 

Most classifiers provide a `decision_function()` or a `predict_proba()` method to
assess degrees of certainty about predictions. Making predictions can be seen as
thresholding the output of decision_function or predict_proba at a certain fixed
point— in binary classification we use 0 for the decision function and 0.5 for
predict_proba.

In logistic regression, we can use the `decision_function()` method to compute the scores.   
°°°"""
# |%%--%%| <HTlJkuQNOt|DpOusNVT5I>
r"""°°°
First let's find a positive sample (using ``np.where`` to find all samples where y label == 1, and uses the first result as sample) and examine the decision score.
°°°"""
# |%%--%%| <DpOusNVT5I|aCEvlCWdBJ>

idx = np.where(y_train == 1)[0][0]
print(idx)

# |%%--%%| <aCEvlCWdBJ|mcUCTFWQtG>

sample_X = X_train_vect[idx]
sample_y = y_train[idx]

y_score = lr_clf.decision_function(sample_X)
print(y_score)

# |%%--%%| <mcUCTFWQtG|1LsijrW9JQ>
r"""°°°
With threshold = 0, the prediction (of positive case, i.e. 1) is correct.
°°°"""
# |%%--%%| <1LsijrW9JQ|bB7oxXk9WG>

threshold = 0
y_some_X_pred = y_score > threshold
print(y_some_X_pred == sample_y)

# |%%--%%| <bB7oxXk9WG|XrpJPkweq5>
r"""°°°
With threshold set at 6, prediction (of positive case, i.e. 1) is wrong. In other words, we failed to detect positive cases (lower recall)
°°°"""
# |%%--%%| <XrpJPkweq5|vlrKyj6L4E>

threshold = 6
y_some_data_pred = y_score > threshold
print(y_some_data_pred == sample_y)

# |%%--%%| <vlrKyj6L4E|tjzLFf6DeW>
r"""°°°
With a higher threshold, it decreases the recall and increases the precision. Conversely, with a lower threshold, we increases recall at the expense of decrease in precision. To decide which threshold to use, get the scores of all instances in the training set using the `cross_val_predict()` function to return decision scores instead of predictions.

Perform cross validation to get the scores for all instances.
°°°"""
# |%%--%%| <tjzLFf6DeW|4NzCMPKqDz>

y_scores = cross_val_predict(
    lr_clf, X_train_vect, y_train, cv=5, method="decision_function"
)

# |%%--%%| <4NzCMPKqDz|mfkluck7uy>
r"""°°°
Compute precision and recall for all possible thresholds using the precision_recall_curve function.
°°°"""
# |%%--%%| <mfkluck7uy|QHtuyIOuX3>

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)

# |%%--%%| <QHtuyIOuX3|NSFO69s9gR>


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16)
    plt.xlabel("Threshold", fontsize=16)
    plt.grid(True)


plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.show()

# |%%--%%| <NSFO69s9gR|eMby5AwYLL>
r"""°°°
Another way to select a good precision/recall trade-off is to plot precision directly against recall.
°°°"""
# |%%--%%| <eMby5AwYLL|iF4hstAiaA>


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.grid(True)


plt.figure(figsize=(8, 4))
plot_precision_vs_recall(precisions, recalls)

plt.show()

# |%%--%%| <iF4hstAiaA|1EfsSBDd3c>
r"""°°°
We want to aim for 80% or better recall, compute the threshold value.
°°°"""
# |%%--%%| <1EfsSBDd3c|NmTE2are3E>

threshold_80_recall = thresholds[np.argmin(recalls >= 0.8)]
threshold_80_recall

# |%%--%%| <NmTE2are3E|63aR44gjm6>

y_train_pred_80 = y_scores >= threshold_80_recall

# |%%--%%| <63aR44gjm6|I1jST3zTv1>
r"""°°°
Compute the precision and recall score
°°°"""
# |%%--%%| <I1jST3zTv1|sedwDciPqT>

precision_score(y_train, y_train_pred_80)

# |%%--%%| <sedwDciPqT|BopvAYqH4S>

recall_score(y_train, y_train_pred_80)

# |%%--%%| <BopvAYqH4S|6cHVDUTMo5>

print(classification_report(y_train, y_train_pred_80))

# |%%--%%| <6cHVDUTMo5|xRoJd58X7S>
r"""°°°
### ROC Curves
°°°"""
# |%%--%%| <xRoJd58X7S|bjRgJXL4ed>
r"""°°°
The receiver operation characteristic (ROC) curve is another common tool used with binary classifiers.  It is similar to the precision/recall curve, but it plots the true positive rate (recall) against the false positive rate.  
°°°"""
# |%%--%%| <bjRgJXL4ed|klXfj1jxjQ>

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train, y_scores)

# |%%--%%| <klXfj1jxjQ|mRqKg5NwI6>


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")  # dashed diagonal
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate (Fall-Out)", fontsize=16)
    plt.ylabel("True Positive Rate (Recall)", fontsize=16)
    plt.grid(True)


plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
plt.show()

# |%%--%%| <mRqKg5NwI6|RxRAiKzujf>
r"""°°°
The higher the recall (TPR), the more false positives (FPR) the classifier produces.  The dotted line represents the ROC curve of a purely random classifier, a good classfier stays as far away from the line as possible.

Let's Compute the area under the curve (AUC) using `roc_auc_score()`
°°°"""
# |%%--%%| <RxRAiKzujf|Kg2nf18SP9>

from sklearn.metrics import roc_auc_score

roc_auc_score(y_train, y_scores)

# |%%--%%| <Kg2nf18SP9|L9QOXLyctw>
r"""°°°
**Exercise 4:**

We are finally done with our binary classification...Wait a minute! Did we just computed all the evaluation metrics on ***training set*** ??!!  Isn't it bad practice to do so.. Don't we need to use ***test set*** to evaluate how good is our model?

Why?

<details><summary>Click here for answer</summary>

We only evaluate our model after we are satisfied with performance of it on our validation set. We will do our model fine-tuning on the validation set and not test set. In our case, since our training set is pretty small, if we are to set aside a validation set, then our training set would be too small. That is why we use ``cross_validation`` to evaluate our model
    
</details>

°°°"""
# |%%--%%| <L9QOXLyctw|h2465F7DHT>

lr_clf.fit(X_train_vect, y_train)

# |%%--%%| <h2465F7DHT|zzlzoBUnYr>

lr_clf.score(X_test_vect, y_test)

# |%%--%%| <zzlzoBUnYr|1aX05kOcMH>
r"""°°°
## Multiclass classification

We will now look at multi-class classification. The dataset we are going to use is the UCI ML hand-written digits datasets https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

The data set contains images of hand-written digits: 10 classes where each class refers to a digit. Each digit is a 8x8 image.  
°°°"""
# |%%--%%| <1aX05kOcMH|HhYvhazEtb>

from sklearn.datasets import load_digits

digits = load_digits()

print(digits.keys())

# |%%--%%| <HhYvhazEtb|QT5MgwU57l>
r"""°°°
**Exercise 5:**

Now create the X (the features) and y (the label) from the digits dataset.  X is a np.array of 64 pixel values, while y is the label e.g. 0, 1, 2, 3, .. 9.

<details><summary>Click here for answer</summary>
    
```python
    
X = digits['data']
y = digits['target']

```
</details>
°°°"""
# |%%--%%| <QT5MgwU57l|NIaaiJAojd>

# Complete your code here

X = None
y = None

# |%%--%%| <NIaaiJAojd|i8Qb2JFB9O>
r"""°°°
Let's plot the image of a particular digit to visualize it.  Before plotting, we need to reshape the 64 numbers into 8 x 8 image arrays so that it can be plotted.
°°°"""
# |%%--%%| <i8Qb2JFB9O|RIo5shobFT>

import matplotlib as mpl

# let's choose any one of the row and plot it
some_digit = X[100]

# print out the corresponding label
print("digit is {}".format(y[100]))

# reshape it to 8 x 8 image
some_digit_image = some_digit.reshape(8, 8)

plt.imshow(some_digit_image, cmap=mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()

# |%%--%%| <RIo5shobFT|PGrrk9urJX>
r"""°°°
**Exercise 6**

Split the data into train and test set, and randomly shuffle the data.


<details><summary>Click here for answer</summary>

```python
    
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)

```
</details>
°°°"""
# |%%--%%| <PGrrk9urJX|8JLIxUNDxc>

## Complete your code here


# |%%--%%| <8JLIxUNDxc|8fvGiSYDsL>
r"""°°°
Multiclass classifiers distinguish between more than two classess.  Scikit-learn detects when you try to use a binary classification algorithm for a multiple class classification task and it automatically runs one-versus-all (OvA)

**Exercise 7**

Use Logistic Regression to train using the training set, and make a prediction of the chosen digit (`some_digit`). Is the prediction correct?

<details><summary>Click here for answer</summary>

```python

lr_clf = LogisticRegression(solver='liblinear', random_state=42)
lr_clf.fit(X_train, y_train)
    
```
</details>
°°°"""
# |%%--%%| <8fvGiSYDsL|U5wyoETNYD>

# Complete the code here

lr_clf = None

# |%%--%%| <U5wyoETNYD|NeMJ3iGoBq>
r"""°°°
Under the hood, Scikit-Learn actually trained 10 binary classifiers, got their decision scores for the image and selected the class with the highest score.  

**Exercise 8**

Compute the scores for `some_digit` using the `decision_function()` method to return 10 scores, one per class.

<details><summary>Click here for answer</summary>

```python
    
some_digit_scores = lr_clf.decision_function([some_digit])
    
```
</details>
°°°"""
# |%%--%%| <NeMJ3iGoBq|T1a2dh0F7R>

# complete the code here

some_digit_scores = lr_clf.decision_function([some_digit])

# |%%--%%| <T1a2dh0F7R|ddzLgnDyH1>

some_digit_scores

# |%%--%%| <ddzLgnDyH1|7hozpBT0VT>
r"""°°°
The highest score is the one corresponding to the correct class.
°°°"""
# |%%--%%| <7hozpBT0VT|EtdUbqehHx>

index = np.argmax(some_digit_scores)
print(index)

# |%%--%%| <EtdUbqehHx|df9eMRbr0L>

lr_clf.classes_[index]

# |%%--%%| <df9eMRbr0L|zJeB7Rt3OI>
r"""°°°
**Exercise 9**

Use `cross_val_score()` to evaluate the classifier's accuracy.

<details><summary>Click here for answer</summary>
    
```python 
    
cross_val_score(lr_clf, X_train, y_train, cv=3, scoring="accuracy")
    
```
</details>  
°°°"""
# |%%--%%| <zJeB7Rt3OI|ghmwICysSk>

# Complete your code here


cross_val_score(lr_clf, X_train, y_train, cv=5, scoring="accuracy").mean()

# |%%--%%| <ghmwICysSk|xtjRXMNwfO>
r"""°°°
**Exercise 10**

Compute the confusion matrix of the classifier. From the confusion matrix, which two digits tend to be confused with each other?

<details><summary>Click here for answer</summary>
    
```python 

y_train_pred = cross_val_predict(lr_clf, X_train, y_train, cv=5)
cm = confusion_matrix(y_train, y_train_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
    
```
<br/>
1 and 8 are confused with each other. 
    
</details>  
°°°"""
# |%%--%%| <xtjRXMNwfO|j3aRqvLhCd>

# Complete your code here


# |%%--%%| <j3aRqvLhCd|HKc58bDAJa>
r"""°°°
**Exercise 11**

Print out the classification_report.  

<details><summary>Click here for answer</summary>
    
```python 

print(classification_report(y_train, y_train_pred))
    
```
</details>  
°°°"""
# |%%--%%| <HKc58bDAJa|L8dWs9Ftx4>

# Complete your code here


# |%%--%%| <L8dWs9Ftx4|MpgUdOoOWq>
r"""°°°
Question: Any algorithm only suitable for Binary classification? Any algorithm only suitable for multiclass classification?
Let try out others classification algorithm and compare the result.
°°°"""
# |%%--%%| <MpgUdOoOWq|mN1qIMtOKf>
