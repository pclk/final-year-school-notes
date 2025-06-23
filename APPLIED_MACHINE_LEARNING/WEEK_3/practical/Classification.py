r"""°°°
<a href="https://colab.research.google.com/github/nyp-sit/sdaai-iti103/blob/master/session-4/classification_text.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
°°°"""

# |%%--%%| <GoECHxTKMt|LAI91FCIOs>
r"""°°°
# Classification

We have worked with regression type of problem in the previous exercise. Let us now take a closer look at classification type of problem.  

We will work with both binary classification and multi-class classification problems, and learn to compute different metrics to evaluate a classification model. 


°°°"""
# |%%--%%| <LAI91FCIOs|VCMg9WEsyA>
r"""°°°
## Binary Classification
°°°"""
# |%%--%%| <VCMg9WEsyA|Lw2XejVQXy>
r"""°°°
### Dataset

We will be using an SMS spam/ham dataset and build a binary classification model to help us predict if a text message is a spam or not. 

Let's go head and load the data into a panda dataframe and look at the first few samples.

°°°"""
# |%%--%%| <Lw2XejVQXy|UiEoHv5UkQ>

import pandas as pd

data_url = "https://nyp-aicourse.s3-ap-southeast-1.amazonaws.com/datasets/smsspamcollection.tsv"
df = pd.read_csv(data_url, sep="\t")
df.head()

# |%%--%%| <UiEoHv5UkQ|JmaKwH7r7z>
r"""°°°
### Data Preparation
°°°"""
# |%%--%%| <JmaKwH7r7z|c0hWQuD3M9>
r"""°°°
Let's see what are the different labels we have. 
°°°"""
# |%%--%%| <c0hWQuD3M9|H3AFzohu3u>

df["label"].unique()

# |%%--%%| <H3AFzohu3u|LbWHaT6azr>
r"""°°°
You will notice that we have two different labels: 'ham' and 'spam', both a text string (dtype=object)

As most of the evaluation metrics in scikit-learn assume (by default) positive label as 1 and negative label as 0, for convenience, we will first convert the label to 1 and 0. As our task is to detect spam, the positive label (label 1) in our case will be for spam.  

Let's create a mapping to map the string label to its corresponding numeric label and use the pandas [map()](https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html)  function to change the label to numeric label. 
°°°"""
# |%%--%%| <LbWHaT6azr|sl9vFVwVO7>

labelmap = {"ham": 0, "spam": 1}

df["label"] = df["label"].map(labelmap)

# |%%--%%| <sl9vFVwVO7|kRV0dbTlAQ>

df.head()

# |%%--%%| <kRV0dbTlAQ|oXAlrA7elM>
r"""°°°
Always a good practice to check if there is any missing values, using ``isnull()`` method of dataframe.
°°°"""
# |%%--%%| <oXAlrA7elM|OC1GRWxd30>

df.isnull().sum()

# |%%--%%| <OC1GRWxd30|DCTnqrSkhQ>
r"""°°°
Let's get a sense of the distribution of positive and negative cases to see if we are dealing with imbalanced dataset.
°°°"""
# |%%--%%| <DCTnqrSkhQ|cHrgekpkv5>

df["label"].value_counts()

# |%%--%%| <cHrgekpkv5|PsPNWLuDZu>
r"""°°°
You will see that we have a lot more 'ham' messages than 'spam' messages: 4825 out of 5572 messages, or 86.6%, are ham. This means that any text classification model we create has to perform **better than 86.6%** to beat random chance. 
°°°"""
# |%%--%%| <PsPNWLuDZu|D6FjBmdXtQ>
r"""°°°
### Split data into train and test set

We will have to first decide what we want to use as features. For this lab, let us just start simply, only use the text message and ignore others like punctuation and message length. 

We then split the data randomly into 80-20 split of train and test set.
°°°"""
# |%%--%%| <D6FjBmdXtQ|1P0TJiTOWg>

from scipy.sparse import random
from sklearn.model_selection import train_test_split

X = df["message"]  # this time we want to look at the text
y = df["label"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# |%%--%%| <1P0TJiTOWg|lyHlkjrR8Z>
r"""°°°
### Text Pre-processing 

We cannot use text string directly as our input features for training our model. It has to be converted into numeric features first. There are many ways to do this, from simple bag-of-words approach to more sophisticated dense embedding using modern neural model. 

In this example, we will use the TF-IDF to represent our string as numeric vector. Text usually has to be pre-processed first, for example removal of punctuation marks, stop words, lower-casing, etc, before convert to numeric vector. Scikit-learn's [TFIDFVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) class conveniently do all these for us, transforming our collection of text into document matrix.

By default TfidfVectorizer will lowercase the text and remove punctuation. We have also removed the English stop_words such as 'the', 'is', etc. and also specify that only words that occurs 2 times or more should be included as part of the vocabulary (min_df=2). By keeping our vocubalary small, we are keeping our number of features small. 
°°°"""
# |%%--%%| <lyHlkjrR8Z|QngLwK6y8R>

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(stop_words="english", min_df=2)

# We will first fit the vectorizer to the training text,
# and transform the training text into dcoument matrix
X_train_vect = tfidf_vect.fit_transform(X_train)
print(X_train_vect.shape)


# |%%--%%| <QngLwK6y8R|3iWUaj5JTD>
r"""°°°
You can print out the vocabulary learnt by the TfidfVectorizer by accessing the instance variable `vocabulary_`. Notice that the vocbulary size is the feature size of your vectorized X_train. 
°°°"""
# |%%--%%| <3iWUaj5JTD|WXo7UFeNwj>

## printout a subset of vocabulary
print("Vocabulary size : ", len(tfidf_vect.vocabulary_))
print("Some words in the vocab : \n", list(tfidf_vect.vocabulary_.items())[:5])

# |%%--%%| <WXo7UFeNwj|KcJREBocj6>
r"""°°°
We will need to transform our X_test as well. We will use the TfidfVectorizer already fitted on train data to transform. There maybe a chance that certain words in the test set are not found in the vocabulary derived from the train set. In this case, the TfidfVectorizer will just ignore the unknown words.
°°°"""
# |%%--%%| <KcJREBocj6|wbyiWz5jg9>

X_test_vect = tfidf_vect.transform(X_test)

# |%%--%%| <wbyiWz5jg9|EkmqbshqIX>
r"""°°°
Now we have gotten our features. Let's go ahead and train our model! 
°°°"""
# |%%--%%| <EkmqbshqIX|2GRRN6Pjxf>
r"""°°°
## Train a classifier 

We will now train a binary classifier capable of distinguishing between ham and spam. 

* Use Logistic Regression and train it on the whole training set. (use liblinear as solver and 42 as random_state)
* Use the trained classifier to predict the test set 
* Calculate the accuracy score 
°°°"""
# |%%--%%| <2GRRN6Pjxf|lmxHYJAu5u>

# import the logistic regressor

from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(solver="liblinear", random_state=42)
lr_clf.fit(X_train_vect, y_train)
y_pred = lr_clf.predict(X_test_vect)


# |%%--%%| <lmxHYJAu5u|ImOcAFgGUe>

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))

# |%%--%%| <ImOcAFgGUe|EvzRqQ6dya>

from sklearn.svm import LinearSVC

svc = LinearSVC(random_state=42)
svc.fit(X_train_vect, y_train)
y_pred = svc.predict(X_test_vect)
print(accuracy_score(y_test, y_pred))

# |%%--%%| <EvzRqQ6dya|UihQZ6K41Q>
r"""°°°
Our accuracy on the chosen test set seems quite decent. But how do we know if it is because we are lucky to pick a 'easy' test set. Since our test set is pretty small, it may not be an accurate reflection of the accuracy of our model. A better way is to use cross-validation.
°°°"""
# |%%--%%| <UihQZ6K41Q|XKO8KOCrM0>
r"""°°°
### Measuring Accuracy using Cross-Validation

°°°"""
# |%%--%%| <XKO8KOCrM0|DZ1UtZVFog>
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
# |%%--%%| <DZ1UtZVFog|Coc7yP3UNB>

from sklearn.model_selection import cross_val_score
import numpy as np

# Complete your code here
val_accuracies = cross_val_score(
    lr_clf, X_train_vect, y_train, cv=5, scoring="accuracy"
)
print(val_accuracies)
print(np.mean(val_accuracies))


# |%%--%%| <Coc7yP3UNB|359cXs0H4v>
r"""°°°
### Confusion Matrix


A much better way to understand how a trained classifier perform is to look at the confusion matrix. We will do the following: 
*   Generate a set of predictions using `cross_val_predict()` on the train data set
*   Compute the confusion matrix using the `confusion_matrix()` function.  Use ConfusionMatrixDisplay to plot the confusion matrix graphically.

°°°"""
# |%%--%%| <359cXs0H4v|Crdnmye1EO>

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(lr_clf, X_train_vect, y_train, cv=5)


# |%%--%%| <Crdnmye1EO|wJeV97gONf>

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_train, y_train_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=lr_clf.classes_)
disp.plot()

# |%%--%%| <wJeV97gONf|1VhCaaK2cK>
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
# |%%--%%| <1VhCaaK2cK|tS0Q8MVBQP>
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
# |%%--%%| <tS0Q8MVBQP|gF7WZqYPYY>
r"""°°°
Now we use the scikit learn's metric function to compute recall, precision and f1_score and compare the values with those manually computed: 
- recall_score()
- precision_score()
- f1_score()

Are they the same as your calculation? 
°°°"""
# |%%--%%| <gF7WZqYPYY|hTLM5pehtj>

from sklearn.metrics import recall_score, precision_score, f1_score

print(recall_score(y_train, y_train_pred))
print(precision_score(y_train, y_train_pred))
print(f1_score(y_train, y_train_pred))

# |%%--%%| <hTLM5pehtj|au3rSVU2hG>
r"""°°°
The is a another useful function called `classification_report()` in scikit-learn that gives all the metrics in one glance. Note that the ``classification_report()`` provides the precision/recall/f1-score values for each of the class. 

Note that we have different precison and recall scores for each class (0 and 1). 
°°°"""
# |%%--%%| <au3rSVU2hG|38u7UBwJTQ>

from sklearn.metrics import classification_report

print(classification_report(y_train, y_train_pred))

# |%%--%%| <38u7UBwJTQ|DT0vXQBQi0>
r"""°°°
Also note that we have different averages for precision, recall and f1 : macro average and weighted average in the classication_report. What is the difference between the two ? You can refer to this [link](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report) for info.  Manually calculate the macro and weighted average to check your understanding. 


°°°"""
# |%%--%%| <DT0vXQBQi0|SA0MZAijKF>
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
# |%%--%%| <SA0MZAijKF|b8thdnMUpP>
r"""°°°
First let's find a positive sample (using ``np.where`` to find all samples where y label == 1, and uses the first result as sample) and examine the decision score.
°°°"""
# |%%--%%| <b8thdnMUpP|3DIo8kyIX3>

idx = np.where(y_train == 1)[0][0]
print(idx)

# |%%--%%| <3DIo8kyIX3|Xjly43c0V2>

sample_X = X_train_vect[idx]
sample_y = y_train[idx]

y_score = lr_clf.decision_function(sample_X)
print(y_score)

# |%%--%%| <Xjly43c0V2|QugQx7n0e2>
r"""°°°
With threshold = 0, the prediction (of positive case, i.e. 1) is correct.
°°°"""
# |%%--%%| <QugQx7n0e2|dDVEhbDCpn>

threshold = 0
y_some_X_pred = y_score > threshold
print(y_some_X_pred == sample_y)

# |%%--%%| <dDVEhbDCpn|kLRiECLTJM>
r"""°°°
With threshold set at 6, prediction (of positive case, i.e. 1) is wrong. In other words, we failed to detect positive cases (lower recall)
°°°"""
# |%%--%%| <kLRiECLTJM|UyXWK9uyjM>

threshold = 6
y_some_data_pred = y_score > threshold
print(y_some_data_pred == sample_y)

# |%%--%%| <UyXWK9uyjM|CoyOOz2SGE>
r"""°°°
With a higher threshold, it decreases the recall and increases the precision. Conversely, with a lower threshold, we increases recall at the expense of decrease in precision. To decide which threshold to use, get the scores of all instances in the training set using the `cross_val_predict()` function to return decision scores instead of predictions.

Perform cross validation to get the scores for all instances.
°°°"""
# |%%--%%| <CoyOOz2SGE|Sv7weiRLsT>

y_scores = cross_val_predict(
    lr_clf, X_train_vect, y_train, cv=5, method="decision_function"
)

# |%%--%%| <Sv7weiRLsT|qkdb7IhLbm>
r"""°°°
Compute precision and recall for all possible thresholds using the precision_recall_curve function.
°°°"""
# |%%--%%| <qkdb7IhLbm|MEHmw5VyeH>

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)

# |%%--%%| <MEHmw5VyeH|Wo35RkB3bg>


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16)
    plt.xlabel("Threshold", fontsize=16)
    plt.grid(True)


plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.show()

# |%%--%%| <Wo35RkB3bg|PylXdNC9JK>
r"""°°°
Another way to select a good precision/recall trade-off is to plot precision directly against recall.
°°°"""
# |%%--%%| <PylXdNC9JK|GC3mjvRgvi>


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.grid(True)


plt.figure(figsize=(8, 4))
plot_precision_vs_recall(precisions, recalls)

plt.show()

# |%%--%%| <GC3mjvRgvi|xchTIiqKvf>
r"""°°°
We want to aim for 80% or better recall, compute the threshold value.
°°°"""
# |%%--%%| <xchTIiqKvf|1YD8JITKno>

threshold_80_recall = thresholds[np.argmin(recalls >= 0.8)]
threshold_80_recall

# |%%--%%| <1YD8JITKno|Vo4HFatacn>

y_train_pred_80 = y_scores >= threshold_80_recall

# |%%--%%| <Vo4HFatacn|akEXXjl2t1>
r"""°°°
Compute the precision and recall score
°°°"""
# |%%--%%| <akEXXjl2t1|sTeQZ368Qv>

precision_score(y_train, y_train_pred_80)

# |%%--%%| <sTeQZ368Qv|qTae2jPOUY>

recall_score(y_train, y_train_pred_80)

# |%%--%%| <qTae2jPOUY|nOiZ4HTpCX>

print(classification_report(y_train, y_train_pred_80))

# |%%--%%| <nOiZ4HTpCX|I4iR6uLjLA>
r"""°°°
### ROC Curves
°°°"""
# |%%--%%| <I4iR6uLjLA|YAISBM5RUz>
r"""°°°
The receiver operation characteristic (ROC) curve is another common tool used with binary classifiers.  It is similar to the precision/recall curve, but it plots the true positive rate (recall) against the false positive rate.  
°°°"""
# |%%--%%| <YAISBM5RUz|Zif35yNBIF>

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train, y_scores)

# |%%--%%| <Zif35yNBIF|d9c9A7irVO>


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

# |%%--%%| <d9c9A7irVO|X1WVpRJyvy>
r"""°°°
The higher the recall (TPR), the more false positives (FPR) the classifier produces.  The dotted line represents the ROC curve of a purely random classifier, a good classfier stays as far away from the line as possible.

Let's Compute the area under the curve (AUC) using `roc_auc_score()`
°°°"""
# |%%--%%| <X1WVpRJyvy|uDI40OuTys>

from sklearn.metrics import roc_auc_score

roc_auc_score(y_train, y_scores)

# |%%--%%| <uDI40OuTys|274GzlFuQP>
r"""°°°
**Exercise 4:**

We are finally done with our binary classification...Wait a minute! Did we just computed all the evaluation metrics on ***training set*** ??!!  Isn't it bad practice to do so.. Don't we need to use ***test set*** to evaluate how good is our model?

Why?

<details><summary>Click here for answer</summary>

We only evaluate our model after we are satisfied with performance of it on our validation set. We will do our model fine-tuning on the validation set and not test set. In our case, since our training set is pretty small, if we are to set aside a validation set, then our training set would be too small. That is why we use ``cross_validation`` to evaluate our model
    
</details>

°°°"""
# |%%--%%| <274GzlFuQP|cxmfhJb4pJ>

lr_clf.fit(X_train_vect, y_train)

# |%%--%%| <cxmfhJb4pJ|VdvN8TuzbY>

lr_clf.score(X_test_vect, y_test)

# |%%--%%| <VdvN8TuzbY|M0AYIB4qi0>
r"""°°°
## Multiclass classification

We will now look at multi-class classification. The dataset we are going to use is the UCI ML hand-written digits datasets https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

The data set contains images of hand-written digits: 10 classes where each class refers to a digit. Each digit is a 8x8 image.  
°°°"""
# |%%--%%| <M0AYIB4qi0|XnfaDJ3JVT>

from sklearn.datasets import load_digits

digits = load_digits()

print(digits.keys())

# |%%--%%| <XnfaDJ3JVT|wuV8ErYzPj>
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
# |%%--%%| <wuV8ErYzPj|pdNJLeWAZU>

# Complete your code here

X = digits["data"]
y = digits["target"]

# |%%--%%| <pdNJLeWAZU|xyF31N4fwX>
r"""°°°
Let's plot the image of a particular digit to visualize it.  Before plotting, we need to reshape the 64 numbers into 8 x 8 image arrays so that it can be plotted.
°°°"""
# |%%--%%| <xyF31N4fwX|cqW7Yh9XMe>

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

# |%%--%%| <cqW7Yh9XMe|5mu5dCbDuo>
r"""°°°
**Exercise 6**

Split the data into train and test set, and randomly shuffle the data.


<details><summary>Click here for answer</summary>

```python
    
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)

```
</details>
°°°"""
# |%%--%%| <5mu5dCbDuo|L26c3Jof5Y>

## Complete your code here
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, shuffle=True, random_state=42
)


# |%%--%%| <L26c3Jof5Y|h0soBxLmBk>
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
# |%%--%%| <h0soBxLmBk|sunOYVjpfs>

# Complete the code here

lr_clf = LogisticRegression(solver="liblinear", random_state=42)
lr_clf.fit(X_train, y_train)

# |%%--%%| <sunOYVjpfs|LffBKGTJFx>
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
# |%%--%%| <LffBKGTJFx|4r7e78b82O>

# complete the code here

some_digit_scores = lr_clf.decision_function([some_digit])

# |%%--%%| <4r7e78b82O|y3vZ2Cjo1d>

some_digit_scores

# |%%--%%| <y3vZ2Cjo1d|PXYMTqv3hg>
r"""°°°
The highest score is the one corresponding to the correct class.
°°°"""
# |%%--%%| <PXYMTqv3hg|DAn3A4M82q>

index = np.argmax(some_digit_scores)
print(index)

# |%%--%%| <DAn3A4M82q|XVF4o9uxuR>

lr_clf.classes_[index]

# |%%--%%| <XVF4o9uxuR|khKxCEquc1>
r"""°°°
**Exercise 9**

Use `cross_val_score()` to evaluate the classifier's accuracy.

<details><summary>Click here for answer</summary>
    
```python 
    
cross_val_score(lr_clf, X_train, y_train, cv=3, scoring="accuracy")
    
```
</details>  
°°°"""
# |%%--%%| <khKxCEquc1|xWO4Y73tkb>

# Complete your code here


cross_val_score(lr_clf, X_train, y_train, cv=5, scoring="accuracy").mean()

# |%%--%%| <xWO4Y73tkb|EkhU9dla3a>
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
# |%%--%%| <EkhU9dla3a|zkCh8K7bz9>

# Complete your code here
y_train_pred = cross_val_predict(lr_clf, X_train, y_train, cv=5)
cm = confusion_matrix(y_train, y_train_pred)
ConfusionMatrixDisplay(cm).plot()


# |%%--%%| <zkCh8K7bz9|M1rziceCt2>
r"""°°°
**Exercise 11**

Print out the classification_report.  

<details><summary>Click here for answer</summary>
    
```python 

print(classification_report(y_train, y_train_pred))
    
```
</details>  
°°°"""
# |%%--%%| <M1rziceCt2|iwpCVzRChy>

# Complete your code here
print(classification_report(y_train, y_train_pred))

# |%%--%%| <iwpCVzRChy|fsU2P10SiS>
r"""°°°
Question: Any algorithm only suitable for Binary classification? Any algorithm only suitable for multiclass classification?
Let try out others classification algorithm and compare the result.
°°°"""
# |%%--%%| <fsU2P10SiS|WZNsVF1in6>
