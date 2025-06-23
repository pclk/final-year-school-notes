r"""°°°
# Lab01: Ensemble Methods using IrisData
°°°"""
# |%%--%%| <ovATx54hF0|ipBVcuqfr2>
r"""°°°
* to practice different ensemble classifiers with Iris dataset, bagging and boosting
* to build and evaluate Random Forest classifier
°°°"""
# |%%--%%| <ipBVcuqfr2|NVovxsy3Xy>

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree
from sklearn.datasets import make_blobs
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report, confusion_matrix

# Suppress warnings about too few trees from the early models
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# |%%--%%| <NVovxsy3Xy|nzhKPloZWw>
r"""°°°
# 1. Prepare Data
°°°"""
# |%%--%%| <nzhKPloZWw|QHHVeh3D25>

iris = datasets.load_iris()
iris.target_names

# |%%--%%| <QHHVeh3D25|acNV4vKKtH>

import pandas as pd   
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()

# |%%--%%| <acNV4vKKtH|X7YdxQ1NCz>

X = iris.data
y = iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=100, shuffle=True)

# |%%--%%| <X7YdxQ1NCz|uIbrxOLcZQ>
r"""°°°
# 2. Decision Tree Classifer
#### First, train a Decision Tree Classifier for the comparision with ensemble models
°°°"""
# |%%--%%| <uIbrxOLcZQ|6MVilZyVBo>

#define DecisionTree model
# ?tree.DecisionTreeClassifier
clf = tree.DecisionTreeClassifier()

#train the model with train data set
clf.fit(X_train,y_train)

# |%%--%%| <6MVilZyVBo|peezBqYPKL>

#check model accracy in terms of test dataset
acc = clf.score(X_test, y_test)
print("Accuracy=", acc)

# |%%--%%| <peezBqYPKL|ed1S3hKsyQ>

#check feature importance
print(iris.feature_names)
print(clf.feature_importances_)

# |%%--%%| <ed1S3hKsyQ|TlaPBj5UTL>

# Visualize the Decision Tree
plt.figure(figsize=(25,20))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names,filled=True)
#plt.savefig("decistion_tree.png")
plt.show()

# |%%--%%| <TlaPBj5UTL|Gi7U0Ud3OK>

#check model accuracy
y_test_pred = clf.predict(X_test)
print(classification_report(y_test, y_test_pred))

#plot a confusion matrix using seaborn
cm = confusion_matrix(y_test, y_test_pred)
sns.set_context('talk')
ax = sns.heatmap(cm, annot=True, fmt='d')
ax.set_xticklabels(iris.target_names);
ax.set_yticklabels(iris.target_names);
ax.set_ylabel('Actual');
ax.set_xlabel('Predicted');

# |%%--%%| <Gi7U0Ud3OK|tlxkkwNZXU>
r"""°°°
# Answer Question

What is  macro avg? What is weighted avg?
°°°"""
# |%%--%%| <tlxkkwNZXU|Z9mkkjUQuN>

from sklearn.metrics import classification_report

classification_report(y_test, y_test_pred)

# macro is simple average, total/number of classes
# weighted takes class sizes into calculation

# |%%--%%| <Z9mkkjUQuN|Rm5GW0Wj0r>
r"""°°°
# 3. Bagging Classifier
°°°"""
# |%%--%%| <Rm5GW0Wj0r|qRMqVVche7>
r"""°°°
* to build base estimators parallelly based on subset data.
* to aggregate their combined predictions by voting or averaging
°°°"""
# |%%--%%| <qRMqVVche7|Nb9hecnjPV>

from sklearn.ensemble import BaggingClassifier
?BaggingClassifier


# |%%--%%| <Nb9hecnjPV|M6j29ufegh>
r"""°°°
# Answer Questions:

What is the default base model of BaggingClassifier?

What is the default number of estimators?
°°°"""
# |%%--%%| <M6j29ufegh|9eKX0qRE77>

bag_clf = BaggingClassifier()
# bag_clf = BaggingClassifier(DecisionTreeClassifier())
# bag_clf = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=0.5)

bag_clf.fit(X_train, y_train)

print('Accuracy: ', bag_clf.score(X_test,y_test) )
print('Total number of trees:', bag_clf.n_estimators)

for tree in bag_clf.estimators_:
    plot_tree(tree, filled=True)
    plt.show()   

# |%%--%%| <9eKX0qRE77|vf3EJ1YTX7>
r"""°°°
# Answer Question
Why there are so many graph above? Not 1?
°°°"""
#|%%--%%| <vf3EJ1YTX7|Xq7g7XMlF6>
r"""°°°
because you did a for loop lmao
°°°"""
# |%%--%%| <Xq7g7XMlF6|QJdPdoUdva>
r"""°°°
# Try your self:

Create a BaggingClassifier with 

base model:  DecisionTree
number of estimators: 20
max_samples=0.2

Compare the number of tree nodes of each estimator with the default parameters
°°°"""
# |%%--%%| <QJdPdoUdva|cyujgYFUG1>
r"""°°°

°°°"""
# |%%--%%| <cyujgYFUG1|r9BFcBJbB2>

bag_clf = BaggingClassifier(n_estimators=20, max_samples=0.2)
bag_clf.fit(X_train, y_train)

print('Accuracy: ', bag_clf.score(X_test,y_test) )
print('Total number of trees:', bag_clf.n_estimators)

for tree in bag_clf.estimators_:
    plot_tree(tree, filled=True)
    plt.show() 

# |%%--%%| <r9BFcBJbB2|itJJ5Q19bU>
r"""°°°
# 4.  AdaBoost classifier
°°°"""
# |%%--%%| <itJJ5Q19bU|pj3tcT4myW>
r"""°°°
* AdaBoost is an iterative algorithm (each iteration is called a boosting round)
* AdaBoost trains base classifiers using random subsets of instances drawn from the training set
* AdaBoost uses adaptive probability distribution. Each instance in the training set is given a weight. The weight determines the probability of being drawn from the training set. AdaBoost adaptively changes the weights at each boosting round
°°°"""
# |%%--%%| <pj3tcT4myW|TZhF58JbSv>

from sklearn.ensemble import AdaBoostClassifier
?AdaBoostClassifier


# |%%--%%| <TZhF58JbSv|YFiDOkdQ77>
r"""°°°
# Question: 

What is the default number of base models in AdaBoostClassifier?

How many tree nodes in each base estimator?
°°°"""
# |%%--%%| <YFiDOkdQ77|xYDcYu0zeH>

ada_clf = AdaBoostClassifier()
# ada_clf = AdaBoostClassifier(DecisionTreeClassifier())
ada_clf.fit(X_train, y_train)
acc = ada_clf.score(X_test,y_test)  

print('Accracy: ', acc)
print('Total number of trees:', ada_clf.n_estimators)

count = 0
for tree in ada_clf.estimators_:
    plot_tree(tree, filled=True)
    plt.show() 
    count += 1
    if count==10: break   



# |%%--%%| <xYDcYu0zeH|t8WRHzyj8y>
r"""°°°
A decision stump is a very short decision tree with only a single split 
°°°"""
# |%%--%%| <t8WRHzyj8y|v2S9ybEOuC>
r"""°°°
# 5. Gradient Boosting classifier
°°°"""
# |%%--%%| <v2S9ybEOuC|bXHRMBP8kD>
r"""°°°
* Gradient Boosting works sequentially by adding predictors to an ensemble, each one correcting its predecessor.
* Instead of reweighting the training instances at every iteration, Gradient Boosting fits the new predictor to the residual errors (loss function) made by the previous predictor.
°°°"""
# |%%--%%| <bXHRMBP8kD|kx0KccGxRn>

from sklearn.ensemble import GradientBoostingClassifier
?GradientBoostingClassifier



# |%%--%%| <kx0KccGxRn|kF1kHNZL8I>
r"""°°°
# Questions: 

What is the loss function applied in GradientBoostingClassifier?

What is the loss function applied in AdaBoostClassifier?
°°°"""
# |%%--%%| <kF1kHNZL8I|TTVIr3fD17>

gr_clf = GradientBoostingClassifier()  #with default parameters
gr_clf.fit(X_train, y_train)

print('Accracy: ', gr_clf.score(X_test,y_test) )
print('Total number of estimator:', gr_clf.n_estimators)

# print(gr_clf.estimators_[0])

# |%%--%%| <TTVIr3fD17|wLptemaqw7>
r"""°°°
# Try your Self: 

Plot the GradientBoostingClassifier map
°°°"""
# |%%--%%| <wLptemaqw7|KDtEeGGMzG>
r"""°°°

°°°"""
# |%%--%%| <KDtEeGGMzG|AjTjzs4Kxb>

count = 1
for e in gr_clf:
  if count>=20:  break
  for clf in e:
    print('Base estimator No.',count) 
    count += 1    
    plot_tree(clf, filled=True) 
    plt.show()

# |%%--%%| <AjTjzs4Kxb|b21wxLTKI4>
r"""°°°
# 6. Random Forest
°°°"""
# |%%--%%| <b21wxLTKI4|dp4paHlFsL>
r"""°°°
* A random forest is an Bagging ensemble of randomized decision trees, resulting in greater tree diversity
* Each decision tree is trained with random samples of data drawn from the training set using the bagging method
°°°"""
# |%%--%%| <dp4paHlFsL|bfULZ89286>

# Random Forest Classifier Using default parameters
from sklearn.ensemble import RandomForestClassifier

?RandomForestClassifier


# |%%--%%| <bfULZ89286|lIFRiArkJl>
r"""°°°
# Questions:

What is the default number of sub-trees in RandomForestClassifier?

what id the max_depth of each sub-tree?
°°°"""
# |%%--%%| <lIFRiArkJl|s0y4Nm8SqO>

import time
clf = RandomForestClassifier()   #with default parameters
# clf = RandomForestClassifier(n_estimators=50)

start=time.time()
clf.fit(X_train,y_train)
acc = clf.score(X_test,y_test) 
print('Accuracy:', acc)
print('Number of trees:', clf.n_estimators)

end=time.time()
diff=end-start
print("Execution time:",diff)

# |%%--%%| <s0y4Nm8SqO|TCXg578GDU>

# Evaluate the model by cross-validation (Self-directed learning)
from sklearn.model_selection import cross_val_score
?cross_val_score 

scores = cross_val_score(clf, X, y, cv=10)
print('Scores', scores)

print(f'Count: {len(scores)}, Mean: {scores.mean():0.2f}, Stdev: {scores.std():0.5f}')

# |%%--%%| <TCXg578GDU|D670BppUPX>
r"""°°°
# 7. Plot Tree
°°°"""
# |%%--%%| <D670BppUPX|0tImoEMaUC>

#plot trees
print('The total number of trees are', len(clf.estimators_))

count = 0
for tree_in_forest in clf.estimators_:
  if count<10:
    plot_tree(tree_in_forest, filled=True)
    plt.show()
    count += 1

# |%%--%%| <0tImoEMaUC|008VvZdR1E>

# Optimization of Parameters: number of trees
clf = RandomForestClassifier(max_depth=5)
acc_list = []

# Iterate through a range of numbers of trees
for n_trees in range(2,10):    
    # Use this to set the number of trees
    clf.set_params(n_estimators=n_trees)
    clf.fit(X_train, y_train)
    
    acc = clf.score(X_test,y_test)    
    acc_list.append(pd.Series({'n_trees': n_trees, 'acc': acc}))

pd.concat(acc_list, axis=1).T.set_index('n_trees')

# |%%--%%| <008VvZdR1E|fGpR3fID0L>

#feature importance
feature_imp = pd.Series(clf.feature_importances_, index=iris.feature_names).sort_values(ascending=False)
fig = plt.figure()
ax = feature_imp.plot(kind='bar')
ax.set(ylabel='Relative Importance');

# |%%--%%| <fGpR3fID0L|nmCT1Zo48g>

#check model accuracy
y_test_pred = clf.predict(X_test)
print(classification_report(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred)
sns.set_context('talk')
ax = sns.heatmap(cm, annot=True, fmt='d')
ax.set_xticklabels(iris.target_names);
ax.set_yticklabels(iris.target_names);
ax.set_ylabel('Actual');
ax.set_xlabel('Predicted');

# |%%--%%| <nmCT1Zo48g|mLAraqOFhQ>
r"""°°°
# 8. XGBoost 
°°°"""
# |%%--%%| <mLAraqOFhQ|xPyNvdjfH5>

!pip install xgboost

# |%%--%%| <xPyNvdjfH5|7focSv27il>

import xgboost as xgb
import time

start=time.time()
xgbc=xgb.XGBClassifier()

xgbc.fit(X_train,y_train)
acc_xgboost = xgbc.score(X_test,y_test) 
print('Accuracy:', acc_xgboost)
print('Number of trees:', xgbc.n_estimators)

end=time.time()
diff=end-start
print("Execution time:",diff)

# |%%--%%| <7focSv27il|bcPdyBG0FS>
r"""°°°

°°°"""
# |%%--%%| <bcPdyBG0FS|ESOuJgjiNw>

!pip install lightgbm

# |%%--%%| <ESOuJgjiNw|HoVyuavbpA>

import lightgbm

start=time.time()
lgbmr=lightgbm.LGBMClassifier()

lgbmr.fit(X_train,y_train)
acc_lgbmr = lgbmr.score(X_test,y_test) 
print('Accuracy:', acc_lgbmr)
print('Number of trees:', lgbmr.n_estimators)

end=time.time()
diff=end-start
print("Execution time:",diff)

# |%%--%%| <HoVyuavbpA|ElQIPmpkrh>
r"""°°°
Which algorithm is better? Explain with Justification.
°°°"""
# |%%--%%| <ElQIPmpkrh|PQn0lCkfiH>


