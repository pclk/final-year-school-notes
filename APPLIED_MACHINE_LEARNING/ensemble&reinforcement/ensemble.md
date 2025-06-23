# deck: AML Ensemble Learning

## Introduction

A group of such models & predictors is called an -ensemble-, while the individual predictors of the ensemble are called -base predictors-.

An -ensemble method- constructs a set of base predictors from the -training data- and makes prediction by -aggregating the predictions- of individual base predictors when given unknown data.

The goal of ensemble methods is to combine the predictions of several base estimators in order to improve -generalizability- and -robustness- over a single estimator.

If done right, the base predictors collectively can get a -better prediction accuracy- than any single predictor in the ensemble.

Ensemble methods have very successful track records in breaking performance barriers on challenging datasets, among the top winners of many prestigious -machine learning competitions-, such as -Kaggle- and -Netflix competitions-

## Outperforming a single classifier

An ensemble will have a -lower error rate- than their individual base estimators, if the base classifiers do -better than random-, and the -errors are uncorrelated-.

## Bias & Variance

The base estimators should have -higher bias-.

-Ensemble- could reduce -bias- and -variance- by aggregation effect.

Ensemble have -comparable bias- but -smaller variance- than a single base estimator.

## Ensemble methods

-Homogeneous ensemble-: -Bagging- and -Boosting- use the same learning algorithm to produce homogeneous base learners, i.e. learners of the same type

-Heterogeneous ensemble-: -Stacking- uses the base learners with different learning algorithms

-Bagging- means to -parallel- build -base estimators- based on -subset data-.

-Bagging- means to aggregate their combined predictions by -voting- or -averaging-

-Bagging- example is -Random Forest-

-Boosting- means to build base estimators -sequentially- by gradually -adjust parameter weights-

-Boosting- means to reduce the -bias- of the combined estimators

-Boosting- example is -AdaBoost-, -Gradient Boosting-

-Stacking- means to build a set of -base-level estimators- independently.

-Stacking- means to train a -meta-level classifier- to combine the outputs of the based-level classifiers

## Bagging

Also known as -Bootstrap Aggregating-

Method is create -separate data sample sets- from the -training dataset-, create a -classifier- for each data sample set, -ensemble- all these multiple classifiers, aggregate the final result by -combination mean-, such as -averaging- or -majority voting-.

Use the -same training algorithm- for every base predictor, but train them on -different random subsets- of the training data.

Bagging is a -parallel ensemble method-, because -base predictors- are trained -independently- of each other.

Once trained, base predictors perform prediction in -parallel-, training can be -parallelized- and therefore quite -scalable-.

-Bootstrap- means to -randomly sample- (with -replacement-) from the -training set- according to -uniform probability distribution- to train on each base estimators.

The -uniform probability distribution- states that each sample in the training set has an -equal chance- to be drawn.

-Aggregating- in bootstraps means to combine -base estimators' predictions- by -voting- or -averaging-.

### Aggregating

For -regressors-, aggregate by -averaging-.

For -classifiers-, aggregate by -voting-.

-Hard voting- means to predict the class that gets the -highest number of votes- from the base classifiers.

The class that gets the highest number of votes is also known as the -majority vote-.

-Soft voting- means to predict the class with the -average probability- across all classifiers.

-Soft voting- often produces -better accuracy- than -hard voting- as it -weights all possible classes- according to their likelihood.

### BaggingClassifier

-base_estimator-, default=-DecisionTree-. The select base estimator to fit the ensemble.

-n_estimators-, default=-10-. The number of base estimators in the ensemble.

-max_samples-, default=-1.0-. The max number of samples to train each base estimator

-max_features-, default=-1.0-. The max number of features to train each base estimator

-bootstrap-, default=-True-. Whether samples are drawn with replacement.

-bootstrap_features-, default=-False-. Whether features are drawn with replacement.

-oob_score-, default=-False-. Whether to use out-of-bag samples to estimate the generalization error. Only apply for bootstrap=True.

-random_state-, default=-None-. Controls the random resampling of the original dataset.

With bagging, because sampling is done with -replacement- or -random sampling-, some instances may be drawn several times from the training set, while others may be omitted altogether.

On average, a bootstrap random subset contains approximately -63%- of the original training data

The other -37%- of the training instances that are not sampled are called -out-of-bag(oob) instances-

The -oob instances- can be used as the -validation set- to evaluate the -training accuracy- of the respective base predictor

The ensemble can be evaluated by the -average oob accuracy- (or error) of all the base predictors

### Using Scikit-Learn

#### Importing base estimators

Decision tree classifier: -from sklearn.tree import DecisionTreeClassifier-

KNN classifiers: -from sklearn.neighbors import KNeighborsClassifier-

#### Defining BaggingClassifier

DecisionTreeClassifier: -BaggingClassifier(DecisionTreeClassifier())-

#### Out-of-bag evaluation

bag_clf = BaggingClassifier(DecisionTreeClassifier())
bag_clf.fit(X_train, y_train)
-bag_clf.oob_score_-

bag_clf.oob_score represents the -average oob accuarcy- of -all base predictors-.

### Random Forest

= -Bagging- + -Decision Tree-

An ensemble of -randomized decision trees-, resulting in -greater tree diversity-

Results in -greater tree diversity-, trades a -higher bias- for a -lower variance-, and often leads to an -overall better prediction-

Each -decision tree- is trained with -random samples- of data drawn from the -training set- using the -bagging method-

The -random training samples- are typically of the -same size- for each subset

Introduces -extra randomness- into the -tree-growing process- while splitting the dataset, by searching for the -best feature- only from a -random subset- of all features

Like other -ensemble methods-, it trades a -higher bias- for a -lower variance- and often leads to an -overall better model- (in terms of -overfitting- and -generalization accuracy-)

An -ensemble learning method- used for -classification- and -regression-.

The -final decision- is based on the -majority vote- from all individually trees

#### Using Scikit-Learn

from sklearn. -ensemble- import 1.-RandomForestClassifier-
rf_clf = 1.-RandomForestClassifier-(n_estimators=500, max_leaf_nodes=16)
-rf_clf.fit(X_train, y_train)-
y_pred = -rf_clf.predict(X_test)-

## Boosting

A -sequential ensemble learning technique-

Constructs an -ensemble- by training and adding predictors -sequentially-, each trying to learn from the -prediction errors- made by its predecessor and progressively improving the ensemble

It starts with a -base classifier- that is prepared on the -training data-. A -second classifier- is then created behind it to focus on the -instances- in the training data that the -first classifier got wrong-. The process continues to add classifiers until a limit is reached in the number of -models- or -accuracy-.

Two popular boosting algorithms are -AdaBoost- and -Gradient Boosting-

### AdaBoost

Also known as -Adaptive Boosting-.

The -base learners- in AdaBoost are usually -decision stumps-

A -decision stump- is a very short -decision tree- with only a -single split-.

Is an -iterative algorithm-.

Each iteration is called a -boosting round-.

Trains -base classifiers- using -random subsets- of instances drawn from the -training set-

Instead of using -uniform probability distribution- for sampling, AdaBoost uses -adaptive probability distribution-

Each -instance- in the training set is given a -weight-. The -weight- determines the -probability- of being drawn from the training set. AdaBoost -adaptively changes- the weights at each -boosting round-

Training rows are assigned -equal weights-, -1/n-, where n is the size of the training set. The -first classifier- will be trained with a -random subset- of data. Rows classified -incorrectly- will have -increased weights-, while classified -correctly- will have -decreased weights-. Subsequent classifiers will focus more on training instances -wrongly classified-.

The -ensemble- is constructed from the -sequence of classifiers- trained in the specified number of -boosting rounds-

The ensemble makes prediction on a new instance by combining the predictions of all the -base classifiers-. The predicted class is the one that has the -highest weighted votes-

#### Using Scikit-Learn

from sklearn. -ensemble- import 1.-AdaBoostClassifier-
ada_clf = 1.-AdaBoostClassifier-(DecisionTreeClassifier(max_depth=1), n_estimators=200, random_state=42)
ada_clf.fit(X_train, y_train)
y_pred = ada_clf.predict(X_test)

In Scikit-Learn, the default base estimator for the -AdaBoostClassifierclass- is the -decision stump-. You can also select other base estimators such as -SVM-.

### Gradient Boosting

Instead of reweighting the training instances at every iteration, -Gradient Boosting- fits the new predictor to the -residual errors- (loss function) made by the previous predictor.

The models built by gradient boosting algorithms are called -additive models-

Its prediction is the -sum of the predictions- of the base predictors

The idea behind gradient boosting algorithm is to -repetitively leverage- the patterns in -residual errors- and strengthen a model with -weak predictions-.

Build the -first model- with simple models and analyze its -errors- with the data. These errors signify -data points- that are difficult to fit by a simple model. Then for later models, particularly focus on these -data hard to fit- to get them right. In the end, combine all the predictors by giving some -weights- to each predictor.

The most common gradient boosting model is -Gradient Tree Boosting (GTB)- – in which base predictors are -decision tree regressors-.

Some well-known GBM are -XGBoost-, -LightGBM- and -CatBoost-.

In boosting, the base predictors are -weak learners-.

A -weak learner- is one that does only -slightly better than random guessing-

The idea of boosting is to train a number of -weak learners- sequentially and combine them together into a -strong learner-

Two key hyperparameters: -n_estimators- & -learning_rate-

-n_estimators-: Number of -boosting stages- / -iterations- to perform

Gradient boosting is fairly -robust to overfitting-, so large -n_estimators- is okay.

A large number of -weak estimators- usually results in -better performance-

-Learning rate- scales the -step length- of the gradient descent procedure. It is a -regularization technique- known as -shrinkage-

Smaller values of -learning_rate- would need larger numbers of -weak learners- to adequately fit the training data, but the ensemble will -generalize better result-.

-n_estimators- and -learning_rate- are interrelated. The usual strategy is to set -n_estimators- to a -large value- and -learning_rate- to a -small value (0.01)-, and stop training when ensemble's prediction does not improve any more (before reaching the set number of estimators)

# Using Scikit-Learn

Classifier:
from sklearn. -ensemble- import 1.-GradientBoostingClassifier-
clf = 1.-GradientBoostingClassifier-(n_estimators=100, learning_rate=0.01)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

Regressor:
from sklearn. -ensemble- import 1.-GradientBoostingRegressor-
gbr = 1.-GradientBoostingRegressor-(max_depth=3, n_estimators=100, learning_rate=0.1, random_state=42)
gbr.fit(X_train, y_train)
y_predict = gbr.predict(X_test)

Hyperparameters for tree growth:
-max_depth- (typically -2 to 8-)
-max_leaf_nodes-
-min_samples_leaf-
-min_samples_split-

### XGBoost

The main difference between -GradientBoosting- and -XGBoost- is that XGboost uses a -regularization technique- in it. In simple words, it is a -regularized form- of the existing gradient-boosting algorithm.

The general principle is we want both a -simple- and -accurate model-. The tradeoff between the two is also referred as -bias-variance tradeoff- in machine learning.

Can be used for both -regression- and -classification- tasks.

Has been designed to work with -large- and -complicated datasets-.

By adding the -regularization-, XGBoost performs -better- than a normal gradient boosting algorithm and much -faster-.

### LightGBM

-Light GBM- (by -Microsoft-) is a -fast-, -distributed-, -high-performance- gradient boosting framework based on -decision tree algorithm-.

Decision trees are grown -leaf wise- with -maximum delta loss- meaning that at a single time only -one leaf- from the whole tree will be grown.

Since the -leaf- is fixed, the -leaf-wise algorithm- has -lower loss- compared to the -level-wise algorithm-. This resulted in -high accuracy- where rarely be achieved by any of the existing boosting algorithms.

LightGBM is surprisingly very -fast-, hence the word 'Light'.

-Leaf wise splits- might lead to increase in -complexity- and may lead to -overfitting-. This could be overcome by specifying the -max-depth parameter- by limiting the splitting.

Advantages of Light GBM: -faster training-, -lower memory usage-, -better accuracy-, Able to use -large datasets-.

Faster training speed and higher efficiency: Light GBM use -histogram based algorithm- i.e it buckets -continuous feature values- into -discrete bins- which fasten the training procedure.

Lower memory usage: Replaces -continuous values- to -discrete bins- which result in -lower memory usage-.

Better accuracy than any other boosting algorithm: It produces much more -complex trees- by following -leaf wise split approach-.

Compatibility with Large Datasets: It is capable of performing equally good with -large datasets- with a significant reduction in -training time- as compared to -XGBOOST-.

If you think that there is a need for -regularization- according to your dataset, then you can definitely use -XGBoost-.

If you want to -fast training speed-, then -LightGBM- perform very well on those types of datasets.

If you need more -community support- for the algorithm then use algorithms which was developed years back.

## Stacking

-Stacking- is a -Heterogeneous ensemble-, its base learners are trained with -different learning algorithms-, such as -KNN-, -Decision Tree-, -Logistic Regression-, -SVM-, etc.

Train a -meta-level classifier- to combine the outputs of the -based-level classifiers-

-Meta classifier- is a classifier that makes a -final prediction- among all the predictions by using those predictions as -features-.

### Using Scikit-Learn
from sklearn. -ensemble- import 1.-VotingClassifier-, 2.-StackingClassifier-
sclf = 1.-VotingClassifier-([estimators list])
-sclf.fit(X_train, y_train)-
y_predict = sclf.predict(X_test)
sclf = 2.-StackingClassifier-(classifiers=[clf1, clf2, clf3], meta_classifier=lr)
