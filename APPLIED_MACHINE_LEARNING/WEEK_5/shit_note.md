Ensemble Model
Dr. Veronica Lim
Learning Outcomes
▰ Upon completion of this session, the learners should be able to:
▰ Articulate the fundamental ideas underlying ensemble learning
▰ Demonstrate an intuitive understanding of ensemble methods by describing:
▻ Bagging and Boosting ensembles and their differences
▻ Random Forest as a bagging ensemble method
▻ AdaBoost and Gradient Boosting ensembles in a stagewise sequential
fashion
▻ XGBoost and LightGSM
1
Introduction
▰ Given a machine learning problem, we would usually experiment with several
different models, then evaluate and compare their performance, and adopt the model
with the best performance.
▰ Instead of a single model, a powerful way to further improve prediction accuracy is to
combine the predictions from multiple models to take the advantages of their relative
strengths.
▰ A group of such models & predictors is called an ensemble, while the individual
predictors of the ensemble are called base predictors.
▰ An ensemble method constructs a set of base predictors from the training data and
makes prediction by aggregating the predictions of individual base predictors when
given unknown data.
2
Ensemble- Gradient Boosting
3
Introduction (cont.)
▰ The goal of ensemble methods is to combine the predictions of several base
estimators in order to improve generalizability and robustness over a single
estimator.
▰ If done right, the base predictors collectively can get a better prediction accuracy
than any single predictor in the ensemble.
▰ Ensemble methods have very successful track records in breaking performance
barriers on challenging datasets, among the top winners of many prestigious
machine learning competitions, such as Kaggle and Netflix competitions
▰ Ensemble methods are widely adopted in industry. (e.g. Microsoft Kinect using
Random Forests [Reference 6])
4
How ensemble outperform a single classifier
▰ Suppose we have an ensemble of 25 binary classifiers
▰ Each base classifier has an error rate 𝜖𝜖 = 0.35
▰ To make prediction on a new instance, the ensemble obtains the
predictions of all individual base classifiers and predicts the
class that gets the most votes
▰ If the base classifiers are identical, the error rate of the ensemble
will be the same as the base classifiers, i.e. 0.35
▰ If the base classifiers are independent and their errors are
therefore uncorrelated, the error rate of the ensemble will be
reduced to 0.06, better than individual classifier
5
Two necessary conditions for ensemble to outperform individual classifiers:
• The base classifiers should do better than mere random guessing
• The base classifiers should be independent of each other (and their errors are uncorrelated)
[Reference 7]
Ensemble’s performance with Bias and
Variance
▰ The individual base predictors are trained on a subset of the training set, each has a
higher bias than if it was trained on the full training set.
▰ An ensemble could reduce both the bias and variance by aggregation effect.
▰ Overall, the ensemble will have a comparable bias but a smaller variance than a
single predictor trained on the original training set.
6
• larger bias
• small variance
• least bias
• large variance
*Bias: refer to the error during training phase *Variance: refer to the error during prediction phase
Main Ensemble Methods
Ensemble Methods
Bagging Boosting
• to build base estimators parallelly
based on subset data.
• to aggregate their combined
predictions by voting or averaging
• such as Random Forest
• to build base estimators sequentially
by gradually adjust parameter weights
• to reduce the bias of the combined
estimators
• such as AdaBoost, Gradient Boosting
Stacking
• to build a set of base-level
estimators independently.
• to train a meta-level
classifier to combine the
outputs of the based-level
classifiers
*Homogeneous ensemble: Bagging and Boosting use the same learning algorithm to produce homogeneous base learners,
i.e. learners of the same type
*Heterogeneous ensemble: Stacking uses the base learners with different learning algorithms
7
Bagging (Bootstrap Aggregating)
Method
▰ Create separate data sample
sets from the training dataset
▰ Create a classifier for each
data sample set
▰ Ensemble all these multiple
classifiers
▰ Aggregate the final result by
combination mean, such as
averaging or majority voting
8
Bootstrap
(data subsets)
Original data
Classifiers
(with same
model)
Aggregating
(by voting or
averaging )
Classifier 1 Classifier 2 Classifier n
Prediction
Bootstrap and Aggregating
9
• Use the same training algorithm for every base predictor, but train them on different
random subsets of the training data
• Bagging is a parallel ensemble method
• Base predictors are trained independently of each other
• Training can be parallelized and therefore quite scalable
• Once trained, base predictors perform prediction in parallel
Bootstrap
The training data for each base predictor is
randomly sampled (with replacement) from the
training set according to a uniform probability
distribution (i.e., each sample in the training set
has an equal chance to be drawn)
Aggregating
The most common way of aggregating the
predictions made by base predictors is by voting
(for classifiers) or averaging (for regressors)
Bootstrap (cont.)
sepal_length sepal_width petal_length petal_width species
5.1 3.5 1.4 0.2 Iris-setosa
4.9 3 1.4 0.2 Iris-setosa
4.7 3.2 1.3 0.2 Iris-setosa
4.6 3.1 1.5 0.2 Iris-setosa
5 3.6 1.4 0.2 Iris-setosa
5.4 3.9 1.7 0.4 Iris-setosa
6.9 3.1 4.9 1.5 Iris-versicolor
5.5 2.3 4 1.3 Iris-versicolor
5.8 4 1.2 0.2 Iris-setosa
5.7 2.8 4.5 1.3 Iris-versicolor
5.4 3.9 1.3 0.4 Iris-setosa
4.9 2.4 3.3 1 Iris-versicolor
6.6 2.9 4.6 1.3 Iris-versicolor
5.2 2.7 3.9 1.4 Iris-versicolor
6.2 2.8 4.8 1.8 Iris-virginica
5.8 2.7 5.1 1.9 Iris-virginica
7.1 3 5.9 2.1 Iris-virginica
6.3 2.9 5.6 1.8 Iris-virginica
6.5 3 5.8 2.2 Iris-virginica
7.6 3 6.6 2.1 Iris-virginica
sepal_length sepal_width petal_length petal_width species
5.1 3.5 1.4 0.2 Iris-setosa
4.9 3 1.4 0.2 Iris-setosa
6.2 2.8 4.8 1.8 Iris-virginica
5.7 2.8 4.5 1.3 Iris-versicolor
5.1 3.5 1.4 0.2 Iris-setosa
5.4 3.9 1.7 0.4 Iris-setosa
6.2 2.8 4.8 1.8 Iris-virginica
5.5 2.3 4 1.3 Iris-versicolor
5.5 2.3 4 1.3 Iris-versicolor
6.3 2.9 5.6 1.8 Iris-virginica
Training instance
not selected
Training instance
randomly selected
Training instance
randomly replaced
Original training set
Bootstrap random subset
Features
Samples/instances
10
Aggregating
▰ Ways of aggregating the predictions of base classifiers by …
▻ Hard voting
▻ Predict the class that gets the highest number of votes (the majority vote) from the base classifiers
▻ Winner-takes-all
▻ Soft voting
▻ Predict the class with the average probability across all classifiers
▻ Often produces better accuracy than hard voting as it weights all possible classes according to their likelihood
Classifier 1 Classifier 2 Classifier 3
Class A 0.90 0.27 0.40
Class B 0 0.63 0.53
Class C 0.10 0.10 0.07
Hard Voting Soft Voting
1 0.52 (A)
2 (B) 0.39
0 0.09
11
Key Parameters of BaggingvClassifier
• base_estimator, default=DecisionTree
The select base estimator to fit the ensemble.
• n_estimators, default=10
The number of base estimators in the ensemble.
• max_samples, default=1.0
The max number of samples to train each base estimator
• max_features, default=1.0
The max number of features to train each base estimator
• bootstrap, default=True
Whether samples are drawn with replacement.
• bootstrap_features, default=False
Whether features are drawn with replacement.
• oob_score, default=False
Whether to use out-of-bag samples to estimate the generalization error. Only apply for bootstrap=True.
• random_state, default=None
Controls the random resampling of the original dataset (sample wise and feature wise). 12
Out-of-bag (oob) Evaluation
• With bagging, because sampling is done with
replacement or random sampling, some instances
may be drawn several times from the training set,
while others may be omitted altogether.
• On average, a bootstrap random subset contains
approximately 63% of the original training data
• The other 37% of the training instances that are not
sampled are called out - of - bag(oob) instances
• The oob instances can be used as the validation set
to evaluate the training accuracy of the respective
base predictor
• The ensemble can be evaluated by the average oob
accuracy (or error) of all the base predictors
• Each bootstrap random subset for
training has the same size as the
original training set (say, 𝑚𝑚)
• Each training instance has a probability
of 1− 1− ⁄ 1
𝑚𝑚
𝑚𝑚 of being selected in
each bootstrap random subset
• If 𝑚𝑚 is sufficiently large, this probability
converges to 1− ⁄ 1
𝑒𝑒 ≅ 𝟎𝟎. 𝟔𝟔𝟔𝟔𝟔𝟔
• Each bootstrap random subset
therefore contains approximately 2/3 of
the training data
13
Bagging Classifier using Scikit-Learn
A bagging ensemble of decision tree classifiers:
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, max_samples=100)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
A bagging ensemble of KNN classifiers:
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
bagging.fit(X_train,y_train)
bagging.score(X_test,y_test)
A bagging classifier with out-of-bag evaluation:
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, bootstrap=True, oob_score=True)
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_ #the average oob accuracy of all base predictors
14
Random Forest
Random Forests Bagging Decision Tree
• It results in greater tree diversity
• It trades a higher bias for a lower variance
• It often leads to an overall better prediction
15
Random Forest
16
▰ A random forest is an ensemble of randomized decision trees, resulting in greater
tree diversity
▰ Each decision tree is trained with random samples of data drawn from the training
set using the bagging method
▰ The random training samples are typically of the same size for each subset
▰ The random forest algorithm introduces extra randomness into the tree-growing
process while splitting the dataset, by searching for the best feature only from a
random subset of all features
▰ Like other ensemble methods, it trades a higher bias for a lower variance and
often leads to an overall better model (in terms of overfitting and generalization
accuracy)
Random Forest Model
▰ An ensemble learning method used for classification and regression.
▰ Build a set of decision trees. Each tree is developed from a bootstrap subset.
▰ The final decision is based on the majority vote from all individually trees
17
RandomForestClassifier using Scikit-Learn
A random forest classifier with 500 decision trees:
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)

---

Boosting
19
Boosting
▰ In the Bagging ensemble methods, training is performed in parallel
▰ In contrast, boosting is a sequential ensemble learning technique
▰ Boosting constructs an ensemble by training and adding predictors sequentially, each trying to learn from the prediction errors made by its predecessor and progressively improving the ensemble
▻ It starts with a base classifier that is prepared on the training data
▻ A second classifier is then created behind it to focus on the instances in the
training data that the first classifier got wrong.
▻ The process continues to add classifiers until a limit is reached in the number of models or accuracy.
20
Two popular boosting algorithms
AdaBoost
(Adaptive Boosting)
Gradient Boosting
21
AdaBoost
• The base learners in
AdaBoost are usually
decision stumps
A decision stump is a very
short decision tree with only
a single split.
Classifier 1 Classifier 2 Classifier 3
Final ensemble – Strong classifier
majority
22
AdaBoost
▰ AdaBoost is an iterative algorithm (each iteration is called a boosting round)
▰ AdaBoost trains base classifiers using random subsets of instances drawn
from the training set
▰ Instead of using uniform probability distribution for sampling, AdaBoost uses
adaptive probability distribution
▻ Each instance in the training set is given a weight
▻ The weight determines the probability of being drawn from the training
set
▻ AdaBoost adaptively changes the weights at each boosting round
23
More details in AdaBoost
▰ Initially, the training instances are assigned equal weights, ⁄ 1 𝑛𝑛, where 𝑛𝑛 is the size of the training set
▰ The first classifier is trained using a random subset drawn from the training set according to the sampling distribution
▰ The weights of the training instances classified incorrectly will be increased, while the classified correctly decreased
▰ This causes the classifiers trained in the subsequent rounds to focus more on training instances wrongly classified
▰ The ensemble is constructed from the sequence of the classifiers trained in the specified number of boosting rounds
▰ The ensemble makes prediction on a new instance by combining the predictions of all the base classifiers. The predicted
class is the one that has the highest weighted votes
24
AdaBoostClassifier using Scikit-Learn
An AdaBoost classifier with 200 decision stumps:
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(
DecisionTreeClassifier(max_depth=1), n_estimators=200,
random_state=42)
ada_clf.fit(X_train, y_train)
y_pred = ada_clf.predict(X_test)
*In Scikit-Learn, the default base estimator for the AdaBoostClassifierclass is the decision stump. You
can also select other base estimators such as SVM.
25
Gradient Boosting
▰ Just like AdaBoost, Gradient Boosting works sequentially by adding predictors to an ensemble, each one
correcting its predecessor.
▰ It is an ensemble boosting method that "boosting" many weak models into a strong one.
▰ Instead of reweighting the training instances at every iteration, Gradient Boosting fits the new predictor to the
residual errors (loss function) made by the previous predictor.
▰ The models built by gradient boosting algorithms are called additive models
▰ It is built stage by stage by adding one new base predicator
▰ Its prediction is the sum of the predictions of the base predictors
26
Gradient Boosting
The ideas behind gradient boosting algorithm is to repetitively
leverage the patterns in residual errors and strengthen a model
with weak predictions. It consist of the following steps:
• Build the first model with simple models and analyze its errors with the
data.
• These errors signify data points that are difficult to fit by a simple model.
• Then for later models, particularly focus on these data hard to fit to get
them right.
• In the end, combine all the predictors by giving some weights to each
predictor.
The most common gradient boosting model is Gradient Tree
Boosting (GTB) – in which base predictors are decision tree
regressors. Some well-known GBM are XGBoost, LightGBM and
CatBoost.
Weak learners
• In boosting, the base
predictors are “weak
learners”
• A weak learner is one that
does only slightly better
than random guessing
• The idea of boosting is to
train a number of weak
learners sequentially and
combine them together
into a strong learner
27
Training of Gradient Boosting Estimators
Source: Reference [1]
y: true target y:prediction result^
28
Key Hyperparameters
Two key hyperparameters among others:
• Number of boosting stages / iterations to perform (n_estimators)
o Gradient boosting is fairly robust to overfitting
o A large number of (weak) estimators usually results in better
performance
• Learning rate (learning_rate)
o Learning rate (𝜂𝜂) scales the step length of the gradient descent
procedure. It is a regularization technique known as shrinkage
o Smaller values of learning_rate would need larger numbers of
weak learners to adequately fit the training data, but the ensemble
will generalize better result.
n_estimators and learning_rate are interrelated. The usual strategy is to
set n_estimators to a large value and learning_rate to a small value (0.01), and
stop training when ensemble’s prediction does not improve any more (before
reaching the set number of estimators)
Underfitting
Overfitting
29
GradientBoostingClassifier using Scikit-Learn
Hyperparameters for ensemble
training:
n_estimators
learning_rate
Hyperparameters for tree growth:
max_depth (typically 2 to 8)
max_leaf_nodes
min_samples_leaf
min_samples_split
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
# Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(max_depth=3, n_estimators=100,
learning_rate=0.1, random_state=42)
gbr.fit(X_train, y_train)
y_predict = gbr.predict(X_test)
30
XGBoost
▰ The main difference between GradientBoosting and XGBoost is that XGbost
uses a regularization technique in it. In simple words, it is a regularized form
of the existing gradient-boosting algorithm.
31
XGBoost
▰ Let fit visually a step function given the input data points on the upper left corner of the image. Which solution among the three do you think is the best fit?
▰ The correct answer is marked in red. The
general principle is we want both
a simple and predictive model. The
tradeoff between the two is also referred
as bias-variance tradeoff in machine
learning.
▰ By add in the regularization, it helps to controls the complexity of the model, which helps us to avoid overfitting.
32
XGBoost
▰ The algorithm can be used for both regression and classification tasks and
has been designed to work with large and complicated datasets.
▰ By adding the regularization, XGBoost performs better than a normal
gradient boosting algorithm and much faster.
33
LightGBM
▰ Light GBM (by Microsoft) is a fast, distributed, high-performance
gradient boosting framework based on decision tree algorithm.
▰ In LightGBM decision trees are grown leaf wise with maximum delta loss meaning
that at a single time only one leaf from the whole tree will be grown.
34
LightGBM
▰ Since the leaf is fixed, the leaf-wise algorithm has lower loss compared
to the level-wise algorithm. This resulted in high accuracy where rarely be
achieved by any of the existing boosting algorithms.
▰ LightGBM is surprisingly very fast, hence the word ‘Light’.
▰ However, leaf wise splits might lead to increase in complexity and may lead
to overfitting. This could be overcome by specifying the max-depth
parameter by limiting the splitting.
35
Advantages of Light GBM
1. Faster training speed and higher efficiency: Light GBM use histogram based
algorithm i.e it buckets continuous feature values into discrete bins which fasten
the training procedure.
2. Lower memory usage: Replaces continuous values to discrete bins which result in
lower memory usage.
3. Better accuracy than any other boosting algorithm: It produces much more
complex trees by following leaf wise split approach rather than a level-wise
approach which is the main factor in achieving higher accuracy. However, it can
sometimes lead to overfitting which can be avoided by setting the max_depth
parameter.
4. Compatibility with Large Datasets: It is capable of performing equally good with
large datasets with a significant reduction in training time as compared to
XGBOOST.
36
When to use XGBoost versus LightGBM?
▰ The answer to these questions can not be a single boosting algorithm
among them as all of them are the best fit solution for a particular type of
problem on which we are working.
▰ For example, If you think that there is a need for regularization according to
your dataset, then you can definitely use XGBoost. If you want to fast
training speed, then LightGBM perform very well on those types of datasets.
If you need more community support for the algorithm then use algorithms
which was developed years back.
37
Stacking
▰ Stacking is a Heterogeneous ensemble, its base learners are trained with different learning algorithms, such as
KNN, Decision Tree, Logistic Regression, SVM, etc.
▰ Train a meta-level classifier to combine the outputs of the based-level classifiers
▰ Meta classifier is a classifier that makes a final prediction among all the predictions by using those predictions as
features.
38
#Stacking using Scikit-Learn
from sklearn.ensemble import VotingClassifier
sclf = VotingClassifier([estimators list])
sclf.fit(X_train, y_train)
y_predict = sclf.predict(X_test)
sclf = StackingClassifier(classifiers=[clf1, clf2,
clf3], meta_classifier=lr)
Summary - Ensemble Methods
Bagging Boosting Stacking
Random Forests AdaBoost Gradient
Boosting
• to build base estimators parallelly
based on subset data.
• to aggregate their combined
predictions by voting or averaging
• to build base estimators sequentially
by gradually adjust parameter weights
• to reduce the bias of the combined
estimators
• to build a set of base-level
estimators independently.
• to train a meta-level
classifier to combine the
outputs
