# Note
model: Cloze

## Text

A group of such models & predictors is called an {{c1::ensemble}}, while the individual predictors of the ensemble are called {{c2::base predictors}}.

## Back Extra


# Note
model: Cloze

## Text

An {{c1::ensemble method}} constructs a set of base predictors from the {{c2::training data}} and makes prediction by {{c3::aggregating the predictions}} of individual base predictors when given unknown data.

## Back Extra


# Note
model: Cloze

## Text

The goal of ensemble methods is to combine the predictions of several base estimators in order to improve {{c1::generalizability}} and {{c2::robustness}} over a single estimator.

## Back Extra


# Note
model: Cloze

## Text

If done right, the base predictors collectively can get a {{c1::better prediction accuracy}} than any single predictor in the ensemble.

## Back Extra


# Note
model: Cloze

## Text

Ensemble methods have very successful track records in breaking performance barriers on challenging datasets, among the top winners of many prestigious {{c1::machine learning competitions}}, such as {{c2::Kaggle}} and {{c3::Netflix competitions}}

## Back Extra


# Note
model: Cloze

## Text

Section: Outperforming a single classifier

An ensemble will have a {{c1::lower error rate}} than their individual base estimators, if the base classifiers do {{c2::better than random}}, and the {{c3::errors are uncorrelated}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Bias & Variance

The base estimators should have {{c1::higher bias}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Bias & Variance

{{c1::Ensemble}} could reduce {{c2::bias}} and {{c3::variance}} by aggregation effect.

## Back Extra


# Note
model: Cloze

## Text

Section: Bias & Variance

Ensemble have {{c1::comparable bias}} but {{c2::smaller variance}} than a single base estimator.

## Back Extra


# Note
model: Cloze

## Text

Section: Ensemble methods

{{c1::Homogeneous ensemble}}: {{c2::Bagging}} and {{c3::Boosting}} use the same learning algorithm to produce homogeneous base learners, i.e. learners of the same type

## Back Extra


# Note
model: Cloze

## Text

Section: Ensemble methods

{{c1::Heterogeneous ensemble}}: {{c2::Stacking}} uses the base learners with different learning algorithms

## Back Extra


# Note
model: Cloze

## Text

Section: Ensemble methods

{{c1::Bagging}} means to {{c2::parallel}} build {{c3::base estimators}} based on {{c4::subset data}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Ensemble methods

{{c1::Bagging}} means to aggregate their combined predictions by {{c2::voting}} or {{c3::averaging}}

## Back Extra


# Note
model: Cloze

## Text

Section: Ensemble methods

{{c1::Bagging}} example is {{c2::Random Forest}}

## Back Extra


# Note
model: Cloze

## Text

Section: Ensemble methods

{{c1::Boosting}} means to build base estimators {{c2::sequentially}} by gradually {{c3::adjust parameter weights}}

## Back Extra


# Note
model: Cloze

## Text

Section: Ensemble methods

{{c1::Boosting}} means to reduce the {{c2::bias}} of the combined estimators

## Back Extra


# Note
model: Cloze

## Text

Section: Ensemble methods

{{c1::Boosting}} example is {{c2::AdaBoost}}, {{c3::Gradient Boosting}}

## Back Extra


# Note
model: Cloze

## Text

Section: Ensemble methods

{{c1::Stacking}} means to build a set of {{c2::base-level estimators}} independently.

## Back Extra


# Note
model: Cloze

## Text

Section: Ensemble methods

{{c1::Stacking}} means to train a {{c2::meta-level classifier}} to combine the outputs of the based-level classifiers

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Also known as {{c1::Bootstrap Aggregating}}

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Method is create {{c1::separate data sample sets}} from the {{c2::training dataset}}, create a {{c3::classifier}} for each data sample set, {{c4::ensemble}} all these multiple classifiers, aggregate the final result by {{c5::combination mean}}, such as {{c6::averaging}} or {{c7::majority voting}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Use the {{c1::same training algorithm}} for every base predictor, but train them on {{c2::different random subsets}} of the training data.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Bagging is a {{c1::parallel ensemble method}}, because {{c2::base predictors}} are trained {{c3::independently}} of each other.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Once trained, base predictors perform prediction in {{c1::parallel}}, training can be {{c2::parallelized}} and therefore quite {{c3::scalable}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

{{c1::Bootstrap}} means to {{c2::randomly sample}} (with {{c3::replacement}}) from the {{c4::training set}} according to {{c5::uniform probability distribution}} to train on each base estimators.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

The {{c1::uniform probability distribution}} states that each sample in the training set has an {{c2::equal chance}} to be drawn.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

{{c1::Aggregating}} in bootstraps means to combine {{c2::base estimators' predictions}} by {{c3::voting}} or {{c4::averaging}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Aggregating

For {{c1::regressors}}, aggregate by {{c2::averaging}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Aggregating

For {{c1::classifiers}}, aggregate by {{c2::voting}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Aggregating

{{c1::Hard voting}} means to predict the class that gets the {{c2::highest number of votes}} from the base classifiers.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Aggregating

The class that gets the highest number of votes is also known as the {{c1::majority vote}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Aggregating

{{c1::Soft voting}} means to predict the class with the {{c2::average probability}} across all classifiers.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Aggregating

{{c1::Soft voting}} often produces {{c2::better accuracy}} than {{c3::hard voting}} as it {{c4::weights all possible classes}} according to their likelihood.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: BaggingClassifier

{{c1::base_estimator}}, default={{c2::DecisionTree}}. The select base estimator to fit the ensemble.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: BaggingClassifier

{{c1::n_estimators}}, default={{c2::10}}. The number of base estimators in the ensemble.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: BaggingClassifier

{{c1::max_samples}}, default={{c2::1.0}}. The max number of samples to train each base estimator

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: BaggingClassifier

{{c1::max_features}}, default={{c2::1.0}}. The max number of features to train each base estimator

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: BaggingClassifier

{{c1::bootstrap}}, default={{c2::True}}. Whether samples are drawn with replacement.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: BaggingClassifier

{{c1::bootstrap_features}}, default={{c2::False}}. Whether features are drawn with replacement.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: BaggingClassifier

{{c1::oob_score}}, default={{c2::False}}. Whether to use out-of-bag samples to estimate the generalization error. Only apply for bootstrap=True.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: BaggingClassifier

{{c1::random_state}}, default={{c2::None}}. Controls the random resampling of the original dataset.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: BaggingClassifier

With bagging, because sampling is done with {{c1::replacement}} or {{c2::random sampling}}, some instances may be drawn several times from the training set, while others may be omitted altogether.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: BaggingClassifier

On average, a bootstrap random subset contains approximately {{c1::63%}} of the original training data

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: BaggingClassifier

The other {{c1::37%}} of the training instances that are not sampled are called {{c2::out-of-bag(oob) instances}}

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: BaggingClassifier

The {{c1::oob instances}} can be used as the {{c2::validation set}} to evaluate the {{c3::training accuracy}} of the respective base predictor

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: BaggingClassifier

The ensemble can be evaluated by the {{c1::average oob accuracy}} (or error) of all the base predictors

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Using Scikit-Learn > Importing base estimators

Decision tree classifier: {{c1::from sklearn.tree import DecisionTreeClassifier}}

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Using Scikit-Learn > Importing base estimators

KNN classifiers: {{c1::from sklearn.neighbors import KNeighborsClassifier}}

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Using Scikit-Learn > Defining BaggingClassifier

DecisionTreeClassifier: {{c1::BaggingClassifier(DecisionTreeClassifier())}}

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Using Scikit-Learn > Out-of-bag evaluation

bag_clf = BaggingClassifier(DecisionTreeClassifier())

bag_clf.fit(X_train, y_train)

{{c1::bag_clf.oob_score_}}

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Using Scikit-Learn > Out-of-bag evaluation

bag_clf.oob_score represents the {{c1::average oob accuarcy}} of {{c2::all base predictors}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Random Forest

= {{c1::Bagging}} + {{c2::Decision Tree}}

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Random Forest

An ensemble of {{c1::randomized decision trees}}, resulting in {{c2::greater tree diversity}}

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Random Forest

Results in {{c1::greater tree diversity}}, trades a {{c2::higher bias}} for a {{c3::lower variance}}, and often leads to an {{c4::overall better prediction}}

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Random Forest

Each {{c1::decision tree}} is trained with {{c2::random samples}} of data drawn from the {{c3::training set}} using the {{c4::bagging method}}

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Random Forest

The {{c1::random training samples}} are typically of the {{c2::same size}} for each subset

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Random Forest

Introduces {{c1::extra randomness}} into the {{c2::tree-growing process}} while splitting the dataset, by searching for the {{c3::best feature}} only from a {{c4::random subset}} of all features

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Random Forest

Like other {{c1::ensemble methods}}, it trades a {{c2::higher bias}} for a {{c3::lower variance}} and often leads to an {{c4::overall better model}} (in terms of {{c5::overfitting}} and {{c6::generalization accuracy}})

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Random Forest

An {{c1::ensemble learning method}} used for {{c2::classification}} and {{c3::regression}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Random Forest

The {{c1::final decision}} is based on the {{c2::majority vote}} from all individually trees

## Back Extra


# Note
model: Cloze

## Text

Section: Bagging

Sub-section: Random Forest > Using Scikit-Learn

from sklearn. {{c1::ensemble}} import {{c2::RandomForestClassifier}}

rf_clf = {{c2::RandomForestClassifier}}(n_estimators=500, max_leaf_nodes=16)

{{c3::rf_clf.fit(X_train, y_train)}}

y_pred = {{c4::rf_clf.predict(X_test)}}

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

A {{c1::sequential ensemble learning technique}}

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Constructs an {{c1::ensemble}} by training and adding predictors {{c2::sequentially}}, each trying to learn from the {{c3::prediction errors}} made by its predecessor and progressively improving the ensemble

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

It starts with a {{c1::base classifier}} that is prepared on the {{c2::training data}}. A {{c3::second classifier}} is then created behind it to focus on the {{c4::instances}} in the training data that the {{c5::first classifier got wrong}}. The process continues to add classifiers until a limit is reached in the number of {{c6::models}} or {{c7::accuracy}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Two popular boosting algorithms are {{c1::AdaBoost}} and {{c2::Gradient Boosting}}

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: AdaBoost

Also known as {{c1::Adaptive Boosting}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: AdaBoost

The {{c1::base learners}} in AdaBoost are usually {{c2::decision stumps}}

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: AdaBoost

A {{c1::decision stump}} is a very short {{c2::decision tree}} with only a {{c3::single split}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: AdaBoost

Is an {{c1::iterative algorithm}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: AdaBoost

Each iteration is called a {{c1::boosting round}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: AdaBoost

Trains {{c1::base classifiers}} using {{c2::random subsets}} of instances drawn from the {{c3::training set}}

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: AdaBoost

Instead of using {{c1::uniform probability distribution}} for sampling, AdaBoost uses {{c2::adaptive probability distribution}}

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: AdaBoost

Each {{c1::instance}} in the training set is given a {{c2::weight}}. The {{c3::weight}} determines the {{c4::probability}} of being drawn from the training set. AdaBoost {{c5::adaptively changes}} the weights at each {{c6::boosting round}}

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: AdaBoost

Training rows are assigned {{c1::equal weights}}, {{c2::1/n}}, where n is the size of the training set. The {{c3::first classifier}} will be trained with a {{c4::random subset}} of data. Rows classified {{c5::incorrectly}} will have {{c6::increased weights}}, while classified {{c7::correctly}} will have {{c8::decreased weights}}. Subsequent classifiers will focus more on training instances {{c9::wrongly classified}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: AdaBoost

The {{c1::ensemble}} is constructed from the {{c2::sequence of classifiers}} trained in the specified number of {{c3::boosting rounds}}

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: AdaBoost

The ensemble makes prediction on a new instance by combining the predictions of all the {{c1::base classifiers}}. The predicted class is the one that has the {{c2::highest weighted votes}}

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: AdaBoost > Using Scikit-Learn

from sklearn. {{c1::ensemble}} import {{c2::AdaBoostClassifier}}

ada_clf = {{c2::AdaBoostClassifier}}(DecisionTreeClassifier(max_depth=1), n_estimators=200, random_state=42)

ada_clf.fit(X_train, y_train)

y_pred = ada_clf.predict(X_test)

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: AdaBoost > Using Scikit-Learn

In Scikit-Learn, the default base estimator for the {{c1::AdaBoostClassifierclass}} is the {{c2::decision stump}}. You can also select other base estimators such as {{c3::SVM}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: Gradient Boosting

Instead of reweighting the training instances at every iteration, {{c1::Gradient Boosting}} fits the new predictor to the {{c2::residual errors}} (loss function) made by the previous predictor.

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: Gradient Boosting

The models built by gradient boosting algorithms are called {{c1::additive models}}

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: Gradient Boosting

Its prediction is the {{c1::sum of the predictions}} of the base predictors

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: Gradient Boosting

The idea behind gradient boosting algorithm is to {{c1::repetitively leverage}} the patterns in {{c2::residual errors}} and strengthen a model with {{c3::weak predictions}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: Gradient Boosting

Build the {{c1::first model}} with simple models and analyze its {{c2::errors}} with the data. These errors signify {{c3::data points}} that are difficult to fit by a simple model. Then for later models, particularly focus on these {{c4::data hard to fit}} to get them right. In the end, combine all the predictors by giving some {{c5::weights}} to each predictor.

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: Gradient Boosting

The most common gradient boosting model is {{c1::Gradient Tree Boosting (GTB)}} – in which base predictors are {{c2::decision tree regressors}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: Gradient Boosting

Some well-known GBM are {{c1::XGBoost}}, {{c2::LightGBM}} and {{c3::CatBoost}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: Gradient Boosting

In boosting, the base predictors are {{c1::weak learners}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: Gradient Boosting

A {{c1::weak learner}} is one that does only {{c2::slightly better than random guessing}}

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: Gradient Boosting

The idea of boosting is to train a number of {{c1::weak learners}} sequentially and combine them together into a {{c2::strong learner}}

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: Gradient Boosting

Two key hyperparameters: {{c1::n_estimators}} & {{c2::learning_rate}}

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: Gradient Boosting

{{c1::n_estimators}}: Number of {{c2::boosting stages}} / {{c3::iterations}} to perform

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: Gradient Boosting

Gradient boosting is fairly {{c1::robust to overfitting}}, so large {{c2::n_estimators}} is okay.

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: Gradient Boosting

A large number of {{c1::weak estimators}} usually results in {{c2::better performance}}

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: Gradient Boosting

{{c1::Learning rate}} scales the {{c2::step length}} of the gradient descent procedure. It is a {{c3::regularization technique}} known as {{c4::shrinkage}}

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: Gradient Boosting

Smaller values of {{c1::learning_rate}} would need larger numbers of {{c2::weak learners}} to adequately fit the training data, but the ensemble will {{c3::generalize better result}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Boosting

Sub-section: Gradient Boosting

{{c1::n_estimators}} and {{c2::learning_rate}} are interrelated. The usual strategy is to set {{c3::n_estimators}} to a {{c4::large value}} and {{c5::learning_rate}} to a {{c6::small value (0.01)}}, and stop training when ensemble's prediction does not improve any more (before reaching the set number of estimators)

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Classifier:

from sklearn. {{c1::ensemble}} import {{c2::GradientBoostingClassifier}}

clf = {{c2::GradientBoostingClassifier}}(n_estimators=100, learning_rate=0.01)

clf.fit(X_train,y_train)

clf.score(X_test,y_test)

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Regressor:

from sklearn. {{c1::ensemble}} import {{c2::GradientBoostingRegressor}}

gbr = {{c2::GradientBoostingRegressor}}(max_depth=3, n_estimators=100, learning_rate=0.1, random_state=42)

gbr.fit(X_train, y_train)

y_predict = gbr.predict(X_test)

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Hyperparameters for tree growth:

{{c1::max_depth}} (typically {{c2::2 to 8}})

{{c3::max_leaf_nodes}}

{{c4::min_samples_leaf}}

{{c5::min_samples_split}}

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Sub-section: XGBoost

The main difference between {{c1::GradientBoosting}} and {{c2::XGBoost}} is that XGboost uses a {{c3::regularization technique}} in it. In simple words, it is a {{c4::regularized form}} of the existing gradient-boosting algorithm.

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Sub-section: XGBoost

The general principle is we want both a {{c1::simple}} and {{c2::accurate model}}. The tradeoff between the two is also referred as {{c3::bias-variance tradeoff}} in machine learning.

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Sub-section: XGBoost

Can be used for both {{c1::regression}} and {{c2::classification}} tasks.

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Sub-section: XGBoost

Has been designed to work with {{c1::large}} and {{c2::complicated datasets}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Sub-section: XGBoost

By adding the {{c1::regularization}}, XGBoost performs {{c2::better}} than a normal gradient boosting algorithm and much {{c3::faster}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Sub-section: LightGBM

{{c1::Light GBM}} (by {{c2::Microsoft}}) is a {{c3::fast}}, {{c4::distributed}}, {{c5::high-performance}} gradient boosting framework based on {{c6::decision tree algorithm}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Sub-section: LightGBM

Decision trees are grown {{c1::leaf wise}} with {{c2::maximum delta loss}} meaning that at a single time only {{c3::one leaf}} from the whole tree will be grown.

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Sub-section: LightGBM

Since the {{c1::leaf}} is fixed, the {{c2::leaf-wise algorithm}} has {{c3::lower loss}} compared to the {{c4::level-wise algorithm}}. This resulted in {{c5::high accuracy}} where rarely be achieved by any of the existing boosting algorithms.

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Sub-section: LightGBM

LightGBM is surprisingly very {{c1::fast}}, hence the word 'Light'.

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Sub-section: LightGBM

{{c1::Leaf wise splits}} might lead to increase in {{c2::complexity}} and may lead to {{c3::overfitting}}. This could be overcome by specifying the {{c4::max-depth parameter}} by limiting the splitting.

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Sub-section: LightGBM

Advantages of Light GBM: {{c1::faster training}}, {{c2::lower memory usage}}, {{c3::better accuracy}}, Able to use {{c4::large datasets}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Sub-section: LightGBM

Faster training speed and higher efficiency: Light GBM use {{c1::histogram based algorithm}} i.e it buckets {{c2::continuous feature values}} into {{c3::discrete bins}} which fasten the training procedure.

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Sub-section: LightGBM

Lower memory usage: Replaces {{c1::continuous values}} to {{c2::discrete bins}} which result in {{c3::lower memory usage}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Sub-section: LightGBM

Better accuracy than any other boosting algorithm: It produces much more {{c1::complex trees}} by following {{c2::leaf wise split approach}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Sub-section: LightGBM

Compatibility with Large Datasets: It is capable of performing equally good with {{c1::large datasets}} with a significant reduction in {{c2::training time}} as compared to {{c3::XGBOOST}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Sub-section: LightGBM

If you think that there is a need for {{c1::regularization}} according to your dataset, then you can definitely use {{c2::XGBoost}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Sub-section: LightGBM

If you want to {{c1::fast training speed}}, then {{c2::LightGBM}} perform very well on those types of datasets.

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Sub-section: LightGBM

If you need more {{c1::community support}} for the algorithm then use algorithms which was developed years back.

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

{{c1::Stacking}} is a {{c2::Heterogeneous ensemble}}, its base learners are trained with {{c3::different learning algorithms}}, such as {{c4::KNN}}, {{c5::Decision Tree}}, {{c6::Logistic Regression}}, {{c7::SVM}}, etc.

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Train a {{c1::meta-level classifier}} to combine the outputs of the {{c2::based-level classifiers}}

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

{{c1::Meta classifier}} is a classifier that makes a {{c2::final prediction}} among all the predictions by using those predictions as {{c3::features}}.

## Back Extra


# Note
model: Cloze

## Text

Section: Using Scikit-Learn

Sub-section: Using Scikit-Learn

from sklearn. {{c1::ensemble}} import {{c2::VotingClassifier}}, {{c3::StackingClassifier}}

sclf = {{c2::VotingClassifier}}([estimators list])

{{c4::sclf.fit(X_train, y_train)}}

y_predict = sclf.predict(X_test)

sclf = {{c3::StackingClassifier}}(classifiers=[clf1, clf2, clf3], meta_classifier=lr)

## Back Extra


