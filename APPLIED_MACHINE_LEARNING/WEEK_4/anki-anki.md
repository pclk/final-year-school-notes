model: Basic

# Note
model: Cloze

## Text
When {{c1::AUC}} is {{c2::0.7}}, it means that that is a {{c3::70}}% chance that the model can differentiate.

## Back Extra


# Note
model: Cloze

## Text
The {{c1::worst}} case is when {{c2::AUC}} is {{c3::0.5}}.

## Back Extra


# Note
model: Cloze

## Text
When {{c1::AUC}} is {{c2::0}}, the model is {{c3::reciprocating}} the classes.

## Back Extra


# Note
model: Cloze

## Text
Logistic curves {{c1::cannot}} exceed the range {{c2::0 and 1}} while Linear {{c3::can}} exceed the range.

## Back Extra


# Note
model: Cloze

## Text
Decision trees usually choose a {{c1::feature}} that has the optimal {{c2::index}}, for example {{c3::entropy}} or {{c4::GINI index}}. Then, they {{c5::split}} the dataset based on the {{c6::chosen feature}}, and {{c7::repeats}} until it meets the {{c8::stopping}} criteria.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Entropy}} is an {{c2::information}} theory metric that measures the {{c3::impurity}} or {{c4::uncertainty}} in a group of observations.

## Back Extra


# Note
model: Cloze

## Text
{{c1::GINI index}} measures the {{c2::probability}} for a {{c3::random}} instance to be {{c4::misclassified}} when chosen {{c5::randomly}}.

## Back Extra


# Note
model: Cloze

## Text
The stopping criteria for decision trees can be: {{c1::5%}} of the data in each {{c2::leaves}}, the leaves' {{c3::purity}}, the trees' {{c4::depth}}, or feature {{c5::selection}}.

## Back Extra


# Note
model: Cloze

## Text
SVM represents the data as {{c1::points}} in space, and tries to separate them by dividing it with a {{c2::wide}} gap.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Imbalanced}} data may not be a problem, unless you have {{c2::small}} sample size, tough class {{c3::separability}}, and {{c4::within}} class {{c5::subclusters}}.

## Back Extra


# Note
model: Cloze

## Text
Small sample size datasets in Imbalanced data refers to the {{c1::minority}} class, where finding {{c2::patterns}} of this class would be hard.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Class separability}} means that if patterns {{c2::overlap}}, it is harder to find rules.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Linearly}} separable domains {{c2::aren't}} sensitive to any amount of {{c3::imbalance}}.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Within class subclusters}} makes it harder to find {{c2::boundaries}} to separate the classes.

## Back Extra


# Note
model: Cloze

## Text
Metrics to use for imbalanced data are the {{c1::confusion matrix}}, {{c2::precision}} and {{c3::recall}} and its {{c4::curve}}, and the {{c5::F1 score}}.

## Back Extra


# Note
model: Cloze

## Text
{{c1::PR}} curve often {{c2::zig-zags}}.

## Back Extra


# Note
model: Cloze

## Text
An {{c1::inbalanced}} dataset would make {{c2::ROC}} curve look {{c3::better}} than it would in a {{c4::balanced}} dataset.

## Back Extra


# Note
model: Cloze

## Text
{{c1::PR}} curves are not impacted by {{c2::imbalanced}} datasets.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Under}}-sampling refers to reducing the number of samples from the {{c2::majority}} class

## Back Extra


# Note
model: Cloze

## Text
{{c1::Over}}-sampling refers to increase the number of samples from the {{c2::minority}} class

## Back Extra


# Note
model: Cloze

## Text
{{c1::Random}} under-sampling randomly {{c2::removes}} samples from the majority class until a certain {{c3::balancing}} ratio has been reached. This is a {{c4::bad}} approach if we have {{c5::little}} data.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Remove noisy neighbor}} removes majority classes that are {{c2::harder}} to classify.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Tomek Links}} are {{c2::2}} samples that are the {{c3::nearest}} neighbors and from a {{c4::different}} class.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Retain close neighbor}} removes majority classes that are {{c2::easier}} to classify

## Back Extra


# Note
model: Cloze

## Text
{{c1::Random}} over-sampling {{c2::duplicates}} observations at random from the {{c3::minority}} class. This however, {{c4::increases}} the likelihood of {{c5::overfitting}}.

## Back Extra


# Note
model: Cloze

## Text
{{c1::SMOTE}} will generate new observations based on its {{c2::neighbors}}.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Borderline}}-SMOTE creates synthethic data only from the {{c2::minority}} class {{c3::closer}} to the boundary between the classes.

## Back Extra


# Note
model: Cloze

## Text
{{c1::SVM}}-SMOTE is similar in behaviour to {{c2::Borderline}}-SMOTE but is given by the {{c3::support vectors}} of {{c4::SVM}}

## Back Extra


# Note
model: Cloze

## Text
{{c1::ADASYS}} uses data that is {{c2::harder}} to classify to generate synthetic data.

## Back Extra


# Note
model: Cloze

## Text
Oversampling should be done {{c1::after}} setting aside the {{c2::validation}} set to avoid {{c3::overfitting}} due to data {{c4::duplication}} process.

## Back Extra


# Note
model: Cloze

## Text
In Over vs Under-sampling, we trade {{c1::noise}} vs {{c2::info loss}}.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Recall}} is also known by {{c2::True Positive Rate}} and {{c3::Sensitivity}}.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Specificity}} is also known by {{c2::True Negative Rate}}.

## Back Extra


