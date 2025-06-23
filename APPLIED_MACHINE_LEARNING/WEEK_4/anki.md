# When {{AUC}} is {{0.7}}, it means that that is a {{70}}% chance that the model can differentiate.

# The {{worst}} case is when {{AUC}} is {{0.5}}.

# When {{AUC}} is {{0}}, the model is {{reciprocating}} the classes.

# Logistic curves {{cannot}} exceed the range {{0 and 1}} while Linear {{can}} exceed the range.

# Decision trees usually choose a {{feature}} that has the optimal {{index}}, for example {{entropy}} or {{GINI index}}. Then, they {{split}} the dataset based on the {{chosen feature}}, and {{repeats}} until it meets the {{stopping}} criteria.

# {{Entropy}} is an {{information}} theory metric that measures the {{impurity}} or {{uncertainty}} in a group of observations.

# {{GINI index}} measures the {{probability}} for a {{random}} instance to be {{misclassified}} when chosen {{randomly}}.

# The stopping criteria for decision trees can be: {{5%}} of the data in each {{leaves}}, the leaves' {{purity}}, the trees' {{depth}}, or feature {{selection}}.

# SVM represents the data as {{points}} in space, and tries to separate them by dividing it with a {{wide}} gap.

# {{Imbalanced}} data may not be a problem, unless you have {{small}} sample size, tough class {{separability}}, and {{within}} class {{subclusters}}.

# Small sample size datasets in Imbalanced data refers to the {{minority}} class, where finding {{patterns}} of this class would be hard.

# {{Class separability}} means that if patterns {{overlap}}, it is harder to find rules.

# {{Linearly}} separable domains {{aren't}} sensitive to any amount of {{imbalance}}.

# {{Within class subclusters}} makes it harder to find {{boundaries}} to separate the classes.

# Metrics to use for imbalanced data are the {{confusion matrix}}, {{precision}} and {{recall}} and its {{curve}}, and the {{F1 score}}.

# {{PR}} curve often {{zig-zags}}.

# An {{inbalanced}} dataset would make {{ROC}} curve look {{better}} than it would in a {{balanced}} dataset.

# {{PR}} curves are not impacted by {{imbalanced}} datasets.

# {{Under}}-sampling refers to reducing the number of samples from the {{majority}} class

# {{Over}}-sampling refers to increase the number of samples from the {{minority}} class

# {{Random}} under-sampling randomly {{removes}} samples from the majority class until a certain {{balancing}} ratio has been reached. This is a {{bad}} approach if we have {{little}} data.

# {{Remove noisy neighbor}} removes majority classes that are {{harder}} to classify.

# {{Tomek Links}} are {{2}} samples that are the {{nearest}} neighbors and from a {{different}} class.  

# {{Retain close neighbor}} removes majority classes that are {{easier}} to classify

# {{Random}} over-sampling {{duplicates}} observations at random from the {{minority}} class. This however, {{increases}} the likelihood of {{overfitting}}.

# {{SMOTE}} will generate new observations based on its {{neighbors}}.

# {{Borderline}}-SMOTE creates synthethic data only from the {{minority}} class {{closer}} to the boundary between the classes.

# {{SVM}}-SMOTE is similar in behaviour to {{Borderline}}-SMOTE but is given by the {{support vectors}} of {{SVM}}

# {{ADASYS}} uses data that is {{harder}} to classify to generate synthetic data.

# Oversampling should be done {{after}} setting aside the {{validation}} set to avoid {{overfitting}} due to data {{duplication}} process.

# In Over vs Under-sampling, we trade {{noise}} vs {{info loss}}.

# {{Recall}} is also known by {{True Positive Rate}} and {{Sensitivity}}.

# {{Specificity}} is also known by {{True Negative Rate}}.
