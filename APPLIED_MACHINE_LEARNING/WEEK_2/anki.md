# When do you use supervised learning?
You know what's the correct output based on the input

# The two types of supervised learning is {{Regression}} and {{Classification}}

# What does regression do?
Estimate Continuous values, also known as Real value outputs

# What does classification do?
Identify unique classes like discrete values, Boolean and categories

# What is the definition of regression?
Give a target prediction value based on independent variables.

# What is regression mostly used for?
find out relationship between variables and forecasting.

# How does Linear regression work?
Find the best fit line by minimizing the distance between all the data points and the distance to the regression line.

# {{Linear regression}} is a {{linear modelling algorithm}} to {{find relationship}} between {{one or more independent variables}} denoted as {{X}} and {{dependent variable (target)}} denoted as {{Y}}

# The 4 regression performance measurements are {{Mean Absolute Error}}, {{Mean Square Error}}, {{Root Mean Square Error}} and {{R-squared Value}}.

# Another term for {{R}} is {{Correlation}}

# Another term for {{R square}} is {{Coefficient of Determination}}

# What is R?
Amount of linear association between two variables

# What is R-square?
Proportion of variation in the dependent variable that can be attributed to the independent variables. 

# Describe R=1, R-squared=1.00. There is a {{Perfect}} {{positive}} linear association. The points are {{exactly on}} the fit line.

# Describe R=0.9, R-squared=0.81. There is a {{Large}} {{positive}} linear association. The points are {{close to}} the fit line.

# Describe R=0.45, R-squared=0.2025. There is a {{Small}} {{positive}} linear association. The points are {{far from}} the fit line.

# Describe R=0.0, R-squared=0.0. There is a {{No}} linear association.

# Describe R=-0.3, R-squared=0.09. There is a {{Small}} {{negative}} linear association. The points are {{far from}} the fit line.

# Describe R=-1, R-squared=1.00. There is a {{Perfect}} {{negative}} linear association. The points are {{exactly}} the fit line.

# What is the difference between MAE and RMSE?
RMSE is sensitive to outlying prediction.

# What does it mean if your RMSE is a lot higher than MAE?
There is a prediction that has very high residual

# What are residuals?
Difference between actual and predicted values

# We can determine whether a predictive model is {{underfitting}} or {{overfitting}} the {{training data}} by {{looking at the prediction error}} on the {{training}} and {{evaluation}} data.

# What does underfitting mean?
Model performs poorly on the training data.

# Why do models underfit?
Unable to capture the relationship between input and target values

# What does overfitting mean?
Model performs well on training data but doesn't perform well on testing data.

# Why do models overfit?
Unable to generalize to unseen examples

# To avoid overfitting, one can do {{Regularization}} and {{Cross-validation}}.

# What is Regularization?
Form of regression that constrains/regularizes or shrinks the coefficient estimates towards zero.

# The three most popular forms of constrains we could use for Regularization are {{Ridge regression}}, {{Lasso}} and {{Elastic Net}}.

# What does high Bias mean?
Model has represented a simple relationship when data points clearly indicate a complex relationship

# What does high Variance mean?
Model learn too much from the fluctuations and noise of the data, and is thus unable to accurately predict new data.

# What is cross-validation?
Generate 5/10 mini train-test splits to tune your model.

# To avoid underfitting, one can {{increase}} the {{size}}, {{number of parameters}}, {{complexity}}, or {{type of the model}}.

# How do you avoid underfitting in neural networks?
Increase training time until cost function is minimized.

# Why do we have Validation data when we already have Train and test data?  
Tune the Hyperparameters based on the result.

# {{Multivariate}} Regression estimates regression model with {{more than one outcome variables}}, while {{Univariate}} Regression estimates regression model with {{only one outcome variable}}. 

# {{Polynomial}} Regression is a regression model in which the {{relationship between independent and dependent variable is modeled as the nth degree of a polynomial}}.

# What do you use if two variables have a non-linear relatioship?
Polynomial regression

# In {{SVR}}, we form {{two lines on either side of the given line}}, and the {{data points lying within it are discarded}} from the {{error}} point of view.

# What are the lines around the best fit line of SVR called?
Decision Boundaries

# What is the best fit line of SVR called?
Hyperplane

# The important parameters of SVR is: {{kernel}}, {{C}} and {{Gamma}}.

# What does the kernel parameter in SVR do?
Defines kernel type to use.

# The most common kernels are {{rbf - Radial Basis Function}}(default), {{poly}} and {{sigmoid}}

# {{C}} is the {{regularization parameter}} of SVRs.

# {{Gamma}} decides how much {{curvature we want in a decision boundary}}.

# What is one trait of SVR?
