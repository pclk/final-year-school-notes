model: Basic

# Note

## Front
When do you use supervised learning?

## Back
You know what's the correct output based on the input

# Note
model: Cloze

## Text
The two types of supervised learning is {{c1::Regression}} and {{c2::Classification}}

## Back Extra


# Note

## Front
What does regression do?

## Back
Estimate Continuous values, also known as Real value outputs

# Note

## Front
What does classification do?

## Back
Identify unique classes like discrete values, Boolean and categories

# Note

## Front
What is the definition of regression?

## Back
Give a target prediction value based on independent variables.

# Note

## Front
What is regression mostly used for?

## Back
find out relationship between variables and forecasting.

# Note

## Front
How does Linear regression work?

## Back
Find the best fit line by minimizing the distance between all the data points and the distance to the regression line.

# Note
model: Cloze

## Text
{{c1::Linear regression}} is a {{c2::linear modelling algorithm}} to {{c3::find relationship}} between {{c4::one or more independent variables}} denoted as {{c5::X}} and {{c6::dependent variable (target)}} denoted as {{c7::Y}}

## Back Extra


# Note
model: Cloze

## Text
The 4 regression performance measurements are {{c1::Mean Absolute Error}}, {{c2::Mean Square Error}}, {{c3::Root Mean Square Error}} and {{c4::R-squared Value}}.

## Back Extra


# Note
model: Cloze

## Text
Another term for {{c1::R}} is {{c2::Correlation}}

## Back Extra


# Note
model: Cloze

## Text
Another term for {{c1::R square}} is {{c2::Coefficient of Determination}}

## Back Extra


# Note

## Front
What is R?

## Back
Amount of linear association between two variables

# Note

## Front
What is R-square?

## Back
Proportion of variation in the dependent variable that can be attributed to the independent variables.

# Note
model: Cloze

## Text
Describe R=1, R-squared=1.00. There is a {{c1::Perfect}} {{c2::positive}} linear association. The points are {{c3::exactly on}} the fit line.

## Back Extra


# Note
model: Cloze

## Text
Describe R=0.9, R-squared=0.81. There is a {{c1::Large}} {{c2::positive}} linear association. The points are {{c3::close to}} the fit line.

## Back Extra


# Note
model: Cloze

## Text
Describe R=0.45, R-squared=0.2025. There is a {{c1::Small}} {{c2::positive}} linear association. The points are {{c3::far from}} the fit line.

## Back Extra


# Note
model: Cloze

## Text
Describe R=0.0, R-squared=0.0. There is a {{c1::No}} linear association.

## Back Extra


# Note
model: Cloze

## Text
Describe R=-0.3, R-squared=0.09. There is a {{c1::Small}} {{c2::negative}} linear association. The points are {{c3::far from}} the fit line.

## Back Extra


# Note
model: Cloze

## Text
Describe R=-1, R-squared=1.00. There is a {{c1::Perfect}} {{c2::negative}} linear association. The points are {{c3::exactly}} the fit line.

## Back Extra


# Note

## Front
What is the difference between MAE and RMSE?

## Back
RMSE is sensitive to outlying prediction.

# Note

## Front
What does it mean if your RMSE is a lot higher than MAE?

## Back
There is a prediction that has very high residual

# Note

## Front
What are residuals?

## Back
Difference between actual and predicted values

# Note
model: Cloze

## Text
We can determine whether a predictive model is {{c1::underfitting}} or {{c2::overfitting}} the {{c3::training data}} by {{c4::looking at the prediction error}} on the {{c5::training}} and {{c6::evaluation}} data.

## Back Extra


# Note

## Front
What does underfitting mean?

## Back
Model performs poorly on the training data.

# Note

## Front
Why do models underfit?

## Back
Unable to capture the relationship between input and target values

# Note

## Front
What does overfitting mean?

## Back
Model performs well on training data but doesn't perform well on testing data.

# Note

## Front
Why do models overfit?

## Back
Unable to generalize to unseen examples

# Note
model: Cloze

## Text
To avoid overfitting, one can do {{c1::Regularization}} and {{c2::Cross-validation}}.

## Back Extra


# Note

## Front
What is Regularization?

## Back
Form of regression that constrains/regularizes or shrinks the coefficient estimates towards zero.

# Note
model: Cloze

## Text
The three most popular forms of constrains we could use for Regularization are {{c1::Ridge regression}}, {{c2::Lasso}} and {{c3::Elastic Net}}.

## Back Extra


# Note

## Front
What does high Bias mean?

## Back
Model has represented a simple relationship when data points clearly indicate a complex relationship

# Note

## Front
What does high Variance mean?

## Back
Model learn too much from the fluctuations and noise of the data, and is thus unable to accurately predict new data.

# Note

## Front
What is cross-validation?

## Back
Generate 5/10 mini train-test splits to tune your model.

# Note
model: Cloze

## Text
To avoid underfitting, one can {{c1::increase}} the {{c2::size}}, {{c3::number of parameters}}, {{c4::complexity}}, or {{c5::type of the model}}.

## Back Extra


# Note

## Front
How do you avoid underfitting in neural networks?

## Back
Increase training time until cost function is minimized.

# Note

## Front
Why do we have Validation data when we already have Train and test data?

## Back
Tune the Hyperparameters based on the result.

# Note
model: Cloze

## Text
{{c1::Multivariate}} Regression estimates regression model with {{c2::more than one outcome variables}}, while {{c3::Univariate}} Regression estimates regression model with {{c4::only one outcome variable}}.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Polynomial}} Regression is a regression model in which the {{c2::relationship between independent and dependent variable is modeled as the nth degree of a polynomial}}.

## Back Extra


# Note

## Front
What do you use if two variables have a non-linear relatioship?

## Back
Polynomial regression

# Note
model: Cloze

## Text
In {{c1::SVR}}, we form {{c2::two lines on either side of the given line}}, and the {{c3::data points lying within it are discarded}} from the {{c4::error}} point of view.

## Back Extra


# Note

## Front
What are the lines around the best fit line of SVR called?

## Back
Decision Boundaries

# Note

## Front
What is the best fit line of SVR called?

## Back
Hyperplane

# Note
model: Cloze

## Text
The important parameters of SVR is: {{c1::kernel}}, {{c2::C}} and {{c3::Gamma}}.

## Back Extra


# Note

## Front
What does the kernel parameter in SVR do?

## Back
Defines kernel type to use.

# Note
model: Cloze

## Text
The most common kernels are {{c1::rbf - Radial Basis Function}}(default), {{c2::poly}} and {{c3::sigmoid}}

## Back Extra


# Note
model: Cloze

## Text
{{c1::C}} is the {{c2::regularization parameter}} of SVRs.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Gamma}} decides how much {{c2::curvature we want in a decision boundary}}.

## Back Extra


# Note

## Front
What is one trait of SVR?

## Back
SVR is robust to outliers.

