# In Linear Regression, the training process is Y = WX + B. What are we adjusting as the training process continues?
W and B (Weights) and (Biases)

# {{Bias}} is the same as MSE.

# Other than cross-validation and regularization, the ways to prevent overfitting is {{Gather more data}}, {{fix data errors}} and {{remove outliers}}.

# An example of increasing the complexity or type of the model to prevent underfitting is {{converting linear to non-linear data}}.

# {{Linear regression}} is one of the most widely known modeling technique.

# To obtain the best fit line in Linear regression, we can easily accomplish it using the {{Least Square Method}}.

# What kind of relationship in the data must be necessary to use Linear Regression?
Linear

# The issues suffered by Multiple Regression are {{multicollinearity}}, {{autocorrelation}} and {{heteroskedasticity}}.

# {{Multicollinearity}} refers to when independent variables are highly correlated with each other.
"many"+"moving in same line"

# {{Autocorrelation}} refers to when data is correlated with its past data.
"self"+"relationship". When past increases leads to more current increases, or vice versa.

# {{Heteroskedasticity}} refers to uneven spread of residuals/errors across predictions
"different"+"dispersion"

# What model is very sensitive to Outliers?
Linear Regression

# {{Multicollinearity}} makes models {{unstable}}, increases {{uncertainty}} in coefficient estimates, and makes it hard to determine the {{important variables}}.

# The {{stepwise}} regression methods are {{standard}}, {{forward selection}}, {{backward selection}}.

# {{Standard stepwise regression}}: Adds and removes predictors as needed in each step.

# {{Forward selection}}: Starts with no variables, adds one at a time.

# {{Backward selection}}: Starts with all variables, removes one at a time.

# Does Logistic regression require linear relationship in data?
no

# A good approach to include all significant variables to avoid under/overfitting, is to use a {{stepwise}} method to estimate 

# {{Ordinary Least square}} is used by Linear Regression, and is another term for the Least Square Method.

# {{Maximum Likelihood Estimates}} is used by Logistic Regression to approximate true distribution and parameters.

# Does Logistic regression require large sample size?
yes

# To gather insights or even improve model fit in presence multicollinearity with Logistic regression, we have an option to include {{interaction effects}}.
of categorical variables. These can capture complex relationships of the variables, and justify real life theoretical or practical interactions. 

# {{Ordinal}} variables are categorical variables with clear ordering.

# {{Ordinal logistic regression}} handles ordinal dependent variables.

# {{Multinomial logistic regression}} handles more than 2 classes of dependent variables.

# If the power of independent variable is more than 1, it is considered a {{polynomial regression}} equation.

# Though polynomial may get {{lower}} error with {{higher}} degrees, this can result in {{overfitting}}.

# Higher polynomial regressions need to be watched at the {{ends}} of the curve.

# {{Extrapolation}} in polynomial regression refers to the predictions of values outside the known data.

# Higher polynomials can produce {{weird}} results on extrapolation.

# The aim of stepwise regression is to {{maximize}} prediction power while with {{minimal}} prediction variables.

# The metrics used for Stepwise regression is {{R^2}}, {{t-stats}}, and {{AIC}} metric.

# {{R^2}} answers: "How well does my model fit the data?"

# {{t-stats}} answers: "Which variables actually matter?"

# {{AIC}} answers: "Is this model more efficient?"

# {{R^2}} always {{increases}} with more variables. 

# {{AIC}} penalizes models for having too many {{variables}}.

# {{Ridge regression}} is used against multicollinearity.

# Ridge regression introduces a regularization parameter which adds a penalty to the {{size of the coefficients}}.

# Ridge regression introduces a small amount of {{bias}} while significantly reducing {{variance}}.

# Ridge regression is also known as {{L2 regression}}.

# {{L2 regression}} adds a penalty proportional to the square of the coefficient.

# Lasso regression is also known as {{L1 regression}}.

# {{L1 regression}} adds a penalty proportional to the absolute value of the coefficient.

# Lasso regression has similar assumptions about the data as the least squared regression except {{normality}}.
Least square regression assumes that the error terms are normally distributed but Lasso doesn't require this normality.

# Lasso regression will shrink the coefficients {{exactly}} to  zero.

# Ridge regression will shrink the coefficients {{close}} to  zero.

# If there is highly correlated group of predictors, Lasso {{picks one and shrinks others to zero}}.

# ElasticNet is a {{hybird}} of {{Lasso}} and {{Ridge}}.

# If there is highly correlated group of predictors, ElasticNet {{picks multiple of them}}.

# ElasticNet can suffer from {{double shrinkage}}, where there might to excessive shrinkage, leading to underfitting.
