model: Basic

# Note

## Front
In Linear Regression, the training process is Y = WX + B. What are we adjusting as the training process continues?

## Back
W and B (Weights) and (Biases)

# Note
model: Cloze

## Text
{{c1::Bias}} is the same as MSE.

## Back Extra


# Note
model: Cloze

## Text
Other than cross-validation and regularization, the ways to prevent overfitting is {{c1::Gather more data}}, {{c2::fix data errors}} and {{c3::remove outliers}}.

## Back Extra


# Note
model: Cloze

## Text
An example of increasing the complexity or type of the model to prevent underfitting is {{c1::converting linear to non-linear data}}.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Linear regression}} is one of the most widely known modeling technique.

## Back Extra


# Note
model: Cloze

## Text
To obtain the best fit line in Linear regression, we can easily accomplish it using the {{c1::Least Square Method}}.

## Back Extra


# Note

## Front
What kind of relationship in the data must be necessary to use Linear Regression?

## Back
Linear

# Note
model: Cloze

## Text
The issues suffered by Multiple Regression are {{c1::multicollinearity}}, {{c2::autocorrelation}} and {{c3::heteroskedasticity}}.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Multicollinearity}} refers to when independent variables are highly correlated with each other.

## Back Extra
"many"+"moving in same line"

# Note
model: Cloze

## Text
{{c1::Autocorrelation}} refers to when data is correlated with its past data.

## Back Extra
"self"+"relationship". When past increases leads to more current increases, or vice versa.

# Note
model: Cloze

## Text
{{c1::Heteroskedasticity}} refers to uneven spread of residuals/errors across predictions

## Back Extra
"different"+"dispersion"

# Note

## Front
What model is very sensitive to Outliers?

## Back
Linear Regression

# Note
model: Cloze

## Text
{{c1::Multicollinearity}} makes models {{c2::unstable}}, increases {{c3::uncertainty}} in coefficient estimates, and makes it hard to determine the {{c4::important variables}}.

## Back Extra


# Note
model: Cloze

## Text
The {{c1::stepwise}} regression methods are {{c2::standard}}, {{c3::forward selection}}, {{c4::backward selection}}.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Standard stepwise regression}}: Adds and removes predictors as needed in each step.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Forward selection}}: Starts with no variables, adds one at a time.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Backward selection}}: Starts with all variables, removes one at a time.

## Back Extra


# Note

## Front
Does Logistic regression require linear relationship in data?

## Back
no

# Note
model: Cloze

## Text
A good approach to include all significant variables to avoid under/overfitting, is to use a {{c1::stepwise}} method to estimate

## Back Extra


# Note
model: Cloze

## Text
{{c1::Ordinary Least square}} is used by Linear Regression, and is another term for the Least Square Method.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Maximum Likelihood Estimates}} is used by Logistic Regression to approximate true distribution and parameters.

## Back Extra


# Note

## Front
Does Logistic regression require large sample size?

## Back
yes

# Note
model: Cloze

## Text
To gather insights or even improve model fit in presence multicollinearity with Logistic regression, we have an option to include {{c1::interaction effects}}.

## Back Extra
of categorical variables. These can capture complex relationships of the variables, and justify real life theoretical or practical interactions.

# Note
model: Cloze

## Text
{{c1::Ordinal}} variables are categorical variables with clear ordering.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Ordinal logistic regression}} handles ordinal dependent variables.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Multinomial logistic regression}} handles more than 2 classes of dependent variables.

## Back Extra


# Note
model: Cloze

## Text
If the power of independent variable is more than 1, it is considered a {{c1::polynomial regression}} equation.

## Back Extra


# Note
model: Cloze

## Text
Though polynomial may get {{c1::lower}} error with {{c2::higher}} degrees, this can result in {{c3::overfitting}}.

## Back Extra


# Note
model: Cloze

## Text
Higher polynomial regressions need to be watched at the {{c1::ends}} of the curve.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Extrapolation}} in polynomial regression refers to the predictions of values outside the known data.

## Back Extra


# Note
model: Cloze

## Text
Higher polynomials can produce {{c1::weird}} results on extrapolation.

## Back Extra


# Note
model: Cloze

## Text
The aim of stepwise regression is to {{c1::maximize}} prediction power while with {{c2::minimal}} prediction variables.

## Back Extra


# Note
model: Cloze

## Text
The metrics used for Stepwise regression is {{c1::R^2}}, {{c2::t-stats}}, and {{c3::AIC}} metric.

## Back Extra


# Note
model: Cloze

## Text
{{c1::R^2}} answers: "How well does my model fit the data?"

## Back Extra


# Note
model: Cloze

## Text
{{c1::t-stats}} answers: "Which variables actually matter?"

## Back Extra


# Note
model: Cloze

## Text
{{c1::AIC}} answers: "Is this model more efficient?"

## Back Extra


# Note
model: Cloze

## Text
{{c1::R^2}} always {{c2::increases}} with more variables.

## Back Extra


# Note
model: Cloze

## Text
{{c1::AIC}} penalizes models for having too many {{c2::variables}}.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Ridge regression}} is used against multicollinearity.

## Back Extra


# Note
model: Cloze

## Text
Ridge regression introduces a regularization parameter which adds a penalty to the {{c1::size of the coefficients}}.

## Back Extra


# Note
model: Cloze

## Text
Ridge regression introduces a small amount of {{c1::bias}} while significantly reducing {{c2::variance}}.

## Back Extra


# Note
model: Cloze

## Text
Ridge regression is also known as {{c1::L2 regression}}.

## Back Extra


# Note
model: Cloze

## Text
{{c1::L2 regression}} adds a penalty proportional to the square of the coefficient.

## Back Extra


# Note
model: Cloze

## Text
Lasso regression is also known as {{c1::L1 regression}}.

## Back Extra


# Note
model: Cloze

## Text
{{c1::L1 regression}} adds a penalty proportional to the absolute value of the coefficient.

## Back Extra


# Note
model: Cloze

## Text
Lasso regression has similar assumptions about the data as the least squared regression except {{c1::normality}}.

## Back Extra
Least square regression assumes that the error terms are normally distributed but Lasso doesn't require this normality.

# Note
model: Cloze

## Text
Lasso regression will shrink the coefficients {{c1::exactly}} to  zero.

## Back Extra


# Note
model: Cloze

## Text
Ridge regression will shrink the coefficients {{c1::close}} to  zero.

## Back Extra


# Note
model: Cloze

## Text
If there is highly correlated group of predictors, Lasso {{c1::picks one and shrinks others to zero}}.

## Back Extra


# Note
model: Cloze

## Text
ElasticNet is a {{c1::hybird}} of {{c2::Lasso}} and {{c3::Ridge}}.

## Back Extra


# Note
model: Cloze

## Text
If there is highly correlated group of predictors, ElasticNet {{c1::picks multiple of them}}.

## Back Extra


# Note
model: Cloze

## Text
ElasticNet can suffer from {{c1::double shrinkage}}, where there might to excessive shrinkage, leading to underfitting.

## Back Extra


