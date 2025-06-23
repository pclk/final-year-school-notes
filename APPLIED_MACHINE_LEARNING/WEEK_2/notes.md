clarify if last week topic 2 is tested

Most important is Mean Absolute Error check the difference between data points and line.

Mean absolute Error, Mean square error, Root Mean Square error?
RMSE is sensisive to error distribution. If there is outliers, it will be much more higher than MAE.
If RMSE is a lot higher than MAE, there are outliers.
Residuals is the difference between predicted and actual value

How to tell whether a model is overfit or underfit?

Split your data. if train data validation is much higher than test data validation.

How to solve if model is underfit?
use more complex models and add more features.
Increase size or number of parameters
Increase training time if using neural model.

How to solve if model is overfit?
Regularization contrains/regularize/shrink the coefficient estimates towards zero. Make it much more simpler to avoid overfit.

Bias and Varience.
Bias when algo has limited flexibility to learn the dataset
is error on training data

Variance when algo is sensitive
related to testing data

Can use Cross validation for overfit/underfit?
standard use 5/10 chunk
maybe the split was unlucky when overfit or underfit.
Good to prove that no matter which is test or train, the result is similar

Why do we need validation data?
Assume test data is future data. 
With validation and training data, can tune the hyperparamaters based on the result.

What is Support Vector Regression (SVR)
It's like linear regression with decision the boundary, but the data points between decision boundary will be ignored

L2 Regularization is ridge regularization
L3 Regularization is lasso regularization
