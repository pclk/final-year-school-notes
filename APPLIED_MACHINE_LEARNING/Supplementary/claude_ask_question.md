1. In a linear regression model predicting employee salaries, what would be the impact of outliers in the "Years of Experience" data, and how would you handle them?
Outliers in years of experience would increase uncertainty in the coefficients of the model. Linear regression is exceptionally sensitive to outliers, so removing them would be top priority. However, it is important to find out whether the outliers are data errors or represent a true real-life relationship which the linear regression is unable to capture.

2. Explain why Mean Square Error (MSE) might not always be the best metric for evaluating regression models. What alternatives would you suggest in different scenarios?
Mean Square Error is equivalent to Bias, which means that if MSE was considered the best metric and it is optimized, the model could have very little bias, but in fact very high variance, indicating that the model has overfit. One example is the polynomial regression, where higher degrees of the polynomial function can lead to lower MSE, but may not be practically useful as it would not extrapolate well and may not generalize well. 

3. In a salary prediction model, if you notice the residual plot shows a clear pattern rather than random scatter, what does this indicate about your model's assumptions?
This means that the residual (also known as error) has some sort of function of x that is unaccounted for in the model. This indicates that the model is likely too simple to catch this relationship, and we should increase it's capabilities by increasing its size, number of parameters, complexity or type of the model.

4. Compare and contrast the implications of underfitting versus overfitting in a linear regression model predicting employee salaries.
An underfit linear regression model indicates that the model performs poorly in the training data. This means that it is unable to capture the relationships to predict employee salaries. This also means that the model is a poor predictor, and increasing the complexity of the model is likely to bring better results.
An overfit linear regression model indicates that the models performs well in the training data, but generalizes poorly in the testing data. This means that the model is unable to generalize, possibly because it follows the fluctuations and noise of the training data too closely. This also means that the model is a poor predictor, and regularizing and cross validating the model is a good next step.

5. How would you handle a situation where the relationship between years of experience and salary appears to be non-linear? What modifications to the linear regression model would you suggest?
I would use a polynomial regression model, where the linear regression model's function of y=wx+b is changed to y=wx^2 + 2wx + 2wb + b. This polynomial regression model would be able to curve and fit the non-linear data better, to represent a non-linear relationship.

6. If you have multiple features for predicting salary (experience, education level, role), explain how you would determine which features are most important for your model.
The process of determining the most important features is called feature selection. There are multiple ways to do so, which is by using the correlation matrix, the 3 stepwise methods of standard, backward and forward, as well as regularization. Notably, Lasso regression, also known as L1 regression, would shrink unimportant features to exactly zero, effectively performing feature selection.

7. In a regression problem, when would you choose Ridge regression over standard linear regression? Provide specific scenarios.
I would choose ridge regression when the standard linear regression model has overfit to the data. For example, if a linear regression has done very well at predicting employee data within the training dataset, but is doing poorly in new employee data, I would employ ridge regression as though it introduces a slight bias into the model, it would significantly decrease the variance, leading to more practical predictions.

8. How would you validate whether your salary prediction model is generalizing well to new data? Describe your approach and reasoning.
I would test my salary prediction model against unseen new data. To do so, I would build a website and host my salary prediction model, and allow participants (that I allow) to key in the predictors that the model uses into the model, and predict the salary. Then, the user can answer whether the prediction is within 5% of their actual salary. From which, we will be able to gather quantitative data to arrive at informed decision on wether the prediction model is doing well.

9. If your linear regression model shows high bias, what steps would you take to improve its performance without introducing high variance?
High bias indicates underfitting. To improve its performance without introducing variance (i.e. overfitting), it would require a balance of increasing the complexity of the model whilst not doing so until the model starts to closely follow the training data's fluctuations and noise. I would proceed to increase the complexity of the model through the results and insights shown by the linear regression model. For example, if I learnt that the relationship between the data and label is a non-linear one, I could consider a polynomial regression model. Then, if my variance starts going too high, i can employ regularization techniques like ridge regression.

10. Explain how feature scaling could affect the interpretation of coefficients in a linear regression model predicting salaries. When is it necessary, and when might it be counterproductive?
Feature scaling means to normalize or affect the distribution of the entire dataset. It is useful when there are a wide spread of data, but it would mean that everytime you test the model, the predictors must also be normalized or scaled with the same normalizer or scaler as used in the model. 
