Model Performance Comparison:
--------------------------------------------------------------------------------

Decision Tree:
RMSE: 853.15
MAE: 654.13
R2 Score: 0.0260
Cross-validation R2: 0.0263 (±0.0005)

XGBoost:
RMSE: 846.52
MAE: 646.46
R2 Score: 0.0411
Cross-validation R2: 0.0430 (±0.0008)

LightGBM:
RMSE: 847.40
MAE: 648.92
R2 Score: 0.0391
Cross-validation R2: 0.0395 (±0.0007)

Random Forest:
RMSE: 852.95
MAE: 654.09
R2 Score: 0.0264
Cross-validation R2: 0.0266 (±0.0006)



While all models were successfully trained and evaluated, the overall predictive performance, as indicated by R² scores, is low across all models.

XGBoost and LightGBM demonstrated slightly better performance compared to Decision Tree and Random Forest, but the practical utility of these models in their current state is limited due to the low R² values.

Significant data quality concerns, previously identified in our initial data exploration, are likely contributing to these suboptimal results.

Low R² Scores: All models exhibit very low R² scores (ranging from 0.0260 to 0.0411). This indicates that these models, in their current configuration and with the given features, explain only a small percentage of the variance in the premium amount.
RMSE and MAE Values: The RMSE values are around 850, and MAE values are around 650. Given the scale of premium amounts in our dataset, these error magnitudes are considered big.
Low R² Scores & AI ethics
An R² of 0.02-0.04 shows around 96-98% of the premium amount variation unexplained.

Possible causes of decreasing R² are missing features, simple models or inherent randomness in data.

However in this case, with the data preparation, the issue is likely with data quality.

Deploying a model with low R² is a serious ethical oversight. This is because of the following reasons:

Accuracy
Low R² typically results in low accuracy, which is true in our case, looking at RMSE and MAE magnitudes. If the model is deployed and used to make AI-Augmented Decisions, we face the following risks:

Undercharging premiums
Overcharging premiums
In the first case, this leads to financial instability for the insurance company in the long run, and cause loss in profits In the second case, this leads to low-risk individuals unfairly charged higher premiums, which may discourage them and push them away from signing a policy with the insurance company, causing loss in profits.

Explainability
Usually Tree-based models are praised for their interpretability. However, intepreting a model that is low R² is ineffective. For example, a detective that is given the physical crime scene to work with, vs a detective that is far away, only given a walkie-talkie from a bystander.

The detective at the crime scene has all the important clues, like fingerprints, witness statements, layout of room etc. Because they have rich and relevant information, they can build and share a very accurate picture of what happened and predict the premium.

However, the detective with a walkie-talkie only gets limited, incomplete, and noisy data, making it very hard to form an accurate picture of what happened and make predictions.

Even if the detective (the random forest) has similar experience (hyperparameters) and similar ability to share ideas (intepretability features), interpretability is more useful with the detective at the crime scene vs the detective with the walkie-talkie.

Summing it up, interpreting a low R² model is ineffective, because it doesn't even have a clear picture of the relationship between the features and target variable to share with us. And this is bad because Insurance companies are required to justify their pricing to customers and regulators.

Deploying such a poor model will make it hard to provide clear, justifiable reasons for premium calculations.

Model Performance Comparison:
--------------------------------------------------------------------------------

Decision Tree:
RMSE: 853.15
MAE: 654.13
R2 Score: 0.0260
Cross-validation R2: 0.0263 (±0.0005)

XGBoost:
RMSE: 846.52
MAE: 646.46
R2 Score: 0.0411
Cross-validation R2: 0.0430 (±0.0008)

LightGBM:
RMSE: 847.40
MAE: 648.92
R2 Score: 0.0391
Cross-validation R2: 0.0395 (±0.0007)

Random Forest:
RMSE: 852.95
MAE: 654.09
R2 Score: 0.0264
Cross-validation R2: 0.0266 (±0.0006)

Linear Regression:
RMSE: 863.23
MAE: 667.20
R2 Score: 0.0028
Cross-validation R2: 0.0030 (±0.0001)

Support Vector Regression:
RMSE: 898.35
MAE: 637.90
R2 Score: -0.0799
Cross-validation R2: -0.0804 (±0.0020)


Linear regression has the lowest R2 of 0.0028, which is significantly lower than even the worst-performing tree-based models. Linear regression also has the highest RMSE. This shows that the relationship between the feature and target variable is not linear, and we can benefit from more complex models.

However, we have interesting results with Linear SVR. A negative R2 means that the model is performing worse than guessing the mean of the target variable. Linear SVR also has the highest RMSE by a significant margin. Paradoxically, it has the lowest MAE among all the models. It suggests that SVR is making some very large errors, which drive up the RMSE, but on average, the absolute errors are smaller than other models. 

Regardless, the combination of PCA and linear models seems to not be a good fit.


### selection verdict

None of these models perform well for real-world deployment. However, for the purpose of moving forward and choosing one model, the XGBoost is the least worst option, given the fact that is has higher R2 and lower RMSE and MAE.
