We're going to build a machine learning model on synthetic data and discuss AI Ethics with google collab.

First, we will mount our drive to access our insurance data, and import required libraries.

The libraries include data manipulation, exploration, modelling and utilities.

We read the insurance csv file into a dataframe, see that we have 20 columns and non-matching null values.

The head shows that we have numerical, categorical and date data.

Date data cannot be fed directly to our machine learning models, so we should convert them to numerical values.

To do this, we can extract cyclical patterns like which day of the year, month of the year and so on. We can also extract temporal distance information through days since earliest date.

Our dataset shows a high amount of missing values. These missing values occur in important columns like Previous Claims, Occupation and Credit score.

Why a value is missing can come from many reasons, and we should communicate with the data procurement team to understand why these values are missing, so that we can creation of interaction terms for the model to learn. For the purpose of this assignment, we will do mean imputation for numerical columns and mode imputation for categorical columns.

Zero values have a high chance of being misinterpreted. Sometimes it can be a placeholder value, or represent 0 occurrences or item. The data scientist should check with the parties regarding zero values as well. We will assume that all zeros do not mean placeholder or other values, and truly mean that the items described are at its zero value.

Categorical variable distribution is taken by df.value_counts converted to percentage. The distribution is so even that its likely synthetic data. These datasets should be labeled or documented as synthetic data.

Correlation matrix is plot using df.corr and sns.heatmap. The highest correlation is between Credit Score and Annual Income, both of which are features of our target variable, signifying multicollinearity. However, the correlation between our target variable, Premium Amount, and the rest of our features, is very weak. Though our correlation matrix measures linear correlation, this is a bad sign that models may perform badly.

Statistical summary is generated with df.describe, and Outlier analysis is based on the standard IQR * 1.5 formula. We've identified 3 columns with outliers, of which are generally in acceptably appropriate range. For example, its not uncommon to see people with the annual income of 150k, even if its uncommon in our dataset. Same applies to the other outlying columns.

Sorting the table with df.sort_values and selecting 3 columns reveals that its not conclusive whether the annual income and previous claims are correlated with Premium Amount, given the rather random values that appear. This is strange because in real life, higher income individuals tend to insure more assets, resulting in higher premiums. Higher previous claims also should result in higher premiums if we follow the industry practice. This points that the data is not only synthetic, but also doesn't reflect reality well.

Skewness is determined with df.skew, and our 3 outlying columns are shown to be highly skewed as expected. We can use scaling to correct this, but usually not for tree columns since they work in thresholds, not distances or gradients. For example, is annual income more than or equal to 45k? We should use scaling for linear regression and SVM, because they are scale dependent.

We prepare our data by extracting date information,

applying mean and mode imputation with df.fillna,

one-hot encoding categorical variables with pd.get_dummies and the drop first argument,

and transforming data using np.log1p, then trying out and settling with stats.boxcox due to better normalization.

Feature engineering portion is moved to after the training and evaluation of Tree-based models. We can also try dimensionality reduction techniques like PCA, but only after the Tree-based models given the opportunity to take advantage of their interpretability.

Modelling data is prepared by separating the feature columns, and splitting the data using train test split. Models and K-fold are initialized by initializing their classes. 

By looping over the models, we train and store their evaluation results in a dictionary. Progress bars and training time are implemented with tqdm and time.time.

Model evaluation results showed XGBoost and LightGBM doing better than Decision Tree and Random Forest. R squared values are around 0.02 to 0.04, meaning around 96-98% of premium amounts are unexplained by the dataset. The likely cause is the dataset. Commercially deploying any of these models to production is an ethical oversight that can result in financial instability to the insurance company, as well as making it hard to explain to customers and regulators how the model derives the predictions, since its trained on data which is noisy and uncorrelated with the premium amount.

Unrealistic feature ranking from the random forest's feature importance also points to poor data quality. It is likely that the models are fitting to the artifacts of the synthetic data generation process, and not real world interactions of the premium amounts and the features.

The error analysis and percentile, calculated with df.describe and np.percentile shows us that these errors are quite large and widely variable. 

Linear regression and SVM training starts with re-assigning new feature columns that replaces the original skewed columns with the transformed columns. Training and evaluation then commences similarly to the tree-based models, then we can evaluate and compare the models by printing the model results dictionary.

Linear regression has by far the worst R sqaured at 0.0028, and also has the second highest RMSE. However, Linear SVC interestingly has a negative R squared, and the highest RMSE by a large margin. Paradoxically, it has the lowest MAE, meaning that the SVR is making very large errors that drive up RMSE but on average, the absolute errors are smaller than the other models. None of these models should be deployed in the real world, but to choose a model, XGBoost has the best metrics.

Best hyperparameters are found using RandomisedCVSearch, which is similar to GridSearchCV but much faster with very similar results. The best cross vaildation score is not much further than the untuned model, signifying that poor model performance is likely not due to suboptimal parameters, but poor data quality.

In conclusion, poor data quality is the biggest issue, and the best next step for the insurance company is to obtain real-world insurance data, documenting and validating its procurement procedures. Interpretability and explainability of the model is important to provide understanding and trust to customers and regulators when needed. Public documents about the how the AI system functions, how the data is derived, and how decisions are made with the predictions will allow more transparency. Sometimes, the most ethical decision is to not deploy models that could perpetuate unfair practices, harm customers, and confuse insurance agents. Thank you.
