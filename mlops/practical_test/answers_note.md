# deck: mlops answers

test -test-

Data Drift in Machine Learning is a situation where the statistical properties of the target variable (what the model is trying to predict) change over time?
False

Only project teams that have many models in production are recommended to apply MLOps due to its high cost and time-consuming process?
False

Feature Generation ensures data and artifacts are versioned to ensure reproducibility?
False

Defining the project is the first stage in Machine Learning Operations which includes defining data and establishing a baseline?
False

In MLOps, we just need to focus on data versioning to ensure reproducibility and facilitate auditing?
False

MLFlow tracking component serves as the registration component where models are stored, annotated and managed in a central repository?
False

An underfitting model sees patterns in noise and seeks to predict every single variation, resulting in a complex model that does not generalise well beyond its training data?
False

Automated feature selection can help to estimate how critical some features will be for the predictive performance of the model?
True

Data Drift in Machine Learning is a situation where the statistical properties of the target variable (what the model is trying to predict) change over time?
False

-Data versioning- ensures reproducibility in model creation as we can store the datasets we used to train a model.

Machine Learning artifact includes libraries with specific versions and environment variables?
True

The throughput of batch scoring cannot be Increased by deploying multiple instances of the model due to the whole datasets are processed using one model only?
False

-Intentionality- in Responsible Al includes assurance that data used for Al projects comes from compliant and unbiased sources plus a collaborative approach to Al projects that ensures multiple checks and balances on potential model bias.

The -data feedback loop- helps the model learn from its mistakes by returning incorrectly predicted data.

Which of the following factors need to be considered when we want to scale out the deployment of the Machine Learning Model?
I. Load-Balance II. Data Partition IV. Caching

A feature that can take three values (e.g., Raspberry, Blueberry, and Strawberry) is transformed into three features that can take only two values-yes or no (e.g., Raspberry yes/no, Blueberry yes/no, Strawberry yes/no). To achieve the outcome, which of the following feature engineering techniques we can use?
one-hot encoding 

Which of the following is NOT a component needed for building a Machine Learning model?
Log

Which parameter in setup is used to control multicollinearity treatment?
remove_multicollinearity

Which of the following is NOT an ML artifact?
Log files

Which of the following are maintenance measures after the release of ML models to production?
I. Resource monitoring II. Health check mechanism IV. ML metrics monitoring

what does the predict_modely tunction return in Pycarets regression module?
A dataframe with original features and predictions

MLOps as the process of automating machine learning using DevOps methodologies. Briefly describe TWO (2) best practices from DevOps that have been adopted in MLOps?
Continuous Integration
Continuous delivery

What is Continuous Integration? It's the process of -continuously testing- a -software project- and -improving the quality- based on -these tests’ results-. It is -automated testing- using -open source-, -SaaS- or -cloud native build servers-.

What is Continuous Integration? 
It's the process of continuously testing a software project and improving the quality based on these tests’ results. It is automated testing using open source, SaaS or cloud native build servers.

What's Continuous delivery? -Delivers code- to a -new environment- without -human intervention-. Is the process of -deploying code automatically-, often through the use of -laC-.

What's Continuous delivery?
Delivers code to a new environment without human intervention. Is the process of deploying code automatically, often through the use of laC.


Briefly describe TWO (2) benefits of using Hydra? -Switch- between -different configuration groups- -easily-. -Automatically record execution results- showing the -code- and the -configuration- used.

Briefly describe TWO (2) benefits of using Hydra?
Switch between different configuration groups easily.
Automatic record execution results showing the code and the configuration used.

Briefly describe TWO (2) components of MLFlow that help in managing the Machine Learning lifecycle? MLFlow features a component known as the -tracking server- wherein you can store -parameters-, -model metrics-, -metadata models- and -artifacts-. This is for -experimentation- phase. The -model registry- component serves as the -registration component- where models are -stored-, -annotated- and -managed- in a -central repository-. This is for -deployment- and -operationalization- phases.

Briefly describe TWO (2) components of MLFlow that help in managing the Machine Learning lifecycle?
MLFlow features a component known as the tracking server wherein you can store parameters, model metrics, metadata models and artifacts. This is for experimentation phase
The model registry component serves as the registration component where models are stored, annotated and managed in a central repository. This is for deployment and operationalization phases.

Experimentation takes place throughout the entire model development process. Briefly describe TWO(2) goals that we can achieve through experimentation? -Finding- the best -modelling parameters- (-algorithms-, -hyperparameters-, -feature preprocessing-, etc.). -Finding- a -balance- between -model improvement- and -computation costs-. (Since there’s always -room for improvement-, how -good is good enough-?)

Experimentation takes place throughout the entire model development process. Briefly describe TWO(2) goals that we can achieve through experimentation?
Finding the best modelling parameters (algorithms, hyperparameters, feature preprocessing, etc.).
Finding a balance between model improvement and improved computation costs. (Since there’s always room for improvement, how good is good enough?)

Depending on the impact of the model's predictions, decisions, or classifications, a more or less deep understanding may be required. Briefly describe TWO (2) reasonable steps data scientists should take to ensure the model is not actively harmful? -Checking- how the -model reacts to different inputs—: e.g., -plot- the -average prediction- (or -probability- for -classification models-) for -different values- of some inputs and see whether there are -oddities- or -extreme variability-. -Splitting- one particular -dimension- and checking the -difference- in -behavior- and -metrics- across different -subpopulations—e.g., is the -error rate- the -same- for -males- and -females-?

Depending on the impact of the model's predictions, decisions, or classifications, a more or less deep understanding may be required. Briefly describe TWO (2) reasonable steps data scientists should take to ensure the model is not actively harmful?
Checking how the model reacts to different inputs—e.g., plot the average prediction (or probability for classification models) for different values of some inputs and see whether there are oddities or extreme variability
Splitting one particular dimension and checking the difference in behavior and metrics across different subpopulations—e.g., is the error rate the same for males and females?

Check how the model reacts to different inputs and see whether there are oddities or extreme variability.
Split a dimension and check the difference in behavior and metrics across different subpopulations.
For example, is the error rate the same for males and females?
