# Note

## Front

Data Drift in Machine Learning is a situation where the statistical properties of the target variable (what the model is trying to predict) change over time?

## Back

False

# Note

## Front

Only project teams that have many models in production are recommended to apply MLOps due to its high cost and time-consuming process?

## Back

False

# Note

## Front

Feature Generation ensures data and artifacts are versioned to ensure reproducibility?

## Back

False

# Note

## Front

Defining the project is the first stage in Machine Learning Operations which includes defining data and establishing a baseline?

## Back

False

# Note

## Front

In MLOps, we just need to focus on data versioning to ensure reproducibility and facilitate auditing?

## Back

False

# Note

## Front

MLFlow tracking component serves as the registration component where models are stored, annotated and managed in a central repository?

## Back

False

# Note

## Front

An underfitting model sees patterns in noise and seeks to predict every single variation, resulting in a complex model that does not generalise well beyond its training data?

## Back

False

# Note

## Front

Automated feature selection can help to estimate how critical some features will be for the predictive performance of the model?

## Back

True

# Note

## Front

Data Drift in Machine Learning is a situation where the statistical properties of the target variable (what the model is trying to predict) change over time?

## Back

False

# Note
model: Cloze

## Text

{{c1::Data versioning}} ensures reproducibility in model creation as we can store the datasets we used to train a model.

## Back Extra


# Note

## Front

Machine Learning artifact includes libraries with specific versions and environment variables?

## Back

True

# Note

## Front

The throughput of batch scoring cannot be Increased by deploying multiple instances of the model due to the whole datasets are processed using one model only?

## Back

False

# Note
model: Cloze

## Text

{{c1::Intentionality}} in Responsible Al includes assurance that data used for Al projects comes from compliant and unbiased sources plus a collaborative approach to Al projects that ensures multiple checks and balances on potential model bias.

## Back Extra


# Note
model: Cloze

## Text

The {{c1::data feedback loop}} helps the model learn from its mistakes by returning incorrectly predicted data.

## Back Extra


# Note

## Front

Which of the following factors need to be considered when we want to scale out the deployment of the Machine Learning Model?

## Back

I. Load-Balance II. Data Partition IV. Caching

# Note

## Front

A feature that can take three values (e.g., Raspberry, Blueberry, and Strawberry) is transformed into three features that can take only two values-yes or no (e.g., Raspberry yes/no, Blueberry yes/no, Strawberry yes/no). To achieve the outcome, which of the following feature engineering techniques we can use?

## Back

one-hot encoding

# Note

## Front

Which of the following is NOT a component needed for building a Machine Learning model?

## Back

Log

# Note

## Front

Which parameter in setup is used to control multicollinearity treatment?

## Back

remove_multicollinearity

# Note

## Front

Which of the following is NOT an ML artifact?

## Back

Log files

# Note

## Front

Which of the following are maintenance measures after the release of ML models to production?

## Back

I. Resource monitoring II. Health check mechanism IV. ML metrics monitoring

# Note

## Front

what does the predict_modely tunction return in Pycarets regression module?

## Back

A dataframe with original features and predictions

# Note

## Front

MLOps as the process of automating machine learning using DevOps methodologies. Briefly describe TWO (2) best practices from DevOps that have been adopted in MLOps?

## Back

Continuous Integration
Continuous delivery

# Note
model: Cloze

## Text

What is Continuous Integration? It's the process of {{c1::continuously testing}} a {{c2::software project}} and {{c3::improving the quality}} based on {{c4::these tests’ results}}. It is {{c5::automated testing}} using {{c6::open source}}, {{c7::SaaS}} or {{c8::cloud native build servers}}.

## Back Extra


# Note

## Front

What is Continuous Integration?

## Back

It's the process of continuously testing a software project and improving the quality based on these tests’ results. It is automated testing using open source, SaaS or cloud native build servers.

# Note
model: Cloze

## Text

What's Continuous delivery? {{c1::Delivers code}} to a {{c2::new environment}} without {{c3::human intervention}}. Is the process of {{c4::deploying code automatically}}, often through the use of {{c5::laC}}.

## Back Extra


# Note

## Front

What's Continuous delivery?

## Back

Delivers code to a new environment without human intervention. Is the process of deploying code automatically, often through the use of laC.

# Note
model: Cloze

## Text

Briefly describe TWO (2) benefits of using Hydra? {{c1::Switch}} between {{c2::different configuration groups}} {{c3::easily}}. {{c4::Automatically record execution results}} showing the {{c5::code}} and the {{c6::configuration}} used.

## Back Extra


# Note

## Front

Briefly describe TWO (2) benefits of using Hydra?

## Back

Switch between different configuration groups easily.
Automatic record execution results showing the code and the configuration used.

# Note
model: Cloze

## Text

Briefly describe TWO (2) components of MLFlow that help in managing the Machine Learning lifecycle? MLFlow features a component known as the {{c1::tracking server}} wherein you can store {{c2::parameters}}, {{c3::model metrics}}, {{c4::metadata models}} and {{c5::artifacts}}. This is for {{c6::experimentation}} phase. The {{c7::model registry}} component serves as the {{c8::registration component}} where models are {{c9::stored}}, {{c10::annotated}} and {{c11::managed}} in a {{c12::central repository}}. This is for {{c13::deployment}} and {{c14::operationalization}} phases.

## Back Extra


# Note

## Front

Briefly describe TWO (2) components of MLFlow that help in managing the Machine Learning lifecycle?

## Back

MLFlow features a component known as the tracking server wherein you can store parameters, model metrics, metadata models and artifacts. This is for experimentation phase
The model registry component serves as the registration component where models are stored, annotated and managed in a central repository. This is for deployment and operationalization phases.

# Note
model: Cloze

## Text

Experimentation takes place throughout the entire model development process. Briefly describe TWO(2) goals that we can achieve through experimentation? {{c1::Finding}} the best {{c2::modelling parameters}} ({{c3::algorithms}}, {{c4::hyperparameters}}, {{c5::feature preprocessing}}, etc.). {{c6::Finding}} a {{c7::balance}} between {{c8::model improvement}} and {{c9::computation costs}}. (Since there’s always {{c10::room for improvement}}, how {{c11::good is good enough}}?)

## Back Extra


# Note

## Front

Experimentation takes place throughout the entire model development process. Briefly describe TWO(2) goals that we can achieve through experimentation?

## Back

Finding the best modelling parameters (algorithms, hyperparameters, feature preprocessing, etc.).
Finding a balance between model improvement and improved computation costs. (Since there’s always room for improvement, how good is good enough?)

# Note
model: Cloze

## Text

Depending on the impact of the model's predictions, decisions, or classifications, a more or less deep understanding may be required. Briefly describe TWO (2) reasonable steps data scientists should take to ensure the model is not actively harmful? {{c1::Checking}} how the {{c2::model reacts to different inputs—: e.g., }}plot{{c3:: the }}average prediction{{c4:: (or }}probability{{c5:: for }}classification models{{c6::) for }}different values{{c7:: of some inputs and see whether there are }}oddities{{c8:: or }}extreme variability{{c9::. }}Splitting{{c10:: one particular }}dimension{{c11:: and checking the }}difference{{c12:: in }}behavior{{c13:: and }}metrics{{c14:: across different }}subpopulations—e.g., is the {{c15::error rate}} the {{c16::same}} for {{c17::males}} and {{c18::females}}?

## Back Extra


# Note

## Front

Depending on the impact of the model's predictions, decisions, or classifications, a more or less deep understanding may be required. Briefly describe TWO (2) reasonable steps data scientists should take to ensure the model is not actively harmful?

## Back

Checking how the model reacts to different inputs—e.g., plot the average prediction (or probability for classification models) for different values of some inputs and see whether there are oddities or extreme variability

# Note

## Front

Splitting one particular dimension and checking the difference in behavior and metrics across different subpopulations—e.g., is the error rate the same for males and females?

## Back



