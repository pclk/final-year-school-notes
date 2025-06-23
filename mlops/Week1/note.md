# deck: mlops wk9

The main reasons in the Challenge of AI implementation are: 1.-lack- of 1.-talent-, -change management- and the 3.-lack- of 3.-automated systems-.

Though a single data scientist can develop a model in a few -weeks-, putting that same model in production will take -months- of work.

It's hard to deploy a model to production, because the model must be -scalable-, -reproducible- and -collaborative-.

Machine Learning Operations (MLOps) is a process that helps organizations and business leaders -generate long\-term value- and 1.-reduce risk- associated with -AI initiatives-.

The benefit of Machine Learning models in production is -increased profit margins-. 

We will see an 1.-exponential increase- in 1.-profit-, due to 2.-rapid adoption- of 2.-machine learning-.

-Scalable- Machine Learning model creation can solve 1.-growing demand- by 1.-increasing resources-.

-Reproducible- Machine Learning model creation can provide the 1.-same performance- and 1.-result- after 1.-several executions-.

-Collaborative- Machine Learning model creation has 1.-different profiles involved- in the 1.-development- of the model.

## MLOps Process

The MLOps processes are -Use case discovery-, -Data Engineering-, -Machine Learning Pipeline-, -Production deployment- and finally, -Production monitoring-.

In -Use Case Discovery-, we 1.-discover- the 1.-most critical aspects- of the 1.-business- and 2.-identify- its 2.-needs-. 

-Business Understanding- seeks to 1.-understand- the 1.-needs- of the 1.-business-.

-Use Case Identification- identifies -different ways- to 1.-solve- the 1.-needs- of the 1.-business-.

In -Data Engineering-, we 1.-analyze- the 1.-available data- and 2.-identify what other data we need-.

-Feasibilty Study- determines if your 1.-data- could 1.-possibly produce- a 1.-model-.

In -Machine Learning Pipeline-, we -train different algorithms- and the 1.-best- is 1.-identified- and 1.-optimized-.

In -Production Deployment-, we 1.-automate- the 1.-deployment- of the 1.-model-.

In -Production Monitoring-, we 1.-monitor- the 1.-performance- of the 1.-model-, and if it 2.-performs worse-, we could 2.-retrain- and 2.-redeploy-.

## DevOps, MLOps, DataOps

DevOps applied to Machine Learning is known as -MLOps-.

DevOps applied to Data is known as -DataOps-.

-DataOps- ensures 1.-high quality data- to 1.-train models-.

The ultimate goal of both DevOps and MLOps is the greater 1.-quality- and 1.-control- of 2.-software applications- and 2.-ML models-.

## Roles in MLOps

The -data scientist- would worry about -model training- and -understanding model results-.

The -data scientist manager- evaluates 1.-when- to 1.-retrain- the 1.-model- and -any changes required-. 

The -product manager- -identifies customer needs- and -publicize model limitations-.

The -data engineer- 1.-obtains-, 1.-extracts-, 1.-corrects- data.

The -quality manager- ensures model 1.-suitability- and 1.-safety-.

The -support- helps the business 1.-apply- the 1.-model-.

The -business stakeholder- 1.-values- the 1.-model- in the 1.-company-.

## Challenges solved by MLOps

-Model and Data versioning- ensures 1.-experiments- are 1.-reproducible-.

-Monitoring- a 1.-model- identifies 1.-data drift-. 

-Reusing functions- 1.-reduces resources- on 1.-Feature Engineering-, so we can focus on the 2.-design- and 2.-testing- of the 2.-model-.

## Parts of MLOps

The parts of MLOps are -Feature store-, -Data versioning-, -Metadata store-, -Model Version Control-, -Model Registration-, -Model Serving-, -Model Monitoring-, -Model Retraining- and -CI/CD-.

-Feature store- 1.-stores- the 1.-functions- used in the 2.-training- of a 2.-model-, to ensure -no duplicated functions-. 

-Features- can also be -searched- and used to -build other models- or -analyze data-.

-Functions- are -versioned- so that you can -revert-.

-Data Versioning- is essential during -audits- to -identify datasets used-.

-Metadata Store- 1.-stores- the 1.-configuration- and 1.-metadata- of the 1.-model-, like -random seed-, -hyper parameters-, -evaluation metrics-.

-Model Version Control- lets you 1.-switch models- in 1.-real time-, and is critical from the 2.-governance POV- of 2.-model- and 2.-compliance-.

-Model Registration- 2.-stores trained models- as a new 2.-record- along with 1.-metadata- and 1.-functions-.

It's better to use 1.-endpoints- with 1.-APIs-, compared to a -model container-, if we want to use the model in -several applications simultaneously-.

-Production Bias- occurs when the deployed model has a -different performance- than local model.

-Model Retraining- -improves performance- and -updates training data- in model.

-Continuous Integration/Continuous Deployment- ensures that -models- are 2.-built- and 2.-deployed frequently-.

-Continuous Delivery- ensures that -code- is -frequently merged- into a -central repository- where -automated builds- and -tests- are applied.

## Tools of MLOps

-Code versioning- tools include -GitHub-, -GitLab-, or -BitBucket-.

-Data labelling- tools include -V7- or -LabelBox-.

V7 is an -automated annotation platform- for -computer vision-.

V7 combines -dataset management-, 1.-image- and 1.-video annotation-, and -AutoML training- to -complete labelling tasks automatically-.

-Data visualization or exploration- tools include -Bokeh-, -Matplotlib-, or -Seaborn-.

-Feature engineering- tools include -Scikit learn-, -Featuretools-, or -Feast-.

-Model training- tools include -Scikit learn-, -Tensorflow-, or -Pycaret-.

-Model debugging- tools include -TensorBoard- or -interpretML-.

-Model tuning- tools include -Optima Tune- or -Keras Tuner-.

-Model tracking- tools include -MLFlow-, -Comet-, -ClearML-, or -TensorBoard-.

-Model packaging- tools include -Kubeflow-, -MLFlow-, -BentoML-, or -Onnx-.

-Model serving- tools include -Amazon Sage Maker-, -FastAPI-, or -BentoML-.

-Model orchestration monitoring- tools include -Kubeflow- or -Apache Airflow-.

-Model uptime monitoring- tools include -Seldon- or -Verta-.

-Data versioning- tools include -DVC-, -Pachyderm-, -Quilt-, or -Comet-.

-Model POC- tools include -Streamlit- or -Dash-.

