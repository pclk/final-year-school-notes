
















































# Folder structure

We have .github, app, property-predictor, tests, used-car-predictor, .gitignore, and README.md.

.github includes a ci-cd.yml, which instructs github to run the test files whenever a new push to main is detected.

app contains our Next.JS app, which we will show later.

property-predictor is my machine learning project, and used-car-predictor is ethan's. 

It contains a .dvc, bentoml which hosts our bentoml service file, configs which hosts our hydra config.yml file, datasets, experiments and their respective .dvc files which would be available with a dvc pull command, notebooks, another .gitignore to exclude the datasets and experiments folder from github, and the pyproject.toml, which allows poetry install to install dependencies, and a requirements.txt file just in case you face any dependencies issues.

# DVC
