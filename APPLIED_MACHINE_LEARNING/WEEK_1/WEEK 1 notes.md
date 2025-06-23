# 15 Oct

## Machine learning process: CRISP-DM
1. Business Understanding
2. Data Understanding
3. Data Preparation
up down
4. Modeling
5. Evaluation
6. Deployment

### Business Understanding
Does this belong to supervised /unsupervised?
Is machine learning necessary?
1. Determine Business Objectives
> (Profit, revenue?)

2. Assess situation
> Determine the following:
  - Resource availability
  - Project requirements 

Assess Risks and contingencies

Conduct cost-benefit analysis

3. Determine data mining goals
> Take a technical view and define success criteria for data mining

4. Product project plan
- Select tech & tools
- Define detailed plans for each project phase
- Understand customer needs; don't just build a model for the sake of it

Formulate the problem.

- Find out what you want to predict
- Frame the problem in terms of input and output?
- Strive for the simplest solution that serves your needs

### Data Understanding
understand and familarize
lets see what data we have
can we use this data?
can we get more data?
can we get higher quality data?

### Data Preparation
fix your data
majority missing values
- drop row because few lines of missing data
- drop column if a lot of missing data in column
- replace with mean data
duplicate values
inconsistent formatting

### Modeling
Determine supervised/unsupervised
if labeled: supervised
if output not number: regress
if output not number: classification
if not labeled: unsupervised

### Evaluation
split train test

#### Supervised
regression metrics:
low is good
- RMSE 
- MSE 

classification metrics:
high is good
- accuracy

### Deployment
if your model is good enough, can check stuff and deploy on webpage
