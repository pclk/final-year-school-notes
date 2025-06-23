# 16 October 2024

## Tests Information

### Test 1
This is a closed book test.

Past students mentioned that this test is difficult because it's hard to tell what is important in the test.

On Week 3, cher will provide a revision for this test and go through what is being tested.

Generally the questions in this test won't feature any practical questions, more of theory.

### Practical Test
Two parts to the test:
- Theory-based MCQ and Written response
> Will be mainly focused on Week 4 and 5 topics
- Practical involving code to perform tests.
> For practical, anything not shown in practical won't be tested. 

> Processing JSON response is important for practical

> For first week's practical, you only need to know how to perform Face, Landmark and Document Text Detection and draw boxes for them

### Assignment
- Vertex AI will be tested for assignment
- Find your own data, put in google data bucket, ingest, process, train a model and deploy
- Assignment topic most likely would be Object Detection-related
- Take note of model training and deploying time, going above specified budget will cost a lot of marks

> Side note: cher mentioned that Deep Learning module will be utilizing local computer, not colab for practical


# Important Points
Study and Memorize the following:
- Machine Learning Workflow
- Implementation types: DIY, Own Model, Own Data, API Service
- Know and understand why the different implementation types hold different levels of risk
- You have to set up infrastructure for DIY but not for other impl types 
- Probably good to have an example in mind for each impl types
  - DIY: Locally hosted & deployed models
  - Own Model: Google Colab
  - Own Data: AutoML
  - API: Google Cloud Vision API
- Service providers
  - Google Cloud Machine Learning
  - Microsoft Cognition Services
  - AWS
- Who usually uses what tools
  - Data Scientist: Vertex AI, Vertex AI Workbench
  - Developer: Vertex AI, AutoML, API
  - Engineer: Deep Learning Containers & VM image
- When to use Vertex AI, AutoML, API, Deep Learning Containers & VM image
- AutoML service types:
  - Vision
  - Tables
  - Natural Language
  - Video Intelligence
  - Translation
- Know how API works 

# Cher drawing session
> cher spent 30 mins going through the implementation types

## DIY
build yourself
- local: need GPU
- cloud: colab, VPS
more flexibility and control than others

## AutoML
- Google cloud: Vertex AI
- Let cloud service provider(CSP) do the training and you provide data
- Pros:
  - Most of the AI tasks probably has a AutoML solution
  - Very easy; Takes care of everything after Data prep
  - Similar to DIY in terms of model potential
- Cons:
  - You don't have as much control as DIY
  - You are limited by CSP. If they don't support LLM training, you have to DIY
  - Technical process of training the model won't be shown to you and is trade secret

## API Service
good and easiest to use, except when specific model for your task may not be available, like detecting cancer cells
