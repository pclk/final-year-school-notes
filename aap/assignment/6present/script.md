# Work Advisor 10 seconds
Welcome everyone, I'm Wei Heng. Together with Ethan and Gavin, we present Work Advisor, your AI career guidance platform.

## Agenda (Wei Heng) 11 seconds
Today's 15 minute presentation will cover the problem and solution, AI models, and a live demo, along with the implemented features and UX principles.

## Problem (Team) 20 seconds
Wei Heng: Our team solves social frustrations that working professionals and students face. For salary, professionals today can't research accurate salary information without sacrificing their free time and energy. It's important to explore career opportunities and negotiate salary.
Ethan: ...
Gavin: ...

## Solution (Team) 15 seconds
Wei Heng: Work Advisor solves this with instant and free salary predictions. In just 30 seconds, you get personalized salary insights, cost of living comparisons and multi-country analysis.
Ethan: ...
Gavin: ...

## Model & Accuracy (Wei Heng) 1 min 5 seconds
We'll go through our modelling process starting with Salary, Post, and Education predictions

### Model & Accuracy - Salary Prediction (Wei Heng)
Clustering: I assigned each skill to clusters based on TF-IDF scores,
Word2Vec: and trained Word2Vec models separately.
Testing: I've tested the above features with a random forest model, and
DistilBERT: also trained a feature extractor DistilBERT model,
Results: which has done better than the ML models, achieving lower values when lower is better, and higher values for R^2.
BERT Size & Time: DistilBERT is also quite lightweight and quick,
Benchmark - predictsalary.com: so benchmark it further, I've found a competitor product which uses a Large Language Model to predict salary. This gave me an idea to...
Benchmark - OpenAI, Claude, Gemini: benchmark DistilBERT against 3 state of the art Large Language Models,
Benchmark - Big Deep: and a custom and larger deep learning model.
Benchmark - Results: DistilBERT managed to out-perform all of the deep learning models,
Benchmark - Conclusion: making it the clear choice for our deployment. Better accuracy, speed, cost, and ownership. Ethan will now share the post prediction approach.

### Model & Accuracy - Post Prediction (Ethan)
...

### Model & Accuracy - Education Prediction (Gavin)
...

## App Demo (2 mins)
### Home page (Wei Heng)
Now for the exciting part - let's see it in action! You can try it yourself at work-advisor.vercel.app.

### Salary (Wei Heng)
We're greeted by a product tour. This follows the UX principle of chunking, where our form is broken down into explained sections.
We have presets for our users to pick from,
these are the fields which our model receives,
and these special fields are looped as individual predictions. Take note that we've chosen US, Singapore and India.
The tour ends by encouraging us to make a prediction.
The streaming predictions follows the UX principle of Doherty, updating the user with new information as soon as possible.
We can see our predictions matching the 3 chosen countries, and immediately after the prediction, we get a generated report. Our AI suggests...
\[Briefly reads the report.]
It provides some guiding questions at the end.
\[Chats with chatbot]
You notice that it knows the median salary and cost of living, 
and you can find the source right here.
Let's save this and try another role.
Now comes the fun part, let's compare it to senior lecturer, and select multiple locations this time.
The grouped salary prediction display follows the Law of Common region, and
now, the best part, we can compare different careers very easily.
\[Read response]
That's our salary predictor! Over to the post predictor!

### Post (Ethan)
...

### Education (Gavin)
...

## Recommendation (Wei Heng) 50 seconds
### Salary (Wei Heng)
Thanks Gavin. The insights from creating salary predictor can be summarized as surprising performance and difference across countries.
I was surprised by DistilBERT outperforming bigger models, especially since the data is manually scraped, dirty and hard to work with.
I've found that India is more challenging to work in, not only because of the low salary, but also the high years of experience and working hours.
The US provides much higher salaries than Singapore despite having similar cost of living expenses. However, we have to consider safety and racial cohesion in Singapore, which may be more important to some.
All in all, companies and individuals can use this tool to gauge salary in the market and explore career opportunities respectively.

### Post (Ethan)
Thanks wei heng

## Thanks (Ethan)
...
