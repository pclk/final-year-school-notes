Generative AI
IT3386 – AI Services in Analytics
IT3386 AI SERVICES IN ANALYTICS - Generative AI
1
Official (Open)
Learning Outcomes
• Explain the differences between traditional machine learning models
and Generative AI models
• State the avenues provided by Google for usage of its Gen AI Services
• Explain the parameters associated with Google GenAI model and the
recommended settings for different use cases
• Describe prompt design concepts and strategies
IT3386 AI SERVICES IN ANALYTICS - Generative AI 2
Official (Open)
Generative AI in a Nutshell
IT3386 AI SERVICES IN ANALYTICS - Generative AI 3
Official (Open)
Introduction to Generative AI
• GenAI is a type of AI that creates
new content based on what it has
learned from existing content
• Process of learning from existing
content is called training and
results in the creation of a
statistical model
• Given a prompt, GenAI uses the
statistical model to predict what an
expected response might be and
this generates new content
IT3386 AI SERVICES IN ANALYTICS - Generative AI 4
Official (Open)
Introduction to Generative AI
IT3386 AI SERVICES IN ANALYTICS - Generative AI 5
• Discriminative
• Classify or predict
• Typically trained on a dataset of
labelled data
• Learns the relationship between the
features of the data points and labels
• Generative
• Generates new data that is similar to
the data it was trained on
• Understands distribution of data and
how likely a given example is
• Predict next word in a sequence
Official (Open)
Introduction to Generative AI
IT3386 AI SERVICES IN ANALYTICS - Generative AI 6
Official (Open)
Introduction to Generative AI
Traditional ML Generative AI
IT3386 AI SERVICES IN ANALYTICS - Generative AI 7
Official (Open)
Generative AI with Google
IT3386 AI SERVICES IN ANALYTICS - Generative AI 8
Official (Open)
Gen AI with Google – Gemini Chat
IT3386 AI SERVICES IN ANALYTICS - Generative AI 9
https://gemini.google.com
Official (Open)
Generative AI with Google
IT3386 AI SERVICES IN ANALYTICS - Generative AI 10
https://aistudio.google.com
Official (Open)
Gen AI with Google – Vertex AI
IT3386 AI SERVICES IN ANALYTICS - Generative AI 11
https://console.cloud.google.com > Vertex AI
Official (Open)
Gen AI with Google – Vertex AI Studio
IT3386 AI SERVICES IN ANALYTICS - Generative AI 12
https://console.cloud.google.com > Vertex AI
Official (Open)
Gen AI with Google – Model Garden
IT3386 AI SERVICES IN ANALYTICS - Generative AI 13
https://console.cloud.google.com > Vertex AI
Official (Open)
Gen AI with Google – Agent Builder
IT3386 AI SERVICES IN ANALYTICS - Generative AI 14
https://console.cloud.google.com > Agent Builder
Official (Open)
Gen AI with Google – 2 types of API
IT3386 AI SERVICES IN ANALYTICS - Generative AI 15
Official (Open)
Large Language Models
• A large language model (LLM) is a statistical
language model, trained on a massive amount
of data, that can be used to generate and
translate text and other content, and perform
other natural language processing (NLP) tasks.
• LLMs are typically based on deep learning
architectures, such as the Transformer
developed by Google in 2017, and can be
trained on billions of text and other content.
• Large
• Number of parameters (weights)
• GPT-3: 175B
• Size of data for training
• GPT-3: 570GB text data (after filtering)
IT3386 AI SERVICES IN ANALYTICS - Generative AI 16
Official (Open)
Large Language Models
IT3386 AI SERVICES IN ANALYTICS - Generative AI 17
Source: O’Reilly
Official (Open)
Vectors, Tokens, Embeddings
• Vector
• A single-dimensional array
• Since machines only understand numbers, all data must
be converted to vectors
• Tokenization
• Tokens are basic units of data processed by LLMs
• A token can be a word, part of a word (subword), or
even a character — depending on the tokenization
process, which varies across LLMs
• Gemini/PaLM2 models, a token is equivalent to about 4
characters. 100 tokens are about 60-80 English words.
• Tokenizer encodes input (prompt) into tokens and
decodes output (response) back to text
• Context length determines how many tokens an LLM
can accept as inputs and generate as outputs; Gemini, 1
million tokens
• Tokens are representations of text in the form of a
vector
• Embeddings
• Tokens with semantic context
• Represent the meaning and context of the text
• Allow LLMs to understand the context, nuance and
subtle meanings of words and phrases.
• Result of model learning from vast amounts of text
data, and encode not just the identity of a token but its
relationships with other tokens
IT3386 AI SERVICES IN ANALYTICS - Generative AI 18
Official (Open)
Model Parameters
• Max output tokens: Specifies the maximum number of tokens
that can be generated in the response
• Temperature
• Controls the degree of randomness in token selection. The
temperature is used for sampling during response generation,
which occurs when topP and topK are applied.
• Lower temperatures are good for prompts that require a more
deterministic or less open-ended response, while higher
temperatures can lead to more diverse or creative results.
• A temperature of 0 is deterministic, meaning that the highest
probability response is always selected.
• High temperatures are good for creating poems and stories but
beware of hallucination
• Hallucinations are words or phrases that are generated by the
model that are often nonsensical or grammatically incorrect.
It happens because:
• Model not trained on enough data
• Model trained on noisy data
• Model not given enough context
• Model not given enough constraints
• topK: changes how the model selects tokens for output. A
topK of 1 means the selected token is the most probable
among all the tokens in the model's vocabulary (also called
greedy decoding), while a topK of 3 means that the next
token is selected from among the 3 most probable. For each
token selection step, the topK tokens with the highest
probabilities are sampled. Tokens are then further filtered
based on topP with the final token selected using
temperature sampling.
• topP: changes how the model selects tokens for output.
Tokens are selected from the most to least probable until the
sum of their probabilities equals the topP value. For example,
if tokens A, B, and C have a probability of 0.3, 0.2, and 0.1 and
the topP value is 0.5, then the model will select either A or B
as the next token by using the temperature and exclude C as
a candidate. The default topP value is 0.95.
• Stop sequence: tells the model to stop generating text.
• Set to "." and the model will stop after one sentence
• Set to custom sequence in few-shot prompts and model will
stop generating more examples
IT3386 AI SERVICES IN ANALYTICS - Generative AI 19
Official (Open)
Prompt Design
• Prompt design is the process of creating prompts that elicit
the desired response from language models
• Writing well structured prompts is an essential part of
ensuring accurate, high quality responses from a language
model
• A prompt is a natural language request submitted to a
language model to receive a response
• Prompts can contain questions, instructions, contextual
information, examples, and partial input for the model to
complete or continue. After the model receives a prompt,
depending on the type of model being used, it can generate
text, embeddings, code, images, videos, music, and more.
• Prompt can include
• Input
• Context
• Examples (Zero-shot, One-shot, Few-shot)
Note
• Unlike traditional software that's designed to a carefully
written specifications, the behavior of generative models is
largely opaque even to the model trainers. As a result, you
often can't predict in advance what types of prompt
structures will work best for a particular model. What's more,
the behavior of a generative model is determined in large
part by its training data, and since models are continually
tuned on new datasets, sometimes the model changes
enough that it inadvertently changes which prompt
structures work best.
• Keep experimenting!
IT3386 AI SERVICES IN ANALYTICS - Generative AI 20
Official (Open)
Prompt
• Zero-shot
• No example
• Q&A, Summarisation, Continuation, etc.
• One-shot
• Single example
• Few-shot
• Multiple examples
• Stop sequence
Summarise the following into two sentences:
Generative artificial intelligence (AI) models
such as the Gemini family of models are able to
create content from varying types of data
input, including text, images, and audio. At
their most basic level, these models operate
like sophisticated autocomplete applications.
Given input text ("You can lead a horse to
water,"), a generative model can predict that
the output text that's statistically likely to
follow ("but you can't make it drink"), based
on patterns learned from their training data.
Generative AI models, like Gemini, create new
content from various inputs like text, images,
and audio. They function as advanced
autocomplete tools, predicting the most
statistically likely output based on patterns
learned from massive datasets.
IT3386 AI SERVICES IN ANALYTICS - Generative AI 21
Official (Open)
Prompt
• Zero-shot
• No example
• Q&A, Summarisation, Continuation, etc.
• One-shot
• Single example
• Few-shot
• Multiple examples
• Stop sequence
I was walking along the street
in Singapore when I saw
I was walking along the street
in Singapore when I saw a group
of people gathered around a man
who was sitting on the ground.
Parameters
- Temperature: 1
- Stop sequence: .
IT3386 AI SERVICES IN ANALYTICS - Generative AI 22
Official (Open)
Prompt
• Zero-shot
• No example
• Q&A, Summarisation, Continuation, etc.
• One-shot
• Single example
• Few-shot
• Multiple examples
• Stop sequence
Food: Apple
Pairs with: Cheese
Food: Orange
Pairs with:
Chocolate
IT3386 AI SERVICES IN ANALYTICS - Generative AI 23
Official (Open)
Prompt
• Zero-shot
• No example
• Q&A, Summarisation, Continuation, etc.
• One-shot
• Single example
• Few-shot
• Multiple examples
• Stop sequence
Generate a grocery shopping list for a week for one
person. Use the JSON format given below.
{"item": "eggs", "quantity": "6"}
{"item": "bread", "quantity": "one loaf"}
[{"item": "eggs", "quantity": "6"},
{"item": "bread", "quantity": "one loaf"},
{"item": "milk", "quantity": "one gallon"},
{"item": "yogurt", "quantity": "one container"},
{"item": "fruit", "quantity": "a few pieces"},
{"item": "vegetables", "quantity": "a few pieces"},
{"item": "meat", "quantity": "one pound"},
{"item": "cheese", "quantity": "one block"},
{"item": "pasta", "quantity": "one box"},
{"item": "rice", "quantity": "one bag"}]
IT3386 AI SERVICES IN ANALYTICS - Generative AI 24
Official (Open)
Prompt
• Zero-shot
• No example
• Q&A, Summarisation, Continuation, etc.
• One-shot
• Single example
• Few-shot
• Multiple examples
• Stop sequence
Write taglines for the following products.
Product description: A bagpack that is lightweight, durable and stylish
1. Effortless Adventure: Carry Your Essentials with Ease
2. Unstoppable Exploration: The Perfect Companion for Every Journey
3. Style and Substance: The Backpack that Combines Form and Function
===
Product description: A cutlery set for a more sustainable world
1. Culinary Revolution: Embrace Sustainable Dining
2. Earth-Conscious Cuisine: Cutlery for a Greener Plate
3. Elevate Your Meals with Eco-Friendly Elegance
===
Product description: A tee that showcases the local culture of Singapore
1. Wear Your Culture, Spread the Story: The Tee That Connects You to Singapore
2. The Heritage of the Lion City: Embroidered with the Essence of Singapore
3. Celebrate Singapore: The Tee That Pays Homage to Its Local Roots
IT3386 AI SERVICES IN ANALYTICS - Generative AI 25
Official (Open)
Prompt
• Other strategies
• Give the models instructions on what to do
• Make the instructions clear and specific
• Specify any constraints or formatting requirements for the output
• Including prompt-response examples in the prompt helps the model learn
how to respond
• Give the model examples of the patterns to follow instead of examples of
patterns to avoid
• Experiment with the number of prompts to include
• Too few examples – ineffective at changing model behaviour
• Too many examples – model overfits
• Use consistent formatting across examples
IT3386 AI SERVICES IN ANALYTICS - Generative AI 26
Official (Open)
Prompt (Specific Use Cases)
• Text classification
• Fraud detection: Classify whether transactions in
financial data are fraudulent or not.
• Spam filtering: Identify whether an email is spam
or not.
• Sentiment analysis: Classify the sentiment
conveyed in text as positive or negative. For
example, you can classify movie reviews or email
as positive or negative.
• Content moderation: Identify and flag content
that might be harmful, such as offensive
language or phishing.
• Best practices
• Try setting the temperature to zero and top-K to
one
• Classification tasks are typically deterministic, so
these settings often produce the best results
• Extraction prompts let you extract specific
pieces of information from text
• Named entity recognition (NER): Extract named
entities from text, including people, places,
organizations, and dates.
• Relation extraction: Extract the relationships
between entities in text, such as family
relationships between people.
• Event extraction: Extract events from text, such
as project milestones and product launches.
• Question answering: Extract information from
text to answer a question.
• Best practices
• Try setting the temperature to zero and top-K to
one
• Extraction tasks are typically deterministic, so
these settings often produce the best results
IT3386 AI SERVICES IN ANALYTICS - Generative AI 27
Official (Open)
Prompt (Specific Use Cases)
• Summarisation tasks extract the most
important information from text. You can
provide information in the prompt to help
the model create a summary, or ask the
model to create a summary on its own.
• Summarize text: Summarize text content
such as the following
• News articles
• Research papers
• Legal documents
• Financial documents
• Technical documents
• Customer feedback
• Content generation: Generate content for
an article, blog, or product description
• Best practices
• Specify any characteristics that you want
the summary to have
• For more creative summaries, specify
higher temperature, top-K, and top-P
values
• When you write your prompt, focus on the
purpose of the summary and what you
want to get out of it
IT3386 AI SERVICES IN ANALYTICS - Generative AI 28
