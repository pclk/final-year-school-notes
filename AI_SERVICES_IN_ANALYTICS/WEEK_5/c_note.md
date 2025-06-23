# 1. Introduction to Generative AI

1. Basic Definition
   - GenAI creates new content based on learning from existing content
   - Training process creates a statistical model to predict expected response from prompts

2. Difference between Traditional ML and Generative AI
   - Discriminative (Traditional ML):
     - Classify or predict
     - Trained on labeled data
     - Learns relationship between features and labels
   
   - Generative:
     - Generates new data similar to training data
     - Understands distribution of data
     - Predicts next word in sequence

# 2. GenAI: 2 Types of API

1. Google AI Gemini API
   - Models: Gemini Pro and Gemini Ultra
   - Sign up: Google account
   - Authentication: API key
   - User interface playground: Google AI Studio
   - API & SDK support: Python, Node.js, Android (Kotlin/Java), Swift, Go
   - Free tier: Yes
   - Quota per minute: 60 (can request increase)
   - Enterprise support: No

2. Google Cloud Vertex AI Gemini API
   - Models: Gemini Pro and Gemini Ultra
   - Sign up: Google Cloud account (requires terms agreement and billing)
   - Authentication: Google Cloud service account
   - User interface playground: Vertex AI Studio
   - SDK support: Python, Node.js, Java, Go
   - Free tier: $300 Google Cloud credit for new users
   - Quota per minute: 60 (can request increase)
   - Enterprise features:
     - Customer encryption key
     - Virtual private cloud
     - Data residency

# 3. Large Language Models (LLMs)

1. Definition
   - Statistical language model trained on massive data
   - Based on deep learning architectures like Transformer (Google, 2017)
   - Used for text generation, translation, and NLP tasks

2. Scale Characteristics
   - Large number of parameters (Example: GPT-3: 175B)
   - Large training data size (Example: GPT-3: 570GB text data after filtering)

3. Technical Components
   - Vectors: Single-dimensional arrays (machines only understand numbers)
   - Tokens:
     - Basic units of data (word, part of word, or character)
     - In Gemini/PaLM2: ~4 characters per token
     - 100 tokens ≈ 60-80 English words
     - Context length in Gemini: 1 million tokens
   - Embeddings:
     - Tokens with semantic context
     - Represent meaning and context
     - Allow understanding of context, nuance, subtle meanings
     - Result from model learning relationships between tokens

# 3. Model Parameters

1. Parameters and Their Functions
   - Max output tokens: Maximum tokens generated in response
   - Temperature:
     - Controls randomness in token selection
     - Lower temperatures: More deterministic, less open-ended
     - Temperature of 0: Highest probability response always selected
     - Higher temperatures: Good for poems/stories but risk hallucination
   
   - topK:
     - Controls token selection
     - topK=1: Most probable token selected
     - topK=3: Selection from 3 most probable tokens
   
   - topP:
     - Default: 0.95
     - Tokens selected from most to least probable until sum equals topP value
   
   - Stop sequence: Stops text generation
     - Can be set to "." for one sentence
     - Can use custom sequence in few-shot prompts

2. Hallucinations
   - Definition: Words/phrases that are nonsensical/grammatically incorrect
   - Causes:
     - Model not trained on enough data
     - Model trained on noisy data
     - Model not given enough context
     - Model not given enough constraints

# 4. Prompt Design

1. Basic Definition
   - Process of creating prompts for desired responses
   - Natural language request to get model response
   - Can include: input, context, examples

2. Types of Prompts
   - Zero-shot: No examples
   - One-shot: Single example
   - Few-shot: Multiple examples

3. Best Practices
   - Give clear instructions on what to do
   - Make instructions clear and specific
   - Specify constraints/formatting requirements
   - Include prompt-response examples
   - Show patterns to follow (not patterns to avoid)
   - Use consistent formatting across examples
   - Experiment with number of examples:
     - Too few: Ineffective at changing behavior
     - Too many: Model overfits

4. Specific Use Cases
   - Text Classification:
     - Uses: Fraud detection, spam filtering, sentiment analysis, content moderation
     - Best practice: Set temperature to zero and top-K to one
   
   - Extraction:
     - Uses: Named entity recognition, relation extraction, event extraction, question answering
     - Best practice: Set temperature to zero and top-K to one
   
   - Summarization:
     - Uses: News articles, research papers, legal documents, financial documents, technical documents, customer feedback
     - Best practices:
       - Specify desired characteristics
       - Use higher temperature, top-K, and top-P for creative summaries
       - Focus on summary purpose in prompt
