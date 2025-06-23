# {{Generative AI}} is a type of artificial intelligence that creates new content based on learning from existing content
The answer is "Generative AI" because this represents a fundamental shift in AI capabilities - unlike traditional systems that simply process or categorize existing data, generative AI can create original content. This creative capability makes it a transformative technology, similar to how humans can create new things based on their learned experiences.

# The training process of Generative AI creates a {{statistical model}} to {{predict expected responses}} from prompts
The answers are:
- statistical model: Because AI doesn't truly "understand" content like humans do - it builds a mathematical representation of patterns in data, which is fundamentally statistical in nature. This helps remember that GenAI is based on probability and patterns, not true comprehension.
- predict expected responses: This highlights the fundamental purpose of the model - it's making educated guesses about what response would be most appropriate given the input, based on its training data patterns.

# Traditional ML (Discriminative) models focus on {{classification}} and {{prediction}}, and are trained on {{labeled data}}
The answers are:
- classification: This emphasizes how traditional ML models sort things into predefined categories, like sorting emails into spam/not spam - a fundamentally different task from generation
- prediction: Traditional ML excels at predicting specific outcomes based on input features, like predicting house prices
- labeled data: This is crucial because it shows how traditional ML needs explicit human guidance through labeled examples, unlike generative AI which can learn from raw, unlabeled data

# In Traditional ML, the model {{learns relationships between features and labels}}, while Generative AI {{understands the distribution of data}} and {{predicts the next word in sequence}}
The answers are:
- learns relationships between features and labels: This highlights how traditional ML is about mapping input characteristics to specific outputs, like connecting visual features to object labels
- understands the distribution of data: This means generative AI learns the overall patterns and structure of the data, allowing it to create new examples that fit those patterns
- predicts the next word in sequence: This is the fundamental mechanism behind many generative AI models - they predict what should come next based on context, whether it's text, images, or other data types

# There are two types of Gemini APIs: {{Google AI Gemini API}} and {{Google Cloud Vertex AI Gemini API}}
The answers are:
- Google AI Gemini API: This is the simpler, more accessible version for individual developers and smaller projects
- Google Cloud Vertex AI Gemini API: This is the enterprise-focused version with additional security and infrastructure features
This distinction is crucial because it shows Google's two-pronged approach to serving different user segments with the same underlying technology.

# Both APIs offer access to the same models: {{Gemini Pro}} and {{Gemini Ultra}}
The answers are:
- Gemini Pro: The standard model suitable for most applications
- Gemini Ultra: The more advanced and powerful model
Understanding this is important because it shows that model access isn't the differentiator between the APIs - the difference lies in the features and infrastructure around them.

# For authentication, Google AI Gemini API uses {{API key}}, while Google Cloud Vertex AI Gemini API uses {{Google Cloud service account}}
The answers are:
- API key: A simpler authentication method typical for developer-focused APIs
- Google Cloud service account: A more sophisticated authentication system that provides better security and access control
This difference reflects the security needs of different user types - individual developers versus enterprise customers.

# The playground interfaces differ: Google AI Gemini API uses {{Google AI Studio}}, while Google Cloud Vertex AI Gemini API uses {{Vertex AI Studio}}
The answers are:
- Google AI Studio: A streamlined interface for developers to experiment with the API
- Vertex AI Studio: A more comprehensive environment integrated with Google Cloud's broader ML tools
This distinction is important as it shows how each interface is tailored to its target user's needs.

# In terms of free tier, Google AI Gemini API offers {{a basic free tier}}, while Google Cloud Vertex AI Gemini API provides {{$300 Google Cloud credit for new users}}
The answers are:
- a basic free tier: Suited for developers to test and build small applications
- $300 Google Cloud credit for new users: A more substantial offering to let businesses evaluate the full enterprise capabilities
This difference in pricing structure reflects the different scales of intended usage.

# Google Cloud Vertex AI Gemini API offers three key enterprise features: {{customer encryption key}}, {{virtual private cloud}}, and {{data residency}}
The answers are:
- customer encryption key: Allows enterprises to maintain control over their data encryption
- virtual private cloud: Provides isolated network infrastructure for enhanced security
- data residency: Enables compliance with data location requirements
These features are crucial for enterprise adoption as they address key security and compliance requirements that large organizations typically have.

# Both APIs have a quota of {{60}} requests per minute, which can be {{increased upon request}}
The answers are:
- 60: The default rate limit that prevents abuse and ensures fair resource distribution
- increased upon request: Flexibility to scale up for legitimate use cases
This is important to understand for planning application architecture and scaling considerations.

I'll create cloze deletion flashcards that capture the key concepts about LLMs while maintaining the hierarchical structure.

# A Large Language Model (LLM) is a {{statistical language model}} trained on {{massive data}}, based on {{deep learning architectures like Transformer}}
The answers are:
- statistical language model: This emphasizes that LLMs are fundamentally probability-based systems, not rule-based like traditional language processing
- massive data: This is crucial because the scale of training data directly impacts the model's capabilities and knowledge breadth
- deep learning architectures like Transformer: The Transformer architecture, introduced by Google in 2017, revolutionized NLP by enabling better understanding of long-range dependencies in text

# LLMs are characterized by scale in two dimensions: {{large number of parameters}} (e.g., GPT-3: 175B) and {{large training data size}} (e.g., GPT-3: 570GB text data)
The answers are:
- large number of parameters: Parameters are the adjustable weights that determine how the model processes information
- large training data size: The amount of text data the model learns from
This scale is what enables LLMs to capture complex patterns in language and knowledge.

# In the technical components of LLMs, {{vectors}} are used because {{machines can only understand numbers}}
The answers are:
- vectors: Single-dimensional arrays that represent data mathematically
- machines can only understand numbers: This fundamental limitation means all text processing must be converted to numerical representations
This helps remember why we need mathematical representations of language for machine learning.

# In tokenization, {{tokens}} are {{basic units of data}} that can be a {{word, part of word, or character}}
The answers are:
- tokens: The fundamental units that LLMs process
- basic units of data: This emphasizes that tokens are the atomic elements of processing
- word, part of word, or character: This flexibility in token size allows for efficient representation of different languages and concepts
Understanding tokenization is crucial because it's how text is converted into a format the model can process.

# In Gemini/PaLM2, {{~4 characters}} make up one token, and {{100 tokens}} is approximately equal to {{60-80 English words}}
The answers are:
- ~4 characters: This helps estimate how much text can fit in a given token limit
- 100 tokens: A standard unit of measurement for LLM capacity
- 60-80 English words: This real-world equivalence helps in planning prompt lengths
These ratios are important for understanding model limitations and planning usage.

# Gemini has a context length of {{1 million tokens}}, which is significant because it determines {{how much information the model can consider at once}}
The answers are:
- 1 million tokens: This large context window is a major advancement
- how much information the model can consider at once: This directly impacts the model's ability to maintain coherence and context in longer conversations
Context length is crucial for complex tasks requiring memory of earlier information.

# {{Embeddings}} are tokens with {{semantic context}} that {{represent meaning}} and allow understanding of {{context, nuance, and subtle meanings}}
The answers are:
- Embeddings: The mathematical representations that capture meaning
- semantic context: The relationship between words and their meanings
- represent meaning: Converting linguistic concepts into mathematical space
- context, nuance, and subtle meanings: This enables the model to understand language beyond literal definitions
Embeddings are crucial because they're how LLMs understand relationships between concepts.

# The {{max output tokens}} parameter determines the maximum length of the generated response
The answer is "max output tokens" because this is the fundamental way to control response length in language models. Understanding this is crucial as it affects both resource usage and cost - longer responses consume more tokens and therefore cost more to generate.

# {{Temperature}} is a parameter that controls {{randomness}} in token selection, where {{lower temperatures}} lead to more deterministic responses
The answers are:
- Temperature: A fundamental parameter that affects the creativity vs. reliability trade-off
- randomness: This describes how the model chooses between different possible next tokens
- lower temperatures: Close to 0, making the model choose the most probable tokens consistently
This parameter is crucial because it lets us control whether we want precise, factual responses (low temperature) or creative, varied outputs (high temperature).

# A temperature of {{0}} means the model will {{always select the highest probability response}}
The answers are:
- 0: The minimum temperature setting
- always select the highest probability response: The most deterministic behavior possible
This is important to remember because temperature=0 is often used for tasks requiring consistency and accuracy, like coding or fact-based responses.

# {{Higher temperatures}} are better for {{creative tasks}} but increase the risk of {{hallucinations}}
The answers are:
- Higher temperatures: Temperature settings closer to 1 or above
- creative tasks: Tasks like poetry or storytelling that benefit from variety
- hallucinations: Generated content that's incorrect or nonsensical
This relationship helps understand the trade-off between creativity and reliability in language models.

# The {{topK}} parameter controls token selection by limiting choices to the {{K most probable tokens}}
The answers are:
- topK: A parameter that provides direct control over the selection pool size
- K most probable tokens: The number of top candidates the model can choose from
Understanding topK is important because it's a more direct way to control randomness compared to temperature.

# {{topP}} sampling selects tokens from most to least probable until their probabilities sum to {{0.95}} by default
The answers are:
- topP: Also known as nucleus sampling, this provides more nuanced control than topK
- 0.95: The default threshold that captures most of the probability mass while excluding unlikely tokens
This parameter is crucial because it adapts to the probability distribution of tokens, unlike topK which uses a fixed number.

# {{Stop sequences}} can be set to {{custom values}} or {{simple punctuation}} like "." to control generation length
The answers are:
- Stop sequences: Special triggers that end text generation
- custom values: Specific phrases or patterns defined by the user
- simple punctuation: Basic markers like periods or newlines
This is important because it provides fine-grained control over response formatting and length.

# Hallucinations in language models are caused by four main factors: 1. {{Insufficient training data}} 2. {{Noisy training data}} 3. {{Insufficient context}} 4. {{Insufficient constraints}}
The answers are:
- Insufficient training data: The model hasn't seen enough examples to learn patterns properly
- Noisy training data: The training data contained incorrect or inconsistent information
- Insufficient context: The prompt doesn't provide enough information for accurate generation
- Insufficient constraints: The model has too much freedom in generating responses
Understanding these causes is crucial for preventing hallucinations by addressing each factor through proper prompt engineering and model selection.

# {{Prompt design}} is the process of creating prompts to get {{desired responses}} from models through {{natural language requests}}
The answers are:
- Prompt design: A crucial skill in working with language models, as it bridges human intent and model capabilities
- desired responses: The specific outputs we want the model to generate
- natural language requests: The human-readable instructions given to the model
This fundamental concept shows how we communicate with AI models effectively.

# There are three main types of prompts: {{zero-shot}} (no examples), {{one-shot}} (single example), and {{few-shot}} (multiple examples)
The answers are:
- zero-shot: The simplest form where the model must understand and execute without examples
- one-shot: Provides one example to guide the model's response
- few-shot: Uses multiple examples to establish patterns
Understanding these types helps choose the right approach based on task complexity and desired accuracy.

# Best practices for prompt design include giving {{clear instructions}}, specifying {{constraints and formatting}}, and showing {{patterns to follow}}
The answers are:
- clear instructions: Explicit directions that leave no room for ambiguity
- constraints and formatting: Specific requirements for the output structure and limitations
- patterns to follow: Good examples that demonstrate the desired behavior
These practices are essential because they help the model understand exactly what we want.

# When using examples in prompts, it's important to maintain {{consistent formatting}} and find the right balance as {{too few examples}} are ineffective while {{too many examples}} lead to overfitting
The answers are:
- consistent formatting: Uniform structure across all examples to establish clear patterns
- too few examples: Not enough to establish the desired pattern
- too many examples: Can cause the model to become too rigid in its responses
This balance is crucial for optimal model performance.

# For text classification tasks like {{fraud detection}}, {{spam filtering}}, and {{sentiment analysis}}, the best practice is to set {{temperature to zero}} and {{top-K to one}}
The answers are:
- fraud detection: Identifying fraudulent activities
- spam filtering: Categorizing unwanted messages
- sentiment analysis: Determining emotional tone
- temperature to zero: Ensures maximum determinism
- top-K to one: Forces selection of most probable token
These settings are important because classification tasks need consistent, deterministic responses.

# Extraction tasks include {{named entity recognition}}, {{relation extraction}}, and {{event extraction}}, and also require {{temperature zero}} for accuracy
The answers are:
- named entity recognition: Identifying specific entities like names, places, organizations
- relation extraction: Understanding relationships between entities
- event extraction: Identifying specific events and their components
- temperature zero: Ensures consistent, reliable extraction
These tasks require high precision, hence the strict parameter settings.

# For summarization tasks, best practices include specifying {{desired characteristics}} and using {{higher temperature}} and {{higher top-K/P values}} for creative summaries
The answers are:
- desired characteristics: Specific qualities wanted in the summary
- higher temperature: Allows for more varied language and expression
- higher top-K/P values: Enables more diverse word choices
These settings are important because summarization often requires a balance between accuracy and creativity.
