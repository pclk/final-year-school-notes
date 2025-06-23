model: Basic

# Note
model: Cloze

## Text
{{c1::Generative AI}} is a type of artificial intelligence that creates new content based on learning from existing content

## Back Extra
The answer is "Generative AI" because this represents a fundamental shift in AI capabilities - unlike traditional systems that simply process or categorize existing data, generative AI can create original content. This creative capability makes it a transformative technology, similar to how humans can create new things based on their learned experiences.

# Note
model: Cloze

## Text
The training process of Generative AI creates a {{c1::statistical model}} to {{c2::predict expected responses}} from prompts

## Back Extra
The answers are:
- statistical model: Because AI doesn't truly "understand" content like humans do - it builds a mathematical representation of patterns in data, which is fundamentally statistical in nature. This helps remember that GenAI is based on probability and patterns, not true comprehension.
- predict expected responses: This highlights the fundamental purpose of the model - it's making educated guesses about what response would be most appropriate given the input, based on its training data patterns.

# Note
model: Cloze

## Text
Traditional ML (Discriminative) models focus on {{c1::classification}} and {{c2::prediction}}, and are trained on {{c3::labeled data}}

## Back Extra
The answers are:
- classification: This emphasizes how traditional ML models sort things into predefined categories, like sorting emails into spam/not spam - a fundamentally different task from generation
- prediction: Traditional ML excels at predicting specific outcomes based on input features, like predicting house prices
- labeled data: This is crucial because it shows how traditional ML needs explicit human guidance through labeled examples, unlike generative AI which can learn from raw, unlabeled data

# Note
model: Cloze

## Text
In Traditional ML, the model {{c1::learns relationships between features and labels}}, while Generative AI {{c2::understands the distribution of data}} and {{c3::predicts the next word in sequence}}

## Back Extra
The answers are:
- learns relationships between features and labels: This highlights how traditional ML is about mapping input characteristics to specific outputs, like connecting visual features to object labels
- understands the distribution of data: This means generative AI learns the overall patterns and structure of the data, allowing it to create new examples that fit those patterns
- predicts the next word in sequence: This is the fundamental mechanism behind many generative AI models - they predict what should come next based on context, whether it's text, images, or other data types

# Note
model: Cloze

## Text
There are two types of Gemini APIs: {{c1::Google AI Gemini API}} and {{c2::Google Cloud Vertex AI Gemini API}}

## Back Extra
The answers are:
- Google AI Gemini API: This is the simpler, more accessible version for individual developers and smaller projects
- Google Cloud Vertex AI Gemini API: This is the enterprise-focused version with additional security and infrastructure features
This distinction is crucial because it shows Google's two-pronged approach to serving different user segments with the same underlying technology.

# Note
model: Cloze

## Text
Both APIs offer access to the same models: {{c1::Gemini Pro}} and {{c2::Gemini Ultra}}

## Back Extra
The answers are:
- Gemini Pro: The standard model suitable for most applications
- Gemini Ultra: The more advanced and powerful model
Understanding this is important because it shows that model access isn't the differentiator between the APIs - the difference lies in the features and infrastructure around them.

# Note
model: Cloze

## Text
For authentication, Google AI Gemini API uses {{c1::API key}}, while Google Cloud Vertex AI Gemini API uses {{c2::Google Cloud service account}}

## Back Extra
The answers are:
- API key: A simpler authentication method typical for developer-focused APIs
- Google Cloud service account: A more sophisticated authentication system that provides better security and access control
This difference reflects the security needs of different user types - individual developers versus enterprise customers.

# Note
model: Cloze

## Text
The playground interfaces differ: Google AI Gemini API uses {{c1::Google AI Studio}}, while Google Cloud Vertex AI Gemini API uses {{c2::Vertex AI Studio}}

## Back Extra
The answers are:
- Google AI Studio: A streamlined interface for developers to experiment with the API
- Vertex AI Studio: A more comprehensive environment integrated with Google Cloud's broader ML tools
This distinction is important as it shows how each interface is tailored to its target user's needs.

# Note
model: Cloze

## Text
In terms of free tier, Google AI Gemini API offers {{c1::a basic free tier}}, while Google Cloud Vertex AI Gemini API provides {{c2::$300 Google Cloud credit for new users}}

## Back Extra
The answers are:
- a basic free tier: Suited for developers to test and build small applications
- $300 Google Cloud credit for new users: A more substantial offering to let businesses evaluate the full enterprise capabilities
This difference in pricing structure reflects the different scales of intended usage.

# Note
model: Cloze

## Text
Google Cloud Vertex AI Gemini API offers three key enterprise features: {{c1::customer encryption key}}, {{c2::virtual private cloud}}, and {{c3::data residency}}

## Back Extra
The answers are:
- customer encryption key: Allows enterprises to maintain control over their data encryption
- virtual private cloud: Provides isolated network infrastructure for enhanced security
- data residency: Enables compliance with data location requirements
These features are crucial for enterprise adoption as they address key security and compliance requirements that large organizations typically have.

# Note
model: Cloze

## Text
Both APIs have a quota of {{c1::60}} requests per minute, which can be {{c2::increased upon request}}

## Back Extra
The answers are:
- 60: The default rate limit that prevents abuse and ensures fair resource distribution
- increased upon request: Flexibility to scale up for legitimate use cases
This is important to understand for planning application architecture and scaling considerations.

I'll create cloze deletion flashcards that capture the key concepts about LLMs while maintaining the hierarchical structure.

# Note
model: Cloze

## Text
A Large Language Model (LLM) is a {{c1::statistical language model}} trained on {{c2::massive data}}, based on {{c3::deep learning architectures like Transformer}}

## Back Extra
The answers are:
- statistical language model: This emphasizes that LLMs are fundamentally probability-based systems, not rule-based like traditional language processing
- massive data: This is crucial because the scale of training data directly impacts the model's capabilities and knowledge breadth
- deep learning architectures like Transformer: The Transformer architecture, introduced by Google in 2017, revolutionized NLP by enabling better understanding of long-range dependencies in text

# Note
model: Cloze

## Text
LLMs are characterized by scale in two dimensions: {{c1::large number of parameters}} (e.g., GPT-3: 175B) and {{c2::large training data size}} (e.g., GPT-3: 570GB text data)

## Back Extra
The answers are:
- large number of parameters: Parameters are the adjustable weights that determine how the model processes information
- large training data size: The amount of text data the model learns from
This scale is what enables LLMs to capture complex patterns in language and knowledge.

# Note
model: Cloze

## Text
In the technical components of LLMs, {{c1::vectors}} are used because {{c2::machines can only understand numbers}}

## Back Extra
The answers are:
- vectors: Single-dimensional arrays that represent data mathematically
- machines can only understand numbers: This fundamental limitation means all text processing must be converted to numerical representations
This helps remember why we need mathematical representations of language for machine learning.

# Note
model: Cloze

## Text
In tokenization, {{c1::tokens}} are {{c2::basic units of data}} that can be a {{c3::word, part of word, or character}}

## Back Extra
The answers are:
- tokens: The fundamental units that LLMs process
- basic units of data: This emphasizes that tokens are the atomic elements of processing
- word, part of word, or character: This flexibility in token size allows for efficient representation of different languages and concepts
Understanding tokenization is crucial because it's how text is converted into a format the model can process.

# Note
model: Cloze

## Text
In Gemini/PaLM2, {{c1::~4 characters}} make up one token, and {{c2::100 tokens}} is approximately equal to {{c3::60-80 English words}}

## Back Extra
The answers are:
- ~4 characters: This helps estimate how much text can fit in a given token limit
- 100 tokens: A standard unit of measurement for LLM capacity
- 60-80 English words: This real-world equivalence helps in planning prompt lengths
These ratios are important for understanding model limitations and planning usage.

# Note
model: Cloze

## Text
Gemini has a context length of {{c1::1 million tokens}}, which is significant because it determines {{c2::how much information the model can consider at once}}

## Back Extra
The answers are:
- 1 million tokens: This large context window is a major advancement
- how much information the model can consider at once: This directly impacts the model's ability to maintain coherence and context in longer conversations
Context length is crucial for complex tasks requiring memory of earlier information.

# Note
model: Cloze

## Text
{{c1::Embeddings}} are tokens with {{c2::semantic context}} that {{c3::represent meaning}} and allow understanding of {{c4::context, nuance, and subtle meanings}}

## Back Extra
The answers are:
- Embeddings: The mathematical representations that capture meaning
- semantic context: The relationship between words and their meanings
- represent meaning: Converting linguistic concepts into mathematical space
- context, nuance, and subtle meanings: This enables the model to understand language beyond literal definitions
Embeddings are crucial because they're how LLMs understand relationships between concepts.

# Note
model: Cloze

## Text
The {{c1::max output tokens}} parameter determines the maximum length of the generated response

## Back Extra
The answer is "max output tokens" because this is the fundamental way to control response length in language models. Understanding this is crucial as it affects both resource usage and cost - longer responses consume more tokens and therefore cost more to generate.

# Note
model: Cloze

## Text
{{c1::Temperature}} is a parameter that controls {{c2::randomness}} in token selection, where {{c3::lower temperatures}} lead to more deterministic responses

## Back Extra
The answers are:
- Temperature: A fundamental parameter that affects the creativity vs. reliability trade-off
- randomness: This describes how the model chooses between different possible next tokens
- lower temperatures: Close to 0, making the model choose the most probable tokens consistently
This parameter is crucial because it lets us control whether we want precise, factual responses (low temperature) or creative, varied outputs (high temperature).

# Note
model: Cloze

## Text
A temperature of {{c1::0}} means the model will {{c2::always select the highest probability response}}

## Back Extra
The answers are:
- 0: The minimum temperature setting
- always select the highest probability response: The most deterministic behavior possible
This is important to remember because temperature=0 is often used for tasks requiring consistency and accuracy, like coding or fact-based responses.

# Note
model: Cloze

## Text
{{c1::Higher temperatures}} are better for {{c2::creative tasks}} but increase the risk of {{c3::hallucinations}}

## Back Extra
The answers are:
- Higher temperatures: Temperature settings closer to 1 or above
- creative tasks: Tasks like poetry or storytelling that benefit from variety
- hallucinations: Generated content that's incorrect or nonsensical
This relationship helps understand the trade-off between creativity and reliability in language models.

# Note
model: Cloze

## Text
The {{c1::topK}} parameter controls token selection by limiting choices to the {{c2::K most probable tokens}}

## Back Extra
The answers are:
- topK: A parameter that provides direct control over the selection pool size
- K most probable tokens: The number of top candidates the model can choose from
Understanding topK is important because it's a more direct way to control randomness compared to temperature.

# Note
model: Cloze

## Text
{{c1::topP}} sampling selects tokens from most to least probable until their probabilities sum to {{c2::0.95}} by default

## Back Extra
The answers are:
- topP: Also known as nucleus sampling, this provides more nuanced control than topK
- 0.95: The default threshold that captures most of the probability mass while excluding unlikely tokens
This parameter is crucial because it adapts to the probability distribution of tokens, unlike topK which uses a fixed number.

# Note
model: Cloze

## Text
{{c1::Stop sequences}} can be set to {{c2::custom values}} or {{c3::simple punctuation}} like "." to control generation length

## Back Extra
The answers are:
- Stop sequences: Special triggers that end text generation
- custom values: Specific phrases or patterns defined by the user
- simple punctuation: Basic markers like periods or newlines
This is important because it provides fine-grained control over response formatting and length.

# Note
model: Cloze

## Text
Hallucinations in language models are caused by four main factors: 1. {{c1::Insufficient training data}} 2. {{c2::Noisy training data}} 3. {{c3::Insufficient context}} 4. {{c4::Insufficient constraints}}

## Back Extra
The answers are:
- Insufficient training data: The model hasn't seen enough examples to learn patterns properly
- Noisy training data: The training data contained incorrect or inconsistent information
- Insufficient context: The prompt doesn't provide enough information for accurate generation
- Insufficient constraints: The model has too much freedom in generating responses
Understanding these causes is crucial for preventing hallucinations by addressing each factor through proper prompt engineering and model selection.

# Note
model: Cloze

## Text
{{c1::Prompt design}} is the process of creating prompts to get {{c2::desired responses}} from models through {{c3::natural language requests}}

## Back Extra
The answers are:
- Prompt design: A crucial skill in working with language models, as it bridges human intent and model capabilities
- desired responses: The specific outputs we want the model to generate
- natural language requests: The human-readable instructions given to the model
This fundamental concept shows how we communicate with AI models effectively.

# Note
model: Cloze

## Text
There are three main types of prompts: {{c1::zero-shot}} (no examples), {{c2::one-shot}} (single example), and {{c3::few-shot}} (multiple examples)

## Back Extra
The answers are:
- zero-shot: The simplest form where the model must understand and execute without examples
- one-shot: Provides one example to guide the model's response
- few-shot: Uses multiple examples to establish patterns
Understanding these types helps choose the right approach based on task complexity and desired accuracy.

# Note
model: Cloze

## Text
Best practices for prompt design include giving {{c1::clear instructions}}, specifying {{c2::constraints and formatting}}, and showing {{c3::patterns to follow}}

## Back Extra
The answers are:
- clear instructions: Explicit directions that leave no room for ambiguity
- constraints and formatting: Specific requirements for the output structure and limitations
- patterns to follow: Good examples that demonstrate the desired behavior
These practices are essential because they help the model understand exactly what we want.

# Note
model: Cloze

## Text
When using examples in prompts, it's important to maintain {{c1::consistent formatting}} and find the right balance as {{c2::too few examples}} are ineffective while {{c3::too many examples}} lead to overfitting

## Back Extra
The answers are:
- consistent formatting: Uniform structure across all examples to establish clear patterns
- too few examples: Not enough to establish the desired pattern
- too many examples: Can cause the model to become too rigid in its responses
This balance is crucial for optimal model performance.

# Note
model: Cloze

## Text
For text classification tasks like {{c1::fraud detection}}, {{c2::spam filtering}}, and {{c3::sentiment analysis}}, the best practice is to set {{c4::temperature to zero}} and {{c5::top-K to one}}

## Back Extra
The answers are:
- fraud detection: Identifying fraudulent activities
- spam filtering: Categorizing unwanted messages
- sentiment analysis: Determining emotional tone
- temperature to zero: Ensures maximum determinism
- top-K to one: Forces selection of most probable token
These settings are important because classification tasks need consistent, deterministic responses.

# Note
model: Cloze

## Text
Extraction tasks include {{c1::named entity recognition}}, {{c2::relation extraction}}, and {{c3::event extraction}}, and also require {{c4::temperature zero}} for accuracy

## Back Extra
The answers are:
- named entity recognition: Identifying specific entities like names, places, organizations
- relation extraction: Understanding relationships between entities
- event extraction: Identifying specific events and their components
- temperature zero: Ensures consistent, reliable extraction
These tasks require high precision, hence the strict parameter settings.

# Note
model: Cloze

## Text
For summarization tasks, best practices include specifying {{c1::desired characteristics}} and using {{c2::higher temperature}} and {{c3::higher top-K/P values}} for creative summaries

## Back Extra
The answers are:
- desired characteristics: Specific qualities wanted in the summary
- higher temperature: Allows for more varied language and expression
- higher top-K/P values: Enables more diverse word choices
These settings are important because summarization often requires a balance between accuracy and creativity.

