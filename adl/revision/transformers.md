# deck: adl transformers

test

How can we speed up RNNs?
Throw away recurrence and rely on attention
---
Transformers came along and said, "Hey, this sequential processing of RNNs is really slowing us down. But this attention thing is brilliant! What if we could get rid of the sequential part and *only* rely on attention?"
This is the core idea of Transformers. They ditch the sequential, step-by-step processing of RNNs ("throws away recurrence"). Instead, they are built entirely on the attention mechanism ("relying entirely on an attention mechanism").
This allows them to look at the *entire* input sequence at once and figure out the relationships between all the words simultaneously.
It's like being able to see the whole page of the book at once and understand how all the words relate to each other, instead of reading it line by line.
**Transformers are like a team meeting around a table:** Everyone has access to all the information at once (the entire input sequence). They can all talk to each other and figure out how different parts of the project relate to each other simultaneously using "attention" – focusing on the most important aspects and contributions from different team members. This is parallel and much faster!

What is the task of Language modelling?
predicting the next word.
---
Language Modeling is about calculating the probability of each possible word being the next word, based on the sequence of words that have come before it.
The "probability distribution" part means that the Language Model doesn't just pick *one* next word. Instead, it assigns a probability to *every* word in its vocabulary.
The words with higher probabilities are the ones the Language Model thinks are more likely to come next.  When your phone suggests words, it's usually showing you the words with the highest probabilities according to its Language Model.

Why are transformers good in language modelling?
Can better understand the current word.
---
Transformers are often used as "language models".  This means that one of their primary tasks is Language Modeling.
Transformers are very good at understanding context and relationships between words in a sequence, thanks to their attention mechanism. 
This makes them excellent at predicting the next word, and therefore, very powerful Language Models.

What component in transformers allows it to excel in language modelling?
Self Attention
---
When the model processes each word in the input sequence, self attention mechanism allows it to pay attention (give more weightage) to other word positions that can help to better encode (understand) the current word.

What are the vectors in Transformers?
Q, K, V
---
Think of the "Encoder" as you, the student, trying to understand a single question from the teacher.  Let's say the question is: "What is the capital of France?".
*   **The Question Itself (Q - Query):**  The question "What is the capital of France?" is your **Query (Q)**. It's what you are currently focusing on and trying to answer. In the Transformer encoder, when processing a word, that word becomes the Query. Let's say we are processing the word "France" in a sentence. "France" is our Query (Q).
*   **Your Knowledge Base (K - Key, V - Value):** To answer the question, you need to look through your knowledge.  Think of your brain as having lots of pieces of information stored.  Each piece of information has two parts:
    *   **Key (K):**  A label or keyword that helps you find the information. For example, "country names", "cities", "capitals".
    *   **Value (V):** The actual information itself. For example, "France", "Paris", "London".
    In the Transformer encoder, for each word in the input sentence, we consider it as both a **Key (K)** and a **Value (V)**.  So, if our input sentence is "France is a country", then "France", "is", "a", "country" are all Keys and Values.

What are the Q, K, and V vectors inside an Encoder (aka self-attention)?
All of them are the word embeddings in the sentence.
---
This is because in this stage, the Transformer is calculating an attention score for each word. And to do this, it has to compare each word of the sentence to each other.
Once the attention score is calculated, the model is enriched with information about its relationships with other words in the sentence.

What are Q values in Encoder-Decoder (specifically the cross-attention)?
Q = of contextualized representations output sentence so far. 
---
Encoder reads the input sentence and uses self-attention to generate a contextual understanding. It outputs a set of K and V, alike to a summary of the input sentence that is useful for generating the output sentence.
Decoder takes the K V vectors as well as the output it has already generated (Q), to generate an contextualized output vector, which is turned into a matrix of equal dimensions to the vocabulary list, so that we can implicitly map this matrix to a vocabulary list to generate a probability distribution of tokens.
Cross-Attention refers to the Decoder referencing from output (Q), with the K V vectors to understand which inputs it should attend to generate the next token.

What are K V values in Encoder-Decoder (specifically the cross-attention)?
K = V = contextualized representations of input sentence
---
Encoder reads the input sentence and uses self-attention to generate a contextual understanding. It outputs a set of K and V, alike to a summary of the input sentence that is useful for generating the output sentence.
Decoder takes the K V vectors as well as the output it has already generated (Q), to generate an contextualized output vector, which is turned into a matrix of equal dimensions to the vocabulary list, so that we can implicitly map this matrix to a vocabulary list to generate a probability distribution of tokens.
Cross-Attention refers to the Decoder referencing from output (Q), with the K V vectors to understand which inputs it should attend to generate the next token.

What are the 3 Transformer Variants?
Autoregressive, Autoencoder, Seq2Seq
---
Memory: ARAES (ah-rays)
Autoregressive: Decoder-only. Only can remember what they've said so far. Examples: GPT2, GPT3, Transformer-XL
Autoencoder: Primarily encoder. Can look at words before and after the sentence to understand its meaning. Examples: BERT, RoBERTa, DistilBERT.
Seq2Seq: Encoder-Decoder. transforms one sequence to another. Examples: BART and T5

What does it mean when BERT uses "Masked language modelling"?
15% of tokens replaced with `[MASK]` token
---
Imagine the detective is given a document with some words blanked out, like a "fill in the blanks" exercise, but for super-detectives!  For example:
"The capital of \[MASK] is Paris."
The detective's job is to figure out the missing word, which is "France".  To do this, BERT looks at the words *around* the blanked-out word ("capital of", "is Paris").  Because BERT is **bidirectional** (we'll get to that soon!), it looks at words both *before* and *after* the masked word to understand the context.
This MLM task forces BERT to deeply understand the context of words and how they fit together in a sentence.  By practicing this on millions of sentences with masked words, BERT becomes incredibly good at predicting words based on their surrounding context.

What does it mean when BERT can do "Next sentence prediction"?
2 sentences and predict whether they're contiguous.
---
Now imagine the detective is given pairs of sentences and asked: "Do these two sentences logically follow each other in a real document?"  For example:
**Pair 1 (Contiguous):**
Sentence A: "The dog barked loudly."
Sentence B: "Its owner came running."
**Pair 2 (Not Contiguous):**
Sentence A: "The sky is blue."
Sentence B: "The price of apples is rising."
The detective needs to predict if Sentence B is likely to come *right after* Sentence A in a real text.  In Pair 1, they are related, so the answer is "Yes, contiguous". In Pair 2, they are unrelated, so the answer is "No, not contiguous".
This NSP task trains BERT to understand relationships *between* sentences and to grasp the flow of text and logical connections between ideas.

What are the 3 Key features of BERT?
Bidirectional, Contextual word embeddings, Fine-tune
---
Memory: BCF
*   **Bidirectional:**  "Looking in both directions"
Unlike older language models that read text only from left to right (like reading a book), BERT is **bidirectional**.  It can look at the entire sentence at once, considering words both before and after a given word to understand its meaning.
Think of it like this: If you're trying to understand the word "bank", and you only read from left to right, you might see "I went to the bank..." and you might not know if it's a river bank or a money bank until you read further.  But if you can look in both directions, and you see "...to deposit money at the bank", the context becomes much clearer immediately.  BERT's bidirectionality gives it a much richer understanding of context.
*   **Contextual word embeddings:** "Words change meaning based on context"
"Word embeddings" are like numerical representations of words that capture their meaning.  Traditional word embeddings (like Word2Vec or GloVe) give each word a *single*, fixed embedding.  So, the word "bank" would always have the same embedding, regardless of whether it's a river bank or a financial bank.
However, BERT creates **contextual word embeddings**. This means that the embedding for a word *changes* depending on the sentence it's in.  So, "bank" in "river bank" will have a *different* embedding than "bank" in "financial bank".  This is a huge advantage because it allows BERT to truly understand the *meaning* of words in context, just like humans do!
*   **Can be fine-tuned:** "Specializing for specific tasks"
After the detective finishes their general training at the academy (pre-training), they can then specialize in specific types of cases, like "missing person cases" or "financial fraud cases".  This "specialization" is like **fine-tuning** BERT.
Fine-tuning means taking the pre-trained BERT model and training it further on a *specific* task with a *specific* dataset.  For example:
*   **Sentiment analysis:** Fine-tuning BERT to classify movie reviews as positive or negative.
*   **Question answering:** Fine-tuning BERT to answer questions based on a given text passage.
*   **Text classification:** Fine-tuning BERT to categorize news articles into topics.
This fine-tuning process allows us to adapt the general language understanding of BERT to perform very specific NLP tasks effectively.  The "Classification Tokens" mentioned in the notes (like `[CLS]`) are used during fine-tuning for classification tasks.

What are the two inputs that BERT can accept?
1 or 2 sentences.
---
**Example Scenarios for Single Sentence Input:**
*   **Sentiment Analysis:** You want to know if a movie review is positive or negative. You give BERT the single sentence review: "This movie was absolutely amazing!" and BERT can tell you it's positive.
*   **Text Classification:** You want to categorize a news headline. You give BERT the headline: "Stock Market Reaches Record High" and BERT can classify it as being about "Business" or "Finance".
Sometimes, you want to send a card that has two parts, like a greeting card where you have a question on the front and an answer inside, or two related sentences that go together.  For example:
> **Sentence 1:** "The cat sat on the mat."
> **Sentence 2:** "It was a very comfortable mat."
This is like giving BERT *two* sentences as input.  When you give BERT two sentences, it needs to understand that they are separate but potentially related. To help BERT understand this "two-part card", we use special "separators" in the input.

What are the two special tokens from BERT?
`[CLS]` and `[SEP]`
---
*   **`[CLS]` (Classification Token):**  Think of this as the "Start of Card" marker.  It's always placed at the very beginning of the *entire* input (even if it's just one sentence, but it's crucial for two-sentence inputs). BERT uses the understanding it gathers at this `[CLS]` token especially for tasks like classification.
*   **`[SEP]` (Separator Token):** This is the "Sentence Divider".  When you have two sentences, you put `[SEP]` between them to clearly tell BERT, "Okay, sentence one ends here, and sentence two begins!"
**So, if you give BERT the two sentences above, the actual input would look like this:**
`[CLS] The cat sat on the mat. [SEP] It was a very comfortable mat. [SEP]`
Notice there's also a `[SEP]` at the very end.  While sometimes optional in simpler examples, it's good practice to include it to clearly mark the end of the input sequence as well.
**Think of it this way:** You're giving BERT a card with two related ideas. The `[CLS]` is like the cover of the card, and the `[SEP]` tokens are like dividers that separate the different parts of your message on the card.

Why is the first token of BERT called Classification?
Represents a summary of entire input.
---
Why is it called "Classification"? Because this `[CLS]` token is particularly useful when you want to use BERT for classification tasks, like figuring out if a sentence is positive or negative, or what topic a document is about.  You can think of the `[CLS]` token as representing the overall meaning of the input, which is helpful for classifying it.

Why doesn't DistilBERT use Segment embedding?
Simpler and faster
---
DistilBERT is a smaller, faster version of BERT. To make it simpler and faster, DistilBERT *doesn't* use segment embeddings.  It still works well for many tasks, even without this feature.  So, segment embeddings are a detail that's in original BERT but not always necessary.

Why are WordPiece embeddings used in BERT?
Reduce vocab size
---
**"WordPiece embeddings is used to encode input sentence e.g. "I have a new GPU!“ ['i', 'have', 'a', 'new', 'gp', '##u', '!’]"**: This is a crucial point! BERT uses something called "WordPiece embeddings".  
Imagine you have LEGO bricks. WordPieces are like special LEGO bricks that can be whole words like "have", "a", "new", or parts of words like "gp" and "##u" from "GPU".
Why break words into pieces?  Because in language, there are so many words! It's impossible for BERT to know *every single word*.  
WordPieces help BERT handle rare words and words it hasn't seen before.  
By breaking words into smaller, more frequent pieces, BERT can understand a wider range of text.

What does it mean when BERT uses "Segment Embedding"?
Recognize whether token belongs to which sentence.
---
**"Segment embedding to indicate if a token belongs to segment A or segment B."**:  When you give BERT two sentences, it needs to know which tokens belong to the first sentence and which belong to the second sentence.  "Segment embeddings" are like color-coding.  BERT assigns a "segment A" color to all tokens in the first sentence and a "segment B" color to all tokens in the second sentence.

How does one fine-tune BERT?
Feed input and output and finetune all parameters
---
In the context of BERT, "fine-tuning" means you take the pre-trained BERT model and train it *further* on a *specific* NLP task, like sentiment analysis (classifying text as positive or negative) or question answering. You feed BERT task-specific data (like movie reviews for sentiment analysis) and adjust *all* of BERT's internal settings (its "parameters") to become really good at that particular task. It's like making BERT a specialist for that one job.

How does one use BERT as an Feature Extractor?
Use it to generate contextualized word embeddings.
---
**"Use the generated embeddings as input features to train our task-specific model (e.g. text classification tasks, NER)"**.  
This is like the culinary consultant giving their flavor profiles to a restaurant chef. The restaurant chef then uses these flavor profiles as *input* to create their *own* dishes.  
The restaurant chef might be a simpler cook who is good at following instructions but doesn't have the deep culinary school training. "Our task-specific model" is like this simpler restaurant chef.  
We use the "contextualized word-embeddings" from BERT as *features* (ingredients) to train a simpler, task-specific model (like a simpler recipe) to perform the NLP task.  For example, we might use BERT's embeddings as input to a simpler logistic regression model to do sentiment analysis.

What is one key advantage of using BERT with the Feature Extractor approach?
Save compute.
---
*   **"Major computational benefits of pre-computing the expensive representation once and runs many cheaper experiments with the pre-computed features."** 
This is a key advantage of the feature extractor approach.  
Creating the "flavor profiles" (contextualized word embeddings) using BERT is computationally expensive – it's like the consultant doing a lot of in-depth analysis. But once you have these profiles, you can use them for *many* different restaurants and dishes without having to re-do the expensive analysis each time.  
"Pre-computing the expensive representation once" means we run BERT once to get the embeddings. 
"Runs many cheaper experiments with the pre-computed features" means we can then train many simpler, faster models using these embeddings as input, without needing to run BERT again and again. 
This saves a lot of time and computing power.

What choices do you have when deciding on using the embeddings of BERT?
Combination of layers.
---
**"Many choices of which embedding to use? 1st layer? 2nd layer? last layer? or combination of layers?"**  
This is like the culinary consultant offering different types of flavor profiles – maybe a profile based on the main ingredients, another based on the cooking techniques, and yet another based on the overall taste.  
With BERT, you can choose to use embeddings from different layers of the model, or even combine embeddings from multiple layers.  
Different layers of BERT capture different kinds of information, so you have flexibility in choosing which embeddings are most useful for your specific task.

Why use DistilBERT over BERT?
Retain 97% of capabilities while being 60% faster and 40% smaller.
---
| Model | Layers | Attention Heads | Parameters |
|---------|---------|-----------------|------------|
| BERT (base) | 12 | 12 | 110 million |
| DistilBERT | 6 | 12 | 66 million |

Multi-task learning means that instead of training a large language model, and using its word embeddings for task-specific models to accomplish tasks like sentiment classification, question answering, machine translation, we can just train a gigantic language model that can accomplish all these tasks.

What is a less complex approach to multi-task learning that wasn't available before LLMs?
Using LLMs for multiple predictive tasks. 
---
In the past, we had to use the embeddings from a language model like BERT, then use a fine tuned smaller task-specific model to predict the output.
Sort of like training good readers and writers to be specialists in various language problems.
Now, we can train a large language model on a gigantic text corpus, which can perform all the tasks like sentiment classification, question answering and machine translation.
Sort of like training excellent readers and writers that can apply their knowledge to solve various language problems.


