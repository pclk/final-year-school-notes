# Google Cloud Natural Language API Guide

## Using API Key with Natural Language API

### Initial Setup
```python
import requests 
import json

googleAPIKey = "AIzaSyBhNkvLI0d7ZcHxP95h7slBOs-5OT8nLa0"
googleurl = "https://language.googleapis.com/v1/documents:annotateText?key=" + googleAPIKey
req_headers = {'Content-Type': 'application/json'}
```

<details>
<summary>API Configuration Details</summary>

- **URL Structure**: Uses annotateText endpoint for combined analysis
- **Headers**: Content-Type ensures proper JSON parsing
- **API Key**: Enables authenticated access to Google Cloud services
</details>

### Document Preparation
```python
document = "Nanyang Polytechnic gives our students the head start they are looking for in their next phase in life with our innovative teaching methods and industry-focused projects. They'll not only be academically prepared, but also future-ready - equipped to tackle whatever life throws at them in their career or further education. Our annual Graduate Employment Surveys show that our students are consistently highly sought-after by employers in multiple industries. Many of our graduates have also gone on to local and overseas universities, where they continue to excel in their field of study."
```

### API Request Structure
```python
data = {
    "document":{
        "type": "PLAIN_TEXT",
        "content": document
    },
    "features":{
        "extractSyntax": True,
        "extractEntities": True,
        "extractDocumentSentiment": True,
        "extractEntitySentiment": True,
        "classifyText": True,
        "moderateText": True
    },
    "encodingType":"UTF8"
}
```

<details>
<summary>Request Parameters Explained</summary>

- **document.type**: Specifies text format (PLAIN_TEXT vs HTML)
- **features**: Boolean flags for different analysis types:
  - extractSyntax: Grammatical structure
  - extractEntities: Named entity recognition
  - extractDocumentSentiment: Overall emotion analysis
  - extractEntitySentiment: Per-entity sentiment
  - classifyText: Content categorization
  - moderateText: Content appropriateness check
- **encodingType**: Character encoding specification
</details>

### Making the Request
```python
r = requests.post(url=googleurl, headers=req_headers, json=data) 

if r.status_code == 200:
    result = r.json()
    print(result)
else:
    print('Error with status')
    print(r.content)
```

### Response Analysis
```python
# Pretty print JSON response
print(json.dumps(result, indent=4))

# View available keys in response
print(result.keys())
```

<details>
<summary>Response Structure</summary>

The API response contains multiple sections:
- **sentences**: Individual sentence analysis
- **tokens**: Word-level analysis
- **entities**: Named entity recognition
- **documentSentiment**: Overall text sentiment
- **categories**: Content classification
</details>

## Syntax Analysis

### Sentence Processing
```python
# Extract sentences from result
sentences = result['sentences']

# Create pandas DataFrame for analysis
import pandas as pd
pd.set_option('display.max_colwidth', 0)

df1 = pd.concat({
    'content': pd.Series([sentence['text']['content'] for sentence in sentences]),
    'magnitude': pd.Series([sentence['sentiment']['magnitude'] for sentence in sentences]),
    'score': pd.Series([sentence['sentiment']['score'] for sentence in sentences])
}, axis=1)
```

<details>
<summary>Sentence Analysis Components</summary>

Each sentence contains:
- **content**: The actual text
- **magnitude**: Intensity of sentiment (0 to ∞)
- **score**: Sentiment polarity (-1 to +1)
- **beginOffset**: Starting position in text
</details>

### Token Analysis
```python
tokens = result['tokens']

# Create DataFrame from part of speech information
df2 = pd.DataFrame([token['partOfSpeech'] for token in tokens])

# Add content and lemma information
df2.insert(0, 'content', pd.Series([token['text']['content'] for token in tokens]))
df2.insert(1, 'lemma', pd.Series([token['lemma'] for token in tokens]))

# Add dependency edge information
df2['d_edge_head_index'] = pd.Series([token['dependencyEdge']['headTokenIndex'] for token in tokens])
df2['d_edge_label'] = pd.Series([token['dependencyEdge']['label'] for token in tokens])
```

<details>
<summary>Token Properties Explained</summary>

Each token contains:
- **content**: Original word
- **lemma**: Base/dictionary form
- **partOfSpeech**: Grammatical properties including:
  - tag: POS tag (NOUN, VERB, etc.)
  - aspect: Verb aspect
  - case: Grammatical case
  - gender: Gender property
  - mood: Verb mood
  - number: Singular/plural
  - person: 1st/2nd/3rd person
  - proper: Proper vs common
  - reciprocity: Reciprocal form
  - tense: Verb tense
  - voice: Active/passive voice
- **dependencyEdge**: Syntactic relationships:
  - headTokenIndex: Parent token reference
  - label: Dependency type
</details>

### Advanced Token Analysis
```python
# Filter for specific person properties
df2.loc[df2['person'] != 'PERSON_UNKNOWN']

# Analyze number distribution
df2['number'].value_counts()

# Find plural tokens
df2.loc[df2['number'] == 'PLURAL']

# Check for mood information
df2.loc[df2['mood'] != 'MOOD_UNKNOWN']
```

<details>
<summary>Token Analysis Features</summary>

- **Person Filter**: Shows pronouns and verbs with person marking
- **Number Analysis**: Identifies singular/plural usage patterns
- **Mood Analysis**: Shows verb moods (indicative, subjunctive, etc.)
- **Custom Filters**: Can be combined for specific linguistic patterns
</details>

## Syntax Analysis Deep Dive

### Understanding Syntax Response
```python
# Response structure for syntax analysis
{
  "sentences": [
    {
      object (Sentence)
    }
  ],
  "tokens": [
    {
      object (Token)
    }
  ],
  "language": string
}
```

### Sentence-Level Analysis
```python
# Extract sentences
sentences = result['sentences']

# Examine first sentence structure
sentences[0]

# Create comprehensive DataFrame
df1 = pd.concat({
    'content': pd.Series([sentence['text']['content'] for sentence in sentences]),
    'magnitude': pd.Series([sentence['sentiment']['magnitude'] for sentence in sentences]),
    'score': pd.Series([sentence['sentiment']['score'] for sentence in sentences])
}, axis=1)
```

<details>
<summary>Sentence Analysis Components</summary>

Each sentence object contains:
- **text**: Raw sentence content
- **sentiment**: Emotional tone metrics
  - magnitude: Strength of emotion
  - score: Positive/negative rating
- **beginOffset**: Position in document
</details>

### Token-Level Analysis
```python
# Extract tokens
tokens = result['tokens']

# Examine token structure
tokens[0]
tokens[0].keys()

# Create token DataFrame with part of speech info
df2 = pd.DataFrame([token['partOfSpeech'] for token in tokens])

# Add content and lemma columns
df2.insert(0, 'content', pd.Series([token['text']['content'] for token in tokens]))
df2.insert(1, 'lemma', pd.Series([token['lemma'] for token in tokens]))
```

<details>
<summary>Token Structure Details</summary>

Each token contains:
- **text**: Original word form
- **lemma**: Dictionary/base form
- **partOfSpeech**: Grammatical properties
- **dependencyEdge**: Syntactic relationships
</details>

### Dependency Edge Analysis
```python
# Examine dependency structure
tokens[0]['dependencyEdge']

# Add dependency information to DataFrame
df2['d_edge_head_index'] = pd.Series([token['dependencyEdge']['headTokenIndex'] for token in tokens])
df2['d_edge_label'] = pd.Series([token['dependencyEdge']['label'] for token in tokens])
```

<details>
<summary>Dependency Edge Components</summary>

- **headTokenIndex**: Points to parent token
- **label**: Grammatical relationship type
- Helps build syntactic tree structure
- Essential for understanding sentence structure
</details>

### Part of Speech Analysis
```python
# Filter for person information
df2.loc[df2['person'] != 'PERSON_UNKNOWN']

# Analyze number distribution
df2['number'].value_counts()

# Find plural tokens
df2.loc[df2['number'] == 'PLURAL']

# Check mood information
df2.loc[df2['mood'] != 'MOOD_UNKNOWN']
```

<details>
<summary>Part of Speech Features</summary>

Available grammatical properties:
- **person**: FIRST, SECOND, THIRD
- **number**: SINGULAR, PLURAL
- **gender**: FEMININE, MASCULINE, NEUTER
- **mood**: INDICATIVE, IMPERATIVE, SUBJUNCTIVE
- **tense**: PAST, PRESENT, FUTURE
- **voice**: ACTIVE, PASSIVE
</details>

## Entity Analysis

### Entity Response Structure
```python
{
  "entities": [
    {
      object (Entity)
    }
  ],
  "language": string
}
```
```python
# Extract entities from response
entities = result['entities']

# Examine first entity structure
entities[0]

# Create DataFrame for analysis
df3 = pd.DataFrame(entities)
df3
```

<details>
<summary>Entity DataFrame Columns</summary>

- **name**: The actual text of the entity
- **type**: Classification (PERSON, LOCATION, ORGANIZATION, etc.)
- **salience**: Importance score from 0-1
- **mentions**: Array of text appearances
- **metadata**: Additional contextual information
- **sentiment**: Entity-specific sentiment scores
</details>

## Sentiment Analysis

### Document-Level Sentiment
```python
# Access available keys
result.keys()

# Extract document sentiment
result['documentSentiment']

# Access sentence-level sentiment details
result['sentences']
```

<details>
<summary>Understanding Sentiment Scores</summary>

Document sentiment contains:
- **score**: Emotional leaning (-1 to +1)
  - -1: Very negative
  - 0: Neutral
  - +1: Very positive
- **magnitude**: Overall emotional intensity (0 to ∞)
  - Higher values indicate stronger emotion
  - Useful for distinguishing neutral from mixed sentiment
</details>

<details>
<summary>Sentence-Level Sentiment</summary>

Each sentence contains:
- **text**: The sentence content
- **sentiment**: Local sentiment scores
  - score: Local emotional leaning
  - magnitude: Local emotional intensity
- Useful for tracking sentiment changes throughout text
</details>

## Entity Sentiment Analysis

### Response Structure
```python
{
  "entities": [
    {
      "name": string,
      "type": enum(Type),
      "metadata": {
        string: string,
        ...
      },
      "salience": number,
      "mentions": [
        {
          "text": {
            "content": string,
            "beginOffset": number
          },
          "type": enum(Type),
          "sentiment": {
            "magnitude": number,
            "score": number
          }
        }
      ],
      "sentiment": {
        "magnitude": number,
        "score": number
      }
    }
  ],
  "language": string
}
```

<details>
<summary>Entity Sentiment Components</summary>

Each entity contains:
- **name**: Entity text
- **type**: Entity classification
  - PERSON
  - LOCATION
  - ORGANIZATION
  - EVENT
  - WORK_OF_ART
  - CONSUMER_GOOD
  - OTHER
- **metadata**: Additional entity information
  - wikipedia_url
  - mid (Machine Intelligence ID)
  - knowledge graph links
- **salience**: Importance in text (0-1)
- **mentions**: List of appearances
  - text content
  - begin offset
  - mention type
  - local sentiment
</details>

## Content Classification

### Accessing Classifications
```python
# View all categories
result['categories']
```

### Example Classification Output
```python
[
    {
        "name": "/Education/Educational stages/Higher education",
        "confidence": 0.85
    },
    {
        "name": "/Business/Employment",
        "confidence": 0.75
    }
]
```

<details>
<summary>Classification Categories Explained</summary>

Category structure:
- **name**: Hierarchical path showing category relationships
  - Forward slashes separate levels
  - More specific categories appear deeper in path
  - Example: /Technology/Computing/Programming
- **confidence**: Certainty score (0-1)
  - Higher values indicate stronger matches
  - Typically threshold at 0.5 for reliability
</details>

<details>
<summary>Common Category Types</summary>

Major category hierarchies include:
- **/Arts & Entertainment**
  - Movies, Music, Television
- **/Business**
  - Finance, Employment, Industries
- **/Computers & Electronics**
  - Programming, Software, Hardware
- **/Health**
  - Medicine, Conditions, Nutrition
- **/Science**
  - Biology, Physics, Mathematics
- **/Education**
  - Academic subjects, Educational stages
</details>

<details>
<summary>Classification Usage Tips</summary>

- Multiple categories can apply to single text
- Confidence scores help prioritize categories
- Hierarchical structure enables both broad and specific classification
- Categories are language-independent
- Minimum content length required (20+ words recommended)
</details>

### Working with Classifications
```python
# Extract category names and confidence scores
categories = result['categories']
for category in categories:
    print(f"Category: {category['name']}")
    print(f"Confidence: {category['confidence']:.2f}")
    print()
```

<details>
<summary>Response Processing</summary>

Key aspects to check:
- Presence of categories array
- Valid confidence scores (0-1)
- Proper category path formatting
- Multiple classification levels
- Category relevance to content
</details>

# Service Account Authentication and Sentiment Analysis

## Service Account Setup

### Environment Configuration
```python
# Verify service key presence
!dir *.json

# Set service account credentials
%env GOOGLE_APPLICATION_CREDENTIALS=it3386-2024-s2.json

# Verify credentials path
%env GOOGLE_APPLICATION_CREDENTIALS
```

<details>
<summary>Service Account vs Developer Account</summary>

Key differences:
- Service accounts have specific permissions
- Not tied to individual user credentials
- Enable programmatic API access
- Required for production environments
- Can be restricted to specific services
</details>

## Client Library Implementation

### Library Setup
```python
from google.cloud import language_v1
```

### Sentiment Analysis Function
```python
def analyse_sentiment(text_content):
    client = language_v1.LanguageServiceClient()

    # Available types: PLAIN_TEXT, HTML
    document_type_in_plain_text = language_v1.Document.Type.PLAIN_TEXT
    language_code = "en"
    
    document = {
        "content": text_content,
        "type_": document_type_in_plain_text
    }

    # Set encoding type
    encoding_type = language_v1.EncodingType.UTF8

    # Make API request
    response = client.analyze_sentiment(
        request={"document": document, "encoding_type": encoding_type}
    )
    
    return response
```

<details>
<summary>Function Parameters Explained</summary>

- **document_type**: 
  - PLAIN_TEXT: Raw text input
  - HTML: Structured web content
- **language_code**: 
  - "en" for English
  - Optional, auto-detected if not specified
- **encoding_type**:
  - UTF8: Standard Unicode encoding
  - UTF16/UTF32: Alternative encodings
</details>

### Sample Analysis
```python
document = "Nanyang Polytechnic gives our students the head start..."

# Perform analysis
annotations = analyse_sentiment(document)

# Extract overall sentiment
score = annotations.document_sentiment.score
magnitude = annotations.document_sentiment.magnitude

# Analyze individual sentences
for index, sentence in enumerate(annotations.sentences):
    sentence_sentiment = sentence.sentiment.score
    print(
        "Sentence {} has a sentiment score of {}".format(index, sentence_sentiment)
    )

print(
    "Overall Sentiment: score of {} with magnitude of {}".format(score, magnitude)
)
```

<details>
<summary>Response Structure</summary>

annotations object contains:
- **document_sentiment**: Overall text sentiment
  - score: Emotional leaning (-1 to +1)
  - magnitude: Emotional intensity (0+)
- **sentences**: Array of sentence-level analysis
  - text: Sentence content
  - sentiment: Local sentiment scores
  - beginOffset: Position in document
</details>

<details>
<summary>Interpreting Results</summary>

Sentiment combinations:
- **Positive score, high magnitude**: Strong positive emotion
- **Negative score, high magnitude**: Strong negative emotion
- **Near-zero score, low magnitude**: Neutral content
- **Near-zero score, high magnitude**: Mixed emotions
</details>

<details>
<summary>Common Use Cases</summary>

- Customer feedback analysis
- Social media monitoring
- Product review processing
- Brand sentiment tracking
- Content moderation
</details>
