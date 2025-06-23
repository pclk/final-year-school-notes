r"""°°°
![nyp.jpg](attachment:nyp.jpg)
°°°"""

# |%%--%%| <6ED61rDnHg|ysK1w1CguA>
r"""°°°
# Google Cloud Natural Language API

In this practical, we are going to learn more about the [Google Cloud Natural Language API](https://cloud.google.com/natural-language/docs).

Cloud Natural Language allows us to perform the following operations:
- [Analyse Syntax](https://cloud.google.com/natural-language/docs/reference/rest/v1/documents/analyzeSyntax)
- [Analyse Entities](https://cloud.google.com/natural-language/docs/reference/rest/v1/documents/analyzeEntities)
- [Analyse Sentiment](https://cloud.google.com/natural-language/docs/reference/rest/v1/documents/analyzeSentiment)
- [Analyse Entity Sentiment](https://cloud.google.com/natural-language/docs/reference/rest/v1/documents/analyzeEntitySentiment)
- [Classify Content](https://cloud.google.com/natural-language/docs/reference/rest/v1/documents/classifyText)

Let's start with exploring [Cloud Natural Language Demo](https://cloud.google.com/natural-language/#natural-language-api-demo).
°°°"""
# |%%--%%| <ysK1w1CguA|qOM2fHJ4PX>
r"""°°°
### Todo

> Try out the demo using your own text. Explore the Entities, Sentiment, Syntax and Categories tabs.
°°°"""
# |%%--%%| <qOM2fHJ4PX|xOGZjr2NiS>
r"""°°°
## Using API Key


We will now connect to the Cloud Natural Language API to perform multiple operations in a single request using the [annotateText](https://cloud.google.com/natural-language/docs/reference/rest/v1/documents/annotateText) method. Click on the link to find out the request format you need to fill in and what response you expect to get.
°°°"""
# |%%--%%| <xOGZjr2NiS|Ykg3VlGQwf>

import requests
import json

# |%%--%%| <Ykg3VlGQwf|74d7rQzaDN>
r"""°°°
These parameters are required to complete the request.
°°°"""
# |%%--%%| <74d7rQzaDN|xEnCYPegJW>

googleAPIKey = "AIzaSyBhNkvLI0d7ZcHxP95h7slBOs-5OT8nLa0"
googleurl = (
    "https://language.googleapis.com/v1/documents:annotateText?key=" + googleAPIKey
)
req_headers = {"Content-Type": "application/json"}

# |%%--%%| <xEnCYPegJW|DlzOkbioNw>

document = "Nanyang Polytechnic gives our students the head start they are looking for in their next phase in life with our innovative teaching methods and industry-focused projects. They'll not only be academically prepared, but also future-ready - equipped to tackle whatever life throws at them in their career or further education. Our annual Graduate Employment Surveys show that our students are consistently highly sought-after by employers in multiple industries. Many of our graduates have also gone on to local and overseas universities, where they continue to excel in their field of study."

# |%%--%%| <DlzOkbioNw|a0baWp8PDo>
r"""°°°
Make an API request. Ensure the request parameters are filled in correctly as required under [annotateText](https://cloud.google.com/natural-language/docs/reference/rest/v1/documents/annotateText#Features).
°°°"""
# |%%--%%| <a0baWp8PDo|zC0j0HHUdj>

data = {
    "document": {"type": "PLAIN_TEXT", "content": document},
    "features": {
        "extractSyntax": True,
        "extractEntities": True,
        "extractDocumentSentiment": True,
        "extractEntitySentiment": True,
        "classifyText": True,
        "moderateText": True,
    },
    "encodingType": "UTF8",
}

r = requests.post(url=googleurl, headers=req_headers, json=data)

# Check and display the results
if r.status_code == 200:
    result = r.json()

    print(result)

    # loop through the response to get the parameters needed


else:
    print("Error with status")
    print(r.content)


# |%%--%%| <zC0j0HHUdj|ULxGe3SZ8c>

# Pretty print JSON response
print(json.dumps(result, indent=4))

# |%%--%%| <ULxGe3SZ8c|oiPg5kYZGM>

print(result.keys())

# |%%--%%| <oiPg5kYZGM|mYFXaVAL2l>
r"""°°°
### Analyse Syntax


°°°"""
# |%%--%%| <mYFXaVAL2l|EVjbf0nSB9>
r"""°°°
Refer to [Analyse Syntax](https://cloud.google.com/natural-language/docs/reference/rest/v1/documents/analyzeSyntax). The expected response is shown below.
```
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
°°°"""
# |%%--%%| <EVjbf0nSB9|JjhmXVbcJM>
r"""°°°
**Sentences**
°°°"""
# |%%--%%| <JjhmXVbcJM|WnX8STaZnX>

# keys are sentences, tokens and language
sentences = result["sentences"]

# |%%--%%| <WnX8STaZnX|vUCdEpHFiW>

sentences

# |%%--%%| <vUCdEpHFiW|UolhpIiFtt>

# explore first sentence
sentences[0]

# |%%--%%| <UolhpIiFtt|INn16sn2d5>

import pandas as pd

# |%%--%%| <INn16sn2d5|9tw3NyWd3p>

pd.set_option("display.max_colwidth", 0)

# |%%--%%| <9tw3NyWd3p|pYeOHIXXmi>

df1 = pd.concat(
    {
        "content": pd.Series([sentence["text"]["content"] for sentence in sentences]),
        "magnitude": pd.Series(
            [sentence["sentiment"]["magnitude"] for sentence in sentences]
        ),
        "score": pd.Series([sentence["sentiment"]["score"] for sentence in sentences]),
    },
    axis=1,
)

# |%%--%%| <pYeOHIXXmi|Pb85ctCtFU>

df1

# |%%--%%| <Pb85ctCtFU|HTY5aDYxE9>
r"""°°°
**Tokens**
°°°"""
# |%%--%%| <HTY5aDYxE9|IlohV5L78j>

tokens = result["tokens"]

# |%%--%%| <IlohV5L78j|IixdyuPCbS>

# explore first token
tokens[0]

# |%%--%%| <IixdyuPCbS|sG9gKRieCW>

tokens[0].keys()

# |%%--%%| <sG9gKRieCW|cJWlflcV8A>

# create dataframe using partOfSpeech inside each token
df2 = pd.DataFrame([token["partOfSpeech"] for token in tokens])

# |%%--%%| <cJWlflcV8A|N5iqY6VHul>

df2

# |%%--%%| <N5iqY6VHul|reQSZ1rIWl>

# create a pandas series using content inside each token
# then insert into first column of df2
df2.insert(0, "content", pd.Series([token["text"]["content"] for token in tokens]))
df2

# |%%--%%| <reQSZ1rIWl|TCQ5H8pbtv>

# do the same for lemma
df2.insert(1, "lemma", pd.Series([token["lemma"] for token in tokens]))
df2

# |%%--%%| <TCQ5H8pbtv|qU4Du3xfBJ>

tokens[0]["dependencyEdge"]

# |%%--%%| <qU4Du3xfBJ|mEU1CtJ1ez>

# add columns for 'dependencyEdge': {'headTokenIndex': 1, 'label': 'NN'},
df2["d_edge_head_index"] = pd.Series(
    [token["dependencyEdge"]["headTokenIndex"] for token in tokens]
)
df2["d_edge_label"] = pd.Series([token["dependencyEdge"]["label"] for token in tokens])
df2

# |%%--%%| <mEU1CtJ1ez|VLAh6Dj1Ns>
r"""°°°
&#128161; **Tip:**

> You can consolidate the commands in the previous cells and create a function `get_tokens_dataframe(tokens)` to simplify the process
°°°"""
# |%%--%%| <VLAh6Dj1Ns|ENvaJWuNIt>
r"""°°°
Explore the different fields inside [Part of Speech](https://cloud.google.com/natural-language/docs/reference/rest/v1/Token#partofspeech).

For example, [Person](https://cloud.google.com/natural-language/docs/reference/rest/v1/Token#Person) indicates different perspectives (i.e. FIRST, SECOND, etc.).
°°°"""
# |%%--%%| <ENvaJWuNIt|w8oTQSm9Yy>

df2.loc[df2["person"] != "PERSON_UNKNOWN"]

# |%%--%%| <w8oTQSm9Yy|BzF8LRvejY>
r"""°°°
What about [Number](https://cloud.google.com/natural-language/docs/reference/rest/v1/Token#number)? What does it represent?
°°°"""
# |%%--%%| <BzF8LRvejY|ddF9Q6PAwD>

df2["number"].value_counts()

# |%%--%%| <ddF9Q6PAwD|pNjbAK6yjw>

# which tokens are plural?
df2.loc[df2["number"] == "PLURAL"]

# |%%--%%| <pNjbAK6yjw|x0NhqtJGFZ>

# any token with known mood?
df2.loc[df2["mood"] != "MOOD_UNKNOWN"]

# |%%--%%| <x0NhqtJGFZ|7BsvpbzFfB>
r"""°°°
Todo

> Try processing different documents and see how the part of speech (e.g. gender, number, person, etc.) changes. 
°°°"""
# |%%--%%| <7BsvpbzFfB|IW96FbLw6D>


# |%%--%%| <IW96FbLw6D|duKuIHHTnH>
r"""°°°
### Analyse Entities
°°°"""
# |%%--%%| <duKuIHHTnH|zCmGILgOUZ>
r"""°°°
Refer to [Analyse Entities](https://cloud.google.com/natural-language/docs/reference/rest/v1/documents/analyzeEntities). The expected response is shown below.
```
{
  "entities": [
    {
      object (Entity)
    }
  ],
  "language": string
}

```
°°°"""
# |%%--%%| <zCmGILgOUZ|5GXSdPj2vn>

entities = result["entities"]
entities[0]

# |%%--%%| <5GXSdPj2vn|nEXyZSOpZ8>

df3 = pd.DataFrame(entities)
df3

# |%%--%%| <nEXyZSOpZ8|jBuugdo0TQ>
r"""°°°
### Analyse Sentiment
°°°"""
# |%%--%%| <jBuugdo0TQ|iFRlUWYu3S>
r"""°°°
Refer to [Analyse Sentiment](https://cloud.google.com/natural-language/docs/reference/rest/v1/documents/analyzeSentiment). The expected response is shown below.
```
{
  "documentSentiment": {
    object (Sentiment)
  },
  "language": string,
  "sentences": [
    {
      object (Sentence)
    }
  ]
}
```
°°°"""
# |%%--%%| <iFRlUWYu3S|VcuxNsGkA9>

result.keys()

# |%%--%%| <VcuxNsGkA9|wII7x4eE7a>

result["documentSentiment"]

# |%%--%%| <wII7x4eE7a|ZZTL0qy8jl>

result["sentences"]

# |%%--%%| <ZZTL0qy8jl|L8yGCNLwdy>


# |%%--%%| <L8yGCNLwdy|5o4yliRoyh>


# |%%--%%| <5o4yliRoyh|LXpJPxaO1L>
r"""°°°
### Analyse Entity Sentiment
°°°"""
# |%%--%%| <LXpJPxaO1L|txsuRyJR67>
r"""°°°
Refer to [Analyse Entity Sentiment](https://cloud.google.com/natural-language/docs/reference/rest/v1/documents/analyzeEntitySentiment). The expected response is shown below.
```
{
  "entities": [
    {
      object (Entity)
    }
  ],
  "language": string
}
```
°°°"""
# |%%--%%| <txsuRyJR67|7GQGzJGDKk>
r"""°°°
Refer to Analyse Entity section on how to process entities.
°°°"""
# |%%--%%| <7GQGzJGDKk|hseT3tQVLE>
r"""°°°
### Classify Content
°°°"""
# |%%--%%| <hseT3tQVLE|LVnnUwScpx>
r"""°°°
Refer to [Classify Content](https://cloud.google.com/natural-language/docs/reference/rest/v1/documents/classifyText). The expected response is shown below.

```
{
  "categories": [
    {
      object (ClassificationCategory)
    }
  ]
}
```
°°°"""
# |%%--%%| <LVnnUwScpx|OLv6HvECnA>

result["categories"]

# |%%--%%| <OLv6HvECnA|9aeVvz9Tuz>
