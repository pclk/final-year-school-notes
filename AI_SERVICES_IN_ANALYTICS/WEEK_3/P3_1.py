r"""°°°
![nyp.jpg](attachment:nyp.jpg)
°°°"""

# |%%--%%| <gUifg8g84N|uWnOdl06ds>
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
# |%%--%%| <uWnOdl06ds|qgVB4TDCh8>
r"""°°°
### Todo

> Try out the demo using your own text. Explore the Entities, Sentiment, Syntax and Categories tabs.
°°°"""
# |%%--%%| <qgVB4TDCh8|atulrKliPU>
r"""°°°
## Using API Key


We will now connect to the Cloud Natural Language API to perform multiple operations in a single request using the [annotateText](https://cloud.google.com/natural-language/docs/reference/rest/v1/documents/annotateText) method. Click on the link to find out the request format you need to fill in and what response you expect to get.
°°°"""
# |%%--%%| <atulrKliPU|Bdvmr0OMn2>

import requests
import json

# |%%--%%| <Bdvmr0OMn2|YF6Akpkqq6>
r"""°°°
These parameters are required to complete the request.
°°°"""
# |%%--%%| <YF6Akpkqq6|tB86lESpoZ>

googleAPIKey = "AIzaSyBhNkvLI0d7ZcHxP95h7slBOs-5OT8nLa0"
googleurl = (
    "https://language.googleapis.com/v1/documents:annotateText?key=" + googleAPIKey
)
req_headers = {"Content-Type": "application/json"}

# |%%--%%| <tB86lESpoZ|K7qjTD0V59>

document = "Nanyang Polytechnic gives our students the head start they are looking for in their next phase in life with our innovative teaching methods and industry-focused projects. They'll not only be academically prepared, but also future-ready - equipped to tackle whatever life throws at them in their career or further education. Our annual Graduate Employment Surveys show that our students are consistently highly sought-after by employers in multiple industries. Many of our graduates have also gone on to local and overseas universities, where they continue to excel in their field of study."

# |%%--%%| <K7qjTD0V59|p6aUxZfI7r>
r"""°°°
Make an API request. Ensure the request parameters are filled in correctly as required under [annotateText](https://cloud.google.com/natural-language/docs/reference/rest/v1/documents/annotateText#Features).
°°°"""
# |%%--%%| <p6aUxZfI7r|hVPpTefLQT>

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


# |%%--%%| <hVPpTefLQT|HPtKRwrxcR>

# Pretty print JSON response
print(json.dumps(result, indent=4))

# |%%--%%| <HPtKRwrxcR|YX9RrPO9Rt>

print(result.keys())

# |%%--%%| <YX9RrPO9Rt|IMqBtWw9dD>
r"""°°°
### Analyse Syntax


°°°"""
# |%%--%%| <IMqBtWw9dD|Hwsk0MwAct>
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
# |%%--%%| <Hwsk0MwAct|d9Kb1JQZY3>
r"""°°°
**Sentences**
°°°"""
# |%%--%%| <d9Kb1JQZY3|Ty8wwBlEKo>

# keys are sentences, tokens and language
sentences = result["sentences"]

# |%%--%%| <Ty8wwBlEKo|oWsBTnDtP6>

sentences

# |%%--%%| <oWsBTnDtP6|UUhRP25Q1G>

# explore first sentence
sentences[0]

# |%%--%%| <UUhRP25Q1G|uQp5fsuhlG>

import pandas as pd

# |%%--%%| <uQp5fsuhlG|xQRaUV3wkD>

pd.set_option("display.max_colwidth", 0)

# |%%--%%| <xQRaUV3wkD|AklmuMtRJ6>

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

# |%%--%%| <AklmuMtRJ6|lCOKwoZzG9>

df1

# |%%--%%| <lCOKwoZzG9|ooFmpxx0mu>
r"""°°°
**Tokens**
°°°"""
# |%%--%%| <ooFmpxx0mu|SK5pjqgUZk>

tokens = result["tokens"]

# |%%--%%| <SK5pjqgUZk|nRVKF087gS>

# explore first token
tokens[0]

# |%%--%%| <nRVKF087gS|1OxGJaYPsC>

tokens[0].keys()

# |%%--%%| <1OxGJaYPsC|fVNXRN7siN>

# create dataframe using partOfSpeech inside each token
df2 = pd.DataFrame([token["partOfSpeech"] for token in tokens])

# |%%--%%| <fVNXRN7siN|7cRjqxIvhZ>

df2

# |%%--%%| <7cRjqxIvhZ|7XKjSuSqtj>

# create a pandas series using content inside each token
# then insert into first column of df2
df2.insert(0, "content", pd.Series([token["text"]["content"] for token in tokens]))
df2

# |%%--%%| <7XKjSuSqtj|L6WvBeLivE>

# do the same for lemma
df2.insert(1, "lemma", pd.Series([token["lemma"] for token in tokens]))
df2

# |%%--%%| <L6WvBeLivE|gJyjA7DCRR>

tokens[0]["dependencyEdge"]

# |%%--%%| <gJyjA7DCRR|1TEEzksIls>

# add columns for 'dependencyEdge': {'headTokenIndex': 1, 'label': 'NN'},
df2["d_edge_head_index"] = pd.Series(
    [token["dependencyEdge"]["headTokenIndex"] for token in tokens]
)
df2["d_edge_label"] = pd.Series([token["dependencyEdge"]["label"] for token in tokens])
df2

# |%%--%%| <1TEEzksIls|OeVuAZJbrN>
r"""°°°
&#128161; **Tip:**

> You can consolidate the commands in the previous cells and create a function `get_tokens_dataframe(tokens)` to simplify the process
°°°"""
# |%%--%%| <OeVuAZJbrN|X9GGcDgjE4>
r"""°°°
Explore the different fields inside [Part of Speech](https://cloud.google.com/natural-language/docs/reference/rest/v1/Token#partofspeech).

For example, [Person](https://cloud.google.com/natural-language/docs/reference/rest/v1/Token#Person) indicates different perspectives (i.e. FIRST, SECOND, etc.).
°°°"""
# |%%--%%| <X9GGcDgjE4|HfjMT495Zw>

df2.loc[df2["person"] != "PERSON_UNKNOWN"]

# |%%--%%| <HfjMT495Zw|mrKVPBa6xY>
r"""°°°
What about [Number](https://cloud.google.com/natural-language/docs/reference/rest/v1/Token#number)? What does it represent?
°°°"""
# |%%--%%| <mrKVPBa6xY|dURqrxzFZk>

df2["number"].value_counts()

# |%%--%%| <dURqrxzFZk|SPG40iEZlT>

# which tokens are plural?
df2.loc[df2["number"] == "PLURAL"]

# |%%--%%| <SPG40iEZlT|C5nCDAtOVn>

# any token with known mood?
df2.loc[df2["mood"] != "MOOD_UNKNOWN"]

# |%%--%%| <C5nCDAtOVn|yVtUDl97Ly>
r"""°°°
Todo

> Try processing different documents and see how the part of speech (e.g. gender, number, person, etc.) changes. 
°°°"""
# |%%--%%| <yVtUDl97Ly|N4K4vWII34>

df2.info()

# |%%--%%| <N4K4vWII34|THNgCxnR7R>
r"""°°°
### Analyse Entities
°°°"""
# |%%--%%| <THNgCxnR7R|Bwl28PPltm>
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
# |%%--%%| <Bwl28PPltm|FMLaR1RuQJ>

entities = result["entities"]
entities[0]

# |%%--%%| <FMLaR1RuQJ|beKgAPM69r>

df3 = pd.DataFrame(entities)
df3

# |%%--%%| <beKgAPM69r|r0eWS1Q2xS>
r"""°°°
### Analyse Sentiment
°°°"""
# |%%--%%| <r0eWS1Q2xS|PNYTxWX46A>
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
# |%%--%%| <PNYTxWX46A|mxu7Z5bUuW>

result.keys()

# |%%--%%| <mxu7Z5bUuW|KqIir8YkdF>

result["documentSentiment"]

# |%%--%%| <KqIir8YkdF|ZsLJMYaezG>

result["sentences"]

# |%%--%%| <ZsLJMYaezG|7p43YunTKQ>
r"""°°°
### Analyse Entity Sentiment
°°°"""
# |%%--%%| <7p43YunTKQ|tuefWPg9o0>
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
# |%%--%%| <tuefWPg9o0|rVHxlu7RTz>
r"""°°°
Refer to Analyse Entity section on how to process entities.
°°°"""
# |%%--%%| <rVHxlu7RTz|trnKtRV6qd>
r"""°°°
### Classify Content
°°°"""
# |%%--%%| <trnKtRV6qd|Zanf9P1VBj>
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
# |%%--%%| <Zanf9P1VBj|kxGdwzSCVj>

result["categories"]

# |%%--%%| <kxGdwzSCVj|7R2bbBoBQs>
