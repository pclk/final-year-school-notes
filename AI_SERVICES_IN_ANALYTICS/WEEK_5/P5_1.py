r"""°°°
##### Copyright 2024 Google LLC.
°°°"""
# |%%--%%| <dH7XFZUGod|NMKR6UcX08>

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# |%%--%%| <NMKR6UcX08|HVkovL0EaB>
r"""°°°
# Get started with the Gemini API: Python
°°°"""
# |%%--%%| <HVkovL0EaB|8CaPs86Hgs>
r"""°°°
<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://ai.google.dev/gemini-api/docs/get-started/python"><img src="https://ai.google.dev/static/site-assets/images/docs/notebook-site-button.png" height="32" width="32" />View on Google AI</a>
  </td>
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/gemini-api/docs/get-started/python.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
  </td>
  <td>
    <a target="_blank" href="https://github.com/google/generative-ai-docs/blob/main/site/en/gemini-api/docs/get-started/python.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
  </td>
</table>
°°°"""
# |%%--%%| <8CaPs86Hgs|FiRDOJgC8h>
r"""°°°
This quickstart demonstrates how to use the Python SDK for the Gemini API, which gives you access to Google's Gemini large language models. In this quickstart, you will learn how to:

1. Set up your development environment and API access to use Gemini.
2. Generate text responses from text inputs.
3. Generate text responses from multimodal inputs (text and images).
4. Use Gemini for multi-turn conversations (chat).

°°°"""
# |%%--%%| <FiRDOJgC8h|Mu3mABpwmq>
r"""°°°
## Prerequisites

You can run this quickstart in [Google Colab](https://colab.research.google.com/github/google/generative-ai-docs/blob/main/site/en/tutorials/python_quickstart.ipynb), which runs this notebook directly in the browser and does not require additional environment configuration.

Alternatively, to complete this quickstart locally, ensure that your development environment meets the following requirements:

-  Python 3.9+
-  An installation of `jupyter` to run the notebook.
°°°"""
# |%%--%%| <Mu3mABpwmq|EHLrX9CzQg>
r"""°°°
## Setup
°°°"""
# |%%--%%| <EHLrX9CzQg|nYv0b0Oc9f>
r"""°°°
### Install the Python SDK

The Python SDK for the Gemini API, is contained in the [`google-generativeai`](https://pypi.org/project/google-generativeai/) package. Install the dependency using pip:
°°°"""
# |%%--%%| <nYv0b0Oc9f|zlAimbdnAk>

!pip install -q -U google-generativeai

# |%%--%%| <zlAimbdnAk|OVZmX5pfJJ>
r"""°°°
### Import packages
°°°"""
# |%%--%%| <OVZmX5pfJJ|QhLEIP5bY5>
r"""°°°
Import the necessary packages.
°°°"""
# |%%--%%| <QhLEIP5bY5|A7gsmYSSyE>

import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# |%%--%%| <A7gsmYSSyE|aCpzcSoAqY>

# Used to securely store your API key
from google.colab import userdata

# |%%--%%| <aCpzcSoAqY|0vIlM8Pr2n>
r"""°°°
### Setup your API key

Before you can use the Gemini API, you must first obtain an API key. If you don't already have one, create a key with one click in Google AI Studio.

<a class="button button-primary" href="https://makersuite.google.com/app/apikey" target="_blank" rel="noopener noreferrer">Get an API key</a>
°°°"""
# |%%--%%| <0vIlM8Pr2n|FuHx4PyFSy>
r"""°°°
In Colab, add the key to the secrets manager under the "🔑" in the left panel. Give it the name `GOOGLE_API_KEY`.
°°°"""
# |%%--%%| <FuHx4PyFSy|EzSa0duOfh>
r"""°°°
Once you have the API key, pass it to the SDK. You can do this in two ways:

* Put the key in the `GOOGLE_API_KEY` environment variable (the SDK will automatically pick it up from there).
* Pass the key to `genai.configure(api_key=...)`
°°°"""
# |%%--%%| <EzSa0duOfh|MqE8wD8BWm>

# Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.
GOOGLE_API_KEY="AIzaSyB3hD-Q6xTZVA_Ea2ItLOJnmGokMDlLccc"

genai.configure(api_key=GOOGLE_API_KEY)

# |%%--%%| <MqE8wD8BWm|D597UFgyrj>
r"""°°°
## List models

Now you're ready to call the Gemini API. Use `list_models` to see the available Gemini models:

* `gemini-pro`: optimized for text-only prompts.
* `gemini-pro-vision`: optimized for text-and-images prompts.
°°°"""
# |%%--%%| <D597UFgyrj|J7SFbpYWff>

for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m)

# |%%--%%| <J7SFbpYWff|yCbLPNfkEN>
r"""°°°
Note: For detailed information about the available models, including their capabilities and rate limits, see [Gemini models](https://ai.google.dev/models/gemini). There are options for requesting [rate limit increases](https://ai.google.dev/docs/increase_quota). The rate limit for Gemini-Pro models is 60 requests per minute (RPM).

The `genai` package also supports the PaLM  family of models, but only the Gemini models support the generic, multimodal capabilities of the `generateContent` method.
°°°"""
# |%%--%%| <yCbLPNfkEN|Okw5Vu7Oxx>
r"""°°°
## Generate text from text inputs

For text-only prompts, use the `gemini-pro` model:
°°°"""
# |%%--%%| <Okw5Vu7Oxx|etxD25aJuW>

model = genai.GenerativeModel('gemini-pro')

# |%%--%%| <etxD25aJuW|9ZkDlF9JiD>
r"""°°°
The `generate_content` method can handle a wide variety of use cases, including multi-turn chat and multimodal input, depending on what the underlying model supports. The available models only support text and images as input, and text as output.

In the simplest case, you can pass a prompt string to the <a href="https://ai.google.dev/api/python/google/generativeai/GenerativeModel#generate_content"><code>GenerativeModel.generate_content</code></a> method:
°°°"""
# |%%--%%| <9ZkDlF9JiD|C0qIjmRt05>

%%time
response = model.generate_content("What is the meaning of life?")

# |%%--%%| <C0qIjmRt05|kYvail3iPM>
r"""°°°
In simple cases, the `response.text` accessor is all you need. To display formatted Markdown text, use the `to_markdown` function:
°°°"""
# |%%--%%| <kYvail3iPM|aWbPp6zfgT>

to_markdown(response.text)

#|%%--%%| <aWbPp6zfgT|QGtlgvFy4e>

response.text

# |%%--%%| <QGtlgvFy4e|Rg38CHoTjO>
r"""°°°
If the API failed to return a result, use `GenerateContentResponse.prompt_feedback` to see if it was blocked due to safety concerns regarding the prompt.
°°°"""
# |%%--%%| <Rg38CHoTjO|OgPgp8bKEz>

response.prompt_feedback

# |%%--%%| <OgPgp8bKEz|Ulo8bStgzt>
r"""°°°
Gemini can generate multiple possible responses for a single prompt. These possible responses are called `candidates`, and you can review them to select the most suitable one as the response.

View the response candidates with <a href="https://ai.google.dev/api/python/google/ai/generativelanguage/GenerateContentResponse#candidates"><code>GenerateContentResponse.candidates</code></a>:
°°°"""
# |%%--%%| <Ulo8bStgzt|Ts8jEDF4hR>

response.candidates

# |%%--%%| <Ts8jEDF4hR|2Y0a02iM32>
r"""°°°
By default, the model returns a response after completing the entire generation process. You can also stream the response as it is being generated, and the model will return chunks of the response as soon as they are generated.

To stream responses, use <a href="https://ai.google.dev/api/python/google/generativeai/GenerativeModel#generate_content"><code>GenerativeModel.generate_content(..., stream=True)</code></a>.
°°°"""
# |%%--%%| <2Y0a02iM32|5pQrruuhLP>

%%time
response = model.generate_content("What is the meaning of life?", stream=True)

# |%%--%%| <5pQrruuhLP|UNi7UD3ywc>

type(response)

# |%%--%%| <UNi7UD3ywc|B5LQYPa0Yd>

for chunk in response:
  print(chunk.text)
  print("_"*80)

# |%%--%%| <B5LQYPa0Yd|oFPoCcChKu>
r"""°°°
When streaming, some response attributes are not available until you've iterated through all the response chunks. This is demonstrated below:
°°°"""
# |%%--%%| <oFPoCcChKu|u3DCfOPf9E>

response = model.generate_content("What is the meaning of life?", stream=True)

# |%%--%%| <u3DCfOPf9E|Jh1cpOKJun>
r"""°°°
The `prompt_feedback` attribute works:
°°°"""
# |%%--%%| <Jh1cpOKJun|we1JWTFHAL>

response.prompt_feedback

# |%%--%%| <we1JWTFHAL|YycoxFkEm0>
r"""°°°
But attributes like <code>text</code> do not:
°°°"""
# |%%--%%| <YycoxFkEm0|1uDULOiX17>

try:
  response.text
except Exception as e:
  print(f'{type(e).__name__}: {e}')

# |%%--%%| <1uDULOiX17|T1PQ9mX4eg>
r"""°°°
## Generate text from image and text inputs

Gemini provides a multimodal model (`gemini-pro-vision`) that accepts both text and images and inputs. The `GenerativeModel.generate_content` API is designed to handle multimodal prompts and returns a text output.

Let's include an image:
°°°"""
# |%%--%%| <T1PQ9mX4eg|lHs4oT77R3>

!curl -o image.jpg https://t0.gstatic.com/licensed-image?q=tbn:ANd9GcQ_Kevbk21QBRy-PgB4kQpS79brbmmEG7m3VOTShAn4PecDU5H5UxrJxE3Dw1JiaG17V88QIol19-3TM2wCHw

# |%%--%%| <lHs4oT77R3|cFyqLl1cOU>

import PIL.Image

img = PIL.Image.open('dinner_table.jpg')
img

# |%%--%%| <cFyqLl1cOU|S2gJUBVZQ8>
r"""°°°
Use the `gemini-pro-vision` model and pass the image to the model with `generate_content`.
°°°"""
# |%%--%%| <S2gJUBVZQ8|Ukz3mtjVhH>

model = genai.GenerativeModel('gemini-pro-vision')

# |%%--%%| <Ukz3mtjVhH|EaSEoajts2>

response = model.generate_content(img)

to_markdown(response.text)


# |%%--%%| <EaSEoajts2|klcNpBp8BT>
r"""°°°
To provide both text and images in a prompt, pass a list containing the strings and images:
°°°"""
# |%%--%%| <klcNpBp8BT|y6emyfSQC4>

response = model.generate_content(["Write a short, engaging blog post based on this picture.", img], stream=True)
response.resolve()

# |%%--%%| <y6emyfSQC4|7UV0MlFYah>

to_markdown(response.text)

# |%%--%%| <7UV0MlFYah|Yz7u6KCJSL>
r"""°°°
## Chat conversations

Gemini enables you to have freeform conversations across multiple turns. The `ChatSession` class simplifies the process by managing the state of the conversation, so unlike with `generate_content`, you do not have to store the conversation history as a list.

Initialize the chat:
°°°"""
# |%%--%%| <Yz7u6KCJSL|60KRwaGKnj>

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])
chat

# |%%--%%| <60KRwaGKnj|hrQ8EXLHyR>
r"""°°°
Note: The vision model `gemini-pro-vision` is not optimized for multi-turn chat.
°°°"""
# |%%--%%| <hrQ8EXLHyR|p5jKWQILUG>
r"""°°°
The `ChatSession.send_message` method returns the same `GenerateContentResponse` type as <a href="https://ai.google.dev/api/python/google/generativeai/GenerativeModel#generate_content"><code>GenerativeModel.generate_content</code></a>. It also appends your message and the response to the chat history:
°°°"""
# |%%--%%| <p5jKWQILUG|9k4ZKELZZR>

response = chat.send_message("In one sentence, explain how a computer works to a young child.")
to_markdown(response.text)

# |%%--%%| <9k4ZKELZZR|qEzcdoKJD2>

chat.history

# |%%--%%| <qEzcdoKJD2|UII7CDiUVm>
r"""°°°
You can keep sending messages to continue the conversation. Use the `stream=True` argument to stream the chat:
°°°"""
# |%%--%%| <UII7CDiUVm|KV0JBmR5pl>

response = chat.send_message("Okay, how about a more detailed explanation to a high schooler?", stream=True)

for chunk in response:
  print(chunk.text)
  print("_"*80)

# |%%--%%| <KV0JBmR5pl|0PyGCha6sm>

chat.history

# |%%--%%| <0PyGCha6sm|pgL1NUkczo>
r"""°°°
`glm.Content` objects contain a list of `glm.Part` objects that each contain either a text (string) or inline_data (`glm.Blob`), where a blob contains binary data and a `mime_type`. The chat history is available as a list of `glm.Content` objects in `ChatSession.history`:
°°°"""
# |%%--%%| <pgL1NUkczo|zNG5sLxEeu>

for message in chat.history:
  display(to_markdown(f'**{message.role}**: {message.parts[0].text}'))

# |%%--%%| <zNG5sLxEeu|aI8eSwjZv6>
r"""°°°
**History makes a difference to the response**
°°°"""
# |%%--%%| <aI8eSwjZv6|9geXSTPtKU>

chat = model.start_chat(history=[])
response = chat.send_message('What is the capital of New Zealand?')
to_markdown(response.text)

# |%%--%%| <9geXSTPtKU|8gmifMvWVt>

chat.history

# |%%--%%| <8gmifMvWVt|C10GAPSm8T>

response = chat.send_message('Is it in the north or south island?')
to_markdown(response.text)

# |%%--%%| <C10GAPSm8T|d2SqriGfoK>

chat.history

# |%%--%%| <d2SqriGfoK|H1ErRkhLp2>
r"""°°°
Compare with a normal search

°°°"""
# |%%--%%| <H1ErRkhLp2|bKWDBrM4CR>

model1 = genai.GenerativeModel('gemini-pro')
response = model1.generate_content('what is the capital of New Zealand?')

# |%%--%%| <bKWDBrM4CR|rK9nyEfFcD>

response.text

# |%%--%%| <rK9nyEfFcD|NZBLBBXLFs>

response1 = model1.generate_content('Is it in the north or south island?')
response1.text

# |%%--%%| <NZBLBBXLFs|xup0Yoe9qU>
r"""°°°
## Count tokens

Large language models have a context window, and the context length is often measured in terms of the **number of tokens**. With the Gemini API, you can determine the number of tokens per any `glm.Content` object. In the simplest case, you can pass a query string to the `GenerativeModel.count_tokens` method as follows:
°°°"""
# |%%--%%| <xup0Yoe9qU|WvyDEOExEv>

model.count_tokens("What is the meaning of life?")

# |%%--%%| <WvyDEOExEv|b1J3UwXmBZ>
r"""°°°
Similarly, you can check `token_count` for your `ChatSession`:
°°°"""
# |%%--%%| <b1J3UwXmBZ|ItfjcKrNp1>

model.count_tokens(chat.history)

# |%%--%%| <ItfjcKrNp1|22ITHYwggu>
r"""°°°
## Advanced use cases

The following sections discuss advanced use cases and lower-level details of the Python SDK for the Gemini API.
°°°"""
# |%%--%%| <22ITHYwggu|hn5g41fi5O>
r"""°°°
### Safety settings

The `safety_settings` argument lets you configure what the model blocks and allows in both prompts and responses. By default, safety settings block content with medium and/or high probability of being unsafe content across all dimensions. Learn more about [Safety settings](https://ai.google.dev/docs/safety_setting).

Enter a questionable prompt and run the model with the default safety settings, and it will not return any candidates:
°°°"""
# |%%--%%| <hn5g41fi5O|pxw7LQUny3>

#response = model.generate_content('[Questionable prompt here]')
#response.candidates

# |%%--%%| <pxw7LQUny3|FAqW4JNXPd>
r"""°°°
The `prompt_feedback` will tell you which safety filter blocked the prompt:
°°°"""
# |%%--%%| <FAqW4JNXPd|BUCmO2JpUI>

response.prompt_feedback

# |%%--%%| <BUCmO2JpUI|RJOdUEIdMq>
r"""°°°
Also note that each candidate has its own `safety_ratings`, in case the prompt passes but the individual responses fail the safety checks.
°°°"""
# |%%--%%| <RJOdUEIdMq|18XQFNfpmB>
r"""°°°
### Encode messages
°°°"""
# |%%--%%| <18XQFNfpmB|IdfP431Daq>
r"""°°°
The previous sections relied on the SDK to make it easy for you to send prompts to the API. This section offers a fully-typed equivalent to the previous example, so you can better understand the lower-level details regarding how the SDK encodes messages.
°°°"""
# |%%--%%| <IdfP431Daq|dUteFKaCPO>
r"""°°°
Underlying the Python SDK is the <a href="https://ai.google.dev/api/python/google/ai/generativelanguage"><code>google.ai.generativelanguage</code></a> client library:
°°°"""
# |%%--%%| <dUteFKaCPO|dFeNC2Oc9R>

import google.ai.generativelanguage as glm

# |%%--%%| <dFeNC2Oc9R|MvC6VfbhSI>
r"""°°°
The SDK attempts to convert your message to a `glm.Content` object, which contains a list of `glm.Part` objects that each contain either:

1. a <a href="https://www.tensorflow.org/text/api_docs/python/text"><code>text</code></a> (string)
2. `inline_data` (`glm.Blob`), where a blob contains binary `data` and a `mime_type`.

You can also pass any of these classes as an equivalent dictionary.

Note: The only accepted mime types are some image types, `image/*`.

So, the fully-typed equivalent to the previous example is:  
°°°"""
# |%%--%%| <MvC6VfbhSI|VrecmeSGUk>

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content('Which war movies did Nicholas Cage starred in?')

# |%%--%%| <VrecmeSGUk|h9RTICAtMB>

response

# |%%--%%| <h9RTICAtMB|RmfzuW4HlL>

response.text

# |%%--%%| <RmfzuW4HlL|keSWWJBSyi>

response.candidates[0].content

# |%%--%%| <keSWWJBSyi|K1uHvI6AqZ>

model = genai.GenerativeModel('gemini-pro-vision')
response = model.generate_content(
    glm.Content(
        parts = [
            glm.Part(text="Write a short, engaging blog post based on this picture."),
            glm.Part(
                inline_data=glm.Blob(
                    mime_type='image/jpeg',
                    data=pathlib.Path('image.jpg').read_bytes()
                )
            ),
        ],
    ),
    stream=True)

# |%%--%%| <K1uHvI6AqZ|uHxcIdTbvN>

response.resolve()
to_markdown(response.text)

# |%%--%%| <uHxcIdTbvN|82P3HyVhZ4>
r"""°°°
### Multi-turn conversations

While the `genai.ChatSession` class shown earlier can handle many use cases, it does make some assumptions. If your use case doesn't fit into this chat implementation it's good to remember that `genai.ChatSession` is just a wrapper around <a href="https://ai.google.dev/api/python/google/generativeai/GenerativeModel#generate_content"><code>GenerativeModel.generate_content</code></a>. In addition to single requests, it can handle multi-turn conversations.

The individual messages are `glm.Content` objects or compatible dictionaries, as seen in previous sections. As a dictionary, the message requires `role` and `parts` keys. The `role` in a conversation can either be the `user`, which provides the prompts, or `model`, which provides the responses.

Pass a list of `glm.Content` objects and it will be treated as multi-turn chat:
°°°"""
# |%%--%%| <82P3HyVhZ4|dKHeU3zCut>

model = genai.GenerativeModel('gemini-pro')

messages = [
    {'role':'user',
     'parts': ["Briefly explain how a computer works to a young child."]}
]
response = model.generate_content(messages)

to_markdown(response.text)

# |%%--%%| <dKHeU3zCut|GjtfHrYdIu>
r"""°°°
To continue the conversation, add the response and another message.

Note: For multi-turn conversations, you need to send the whole conversation history with each request. The API is **stateless**.
°°°"""
# |%%--%%| <GjtfHrYdIu|gSSvsNolI0>

messages.append({'role':'model',
                 'parts':[response.text]})

messages.append({'role':'user',
                 'parts':["Okay, how about a more detailed explanation to a high school student?"]})

response = model.generate_content(messages)
print(response.text)
to_markdown(response.text)

# |%%--%%| <gSSvsNolI0|ilssjRnkIx>

response.candidates

# |%%--%%| <ilssjRnkIx|iVf61fZ7i5>
r"""°°°
### Generation configuration

The `generation_config` argument allows you to modify the generation parameters. Every prompt you send to the model includes parameter values that control how the model generates responses.
°°°"""
# |%%--%%| <iVf61fZ7i5|LpLTil0lE1>

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content(
    'Tell me a story about a magic backpack.',
    generation_config=genai.types.GenerationConfig(
        # Only one candidate for now.
        candidate_count=1,
        stop_sequences=['town'],
        max_output_tokens=20,
        temperature=1.0)
)

# |%%--%%| <LpLTil0lE1|mqUKSVJXyg>

# stopped story at town
response

# |%%--%%| <mqUKSVJXyg|Z86JORtaTj>
r"""°°°
https://cloud.google.com/python/docs/reference/aiplatform/latest/vertexai.generative_models.FinishReason

```
FINISH_REASON_UNSPECIFIED (0):
    The finish reason is unspecified.
STOP (1):
    Natural stop point of the model or provided
    stop sequence.
MAX_TOKENS (2):
    The maximum number of tokens as specified in
    the request was reached.
SAFETY (3):
```
°°°"""
# |%%--%%| <Z86JORtaTj|lFlJXz9Bc9>

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content(
    'Tell me a story about a magic backpack.',
    generation_config=genai.types.GenerationConfig(
        # Only one candidate for now.
        candidate_count=1,
        stop_sequences=['x'],
        max_output_tokens=200,
        temperature=1.0)
)

# |%%--%%| <lFlJXz9Bc9|qMA2FvWHCn>

response

# |%%--%%| <qMA2FvWHCn|ujJseg8Ba9>

response.candidates[0].finish_reason.name

# |%%--%%| <ujJseg8Ba9|v2ksa4eIKe>

# did not get max token because reason is stop, even when max tokens reached
text = response.text

if response.candidates[0].finish_reason.name == "MAX_TOKENS":
    text += '...'

to_markdown(text)

# |%%--%%| <v2ksa4eIKe|ToJJ7iPsGs>

response = model.generate_content(
    'Tell me a story about a magic backpack.',
    generation_config=genai.types.GenerationConfig(
        # Only one candidate for now.
        candidate_count=1,
        stop_sequences=['x'],
        max_output_tokens=1000,
        temperature=1.0)
)

# |%%--%%| <ToJJ7iPsGs|WswsCjoJDO>

response

# |%%--%%| <WswsCjoJDO|KSCKVYAMLM>
r"""°°°
## What's next

-   Prompt design is the process of creating prompts that elicit the desired response from language models. Writing well structured prompts is an essential part of ensuring accurate, high quality responses from a language model. Learn about best practices for [prompt writing](https://ai.google.dev/docs/prompt_best_practices).
-   Gemini offers several model variations to meet the needs of different use cases, such as input types and complexity, implementations for chat or other dialog language tasks, and size constraints. Learn about the available [Gemini models](https://ai.google.dev/models/gemini).
-   Gemini offers options for requesting [rate limit increases](https://ai.google.dev/docs/increase_quota). The rate limit for Gemini-Pro models is 60 requests per minute (RPM).
°°°"""
