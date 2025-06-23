r"""°°°
![nyp.jpg](attachment:nyp.jpg)
°°°"""
# |%%--%%| <hxcwOEfuly|vwMqv5kNwj>
r"""°°°
## Google Credentials and Authentication
°°°"""
# |%%--%%| <vwMqv5kNwj|rdLwmBVL0Z>
r"""°°°
Previously in practical 2, you have
1. Installed Google Cloud SDK
2. Performed "gcloud init" after installation

After step 2, if you have given permission for Google Cloud SDK to access your Google Account and manage your resources (e.g. Compute Engine, Cloud SQL, etc.), your gcloud credentials will be stored in your local terminal (PC/laptop). Read on to find out where your gcloud credentials are stored.
°°°"""
# |%%--%%| <rdLwmBVL0Z|45IIkaqFff>
r"""°°°
> &#128161; **Tip**
> - You can run [gcloud](https://cloud.google.com/sdk/gcloud/reference) commands directly inside these shells (without the exclaimation mark):
        - Cloud SDK Shell (Start > Google Cloud SDK > Google Cloud SDK Shell)
        - Conda Prompt
        - Command prompt
> - Sometimes, when you execute gcloud commands in a jupyter notebook, the cell appears to "hang" because an error has been encountered but the error is not displayed explicitly. Run the gcloud command inside a shell to see the actual errors.
°°°"""
# |%%--%%| <45IIkaqFff|nsJ8FKfOwf>
r"""°°°
You can check your active account below.
°°°"""
# |%%--%%| <nsJ8FKfOwf|tznDVDshxD>

# likely a single gmail address with an * indicating it is active
!gcloud auth list

# |%%--%%| <tznDVDshxD|ksds29CoUg>
r"""°°°
*Sample Output:*
```
                   Credentialed Accounts
ACTIVE  ACCOUNT
*       example@gmail.com
```
°°°"""
# |%%--%%| <ksds29CoUg|IByUOnnfUz>
r"""°°°
### 1. Authenticating via gcloud credentials
°°°"""
# |%%--%%| <IByUOnnfUz|JOYXsPV1CE>
r"""°°°
At the start, when `gcloud init` was executed, you were indirectly authenticating as a user via `gcloud auth login`. After `gcloud auth login` is completed, you will be able to use your gcloud credentials to perform administrative tasks (e.g. set up project, roles, etc.) remotely.

It is equivalent to you signing into [console.cloud.google.com](console.cloud.google.com) to carry out the tasks but instead of doing it through the browser, you will be able to execute gcloud or curl commands to do it.

Run the following gcloud command to list the projects in your Google account. See [gcloud projects](https://cloud.google.com/sdk/gcloud/reference/projects) for commands to manage your projects.
°°°"""
# |%%--%%| <JOYXsPV1CE|7MNBNyI6tY>

!gcloud projects list --sort-by=projectId --limit=5

# |%%--%%| <7MNBNyI6tY|TKxdQcsy8H>
r"""°°°
Let's create a new project with your gcloud credentials
°°°"""
# |%%--%%| <TKxdQcsy8H|dmHin4FD3c>

# create a project with id and name 
!gcloud projects create proj-3386-testing-1 --name="it3386 test"

# |%%--%%| <dmHin4FD3c|nIPEfFE1VH>
r"""°°°
> If there is an error on "agree to terms and conditons", you will need to log into console.cloud.google.com and accept terms and conditions in the home page for first time use. See https://stackoverflow.com/questions/57145441/solution-google-cloud-sdk-issue-callers-must-accept-terms-of-service
°°°"""
# |%%--%%| <nIPEfFE1VH|eNfq8hhyMb>
r"""°°°
List the projects that are in your Google account. You should see the newly created project.
°°°"""
# |%%--%%| <eNfq8hhyMb|6e5y4W08Gr>

!gcloud projects list --sort-by=projectId --limit=5

# |%%--%%| <6e5y4W08Gr|VCIYrlhDfa>
r"""°°°
As a project [Owner](https://cloud.google.com/iam/docs/understanding-roles), you have permission to list details about your project.
°°°"""
# |%%--%%| <VCIYrlhDfa|coOfCPqm5h>

# ref https://cloud.google.com/sdk/gcloud/reference/projects/describe
!gcloud projects describe proj-3386-testing-1

# |%%--%%| <coOfCPqm5h|vjbyZpRiAM>
r"""°°°
As a project owner, you can also query members
°°°"""
# |%%--%%| <vjbyZpRiAM|cAvoqiO2XO>

!gcloud projects get-iam-policy proj-3386-testing-1 --format=json

# |%%--%%| <cAvoqiO2XO|GJXUw70AGD>



# |%%--%%| <GJXUw70AGD|8QnnQf4oCh>
r"""°°°
Back to the question, where are your gcloud credentials stored? 
> They are stored in your `User Config Directory`
°°°"""
# |%%--%%| <8QnnQf4oCh|o0KzrUdlv6>

!gcloud info

# |%%--%%| <o0KzrUdlv6|ROrW5fsgUe>
r"""°°°
*Sample Output*

```
...
User Config Directory: [C:\Users\sit\AppData\Roaming\gcloud]
...
```
°°°"""
# |%%--%%| <ROrW5fsgUe|Dm5ItPKXmY>



# |%%--%%| <Dm5ItPKXmY|xIIv3UCQ5K>
r"""°°°
Your gcloud credentials are stored in file `access_tokens.db` in the `User Config Directory`. Notice the timestamp of this file is updated every time you refresh your credentials via:
- `gcloud init`
- `gcloud auth login`
- `gcloud auth print-access-token`
°°°"""
# |%%--%%| <xIIv3UCQ5K|T944Z55cCN>
r"""°°°
Other than using gcloud commands, you can also use a curl command to issue a GET request with an access token and get some responses.
°°°"""
# |%%--%%| <T944Z55cCN|PfThEnJIrc>

access_token = !gcloud auth print-access-token
print(access_token)

# |%%--%%| <PfThEnJIrc|dGUWagcAU4>

# todo: fill in your own project id
project_id = 'proj-3386-testing-1'

# |%%--%%| <dGUWagcAU4|huKUNGAWuc>

endpoint_rm = 'https://cloudresourcemanager.googleapis.com/v3/projects/{}'.format(project_id)
print(endpoint_rm)

# |%%--%%| <huKUNGAWuc|FJzb5LlRHy>

header_token = 'Authorization: Bearer {}'.format(access_token[0])
print(header_token)

# |%%--%%| <FJzb5LlRHy|xemvosiCQY>

arg = '-X GET -H "{}" "{}"'.format(header_token, endpoint_rm)

# |%%--%%| <xemvosiCQY|PrTExPtMJr>

print(arg)

# |%%--%%| <PrTExPtMJr|efxZ0ToVIj>

# finding out project details using curl
!curl {arg}

# |%%--%%| <efxZ0ToVIj|ylecZoatu0>



# |%%--%%| <ylecZoatu0|eWmTIeGNaZ>
r"""°°°
### 2. Authenticating via Application Default Credentials (ADC)
°°°"""
# |%%--%%| <eWmTIeGNaZ|V4BErIjnBe>
r"""°°°
If you are a developer who wish to use Cloud Client Libraries (e.g. Python) to call the Google services (e.g. Cloud Natural Language), you should  authenticate as a user via `gcloud auth application-default login`:
- this command creates a credentials JSON named `application_default_credentials.json` (ADC) which can also be found in the `User Config Directory`
- if the `GOOGLE_APPLICATON_CREDENTIALS` environment is <ins>not set</ins> (to load a service key), ADC is the credential that will be used by your Cloud Client Library
°°°"""
# |%%--%%| <V4BErIjnBe|et1i3w7Yfo>
r"""°°°
At this point, you probably do not have `application_default_credentials.json` yet. Complete the following:
1. Launch any shell (i.e. Cloud SDK Shell, Conda/Command Prompt)
2. Execute `gcloud auth application-default login`
3. Allow permissions in the browser
4. Complete the process

Now, you should be able to see your ADC credentials using File Explorer.
°°°"""
# |%%--%%| <et1i3w7Yfo|MwDz5T4JtR>
r"""°°°
In order to use your ADC to call a Google service (e.g. Sentiment Analysis), you will need to enable the following:
1. The service API (e.g. Cloud Natural Language API) in the corresponding project specified in your ADC
2. Billing for your account

To enable billing, you will need a debit/credit card. Since you might not have a card, we will skip the process of using ADC. You can just refer to the codes in the next section to familiarise with the process.
°°°"""
# |%%--%%| <MwDz5T4JtR|jMxaElW4W2>



# |%%--%%| <jMxaElW4W2|EbNgcVQFOr>
r"""°°°
### 3. Entity Analysis using CURL + ADC (Walk-through)

This section of the codes assumes you have already enabled the service API and billing.

[Entity Analysis](https://cloud.google.com/natural-language/docs/sentiment-analysis-gcloud#quickstart-analyze-entities-cli).
°°°"""
# |%%--%%| <EbNgcVQFOr|lp2lqFUnIO>

adc_token = !gcloud auth application-default print-access-token

# |%%--%%| <lp2lqFUnIO|nkn7b98KWC>

print(adc_token)

# |%%--%%| <nkn7b98KWC|iJQEPT6btI>
r"""°°°
We will use <code>to_analyse.json</code> file to specify the input request. Make sure the file is present in your current directory.
°°°"""
# |%%--%%| <iJQEPT6btI|PtZjdO8iDN>

# view the content of file
!type ./to_analyse.json

#|%%--%%| <PtZjdO8iDN|rxKU0hDvSl>

!cat to_analyse.json

# |%%--%%| <rxKU0hDvSl|koRMA4ocUv>

header_project = 'x-goog-user-project: {}'.format(project_id)
content_type = 'Content-Type: application/json; charset=utf-8'
endpoint = 'https://language.googleapis.com/v1/documents:analyzeEntities'

# |%%--%%| <koRMA4ocUv|XcNvycv0jD>

arg = '-X POST -H "Authorization: Bearer {}" -H "{}" -H "{}" --data @to_analyse.json "{}"'.format(adc_token[0], header_project, content_type, endpoint)
#print(arg)

# |%%--%%| <XcNvycv0jD|JWHrDZ56C7>

!curl {arg}

# |%%--%%| <JWHrDZ56C7|4u0IfJ1URz>



# |%%--%%| <4u0IfJ1URz|umuXPiYeNk>



# |%%--%%| <umuXPiYeNk|Ty281GetP4>
r"""°°°
### 4. Sentiment Analysis using Cloud Client Library + Service Account Key

As you have seen earlier, without billing and Cloud Natural Language API service enabled for your newly created project, you will not be able to perform sentiment analysis using your "developer" (i.e. `gcloud auth application-default login`) account. 

This is where a service account key can be used. You can define a service key in the environment `GOOGLE_APPLICATION_CREDENTIALS`. Remember, this environment is the first place Google will look for credentials to perform any services.
°°°"""
# |%%--%%| <Ty281GetP4|Et0AnU3umU>
r"""°°°
You will use the Cloud Client Library with a service account key to analyse sentiment. Copy and paste the service account key you have used in Practical 2 to the current directory.
°°°"""
# |%%--%%| <Et0AnU3umU|5ZPYIgUzWx>

# you should see the service key it3386_practical.json in this directory
!dir *.json

#|%%--%%| <5ZPYIgUzWx|UUgz8ydgPZ>

!cat ../WEEK_2/it3386-2024-s2.json

# |%%--%%| <UUgz8ydgPZ|jDENmVaACx>

%env GOOGLE_APPLICATION_CREDENTIALS=../WEEK_2/it3386-2024-s2.json

# |%%--%%| <jDENmVaACx|ViFHUHnq1M>

%env GOOGLE_APPLICATION_CREDENTIALS

# |%%--%%| <ViFHUHnq1M|1DhlMrDqcT>

# if error, go to conda prompt and execute command below
# conda install -c conda-forge google-cloud-language
from google.cloud import language_v1

# |%%--%%| <1DhlMrDqcT|mgE6hHdvJ8>

# ref https://cloud.google.com/natural-language/docs/analyzing-entities
def analyse_sentiment(text_content):
    """
    Analyzes Sentiment in a string.

    Args:
      text_content: The text content to analyze.
    """

    client = language_v1.LanguageServiceClient()

    # Available types: PLAIN_TEXT, HTML
    document_type_in_plain_text = language_v1.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages
    language_code = "en"
    document = {
        "content": text_content,
        "type_": document_type_in_plain_text
    }

    # Available values: NONE, UTF8, UTF16, UTF32
    # See https://cloud.google.com/natural-language/docs/reference/rest/v2/EncodingType.
    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_sentiment(
        request={"document": document, "encoding_type": encoding_type}
    )
    
    return response

# |%%--%%| <mgE6hHdvJ8|h8PUxn6M0v>

document = "Nanyang Polytechnic gives our students the head start they are looking for in their next phase in life with our innovative teaching methods and industry-focused projects. They'll not only be academically prepared, but also future-ready - equipped to tackle whatever life throws at them in their career or further education. Our annual Graduate Employment Surveys show that our students are consistently highly sought-after by employers in multiple industries. Many of our graduates have also gone on to local and overseas universities, where they continue to excel in their field of study."

# |%%--%%| <h8PUxn6M0v|YYOHC5851y>

print(document)

# |%%--%%| <YYOHC5851y|jPavkBgqdS>

annotations = analyse_sentiment(document)

# |%%--%%| <jPavkBgqdS|P9xwvYjtX8>

# raw annotations
annotations

# |%%--%%| <P9xwvYjtX8|5bUZPkcXg9>

score = annotations.document_sentiment.score
magnitude = annotations.document_sentiment.magnitude

for index, sentence in enumerate(annotations.sentences):
    sentence_sentiment = sentence.sentiment.score
    print(
        "Sentence {} has a sentiment score of {}".format(index, sentence_sentiment)
    )

print(
    "Overall Sentiment: score of {} with magnitude of {}".format(score, magnitude)
)

# |%%--%%| <5bUZPkcXg9|yyZQbcQCti>



# |%%--%%| <yyZQbcQCti|0XGDx4ZL17>
r"""°°°
Todo

> Refer to the documentation and try the rest of the operations (e.g. analyse entities)
°°°"""
# |%%--%%| <0XGDx4ZL17|faV1u3i37c>



# |%%--%%| <faV1u3i37c|wm9F1mqpG3>
r"""°°°
### Recap

You are now able to access Google Cloud API Services using:
- API Key
- Cloud Client Library
- gcloud or curl commands using gcloud and ADC credentials

Summary:
- For administrative tasks, you can execute gcloud or curl commands using gcloud credentials
- For calling Google services, 
    - Google will look for environmental variable GOOGLE_APPLICATION_CREDENTIALS first. If you have a service key, assign it to that env.
    - If GOOGLE_APPLICATION_CREDENTIALS is not found, ADC will be used

°°°"""
# |%%--%%| <wm9F1mqpG3|wBAz5Kj6of>


