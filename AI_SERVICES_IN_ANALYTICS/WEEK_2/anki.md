# Any Google Cloud {{resources}} that you {{allocate}} and {{use}} belong to a {{project}}.

# A {{project}} is made up of the {{settings}}, {{permissions}} and other {{metadata}} that describe your {{applications}}.

# Each Google Cloud project has a {{project name}} which is provided by {{you}}, {{project ID}} which is provided by {{you or Google }}, and a {{project number}} which is provided by {{Google}}.

# Each {{project}} is associated with {{one}} {{billing account}}.

# To interact with {{Google Services}}, you can {{use Google Cloud console}}, {{start Local terminal}} or {{Cloud shell}} and {{use Google Client Libraries}} or {{Google API Client Libraries}}.

# The recommended option for accessing {{Cloud APIs}} programmatically is using {{Cloud Client Libraries}}, not {{Cloud API Client Libraries}}.

# The benefits of {{Cloud Client Libraries}} are {{idiomatic code in each language}}, {{consistent style across client libraries}} and {{potential performance benefits with gRPC}}.

# One should use {{Cloud API Client Libraries over {{Cloud Client Libraries}} if {{there isn't a Client Library for your language}}, or {{your project already uses it}}.

# The downsides to using {{Cloud API Client Libraries}} is: {{gRPC is not supported}} and instead, it only has the {{REST interface}}, {{auto-generated interface code}} which is {{not as idiomatic}}, and {{a Single client for all libraries}},  where you're {{unable to choose which API to install}}.

# One should use Cloud Client and Cloud API Client Libraries because they provide {{abstraction}} of {{low-level communication}}, including {{authentication}}, and {{npm and pip installation}}.

# {{Principal}} is an {{identity}} that can be {{granted access}} to a {{resource}}.

# {{Authentication}} is the process of {{determining}} the {{identity of the principal}} attempting to access a {{resource}}.

# {{Authorization}} is determining whether the {{principal}} or {{application}} attempting to {{access a resource}} has been {{authorized}}

# {{Credentials}} are {{digital objects}} that provides {{proof of identity}}

# We need to provide {{credentials}} to access a {{protected}} resource, so as to {{prove your identity}} as a {{principal}} that has the correct level of {{authorization}}.

# {{Tokens}} are {{digital objects}} that proves that the user provided {{proper credentials}}

# Tokens contain information about {{the identity of the principal}} and {{what access}} they have.

# Google supports two types of {{principals}}, {{User accounts}} and {{Service accounts}}.

# {{User accounts}} represent a {{developer}}, {{administrator}} or any other person that {{interacts}} with {{Google APIs}} and {{services}}

# {{Service accounts}} do not represent a {{human user}}, and is managed by {{Identity and Access Management (IAM)}}

# To authenticate with a {{user}} account using {{user credentials}}, you can {{login to CLI}} or {{impersonate service account}} 

# To authenticate with a {{user}} account using {{gcloud CLI}}, you can setup {{Application Default Credential (ADC)}} or {{generate access token}}.

# To authenticate with a {{service}} account using {{not recommended}} processes, you can use {{default service account}} or {{service account key}}.

# To authenticate with a {{service}} account using {{recommended}} processes, you can {{attach a user-managed service}} and {{use Application Default Credentials}}, or to {{impersonate service account}}, which can allow you to temporarily {{grant more privileges}}.

# An {{API key}} is a Token that associates a {{request}} with a {{project}} for {{billing}} and {{quota}} purposes.

# Because {{API keys}} do not {{identify}} the caller, they're often used for accessing {{public}} data or resources

# How much API keys can you create in your project? 
300

# To use API Keys in Google's REST interface, POST https://vision.googleapis.com/v1/images:annotate{{?key=API_KEY}}

# The best practices for API keys are to {{add API key restrictions}}, {{delete unneeded API keys}} and {{regularly refresh API key}}.

# We add {{API key restrictions}} to our key to {{reduce impact}} of a {{compromised API key}}.

# We delete {{unneeded}} API keys to {{minimize attack potential}}.

# What is ADC?
Application Default Credentials

# {{ADC}} is a strategy used by Google authentication libraries to automatically {{find credentials available}} to {{Cloud Client Libraries}}.

# If you have set up the necessary {{IAM permissions}}, you should use {{gcloud credentials}}.

# The {{easiest}} and {{most secure}} way to provide credentials to REST in a {{local development}} environment is {{gcloud credentials}}.

# The preferred method to authenticate a REST call in {{production}} environment is {{ADC}}.

# You can use the {{gcloud CLI}} to create {{ADC}} credentials.

# If you can't use an {{attached service account}}, or you want to {{test}} whether a {{service account}} has {{sufficient IAM permissions}}, provide credentials by {{impersonating a service account}}.

# The credentials to login to {{google cloud console}} is separate from {{client libraries}}
