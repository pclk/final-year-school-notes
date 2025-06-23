model: Basic

# Note
model: Cloze

## Text
Any Google Cloud {{c1::resources}} that you {{c2::allocate}} and {{c3::use}} belong to a {{c4::project}}.

## Back Extra


# Note
model: Cloze

## Text
A {{c1::project}} is made up of the {{c2::settings}}, {{c3::permissions}} and other {{c4::metadata}} that describe your {{c5::applications}}.

## Back Extra


# Note
model: Cloze

## Text
Each Google Cloud project has a {{c1::project name}} which is provided by {{c2::you}}, {{c3::project ID}} which is provided by {{c4::you or Google }}, and a {{c5::project number}} which is provided by {{c6::Google}}.

## Back Extra


# Note
model: Cloze

## Text
Each {{c1::project}} is associated with {{c2::one}} {{c3::billing account}}.

## Back Extra


# Note
model: Cloze

## Text
To interact with {{c1::Google Services}}, you can {{c2::use Google Cloud console}}, {{c3::start Local terminal}} or {{c4::Cloud shell}} and {{c5::use Google Client Libraries}} or {{c6::Google API Client Libraries}}.

## Back Extra


# Note
model: Cloze

## Text
The recommended option for accessing {{c1::Cloud APIs}} programmatically is using {{c2::Cloud Client Libraries}}, not {{c3::Cloud API Client Libraries}}.

## Back Extra


# Note
model: Cloze

## Text
The benefits of {{c1::Cloud Client Libraries}} are {{c2::idiomatic code in each language}}, {{c3::consistent style across client libraries}} and {{c4::potential performance benefits with gRPC}}.

## Back Extra


# Note
model: Cloze

## Text
One should use {{c1::Cloud API Client Libraries over {{Cloud Client Libraries}} if {{c2::there isn't a Client Library for your language}}, or {{c3::your project already uses it}}.

## Back Extra


# Note
model: Cloze

## Text
The downsides to using {{c1::Cloud API Client Libraries}} is: {{c2::gRPC is not supported}} and instead, it only has the {{c3::REST interface}}, {{c4::auto-generated interface code}} which is {{c5::not as idiomatic}}, and {{c6::a Single client for all libraries}},  where you're {{c7::unable to choose which API to install}}.

## Back Extra


# Note
model: Cloze

## Text
One should use Cloud Client and Cloud API Client Libraries because they provide {{c1::abstraction}} of {{c2::low-level communication}}, including {{c3::authentication}}, and {{c4::npm and pip installation}}.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Principal}} is an {{c2::identity}} that can be {{c3::granted access}} to a {{c4::resource}}.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Authentication}} is the process of {{c2::determining}} the {{c3::identity of the principal}} attempting to access a {{c4::resource}}.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Authorization}} is determining whether the {{c2::principal}} or {{c3::application}} attempting to {{c4::access a resource}} has been {{c5::authorized}}

## Back Extra


# Note
model: Cloze

## Text
{{c1::Credentials}} are {{c2::digital objects}} that provides {{c3::proof of identity}}

## Back Extra


# Note
model: Cloze

## Text
We need to provide {{c1::credentials}} to access a {{c2::protected}} resource, so as to {{c3::prove your identity}} as a {{c4::principal}} that has the correct level of {{c5::authorization}}.

## Back Extra


# Note
model: Cloze

## Text
{{c1::Tokens}} are {{c2::digital objects}} that proves that the user provided {{c3::proper credentials}}

## Back Extra


# Note
model: Cloze

## Text
Tokens contain information about {{c1::the identity of the principal}} and {{c2::what access}} they have.

## Back Extra


# Note
model: Cloze

## Text
Google supports two types of {{c1::principals}}, {{c2::User accounts}} and {{c3::Service accounts}}.

## Back Extra


# Note
model: Cloze

## Text
{{c1::User accounts}} represent a {{c2::developer}}, {{c3::administrator}} or any other person that {{c4::interacts}} with {{c5::Google APIs}} and {{c6::services}}

## Back Extra


# Note
model: Cloze

## Text
{{c1::Service accounts}} do not represent a {{c2::human user}}, and is managed by {{c3::Identity and Access Management (IAM)}}

## Back Extra


# Note
model: Cloze

## Text
To authenticate with a {{c1::user}} account using {{c2::user credentials}}, you can {{c3::login to CLI}} or {{c4::impersonate service account}}

## Back Extra


# Note
model: Cloze

## Text
To authenticate with a {{c1::user}} account using {{c2::gcloud CLI}}, you can setup {{c3::Application Default Credential (ADC)}} or {{c4::generate access token}}.

## Back Extra


# Note
model: Cloze

## Text
To authenticate with a {{c1::service}} account using {{c2::not recommended}} processes, you can use {{c3::default service account}} or {{c4::service account key}}.

## Back Extra


# Note
model: Cloze

## Text
To authenticate with a {{c1::service}} account using {{c2::recommended}} processes, you can {{c3::attach a user-managed service}} and {{c4::use Application Default Credentials}}, or to {{c5::impersonate service account}}, which can allow you to temporarily {{c6::grant more privileges}}.

## Back Extra


# Note
model: Cloze

## Text
An {{c1::API key}} is a Token that associates a {{c2::request}} with a {{c3::project}} for {{c4::billing}} and {{c5::quota}} purposes.

## Back Extra


# Note
model: Cloze

## Text
Because {{c1::API keys}} do not {{c2::identify}} the caller, they're often used for accessing {{c3::public}} data or resources

## Back Extra


# Note

## Front
How much API keys can you create in your project?

## Back
300

# Note
model: Cloze

## Text
To use API Keys in Google's REST interface, POST https://vision.googleapis.com/v1/images:annotate{{c1::?key=API_KEY}}

## Back Extra


# Note
model: Cloze

## Text
The best practices for API keys are to {{c1::add API key restrictions}}, {{c2::delete unneeded API keys}} and {{c3::regularly refresh API key}}.

## Back Extra


# Note
model: Cloze

## Text
We add {{c1::API key restrictions}} to our key to {{c2::reduce impact}} of a {{c3::compromised API key}}.

## Back Extra


# Note
model: Cloze

## Text
We delete {{c1::unneeded}} API keys to {{c2::minimize attack potential}}.

## Back Extra


# Note

## Front
What is ADC?

## Back
Application Default Credentials

# Note
model: Cloze

## Text
{{c1::ADC}} is a strategy used by Google authentication libraries to automatically {{c2::find credentials available}} to {{c3::Cloud Client Libraries}}.

## Back Extra


# Note
model: Cloze

## Text
If you have set up the necessary {{c1::IAM permissions}}, you should use {{c2::gcloud credentials}}.

## Back Extra


# Note
model: Cloze

## Text
The {{c1::easiest}} and {{c2::most secure}} way to provide credentials to REST in a {{c3::local development}} environment is {{c4::gcloud credentials}}.

## Back Extra


# Note
model: Cloze

## Text
The preferred method to authenticate a REST call in {{c1::production}} environment is {{c2::ADC}}.

## Back Extra


# Note
model: Cloze

## Text
You can use the {{c1::gcloud CLI}} to create {{c2::ADC}} credentials.

## Back Extra


# Note
model: Cloze

## Text
If you can't use an {{c1::attached service account}}, or you want to {{c2::test}} whether a {{c3::service account}} has {{c4::sufficient IAM permissions}}, provide credentials by {{c5::impersonating a service account}}.

## Back Extra


# Note
model: Cloze

## Text
The credentials to login to {{c1::google cloud console}} is separate from {{c2::client libraries}}

## Back Extra


