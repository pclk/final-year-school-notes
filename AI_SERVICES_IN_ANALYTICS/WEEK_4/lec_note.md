Learning Outcomes
• Describe the Turing test
• Explain the Chinese Room Argument leading to the concepts of
Strong and Weak AI
• Describe alternatives to Turing test that are of relevance to chatbots
• Describe the basic concepts of agents, intents and entities in
Dialogflow
• Describe the intermediate concepts of integration, context and
fulfillment in Dialogflow
• Explain the best practices in general agent design
Introduction
• Alan Turing
• Father of computer science and AI
• WWII work unknown until docs
declassified in 1970s
• Turing machines – mathematical
problem solving machines – led to
modern day computers
• Side fact: Hungarian mathematician
Von Neumann established the
fundamental architecture of modern
computers
• 1950, Turing test – a measure if a
computer is intelligent or not

ELIZA chatbot
• ELIZA
• First example of chatbot
• Written by German American MIT
computer scientist Joseph Weizenbaum in
the mid-1960s
• Named after Eliza Doolittle (based on
George Bernard Shaw’s play Pygmalion)
• Takes the role of a psychiatrist talking to a
patient, asking open-ended questions
• Does not understand the dialogue
• Never intended for use in Turing test
• Ancestor of many non-AI chatbots
• Legacy lives on for many years; Loebner
Prize competition
Controversy with Turing test
• Chinese Room Argument
• 1980, by American philosopher John Searle
• Searle imagining himself following a symbol-processing
manual written in English. The English speaker (Searle)
sitting in the room follows English instructions for
manipulating Chinese symbols, whereas a computer
“follows” (in some sense) a program written in a
computing language. The human produces the
appearance of understanding Chinese by following the
symbol manipulating instructions, but does not thereby
come to understand Chinese. Since a computer just
does what the human does – manipulate symbols on
the basis of their syntax alone – no computer, merely
by following a program, comes to genuinely understand
Chinese.
• As far as the Turing test is concerned, the computer is
doing something indistinguishable from human
behavior => passes Turing test
• Narrow conclusion – programming a computer may
make it appear to understand language but could not
produce real understanding => Turing Test is
inadequate
• Broad conclusion - human minds do not work like
computers
• Two interpretations even if a computer passes
Turing test
• Computer actually understands the dialogue like a
human
• Computer does not understand, but can simulate
understanding
• Leading to
• Strong AI – goal of building programs that really do
have understanding (consciousness, etc.) in the way
that people do
• Weak AI – the weaker goal, of building programs that
demonstrate the same capability but without any claim
that they actually possess these attributes
Alternatives to Turing test
• Winograd Schemas (Terry Winograd,
professor of computer science at Stanford)
• 1a: The trophy doesn’t fit into the brown
suitcase because it is too small
• 1b: The trophy doesn’t fit into the brown
suitcase because it is too large.
• Question: What is too small/large?
• LAMBADA (LAnguage Modeling Broadened to
Account for Discourse Aspects) benchmark
• The Wozniak (Apple co-founder) “coffee test”
– go into an average American house and
figure out how to make coffee, including
identifying the coffee machine, figuring out
what the buttons do, finding the coffee in the
cabinet, etc.
• Goertzel Tests
• Story understanding – reading a story, or
watching it on video, and then answering
questions about what happened (including
questions at various levels of abstraction)
• Passing the elementary school reading
curriculum (which involves reading and an-
swering questions about some picture books as
well as purely textual ones)
• Graduating (virtual-world or robotic) preschool;
depends a lot on a country and a specific
preschool – requirements differ widely and the
cognitive abilities to pass this test
• Employment Test (Nilsson)
Dialogflow
• Chatbots can be divided into two primary types
• Goal oriented; focused on tasks
• Conversational; provide responses that are almost
entirely artificially generated, or the intention is
that they be perceived that way
• Dialogflow is fundamentally a rules-based chatbot
that uses natural language to construct, not rules,
but intents (i.e. intended actions of the user)
• The chatbot can identify parts of conversation that
may match intended actions
• After the chatbot identifies the intended action, it
can probe for more information relevant to that
action until it has enough to proceed with the task
• Dialogflow provides a web user interface called the
Dialogflow Console for you to create, build, and
test agents
• The Dialogflow Console is different from the
Google Cloud Platform (GCP) Console; Dialogflow
Console is used to manage Dialogflow agents,
while the GCP Console is used to manage GCP-
specific Dialogflow settings (for example, billing)
and other GCP resources
• In most cases you should use the Dialogflow
Console to build agents, but you can also use the
Dialogflow API to build agents for advanced
scenarios
Dialogflow
Trial Edition
A free edition that provides most of
the features of the standard ES
agent type. It offers limited quota
and support by community and e-
mail. This edition is suitable to
experiment with Dialogflow.
Dialogflow ES
Edition
The Dialogflow Essentials (ES)
Edition is a pay-as-you-go edition
that provides the standard ES
agent type. The Essentials Edition
offers production-ready quotas and
support from Google Cloud
support.
Dialogflow
CX Edition
The Dialogflow Customer
Experience (CX) Edition is a pay-as-
you-go edition that provides the
advanced CX agent type. The CX
Edition offers production-ready
quotas and support from Google
Cloud support.
Editions Quotas & Limits
Agent
• A virtual agent that handles concurrent
conversations with your end-users.
• A natural language understanding (NLU)
module that understands the nuances of
human language. Dialogflow translates end-
user text or audio during a conversation to
structured data that your apps and services
can understand. You design and build a
Dialogflow agent to handle the types of
conversations required for your system.
• Similar to a human call center agent. You train
them both to handle expected conversation
scenarios, and your training does not need to
be overly explicit.
• Dialogflow agents use machine learning
algorithms to understand end-user
expressions, match them to intents, and
extract structured data. An agent learns from
training phrases that you provide and the
language models built into Dialogflow. Based
on this data, it builds a model for making
decisions about which intent should be
matched to an end-user expression.
• By default, Dialogflow updates your agent's
machine learning model every time you make
changes to intents and entities, import or
restore an agent, or train your agent.
Intent
• An intent categorizes an end-user's intention for one
conversation turn.
• For each agent, you define many intents, where your
combined intents can handle a complete conversation.
• When an end-user writes or says something (i.e. an end-user
expression), Dialogflow matches the end-user expression to
the best intent in your agent. Matching an intent is also
known as intent classification.
• In short, an intent represents a mapping between what a
user says and what action should be taken by your agent.
• Dialogflow uses training phrases as examples for a machine
learning model to match end-user expressions to intents.
• The model checks the expression against every intent in the
agent, gives every intent a score, and the highest scoring
intent is matched. If the highest scoring intent has a very low
score, the fallback intent is matched.
Intent
Basic intent
• Training phases - Example phrases for what end-users might
say. Dialogflow's built-in machine learning expands on your
list with other similar phrases. When an end-user expression
resembles one of these phrases, Dialogflow matches the
intent.
• Action - You can define an action for each intent. When an
intent is matched, Dialogflow provides the action to your
system, and you can use the action to trigger certain actions
defined in your system.
• Parameters - When an intent is matched at runtime,
Dialogflow provides the extracted values from the end-user
expression as parameters. Each parameter has a type, called
the entity type, which dictates exactly how the data is
extracted.
• Responses – You define text, speech, or visual responses to
return to the end-user. These may provide the end-user with
answers, ask the end-user for more information, or terminate
the conversation.
Complex Intent
• Context - Dialogflow contexts are similar to natural language
context. If a person says to you "they are orange", you need
context to understand what the person is referring to.
• Events - With events, you can invoke an intent based on
something that has happened, instead of what an end-user
communicates. Example: Default Welcome Intent.
Intent
Training phrases
• You should create at least 10-20 (depending on
complexity of intent) training phrases, so your
agent can recognize a variety of end-user
expressions.
• Annotation – control how data is extracted by
annotating parts of your training phrases and
configuring the associated parameters
Intent
Slot filling
• When an intent is matched at runtime, the
Dialogflow agent continues collecting
information from the end-user until all data
for every required parameters have been
provided
• When building an agent, you provide prompts
that the agent will use to get parameter data
from the end-user. You can also provide
prompt variations, so the agent doesn't
always ask the same question.
• What date?
• What date would you like?
• Which date is good for you?
• Ordering parameters
Intent
Responses
• Intents have a built-in response handler that can
return responses after the intent is matched.
• Use parameter references in these responses to
make them more dynamic. This is helpful for
recapping information provided by the end-user.
Example: "Okay, I booked a room for you on
$date".
• Agents typically use a combination of static and
dynamic responses. If you define more than one
response variation, your agent will select a
response at random. You should add several
response variations to make your agent more
conversational.
• Rich responses are supported using custom
payload when integrating with Dialogflow
Messenger (Beta)
Entity
• Dictates exactly how data from an end-user expression is
extracted. Any important data you want to get from a user's
request will have a corresponding entity.
• Each entity entry provides a set of words or phrases that are
considered equivalent. For example, if size is an entity type,
you could define three entity entries: S, M, L
• Reference value and synonyms – Some entity entries have
multiple words or phrases that are considered equivalent;
provide one reference value and one or more synonyms.
• 3 types
• System entity – built into Dialogflow
• Date and time: @sys.time @sys.date
• Numbers: @sys.number
• Geography: @sys.address, @sys.airport
• Contacts: @sys.email, @sys.phone-number
• Names: @sys.last-name
• Developer entity – define your own
• Create via console, API, CSV
• Only for those needed for actionable data
• User/Session – created via API for transient match
Context
• Dialogflow contexts are similar to natural language context. If
a person says to you "they are orange", you need context in
order to understand what "they" is referring to.
• Using contexts, you can control the flow of a conversation.
• Process Flow
1. The end-user asks for information about their checking account.
2. Dialogflow matches this end-user expression to the CheckingInfo
intent. This intent has a checking output context, so that context
becomes active.
3. The agent asks the end-user for the type of information they
want about their checking account.
4. The end-user responds with "my balance".
5. Dialogflow matches this end-user expression to the
CheckingBalance intent. This intent has a checking input context,
which needs to be active to match this intent. A similar
SavingsBalance intent may also exist for matching the same end-
user expression when a savings context is active.
6. After your system performs the necessary database queries, the
agent responds with the checking account balance
Context
• Input and output contexts are applied to
intents. They work together to control
conversation flow:
• Output contexts control active contexts.
When an intent is matched, any configured
output contexts for that intent become
active.
• Input contexts control intent matching.
While contexts are active, Dialogflow is
more likely to match intents that are
configured with input contexts that are a
subset of currently active contexts.
• With contexts, you can:
• Control the order of intent matching.
• Create context-specific intents with the
same training phrases.
pet-init
What kind of pet you like?
I like dogs
What to know about dogs?
What they look like?
Here is a pic of a dog
Hi, looking for pet
pet-select-dogs
dog-show?
cat-show?
dogs
Context
• Follow-up Intent
• You can use follow-up intents to
automatically set contexts for pairs of
intents.
• A follow-up intent is a child of its
associated parent intent. When you
create a follow-up intent, an output
context is automatically added to the
parent intent and an input context of the
same name is added to the follow-up
intent.
• A follow-up intent is only matched when
the parent intent is matched in the
previous conversational turn. You can
also create multiple levels of nested
follow-up intents.
Fulfillment
• By default, your agent responds to a matched intent with a
static response. You can provide a more dynamic response by
using fulfillment through an integration.
• When an intent with fulfillment enabled is matched,
Dialogflow sends a request to your webhook service with
information about the matched intent. Your system can
perform any required actions and respond to Dialogflow.
• Process flow
1. The end-user types or speaks an expression.
2. Dialogflow matches the end-user expression to an intent and
extracts parameters.
3. Dialogflow sends a webhook request message to your webhook
service. This JSON message contains information about the
matched intent, the action, the parameters, values of entities and
the response defined for the intent.
4. Your webhook service performs actions and trigger business
logic as needed, like database queries or external API calls.
5. Your webhook service sends a response message to Dialogflow.
6. Dialogflow sends the response to the end-user.
7. The end-user sees or hears the response.
Agent Design Best Practices
• Reference
• Consider the overall objective of your agent:
• What is your business trying to achieve?
• What will your users expect from your agent?
• How often will users interact with your agent?
• Consider how users will access your agent – Review the
platforms supported by Dialogflow before creating content.
When you choose platforms to support, prepare your content
accordingly. Some of Dialogflow's platform integrations
support rich messages that can include elements like images,
links, and suggestion chips.
• Build agents iteratively
• When building your agent, it's best to define your entities
prior to adding training phrases to your intents. The console
will automatically annotate your training phrases with
existing entities. However, if you create entities after training
phrases, you can manually annotate the phrases.
Categories
• Greetings and goodbyes
• ML and training
• Intent naming
• Conversation repair
• Persona
• Designing for voice
• Protection of consumer privacy
• Implementing Dialogflow APIs
• Testing
