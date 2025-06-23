Dialogflow creates the following two default intents as a part of the agent:

Default welcome intent greets your users.
Default fallback intent catches all the questions that your bot doesn't understand.


To test the agent, type "Hi" where it says Try it now. The agent should respond with the default greeting defined in the default welcome intent. It should say, "Greetings! How can I assist?" You can modify the response.


Now, if you enter "set an appointment," the agent doesn't know what to do, so it initiates the default fallback intent. That's because you haven't created any intent to catch that particular question!

To create the intent, click on Intents > Create Intent. Enter "Schedule Appointment" as the Intent name.

Click Training phrases and enter the following phrases.

As you enter the phrases, you'll see time and date are automatically identified as system entities @sys.date and @sys.time.

Scroll to Responses, enter "You are all set. See you then!" as a response or you could make it more interesting and enter "You are all set for $date at $time. See you then!" (Dollar($) sign here helps you access the entity values.) Click Add Responses.

Slot filling
Now, test "set an appointment." That's not very specific and you haven't handled that scenario, so it should be handled by the default fallback intent. To support that, you can use something called slot filling.

Slot filling allows you to design a conversation flow for parameter-value collection in a single intent. It's useful when an action can't be completed without a specific set of parameter values.
Click Actions and parameters. Make the entities as required, and Dialogflow asks for date and time before it responds.
Dialogflow provides many types of integration for your chatbot. Take a look at a sample web user interface for the chatbot.

Click Integrations in the Dialogflow console.

Enable Web Demo.
You'll notice training phrases like "Set an appointment for 4 PM tomorrow," where Date and Time are automatically extracted as @sys.date and @sys.time. Feel free to add more training phrases to see how Dialogflow automatically extracts the system entities.
As you saw, system entities allow agents to extract information about a wide range of concepts without any additional configuration. Data like address, emails, currency, and phone numbers are some of the common examples of system entities. Find more, see System Entities.
You can select the Allow automated expansion checkbox to automatically add more entities. Automated expansion of developer entities allows an agent to recognize values that haven't been explicitly listed in the entity. If a user's request includes an item that isn't listed in the entity, automatic expansion recognizes the undefined item as a parameter in the entity. The agent sees the user's request is similar to the examples provided, so it can derive what the item is in the request.
If the user only provided one or two pieces of information, then Dialogflow will ask for the leftover information before it acts on the response. That feature is called slot filling.

.original to get unformatted value of parameter
