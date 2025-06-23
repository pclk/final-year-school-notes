Hi, I'm Allison Walther with Confluent.

Let's take a look at some key concepts and semantics in ksqlDB.

While ksqlDB reads and writes from Kafka topics, it can model how it works with that data in different ways.

## Sample Data
We've got some sample data here showing where people are going. Each event has a person's name and their current location. We're going to see how these events can be modeled as either a stream or a table and what the difference is. We have a stream on the left and a table on the right. With only one event, they appear identical.

## Streams
As we add a second event, the stream grows. Streams behave the same as their underlying Kafka topics. They are unbounded and append only, so they grow with each new event.

Tables, on the other hand, store the latest value of a given key. It's important to note that a key is required in order to have a table. The only way that a table grows in size is as new keys are added - a new person in our case. Now, any events arriving with either key will cause an update to our table.

While our stream continues to grow with each new event, a table is a fast way to get the current state of a given key. We can easily tell that Robin is in Ilkley and Allison is in Boulder. But if we wanted to replay the history of their recent movements, the table wouldn't help us. We'd need a stream for that.

We can see that streams and tables are just different ways of working with events, each with its own strengths. In our example, we are using the same events for both, but a more realistic example might have inventory as a table and orders as a stream. Together, streams and tables are a powerful combination for building event streaming applications.
