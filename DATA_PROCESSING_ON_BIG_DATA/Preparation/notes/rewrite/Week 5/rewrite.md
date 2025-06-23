# Week 5: Data Engineering - Building Batch Data Pipelines in Google Cloud Part 2

## Introduction to Dataflow

Let's start by exploring Dataflow in more detail. The reason Dataflow is the preferred way to do data processing on Google Cloud is that Dataflow is serverless. You don't have to manage clusters at all. Unlike with Dataproc, the auto scaling in Dataflow scales step by step, it's very fine-grained.

Plus, as we will see in the next course, Dataflow allows you to use the same code for both batch and stream. This is becoming increasingly important. When building a new data processing pipeline, we recommend that you use Dataflow.

If, on the other hand, you have existing pipelines written using Hadoop technologies, it may not be worth it to rewrite everything, migrate it over to Google Cloud using Dataproc and then modernize it as necessary. As a data engineer, we recommend that you learn both Dataflow and Dataproc and make the choice based on what's best for a specific use case. If the project has existing Hadoop or Spark dependencies, use Dataproc.

Please keep in mind that there are many subjective issues when making this decision, and that no simple guide will fit every use case. Sometimes the production team might be much more comfortable with a DevOps approach where they provision machines than with a serverless approach. In that case too, you might pick Dataproc. If you don't care about streaming and your primary goal is to move existing workloads, then Dataproc would be fine.

Dataflow, however, is our recommended approach for building pipelines. Dataflow provides a serverless way to execute pipelines on batch and streaming data, it's scalable to process more data, Dataflow will scale out to more machines, it will do this transparently. The stream processing capability also makes it low latency, you can process the data as it comes in.

This ability to process batch and stream with the same code is rather unique. For a long time, batch programming and data processing used to be two very separate and different things. Batch programming dates to the 1940s and the early days of computing where it was realized that you can think of two separate concepts, code and data. Use code to process data. Of course, both of these were on punch cards. So that's what you were processing, a box of punch cards called a batch. It was a job that started and ended when the data was fully processed.

Stream processing, on the other hand, is more fluid. It arose in the 1970s with the idea of data processing being something that is ongoing. The idea is that data keeps coming in and you process the data, the processing itself tended to be done in micro batches.

The genius of beam is that it provides abstractions that unify traditional batch programming concepts and traditional data processing concepts. Unifying programming and processing is a big innovation in data engineering. The four main concepts are P transforms, P collections, pipelines and pipeline runners.

A pipeline identifies the data to be processed and the actions to be taken on the data. The data is held in a distributed data abstraction called a P collection. The P collection is immutable. Any change that happens in a pipeline ingests one P collection and creates a new one. It does not change the incoming P collection.

The action or code is contained in an abstraction called a P transform. P transform handles input, transformation, and output of the data. The data in the P collection is passed along a graph from one P transform to another. Pipeline runners are analogous to container hosts such as Google Kubernetes Engine. The identical pipeline can be run on a local computer, data center VM, or on a service such as Dataflow in the Cloud. The only difference is scale and access to platform-specific services. The services the runner uses to execute the code is called a backend system.

Immutable data is one of the key differences between batch programming and data processing. Immutable data where each transform results in a new copy means there is no need to coordinate access control or sharing of the original ingest data. So it enables or at least simplifies distributed processing.

The shape of a pipeline is not actually just a singular linear progression but rather a directed graph with branches and aggregations. For historical reasons, we refer to it as a pipeline, but a data graph or Dataflow might be a more accurate description.

A P collection represents both streaming data and batch data. There is no size limit to a P collection. Streaming data is an unbounded P collection that doesn't end. Each element inside a P collection can be individually accessed and processed. This is how distributed processing of the P collection is implemented. So you define the pipeline and the transforms on the P collection and the runner handles implementing the transformations on each element distributing the work as needed for scale and with available resources.

Once an element is created in a P collection, it is immutable, so it can never be changed or deleted. Elements represent different data types. In traditional programs, a data type is stored in memory with a format that favors processing. Integers in memory are different from characters which are different from strings and compound data types.

In a P collection, all data types are stored in a serialized state as byte strings. This way, there is no need to serialize data prior to network transfer and deserialize it when it is received. Instead, the data moves through the system in a serialized state and is only deserialized when necessary for the actions of a P transform.


## Why customers value Dataflow

So we've discussed what Dataflow is, but why do data engineers value Dataflow over other alternatives for data processing? To understand that, it helps to understand a bit about how Dataflow works.

Dataflow provides an efficient execution mechanism for Apache Beam. The Beam pipeline specifies what has to be done. The Dataflow service chooses how to run the pipeline. The pipeline typically consists of reading data from one or more sources, applying processing to the data and writing it to one or more sinks.

In order to execute the pipeline, the Dataflow service first optimizes the graph by, for example, fusing transforms together. It then breaks the jobs into units of work and schedules them to various workers. One of the great things about Dataflow is that the optimization is always ongoing. Units of work are continually rebalanced.

Resources, both compute and storage, are deployed on demand and on a per job basis. Resources are torn down at the end of a job stage or on downscaling. Work scheduled on a resource is guaranteed to be processed. Work can be dynamically rebalanced across resources. This provides fault tolerance.

The watermarking handles late arrivals of data and comes with restarts, monitoring and logging. No more waiting for other jobs to finish. No more preemptive scheduling. Dataflow provides a reliable, serverless, job-specific way to process your data.

To summarize, the advantages of Dataflow are:
First, Dataflow is fully managed and auto-configured. Just deploy your pipeline.

Second, Dataflow doesn't just execute the Apache Beam transforms as is. It optimizes the graph, fusing operations, as we see with C and D. Also, it doesn't wait for a previous step to finish before starting a new step. We see this with A and the Group By Key.

Third, autoscaling happens step-by-step in the middle of a job. As the job needs more resources, it receives more resources. You don't have to manually scale resources to match job needs. If some machines have finished their tasks and others are still going on, the tasks queued up for the busy ones are rebalanced out to the idle machines. This way, the overall job finishes faster.

Dynamic work rebalancing in mid-job removes the need to spend operational or analyst resource time hunting down hotkeys. All this happens while maintaining strong streaming semantics. Aggregations, like sums and counts, are correct even if the input source sends duplicate records. Dataflow is able to handle late arriving records.

Finally, Dataflow functions as the glue that ties together many of the services on Google Cloud. Do you need to read from BigQuery and write to BigTable? Use Dataflow. Do you need to read from Pub/Sub and write to Cloud SQL? Use Dataflow.

## Building Dataflow pipelines in code

Let's look in greater detail at an example Dataflow pipeline. Here's how to construct a simple pipeline where you have an input PCollection and pass it through three PTransforms and get an output PCollection. The syntax is shown in Python.

You have the input, the pipe symbol, the first PTransform, the pipe symbol, the second PTransform, et cetera. The pipe operator essentially applies the transform to the input PCollection and sends out an output PCollection. The first three times, we don't give the output a name, simply pass it on the next step. The output of PTransform_3, though, we save into a PCollection variable named PCollection_out.

In Java, it is the same thing, except that, instead of the pipe symbol, we use the apply method. If you want to do branching, just send the same PCollection through two different transforms. Give the output PCollection variable in each case a name. Then you can use it in the remainder of your program.

Here, for example, we take the PCollection_in and pass the collection first through both PTransform_1 then through PTransform_2. The result of the first case, we store as PCollection_out_1. In the second case, we store it as PCollection_out_2.

What we showed you so far is the middle part of a pipeline. You already had a PCollection, and you applied a bunch of transforms, and you end up with a PCollection, but where does the pipeline start? How do you get the first PCollection? You get it from a source. What does a pipeline do with the final PCollection? Typically, it writes out to a sink. That's what we are showing here.

This is Python. We create a PCollection by taking the pipeline object P and passing it over a text file in cloud storage. That's the read from text line. Then, we apply the PTransform called FlatMap to the lines read from the text file. What FlatMap does is that it applies a function to each row of the input and concatenates all the outputs. When the function is applied to a row, it may return zero or more elements that go to the output PCollection.

The function in this case is the function called count_words. It takes a line of text and returns an integer. The output PCollection then consists of a set of integers. These integers are written to a text file in cloud storage. Because the pipeline was created in a with clause and because this is not a streaming pipeline, exiting the with clause automatically stops the pipeline.

Once you have written the pipeline, it's time to run it. Executing the Python program on the previous slide will run the program. By default, the program is run using the default runner, which runs on the same machine where the Python program was executed.

When you create the pipeline, you can pass in a set of options. One of these options is the runner. Specify that as Dataflow to have the pipeline run on Google Cloud. This example contains hard-coded variables, which in most cases is not a preferred practice for programming at scale. Of course, normally you will set up command-line parameters to transparently switch between local and cloud.

Simply running main runs the pipeline locally. To run on cloud, specify cloud parameters.

## Key considerations with designing pipelines

To design pipelines, you need to know how each step works on the individual data elements contained inside of a PCollection.

Let's start with the input and outputs of the pipeline. First, we set up our Beam pipeline with beam.Pipeline and pass through any options. Here, we'll call the pipeline P. Now it's time to get some data as input.

If we wanted to read a series of CSV files in Cloud Storage, we could use beam.io.ReadFromText and simply parse in the Cloud Storage bucket and file name. Note the use of an asterisk wildcard can handle multiple files.

If we wanted to read instead from a Pub/Sub topic, you would still use beam.io, but instead it's ReadStringsFromPubSub, and you'd have to parse in the topic name.

What about if you wanted to read in data that's already in BigQuery? Here's how that would look. You'd prepare your SQL query and specify BigQuery as your input source and then parse in the query and source as a read function to Dataflow.

These are just a few of the data sources from which Dataflow can read. But now what about writing to sinks?

Take the BigQuery example but as a data sink this time. With Dataflow, you can write to a BigQuery table, as you can see here. First, you establish the reference to the BigQuery table with what BigQuery expects: your project ID, dataset ID and table name. Then you use beam.io.WriteToBigQuery as a sink to your pipeline.

Note that we are using the normal BigQuery options here for write_disposition. Here, we're truncating the table if it exists, meaning to drop data rows. If the table doesn't exist, we can create it if needed.

Naturally, this is a batch pipeline if we're truncating the table with each load.

You can also create a PCollection in memory without reading from a particular source. Why might you do this? If you have a small dataset, like a lookup table or a hard-coded list, you could create the PCollection yourself, as you can see here. Then we can call a pipeline step on this new PCollection just as if we sourced it from somewhere else.

## Transforming data with PTransforms

Now that we have looked at how to get the data in, let's look at how we transform each data element in the PCollection with PTransforms.

The first step of any map-reduce process is the map phase, where you're doing something in parallel.

In the word length example, there is one length output for each word input, so the word "dog" would map to three for length.

In the bottom graph example, the function my_grep returns each instance of the term it's searching for in the line. There may be multiple instances of the term in a single line in a one-to-many relationship.

In this case, you may want my_grep to return the next instance each time it's called, which is why the function has been implemented with a generator using yields. The yield command has the effect of preserving the state of the function so that the next time it's called, it can continue from where it left off.

FlatMap has the effect of iterating over one-to-many relationships. The map example returns a key-value pair. In Python, this is simply a two-tuple for each word. The FlatMap example yields the line only for lines that contain the search term.

ParDo is a common intermediate step in a pipeline. You might use it to extract certain fields from a set of raw input records or convert raw input into a different format. You might also use ParDo to convert process data into an output format, like table rows for BigQuery or strings for printing.

You can use ParDo to consider each element in a PCollection and either output that element to a new collection or discard it. If your input PCollection contains elements that are of a different type or format than you want, you can use ParDo to perform a conversion on each element and output the result to a new PCollection.

If you have a PCollection of records with multiple fields, for example, you can use a ParDo to parse out just the fields you want to consider into a new PCollection. You can use ParDo to perform simple or complex computations on every element or certain elements of a PCollection and output the results as a new PCollection.

When you apply a ParDo transform, you need to provide code in the form of a DoFn object. A DoFn is a Beam SDK class that defines a distributed processing function. Your DoFn code must be fully serializable, idempotent and thread-safe.

In this example, we're just counting the number of words in a line and returning the length of the line. Transformations are always going to work on one element at a time here.

Here we have an example from Python which can return multiple variables. In this example, we have below and above, some cutoff in our data elements and return two different types, below and above, two different variables by referencing these properties of the results.
