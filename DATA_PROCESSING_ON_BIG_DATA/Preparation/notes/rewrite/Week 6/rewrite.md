# Week 6: Data Engineering - Building Batch Data Pipelines in Google Cloud Part 3

## Introduction to Cloud Data Fusion

Let's start with an introduction to Cloud Data Fusion. Cloud Data Fusion provides a graphical user interface and APIs that increase time efficiency and reduce complexity. It equips business users, developers, and data scientists to quickly and easily build, deploy, and manage data integration pipelines.

Cloud Data Fusion is essentially a graphical no-code tool to build data pipelines. Cloud Data Fusion is used by developers, data scientists, and business analysts alike. For developers, Cloud Data Fusion allows you to cleanse, match, remove duplicates, blend, transform, partition, transfer, standardize, automate, and monitor data.

Data scientists can use Cloud Data Fusion to visually build integration pipelines, test, debug, and deploy applications. Business analysts can run Cloud Data Fusion at scale on Google Cloud, operationalize pipelines, and inspect rich integration metadata.

Cloud Data Fusion offers a number of benefits:
- Integrate with any data - through a rich ecosystem of connectors for a variety of legacy and modern systems, relational databases, file systems, cloud services, object stores, NoSQL, EBCDIC, and more.
- Increase productivity - If you have to constantly move between numerous systems to gather insight, your productivity is significantly reduced. With Cloud Data Fusion, your data from all the different sources can be pooled into a view like in BigQuery, Spanner, or any other Google Cloud technologies, allowing you to be more productive faster.
- Reduce complexity - through a visual interface for building data pipelines, code-free transformations, and reusable pipeline templates.
- Increase flexibility - through support for on-premises and cloud environments, interoperability with the open-source software CDAP.

At a high level, Cloud Data Fusion provides you with a graphical user interface to build data pipelines with no code. You can use existing templates, connectors to Google Cloud, and other Cloud services providers, and an entire library of transformations to help you get your data in the format and quality you want.

Also, you can test and debug the pipeline and follow along with each node as it receives and processes data. As you will see in the next lesson, you can tag pipelines to help organize them more efficiently for your team, and you can use the unified search functionality to quickly find field values or other keywords across your pipelines and schemas.

Lastly, we'll talk about how Cloud Data Fusion tracks the lineage of transformations that happen before and after any given field on your dataset. One of the advantages of Cloud Data Fusion is that it's extensible. This includes the ability to templatize pipelines, create conditional triggers, and manage and templatize plugins. There is a UI widget plugin as well as custom provisioners, custom compute profiles, and the ability to integrate to hubs.

## Components of Cloud Data Fusion

The two major user interface components we will focus our attention on in this course are the Wrangler UI for exploring data sets visually and building pipelines with no code, and the Data Pipeline UI for drawing pipelines right onto a canvas. You can choose from existing templates for common data processing tasks like Cloud Storage to BigQuery.

There are other features of Cloud Data Fusion that you should be aware of too. There's an integrated rules engine where business users can program in their pre-defined checks and transformations, and store them in a single place. Then data engineers can call these rules as part of a rule book or pipeline later.

We mentioned data lineage as part of field metadata earlier. You can use the metadata aggregator to access the lineage of each field in a single UI and analyze other rich metadata about your pipelines and schemas as well. For example, you can create and share a data dictionary for your schemas directly within the tool.

Other features, such as the microservice framework, allow you to build specialized logic for processing data. You can also use the Event Condition Action (ECA) Application to parse any event, trigger conditions, and execute an action based on those conditions.

## Cloud Data Fusion UI

Managing your pipelines is easiest when you have the right tools. We'll now take a high level look at the Cloud Data Fusion UI as you saw in the component overview. Here are some of the key user interface elements that you will encounter when using Data Fusion. Let's look at each of them in turn.

Under Control Center is the section for applications, artifacts and a dataset. Here you could have multiple pipelines associated with a particular application. The Control Center gives you the ability to see everything at a glance and search for what you need, whether it's a particular dataset, pipeline or other artifact, like a data dictionary, for example.

Under the Pipeline section, you have a developer studio. You can preview, export, schedule a job or project. You also have a connector and a function palette and a navigation section.

Under the Wrangler section, you have connections, transforms, data quality, insights and functions.

Under the Integration metadata section, you can search, add tags and properties and see the data lineage for field and data.

The Hub allows you to see all the available plugins, sample use cases and prebuilt pipelines.

Entities include the ability to create pipelines, upload an application, plugin, driver, library and directives.

There are two components in Administration, management and configuration. Under management, you have services and metrics. Under configuration, you have namespace, compute profiles, preferences, system artifacts and the REST client.

## Build a pipeline

Now that we've looked at the components in the UI, we'll discuss the process of building a data pipeline. A pipeline is represented visually as a series of stages arranged in a graph. These graphs are called DAGs, or Directed Acyclic Graphs, because they flow from one direction to another, and they cannot feed into themselves. Acyclic simply means not a circle.

Each stage is a node. And as you can see here, nodes can be of a different type, you may start with a node that pulls data from Cloud Storage, then passes it on to a node that parses a CSV. The next node takes multiple nodes, has an input and joins them together before passing the joined data to two separate data sink nodes.

As you saw in our previous example, you can have multiple nodes fork out from a single parent node. This is useful because you may want to kick off another data processing work stream that should not be blocked by any processing on a separate series of nodes. You can combine data from two or more nodes into a single output in a sink.

In Cloud Data Fusion, the studio is the user interface where you author and create new pipelines. The area where you create nodes and chain them together in your pipeline is your canvas. If you have many nodes in a pipeline, the canvas can get visually cluttered, so use the mini map to help navigate around a huge pipeline quickly. You can interact with the canvas and add objects by using the Canvas control panel.

When you're ready to save and run the entire pipeline, you can do so with the pipeline actions toolbar at the top. Don't forget to give your pipeline a name and description, as well as make use of the many pre-existing templates and plugins so you don't have to write your pipeline from scratch.

Here, we've used a template for data pipeline batch, which gives us the three nodes you see here to move data from a Cloud Storage file, process it in a wrangler and output it to BigQuery. You should make use of preview mode before you deploy and run your pipeline in production to ensure everything will run properly. While a pipeline is in preview, you can click on each node and see any sample data or errors that you will need to correct before deploying.

After deployment, you can monitor the health of your pipeline and collect key summary stats of each execution. Here, we're ingesting data from Twitter and Google Cloud and parsing each tweet before loading them into a variety of data sinks. If you have multiple pipelines, it's recommended that you make liberal use of the tags feature to help you quickly find and organize each pipeline for your organization.

You can view the start time, the duration of the pipeline run and the overall summary across runs for each pipeline. You can quickly see the data throughput at each node in the pipeline simply by interacting with the node. Note the compute profile used in the Cloud. Clicking on a node gives you detail on the inputs, outputs and errors for that given node.

Here, we are integrating with the speech-to-text API to process audio files into searchable text. You can track the individual health of each node and get useful metrics like records out per second, average processing time, and max processing time, which can alert you to any anomalies in your pipeline.

You can set your pipelines to run automatically at certain intervals. If your pipeline normally takes a long time to process the entire dataset, you can also specify a maximum number of concurrent runs to help avoid processing data unnecessarily. Keep in mind that Cloud Data Fusion is designed for batch data pipelines. We'll dive into streaming data pipelines in future modules.

One of the big features of Cloud Data Fusion is the ability to track the lineage of a given field value. Let's take this example of a campaign field for DoubleClick dataset and track every transform operation that happened before and after this field. Here, you can see the lineage of operations that are applied to the campaign field between the campaign dataset and the DoubleClick dataset. Note the time this field was last changed by a pipeline run and each of the input fields and descriptions that interacted with the field as part of processing it between datasets.

Imagine the use cases if you've inherited a set of analytical reports and you want to walk back upstream all of the logic that went into a certain field. Well, now you can.

## Explore data using wrangler

We've discussed the core components, tools and processes of building data pipelines. Now we'll look at using Wrangler to explore the data set. So far in the course, we have focused on building new pipelines for our data sets.

That presumes we know what the data is and what transformations need to be made already. Oftentimes, a new data set still needs to be explored and analyzed for insights. The Wrangler UI is the cloud data fusion environment for exploring new data sets visually for insights.

Here, you can inspect the data set and build a series of transformation steps called directives to stitch together a pipeline. Here's what the Wrangler UI looks like. Starting from the left, you have your connections to existing data sets.

You can add new connections to a variety of data sources like Google Cloud Storage, BigQuery or even other cloud providers. Once you specify your connection, you can browse all of the files and tables in that source. Here, you can see a cloud storage bucket of demo data sets and all the CSV files of customer complaints.

Once you've found an example data set like customers.csv here, you can explore the rows and columns visually and view sample insights. As you explore the data, you might want to create new calculated fields, drop columns, filter rows or otherwise wrangle the data.

You can do so using the Wrangler UI by adding new directives to form a data transformation recipe. When you're happy with your transformations, you can create a pipeline that you can then run at regular intervals.
