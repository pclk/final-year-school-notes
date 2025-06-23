
1i, I'm weiheng. right now, we're going to analyse Data Center Temperature data through Pub/Sub, Big query and Looker studio.

I'm uploading the data center csv file on our bucket

I change the bucket file path to match our csv file.

I change the input file path, as well as the datetime format.

I open a new SSH connection, so that I can my pub/sub delivering while i build the dataflow in Java.

I open the AverageSpeeds file, change the average interval from 60 to 5 minutes, change the table name from demo average speeds to demo average temperature, change the field names, and remove unneeded fields.

I open the LaneInfo file, change the ordering of the enum field to suit my data, and change the getSensorId to directly retrieve the sensorId field.

I run the script which compiles and executes the AverageSpeeds java.

I create the Bigquery dataset demos.

I check my dataflow to make sure everything is working.

I begin to query my average_temperature table

Take a look at all the fields.

This is the latest timestamp of the table.

This is the subset of rows that existed 1 minute ago.

There is now a rate of change for temperature with this query.

We can see how many times a sensor indicated temperature that was above our threshold. 

I look at monitoring

I create an alert policy for system lag in dataflow, and set it to notify my email address.

I create a dashboard that shows system lag over time.

Let's open looker studio, going past the initialization messages, we select our big query table and start building our report

I add a pie chart, that symbolizes the ratio of the quantity of sensor data from all 3 locations. It looks like they're very even.

I add a line chart, that shows the average temperature of the 3 locations over time. 

It looks like they're very stable.

Lastly, I add a bar chart, that shows the average, minimum and maximum temperature over the entire bigquery table. To do this, I need to add a custom query from the bigquery table.

Overall, the difference between the average, minimum and maximum do not deviate much. This means that the data centers are stable and properly managed.

When sensors are faulty, the pie chart will show an imbalance. When temperature is high, the line chart and barchart will show a higher value. The difference between the line and bar chart is that the bar will give an aggregated view, while the line will show change over time. As a result, this dashboard is effective in monitoring data center temperature.


