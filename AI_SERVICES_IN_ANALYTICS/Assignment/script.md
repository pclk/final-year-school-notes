# VOC
We start by initializing and making the directory paths, as well as annotations.

Annotations are lists which represent the desired csv format by google.

We also have an annotation limit to retrieve the desired amount of 300 annotations.

Looking at the main function which is process underscore files, we are checking our file issues. This would check that the xml and jpg pairs exist, contains annotations, contains annotations for both Persons and Labels, make sure that the jpg file has a resolution of 640 by 640, and the bounding box is valid.

Valid bounding boxes mean that the coordinates follow logic here that make it possible to draw a rectangle. However, if the bounding boxes have a value of 641, which was quite common, we will correct it to 640.

If the file pairs do not meet the criteria, then we call move to unwanted, which moves the file pairs to the categorized unwanted directory.

If the file pairs meet the criteria, then we call process XML, which extracts the annotations, normalize the coordinates, add to self.annotations and increment the self.annotation count. Once the count reaches the specified limit, the loop will break, and no more files will be processed.

But what are the two extra functions below? Create csv basically takes the self.annotations, convert it into a dataframe, and export it as csv. Print summary counts the number of files in each directory and prints the summary message like this.

Now let's move on to the verification of the csv file.

# verification of csv
Scrolling past the file processing, and looking at the CSV Validator, we understand that the checks it does are to ensure that 

the number of columns add up to 10,
image path starts with gs://,
checks the label, 
make sure that the coordinates are at the correct locations, 
check that xmin and ymin is less than xmax and ymax,
make sure that the columns which are supposed to be empty are empty for each row.

It will also check across the dataset that 
the annotation count is 300,
check the class balance,
and prints a report that we can see here.

we are also checking that the uploaded csv matches 1 to 1 with our training directory by using the set() function in python.

Finally, we upload the file,
retrieve an image,
and get the bucket size, which is 5.32 MB.

Let's move on to the training, evaluation and deployment of the object detection model.

# Evaluation
In terms of Training the model, all the instructions are followed.

So why not let's take a look at the model evaluation. The ladder's precision is very good, but not for the person's precision. Probably because ladders has this sort of iconic A-frame image, whereas persons can look very different, with sometimes only the legs being shown in the image.

Looking at the Confidence and Intersection over Union thresholds, let's first define what would happen when we adjust them. 

So, higher confidence means that precision will increase but recall will decrease. This is useful when false positives, like warning all the supervisors, about ladder misuse could be disruptive.

Lower confidence has the opposite behavior of precision, which is useful for monitoring or cases where you don't want false negatives, like if you want to be certain that all ladder activity are captured.

Higher IoU means that the model's predictions must fit the strict positioning information set by high IoU. This would reduce recall, by rejecting correct detections with less accurate bounding box coordinates.

Lower IoU means that the bounding box coordinates are allowed to be more loose, which would increasing recall by accepting partially accurate detections.

Looking at our options, I recommend setting a high confidence threshold for ladder, maybe 0.7 and a medium confidence threshold for person, maybe 0.5. The IoU can be left at 0.5 because it's the industry standard.

To expand on this, specifically to the use case of detecting persons on ladders, this dual-threshold approach makes sense because the safety cost of not identifying a person on a ladder is higher than identifying a ladder without a person. 

# Deployment and testing

To select testing images, we create a test directory, get the set of images that were used in training, then the set of all the images available, and find the complement of these two set, select and move the 10 of the images from complement into the test directory.

Then, we create an endpoint.

We call the test_images function, which deploys the model to the endpoint, gets the images from the test directory, predict with the base64 encoded image, save the results to a json, and undeploy the model immediately.

Then, we draw the predictions with the draw predictions function, which takes the predictions json, filter the 40 predictions per image into the top p or top k, draw the boxes with the label and the confidence, and save it to the predictions_output directory.

From there, we're able to see 3 examples of predicted images.

Across the examples, the images had a high person detection confidence of 91% in clear full-body images, moderate confidence 77% in partial views, and unfortunately, the failure to detect a partially visible foot in the last image.

The ladder detection had some interesting patterns, with confidence scores ranging from 56% to 77%. Notably, the model is more confident with a clearly visible ladder in isolation and flat ground, but lower confidence to ladders in active use. This suggests that occlusion and interaction from the person to the ladder might affect ladder confidence.

# Reflection

If we wanted to expand use case of the model, 

We could generate confidence metadata in the annotation process of the training data, which could help with the confidence distribution of the model.

We could also have additional labels or classification models that check for safe and unsafe practices

Finally, we could consider having a logic to flag scenarios where ladder detection confidence drops during human interaction, which can indicate ladder usage, or potentially unconventional and as a result, unsafe usage patterns.

My learnings are basically that especially in object detection, general accuracy and precision isn't everything. Seeing how the relationship between ladder and person confidence score differ on when the person is climbing the ladder, on the very top of the ladder, obstructing the ladder, can influence the confidence score in a way that is hopefully rather consistent. This can lead to stronger usecases than simply detecting objects, like the different usage or positioning of the ladder.

From a deployment standpoint, when I heard that the requirement was to take down the deployment as fast as I can, the first solution that I had was to build the undeployment as part of the function that tested the image. I know that because of this decision, the cloud computing resource was minimized as much as it could be.

From a practical standpoint, separating the testing pipeline into deployment predictions and the drawing of the boxes with the test image and draw prediction function, really made it much more easier to debug, and even allow for advanced functionality like adjusting top p and k values.

From the Pascal VOC cleaning and CSV validation, I honestly find it quite fun, and the VOC processor was probably one of the most meticulous class that I've built for data wrangling and processing. I made sure all the requirements were met, have good logging, make sure all the unwanted cases were stored neatly for later analysis. 

Plus, to make sure that the requirements of VOC processor was met, I even created a python script, and used pytest to test against the VOC processor, to make sure that all the checking functionality was working properly.

This was a pretty fun project, thanks for watching.
