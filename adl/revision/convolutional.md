# deck: adl convolutional

-Image coordinate system- has origin at -the top left corner-.
> This is following the Cartesian coordinates

In Image coordinate system, when x and y increases, Move 1.-right- and 1.-down-.
> This is following the Cartesian coordinates

-Pixels- are the building blocks of -images-.

-Pixels- can be accessed by specifying their -(x, y) coordinates-. 

What are the two ways color channels can be encoded?
Channel first and channel last.

What is the order of channel first?
channels, height, width

What is the order of channel last?
height, width, channel
---
height, width is a convention to describe image.

What are the two frameworks that uses Channel first?
Theano and Pytorch.
---
Memory: Theano's "Th" sounds like first, and PyTorch is known for being the "first" in many research areas.
Theano is one of the earliest deep learning frameworks. However its no longer actively developed.

What are the two frameworks that uses Channel last?
CNTK and Tensorflow
---
Memory: Tensorflow loses out to Pytorch, so its "last".
CNTK is developed by Microsoft. its no longer developed.

What properties of Image data can we exploit using CNN?
Feature hierarchy, Translation invariance, and Pixel correlation.
---
Memory: FTP: File Transfer Protocol, related to "data".
Feature hierarchy because cat ears are formed with a hierarchy of edges, colors etc.
Translation invariance because it doesn't matter where the cat is.
Pixel correlation because when pixels are close together, its probably an important feature.

Why not feedforward neural network for images?
It treats pixels as independent inputs and ignores spatial relationships between adjacent pixels.
---
This is done by flattening.

What are the 3 hand-tuned filters?
Blurring, Sharpening, and Edge Detection.
---
Memory: Blurring reduces detail, Sharpening enhances detail, Edge Detection filters detail.
Blurring reduces noise and smooths out details, to simplify an image and focus on larger structures.
Sharpening enhances edges and make features more prominent, highlighting important structures in an image.
Edge detection identifies boundaries between objects or regions, and this is useful because they often correspond to object contours and shapes.

Why doesn't CNN need hand-tuned filters?
CNN learns the filter values during training.
---
CNNs can learn optimal filters for a specific task from data, rather than relying on pre-defined filters.

What is a "Filter Size"?
The receptive field
---
This refers to the number of pixels a filter sees at once.

Why do we want a "center" pixel in our filters?
Can be useful for padding

How do we get a center pixel in our filter?
Use odd numbers in height and width of filter size.

Filters are often -square- but they -don't have to be-.

The -Convolutional Operation- is where we take the -image- and the -convolutional filter-, and perform 1.-element-wise multiplication- and 1.-sum- their values.

The Convolutional filter is also known as a -kernel-.

An -RGB- image has -3- channels. 

A -CMYK- image has -4- channels.

The number of -filters- must be the same as the number of -channels-.
> We don't want a single filter for all channels, because it would not allow the model to learn complex features.

A -convolutional layer- is a set of -convolutional filters- that are applied to the input data.

A convolutional operation with an image of -3- channels and -3- convolutional filters will result in -1- output.
> The element-wise multiplicated values are summed to generate 1 output. This enables the filter to learn and detect complex relationships in combination of all the channels.

For a convolutional operation to generate -2- outputs, you need -2- sets of convolutional filters.

The outputs from the convolution layer is called -feature maps-.
> The purpose of Convolutional layers are to extract features from input data.
> Outputs are spatial maps indicating the presence and location of learned features.

What does it mean for something to "Convolve"?
Combine or merge.
---
In the context of CNN, it describes merging the filter's info with the local image patch.

What can the term "Convolutional" Neural Network be roughly translated to?
Neural Networks that spatially combine and merge input features to form more complex features.

With padding, we pad the -outside- of the -input- with -zeros-.
> This is to control the output size of the convolutional layers.
> Padding the filter is computationally inefficient.

Why should we do padding?
Retain spatial dimensions of the feature map.
---
With many convolutional layers, this could lead to a rapid reduction in size, making it difficult to design deeper architecture.

What are the two downsampling techniques?
Increasing stride and Max pooling.

What is stride?
The step size of filter.

What happens when stride is greater?
The feature map will be smaller.

Why would we increase stride?
Reduce computation & Downsampling.
---
Reduce computation: Smaller feature maps mean fewer computation later on.

What is Max Pooling?
Choose max value in each patch.
---
Max pooling is a downsampling technique. It divides the feature map into non-overlapping rectangles, typically 2x2.  Then, in each rectangle, the max value (aka the dominant features), is chosen.
This emphasizes the most important features and avoid over-fitting.

What does not have learnable weights in a CNN?
Pooling layer
---
In other words, pooling layers do not have **learnable parameters** (weights that are adjusted during training).  The *type* of summary statistic (maximum, average, sum, etc.) is a hyperparameter that can be chosen. It simply replaces the feature map with a summary statistic of the nearby outputs.

What's a typical CNN layout?
A series of Convolutional and Pooling layer followed by fully connected dense layers.
---
Conv + Pooling create a hierarchy of features.
Meaning, as early Conv layers learn more complex and abstract features, the deeper Conv layers learn more complex and abstract features.
Pooling helps to generalize these features and reduce dimensionality.
This layout is efficient because Conv + Pooling reduce the dimensions in each layer.

In a typical CNN layout, other than using down-sampling methods like Pooling, what can be used?
Batch Normalization
---
This helps with vanishing/exploding gradients during training, and helps with regularization.
During training, as a deep neural network is trained, the earlier layers change. Thus, the **distribution** inputs to the later layers also change, causing it difficult for the later layers to learn effectively.
Batch Normalization combats this by normalizing the distribution of data so that it doesn't change.

What type of layer learns global patterns?
Traditional Fully Connected layers 
---
It looks at the entire input to find patterns.
Conv layers focus on small, local areas of the input.


What part of CNN allows it to learn local patterns?
Convolutional layer
---
Traditional Fully Connected layers look at the entire input to find patterns.

Why does FCN have more parameters than CNN?
Parameter sharing
---
for FCN with single hidden layer of 512 neurons: parameters = (100x100x3) x 512 + 512 parameters = 15,360,512
for a Convolutional Neural Network with a 512 filters of size 3x3: parameters = (3x3x3+1)*512 = 14,336

What are some other CNN architectures?
VGG, Inception, Resnet
---
Memory: **VIR**al, very popular in the CNN world.
VGG: Visual Geometry Group. very small convolutional filters stacked in deep sequences.
Inception: i don't understand
ResNet: Residual Network. Introduces "residual connections", which train networks hundreds or thousands of layers.
Many modern CNN architectures build upon ideas from these 

What are some applications of CNNs?
Image classification, Object detection, Localization and Segmentation.
---
Object detection: Labeling whole scene + drawing a box, multiple object

What are some applications of CNN that involves 1 object?
Image classification and Localization
---
Image classification: Labeling whole scene
Localization: Labelling whole scene + drawing a box.

What are some applications of CNN that involves multiple objects?
Object detection and Segmentation
---
Object detection: Labelling each object and drawing a box around them.
Segmentation: Object detection but instead of box, is pixel-by-pixel border outline.
