You are a syllable shortener. Your job is to optimize the script such that the syllables spoken are as simple and as little as possible, while carrying the same information.

limit: 250wpm: 5*250 = 1250 words
## Page 1
Welcome, this is a deep learning experiment comparing 4 ML approaches.

Part 1 checks training from scratch and transfer learning.

Part 2 checks direct and indirect training data. Let's start part 1.

## Page 2
The goal of comparing training from scratch and transfer learning is to understand their pros and cons.

Training from scratch means training a brand new neural network, with random weights.

Transfer learning means using a neural network already trained on big datasets, and fine-tuning it, or using it to get features.

In ML, choosing between these is common. I want to compare the speed, ease, and compute cost of both.

## Page 3
The Face Emotion computer vision dataset is from Roboflow. It has 7 emotions and 5000 images, which isn't a lot.

It's split into 3500 train, 1000 valid and 500 test. 

Given the domain gap of ImageNet and this task, fine-tuning will be necessary.

## Page 4
First, I will train 3 versions of our custom network, EmotionNet.

Then, I will train ResNet18 followed by ResNet50. 

I chose Resnet because their architectures are well-regarded in computer vision, known for their strong performance. 

ResNet models were trained on ImageNet,

and are easily accessible on Pytorch.

## Page 5
Imports can be broken down into Standard library, Data handling & visualization, Machine Learning and Local imports.

## Page 6
MPS is available on my Macbook M3 pro to accelerate PyTorch training. For reference, one hour on CPU takes 10 mins.

## Page 7
The Images were augmented with 

cropping,

flipping,

rotating,

and color jittering.

This makes networks insensitive to these transformations.

These transformations are only done to training samples because when we test the model, we want to see how well it does on unaltered data. Let's start with EmotionNet

## Page 8
EmotionNet combines local and global features into a layer,

which is then flattened and fed to a classifier which makes a prediction.

## Page 9
Using OOP in Pytorch, the model is easy to understand.

Local feature pathway has smaller kernels to focus on detailed information, while global looks at overall expression.

## Page 10
We get a simple model with many parameters. 

## Page 11
The loss and accuracy graph over epochs show 

early convergence 

and signs of overfitting.

## Page 12
10 test image samples show that EmotionNet is not performant. It predicted happiness a lot, maybe since happiness is the majority class.

## Page 13
Left confusion matrix belongs to our EmotionNet, 

and Right represents the ideal.

EmotionNet seems to always predict happiness, causing poor performance.

## Page 14
EmotionNet has overfitted. We can use regularization techniques like Batch Normalization and Pooling.

Huge model size means we have to redesign a more efficient architecture, with less parameters and more efficient layers.

Cross pathway attention lets local and global feature pathways interact and influence each other. 

For example, the global pathway could guide the local pathway to focus on the relevant local features, and vice versa.

Self-attention weighs the importance of different parts of the input. 

For example, self-attention can help the model look at the most emotionally relevant features of the face.

These improvements would be what we're striving for in EmotionNetV2.

## Page 15
EmotionNetV2 is more complex.

First, the residual block helps with overfitting and efficiency. 

Second, the SelfAttention module is used for attention mechanisms in different parts of the network. 

Lastly, we'll learn about the other layers and see how they're initialized.

## Page 16
The residual block allows the network to go deeper without the vanishing gradient problem, similar to the Batch Normalization. This is more efficient because we can go deeper rather than wider.

## Page 17
Self attention uses the QKV as three convolutional layers,

to turn the input feature map into QKV representations,

so that we can calculate an attention map with the dot product between QK,

followed by a softmax operation. 

The attention map weighs V, letting it focus on the important parts.

The output is scaled by the learnable parameter gamma. We start with zero so it can gradually learn to incorporate attention.

## Page 18
You may not see Convolutional layers in the local and global pathways, 

but they're actually within the ResidualBlock. These ResidualBlock layers extract local and global features, 

MaxPool2d reduces spatial dimensions, 

and SelfAttention can weigh features. 

While the design is exactly the same between local and global,

this is more of a goal, and not enforced.

What's more important is that both pathways learn different sets of features, and even if they start similarly, the optimization process will push them to specialize in different aspects of the input.

## Page 19
The Cross-Pathway attention weighs the local features in context of global features, and vice versa, leading to a more integrated representation.

## Page 20
The Fusion Module prepares the combined features for classification, 

with a ResidualBlock to further process them,

SelfAttention to weigh features afterwards,

and a AdaptiveAvgPool2d to reduce the dimensions to a single value, producing a feature vector.

We also have a dropout2d to prevent overfitting.

## Page 21
Finally, the classifier maps the feature vector to the emotion class.

## Page 22
The resulting model is a deeper model,

but with much less parameters.

## Page 23
The loss and accuracy graph shows that

validation score is higher than training, which is desired.

## Page 24
The model is no longer

always predicting happiness.

## Page 25
However, we can see that the model still

predicts happiness quite frequently, which leaves much to be desired.

## Page 26
The benchmark shows much

smaller size and slightly lower inference time, so its more cheaper to run.

However, we see higher performance.

## Page 27
One last optimization we can make is to create an 

even smaller model with faster inference time. 

We can combine concepts from V1 and V2 to generate V3. 

## Page 28
EmotionNetV3 starts with shared features to save space.

## Page 28
Local and global blocks are inspired by V1, and uses 

smaller and larger kernels respectively.

## Page 29
Instead of SelfAttention, we have Channel and Spatial Attention, focusing on channel and spatial-wise feature importance.

## Page 30
The final layers are simpler, and uses a Conv2d block with a basic classifier.

## Page 31
The model did worse than V2, but it's smaller and faster.

## Page 32
To sum up, less is more, and a

good design can sometimes outperform more parameters. Onto Transfer Learning.

## Page 33
Look at how easy transfer learning is.

These two lines are all that is needed to initialize the model.

## Page 34
The loss and accuracy graph shows

slight overfitting. However,

the accuracy graph shows that our model hit 60%, reaching predictive levels we've not seen before.

## Page 35
The sample shows a huge jump in accuracy, scoring 7/10.

## Page 36
The misclassified images seem to be rather debatable too, even to the human eye.

## Page 37
We can see a beautiful 

diagonal line that matches the ideal confusion matrix

## Page 38
(No skipping)
Even with double accuracy 

and more parameters,

the inference time is much lesser than EmotionNetV3.

This is because of the highly computationally efficient optimizations made in the ResNet architecture.

There might be also be some library level optimizations in PyTorch that weren't applied on custom trained models. 

This proves that a model with more parameters may not be slower to infer, because the operations and computational cost per parameter may differ.

## Page 39
ResNet50 is like ResNet18

but instead of 18 layers,

it has 50, giving it higher predictive power.

## Page 40
Surprisingly, it is not that accurate. 

The matrix shows that it misclassifies sadness as anger,

fear as surprise, and though these emotions may look similar,

it has under performed ResNet18.

## Page 41
I would say the primary reason it underperformed

is the small dataset, which provides difficulty for deeper networks to converge to good performance levels.

## Page 42
In short, with carefully designed custom architectures,

we can achieve good performance with fewer parameters and smaller model size.

However, in the case where we have little data, transfer learning is best.

ResNet18 doubles the accuracy and efficiency. And this small dataset severely limits the use case of fine-tuning a deeper pre-trained model like the ResNet50.
