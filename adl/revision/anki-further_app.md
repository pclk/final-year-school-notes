# Note

## Front

Why does Convolutional Networks need a lot of data?

## Back

A lot of parameters

---

If less data, overfitting may occur.

Just like a puppy needs to see the ball many times to understand what "fetch" means, a CNN needs to see *lots* of images to learn effectively.  

This is because CNNs have "large number of parameters."  

Think of parameters as the tiny little knobs and dials inside the puppy's brain that it adjusts as it learns.  

Lots of knobs mean lots of things to adjust, and to adjust them correctly, you need lots of examples.

# Note

## Front

Why is getting quality annotated data hard?

## Back

Time and resources

---

And here's the real kicker: getting good examples for a CNN is not just about quantity, but also *quality*.  

We need "quality annotated (labelled) data!"  This means for every image we show the CNN, we need to tell it what's in the image – "This is a cat," "This is a dog," etc.  

This "labeling" or "annotating" is often done by humans, and it's time-consuming, expensive, and sometimes even difficult to get right, especially for complex tasks.

# Note

## Front

What are some common techniques to tackle insufficient data?

## Back

Unsupervised learning, Data augmentation and Transfer Learning

---

Memory: UDT (user defined type, defining your own types)

Unsupervised: Easy to get unlabelled data vs labelled.

Data augmentation: take few pictures, alter many times and save them, get more pictures.

Transfer Learning: take trained model and fine-tune. "Make faster progress using pretrained weights as initial weights for your network instead of some random initialized values, especially when you have small dataset.

Instead of starting with a completely untrained CNN (randomly initialized "weights"), we can take a pre-trained CNN and use its learned "weights" as a starting point.  

This is especially helpful when we have a small dataset for our specific task. It's like giving our puppy a head start in its training because it already knows some basic commands.

# Note

## Front

What are the 5 image transformations?

## Back

Zoom, Rotate, Crop, Contrast, Flip

---

Memory: ZRCCF ("Zooming Robots Can't Catch Fish")

# Note

## Front

Why could models learn better with Data Augmentation?

## Back

Recognize the object despite looking a little different.

---

By showing all these *augmented* images, you're helping your puppy learn that a Golden Retriever is still a Golden Retriever, even if it looks a little different.  You're making it more robust and able to recognize Golden Retrievers in all sorts of situations.

# Note

## Front

Why can we apply Data Augmentation technique to transfer learning as well?

## Back

Still provides the same benefits

---

This means no matter how you're training your model – whether you're starting from zero or using a pre-trained model, Data Augmentation can be helpful.

# Note

## Front

Why can't we apply data augmentation to our test/validation data?

## Back

See its performance on real, unaltered data.

---

This is crucial! We only augment our *training* data.  Why? Because we want to *teach* the model to be more flexible.

But when we *test* or *validate* the model, we want to see how well it performs on *real, unaltered* data.  We're not trying to trick ourselves into thinking the model is better than it is by augmenting the test data.

# Note

## Front

What could possibly be an unintended consequence of Data Augmentation?

## Back

Flipping them may change its meaning

---

For images, flipping an image of a cat is usually fine, it's still a cat. But if you were dealing with images of handwritten digits '6' and '9', vertically flipping them would change them into each other, which would be disastrous!

# Note

## Front

How can we train an accurate model without large amounts of training data and computing hours?

## Back

Transfer Learning

---

**Transfer Learning** is like giving your puppy that head start.  

Instead of training a model from scratch (from random "puppy brain" settings), we use a model that's already been trained on a *massive* dataset, like ImageNet. 

ImageNet is a dataset with millions of images of all sorts of things – cats, dogs, cars, trees, you name it.

# Note

## Front

What are the two main approaches of Transfer Learning?

## Back

Feature Extractor and Fine-Tuning

---

Feature Extractor: Imagine the pre-trained model as a super-smart "feature-finding machine". It takes an image and extracts all the important features – edges, textures, shapes, etc. 

We can use this machine to process our dog breed images and get a set of features for each image.

Fine-Tuning: Fine-tuning means we *unfreeze* some of the layers of the pre-trained model (especially the later layers, which are more task-specific) and train them on our dog breed data *along with* the new classifier layers.  

We're essentially adjusting the pre-trained model to be more specialized for our specific task.

# Note

## Front

What do we replace in a Pre-trained model when we want to use them as a Feature Extractor?

## Back

Classification head

---

Last few dense layers.

Then, we just need to train a *simple* classifier (like a few dense layers, as mentioned in the text) on top of these extracted features to actually learn to distinguish between dog breeds.  We "freeze" the "feature-finding machine" part (the convolutional base) so it doesn't change, and only train the new classifier part.

It's like using a pre-built engine in a car. The engine (feature extractor) is already great, we just need to build a new body and interior (classifier) that's specific to our needs.

# Note

## Front

What are the two factors of level of fine-tuning?

## Back

Size of dataset and Similarity of dataset

---

Size: If you have a *small* dataset for your new task, you should do *less* fine-tuning.  Why? Because if you fine-tune too much with a small dataset, you risk *overfitting*.  

The model might start to memorize your small dataset instead of learning generalizable features. In this case, using the pre-trained model as a feature extractor or fine-tuning only the very last layers is safer.

Similarity: If your task is very *similar* (e.g., classifying different types of flowers, which is still image classification and somewhat similar to ImageNet), you might need *less* fine-tuning. The pre-trained features are already quite relevant.

If your task is very *different* (e.g., classifying X-ray images for lung cancer, which is a very different domain from everyday images in ImageNet), you'll likely need *more* fine-tuning. You need to adapt the pre-trained features more significantly to the new domain.

# Note

## Front

What if our dataset is small and similar?

## Back

Feature Extractor

---

The model might start to memorize your small dataset instead of learning generalizable features. In this case, using the pre-trained model as a feature extractor or fine-tuning only the very last layers is safer.

# Note

## Front

What if our dataset is large and similar?

## Back

Fine tune through full network

---

With a large dataset, we can be more aggressive with fine-tuning. We can allow the model to adjust more of its internal "knobs and dials" because we have enough data to guide it in the right direction and prevent overfitting.

# Note

## Front

What if our dataset is small and different?

## Back

Fine tune from activations earlier in the network

---

You still want to leverage *some* of its learned features. You might unfreeze and fine-tune some of the *earlier* layers of the network.

Earlier layers in CNNs tend to learn more general features (like edges, textures), which might still be useful even for a different domain.

You'd be more cautious and fine-tune fewer layers, especially the later, more task-specific layers, because you have a small dataset and don't want to overfit.

# Note

## Front

What if our dataset is large and different?

## Back

Fine tune through full network.

---

Even though the task is different, you have a lot of data to work with. "Fine-tune through full network" is again the recommended approach. You can unfreeze and retrain a large portion or all of the pre-trained model. The large dataset allows you to significantly adapt the model to the new domain.  It's like saying, "Puppy, 'cat language' is very different, but we have tons of resources to learn it! Let's really dive in and retrain you extensively so you become fluent in 'cat language'."

# Note

## Front

What stages are there in the multi-stage Object detection algorithms?

## Back

Region proposal and classification/regression.

---

**Step 1: Region Proposal (Finding Potential Toys):** First, you teach your puppy to scan the room and point out *areas* where there *might* be a toy.  It's like saying, "Puppy, look around and show me all the spots that *could* contain a toy."  

The puppy might point to a corner, under the sofa, or on the rug. It's not sure if there's a toy there yet, but these are just *potential* toy locations.  

In the world of algorithms, this is like the **region proposal stage**. Algorithms like **R-CNN, Fast R-CNN, and Faster R-CNN** work this way. They first propose regions in the image that might contain an object.

**Step 2: Classification and Regression (Identifying and Locating the Toy):** Once the puppy has pointed out potential toy locations, you go to each spot and teach it to:

**Classify:**  "Puppy, is there a toy here? If yes, is it a ball or a frisbee?"  This is like the **classification stage**, where the algorithm figures out *what* object is in the proposed region (if any).

**Regression (Locating):** "Okay, puppy, if it's a toy, can you draw a box around it to show me exactly where it is?" This is like the **regression stage**, where the algorithm refines the location of the object and draws a bounding box around it.

# Note

## Front

Why would you use Single-stage Object detection Algorithms?

## Back

Very fast, suitable for real-time detection.

---

**Pros: Very Fast.** Because they do everything in one stage, these algorithms are much faster. They are **suitable for real-time detection**.  This puppy is perfect for a fast-paced game of fetch!

**Cons: Not as Accurate.**  Being fast sometimes means sacrificing a bit of accuracy.  Since they are doing everything quickly, single-stage algorithms might not be as precise as multi-stage ones, especially in complex scenes with many objects.  The speedy puppy might sometimes mistake a crumpled napkin for a ball if it's not careful!

# Note

## Front

Why would you use Multi-stage Object detection Algorithms?

## Back

Very accurate.

---

**Multi-stage (Accuracy Focused):**  Think of applications where accuracy is paramount, even if it takes a bit longer, like medical image analysis (detecting tumors) or high-precision object recognition in security systems.

# Note

## Front

When both the input and output are sequence, what kind of RNN architecture are we looking at?

## Back

Seq2Seq

---

Seq2Seq models are designed for situations where both the input and the output are sequences, and importantly, these sequences can be of *different lengths*.  "Good dog, sit down" is four words, "Bon chien, assieds-toi" is three words.  This flexibility is key!

# Note

## Front

What component of our RNN architecture plays the role of listening and understanding?

## Back

Encoder

---

Imagine the Encoder as the puppy's *listening and understanding* part of the brain.  It takes the input sequence, word by word:

*   "Good" -> Encoder processes it

*   "dog"  -> Encoder processes it, *remembering* "Good"

*   "sit"  -> Encoder processes it, *remembering* "Good dog"

*   "down" -> Encoder processes it, *remembering* "Good dog sit"

As it reads each word, the Encoder builds up a **context vector**. Think of this context vector as a summary, a little thought bubble in the puppy's brain that captures the *meaning* of the entire input sentence "Good dog, sit down."  It's like the puppy now *gets* what you're asking it to do.

# Note

## Front

What component of our RNN architecture plays the role of generating the output sequence?

## Back

Decoder

---

Once the Encoder has created this "understanding" (the context vector), the **Decoder** comes into play.  The Decoder is like the puppy's *action-taking* or *speaking* part of the brain.  It takes the context vector (the "understanding" of "Good dog, sit down") and uses it to generate the *output sequence*, which could be:

*   The puppy actually sitting down (if we were teaching a command)

*   The French translation "Bon chien, assieds-toi" (for machine translation)

The Decoder generates the output sequence word by word, based on the context vector.  It's like the puppy, having understood "Good dog, sit down," now *thinks* in French and starts "speaking" (outputting) "Bon", then "chien", then "assieds-toi".

# Note

## Front

What RNN architecture is the following example: Image captioning that takes an image and outputs a sentence of words?

## Back

One to Many

---

Use when you want to *expand* a single idea into a sequence of details (like captioning an image).

**1. One to Many: The "Recipe Expander" Chef**

*   **Concept:** Takes a single input and produces a sequence of outputs.

*   **Analogy:** Imagine you give our chef *one* ingredient, let's say, "chicken".  The chef's task is to create a whole *menu* of dishes you can make with chicken: "Chicken soup, Chicken salad, Roasted chicken, Chicken stir-fry..."  One input (chicken) expands into a sequence of outputs (menu items).

*   **Example (Image Captioning):**  You give the RNN *one* input – an image of a cat. The RNN then generates a *sequence* of words describing the image: "A fluffy cat is sitting on a mat."

*   **In essence:**  Starting from a single point, we branch out into a sequence.

# Note

## Front

What RNN architecture is the following example: Sentiment Analysis, given a sentence and classify as positive or negative?

## Back

Many to one

---

Use when you want to *summarize* a sequence of information into a single takeaway (like getting the sentiment of a review).

**2. Many to One: The "Summarizer" Chef**

*   **Concept:** Takes a sequence of inputs and produces a single output.

*   **Analogy:**  Imagine you give our chef a *long recipe* – a sequence of instructions: "Chop onions, sauté garlic, add tomatoes, simmer for 30 minutes..." The chef reads through all the instructions and then gives you *one* final result: "Tomato Soup".  Many inputs (recipe steps) condense into a single output (the dish).

*   **Example (Sentiment Analysis):** You give the RNN a *sequence* of words – a movie review: "This movie was amazing! The acting was superb, and the plot kept me hooked." The RNN analyzes the entire review and gives *one* output: "Positive Sentiment".

*   **In essence:** We process a whole sequence to get a single, summarized understanding.

# Note

## Front

What RNN architecture is the following example: Machine translation from english to french?

## Back

Many to Many (Seq2Seq)

---

Use when you need to *translate* or transform one sequence into another sequence, possibly of different length (like translating languages).

# Note

## Front

What RNN architecture is the following example: Video classification, labelling each frame of the video?

## Back

Many to Many (not Seq2Seq)

---

Use when you need to *label* or analyze each element of a sequence in relation to its context within the sequence (like labeling frames in a video).

