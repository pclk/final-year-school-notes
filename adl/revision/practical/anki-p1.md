# Note

## Front

How do we get our conda environment activated in jupyterlab?

## Back

pip install ipykernel

python -m ipykernel install --user

# Note

## Front

How do we load a dataset called "mnist" using keras?

## Back

(train_x, train_y), (test_x, test_y) = keras.datasets.mnist.load_data()

---

this is quite hard to memorize, AI please help

# Note

## Front

What extension does keras.datasets.mnist.load_data() download?

## Back

npz

# Note

## Front

How do you print a shape of an numpy array called train_x?

## Back

train_x.shape

# Note

## Front

How do you print the first 10 images of a dataset?

## Back

rows = 2

cols = 6

fig, axes = plt.subplots(rows, cols)

for j in range(rows):

* for i in range(cols):

    * img=train_x`[j*cols+i]`

    * axes`[j, i]`.imshow(img,cmap='gray')

---

1. Iterates through each figure.

    These are nested `for` loops.  They are used to iterate through each position in our 2x5 grid of subplots.

    *   **`for j in range(2):`**: The outer loop uses the variable `j`. `range(2)` generates numbers 0 and 1. We can think of `j` as representing the *row index*. So, `j` will be 0 for the first row and 1 for the second row.

    *   **`for i in range(cols):`**: The inner loop uses the variable `i`. `range(cols)` which is `range(5)` generates numbers 0, 1, 2, 3, and 4. We can think of `i` as representing the *column index*. So, `i` will go from 0 to 4 for each row.

    Together, these nested loops will go through all the combinations of `(j, i)`:  (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4).  These pairs of indices perfectly correspond to the positions in our 2x5 grid of subplots.

2. Selects the correct image.

3. Shows the image.

# Note

## Front

How to scale numerical values to 0 to 1?

## Back

Divide by itself

---

train_x_scaled = train_x / 255

test_x_scaled = test_x / 255

# Note

## Front

How to sample first 1000 rows of a dataset?

## Back

train_x_1000 = train_x_scaled`[:1000]`

train_y_1000 = train_y`[:1000]`

test_x_500 = test_x_scaled`[:1000]`

test_y_500 = test_y`[:1000]`

# Note

## Front

How to flatten a 1000 long numpy array containing 28x28 numpy arrays called `train_x_1000`?

## Back

train_x_1000 = train_x_1000.reshape(1000, 784)

# Note

## Front

How to flatten a 500 long numpy array containing 28x28 numpy arrays called `test_x_1000`?

## Back

test_x_500 = test_x_500.reshape(500, 784)

# Note

## Front

How to create a sequential model?

## Back

model = keras.Sequential()

---

`model = keras.Sequential()`:  Think of `keras.Sequential()` as creating an empty factory assembly line.  "Sequential" means that the data will flow through the layers in the order we add them, step-by-step, just like items on an assembly line. We are naming our factory `model`.

# Note

## Front

How to add a dense hidden layer? 128 neurons, where the input has 784 features?

## Back

model.add( 

* keras.layers.Dense(128, input_shape=(784,), activation="relu")

)

---

`model.add(...)`: This is like adding a processing unit to our factory assembly line.

`keras.layers.Dense(...)`:  We are adding a specific type of processing unit called a "Dense layer". "Dense" or "fully-connected" means that every input from the previous step is connected to every "neuron" in this layer. Imagine every worker in this unit is connected to all the information coming from the previous unit.

`128`: This is the number of "neurons" or "units" in this Dense layer. Think of these as 128 workers in our processing unit. More workers can potentially learn more complex patterns, but also increase the complexity and computational cost.

`input_shape=(784,)`: This is very important for the *first* layer in a `Sequential` model. It tells the factory what kind of input to expect.  Remember earlier in the practical, we flattened our 28x28 images into a 1-dimensional array of 784 features (28 \* 28 = 784). So, `input_shape=(784,)` says: "Hey factory, expect to receive 784 input values for each image."  This is like telling the first processing unit that each item on the assembly line will have 784 parts.

`activation="relu"`: This is where we need to fill in the activation function!  Activation functions are like decision-making rules for each neuron. They decide whether a neuron should "fire" or not based on the input it receives.  For now, the code is asking you to fill this in.  A common choice for hidden layers is `'relu'` (Rectified Linear Unit).  We'll talk more about activation functions later, but for now, think of it as adding a specific type of logic to our processing unit.

# Note

## Front

How to add an output layer? You need to classify digits from 0 to 9?

## Back

model.add(

* keras.layers.Dense(10, activation="softmax")

)

---

We are adding *another* Dense layer, another processing unit to our factory.

`10`: This time, we have `10` units. Why 10? Because we want to classify digits from 0 to 9, which are 10 classes!  This output layer will have 10 neurons, each corresponding to one digit class (0, 1, 2, ..., 9).

`activation=""`: Again, we need to fill in the activation function. For the output layer in a classification problem with multiple classes (like digits 0-9), a very common and suitable choice is `'softmax'`.  `softmax` will convert the outputs of these 10 neurons into probabilities.  Each output will be a number between 0 and 1, and the sum of all 10 outputs will be 1.  These probabilities represent the model's confidence that the input image belongs to each of the 10 digit classes.

# Note

## Front

How to print a summary of the model?

## Back

print(model.summary())

---

`model.summary()`: This is like getting a blueprint or a report of our factory. It prints out a summary of the model's architecture. When you run this, it will show you:

The type of each layer (Dense).

The output shape of each layer (how the data changes as it passes through each layer).

The number of parameters (weights and biases) in each layer. Parameters are the things the model *learns* during training.  More parameters mean a more complex model.

The total number of trainable parameters.

# Note

## Front

What are the steps to get a trained model?

## Back

Initialize > Layers > Compile > Fit

---

Step 1:

model = keras.Sequential()

Step 2:

model.add(keras.layers.Dense(128, input_shape=(784,), activation=""))

model.add(keras.layers.Dense(10, activation=""))

Step 3:

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=`["accuracy"]`)

Step 4:

model.fit(train_x_1000, train_y_1000, epochs=5, validation_data=(test_x_500, test_y_500))

# Note

## Front

How to compile the model?

## Back

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

---

`model.compile(...)`:  This is like setting up the factory for the learning process. We need to tell it:

`optimizer="adam"`:  The "optimizer" is like the method our factory will use to improve itself.  "Adam" is a popular and efficient optimization algorithm.  Think of it as the factory's learning strategy. It helps the model adjust its internal parameters (weights and biases) to minimize errors.

`loss="sparse_categorical_crossentropy"`: The "loss function" is how we measure how badly our factory is performing. "sparse\_categorical\_crossentropy" is a suitable loss function for multi-class classification problems like digit recognition. It measures the difference between the model's predicted probabilities and the true labels.  A higher loss means the model is doing worse, and a lower loss means it's doing better.  Think of it as a "mistake counter" for the factory.

`metrics=["accuracy"]`: "Metrics" are what we want to track to evaluate our factory's performance. "accuracy" is a common metric for classification. It simply measures the percentage of images that the model classifies correctly.  This is like tracking the "success rate" of our factory.

# Note

## Front

How to fit the model?

## Back

model.fit(train_x_1000, train_y_1000, epochs=5, validation_data=(test_x_500, test_y_500))

---

`model.fit(...)`: This is where the actual learning happens!  It's like feeding our factory with training data and letting it learn to recognize digits.

`train_x_1000`, `train_y_1000`: These are our training data. `train_x_1000` is the input images (flattened to 784 features), and `train_y_1000` are the corresponding correct labels (the actual digits, like 0, 1, 2, ..., 9). We are using the first 1000 samples of our training data here.

`epochs=5`:  "Epochs" means how many times we want to go through the *entire* training dataset. `epochs=5` means we will show the factory the entire training dataset 5 times.  More epochs can lead to better learning, but also risk "overfitting" (memorizing the training data too well and not generalizing to new data).

`validation_data=(test_x_500, test_y_500)`: This is our validation data.  While the model is training on the `train_x_1000` data, it will also periodically check its performance on the `test_x_500` data. This helps us monitor if the model is generalizing well to unseen data and prevents overfitting. We are using the first 500 samples of our test data for validation here.

# Note

## Front

What does model.fit do repeatedly in a epoch?

## Back

Feedforward, Loss, Backpropagation, Validation

---

1.  **Feed data forward:** It takes a batch of training images (`train_x_1000`) and passes them through the factory (neural network). Each layer processes the data, applying its weights, biases, and activation functions.  The factory produces predictions (probabilities for each digit class).

2.  **Calculate loss:** It compares the predictions with the true labels (`train_y_1000`) using the loss function (`sparse_categorical_crossentropy`). This tells us how wrong the predictions were.

3.  **Backpropagation and Optimization:**  Using the optimizer (`adam`), the model adjusts its internal parameters (weights and biases) to reduce the loss.  This is the "learning" step. It's like the factory workers adjusting their processes to make fewer mistakes in the future.

4.  **Validation (optional):** After each epoch (or a set of epochs), it evaluates the model's performance on the `validation_data` to see how well it generalizes.

# Note

## Front

How do we make the model predict against test_x_500?

## Back

model.predict(test_x_500)

---

**`model.predict(...)`**: This is the core action - it's the student *taking the exam*. We are using the `predict()` function of our trained model.

**`test_x_500`**:  This is the input to the `predict()` function. It's the set of *test images* (remember, we've flattened them into 784 features each).  These are the new questions for our student.

**`predictions = ...`**: The `predict()` function doesn't just give us a single answer. For each test image, it gives us a *list of confidence levels* for each digit class (0 to 9).  Think of it like this: for each question, the student provides a score (confidence level) for each possible answer (digit 0 to 9).  The `predictions` variable will store all these confidence scores for all the test images.

This line gets predictions for all 500 test images at once. For each image, it returns 10 confidence scores (one for each digit 0-9).

# Note

## Front

How do you print the test label?

## Back

for i in range(len(test_x_500)):

* print("Test Label: ", test_y_500[i])

---

Shows the actual correct digit (ground truth).

# Note

## Front

How do you print the confidence scores?

## Back

for i in range(len(test_x_500)):

* print("Confidence: ", predictions[i])

---

Shows the confidence scores for all 10 digits (0-9). For example:

`[5.9525709e\-04 9.4538809e\-06 2.4333365e\-05 5.7085850e\-05 1.0240891e\-02 7.4954827e\-05 8.9236892e\-05 3.0947337e\-01 6.9667515e\-04 6.7873871e\-01]`

# Note

## Front

How do you print the indices of the prediction with the highest confidence?

## Back

for i in range(len(test_x_500)):

* print("Prediction made: ", np.argmax(predictions[i])

---

`np.argmax()` finds the position of the highest confidence score. If position 2 has the highest score (0.85 in our example), then the model predicts this image is a "2".

