## Next section
Onto part 2. First, let's define our new dataset. It's a dataset of reviews of drugs, and in it,

We have ratings by the reviewers on the drugs.

Our goal is to predict sentiment from the reviews,

and the ratings will be our output variable.

## Next section

Direct and Indirect training means that while we test both of the generated models on the drug review test data split,

Indirect will be tested on another training dataset instead of Drug review Train, for example, IMDB reviews.

## Next section

For this experiment, we will not only have an equivalent model architecture of Training from scratch for both sides,

but also a Transfer learning component. This will bring more insights into how direct and indirect training affects these approaches.

## Next section

We import similar libraries to part 1, with the change from ResNet to AutoModel, allowing us to import 

MiniLM, an encoder transformer model that is smaller than even DistilBERT yet with good performance.

## Next section

Our correlation matrix of the training data shows that all columns exhibit weak correlation with each other, except for usefulCount and ratings.

The usefulCount shouldn't be included because it appears together with ratings. Therefore, in cases where you need ratings, you probably wouldn't have usefulCount as well.

Our imbalanced class distribution across ratings means that 

we have to include class weights that adjust our criterion.

For more alternative dataset options for our indirect training,

we need to bin our values into negative and positive sentiment.

To do so, I've read the sample reviews and

came up with the splitting threshold of between 6-7.

This is because at ratings 1-6, there's a strong focus on the negative effects of the drugs, whereas at ratings 7-10, even though they may be some initial side effects, they emphasize more on the drug's treatment effectiveness.

## Next section

After some basic preprocessing, 

and data splitting sorted by date to better reflect its real-word utility,

## Next section

We train our Direct training transfer learning, which shows a large amount of overfitting.

Our custom model is a bidirectional LSTM-based neural network for sentiment classification. 

It starts by embedding words into dense vectors,

processing these embeddings using a bidirectional LSTM to capture sequential context,

and then uses fully connected layers to classify the sentiment based on the LSTM's output.

Even larger amounts of overfitting are seen. Overfitting is a more common phenomenon compared to structured data, since text data is much more complex. Especially with LSTM-based networks.

If you recall, the previous graph showed that the pre-trained model overfitted the more epochs there were.

Thus, its interesting to see that the model initialized from random immediately had a big gap between training and validation metrics.

For indirect training, with the IMDB review dateset, 

we can see that our pretrained-model was also overfitted.

However, our model from scratch, although with graphs that can be considered as overfitting, has by far the least overfitting.

One additional reason our model could be overfitting was also the sort by date decision. 

If it is true, then it means that the ratings of the reviews may somewhat be correlated with some temporal function, maybe 

product quality increasing vs decreasing overtime, 

external events like a certain disease being more common, and so on.

Seeing this, we learnt a new potential factor. However,

we should still keep the current configuration because in production,

our model will always be predicting on future data, and date-based splitting will give us a more honest assessment of how our model performs.

## Next section

Our model comparison shows that the direct transformer achieved the highest performance across all metrics.

This is highly expected since pre-trained models are often more optimized for common tasks like sentiment classification and have been trained on large data.

Indirect pre-trained models achieved good performance, and seeing that transfer learning outperformed training from scratch, it makes sense that transfer learning is the best option.

Direct custom model performed just above random chance, with a significant drop of precision.

Indirect custom model performed worse than random chance, indicating that the indirect data probably gave it contradicting insights and lured it against the wrong option.

With this, it can be concluded that direct training consistently outperforms indirect training.

While its possible not have access to the domain-specific data due to resource constraints,

direct training is still the optimal solution for production systems.

And if possible, always experiment with transfer learning, because it is highly performant and easily trainable.
