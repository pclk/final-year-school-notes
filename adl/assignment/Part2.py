r"""°°°
# Assignment 2024 S2
## Part 2: Structured Data - Direct vs Indirect Training Data

Citation: Some code has been generated with the help of Claude 3.5 Sonnet by Anthropic, and some decisions and further clarifications were made with gemini-2.0-flash-thinking-exp-01-21

drugName (categorical): name of drug

condition (categorical): name of condition

review (text): patient review

rating (numerical): 10 star patient rating

date (date): date of review entry

usefulCount (numerical): number of users who found review useful
°°°"""
# |%%--%%| <g1FDS91ZE5|S7Gp3YhJ0H>

# Standard library imports
import logging
import time
from pathlib import Path

# Data handling & visualization
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pandas as pd
from ydata_profiling import ProfileReport
from datasets import load_dataset

# Data preprocessing and feature engineering
import re
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from dataset2 import DrugReviewDataset, IndirectDataset, TextDatasetFromScratch
from collections import Counter
import torch.nn.functional as F

import importlib
import dataset2
importlib.reload(dataset2)
from dataset2 import TextDatasetFromScratch, DrugReviewDataset, IndirectDataset

# |%%--%%| <S7Gp3YhJ0H|fwgkj7LEKl>
r"""°°°
## Data Understanding
°°°"""
# |%%--%%| <fwgkj7LEKl|3zYeXQG5k7>

INPUT_TRAIN = "drug_review_train.csv"
INPUT_TEST = "drug_review_test.csv"

INPUT_TRAIN = "drug_review_train.csv"
INPUT_TEST = "drug_review_test.csv"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

df = pd.read_csv(INPUT_TRAIN)
df_test = pd.read_csv(INPUT_TEST)

MAX_LENGTH = 256
NUM_EPOCH = 20

# |%%--%%| <3zYeXQG5k7|uiqxqomfsp>

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# |%%--%%| <uiqxqomfsp|RDHJLHlHuW>

# # --- Data Profiling ---
# print("Profiling train data...")
# profile_train = ProfileReport(
#     df,
#     title="Drug Review Train Data Profiling Report",
#     explorative=True,
# )
# profile_train.to_file("drug_review_train_profiling.html")

# print("Profiling test data...")
# profile_test = ProfileReport(
#     df_test,
#     title="Drug Review Test Data Profiling Report",
#     explorative=True,
# )
# profile_test.to_file("drug_review_test_profiling.html")
# print("Profiling complete. Reports saved as HTML files.")

# |%%--%%| <RDHJLHlHuW|fS0yA3I1wk>

# Create a DataFrame with column descriptions
column_info = {
    'Column Name': ['drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
    'Data Type': ['categorical', 'categorical', 'text', 'numerical', 'date', 'numerical'],
    'Description': [
        'Name of the drug',
        'Name of the medical condition',
        'Patient review text',
        '10-star patient rating',
        'Date of review entry',
        'Number of users who found review useful'
    ]
}

column_df = pd.DataFrame(column_info)
column_df

# |%%--%%| <fS0yA3I1wk|yQOeMntITI>

# Analyze usefulCount zeros
df = pd.read_csv(INPUT_TRAIN)
zero_useful_count = (df["usefulCount"] == 0).sum()
total_reviews = len(df)
zero_percentage = (zero_useful_count / total_reviews) * 100

print(f"\n--- UsefulCount Analysis ---")
print(f"Total reviews: {total_reviews:,}")
print(f"Reviews with zero useful votes: {zero_useful_count:,}")
print(f"Percentage of zero useful votes: {zero_percentage:.2f}%")

# |%%--%%| <yQOeMntITI|7ldxiN6lHp>
r"""°°°
### Analysis of Zero UsefulCount Impact on Sentiment:

1. Silent Majority Phenomenon:
   - The high percentage of zero useful votes suggests a classic "lurker" behavior in online communities
   - Most users read but don't interact, creating a participation inequality
   - This means our sentiment analysis might be biased towards more "engaging" content

2. Sentiment Validation Gap:
   - Reviews with zero useful votes lack community validation
   - We can't assume these reviews are less valuable - they might be newer or simply not seen by many users
   - This creates a potential temporal bias in our sentiment understanding

3. Engagement vs. Sentiment Relationship:
   - Higher useful counts might indicate more polarizing content rather than more accurate sentiment
   - Extreme opinions (very positive or very negative) tend to attract more engagement
   - This suggests we should be cautious about weighing sentiment by useful counts

4. Data Quality Implications:
   - Zero useful counts might indicate:
     a) Fresh reviews that haven't had time to accumulate votes
     b) Reviews that didn't reach many readers
     c) Reviews that readers found neither particularly helpful nor controversial
   - This impacts how we should approach sentiment weighting in our analysis
°°°"""
# |%%--%%| <7ldxiN6lHp|0sB2ASMC2a>

print(f"\n--- UsefulCount Analysis ---")
print(f"Total reviews: {total_reviews:,}")
print(f"Reviews with zero useful votes: {zero_useful_count:,}")
print(f"Percentage of zero useful votes: {zero_percentage:.2f}%")

# Create correlation matrix
print("\n--- Correlation Matrix Analysis ---")
# Select only numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = df[numerical_cols].corr()

# Create correlation heatmap
plt.rcParams.update({"font.size": 14})
plt.figure(figsize=(10, 8), dpi=400)
sns.heatmap(correlation_matrix, 
            annot=True,  # Show correlation values
            fmt='.3f')  # Format correlation values to 2 decimal places
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()

# |%%--%%| <0sB2ASMC2a|THaKZupoJK>
r"""°°°
### Key Correlation Findings:
- Correlation between rating and usefulCount: 0.243
- Most other numerical features show weak or no correlation
- This suggests that higher rated reviews tend to be found slightly more useful by readers, or vice versa

### Should we include usefulCount?
usefulCount is a similar metric to ratings, and its logical to think that in situations you don't have ratings, you probably wouldn't have the usefulCount too. 

Some cases of situations where you don't have ratings are if you're predicting how positive a review is. Usually, you would only have the review and the patientid only, while the review_length can be determined from review. Thus, it may not be fair to include usefulCount in our machine learning, especially if we're focusing on text sentiment classification.
°°°"""
# |%%--%%| <THaKZupoJK|NF94lyETUC>

# Create histogram of ratings with percentage labels
plt.figure(figsize=(12, 6), dpi=400)


# Calculate histogram data
counts, bins, _ = plt.hist(df['rating'], bins=10, edgecolor='black')
total = len(df['rating'])

for i in range(len(counts)):
    percentage = (counts[i]/total) * 100
    plt.text(bins[i], counts[i], 
             f'{percentage:.1f}%', 
             va='bottom')

plt.title('Distribution of Drug Ratings')
plt.xlabel('Rating (0-10)')
plt.ylabel('Number of Reviews')
plt.tight_layout()
plt.show()



# |%%--%%| <NF94lyETUC|igNU20y8r5>
r"""°°°
The question is how can we split this into positive and negative sentiment?

1. **Highly Imbalanced Distribution**:
   - Ratings 9 and 10 dominate (48.4% combined)
   - Rating 10 alone is 30.9% of all reviews
   - Ratings 3, 4, and 6 are severely underrepresented (each around 3-4%)
   - This creates a significant class imbalance problem

2. **Bimodal Distribution**:
   - There are two peaks: Rating 1 (12.9%) and Ratings 9-10 (48.4%)
   - This suggests strong polarization in reviews
   - Middle ratings (3-6) are less common
   - This validates our earlier decision to split into two classes (1-6 vs 7-10)

3. **Machine Learning Implications**:

   a) **Class Imbalance Solutions Needed**:
   - Consider using class weights
   - Implement oversampling (SMOTE) for minority classes
   - Use undersampling for majority classes
   - Or combine both (SMOTEENN, SMOTETomek)

   b) **Evaluation Metrics**:
   - Accuracy alone would be misleading
   - Need to focus on:
     * F1-score
     * Precision and Recall
     * ROC-AUC
     * Confusion matrix analysis

   c) **Model Selection**:
   - Choose algorithms that handle imbalanced data well
   - Consider ensemble methods
   - Use stratification in train/test splits

4. **Neural Network Considerations**:
   - The imbalanced distribution affects deep learning models differently than traditional ML:
     * Deep learning models often need MORE data per class for effective learning
     * The severe underrepresentation of ratings 3-6 (each ~3-4%) is particularly problematic
     * The dominance of rating 10 (30.9%) could cause model bias

5. **Deep Learning Specific Solutions**:

   a) **Data Augmentation**:
   - For text data, we can use:
     * Back-translation
     * Synonym replacement
     * Text generation using LLMs
     * EDA (Easy Data Augmentation) techniques
   - These help increase samples for underrepresented ratings

   b) **Loss Functions**:
   - Use specialized loss functions:
     * Weighted Cross-Entropy Loss
     * Focal Loss (reduces impact of easy, common samples)
     * Class-Balanced Loss
   - These help handle class imbalance during training

   c) **Architecture Choices**:
   - Consider:
     * Pre-trained language models (BERT, RoBERTa)
     * Multi-task learning approaches
     * Attention mechanisms to focus on important parts of reviews
   - These help leverage the bimodal nature of the distribution

6. **Training Strategies**:
   - Implement:
     * Gradient accumulation
     * Progressive resizing
     * Curriculum learning (start with balanced subsets)
   - Use dynamic batch sampling:
     * Over-sample minority classes within batches
     * Ensure each batch sees all rating classes

7. **Validation Considerations**:
   - Use:
     * Stratified k-fold cross-validation
     * Balanced validation sets
     * Multiple evaluation metrics
   - Monitor for:
     * Overfitting on majority classes
     * Underfitting on minority classes
     * Class-wise performance metrics

°°°"""
# |%%--%%| <igNU20y8r5|r9rwdMeJdx>

def wrap_text(text, width=80, indent=4):
    """
    Custom function to wrap text with indentation
    Args:
        text (str): The text to wrap
        width (int): Maximum width of each line
        indent (int): Number of spaces for indentation
    Returns:
        str: Wrapped and indented text
    """
    # Split text into words
    words = text.split()
    # Initialize variables
    lines = []
    current_line = " " * indent  # Start with indentation
    current_width = indent

    for word in words:
        # Calculate width if we add this word
        if current_width + len(word) + 1 <= width:
            # Add word with a space
            if current_width > indent:  # If not the first word in line
                current_line += " "
                current_width += 1
            current_line += word
            current_width += len(word)
        else:
            # Line is full, start a new line
            lines.append(current_line)
            current_line = " " * indent + word
            current_width = indent + len(word)
    
    # Add the last line
    if current_line:
        lines.append(current_line)
    
    return "\n".join(lines)

# Sample reviews from each rating category
N_REVIEWS = 10
print("\n=== Sample Reviews by Rating ===")
for rating in sorted(df["rating"].unique()):
    print(f"\nRating {rating:.1f} - {N_REVIEWS} Sample Reviews:")
    print("-" * 80)
    sample_reviews = df[df["rating"] == rating].sample(
        n=min(N_REVIEWS, len(df[df["rating"] == rating]))
    )
    # Display in a more readable format
    for idx, row in sample_reviews.iterrows():
        print(f"Drug: {row['drugName']}")
        print(f"Condition: {row['condition']}")
        print(f"UsefulCount: {row['usefulCount']}")
        print("Review:")
        # Use our custom wrap_text function
        wrapped_review = wrap_text(row["review"], width=80, indent=4)
        print(wrapped_review)
        print("-" * 80)

# |%%--%%| <r9rwdMeJdx|7XcbZlmLlb>
r"""°°°
Summary:

The reviews highlight a wide range of experiences, from severe negative side effects and dissatisfaction to significant relief and positive outcomes. Many reviews, especially at the lower ratings, focus on negative side effects such as nausea, weight gain, bleeding, mood changes, and digestive issues. Higher-rated reviews often acknowledge some initial side effects but emphasize the drug's effectiveness in treating the condition. Some medium-rated reviews acknowledge postive side effects of the drug, but not effective overall.

Sentiment Threshold:

Based on the provided samples, the sentiment threshold appears to be around a rating of 7.0.

Sentiment Threshold Analysis:

Looking closely at the reviews within each rating level, and paying attention to how the language and described experiences change, here's a refined breakdown and the apparent threshold:

1.0: Almost universally extremely negative. Users describe severe, debilitating side effects, complete lack of effectiveness, and often dangerous reactions. Words like "horrible," "awful," "die," "severe pain," and descriptions of emergency room visits are common. These are clearly negative experiences.

2.0: Still overwhelmingly negative. The language is similar to the 1.0 reviews, focusing on significant side effects, lack of efficacy, and regret. There's a sense of frustration and disappointment. Some reviews mention stopping the medication due to the negative experience.

3.0: Predominantly negative, but with hints of mixed experiences. While many reviews still detail significant side effects and problems, some acknowledge potential benefits or that the drug might work for others, even if it didn't work for them. There's more ambivalence here, but the overall tone leans negative. We see phrases like "takes some getting used to," "overwhelming," "not worth it," and descriptions of weight gain, mood changes, and other undesirable effects.

4.0: A definite mix of negative and slightly more neutral experiences, but still leaning negative overall. Users often describe a trade-off: the medication might help with the condition to some extent, but the side effects are significant and disruptive. There's a sense of weighing pros and cons, and often the cons are still winning. We see mentions of both positive effects (e.g., "worked for a few years," "pain was less") and negative ones ("wasn't pleasant," "side effects such as lack of concentration," "gaining weight").

5.0: Truly mixed, and the most difficult to categorize neatly. These reviews represent a clear "tipping point." Some users report positive effects on the condition, but significant side effects often counterbalance those benefits. Other users report minimal benefits and persistent problems. There's a strong sense of individual variability and uncertainty. The language is less intensely negative than lower ratings, but still expresses concern and dissatisfaction. Key phrases: "hit or miss," "side effects were slim in the first couple of months but soon after...," "worked really good for that [one symptom]... [but had other significant negative effects]."

6.0: Similar to 5.0, a mix of positive and negative, but with a slight shift towards acknowledging benefits, even with ongoing issues. The reviews often describe a situation where the drug helps, but the side effects are still a significant factor, leading to a less-than-ideal experience. There's a sense of compromise and ongoing evaluation. We see phrases like "love/hate relationship," "better than [previous medication]," "side effects improved," and "debating whether i should stop."

7.0: This is where the sentiment generally shifts to positive, but with caveats. Users often describe a "learning curve" or initial side effects that diminished over time. The reviews tend to emphasize the drug's effectiveness in managing the condition, while still acknowledging some lingering drawbacks or individual concerns. There's more optimism and a sense of finding a workable solution. Key phrases: "helped me a lot," "pros definitely outweigh the cons," "worked great for the first 2 years, but...," "worked really well [but had side effects]."

8.0: More consistently positive. Users report good results and often express satisfaction with the medication. Side effects are either minimal, manageable, or considered worth enduring for the benefits. There's a sense of finding a good balance and a willingness to continue treatment. Phrases like "worked miracles," "feel so much better," "life saver," and "good cushion for my knees" appear.

9.0: Strongly positive, with users often describing significant improvements and a high level of satisfaction. Side effects are mentioned less frequently, and when they are, they're typically described as minor or temporary. There's a clear endorsement of the medication.

10.0: Almost universally positive, with users expressing great satisfaction and often describing the medication as life-changing or highly effective. Side effects are rarely mentioned, and if they are, they are downplayed or considered insignificant compared to the benefits.

Conclusion:

Based on this more detailed analysis, the sentiment threshold is still around the 6.0 to 7.0 range, the sentiment leans to be more positive closer to 7.0.

Negative Sentiment: Ratings 1.0 to 6.0

Positive Sentiment: Ratings 7.0 to 10.0

The key difference is the increased nuance we see in the 4.0, 5.0, and 6.0 ratings. These are not clearly negative in the same way as the 1.0-3.0 ratings, but they represent a mixed bag of experiences where the negative aspects often outweigh or significantly detract from the positive ones. The 7.0 rating represents the point where the balance generally tips towards a positive overall experience, despite potential drawbacks.
°°°"""
# |%%--%%| <7XcbZlmLlb|t4AJVc3U6x>

# Create histogram with two class distributions
plt.figure(figsize=(12, 8), dpi=400)

# Define the bins for negative (1-6) and positive (7-10) classes
bins = [0, 6.5, 10]  # Using 6.5 as the boundary to properly separate 6 and 7
counts, bins, patches = plt.hist(df['rating'], bins=bins, edgecolor='black')
total = len(df['rating'])

# Color the bars differently
patches[0].set_facecolor('salmon')  # Negative class (1-6)
patches[1].set_facecolor('lightgreen')  # Positive class (7-10)

# Calculate maximum y value needed
max_count = max(counts)
y_margin = max_count * 0.15  # Add 15% margin for labels

# Set y-axis limit
plt.ylim(0, max_count + y_margin)


# Add percentage labels on top of each bar
for i in range(len(counts)):
    percentage = (counts[i]/total) * 100
    plt.text(bins[i], counts[i], 
             f'{counts[i]:,.0f}\n({percentage:.1f}%)', 
             va='bottom')

plt.title('Distribution of Drug Ratings by Class (Negative: 1-6, Positive: 7-10)')
plt.xlabel('Rating Classes')
plt.ylabel('Number of Reviews')
plt.grid(True, alpha=0.3)

# Customize x-axis labels
plt.xticks([3.25, 8.25], ['Negative\n(1-6)', 'Positive\n(7-10)'])

plt.tight_layout()
plt.show()

# Print detailed statistics
print("\n--- Class Distribution Statistics ---")
print(f"Total reviews: {total:,}")
print(f"Negative class (1-6): {counts[0]:,.0f} reviews ({(counts[0]/total)*100:.1f}%)")
print(f"Positive class (7-10): {counts[1]:,.0f} reviews ({(counts[1]/total)*100:.1f}%)")


# |%%--%%| <t4AJVc3U6x|coTRTZocbq>
r"""°°°
1. **Purpose**: This histogram shows how the drug reviews are distributed when split into two classes based on ratings:
   - Negative class: Ratings from 1-6
   - Positive class: Ratings from 7-10

2. **Visual Elements**:
   - Red (salmon) bar: Represents negative reviews (ratings 1-6)
   - Green bar: Represents positive reviews (ratings 7-10)
   - Each bar shows both the count and percentage of reviews

3. **Key Findings**:
   - Total Dataset Size: 110,811 reviews
   - Negative Reviews (1-6): 37,173 reviews (33.5%)
   - Positive Reviews (7-10): 73,638 reviews (66.5%)

4. **Interpretation**:
   - The data is imbalanced, with about twice as many positive reviews as negative ones
   - Roughly 2/3 of all reviews are positive (7-10 rating)
   - Only 1/3 of reviews are negative (1-6 rating)

5. **Implications**:
   - This imbalance suggests people are more likely to leave positive reviews
   - We might need to consider techniques to handle class imbalance (like oversampling, undersampling, or class weights)
°°°"""
# |%%--%%| <coTRTZocbq|IvUGW91IrO>
r"""°°°
## Data pre-processing

The dataset has been mentioned to be clean. Therefore, we just need to remove the row index. This has to be done during testing as well.
°°°"""
# |%%--%%| <IvUGW91IrO|qtAtC7peva>
r"""°°°
## Feature Engineering
   - Create text-based features from reviews (length, sentiment scores, etc.)
   - Handle date features (extract year, month, etc.)
   - Encode categorical variables (drugName, condition)
°°°"""
# |%%--%%| <qtAtC7peva|jrdDZpUb5E>

print("\n=== Starting Data Preprocessing ===")

# Remove Unnamed column if it exists
if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

# Create binary sentiment labels (0 for ratings 1-6, 1 for ratings 7-10)
print("\nCreating sentiment labels...")
df["sentiment_label"] = (df["rating"] >= 7).astype(int)

# For DistilBERT, we'll just use the raw review text
# No need for text preprocessing as the model's tokenizer will handle it

print("\n=== Basic Preprocessing Complete ===")
print(f"Total reviews: {len(df):,}")
print(f"Positive reviews (rating >= 7): {df['sentiment_label'].sum():,}")
print(f"Negative reviews (rating < 7): {len(df) - df['sentiment_label'].sum():,}")

# |%%--%%| <jrdDZpUb5E|XGKuw0dchh>
r"""°°°
## Data Splitting
   - Split data into training and validation sets
   - Consider temporal splits given the date column
   - Ensure balanced distribution of classes

°°°"""
# |%%--%%| <XGKuw0dchh|948W3nSGG8>

# Convert date column to datetime
print("\n=== Preparing Data Split ===")
df["date"] = pd.to_datetime(df["date"])

# Sort by date
df = df.sort_values("date")

# Calculate split points (80% train, 10% val, 10% test)
train_end_idx = int(len(df) * 0.8)
val_end_idx = int(len(df) * 0.9)

# Split the data
train_df = df[:train_end_idx]
val_df = df[train_end_idx:val_end_idx]
test_df = df[val_end_idx:]

# Print split statistics
print("\n=== Data Split Statistics ===")
print(f"Training set: {len(train_df):,} reviews")
print( f"  Positive: {train_df['sentiment_label'].sum():,} ({train_df['sentiment_label'].mean()*100:.1f}%)")
print( f"  Negative: {len(train_df) - train_df['sentiment_label'].sum():,} ({(1-train_df['sentiment_label'].mean())*100:.1f}%)")
print( f"  Date range: {train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')}")

print(f"\nValidation set: {len(val_df):,} reviews")
print( f"  Positive: {val_df['sentiment_label'].sum():,} ({val_df['sentiment_label'].mean()*100:.1f}%)")
print( f"  Negative: {len(val_df) - val_df['sentiment_label'].sum():,} ({(1-val_df['sentiment_label'].mean())*100:.1f}%)")
print( f"  Date range: {val_df['date'].min().strftime('%Y-%m-%d')} to {val_df['date'].max().strftime('%Y-%m-%d')}")

print(f"\nTest set: {len(test_df):,} reviews")
print( f"  Positive: {test_df['sentiment_label'].sum():,} ({test_df['sentiment_label'].mean()*100:.1f}%)")
print( f"  Negative: {len(test_df) - test_df['sentiment_label'].sum():,} ({(1-test_df['sentiment_label'].mean())*100:.1f}%)")
print( f"  Date range: {test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')}")

# |%%--%%| <948W3nSGG8|n7Wir1cWm6>
r"""°°°
## Model Development
### Direct Training approach
- Define target variable (likely rating)
- Build and train deep learning models
- Evaluate validation data

We'll train two models to compare: one trained from scratch, and another transfer learning.
°°°"""
# |%%--%%| <n7Wir1cWm6|AeVrD5hAEW>

# Can't fully reuse part 1 sadly.

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def train_model(
    model,
    dataloaders,
    criterion,
    optimizer,
    scheduler,
    num_epochs=NUM_EPOCH,
    model_name="Model",
    class_weights=None,
    early_stopping_patience=3,
    gradient_accumulation_steps=1,
):
    """
    Train a model for sentiment classification with advanced training features.

    Args:
        model: The neural network model
        dataloaders: Dictionary containing 'train' and 'valid' dataloaders
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        model_name: Name for saving the model
        class_weights: Weights for handling class imbalance
        early_stopping_patience: Number of epochs to wait before early stopping
        gradient_accumulation_steps: Number of steps to accumulate gradients
    """
    # Initialize tracking variables
    best_val_f1 = 0.0
    patience_counter = 0
    train_metrics = {"loss": [], "acc": [], "f1": []}
    val_metrics = {"loss": [], "acc": [], "f1": []}

    # Move model and criterion to device
    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        optimizer.zero_grad(set_to_none=True)

        train_pbar = tqdm(
            dataloaders["train"], desc="Training", position=1, leave=False
        )

        for batch_idx, batch in enumerate(train_pbar):
            # Move data to device
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss = loss / gradient_accumulation_steps  # Scale loss

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Track metrics
            running_loss += loss.item() * gradient_accumulation_steps
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            train_pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                }
            )

        # Calculate epoch metrics
        epoch_loss = running_loss / len(dataloaders["train"])
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average="weighted")

        train_metrics["loss"].append(epoch_loss)
        train_metrics["acc"].append(epoch_acc)
        train_metrics["f1"].append(epoch_f1)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            val_pbar = tqdm(
                dataloaders["valid"], desc="Validation", position=1, leave=False
            )

            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        # Calculate validation metrics
        val_loss = val_loss / len(dataloaders["valid"])
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_f1 = f1_score(all_val_labels, all_val_preds, average="weighted")

        val_metrics["loss"].append(val_loss)
        val_metrics["acc"].append(val_acc)
        val_metrics["f1"].append(val_f1)

        # Log metrics
        logger.info(
            f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}"
        )
        logger.info(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Model checkpointing based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            logger.info("Saving best model checkpoint...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_f1": best_val_f1,
                },
                f"best_{model_name}.pth",
            )
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Plot training history
    plot_training_history(train_metrics, val_metrics, model_name)

    return train_metrics, val_metrics


def plot_training_history(train_metrics, val_metrics, model_name):
    """Plot training and validation metrics history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot loss
    axes[0].plot(train_metrics["loss"], label="Train")
    axes[0].plot(val_metrics["loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Plot accuracy
    axes[1].plot(train_metrics["acc"], label="Train")
    axes[1].plot(val_metrics["acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    # Plot F1 score
    axes[2].plot(train_metrics["f1"], label="Train")
    axes[2].plot(val_metrics["f1"], label="Val")
    axes[2].set_title("F1 Score")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.suptitle(f"Training History - {model_name}")
    plt.tight_layout()
    plt.show()

# |%%--%%| <AeVrD5hAEW|XSxJJW08Wm>

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=2):
        super(SentimentClassifier, self).__init__()
        self.minilm = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/MiniLM-L12-H384-uncased",
            num_labels=n_classes
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.minilm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits

# |%%--%%| <XSxJJW08Wm|IrGMc21Ppj>

# Initialize training components
def initialize_training(train_df, val_df, batch_size=16, max_length=64):
    """
    Initialize all components needed for training
    Args:
        train_df (DataFrame): Training dataframe
        val_df (DataFrame): Validation dataframe
        batch_size (int): Batch size for training
        max_length (int): Maximum sequence length
    """
    # Set device
    print(f"Using device: {device}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")

    # Create datasets
    train_dataset = DrugReviewDataset(
        train_df["review"].values,
        train_df["sentiment_label"].values,
        tokenizer,
        max_length,
    )

    val_dataset = DrugReviewDataset(
        val_df["review"].values, val_df["sentiment_label"].values, tokenizer, max_length
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Calculate class weights for imbalanced dataset
    total_samples = len(train_df)
    neg_samples = (train_df["sentiment_label"] == 0).sum()
    pos_samples = (train_df["sentiment_label"] == 1).sum()

    class_weights = torch.tensor(
        [total_samples / (2 * neg_samples), total_samples / (2 * pos_samples)],
                dtype=torch.float32,
    ).to(device)

    # Initialize model
    model = SentimentClassifier().to(device)

    # Initialize optimizer with weight decay
    optimizer = AdamW(
        model.parameters(), lr=2e-5, weight_decay=0.01
    )

    # Initialize scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2
    )

    # Initialize loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    return {
        "device": device,
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "criterion": criterion,
        "class_weights": class_weights,
    }


training_components = initialize_training(train_df, val_df)

# Create dataloaders dictionary for the train_model function
dataloaders = {
    "train": training_components["train_loader"],
    "valid": training_components["val_loader"],
}

# Train the model
train_metrics, val_metrics = train_model(
    model=training_components["model"],
    dataloaders=dataloaders,
    criterion=training_components["criterion"],
    optimizer=training_components["optimizer"],
    scheduler=training_components["scheduler"],
    model_name="DrugReviewSentiment",
    class_weights=training_components["class_weights"],
    early_stopping_patience=3,
    gradient_accumulation_steps=4,
)

# |%%--%%| <IrGMc21Ppj|OO4yFFj1Lm>

class SentimentClassifierFromScratch(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256, n_layers=2, dropout=0.5):
        super(SentimentClassifierFromScratch, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim  # Save hidden_dim as instance variable
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 2)
        
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        # Move tensors to the same device as the model
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # Get embeddings
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(2 * 2, batch_size, self.hidden_dim).to(device)  # 2 for bidirectional
        c0 = torch.zeros(2 * 2, batch_size, self.hidden_dim).to(device)
        
        # Pass through LSTM with initial states
        lstm_out, (hidden, cell) = self.lstm(embedded, (h0, c0))
        
        # Get final hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Pass through fully connected layers
        dense1 = self.dropout(self.relu(self.fc1(hidden)))
        output = self.fc2(dense1)
        
        return output

# |%%--%%| <OO4yFFj1Lm|ZlaRreUqUT>

# Vocabulary builder
def build_vocabulary(texts, min_freq=5):
    """
    Build vocabulary from texts
    
    Args:
        texts: List of text strings
        min_freq: Minimum frequency for a word to be included
    """
    word_freq = Counter()
    for text in texts:
        word_freq.update(text.split())
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab

# Training function
def train_from_scratch(model, train_loader, val_loader, criterion, optimizer, n_epochs=NUM_EPOCH, device='mps'):
    """
    Train the from-scratch model
    """
    best_val_loss = float('inf')
    model = model.to(device)  # Ensure model is on correct device
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for batch in train_pbar:
            # Ensure tensors are on correct device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            try:
                # Forward pass
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except RuntimeError as e:
                print(f"Error during training: {e}")
                print(f"Input shapes - ids: {input_ids.shape}, mask: {attention_mask.shape}")
                raise e
        
        # Calculate training metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, F1: {epoch_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'vocab_size': len(vocab),
                'vocab': vocab  # Save the vocabulary
            }, 'best_scratch_model.pth')



# Build vocabulary
vocab = build_vocabulary(train_df['review'])

# Create datasets
train_dataset = TextDatasetFromScratch(
    train_df['review'].values,
    train_df['sentiment_label'].values,
    vocab
)

# Create model
model = SentimentClassifierFromScratch(
    vocab_size=len(vocab),
    embedding_dim=100,
    hidden_dim=256
)

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

train_from_scratch(model, training_components["train_loader"], training_components["val_loader"], criterion, optimizer)

# |%%--%%| <ZlaRreUqUT|TKWatTHGFQ>
r"""°°°
### Indirect Training approach
- Define indirect signals/proxies
- Build and train models
- Compare with direct training resultss

let's use the popular sentiment classification dataset: IMDB movie reviews.

This is chosen because its well cleaned and well labelled.
°°°"""
# |%%--%%| <TKWatTHGFQ|SEnQdLcLWW>

def prepare_indirect_training(batch_size=16, max_length=64):
    """
    Prepare indirect training components using IMDB dataset
    """
    # Load IMDB dataset
    logging.info("Loading IMDB dataset...")
    imdb_dataset = load_dataset("imdb")
    
    # Initialize tokenizer (using the same model as direct training)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
    
    # Prepare training data
    train_dataset = IndirectDataset(
        imdb_dataset["train"]["text"],
        imdb_dataset["train"]["label"],
        tokenizer,
        max_length
    )
    
    # Prepare validation data
    val_dataset = IndirectDataset(
        imdb_dataset["test"]["text"],  # Using IMDB test set as validation
        imdb_dataset["test"]["label"],
        tokenizer,
        max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "tokenizer": tokenizer
    }

def train_indirect_model(
    model,
    train_loader,
    val_loader,
    num_epochs=NUM_EPOCH,
    learning_rate=2e-5,
    device='mps'
):
    """
    Train model on indirect data (IMDB)
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    best_val_f1 = 0
    train_metrics = {"loss": [], "acc": [], "f1": []}
    val_metrics = {"loss": [], "acc": [], "f1": []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in train_pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate training metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        train_metrics["loss"].append(epoch_loss)
        train_metrics["acc"].append(epoch_acc)
        train_metrics["f1"].append(epoch_f1)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        val_metrics["loss"].append(val_loss)
        val_metrics["acc"].append(val_acc)
        val_metrics["f1"].append(val_f1)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_indirect_model.pth')
        
        logging.info(f"Epoch {epoch+1}/{num_epochs}")
        logging.info(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, F1: {epoch_f1:.4f}")
        logging.info(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
    
    return model, train_metrics, val_metrics

def evaluate_on_drug_reviews(model, test_loader, device='mps'):
    """
    Evaluate direct or indirect-trained model on drug review test set
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing on drug reviews"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    results = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='weighted'),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted')
    }
    
    return results

# |%%--%%| <SEnQdLcLWW|63eRA0dpgi>

# Prepare indirect training components
indirect_components = prepare_indirect_training()

# Initialize model
model = SentimentClassifier()  # Your model class

# Train on indirect data (IMDB)
model, train_metrics, val_metrics = train_indirect_model(
    model,
    indirect_components["train_loader"],
    indirect_components["val_loader"]
)


# |%%--%%| <63eRA0dpgi|LSh3DiAyzu>

# Create test dataset using DrugReviewDataset
test_dataset = DrugReviewDataset(
    reviews=test_df["review"].values,  # Using 'review' column
    labels=test_df["sentiment_label"].values,  # Using 'sentiment_label' column
    tokenizer=indirect_components["tokenizer"],
    max_length=MAX_LENGTH
)

# Create test dataloader
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Evaluate on drug review test set
test_results = evaluate_on_drug_reviews(model, test_loader)

# |%%--%%| <LSh3DiAyzu|UQizDaMUYs>

# Print results
print("\nTest Results on Drug Reviews:")
for metric, value in test_results.items():
    print(f"{metric}: {value}")

# |%%--%%| <UQizDaMUYs|F1K0RaXvbF>

def prepare_indirect_scratch_training(batch_size=16, max_length=MAX_LENGTH):
    """
    Prepare indirect training components using IMDB dataset
    """
    # Load IMDB dataset
    print("Loading IMDB dataset...")
    imdb_dataset = load_dataset("imdb")
    
    # Initialize tokenizer for vocabulary building
    tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
    
    # Build vocabulary from IMDB training data
    vocab = build_vocabulary([text for text in imdb_dataset["train"]["text"]], min_freq=5)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    train_dataset = TextDatasetFromScratch(
        texts=imdb_dataset["train"]["text"],
        labels=imdb_dataset["train"]["label"],
        vocab=vocab,
        max_length=max_length
    )
    
    val_dataset = TextDatasetFromScratch(
        texts=imdb_dataset["test"]["text"],
        labels=imdb_dataset["test"]["label"],
        vocab=vocab,
        max_length=max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "vocab": vocab
    }

def train_indirect_scratch(model, train_loader, val_loader, criterion, optimizer, n_epochs=NUM_EPOCH, device='mps'):
    """
    Train the from-scratch model on indirect data (IMDB)
    """
    best_val_loss = float('inf')
    model = model.to(device)
    
    train_metrics = {"loss": [], "acc": [], "f1": []}
    val_metrics = {"loss": [], "acc": [], "f1": []}
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for batch in train_pbar:
            # Get batch data
            input_ids = batch['text'].to(device)  # Note: using 'text' instead of 'input_ids'
            lengths = batch['length'].to(device)
            labels = batch['label'].to(device)  # Note: using 'label' instead of 'labels'
            
            optimizer.zero_grad()
            
            try:
                # Forward pass
                outputs = model(input_ids, lengths)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except RuntimeError as e:
                print(f"Error during training: {e}")
                print(f"Input shapes - ids: {input_ids.shape}, lengths: {lengths.shape}")
                raise e
        
        # Calculate training metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        train_metrics["loss"].append(epoch_loss)
        train_metrics["acc"].append(epoch_acc)
        train_metrics["f1"].append(epoch_f1)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['text'].to(device)
                lengths = batch['length'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, lengths)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        val_metrics["loss"].append(val_loss)
        val_metrics["acc"].append(val_acc)
        val_metrics["f1"].append(val_f1)
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, F1: {epoch_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_scratch_indirect_model.pth')
    
    return model, train_metrics, val_metrics

def run_indirect_scratch_experiment():
    """
    Run complete indirect training experiment with from-scratch model
    """
    # Prepare indirect training components
    print("Preparing training components...")
    components = prepare_indirect_scratch_training()
    
    # Initialize model and training components
    model = SentimentClassifierFromScratch(
        vocab_size=len(components["vocab"]),
        embedding_dim=100,
        hidden_dim=256
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    print("\nStarting training...")
    model, train_metrics, val_metrics = train_indirect_scratch(
        model=model,
        train_loader=components["train_loader"],
        val_loader=components["val_loader"],
        criterion=criterion,
        optimizer=optimizer
    )
    
    return model, components["vocab"], train_metrics, val_metrics

# Run the experiment
model, vocab, train_metrics, val_metrics = run_indirect_scratch_experiment()

# Plot training history
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(train_metrics['loss'], label='Train')
plt.plot(val_metrics['loss'], label='Val')
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(train_metrics['acc'], label='Train')
plt.plot(val_metrics['acc'], label='Val')
plt.title('Accuracy History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# |%%--%%| <F1K0RaXvbF|PWnNOItDVD>
r"""°°°
## Model Comparison
   - Compare direct vs indirect training approaches
   - Analyze pros and cons of each method
   - Discuss real-world applicability
°°°"""
# |%%--%%| <PWnNOItDVD|B0nZze7HN9>

def load_model_state(path):
    """Helper function to load model state regardless of saving format"""
    checkpoint = torch.load(path)
    if isinstance(checkpoint, dict):
        # If checkpoint is a dictionary with multiple keys
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        else:
            # If it's a dictionary but just the state dict
            return checkpoint
    else:
        # If it's directly the state dict
        return checkpoint

# Load both models
direct_model = SentimentClassifier()
indirect_model = SentimentClassifier()

# Load saved states with flexible loading
try:
    direct_state = load_model_state('best_DrugReviewSentiment.pth')
    indirect_state = load_model_state('best_indirect_model.pth')
    
    direct_model.load_state_dict(direct_state)
    indirect_model.load_state_dict(indirect_state)
except Exception as e:
    print(f"Error loading models: {e}")
    # Print the structure of the saved files for debugging
    print("\nDirect model checkpoint structure:", torch.load('best_DrugReviewSentiment.pth').keys())
    print("\nIndirect model checkpoint structure:", torch.load('best_indirect_model.pth').keys())
    raise

# Move to device
direct_model = direct_model.to(device)
indirect_model = indirect_model.to(device)

# Set to evaluation mode
direct_model.eval()
indirect_model.eval()

# Evaluate both models
print("\nEvaluating Direct Training Model:")
direct_results = evaluate_on_drug_reviews(direct_model, test_loader, device)

print("\nEvaluating Indirect Training Model:")
indirect_results = evaluate_on_drug_reviews(indirect_model, test_loader, device)

# Print comparison
print("\n=== Model Performance Comparison ===")
metrics = ['accuracy', 'f1', 'precision', 'recall']

print("\n{:<10} {:<12} {:<12}".format("Metric", "Direct", "Indirect"))
print("-" * 34)
for metric in metrics:
    print("{:<10} {:<12.4f} {:<12.4f}".format(
        metric,
        direct_results[metric],
        indirect_results[metric]
    ))


# |%%--%%| <B0nZze7HN9|NRDp0EhkBv>

# Load the saved model and print its structure
checkpoint = torch.load('best_DrugReviewSentiment.pth')

# Print model state dict structure
print("Model state dict keys:")
for key, value in checkpoint['model_state_dict'].items():
    print(f"{key}: Shape {value.shape}")

# Print optimizer state
print("\nOptimizer state dict keys:", checkpoint['optimizer_state_dict'].keys())

# Print other metadata
print("\nMetadata:")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Best Val F1: {checkpoint['best_val_f1']}")

# For the DrugReviewDataset issue, let's try recreating the test loader:
test_dataset = DrugReviewDataset(
    reviews=test_df["review"].values,
    labels=test_df["sentiment_label"].values,
    tokenizer=AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased"),
    max_length=MAX_LENGTH
)

# Create a new test loader with num_workers=0 to avoid pickling issues
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,  # Set to 0 to avoid pickling issues
    pin_memory=True
)


# |%%--%%| <NRDp0EhkBv|mYrCL1F1lY>

def compare_all_models(test_loader, vocab_size, device='mps'):
    """
    Compare all trained models on the drug review test set
    """
    # Load all models
    models = {
        'Direct Transformer': {
            'model': SentimentClassifier(),
            'path': 'best_DrugReviewSentiment.pth',
            'state_dict_key': 'model_state_dict'  # Uses nested state dict
        },
        'Indirect Transformer': {
            'model': SentimentClassifier(),
            'path': 'best_indirect_model.pth',
            'state_dict_key': None  # Direct state dict
        },
        'Direct Scratch': {
            'model': SentimentClassifierFromScratch(vocab_size=vocab_size),  # Use correct vocab size
            'path': 'best_scratch_model.pth',
            'state_dict_key': 'model_state_dict'
        },
        'Indirect Scratch': {
            'model': SentimentClassifierFromScratch(vocab_size=vocab_size),
            'path': 'best_scratch_indirect_model.pth',
            'state_dict_key': 'model_state_dict'
        }
    }
    
    results = {}
    
    # Evaluate each model
    for name, model_info in models.items():
        print(f"\nEvaluating {name}...")
        
        try:
            # Load model state
            checkpoint = torch.load(model_info['path'])
            
            # Handle different saving formats
            if model_info['state_dict_key'] is not None:
                # Nested state dict
                state_dict = checkpoint[model_info['state_dict_key']]
            else:
                # Direct state dict
                state_dict = checkpoint
            
            # Load state dict
            try:
                model_info['model'].load_state_dict(state_dict)
            except RuntimeError as e:
                print(f"Error loading state dict for {name}: {e}")
                print(f"Model vocab size: {vocab_size}")
                print(f"Checkpoint embedding size: {next(iter(state_dict.values())).shape}")
                continue
            
            # Move to device and evaluate
            model_info['model'].to(device)
            model_info['model'].eval()
            
            # Get predictions
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Testing {name}"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model_info['model'](input_ids, attention_mask)
                    preds = torch.argmax(outputs, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            results[name] = {
                'accuracy': accuracy_score(all_labels, all_preds),
                'f1': f1_score(all_labels, all_preds, average='weighted'),
                'precision': precision_score(all_labels, all_preds, average='weighted'),
                'recall': recall_score(all_labels, all_preds, average='weighted')
            }
            
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            results[name] = None
    
    # Print comparison table
    print("\n=== Model Performance Comparison ===")
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    
    print("\n{:<20} {:<12} {:<12} {:<12} {:<12}".format(
        "Model", "Accuracy", "F1 Score", "Precision", "Recall"
    ))
    print("-" * 68)
    
    for model_name, model_results in results.items():
        if model_results is not None:
            print("{:<20} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
                model_name,
                model_results['accuracy'],
                model_results['f1'],
                model_results['precision'],
                model_results['recall']
            ))
    
    return results


# Usage:
# First, get the vocabulary size from your training data
vocab_size = len(vocab)  # Make sure this matches your training vocab

# Then run comparison
results = compare_all_models(test_loader, vocab_size=vocab_size)


# |%%--%%| <mYrCL1F1lY|p1ixQKWyg5>
r"""°°°
Key Findings:
1. **Performance Gap**: 
   - Direct training outperforms indirect training by about 12 percentage points (81.66% vs 69.61%)
   - This is expected as direct training learns from domain-specific data

2. **Domain Transfer Challenges**:
   - The indirect model (trained on movie reviews) still achieves ~70% accuracy on drug reviews
   - This shows sentiment analysis skills do transfer across domains, but with significant performance loss

3. **Precision vs Recall**:
   - Both models maintain similar precision and recall values
   - The indirect model has higher precision (0.7328) than recall (0.6961), suggesting it's more conservative in making positive predictions

Implications:
1. **Domain Specificity Matters**:
   - Drug review sentiment has unique characteristics that general sentiment models miss
   - Medical terminology and context are important for accurate prediction

2. **Transfer Learning Limitations**:
   - While sentiment analysis skills transfer across domains, there's a significant performance cost
   - The ~70% accuracy of indirect training suggests it learns useful general sentiment features

3. **Practical Applications**:
   - If you have domain-specific data, direct training is clearly superior
   - Indirect training could be useful for:
     * Initial model deployment before collecting domain data
     * Low-resource scenarios where drug review data isn't available
     * Pre-training before fine-tuning on drug reviews


°°°"""
# |%%--%%| <p1ixQKWyg5|v2JY1N6ccC>
r"""°°°
## Results and Discussion
   - Present key findings
   - Discuss limitations
   - Suggest improvements
°°°"""