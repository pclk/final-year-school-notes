# Classification
Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 = (2 * Precision * Recall) / (Precision + Recall)

False Positive Rate (FPR) = FP / (FP + TN)

True Negative Rate (TNR) = TN / (TN + FP)

True Positive Rate (TPR) = Sensitivity = Recall

FPR = 1 - TNR

Create 10 tricky questions where the answers are whether the scenario would lead to higher Recall, Precision, FPR, TNR 

1. In a medical screening test, if we decide to flag more patients as potentially having a disease (lowering the classification threshold), this would primarily increase:
var Positive: have disease
var Negative: don't have disease
var Direction: flag more patients having disease: positives increase, negatives decrease.
TP + FP: increase
TN + FN: decrease
Accuracy = depends on previous results
Precision = decrease since though TP increase, FP increase likely will drop precision
Recall = increase since TP increase and FN decrease
F1 = depends on how much precision and recall
FPR = increase since FP increase while TN decrease
TNR = decrease

2. If we modify our spam detection algorithm to be more strict about what it classifies as spam, requiring multiple spam indicators to be present, this would likely increase:
var Positive = is spam
var Negative = is not spam
var Direction = more strict and means going towards Negative.
Precision = Typically increase since FP will decrease less<!more!> than TP
Recall = Definitely decrease since Positive decrease and Negative up
FPR = Definitely decrease since TN will increase and FP will decrease, 

3. In a fraud detection system, if we focus on reducing false positives by being more conservative in our fraud predictions, this would increase:
+ = fraud yes
- = fraud no
direction = going towards neg
Precision= Typically increase depending on Positive distribution because FP might decrease more than TP
Recall = Definitely decrease because TP down and FN up.
FPR = Definitely decrease since FP down and TN up.
TNR = Definitely increase TN up FP down

4. When dealing with imbalanced classes in fraud detection, if we want to find the best balance between precision and recall, we should optimize for:
F1 score

5. In a quality control system for manufacturing, if we adjust our defect detection algorithm to catch more potential defects, even if some are false alarms, this would increase:
+ = yes defect
- = no defect
direction = more pos
Recall = TP increase FN decrease, definite increase
Precision = Typically decrease since FP probably increase more.
FPR = FP increase TN decrease, definite increase
TNR = definite decrease since TN decrease while FP increase

6. If we have a sentiment analysis model where we want to be very confident about our positive predictions, even if we miss some positive cases, we should optimize for:
direction= move towards negative.
Recall = decrease
Precision = increase
FPR = decrease
TNR = increase

7. In a search engine ranking system, if we want to ensure that the top results are highly relevant, even if we miss some relevant documents, we should focus on:
direction = neg
Recall = decrease
Precision = increase
FPR = decrease
TNR = increase

8. If we modify our model to reduce both false positives and false negatives proportionally while maintaining their ratio, this would primarily improve:
accuracy

9. In a cancer detection system, if we adjust our model to ensure we don't miss any positive cases, even at the cost of more false positives, this would maximize:
direction = positive
Recall = increase
Precision = decrease
FPR = increase
TNR = decrease

10. If we have a content moderation system where we want to minimize false accusations of policy violations while accepting that some violations might slip through, we should optimize for:
direction = false positive go down, positive go down, neg
recall = decrease
precision= increase
FPR = decrease
TNR = increase
