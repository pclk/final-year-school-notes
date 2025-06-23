Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 = (2 * Precision * Recall) / (Precision + Recall)

False Positive Rate (FPR) = FP / (FP + TN)

True Negative Rate (TNR) = TN / (TN + FP)

True Positive Rate (TPR) = Sensitivity = Recall

FPR = 1 - TNR

Question 1:
Given the confusion matrix:
```
                Predicted Positive    Predicted Negative
Actual Positive        85                   15
Actual Negative        25                   75
```
Calculate:
a) Precision
85 / 85 + 25 
85 / 110
Ans: 77%
b) TNR
75 / 75 + 25
Ans: 75%
c) FPR
25 / 25 + 75
Ans: 25%
d) F1 score
(2 * 0.85 * (0.77))/(0.85 + 0.77)
1.309 / 1.62
Ans: 0.8080246914

Question 2:
Given a highly imbalanced confusion matrix:
```
                Predicted Positive    Predicted Negative
Actual Positive         5                   45
Actual Negative         2                   948
```
Calculate:
a) Accuracy
(5 + 948)/ 50 + 950
953 / 1000
Ans: 95.3%
b) Recall
5/5+2
Ans : 71.4%
c) Precision
5/5+45
5/50
1/10
Ans: 10% 
d) TNR
948/948+2
Ans: 99.8%

Question 3:
Given a confusion matrix with some tricky numbers:
```
                Predicted Positive    Predicted Negative
Actual Positive        199                  1
Actual Negative        99                   1
```
Calculate:
a) FPR
99/99+1
Ans: 99%
b) Recall
199/199+1
Ans: 99.5%
c) F1 score
precision = 199/199+99 = 66.8%
(2 * 0.995 * 0.668)/(0.995 + 0.668)
Ans: 0.7993505713

d) Accuracy
(199 + 1)/300
2/3
Ans: 66.7%
These questions are designed to test understanding of edge cases and working with imbalanced data. Would you like to try solving them?
