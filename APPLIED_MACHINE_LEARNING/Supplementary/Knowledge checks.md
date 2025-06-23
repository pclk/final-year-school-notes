1. In a scenario where false positives are extremely costly but false negatives are relatively acceptable, would you prioritize precision or recall? Why?
In this question, the scenario prefers going towards negative because we can and should make more negative assumptions.
Precision means among all those which we predicted positive, which ones were correct?
Precision = TP/TP+FP
Recall means among all actual positive cases, which ones are correctly identified as positive?
Recall = TP/TP+FN
As such, I will prioritize precision, because I want all of my predictions to be positive, since we're minimizing false positives. I will not prioritize Recall because it would increase if I predict more positives as it doesn't consider cases which were false positive, leading to a metric that does not align with my goals.


2. If you have a model with 99.9% accuracy on a dataset where the positive class appears only 0.1% of the time, what might be wrong with relying on accuracy?
The model could predict the majority class 100% of the time, disregarding the positive class altogether, and achieve 99.9% accuracy. This is bad in scenarios where imbalanced data is common, like spam and fraud detection.

3. When would a model with 60% accuracy potentially be more valuable than one with 95% accuracy?
When the data is heavily imbalanced, and the model with 60% accuracy has lesser false predictions of the minority class, the model with 60% accuracy would be more valuable at identifying the minority class.

4. In the "Boy Who Cried Wolf" scenario, what would be the consequences of optimizing solely for precision? For recall?
Optimizing for precision will result in a healthier relationship, because as FP is minimized, the villagers would trust the shepherd more. However, there will be times where the sheeps are eaten because the shepherd isn't that sure whether the wolf is present and would rather not disturb the villager than risk his sheeps being eaten.
Optimizing for recall will result in the villagers ignoring the boy, because a full optimization for recall will result mean the threshold is all the way down, where you call (positive) for the villagers on every occasion. This will result in all the sheeps being eaten because the villagers will not bother helping even if the boy asks for it.

5. If increasing the classification threshold improves precision but decreases recall, how do you determine the optimal threshold?
You need to adjust the classification threshold such that the F1 score, which is the harmonic mean of precesion and recall, is the highest. 

6. Why might ROC curves be misleading for highly imbalanced datasets?
ROC curves look better than it should for imbalanced datasets. This is because of their axis, TPR and FPR, is blah blah more ... im not sure, let's expand on these. But for Precision Recall curves, since the formula covers TP, FP, FN, they provide a more holistic view of the dataset where if the dataset's majority class is positive, TP and FP would increase while FN will decrease. And the changes will cancel each other out.

7. In a medical diagnosis system, why might you want different thresholds for different diseases?
Some diseases are less serious than others. thresholds should be higher the less serious the disease is. non-serious disease can have lower thresholds because their effects on the individual is not as serious, where the cost of the disease not being spotted is not as high as admitting and providing for more people falsely identified as having the disease.

8. How could a model achieve high precision but still be practically useless?
High precision means that when the model predicts positive, they're precise in their prediction. However, they can prioritize precision so much that their recall is so bad, that they miss out on almost all of the actual positive cases. It's like predicting that the amount of years it would take for AI to be smarter than humans, is at most 1000 years. Sure, you're precise and probably right, but its not a very useful prediction as its not insightful. Similar to high precision low recall models, their precise prediction means that only very obviously positive cases are predicted.

9. In what scenario would you prefer a model with 70% precision and 70% recall over one with 90% precision and 40% recall?
similar to previous question

10. Using the roulette example (4% accuracy), explain why traditional accuracy metrics might need context for proper interpretation.
Roulette has very high negative and very low positive. However, if you score it, there is very high returns, where the cost of each chance to play the roulette is far lesser than the prize of the roulette. Thus, you have a very valuable model if your accuracy is good enough such that the cost * number of plays <= prize.

11. If you have two models, one with higher precision and one with higher recall, what business factors would influence your choice between them?
It mostly comes down to the cost of false positives. If cost of false positive is high, and you need to be more precise when you identify a positive case, go with precision. If cost of false positive is low, and you can afford to trade some false positive predictions for generally better accuracy, go with recall.

12. Why might a model with perfect recall be problematic in real-world applications?
Recall will always increase when the number of positive predictions increase, when you have higher TP, whilst disregarding FP. This means that a perfect recall model may just predict every case as positive, making the model quite useless, or not insightful.

13. In the cancer detection case study (Case study 2), why might the high accuracy be misleading?
cancer has very high negative and very low positive. Similar to the roulette example, except that the cost of not providing for cancerous patients will be relatively higher than if they falsely classify some individuals to have cancer. Having high accuracy however, means that you could predict all of negatives, but still reach over 90% accuracy. Despite the nice accuracy, the amount of untreated cancer patients will definitely be frightening.

14. How would you adjust your model if false positives were twice as costly as false negatives?
increase precision decrease recall. Do so by shifting threshold up, aka make model predict less positives.

15. When would you choose to optimize for F1 score instead of accuracy?
When dataset is imbalanced.

16. In a spam detection system, explain why increasing precision might actually harm user experience in some cases.
higher precision means that the model is more right when it predicts spam, resulting in lower false positives. And generally when focusing on user experience, lower false positives are better, because you don't flag and question innocent users. However, when you strive to improve precision when the precision is very high already, the number of actual spammers that are not correctly identified (false negative), would increase. This will result in more spammers that harm user experience.

17. How could a model have high precision and high recall but still be practically useless?
i have no idea, i thought a model with high precision and recall would be always useful. i guess when the training and testing data is shit?

18. Why might a model's performance metrics look different in production compared to testing, even with the same threshold?
perhaps the training and testing data did not represent the production data well enough.

19. In what scenario would you prefer a model that makes fewer but more confident predictions over one that makes more predictions with less confidence?
a model that makes fewer but more confident predictions is the high precision model, and the ones that makes more predictions with less confidence is the high recall model. I would prefer the high precision model when the cost of false positive predictions are high.

20. Using the robotic chicken example, explain why even 99.99% accuracy might not be good enough, and what metric might be more appropriate.
The cost of getting banged by a car is very high apparently, so the cost of a false positive (where the robot chicken is able to cross the road) is very high too. As such, precision will be the highest importance, to make sure that the robotic chicken can continue crossing the road for long periods of time.


1. "You're building a spam filter for emails. Which would be worse:
   a) Missing some spam emails (they get to inbox)
   b) Marking real emails as spam
   Explain why, using precision and recall concepts!"


2. "Imagine you're playing 'Heads or Tails' and your model predicts 'Heads' 100% of the time. If 'Heads' happens 95% of the time, your accuracy would be 95%! Why is this not as amazing as it sounds?"

3. "Your friend says their model is perfect because it has 99% accuracy. What important question should you ask them first?"

4. "Using the Boy Who Cried Wolf story:
   - What happens if the boy calls 'wolf' every single time he sees anything move (even leaves)?
   - What happens if he only calls 'wolf' when he's 100% sure?
   Which approach is more dangerous and why?"

5. "In a simple example, explain why Netflix might prefer to:
   a) Show you some movies you won't like
   b) Miss showing you some movies you would like
   Use precision and recall to explain!"

6. "You have two heart disease detection models:
   Model A: Catches 90% of heart problems but has many false alarms
   Model B: Only raises alarm when very sure, but misses some cases
   Which would you use in:
   - A general checkup?
   - An emergency room?"

7. "Why is getting 90% accuracy on predicting cat photos different from getting 90% accuracy on predicting rare diseases?"

8. "Explain in simple terms why a model might do great in testing but fail in the real world. Use an everyday example!"

9. "Your model is really good at predicting when people will buy ice cream (95% accurate). Why might this model become less accurate in:
   a) Winter
   b) A different country
   Use this to explain why production results might differ from testing!"

10. "You're building a model to detect if photos contain dogs. Which would you prefer:
    a) A model that only identifies obvious dogs but is almost never wrong
    b) A model that catches most dogs but sometimes mistakes cats for dogs
    Explain using precision and recall!"

These questions are designed to:
- Connect complex concepts to everyday situations
- Focus on practical understanding rather than mathematical formulas
- Test core concepts without requiring deep technical knowledge
- Encourage thinking about real-world applications

Would you like to try answering any of these?
