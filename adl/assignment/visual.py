       anger      0.176     0.075     0.105        80
     disgust      0.000     0.000     0.000        42
        fear      0.000     0.000     0.000        54
   happiness      0.268     0.939     0.417       132
  neutrality      0.000     0.000     0.000        58
     sadness      0.154     0.026     0.044        77
    surprise      0.000     0.000     0.000        67

# Define class names
class_names = ['anger', 'disgust', 'fear', 'happiness', 'neutrality', 'sadness', 'surprise']

# Create ideal confusion matrix
ideal_cm = np.zeros((7, 7), dtype=int)
# Set diagonal values to the support counts
support_values = [80, 42, 54, 132, 58, 77, 67]  # from the support column
np.fill_diagonal(ideal_cm, support_values)

# Plot ideal confusion matrix
plt.figure(figsize=(10, 8), dpi=400)
sns.heatmap(ideal_cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title(f"Ideal Confusion Matrix - Perfect Predictions")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
