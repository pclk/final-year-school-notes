import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data preparation
metrics = ["RMSE", "R2"]
openai_values = [51507.76, 0.4403]
claude_values = [52500.23, 0.4186]
gemini_values = [38608.09, 0.6856]
bert_values = [27143.14, 0.8446]


# Create DataFrame
df = pd.DataFrame(
    {
        "Metric": metrics,
        "OpenAI": openai_values,
        "Claude": claude_values,
        "Gemini": gemini_values,
        "BERT": bert_values,
    }
)

# Set style and custom color palette
plt.style.use("seaborn")
custom_palette = [
    "#415866",
    "#4f2422",
    "#8b786e",
    "#F57229",
]  # Steel blue, Dark red, Taupe
sns.set_palette(custom_palette)

# Set figure background color to white for better contrast
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

# Create high resolution figure with multiple subplots
plt.rcParams["figure.dpi"] = 600  # Set high DPI for display
plt.rcParams["figure.figsize"] = [16, 20]  # Larger figure size
plt.rcParams["font.size"] = 20  # Larger default font size
plt.rcParams["axes.linewidth"] = 2  # Thicker axes lines
plt.rcParams["axes.labelsize"] = 20  # Larger axis labels
plt.rcParams["xtick.labelsize"] = 20  # Larger tick labels
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["figure.titlesize"] = 20

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot RMSE
rmse_data = df[df["Metric"] == "RMSE"].melt(
    id_vars=["Metric"], var_name="Method", value_name="Value"
)
sns.barplot(
    data=rmse_data,
    x="Method",
    y="Value",
    ax=ax1,
    palette=custom_palette,
)
ax1.set_title("RMSE Comparison", pad=20, fontsize=20, fontweight="bold")
ax1.set_ylabel("RMSE (lower is better)")
ax1.grid(True, alpha=0.3)
# Add value labels on the bars
for container in ax1.containers:
    ax1.bar_label(container, fmt="%.2f", padding=3)

# Plot R²
r2_data = df[df["Metric"] == "R2"].melt(
    id_vars=["Metric"], var_name="Method", value_name="Value"
)
sns.barplot(
    data=r2_data,
    x="Method",
    y="Value",
    ax=ax2,
    palette=custom_palette,
)
ax2.set_title("R² Score Comparison", pad=20, fontsize=20, fontweight="bold")
ax2.set_ylabel("R² Score (higher is better)")
ax2.grid(True, alpha=0.3)
# Set y-axis limits for R² plot to start from 0
ax2.set_ylim(0, 1)
# Add value labels on the bars
for container in ax2.containers:
    ax2.bar_label(container, fmt="%.4f", padding=3)

# Enhance grid and spines for both plots
for ax in [ax1, ax2]:
    ax.grid(True, linestyle="--", alpha=0.7, linewidth=1.5)
    ax.spines["top"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.set_facecolor("white")
    # Rotate x-axis labels for better readability
    ax.tick_params(axis="x", rotation=45)

# Add overall title
fig.suptitle("Model Performance Metrics", y=1.05, fontsize=24, fontweight="bold")


# Enhance grid and spines
for ax in axes:
    ax.grid(True, linestyle="--", alpha=0.7, linewidth=1.5)
    ax.spines["top"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.set_facecolor("white")

# Adjust layout
plt.tight_layout()
