import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set Times New Roman font globally
plt.rcParams['font.family'] = 'Times New Roman'

# Confusion matrices
models = {
    "Logistic Regression": np.array([[67, 15],[10, 92]]),
    "SVM": np.array([[67, 15],[15, 87]]),
    "Decision Tree": np.array([[65, 17],[18, 84]]),
    "Random Forest": np.array([[71, 11],[10, 92]]),
    "XGBoost": np.array([[72, 10],[17, 85]]),
    "Gradient Boosting": np.array([[71, 11],[15, 87]])
}

# Loop and save each image
for name, cm in models.items():
    plt.figure(figsize=(6,5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                annot_kws={"fontsize": 14, "fontname": "Times New Roman"})
    
    plt.title(f"{name} - Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    
    plt.tight_layout()
    
    # Save image
    filename = name.replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=300)
    plt.close()

print("All images saved successfully ✅")