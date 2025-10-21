import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

# ==================== STEP 8: Load Data ====================
# - Load the preprocessed dataset containing features and labels
# - Separate features (X) from target labels (y)
# - Print shapes to verify correct loading
final_dataset = pd.read_csv('final_dataset_with_labels.csv')
X = final_dataset.drop(columns=['Label'])
y = final_dataset['Label']

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# ==================== STEP 9: Calculate Mutual Information ====================
# - Mutual Information (MI) measures dependency between each feature and the target
# - Higher MI → more informative feature for prediction
# - Calculate MI for all genes (features) and sort by importance
print("\nCalculating mutual information scores...")

mi_scores = mutual_info_classif(X, y, random_state=42)
mi_scores_series = pd.Series(mi_scores, index=X.columns, name="MI Scores")
mi_scores_df = mi_scores_series.to_frame().sort_values(by="MI Scores", ascending=False)

mi_scores_df.to_csv("mi_scores.csv")

# ==================== STEP 10: Visualize MI Scores ====================
# - Create a bar plot of MI scores to understand feature importance distribution
# - Use colors to distinguish between informative (MI > 0) and less informative (MI <= 0) features
plt.figure(figsize=(12, 6))
colors = ['skyblue' if score > 0 else 'salmon' for score in mi_scores_df["MI Scores"]]

plt.bar(range(len(mi_scores_df)), mi_scores_df["MI Scores"], color=colors)
plt.xlabel("Feature Index", weight='bold', fontsize=14)
plt.ylabel("Mutual Information Score", weight='bold', fontsize=14)
plt.title("Mutual Information Scores of Features", weight='bold', fontsize=16)
plt.tight_layout()
plt.savefig("mi_scores_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# ==================== STEP 11: Feature Selection Using Threshold ====================
# - Define a threshold for MI scores
# - Keep only features with MI > threshold
# - Save the final selected feature set for downstream modeling
threshold = 0.0
selected_features = mi_scores_df[mi_scores_df["MI Scores"] > threshold].index.tolist()

print(f"\nNumber of selected features: {len(selected_features)}")
print("Selected Features:")
print(selected_features)

final_selected_df = final_dataset[selected_features + ["Label"]]
final_selected_df.to_csv('final_selected_features.csv', index=False)

print(f"\nFinal selected dataset shape: {final_selected_df.shape}")
print(final_selected_df.head())

print("\nFeature selection pipeline completed successfully (using MI threshold).")
