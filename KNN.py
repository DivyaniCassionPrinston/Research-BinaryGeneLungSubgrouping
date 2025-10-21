import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
import psutil

# ==================== STEP 1: LOAD SELECTED FEATURES ====================
# Load dataset containing previously selected top features and separate features and labels
print("Loading selected features from feature selection pipeline...")
final_selected_df = pd.read_csv("final_selected_features.csv")

print(f"Dataset shape after feature selection: {final_selected_df.shape}")
print(f"Number of selected features: {final_selected_df.shape[1] - 1}")

X = final_selected_df.drop("Label", axis=1)
y = final_selected_df["Label"]

print(f"\nFeatures shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Class distribution:\n{y.value_counts()}")

# ==================== STEP 2: PCA COMPONENT SELECTION ====================
# Determine optimal number of PCA components using cross-validation
print(f"\n{'=' * 60}")
print("FINDING OPTIMAL PCA COMPONENTS")
print(f"{'=' * 60}\n")

max_possible_components = min(X.shape[0], X.shape[1])
component_range = range(50, min(1600, max_possible_components), 50)

print(f"Testing component range: {list(component_range)}")
print("Running cross-validation for each component number...\n")

best_score = 0
optimal_num_features = None
cv_results = []

component_selection_start = time.time()

for n_comp in component_range:
    # Apply PCA for dimensionality reduction
    pca_temp = PCA(n_components=n_comp)
    X_pca_temp = pca_temp.fit_transform(X)

    # Evaluate KNN using cross-validation
    knn_temp = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(knn_temp, X_pca_temp, y, cv=10, scoring="accuracy", n_jobs=-1)

    mean_score = scores.mean()
    std_score = scores.std()
    cv_results.append((n_comp, mean_score, std_score))

    print(f"  Components: {n_comp:3d} | CV Accuracy: {mean_score:.4f} ± {std_score:.4f}")

    if mean_score > best_score:
        best_score = mean_score
        optimal_num_features = n_comp

component_selection_time = time.time() - component_selection_start

print(f"\n{'=' * 60}")
print(f"Optimal number of components: {optimal_num_features}")
print(f"Best CV accuracy: {best_score:.4f}")
print(f"Component selection completed in: {component_selection_time:.2f}s")
print(f"{'=' * 60}\n")

# ==================== STEP 3: SETUP 10-FOLD CROSS VALIDATION ====================
# Setup StratifiedKFold to preserve class proportions in each fold
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize arrays to store metrics and memory usage
pca_times, training_times = [], []
pca_memory, knn_memory = [], []
fold_accuracies = []
all_y_true, all_y_pred = [], []

print(f"{'=' * 60}")
print(f"Starting {n_splits}-Fold Cross-Validation with {optimal_num_features} Components")
print(f"{'=' * 60}\n")

# ==================== STEP 4: CROSS-VALIDATION LOOP ====================
# Perform fold-wise training, PCA transformation, KNN training, and evaluation
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    print(f"Processing Fold {fold}/{n_splits}...")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # PCA dimensionality reduction
    process = psutil.Process()
    start_time = time.time()
    memory_before = process.memory_info().rss

    pca = PCA(n_components=optimal_num_features)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    memory_after = process.memory_info().rss
    pca_time = time.time() - start_time
    pca_times.append(pca_time)
    pca_memory.append((memory_after - memory_before) / (1024**2))

    explained_variance = np.sum(pca.explained_variance_ratio_)

    # Train KNN classifier
    start_time = time.time()
    memory_before = process.memory_info().rss

    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train_pca, y_train)

    memory_after = process.memory_info().rss
    training_time = time.time() - start_time
    training_times.append(training_time)
    knn_memory.append((memory_after - memory_before) / (1024**2))

    # Evaluate fold
    y_pred = knn_classifier.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracies.append(accuracy)

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    print(f"  PCA: {pca_time:.2f}s | Variance: {explained_variance:.4f}")
    print(f"  KNN Training: {training_time:.2f}s | Accuracy: {accuracy:.4f}")
    print(f"  Memory - PCA: {pca_memory[-1]:.2f}MB | KNN: {knn_memory[-1]:.2f}MB\n")

# ==================== STEP 5: CROSS-VALIDATION SUMMARY ====================
# Display average and range of accuracy, time, and memory usage across folds
print(f"{'=' * 60}")
print("CROSS-VALIDATION SUMMARY")
print(f"{'=' * 60}")

print(f"\nAverage Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"Min Accuracy: {np.min(fold_accuracies):.4f}")
print(f"Max Accuracy: {np.max(fold_accuracies):.4f}")

print(f"\nAverage PCA Time: {np.mean(pca_times):.2f}s ± {np.std(pca_times):.2f}s")
print(f"Average KNN Training Time: {np.mean(training_times):.2f}s ± {np.std(training_times):.2f}s")
print(f"Total Time: {sum(pca_times) + sum(training_times):.2f}s")

print(f"\nAverage PCA Memory: {np.mean(pca_memory):.2f}MB ± {np.std(pca_memory):.2f}MB")
print(f"Average KNN Memory: {np.mean(knn_memory):.2f}MB ± {np.std(knn_memory):.2f}MB")

# ==================== STEP 6: OVERALL MODEL EVALUATION ====================
# Classification report and confusion matrix on all folds combined
print(f"\n{'=' * 60}")
print("OVERALL MODEL EVALUATION (All Folds Combined)")
print(f"{'=' * 60}")

print("\nClassification Report:")
print(classification_report(all_y_true, all_y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(all_y_true, all_y_pred)
print(cm)

# ==================== STEP 7: TRAIN FINAL MODEL ON FULL DATASET ====================
# Fit PCA and KNN on full dataset
print(f"\n{'=' * 60}")
print("TRAINING FINAL MODEL ON FULL DATASET")
print(f"{'=' * 60}\n")

print(f"Applying PCA with {optimal_num_features} components...")
start_time = time.time()
pca_full = PCA(n_components=optimal_num_features)
X_pca_full = pca_full.fit_transform(X)
pca_full_time = time.time() - start_time
explained_variance_full = np.sum(pca_full.explained_variance_ratio_)

print(f"PCA completed in {pca_full_time:.2f} seconds.")
print(f"Cumulative explained variance: {explained_variance_full:.4f}")

print("\nTraining final KNN model...")
start_time = time.time()
final_knn = KNeighborsClassifier(n_neighbors=5)
final_knn.fit(X_pca_full, y)
final_training_time = time.time() - start_time

print(f"KNN training completed in {final_training_time:.2f} seconds.")
final_accuracy = final_knn.score(X_pca_full, y)
print(f"Final Model Training Accuracy: {final_accuracy:.4f}")

# ==================== STEP 8: COMPONENT SELECTION RESULTS ====================
# Print all tested PCA configurations
print(f"\n{'=' * 60}")
print("PCA COMPONENT SELECTION RESULTS")
print(f"{'=' * 60}\n")

print("All tested configurations:")
for n_comp, mean_acc, std_acc in cv_results:
    marker = " ← SELECTED" if n_comp == optimal_num_features else ""
    print(f"  {n_comp:3d} components: {mean_acc:.4f} ± {std_acc:.4f}{marker}")

# ==================== STEP 9: FINAL PIPELINE SUMMARY ====================
# Display total pipeline summary
print(f"\n{'=' * 60}")
print("PIPELINE SUMMARY")
print(f"{'=' * 60}")

print(f"\nTotal Features Selected: {X.shape[1]}")
print(f"PCA Components Used: {optimal_num_features} (CV-optimized)")
print(f"Total Samples: {X.shape[0]}")
print(f"Number of Folds: {n_splits}")

print(f"\nComponent Selection:")
print(f"  - Time Taken: {component_selection_time:.2f}s")
print(f"  - Best CV Accuracy: {best_score:.4f}")

print(f"\nCross-Validation Results:")
print(f"  - Average Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"  - Average Time per Fold: {np.mean(pca_times) + np.mean(training_times):.2f}s")

print(f"\nFinal Model (Full Dataset):")
print(f"  - Training Accuracy: {final_accuracy:.4f}")
print(f"  - Total Training Time: {pca_full_time + final_training_time:.2f}s")

print(f"\nTotal Pipeline Time: {component_selection_time + sum(pca_times) + sum(training_times) + pca_full_time + final_training_time:.2f}s")

print("\n✓ Model training pipeline completed successfully!")
