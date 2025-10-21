# EFFICIENCY OF USING BINARY GENE EXPRESSION DATA IN LUNG CANCER SUBGROUPING

## Overview

This research project evaluates the performance and efficiency of using **binary gene expression data** compared to traditional **continuous gene expression data** for classifying lung cancer patients into specific subgroups. The goal is to develop effective and computationally efficient classification models that can identify key genetic markers associated with different lung cancer types, ultimately aiming to improve diagnostic precision and treatment outcomes.

## 1. Project Information

*   **Supervisor:** Dr. Pratheeba Jeyananthan.
*   **Affiliation:** Department of Computer Engineering, Faculty of Engineering, University of Jaffna.

## 2. Motivation and Objectives

Lung cancer is a leading cause of death worldwide, with Non-Small Cell Lung Cancer (NSCLC) being the most common subgroup. Finding genetic types of lung cancer using gene expression profiling is an effective tool to guide specific therapies.

The motivation for this research is to evaluate if binary gene expression data simplifies the analysis of high-dimensional gene expression data, thereby improving its usability for clinical applications and potentially increasing the identification of new lung cancer subtypes. Binary data reduces gene activity to a binary state (expressed or not expressed), which can help reduce dimensionality and noise.

### Objectives

1.  **Evaluate Binary Gene Expression Data:** Assessing the efficiency of using binary gene expression data for subgrouping lung cancer patients.
2.  **Identify Genetic Markers:** Discovering distinct genetic markers essential for differentiating between various types of lung cancer.
3.  **Develop Accurate Models:** Creating effective classification algorithms based on identified markers to precisely classify lung cancer into its various classifications.
4.  **Compare Accuracy and Efficiency:** Directly comparing the performance (accuracy) and computational efficiency (time and memory usage) between models trained on binary and continuous gene expression data.

### Research Gap Addressed

The research addresses the limitation that **only a limited number of researchers have used Binary Gene Expression Data**. Furthermore, there is a **lack of a direct comparison** between binary and continuous gene expression data in terms of their effectiveness for cancer subgrouping.

## 3. Methodology

The overall methodology follows a sequence of data collection, preprocessing, feature selection, model application, and performance comparison.

### 3.1 Data and Preprocessing

*   **Data Types:** Gene expression RNAseq, miRNA strand expression RNAseq, and Microarray data are used.
*   **Primary Dataset:** **GSE135304** (Non-Small Cell Lung Cancer, Peripheral Immunity, Extending the Tumor Macroenvironment) was selected for implementation.
*   **Preprocessing Steps:** Includes null value elimination, duplicate value elimination, and outlier removal.
*   **Normalization:** Continuous gene expression values were scaled between $$ using Min-Max scaling. Normalization is generally not needed for the binarized scenario.
*   **Adaptive Binarization:** Gene expression data is converted to a simple yes/no format (1 for "expressed" and 0 for "not expressed"). This uses an adaptive threshold set for each gene, often determined by the median expression value among all samples.
*   **Target Encoding:** Cancer subtypes were encoded, for example: 0: Adenocarcinoma, 4: Squamous cell carcinoma.

### 3.2 Feature Selection

Feature selection was used to identify the most informative genes from the high-dimensional data (e.g., from 47,323 features in GSE135304).

*   **Mutual Information (MI):** Measures the dependency between each gene feature and the target variable, where genes with higher MI scores are retained.
*   **Feature Clustering:** Features are clustered based on their MI scores, often using K-means clustering to generate clusters of highly informative features.

### 3.3 Machine Learning Models

Six machine learning models were applied and evaluated on both the continuous and binarized versions of the dataset.

1.  **Support Vector Machines (SVM):** Effective for high-dimensional data. The Tanimoto kernel was utilized for binary vectors.
2.  **Random Forest (RF):** Handles high-dimensional data by building multiple decision trees, improving accuracy and robustness.
3.  **K-Nearest Neighbors (K-NN):** Classifies samples based on the majority class of their nearest neighbors.
4.  **Logistic Regression (LR):** Used as a statistical baseline method.
5.  **Naïve-Bayes (NB):** Calculates the probability of each subgroup assuming conditional independence of gene expressions.
6.  **Decision Tree (DT):** A supervised learning technique that iteratively divides data based on features.

### 3.4 Performance Comparison

Model performance was compared using metrics and techniques such as:

*   **Accuracy (ACC):** The ratio of correct predictions to total predictions.
    $$ Accuracy = \frac{TP + TN}{TP + TN + FP + FN} $$
*   **Sensitivity (SN) / Recall:** The ratio of true positives (TP) to all actual positives ($$TP + FN$$).
    $$ Sensitivity = \frac{TP}{TP + FN} $$
*   **Specificity (SP):** The portion of real negatives (TN) identified accurately.
    $$ Specificity = \frac{TN}{TN + FP} $$
*   **Precision:** Percentage of accurate positive forecasts amidst all positive predictions ($$TP + FP$$).
    $$ Precision = \frac{TP}{TP + FP} $$
*   **F1-Measure:** A weighted average of precision and recall.
    $$ F1\; measure = \frac{2 \times Recall \times Precision}{Recall + Precision} $$
*   **Mathew’s Correlation Coefficient (MCC):** Assesses binary classifications quality, considered a fair metric even with varying group sizes.
*   **Area Under the Curve (AUC):** Derived from the Receiver Operating Characteristic (ROC) curve, summarizing model performance across thresholds.
*   **Cross-Validation:** Leave-One-Out Cross-Validation (LOOCV) and K-fold Cross Validation (e.g., 10-fold) were used for robust performance estimation.
*   **Efficiency Metrics:** Computational time taken and memory consumed for model building and training were tracked.

## 4. Key Results and Findings

The experimental results demonstrate a comparative analysis across the six models using both continuous and binarized data representations.

| Model | Continuous Data Accuracy | Binarized Data Accuracy | Time Improvement (Binary vs. Continuous) | Memory Reduction (Binary vs. Continuous) |
| :--- | :--- | :--- | :--- | :--- |
| **SVM** | 0.77 | **0.80** | **Significantly faster** (e.g., 0.62s vs 40.52s) | Slightly less (e.g., 49.91 MB vs 50.71 MB) |
| **Naïve Bayes** | 0.63 | 0.72 | Not specified, generally improved | Not specified, generally improved |
| **Random Forest** | 0.63 | 0.69 | Significant reduction (e.g., 21.22s vs 51.3s) | **Major reduction** (e.g., 144.3 MB vs 1118.44 MB) |
| **K-NN** | 0.56 | 0.69 | Significant reduction (e.g., 1.47s vs 47.32s) | **Major reduction** (e.g., 205.52 MB vs 1245.56 MB) |
| **Logistic Regression** | 0.75 | 0.77 | Not specified, generally improved | Not specified, generally improved |
| **Decision Tree** | 0.57 | 0.68 | Not specified, generally improved | Not specified, generally improved |

### Summary

*   **Binarized data generally improves performance** of all tested machine learning models compared to continuous data.
*   **SVM performed the best overall,** achieving the highest accuracy (0.80) when using binarized data.
*   Binarized data proved to be **significantly more computationally efficient** across models like SVM, Random Forest, and K-NN, showing substantial reductions in computational time and memory usage compared to continuous data.
*   Models sensitive to data preprocessing, such as K-NN and Decision Tree, showed the **most significant improvements** in accuracy when using binarized data.

## 5. Conclusion

The research concludes that **binarized gene expression data is a practical, efficient, and effective approach** for classification tasks in lung cancer subgrouping, simplifying the dataset, reducing noise, and enhancing accuracy across multiple machine learning models.

## 6. Dependencies and Setup

*   The implementation involves data preprocessing, feature selection (using Mutual Information and clustering), and applying various ML algorithms (SVM, RF, KNN, LR, NB, DT).
*   Libraries typically required include those for data handling (e.g., Pandas), mathematical operations, feature selection (e.g., `sklearn.feature_selection`), classification models (e.g., `sklearn.svm`, `sklearn.ensemble`), and performance metrics.

## 7. Data Sources Used

The code utilizes data from public repositories, primarily Gene Expression Omnibus (GEO).

*   **GSE135304:** Expression profiling by array, focused on Non-Small Cell Lung Cancer (NSCLC).
*   **GSE43580:** Gene expression profiles of lung cancer tumors (adenocarcinomas and Squamous cell carcinomas).
*   **GSE23739:** miRNA expressions.
*   **Other Datasets Referenced:** GSE10799, GSE13255, TCGA (The Cancer Genome Atlas Research Network).
