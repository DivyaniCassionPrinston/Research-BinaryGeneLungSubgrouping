# ==================== STEP 4: Prepare Data for Analysis ====================
# - Transpose expression data: rows = samples, columns = genes
# - Check for missing values
# - Check for duplicate rows (identical samples)
from sklearn.preprocessing import LabelEncoder

transposed_data = data.T
print(f"\nTransposed data shape: {transposed_data.shape}")

missing_values = transposed_data.isnull().sum().sum()
print(f"Total missing values: {missing_values}")

num_duplicates = transposed_data.duplicated().sum()
print(f"Number of duplicate rows: {num_duplicates}")

# ==================== STEP 5: Normalize Data (ONLY ONCE) ====================
# - Normalize all gene expression values between 0 and 1
# - Use MinMaxScaler for scaling
# - Save the normalized dataset for later use
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
normalized_data = pd.DataFrame(
    scaler.fit_transform(transposed_data),
    index=transposed_data.index,
    columns=transposed_data.columns,
)

print(f"\nNormalized data shape: {normalized_data.shape}")
normalized_data.to_csv("normalized_data.csv")

# ==================== STEP 6: Merge with Labels ====================
# - Add Sample_ID column to normalized data for merging
# - Merge with metadata containing cancer type labels
# - Encode cancer types into numeric values using LabelEncoder
# - Drop unnecessary columns and save the final dataset

normalized_data["Sample_ID"] = normalized_data.index
final_dataset = normalized_data.merge(metadata_df, on="Sample_ID", how="left")
final_dataset = final_dataset.drop(columns=["Sample_ID"])

label_encoder = LabelEncoder()
final_dataset["Label"] = label_encoder.fit_transform(final_dataset["Cancer_Type"])
final_dataset = final_dataset.drop(columns=["Cancer_Type"])

label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
print(f"\nLabel mapping: {label_mapping}")

final_dataset.to_csv("final_dataset_with_labels.csv", index=False)

# ==================== STEP 6.1: Final Check ====================
# - Print dataset shape and preview to confirm correct processing
print(f"\nFinal dataset shape: {final_dataset.shape}")
print(final_dataset.head())
