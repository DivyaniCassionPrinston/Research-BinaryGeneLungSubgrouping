import GEOparse
import os
import wget
import pandas as pd

# ==================== STEP 1: Download and Load GEO Dataset ====================
# - Define GEO dataset ID and construct the download URL (.soft.gz file)
# - Check if the file already exists locally; if not, download it
# - Load the GEO dataset into a GEOparse object
geo_id = "GSE135304"
url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE135nnn/{geo_id}/soft/{geo_id}_family.soft.gz"

if not os.path.exists(f"{geo_id}_family.soft.gz"):
    filename = wget.download(url, out=f"{geo_id}_family.soft.gz")
else:
    filename = f"{geo_id}_family.soft.gz"

gse = GEOparse.get_GEO(filepath=filename)
print(f"Dataset loaded: {geo_id}")

# ==================== STEP 1.1: Preview First Sample ====================
# - Each key in gse.gsms is a sample ID (e.g., GSM1234567)
# - Each value is a GEOparse object containing expression table + metadata
# - Print the head of the first sample's expression table
first_sample = list(gse.gsms.keys())[0]
print("\nFirst sample preview:")
print(gse.gsms[first_sample].table.head())

# ==================== STEP 2: Extract Expression Matrix ====================
# - Convert dataset into a matrix: genes as rows, samples as columns
# - Save the resulting expression matrix as a CSV file
data = gse.pivot_samples("VALUE")
print(f"\nOriginal data shape: {data.shape}")
data.to_csv(f"{geo_id}_raw_data.csv")

# ==================== STEP 3: Extract Metadata ====================
# - Loop through each sample to collect metadata
# - Look for "cancer type" in 'characteristics_ch1' field
# - Extract the cancer type and store it in a structured DataFrame
metadata = []
for gsm_name, gsm in gse.gsms.items():
    cancer_type = None
    for char in gsm.metadata['characteristics_ch1']:
        if "cancer type" in char.lower():
            cancer_type = char.split(": ")[1]
            break
    metadata.append({'Sample_ID': gsm_name, 'Cancer_Type': cancer_type})

metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv('metadata.csv', index=False)
print(f"\nMetadata shape: {metadata_df.shape}")
print(metadata_df.head())
