# ===============================
# Genomic Analysis of CRC Liver Metastasis
# ===============================

# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# -------------------------------
# 2. Load Dataset
# (Replace with path to your GEO dataset, e.g., GSE10961 expression file)
# -------------------------------
# Example: dataset.csv should have genes in rows, samples in columns
df = pd.read_csv("GSE10961_dataset.csv", index_col=0)

# Metadata to separate synchronous vs metachronous
# Example: replace with actual sample labels
synchronous_samples = [col for col in df.columns if "Sync" in col]
metachronous_samples = [col for col in df.columns if "Meta" in col]

print(f"Synchronous samples: {len(synchronous_samples)}")
print(f"Metachronous samples: {len(metachronous_samples)}")

# -------------------------------
# 3. Preprocessing
# -------------------------------
# Remove missing values
df = df.dropna()

# Log2 transformation
df = np.log2(df + 1)

print("After preprocessing:", df.shape)

# -------------------------------
# 4. Differential Expression Analysis
# -------------------------------
results = []
for gene in df.index:
    sync_values = df.loc[gene, synchronous_samples]
    meta_values = df.loc[gene, metachronous_samples]

    # t-test
    t_stat, p_val = ttest_ind(sync_values, meta_values, equal_var=False)

    # fold change (log2)
    mean_sync = np.mean(sync_values)
    mean_meta = np.mean(meta_values)
    log2fc = mean_sync - mean_meta

    results.append([gene, log2fc, p_val])

# Create results DataFrame
deg_df = pd.DataFrame(results, columns=["Gene", "log2FC", "p_value"])

# Multiple testing correction (FDR)
deg_df["adj_p_value"] = multipletests(deg_df["p_value"], method="fdr_bh")[1]

# Significant genes
significant_genes = deg_df[(abs(deg_df["log2FC"]) >= 1) & (deg_df["adj_p_value"] < 0.05)]
print("Number of significant genes:", significant_genes.shape[0])

# -------------------------------
# 5. Classification of Genes
# -------------------------------
upregulated = significant_genes[significant_genes["log2FC"] >= 1]
downregulated = significant_genes[significant_genes["log2FC"] <= -1]

print(f"Upregulated: {len(upregulated)}, Downregulated: {len(downregulated)}")

# -------------------------------
# 6. Volcano Plot
# -------------------------------
plt.figure(figsize=(8,6))
plt.scatter(deg_df["log2FC"], -np.log10(deg_df["p_value"]), color="grey", alpha=0.5)
plt.scatter(upregulated["log2FC"], -np.log10(upregulated["p_value"]), color="red", label="Upregulated")
plt.scatter(downregulated["log2FC"], -np.log10(downregulated["p_value"]), color="blue", label="Downregulated")
plt.axhline(-np.log10(0.05), color="black", linestyle="--")
plt.axvline(1, color="red", linestyle="--")
plt.axvline(-1, color="blue", linestyle="--")
plt.xlabel("log2 Fold Change")
plt.ylabel("-log10(p-value)")
plt.title("Volcano Plot")
plt.legend()
plt.show()

# -------------------------------
# 7. Correlation Analysis
# -------------------------------
sig_gene_data = df.loc[significant_genes["Gene"]]
corr_matrix = sig_gene_data.T.corr(method="pearson")

# Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Significant Genes")
plt.show()

# -------------------------------
# 8. Strongly Correlated Genes
# -------------------------------
strong_pos = []
strong_neg = []
for g1 in corr_matrix.columns:
    for g2 in corr_matrix.columns:
        if g1 < g2:  # avoid duplicates
            if corr_matrix.loc[g1, g2] >= 0.7:
                strong_pos.append((g1, g2, corr_matrix.loc[g1, g2]))
            elif corr_matrix.loc[g1, g2] <= -0.7:
                strong_neg.append((g1, g2, corr_matrix.loc[g1, g2]))

print("Strong Positive Correlations:", len(strong_pos))
print("Strong Negative Correlations:", len(strong_neg))

# Export for Cytoscape
pd.DataFrame(strong_pos, columns=["Gene1", "Gene2", "Correlation"]).to_csv("positive_network.csv", index=False)
pd.DataFrame(strong_neg, columns=["Gene1", "Gene2", "Correlation"]).to_csv("negative_network.csv", index=False)

print("Networks saved for Cytoscape visualization.")
