# PCA_gene_exp
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df0 = pd.read_csv('TCGA Lung Cancer (LUNG).csv')
df = df0.dropna()

# Select features for PCA
#Choose numeric columns representing gene markers
num_col = [
    'PDCD1', 'CD274', 'CTLA4', 'EGFR', 'ERBB2', 'TP53', 'CD276', 'VTCN1', 
    'C10orf54', 'TNFRSF9', 'TIGIT', 'HAVCR2'
]
X = df[num_col]

# Standardization
#Ensure all features are on the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Principal Component Analysis
#Create PCA object and apply dimensionality reduction
pca = PCA()
X_pca = pca.fit_transform(X_scaled)



# Calculate explained variance ratio
#Determine how much variance each principal component explains
explained_variance_ratio = pca.explained_variance_ratio_

#Scree Plot (explained variance ratio)
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.tight_layout()
plt.show()
#plt.savefig('pca_scree_plot.png', dpi=300)
plt.close()

#Print explained variance ratio
print("Explained Variance Ratio:")
for i, ratio in enumerate(explained_variance_ratio, 1):
    print(f"PC{i}: {ratio:.4f} ({ratio*100:.2f}%)")
# Calculate cumulative explained variance
print("\nCumulative Explained Variance Ratio:")
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
for i, cum_ratio in enumerate(cumulative_variance_ratio, 1):
    print(f"Top {i} PCs: {cum_ratio:.4f} ({cum_ratio*100:.2f}%)")



# Scatter plot of the first two principal components
#Color points by sample type
#Points: each patient sample, X and Y axes represent positions in PC1 and PC2 space
codes, uniques = pd.factorize(df['sample_type'])
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                      c=pd.factorize(df['sample_type'])[0], 
                      cmap='viridis')
plt.xlabel(f'First PC (explained variance: {explained_variance_ratio[0]*100:.2f}%)')
plt.ylabel(f'Second PC (explained variance: {explained_variance_ratio[1]*100:.2f}%)')
plt.title('PCA')
plt.colorbar(scatter, label='Sample Type')

import matplotlib.lines as mlines
legend_elements = [
    mlines.Line2D([0], [0], marker='o', color='w', label=cat,
                  markerfacecolor=plt.cm.viridis(i/(len(uniques)-1)), markersize=8)
    for i, cat in enumerate(uniques)
]
plt.legend(handles=legend_elements, title='Sample Type:')
plt.tight_layout()
plt.show()
#plt.savefig('pca_scatter_plot.png', dpi=300)
plt.close()



# Calculate component loadings to analyze contribution of each feature to PCs
component_loadings = pd.DataFrame(
    pca.components_.T, 
    columns=[f'PC{i+1}' for i in range(pca.n_components_)], 
    index=num_col
)
print("\nComponent Loadings:")
print(component_loadings)

#Save to CSV
component_loadings.to_csv('pca_component_loadings.csv')

# top 3 components
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)

print(pca.explained_variance_ratio_)

top_3 = sum(pca.explained_variance_ratio_[:3])
print(f"Top 3 PCs explain {top_3:.2%} of total variance")

# 3D PCA
from mpl_toolkits.mplot3d import Axes3D  

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
           c=codes, cmap='viridis', s=50, alpha=0.8)

ax.set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.2f}%)')
ax.set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.2f}%)')
ax.set_zlabel(f'PC3 ({explained_variance_ratio[2]*100:.2f}%)')
ax.set_title('3D PCA Projection by Sample Type')

legend_elements_3d = [
    mlines.Line2D([0], [0], marker='o', color='w', label=cat,
                  markerfacecolor=plt.cm.viridis(i/(len(uniques)-1)), markersize=8)
    for i, cat in enumerate(uniques)
]
ax.legend(handles=legend_elements_3d, title='sample type :')

plt.tight_layout()
plt.show()
