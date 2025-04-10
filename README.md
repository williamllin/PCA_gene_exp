# PCA_gene_exp
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df0 = pd.read_csv('TCGA Lung Cancer (LUNG).csv')
df = df0.dropna()

#選擇主成分分析的特徵
#選擇代表基因標記數值欄位
num_col = [
    'PDCD1', 'CD274', 'CTLA4', 'EGFR', 'ERBB2', 'TP53', 'CD276', 'VTCN1', 
    'C10orf54', 'TNFRSF9', 'TIGIT', 'HAVCR2'
]
X = df[num_col]

#標準化
#確保所有特徵在同一尺度上
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#主成分分析
#建立PCA物件、降維轉換
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

#========================================================================

#計算解釋方差比例
#確定每個主成分解釋的數據變異程度（貢獻多少）
explained_variance_ratio = pca.explained_variance_ratio_

# Scree Plot (解釋方差比例)
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.xlabel('Principle Component')
plt.ylabel('Explained variance ratio')
plt.title('Scree Plot')
plt.tight_layout()
plt.show()
#plt.savefig('pca_scree_plot.png', dpi=300)
plt.close()

#解釋方差比例
print("解釋方差比例：")
for i, ratio in enumerate(explained_variance_ratio, 1):
    print(f"第{i}主成分: {ratio:.4f} ({ratio*100:.2f}%)")
#計算累積解釋方差，了解前幾個主成分能解釋多少總體變異
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
print("\n累積解釋方差比例：")
for i, cum_ratio in enumerate(cumulative_variance_ratio, 1):
    print(f"前{i}個主成分: {cum_ratio:.4f} ({cum_ratio*100:.2f}%)")

#========================================================================

#繪製前兩個主成分的散點圖
#用不同顏色區分樣本類型
#顏色：不同腫瘤樣本
#點：每個病患樣本，對應的XY軸為第一第二主成分空間中的位置
codes, uniques = pd.factorize(df['sample_type'])
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                      c=pd.factorize(df['sample_type'])[0], 
                      cmap='viridis')
plt.xlabel(f'First PC (explained variance: {explained_variance_ratio[0]*100:.2f}%)')
plt.ylabel(f'Second PC (explained variance: {explained_variance_ratio[1]*100:.2f}%)')
plt.title('PCA')
plt.colorbar(scatter, label='sample type(tumor)')

import matplotlib.lines as mlines
legend_elements = [
    mlines.Line2D([0], [0], marker='o', color='w', label=cat,
                  markerfacecolor=plt.cm.viridis(i/(len(uniques)-1)), markersize=8)
    for i, cat in enumerate(uniques)
]
plt.legend(handles=legend_elements, title='sample type :')
plt.tight_layout()
plt.show()
#plt.savefig('pca_scatter_plot.png', dpi=300)
plt.close()

#========================================================================

#計算成分載荷，分析每個原始特徵對主成分的貢獻
component_loadings = pd.DataFrame(
    pca.components_.T, 
    columns=[f'主成分{i+1}' for i in range(pca.n_components_)], 
    index=num_col
)
print("\n成分載荷：")
print(component_loadings)

#CSV
component_loadings.to_csv('pca_component_loadings.csv')
