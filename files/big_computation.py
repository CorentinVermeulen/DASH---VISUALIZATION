### File used for the creation of non-responsive plots

import plotly.express as px
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import time

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE

tik = time.perf_counter()
data = pd.read_csv('files/data/transcriptomics_data.csv')

X = data.iloc[:, :-2]
y = data['cell_type']
genes = list(data.columns[:-2])
colors = {}
print(f'Time to import data: {time.perf_counter()-tik:.3f} sec.')
## HIERARCHICAL CLUSTER --------
tik = time.perf_counter()
linkage_data = linkage(data.iloc[:, :-1], method='ward', metric='euclidean')
np.savetxt("files/data/linkage_data.csv", linkage_data,delimiter = ",")
print(f'Time for linkage: {time.perf_counter()-tik:.3f} sec.')

tik = time.perf_counter()
plt.figure(figsize=(12,5))
fig_hc = dendrogram(linkage_data, truncate_mode='lastp', no_labels=True, color_threshold=0.4*max(linkage_data[:,2]))
#plt.title('Hierarchical clustering dendrogram')
plt.tight_layout()
plt.savefig('assets/fig_hc.png')
plt.show()
print(f'Time for dendrogram: {time.perf_counter()-tik:.3f} sec.')
# ------------------------------

# t-SNE -----------------------
tik = time.perf_counter()
X_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=60).fit_transform(X)
np.savetxt("files/data/tsne_data.csv", X_tsne,delimiter = ",")

fig_tsne = px.scatter(None, x=X_tsne[:, 0], y=X_tsne[:, 1], color=y, labels={'sort': False, 'color':'Cell Type'})
#fig_tsne.write_json("files/assets/fig_tsne.json")
fig_tsne.write_html("assets/fig_tsne.html")
print(f'Time for t-SNE: {time.perf_counter()-tik:.3f} sec.')
# ------------------------------




