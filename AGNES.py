from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler
from matplotlib.gridspec import GridSpec
from sklearn import metrics

### input
data = pd.read_excel(r'E:\Yibin\clustering\landslide.xls')

X = np.array(data)

### CH
ch_scores=[]
ch_scores1=[]
ch_scores2=[]
for i in range(2,15):
    clustering = AgglomerativeClustering(linkage='ward', n_clusters=i)
    clustering1 = AgglomerativeClustering(linkage='complete', n_clusters=i)
    clustering2 = AgglomerativeClustering(linkage='average', n_clusters=i)
    res = clustering.fit(X)
    res1 = clustering1.fit(X)
    res2= clustering2.fit(X)
    ch_scores.append(metrics.calinski_harabasz_score(X,clustering.labels_))
    ch_scores1.append(metrics.calinski_harabasz_score(X,clustering1.labels_))
    ch_scores2.append(metrics.calinski_harabasz_score(X,clustering2.labels_))

plt.figure(dpi=300)
plt.plot(range(2,15),ch_scores,marker='o')
plt.plot(range(2,15),ch_scores1,marker='o')
plt.plot(range(2,15),ch_scores2,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('calinski_harabaz_score')
# plt.ylabel('silhousette_score')
plt.legend(['d$_{min}$', 'd$_{max}$', 'd$_{avg}$'], loc='upper right')
plt.show()

### AGNES
clustering = AgglomerativeClustering(linkage='average', n_clusters=3)

res = clustering.fit(X)

print("各个簇的样本数目：")
print(pd.Series(clustering.labels_).value_counts())

plt.figure()
d0 = X[clustering.labels_ == 0]
plt.plot(d0[:, 0], d0[:, 1], 'r.')
d1 = X[clustering.labels_ == 1]
plt.plot(d1[:, 0], d1[:, 1], 'g.')
d2 = X[clustering.labels_ == 2]
plt.plot(d2[:, 0], d2[:, 1], 'b.')

plt.xlabel("X")
plt.ylabel("Y")
plt.title("AGNES Clustering")
plt.show()