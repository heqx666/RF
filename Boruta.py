from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data= pd.read_excel(r'E:\Yibin\train\Cluster1.xlsx')
X=data.iloc[:,1:16]
Y=data.iloc[:,data.columns=="label"]

x = np.array(X.values)
y = np.array(Y.values)
results = []

rf = RandomForestClassifier(n_estimators=200,
                                min_samples_leaf=5,
                                n_jobs=-1,
                                oob_score=True,
                                random_state=0)

# define Boruta feature selection method
feat_selector = BorutaPy(rf, verbose=2, random_state=0)

# find all relevant features - 5 features should be selected
feat_selector.fit(x,y)
ranking = feat_selector.ranking_

results.append(ranking)

# check selected features - first 5 features are selected
print(feat_selector.support_)

# check ranking of features
print(feat_selector.ranking_)

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(x)

green_area = X.columns[feat_selector.support_].to_list()
blue_area  = X.columns[feat_selector.support_weak_].to_list()
print('features in the green area:', green_area)
print('features in the blue area:', blue_area)

# RF importance
rf.fit(x,y)
feat_labels = data.columns[1:]
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1] #[::-1]
for f in range(x.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))