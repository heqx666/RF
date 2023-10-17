import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,cohen_kappa_score, roc_auc_score
import matplotlib.pyplot as plt

### input data
data= pd.read_excel(r'E:\Yibin\train\Cluster1.xlsx')
x=data.iloc[:,1:12]
y=data.iloc[:,data.columns=="label"]

### Partition data
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=0)

### parameter optimization
RFclf= RandomForestClassifier()
n_estimators = [int(x) for x in np.linspace(start = 20,stop =200,num = 10)]
min_samples_split = [2,3]
min_samples_leaf = [1,2]
max_depth = [6,7,8,9]
max_features = ['auto','sqrt']
criterion=['entropy','gini']
bootstrap = [True,False]
random_params_group = {'n_estimators':n_estimators,
                      'min_samples_split':min_samples_split,
                      'min_samples_leaf':min_samples_leaf,
                      'max_depth':max_depth,
                      'max_features':max_features,
                       'criterion':criterion,
                      'bootstrap':bootstrap}
random_model =RandomizedSearchCV(RFclf,param_distributions = random_params_group,n_iter = 50,
scoring = 'neg_mean_squared_error',verbose = 2,n_jobs = -1,cv = 10,random_state = 0)
random_model.fit(xtrain,ytrain)
#Get the optimal parameters
print(random_model.best_params_)

### Instantiate with optimal parameters
RFclf1 = RandomForestClassifier(criterion='gini', max_depth=3, min_samples_split=2, n_estimators=60,max_features='auto',
                               min_samples_leaf=2, random_state=0)
rf=RFclf1.fit(xtrain, ytrain)

### accuracy evaluation
p_yest = rf.predict(xtest)
predprob_ytest = rf.predict_proba(xtest)[:, 1]
auc_score = roc_auc_score(ytest,predprob_ytest)
ACC = accuracy_score(ytest,p_yest)
PE = precision_score(ytest,p_yest)
RC = recall_score(ytest,p_yest)
F1 = f1_score(ytest,p_yest)

###predict test
CE=pd.read_csv(r'E:\Yibin\test\Cluster.csv')
CS=CE.iloc[:,1:12]
RF=rf.predict_proba(CS)[:, 1]

### output
df = pd.DataFrame(RF)
df.columns=["p"]
df.to_csv(r'E:\Yibin\result\Proposed_cluster1.csv', encoding="UTF-8", index=False)